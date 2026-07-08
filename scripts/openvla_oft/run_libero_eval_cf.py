"""
run_libero_eval_cf.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
With Counterfactual (CF) reasoning: computes effect_vlm and effect_prop at each step.
"""

import json
import logging
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb
import torch
from PIL import Image

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
    prepare_images_for_vla,
    normalize_proprio,
    center_crop_image,
    OPENVLA_IMAGE_SIZE,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Force GPU 0
DEVICE = torch.device("cuda:0")


@dataclass
class GenerateConfig:
    # fmt: off

    # Model-specific parameters
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""

    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True

    center_crop: bool = True
    num_open_loop_steps: int = 8

    lora_rank: int = 32

    unnorm_key: Union[str, Path] = ""

    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # LIBERO environment-specific parameters
    task_suite_name: str = TaskSuite.LIBERO_10
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256

    # Utils
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"

    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"

    seed: int = 7

    # Counterfactual evaluation parameters
    enable_cf_eval: bool = True  # Enable counterfactual reasoning
    use_cf_reweight: bool = False  # Whether to use CF to modify actions (baseline: False)
    cf_guidance_scale: float = 0.1  # CF guidance scale for action modification
    vlm_effect_upper_threshold: float = 0.90  # If effect_vlm > this, use baseline
    vlm_effect_lower_threshold: float = 0.35  # If effect_vlm < this, use baseline
    cf_log_dir: str = "./experiments/cf_logs"  # Directory to save CF effect logs

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    model = get_model(cfg)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,
        )

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    unnorm_key = cfg.task_suite_name

    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    run_id = f"EVAL-CF-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Create CF log directory
    os.makedirs(cfg.cf_log_dir, exist_ok=True)

    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    initial_states = task_suite.get_task_init_states(task_id)

    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img


def process_action(action, model_family):
    """Process action before sending to environment."""
    action = normalize_gripper_action(action, binarize=True)

    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def compute_action_diff(actions_base: List[np.ndarray], actions_cf: List[np.ndarray]) -> float:
    """Compute L2 distance between two action lists.

    Args:
        actions_base: Baseline actions
        actions_cf: Counterfactual actions

    Returns:
        float: Mean L2 distance across all actions
    """
    if not actions_base or not actions_cf:
        return 0.0

    n = min(len(actions_base), len(actions_cf))
    diffs = []

    for i in range(n):
        a_base = actions_base[i]
        a_cf = actions_cf[i]

        # Both should be numpy arrays
        if isinstance(a_base, np.ndarray) and isinstance(a_cf, np.ndarray):
            diff = np.linalg.norm(a_base - a_cf)
            diffs.append(diff)

    if not diffs:
        return 0.0

    return float(np.mean(diffs))


def get_vla_action_with_cf(
    cfg: Any,
    vla: torch.nn.Module,
    processor: Any,
    obs: Dict[str, Any],
    task_label: str,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    noisy_action_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False,
) -> tuple:
    """
    Generate action predictions with counterfactual reasoning.

    Returns:
        tuple: (actions_baseline, effect_vlm, effect_prop)
    """
    effect_vlm = 0.0
    effect_prop = 0.0

    with torch.inference_mode():
        # === Prepare inputs ===
        all_images = [obs["full_image"]]
        if cfg.num_images_in_input > 1:
            all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

        all_images = prepare_images_for_vla(all_images, cfg)
        primary_image = all_images.pop(0)

        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)

        if all_images:
            all_wrist_inputs = [
                processor(prompt, image_wrist).to(DEVICE, dtype=torch.bfloat16) for image_wrist in all_images
            ]
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        # Process proprioception data - convert to tensor
        proprio = None
        proprio_tensor = None
        if cfg.use_proprio:
            proprio_raw = obs["state"]
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            proprio_normalized = normalize_proprio(proprio_raw, proprio_norm_stats)
            # Convert to tensor for passing to model
            proprio_tensor = torch.tensor(proprio_normalized).to(DEVICE, dtype=torch.bfloat16)

        # === Baseline prediction ===
        if action_head is None:
            action_base, _ = vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False)
        else:
            action_base, _ = vla.predict_action(
                **inputs,
                unnorm_key=cfg.unnorm_key,
                do_sample=False,
                proprio=proprio_tensor,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                action_head=action_head,
                use_film=use_film,
            )

        actions_base = [action_base[i] for i in range(len(action_base))]

        # === Counterfactual: Image zero ===
        # This measures the effect of vision (VLM) on action prediction
        if cfg.enable_cf_eval:
            # Clone inputs and zero out images
            inputs_img_zero = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "pixel_values": torch.zeros_like(inputs["pixel_values"]),
            }

            if action_head is None:
                action_img_zero, _ = vla.predict_action(**inputs_img_zero, unnorm_key=cfg.unnorm_key, do_sample=False)
            else:
                action_img_zero, _ = vla.predict_action(
                    **inputs_img_zero,
                    unnorm_key=cfg.unnorm_key,
                    do_sample=False,
                    proprio=proprio_tensor,  # Keep proprio unchanged
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    action_head=action_head,
                    use_film=use_film,
                )

            actions_img_zero = [action_img_zero[i] for i in range(len(action_img_zero))]
            effect_vlm = compute_action_diff(actions_base, actions_img_zero)

        # === Counterfactual: Proprio zero ===
        # This measures the effect of proprioceptive state on action prediction
        # IMPORTANT: We pass proprio=None (not zeros) to truly remove proprio information.
        # Passing zeros through ProprioProjector would still produce non-zero embeddings due to bias!
        if cfg.enable_cf_eval and cfg.use_proprio:
            # Pass proprio=None to skip proprio processing entirely
            # This is the correct way to measure proprio effect

            if action_head is None:
                # For discrete action prediction, proprio is not used anyway
                action_prop_zero, _ = vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False)
            else:
                # Pass proprio=None and proprio_projector=None to completely skip proprio
                action_prop_zero, _ = vla.predict_action(
                    **inputs,
                    unnorm_key=cfg.unnorm_key,
                    do_sample=False,
                    proprio=None,  # None instead of zeros - truly removes proprio
                    proprio_projector=None,  # Also None to skip processing
                    noisy_action_projector=noisy_action_projector,
                    action_head=action_head,
                    use_film=use_film,
                )

            actions_prop_zero = [action_prop_zero[i] for i in range(len(action_prop_zero))]
            effect_prop = compute_action_diff(actions_base, actions_prop_zero)

    return actions_base, effect_vlm, effect_prop


def get_action_with_cf(
    cfg: Any,
    model: torch.nn.Module,
    obs: Dict[str, Any],
    task_label: str,
    processor: Optional[Any] = None,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    noisy_action_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False,
) -> tuple:
    """
    Query the model to get action predictions with CF effects.

    Returns:
        tuple: (actions, effect_vlm, effect_prop)
    """
    with torch.no_grad():
        if cfg.model_family == "openvla":
            actions, effect_vlm, effect_prop = get_vla_action_with_cf(
                cfg=cfg,
                vla=model,
                processor=processor,
                obs=obs,
                task_label=task_label,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=use_film,
            )
        else:
            raise ValueError(f"Unsupported model family: {cfg.model_family}")

    return actions, effect_vlm, effect_prop


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
    cf_records=None,
    episode_idx=0,
    task_id=0,
):
    """Run a single episode in the environment with CF logging.

    Note: Always executes baseline actions (not modified by CF).
    """
    env.reset()

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match NUM_ACTIONS_CHUNK")

    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    success = False
    episode_actions = []  # Record all actions for this episode

    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            if len(action_queue) == 0:
                # Query model with CF reasoning - returns baseline actions
                actions, effect_vlm, effect_prop = get_action_with_cf(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )

                # IMPORTANT: We use baseline actions directly (not modified by CF)
                action_queue.extend(actions)

                # Log CF effects and actions
                if cfg.enable_cf_eval and cf_records is not None:
                    # Record baseline actions (before gripper processing)
                    baseline_actions_raw = [a.copy() if isinstance(a, np.ndarray) else np.array(a) for a in actions]

                    cf_record = {
                        "task_id": task_id,
                        "task_description": task_description,
                        "episode_idx": episode_idx,
                        "timestep": t - cfg.num_steps_wait,
                        "effect_vlm": effect_vlm,
                        "effect_prop": effect_prop,
                        "total_effect": effect_vlm + effect_prop,
                        "baseline_actions": [a.tolist() for a in baseline_actions_raw],  # List of 7-dim actions
                        "proprio_state": observation["state"].tolist(),  # 8-dim proprio
                        "num_actions_in_chunk": len(actions),
                    }
                    cf_records.append(cf_record)

                    # Log to console when effects are significant
                    if effect_vlm > 0.3 or effect_prop > 0.3:
                        log_message(
                            f"t={t - cfg.num_steps_wait}: effect_vlm={effect_vlm:.3f}, effect_prop={effect_prop:.3f}, actions[0]={actions[0][:4]}",
                            log_file
                        )

            # Execute baseline action (unchanged by CF)
            action = action_queue.popleft()
            action_processed = process_action(action, cfg.model_family)

            # Record executed action
            episode_actions.append({
                "timestep": t - cfg.num_steps_wait,
                "action_raw": action.tolist(),
                "action_processed": action_processed.tolist(),
            })

            obs, reward, done, info = env.step(action_processed.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)
        import traceback
        log_message(traceback.format_exc(), log_file)

    return success, replay_images, episode_actions


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    cf_records=None,
    all_episode_records=None,
):
    """Run evaluation for a single task with CF logging."""
    task = task_suite.get_task(task_id)

    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    task_episodes, task_successes = 0, 0

    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"Task {task_id}: {task_description[:30]}"):
        log_message(f"\nTask: {task_description}", log_file)

        if cfg.initial_states_path == "DEFAULT":
            initial_state = initial_states[episode_idx]
        else:
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        success, replay_images, episode_actions = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
            cf_records,
            episode_idx,
            task_id,
        )

        # Record episode summary
        if all_episode_records is not None:
            episode_record = {
                "task_id": task_id,
                "task_description": task_description,
                "episode_idx": episode_idx,
                "success": success,
                "num_actions": len(episode_actions),
                "actions": episode_actions,
            }
            all_episode_records.append(episode_record)

        task_episodes += 1
        total_episodes += 1

        if success:
            task_successes += 1
            total_successes += 1

        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file
        )

        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero_cf(cfg: GenerateConfig) -> float:
    """Main function to evaluate policy with counterfactual reasoning.

    IMPORTANT: This evaluation uses ONLY baseline actions for execution.
    CF effects are computed and logged but DO NOT modify the executed actions.
    """
    validate_config(cfg)

    set_seed_everywhere(cfg.seed)

    log_message("=" * 60)
    log_message("Starting LIBERO evaluation with Counterfactual Analysis")
    log_message("=" * 60)
    log_message(f"GPU: cuda:0 (forced)")
    log_message(f"Model: {cfg.pretrained_checkpoint}")
    log_message(f"Task suite: {cfg.task_suite_name}")
    log_message(f"CF eval enabled: {cfg.enable_cf_eval}")
    log_message("IMPORTANT: Executing BASELINE actions only (not CF-modified)")
    log_message("=" * 60)

    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    resize_size = get_image_resize_size(cfg)

    log_file, local_log_filepath, run_id = setup_logging(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Number of tasks: {num_tasks}", log_file)
    log_message(f"Trials per task: {cfg.num_trials_per_task}", log_file)
    log_message(f"Counterfactual evaluation enabled: {cfg.enable_cf_eval}", log_file)

    # Initialize CF records (per-timestep) and episode records (per-episode)
    cf_records = []
    all_episode_records = []

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
            cf_records,
            all_episode_records,
        )

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message("=" * 60, log_file)
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    log_message("=" * 60, log_file)

    # Save CF records (per-timestep effects)
    if cfg.enable_cf_eval and cf_records:
        cf_log_filepath = os.path.join(cfg.cf_log_dir, run_id + "_cf_effects.json")
        with open(cf_log_filepath, "w") as f:
            json.dump(cf_records, f, indent=2)
        log_message(f"CF effects saved to: {cf_log_filepath}", log_file)
        log_message(f"Total CF records: {len(cf_records)}", log_file)

        # Compute and log summary statistics
        if cf_records:
            mean_effect_vlm = np.mean([r["effect_vlm"] for r in cf_records])
            mean_effect_prop = np.mean([r["effect_prop"] for r in cf_records])
            max_effect_vlm = np.max([r["effect_vlm"] for r in cf_records])
            max_effect_prop = np.max([r["effect_prop"] for r in cf_records])

            log_message(f"Mean effect_vlm: {mean_effect_vlm:.4f}", log_file)
            log_message(f"Mean effect_prop: {mean_effect_prop:.4f}", log_file)
            log_message(f"Max effect_vlm: {max_effect_vlm:.4f}", log_file)
            log_message(f"Max effect_prop: {max_effect_prop:.4f}", log_file)

            # Per-task statistics
            task_effects = {}
            for r in cf_records:
                task_id = r["task_id"]
                if task_id not in task_effects:
                    task_effects[task_id] = {"vlm": [], "prop": []}
                task_effects[task_id]["vlm"].append(r["effect_vlm"])
                task_effects[task_id]["prop"].append(r["effect_prop"])

            log_message("\nPer-task effect statistics:", log_file)
            for task_id, effects in task_effects.items():
                log_message(
                    f"Task {task_id}: mean_vlm={np.mean(effects['vlm']):.4f}, mean_prop={np.mean(effects['prop']):.4f}",
                    log_file
                )

    # Save episode records (per-episode with all actions)
    if all_episode_records:
        episode_log_filepath = os.path.join(cfg.cf_log_dir, run_id + "_episode_records.json")
        with open(episode_log_filepath, "w") as f:
            json.dump(all_episode_records, f, indent=2)
        log_message(f"Episode records saved to: {episode_log_filepath}", log_file)
        log_message(f"Total episodes recorded: {len(all_episode_records)}", log_file)

    # Create summary JSON
    summary = {
        "run_id": run_id,
        "task_suite": cfg.task_suite_name,
        "model_path": str(cfg.pretrained_checkpoint),
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": final_success_rate,
        "num_cf_records": len(cf_records),
        "mean_effect_vlm": mean_effect_vlm if cf_records else 0,
        "mean_effect_prop": mean_effect_prop if cf_records else 0,
        "max_effect_vlm": max_effect_vlm if cf_records else 0,
        "max_effect_prop": max_effect_prop if cf_records else 0,
    }
    summary_filepath = os.path.join(cfg.cf_log_dir, run_id + "_summary.json")
    with open(summary_filepath, "w") as f:
        json.dump(summary, f, indent=2)
    log_message(f"Summary saved to: {summary_filepath}", log_file)

    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)
        if cfg.enable_cf_eval and cf_records:
            wandb.save(cf_log_filepath)
            wandb.save(episode_log_filepath)
            wandb.save(summary_filepath)

    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero_cf()