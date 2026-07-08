#!/usr/bin/env python3
"""Evaluate CF Attention on LIBERO benchmark.

This script evaluates the counterfactual attention mechanism on LIBERO tasks,
supporting multiple CF modes and detailed effect analysis.

Usage:
    python scripts/eval_cf_attn_libero.py \
        --checkpoint_dir /path/to/pi05_libero \
        --cf_mode E \
        --task_suite libero_spatial

Output:
    - Success rate statistics
    - Rollout videos
    - CF effect analysis data
"""

import collections
import dataclasses
import enum
import json
import logging
import math
import pathlib
import time

import imageio
import numpy as np
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import tqdm
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


class CfMode(enum.Enum):
    """Counterfactual reweighting mode for model.sample_actions_with_cf."""
    BASE = "base"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


@dataclasses.dataclass
class Args:
    """Arguments for CF Attention evaluation on LIBERO."""
    
    # Model configuration
    checkpoint_dir: str = ""  # Set via argument, e.g., ~/.cache/openpi/checkpoints/pi05_libero
    config_name: str = "pi05_libero"
    
    # CF configuration
    cf_mode: CfMode = CfMode.E
    use_cf_sampling: bool = True
    cf_guidance_scale: float = 0.1
    vlm_effect_upper_threshold: float = 0.5
    reweight_action_with_cf: bool = True
    clear_prompt_for_image_cf: bool = True
    return_cf_metrics: bool = True
    
    # LIBERO environment parameters
    task_suite_name: str = ""  # Single task suite (deprecated, use task_suite_names)
    task_suite_names: str = "libero_spatial"  # Comma-separated list of task suites
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10
    num_trials_per_task: int = 10
    
    # Output configuration
    video_out_path: str = "data/cf_attn_libero"
    save_videos: bool = True
    save_cf_effects: bool = True
    
    # Random seed
    seed: int = 7
    
    # GPU configuration
    gpu_id: int = 0
    
    def get_task_suites(self) -> list[str]:
        """Get list of task suites to evaluate."""
        # Support both old task_suite_name and new task_suite_names
        if self.task_suite_name:
            return [self.task_suite_name]
        return [s.strip() for s in self.task_suite_names.split(",") if s.strip()]


def create_cf_policy(args: Args) -> _policy.Policy:
    """Create a policy with CF sampling enabled."""
    train_config = _config.get_config(args.config_name)
    
    sample_kwargs = None
    if args.use_cf_sampling:
        sample_kwargs = {
            "reweight_action_with_cf": args.reweight_action_with_cf,
            "cf_guidance_scale": args.cf_guidance_scale,
            "vlm_effect_upper_threshold": args.vlm_effect_upper_threshold,
            "clear_prompt_for_image_cf": args.clear_prompt_for_image_cf,
            "cf_mode": args.cf_mode.value,
            "return_cf_metrics": args.return_cf_metrics,
        }
    
    return _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        use_cf_sampling=args.use_cf_sampling,
        sample_kwargs=sample_kwargs,
    )


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle representation."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def resize_image(img, target_size):
    """Resize image with padding to target size."""
    from openpi_client import image_tools
    return image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, target_size, target_size)
    )


@dataclasses.dataclass
class EpisodeResult:
    """Result of a single episode."""
    task_id: int
    task_description: str
    episode_idx: int
    success: bool
    total_steps: int
    cf_effects: list  # List of CF effect data per step
    replay_images: list = dataclasses.field(default_factory=list)  # Images for video
    
    def to_dict(self):
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "episode_idx": self.episode_idx,
            "success": self.success,
            "total_steps": self.total_steps,
            "cf_effects": self.cf_effects,
        }


def run_episode(
    env,
    task_description: str,
    policy: _policy.Policy,
    args: Args,
    episode_idx: int,
    task_id: int,
    current_task_suite: str,
) -> EpisodeResult:
    """Run a single episode and return the result."""
    # Reset environment
    env.reset()
    action_plan = collections.deque()
    
    # Setup
    t = 0
    replay_images = []
    cf_effects_data = []
    
    logging.info(f"Starting episode {episode_idx + 1}...")
    
    while t < get_max_steps(current_task_suite) + args.num_steps_wait:
        try:
            # Wait for objects to stabilize
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue
            
            # Get preprocessed images (rotate 180 degrees to match train preprocessing)
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = resize_image(img, args.resize_size)
            wrist_img = resize_image(wrist_img, args.resize_size)
            
            # Save image for replay video
            if args.save_videos:
                replay_images.append(img)
            
            if not action_plan:
                # Prepare observation dict
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": str(task_description),
                }
                
                # Query policy
                result = policy.infer(element)
                action_chunk = result["actions"]
                
                # Extract CF metrics/effects if available
                if args.save_cf_effects and ("cf_metrics" in result or "cf_effects" in result):
                    metrics_payload = result.get("cf_metrics", result.get("cf_effects"))
                    cf_effects_data.append({
                        "step": t,
                        "cf_effects": metrics_payload,
                    })
                
                assert len(action_chunk) >= args.replan_steps
                action_plan.extend(action_chunk[: args.replan_steps])
            
            action = action_plan.popleft()
            
            # Execute action
            obs, reward, done, info = env.step(action.tolist())
            
            if done:
                return EpisodeResult(
                    task_id=task_id,
                    task_description=task_description,
                    episode_idx=episode_idx,
                    success=True,
                    total_steps=t - args.num_steps_wait,
                    cf_effects=cf_effects_data,
                    replay_images=replay_images,
                )
            
            t += 1
            
        except Exception as e:
            logging.error(f"Episode error: {e}")
            break
    
    return EpisodeResult(
        task_id=task_id,
        task_description=task_description,
        episode_idx=episode_idx,
        success=False,
        total_steps=t - args.num_steps_wait if t > args.num_steps_wait else 0,
        cf_effects=cf_effects_data,
        replay_images=replay_images,
    )


def get_max_steps(task_suite_name: str) -> int:
    """Get max steps for a task suite."""
    if task_suite_name == "libero_spatial":
        return 220
    elif task_suite_name == "libero_object":
        return 280
    elif task_suite_name == "libero_goal":
        return 300
    elif task_suite_name == "libero_10":
        return 520
    elif task_suite_name == "libero_90":
        return 400
    else:
        raise ValueError(f"Unknown task suite: {task_suite_name}")


def save_episode_video(video_path: pathlib.Path, images: list, fps: int = 10):
    """Save episode replay video."""
    imageio.mimwrite(video_path, [np.asarray(x) for x in images], fps=fps)


def eval_single_task_suite(args: Args, task_suite_name: str, policy: _policy.Policy) -> dict:
    """Evaluate a single task suite."""
    # Initialize LIBERO task suite
    logging.info(f"\n{'='*60}")
    logging.info(f"Task suite: {task_suite_name}")
    logging.info(f"{'='*60}")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    
    # Create output directory
    output_dir = pathlib.Path(args.video_out_path) / f"{args.cf_mode.value}_{task_suite_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    if args.save_videos:
        videos_dir.mkdir(parents=True, exist_ok=True)
    
    # Start evaluation
    total_episodes, total_successes = 0, 0
    all_episode_results = []
    
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        
        logging.info(f"\nTask {task_id}: {task_description}")
        
        task_episodes, task_successes = 0, 0
        
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            # Set initial state
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            
            # Run episode
            result = run_episode(
                env, task_description, policy, args, episode_idx, task_id, task_suite_name
            )
            all_episode_results.append(result)
            
            if result.success:
                task_successes += 1
                total_successes += 1
            
            task_episodes += 1
            total_episodes += 1
            
            # Save video
            if args.save_videos and result.replay_images:
                suffix = "success" if result.success else "failure"
                task_segment = task_description.replace(" ", "_")
                video_path = videos_dir / f"task{task_id}_ep{episode_idx}_{suffix}.mp4"
                save_episode_video(video_path, result.replay_images)
            
            # Log progress
            logging.info(f"Episode {episode_idx + 1}: {result.success}")
            logging.info(f"Task success rate: {task_successes / task_episodes * 100:.1f}%")
            logging.info(f"Total success rate: {total_successes / total_episodes * 100:.1f}%")
        
        # Log task results
        logging.info(f"\nTask {task_id} final success rate: {task_successes / task_episodes * 100:.1f}%")
    
    # Calculate final results
    final_success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    
    # Save results
    results_file = output_dir / "results.txt"
    results_file.write_text(
        f"CF Mode: {args.cf_mode.value}\n"
        f"Task Suite: {task_suite_name}\n"
        f"Total episodes: {total_episodes}\n"
        f"Total successes: {total_successes}\n"
        f"Success rate: {final_success_rate:.4f}\n"
        f"Checkpoint: {args.checkpoint_dir}\n"
        f"CF guidance scale: {args.cf_guidance_scale}\n"
        f"VLM effect threshold: {args.vlm_effect_upper_threshold}\n"
    )
    
    # Save detailed episode results
    if args.save_cf_effects:
        effects_file = output_dir / "cf_attn_effects.json"
        effects_data = {
            "config": {
                "cf_mode": args.cf_mode.value,
                "task_suite": task_suite_name,
                "cf_guidance_scale": args.cf_guidance_scale,
                "vlm_effect_threshold": args.vlm_effect_upper_threshold,
                "reweight_action_with_cf": args.reweight_action_with_cf,
            },
            "summary": {
                "total_episodes": total_episodes,
                "total_successes": total_successes,
                "success_rate": final_success_rate,
            },
            "episodes": [r.to_dict() for r in all_episode_results],
        }
        effects_file.write_text(json.dumps(effects_data, indent=2))
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Task suite evaluation complete!")
    logging.info(f"CF Mode: {args.cf_mode.value}")
    logging.info(f"Task Suite: {task_suite_name}")
    logging.info(f"Success rate: {final_success_rate * 100:.1f}%")
    logging.info(f"Results saved to: {output_dir}")
    logging.info(f"{'='*60}")
    
    return {
        "success_rate": final_success_rate,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "output_dir": str(output_dir),
    }


def eval_cf_attn_libero(args: Args) -> dict:
    """Evaluate CF attention on LIBERO benchmark across multiple task suites."""
    # Set random seed
    np.random.seed(args.seed)
    
    # Set GPU
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    # Get task suites to evaluate
    task_suites = args.get_task_suites()
    logging.info(f"Will evaluate {len(task_suites)} task suites: {task_suites}")
    
    # Create policy (shared across all task suites)
    logging.info(f"Creating policy with CF mode: {args.cf_mode.value}")
    logging.info(f"Checkpoint: {args.checkpoint_dir}")
    policy = create_cf_policy(args)
    
    # Evaluate each task suite
    all_results = {}
    total_episodes_all = 0
    total_successes_all = 0
    
    start_time = time.time()
    
    for task_suite_name in task_suites:
        logging.info(f"\n{'#'*60}")
        logging.info(f"Evaluating task suite: {task_suite_name}")
        logging.info(f"{'#'*60}")
        
        result = eval_single_task_suite(args, task_suite_name, policy)
        all_results[task_suite_name] = result
        total_episodes_all += result["total_episodes"]
        total_successes_all += result["total_successes"]
    
    elapsed_time = time.time() - start_time
    
    # Generate summary report
    overall_success_rate = total_successes_all / total_episodes_all if total_episodes_all > 0 else 0.0
    
    summary_dir = pathlib.Path(args.video_out_path) / f"{args.cf_mode.value}_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = summary_dir / "overall_results.txt"
    summary_content = (
        f"CF Mode: {args.cf_mode.value}\n"
        f"Task Suites: {', '.join(task_suites)}\n"
        f"Total episodes: {total_episodes_all}\n"
        f"Total successes: {total_successes_all}\n"
        f"Overall success rate: {overall_success_rate:.4f}\n"
        f"Total evaluation time: {elapsed_time / 60:.1f} minutes\n"
        f"Checkpoint: {args.checkpoint_dir}\n"
        f"\n--- Per Task Suite Results ---\n"
    )
    for suite_name, result in all_results.items():
        summary_content += f"{suite_name}: {result['success_rate'] * 100:.1f}% ({result['total_successes']}/{result['total_episodes']})\n"
    summary_file.write_text(summary_content)
    
    # Log final summary
    logging.info(f"\n{'='*60}")
    logging.info(f"ALL EVALUATIONS COMPLETE!")
    logging.info(f"{'='*60}")
    logging.info(f"CF Mode: {args.cf_mode.value}")
    logging.info(f"Task Suites evaluated: {len(task_suites)}")
    logging.info(f"Overall success rate: {overall_success_rate * 100:.1f}%")
    logging.info(f"Total episodes: {total_episodes_all}")
    logging.info(f"Total successes: {total_successes_all}")
    logging.info(f"Total evaluation time: {elapsed_time / 60:.1f} minutes")
    logging.info(f"Summary saved to: {summary_file}")
    logging.info(f"{'='*60}")
    
    return {
        "overall_success_rate": overall_success_rate,
        "total_episodes": total_episodes_all,
        "total_successes": total_successes_all,
        "task_suite_results": all_results,
        "summary_file": str(summary_file),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = tyro.cli(Args)
    
    results = eval_cf_attn_libero(args)


if __name__ == "__main__":
    main()
