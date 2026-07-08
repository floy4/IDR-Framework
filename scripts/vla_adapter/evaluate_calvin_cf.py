"""
evaluate_calvin_cf.py

Evaluate VLA on CALVIN benchmark with Counterfactual Inference.
Based on evaluate_calvin.py with CF support for zeroing visual and proprioceptive features.
"""

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import json
import logging
import os
from collections import deque
from pathlib import Path
import time
import copy

try:
    from moviepy import ImageSequenceClip  # moviepy v2.x
except ImportError:
    from moviepy.editor import ImageSequenceClip  # moviepy v1.x
from accelerate import Accelerator
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

from calvin_agent.models.calvin_base_model import CalvinBaseModel
from prismatic.models.projectors import NoisyActionProjector, ProprioProjector
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
)
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from vla_evaluation import DualSystemCalvinEvaluation

from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
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

os.environ["FFMPEG_BINARY"] = "auto-detect"
CALVIN_ROOT_DEFAULT = "/path/to/calvin"
os.environ.setdefault("CALVIN_ROOT", CALVIN_ROOT_DEFAULT)
CALVIN_ROOT = os.environ["CALVIN_ROOT"]

from collections import Counter
import json
import numpy as np
from typing import Optional, Union
from pathlib import Path
from dataclasses import dataclass
import draccus

import os
import torch


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "../outputs/calvin-abc"

    use_minivla: bool = False

    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_x0_prediction: bool = False
    num_diffusion_steps: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True

    center_crop: bool = False
    num_open_loop_steps: int = 8

    unnorm_key: Union[str, Path] = ""

    load_in_8bit: bool = False
    load_in_4bit: bool = False

    #################################################################################################################
    # CALVIN environment-specific parameters
    #################################################################################################################
    calvin_root: str = "/path/to/calvin"                             # Path to CALVIN repo (for configs)
    calvin_dataset: str = "/path/to/calvin/task_ABC_D"      # Path to CALVIN dataset
    log_dir: str = "log"
    with_depth: bool = True
    with_gripper: bool = True
    with_cfg: bool = True
    enrich_lang: bool = False

    #################################################################################################################
    # Counterfactual Inference parameters
    #################################################################################################################
    use_cf: bool = False
    cf_method: str = "input_zeroing"
    cf_mode: str = "E"
    cf_guidance_scale: float = 0.1
    vlm_effect_threshold: float = 0.5
    cfg_scale: float = 1.0

    use_proprio_cf: bool = False
    cf_method_proprio: str = "input_zeroing"
    cfg_scale_proprio: float = 1.0

    cf_verbose: bool = False

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs-pro-cf"
    seed: int = 7
    save_version: str = "Pro"

    # fmt: on


def print_and_save(results, sequences, eval_result_path, task_name=None, epoch=None):
    current_data = {}
    print(f"Results for Epoch {epoch}:")
    avg_seq_len = np.mean(results)
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
    print(f"Average successful sequence length: {avg_seq_len}")
    print("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        print(f"{i}: {sr * 100:.1f}%")

    cnt_success = Counter()
    cnt_fail = Counter()

    for result, (_, sequence) in zip(results, sequences):
        for successful_tasks in sequence[:result]:
            cnt_success[successful_tasks] += 1
        if result < len(sequence):
            failed_task = sequence[result]
            cnt_fail[failed_task] += 1

    total = cnt_success + cnt_fail
    task_info = {}
    for task in total:
        task_info[task] = {"success": cnt_success[task], "total": total[task]}
        print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

    data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}
    current_data[epoch] = data

    if not os.path.isdir(f'./{task_name}'):
        os.mkdir(f'./{task_name}')
    with open(f'./{task_name}/split_{torch.cuda.current_device()}.json', "w") as file:
        json.dump(chain_sr, file)

    print()
    previous_data = {}
    json_data = {**previous_data, **current_data}
    with open(eval_result_path, "w") as file:
        json.dump(json_data, file)
    print(
        f"Best model: epoch {max(json_data, key=lambda x: json_data[x]['avg_seq_len'])} "
        f"with average sequences length of {max(map(lambda x: x['avg_seq_len'], json_data.values()))}"
    )


def make_env(dataset_path, observation_space, device):
    val_folder = Path(dataset_path) / "validation"
    from calvin_env_wrapper import CalvinEnvWrapperRaw
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    return env


def evaluate_policy(model, env, eval_sr_path, eval_result_path, num_procs, procs_id, eval_dir, ep_len, num_sequences, task_name='test', enrich_lang=False, debug=False, calvin_root="/path/to/calvin"):
    conf_dir = Path(f"{calvin_root}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    if enrich_lang:
        with open('/root/RoboDual/vla-scripts/enrich_lang_annotations.json', 'r') as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_dir = get_log_dir(eval_dir)
    eval_sequences = get_sequences(num_sequences)

    num_seq_per_procs = num_sequences // num_procs
    eval_sequences = eval_sequences[num_seq_per_procs * procs_id:num_seq_per_procs * (procs_id + 1)]

    results = []
    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    sequence_i = 0
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i, ep_len)
        results.append(result)
        if not debug:
            success_list = count_success(results)
            with open(eval_sr_path, 'a') as f:
                line = f"{sequence_i}/{num_sequences}: "
                for sr in success_list:
                    line += f"{sr:.3f} | "
                sequence_i += 1
                line += "\n"
                f.write(line)
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]) + "|"
            )
        else:
            sequence_i += 1
    print_and_save(results, eval_sequences, eval_result_path, task_name, None)
    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i, ep_len):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout_hi3(env, model, task_checker, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    normalized_action = action.copy()
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    if binarize:
        sign = np.sign(normalized_action[..., -1])
        sign = np.array(sign)
        sign[sign == 0.0] = 1
        sign[sign == -0.0] = -1
        normalized_action[..., -1] = sign
    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    inverted_action = action.copy()
    inverted_action[..., -1] *= -1.0
    return inverted_action


def process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


def rollout_hi3(env, model, task_oracle, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len):
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)

    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    img_dict = {
        'static': [],
        'gripper': [],
    }

    for step in range(80):
        action_buffers = [None, None, None]

        action_buffers[0] = model.step(obs, lang_annotation, 0)
        action = action_buffers[0][0]
        action = process_action(action, "openvla")
        obs, reward, done, current_info = env.step(action.tolist())

        img_dict['static'].append(copy.deepcopy(obs['rgb_obs']['rgb_static']))
        img_dict['gripper'].append(copy.deepcopy(obs['rgb_obs']['rgb_gripper']))

        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            print(colored("success", "green"), end=" ")
            for key in img_dict.keys():
                clip = ImageSequenceClip(img_dict[key], fps=50)
                clip.write_videofile(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-succ.mp4'), fps=50, codec='libx264', bitrate="5000k")
            return True

        action_buffers[1] = model.step(obs, lang_annotation, 1)
        action = (action_buffers[0][1] + action_buffers[1][0]) / 2
        action = process_action(action, "openvla")
        obs, reward, done, current_info = env.step(action.tolist())

        img_dict['static'].append(copy.deepcopy(obs['rgb_obs']['rgb_static']))
        img_dict['gripper'].append(copy.deepcopy(obs['rgb_obs']['rgb_gripper']))

        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            print(colored("success", "green"), end=" ")
            for key in img_dict.keys():
                clip = ImageSequenceClip(img_dict[key], fps=50)
                clip.write_videofile(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-succ.mp4'), fps=50, codec='libx264', bitrate="5000k")
            return True

        action_buffers[2] = model.step(obs, lang_annotation, 2)
        action = (action_buffers[0][2] + action_buffers[1][1] + action_buffers[2][0]) / 3
        action = process_action(action, "openvla")
        obs, reward, done, current_info = env.step(action.tolist())

        img_dict['static'].append(copy.deepcopy(obs['rgb_obs']['rgb_static']))
        img_dict['gripper'].append(copy.deepcopy(obs['rgb_obs']['rgb_gripper']))

        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            print(colored("success", "green"), end=" ")
            for key in img_dict.keys():
                clip = ImageSequenceClip(img_dict[key], fps=50)
                clip.write_videofile(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-succ.mp4'), fps=50, codec='libx264', bitrate="5000k")
            return True

        for t in range(2, 7):
            action = (action_buffers[0][t] + action_buffers[1][t-1] + action_buffers[2][t-2]) / 3
            action = process_action(action, "openvla")
            obs, reward, done, current_info = env.step(action.tolist())

            img_dict['static'].append(copy.deepcopy(obs['rgb_obs']['rgb_static']))
            img_dict['gripper'].append(copy.deepcopy(obs['rgb_obs']['rgb_gripper']))

            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                print(colored("success", "green"), end=" ")
                for key in img_dict.keys():
                    clip = ImageSequenceClip(img_dict[key], fps=50)
                    clip.write_videofile(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-succ.mp4'), fps=50, codec='libx264', bitrate="5000k")
                return True

        action = (action_buffers[1][7] + action_buffers[2][6]) / 2
        action = process_action(action, "openvla")
        obs, reward, done, current_info = env.step(action.tolist())

        img_dict['static'].append(copy.deepcopy(obs['rgb_obs']['rgb_static']))
        img_dict['gripper'].append(copy.deepcopy(obs['rgb_obs']['rgb_gripper']))

        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            print(colored("success", "green"), end=" ")
            for key in img_dict.keys():
                clip = ImageSequenceClip(img_dict[key], fps=50)
                clip.write_videofile(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-succ.mp4'), fps=50, codec='libx264', bitrate="5000k")
            return True

        action = action_buffers[2][7]
        action = process_action(action, "openvla")
        obs, reward, done, current_info = env.step(action.tolist())

        img_dict['static'].append(copy.deepcopy(obs['rgb_obs']['rgb_static']))
        img_dict['gripper'].append(copy.deepcopy(obs['rgb_obs']['rgb_gripper']))

        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            print(colored("success", "green"), end=" ")
            for key in img_dict.keys():
                clip = ImageSequenceClip(img_dict[key], fps=50)
                clip.write_videofile(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-succ.mp4'), fps=50, codec='libx264', bitrate="5000k")
            return True

    print(colored("fail", "red"), end=" ")
    for key in img_dict.keys():
        clip = ImageSequenceClip(img_dict[key], fps=50)
        clip.write_videofile(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-fail.mp4'), fps=50, codec='libx264', bitrate="5000k")
    return False


from huggingface_hub import HfApi, hf_hub_download
import shutil
from datetime import datetime
import filecmp
from typing import Any, Dict, List, Optional, Tuple, Union


def model_is_on_hf_hub(model_path: str) -> bool:
    try:
        HfApi().model_info(model_path)
        return True
    except Exception:
        return False


def update_auto_map(pretrained_checkpoint: str) -> None:
    if not os.path.isdir(pretrained_checkpoint):
        return
    config_path = os.path.join(pretrained_checkpoint, "config.json")
    if not os.path.exists(config_path):
        print(f"Warning: No config.json found at {config_path}")
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(pretrained_checkpoint, f"config.json.back.{timestamp}")
    shutil.copy2(config_path, backup_path)
    print(f"Created backup of original config at: {os.path.abspath(backup_path)}")
    with open(config_path, "r") as f:
        config = json.load(f)
    config["auto_map"] = {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Updated config.json at: {os.path.abspath(config_path)}")


def check_identical_files(path1: Union[str, Path], path2: Union[str, Path]) -> bool:
    path1, path2 = Path(path1), Path(path2)
    if path1.stat().st_size != path2.stat().st_size:
        return False
    return filecmp.cmp(path1, path2, shallow=False)


def _handle_file_sync(curr_filepath: str, checkpoint_filepath: str, file_type: str) -> None:
    if os.path.exists(checkpoint_filepath):
        match = check_identical_files(curr_filepath, checkpoint_filepath)
        if not match:
            print(
                "\n------------------------------------------------------------------------------------------------\n"
                f"Found mismatch between:\n"
                f"Current:   {curr_filepath}\n"
                f"Checkpoint: {checkpoint_filepath}\n"
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{checkpoint_filepath}.back.{timestamp}"
            shutil.copy2(checkpoint_filepath, backup_path)
            print(f"Created backup of original checkpoint file at: {os.path.abspath(backup_path)}")
            shutil.copy2(curr_filepath, checkpoint_filepath)
            print(f"Copied current version to checkpoint at: {os.path.abspath(checkpoint_filepath)}")
            print(
                f"Changes complete. The checkpoint will now use the current version of {file_type}"
                "\n------------------------------------------------------------------------------------------------\n"
            )
    else:
        shutil.copy2(curr_filepath, checkpoint_filepath)
        print(
            "\n------------------------------------------------------------------------------------------------\n"
            f"No {file_type} found in checkpoint directory.\n"
            f"Copied current version from: {curr_filepath}\n"
            f"To checkpoint location: {os.path.abspath(checkpoint_filepath)}"
            "\n------------------------------------------------------------------------------------------------\n"
        )


def check_model_logic_mismatch(pretrained_checkpoint: str) -> None:
    if not os.path.isdir(pretrained_checkpoint):
        return
    curr_files = {"modeling_prismatic.py": None, "configuration_prismatic.py": None}
    for root, _, files in os.walk("./prismatic/"):
        for filename in curr_files.keys():
            if filename in files and curr_files[filename] is None:
                curr_files[filename] = os.path.join(root, filename)
    for filename, curr_filepath in curr_files.items():
        if curr_filepath is None:
            print(f"WARNING: `{filename}` is not found anywhere in the current directory.")
            continue
        checkpoint_filepath = os.path.join(pretrained_checkpoint, filename)
        _handle_file_sync(curr_filepath, checkpoint_filepath, filename)


def load_component_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    state_dict = torch.load(checkpoint_path, weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    assert os.path.isdir(pretrained_checkpoint), f"Checkpoint path must be a directory: {pretrained_checkpoint}"
    checkpoint_files = []
    for filename in os.listdir(pretrained_checkpoint):
        if file_pattern in filename and "checkpoint" in filename:
            full_path = os.path.join(pretrained_checkpoint, filename)
            checkpoint_files.append(full_path)
    assert len(checkpoint_files) == 1, (
        f"Expected exactly 1 {file_pattern} checkpoint but found {len(checkpoint_files)} in directory: {pretrained_checkpoint}"
    )
    return checkpoint_files[0]


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    unnorm_key = cfg.task_suite_name
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"
    cfg.unnorm_key = unnorm_key


DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _load_dataset_stats(vla: torch.nn.Module, checkpoint_path: str) -> None:
    if model_is_on_hf_hub(checkpoint_path):
        dataset_statistics_path = hf_hub_download(
            repo_id=checkpoint_path,
            filename="dataset_statistics.json",
        )
    else:
        dataset_statistics_path = os.path.join(checkpoint_path, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )


MODEL_IMAGE_SIZES = {
    "openvla": 224,
}


def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"


def get_image_resize_size(model_family) -> Union[int, tuple]:
    return MODEL_IMAGE_SIZES[model_family]


def initialize_model(cfg: GenerateConfig):
    model = get_model(cfg)
    model.set_version(cfg.save_version)

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

    return model, action_head, proprio_projector, noisy_action_projector, processor


@draccus.wrap()
def main(cfg: GenerateConfig):
    seed_everything(cfg.seed)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    acc = Accelerator(kwargs_handlers=[kwargs])
    validate_config(cfg)

    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")

    save_path = f'./evaluation_results'
    observation_space = {
        'rgb_obs': ['rgb_static', 'rgb_gripper'],
        'depth_obs': ['depth_static', 'depth_gripper'],
        'state_obs': ['robot_obs'],
        'actions': ['rel_actions'],
        'language': ['language']}

    cf_suffix = f"-cf_{cfg.cf_method}-{cfg.cf_mode}" if cfg.use_cf else ""
    if cfg.use_proprio_cf:
        cf_suffix += f"-proprio_cf_{cfg.cf_method_proprio}"

    eval_dir = save_path + f'/calvin/{current_time}_{cfg.pretrained_checkpoint.split("/")[-1]}{cf_suffix}/'
    os.makedirs(eval_dir, exist_ok=True)
    os.environ["CALVIN_ROOT"] = cfg.calvin_root
    env = make_env(cfg.calvin_dataset, observation_space, DEVICE)

    eva = DualSystemCalvinEvaluation(
        model, proprio_projector, noisy_action_projector, action_head, processor,
        use_x0_prediction=cfg.use_x0_prediction,
        use_cf=cfg.use_cf,
        cf_method=cfg.cf_method,
        cf_mode=cfg.cf_mode,
        cf_guidance_scale=cfg.cf_guidance_scale,
        vlm_effect_threshold=cfg.vlm_effect_threshold,
        cfg_scale=cfg.cfg_scale,
        use_proprio_cf=cfg.use_proprio_cf,
        cf_method_proprio=cfg.cf_method_proprio,
        cfg_scale_proprio=cfg.cfg_scale_proprio,
        cf_verbose=cfg.cf_verbose,
    )

    print(f"[CF Config] use_cf={cfg.use_cf}, cf_method={cfg.cf_method}, cf_mode={cfg.cf_mode}, "
          f"cf_guidance_scale={cfg.cf_guidance_scale}, vlm_effect_threshold={cfg.vlm_effect_threshold}, "
          f"use_proprio_cf={cfg.use_proprio_cf}, cf_method_proprio={cfg.cf_method_proprio}")

    avg_reward = torch.tensor(evaluate_policy(
        eva,
        env,
        eval_dir + 'success_rate.txt',
        eval_dir + 'result.txt',
        acc.num_processes,
        acc.process_index,
        eval_dir=eval_dir,
        ep_len=360,
        num_sequences=1000,
        enrich_lang=cfg.enrich_lang,
        debug=False,
        calvin_root=cfg.calvin_root,
    )).float().mean().to(DEVICE)

    acc.wait_for_everyone()
    avg_reward = acc.gather_for_metrics(avg_reward).mean()
    if acc.is_main_process:
        print('average success rate ', avg_reward)


if __name__ == "__main__":
    main()
