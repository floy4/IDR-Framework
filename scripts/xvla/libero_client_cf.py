#!/usr/bin/env python3
"""
Libero evaluation client with CF Attention Delta Weighting support.

This client uses the /cf_act endpoint with weight_mode parameter to run
counterfactual-guided action generation.

Usage:
    python libero_client_cf.py \
        --server_ip 127.0.0.1 \
        --server_port 9997 \
        --output_dir logs_cf_E \
        --weight_mode E \
        --task_suites libero_goal libero_spatial
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import imageio
import json_numpy
import numpy as np
import requests
import torch  # noqa: F401
import torchvision.transforms as transforms  # noqa: F401
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from evaluation.utils import save_query_image, ensure_query_dir
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv  # noqa: F401
import robosuite.utils.transform_utils as T

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
EPS = 1e-6

LIBERO_DATASETS = {
    "libero_goal": ["libero_goal"],
    "libero_object": ["libero_object"],
    "libero_spatial": ["libero_spatial"],
    "libero_10": ["libero_10"],
    "libero_90": ["libero_90"],
    "libero30": ["libero_goal", "libero_object", "libero_spatial"],
    "libero130": ["libero_goal", "libero_object", "libero_spatial", "libero_10", "libero_90"],
}

LIBERO_DATASETS_HORIZON = {
    "libero_goal": 800,
    "libero_object": 800,
    "libero_spatial": 800,
    "libero_10": 900,
    "libero_90": 800,
    "libero30": 800,
    "libero130": 800,
}

benchmark_dict = benchmark.get_benchmark_dict()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _flip_agentview(img: np.ndarray) -> np.ndarray:
    """Match original code behavior: vertical+horizontal flips."""
    return np.flip(np.flip(img, 0), 1)


# -----------------------------------------------------------------------------
# Action processing
# -----------------------------------------------------------------------------
class LiberoAbsActionProcessor:
    """Helpers to convert between 6D rotation (Zhou et al.) and axis-angle."""

    def Rotate6D_to_AxisAngle(self, r6d: np.ndarray) -> np.ndarray:
        """Convert 6D rotation representation to axis-angle."""
        single = False
        if r6d.ndim == 1:
            r6d = r6d[None, :]
            single = True

        a1 = r6d[:, 0:3]
        a2 = r6d[:, 3:6]

        b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + EPS)
        dot_prod = np.sum(b1 * a2, axis=-1, keepdims=True)
        b2_orth = a2 - dot_prod * b1
        b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + EPS)
        b3 = np.cross(b1, b2, axis=-1)

        R = np.stack([b1, b2, b3], axis=-1)

        axis_angle_list: List[np.ndarray] = []
        for i in range(R.shape[0]):
            quat = T.mat2quat(R[i])
            axis_angle = T.quat2axisangle(quat)
            axis_angle_list.append(axis_angle)

        axis_angle_array = np.stack(axis_angle_list, axis=0)
        return axis_angle_array[0] if single else axis_angle_array

    def Mat_to_Rotate6D(self, R: np.ndarray) -> np.ndarray:
        if R.ndim == 2:
            return np.concatenate([R[:3, 0], R[:3, 1]], axis=-1)
        elif R.ndim == 3:
            return np.concatenate([R[:, :3, 0], R[:, :3, 1]], axis=-1)
        else:
            raise ValueError("Rotation matrix must be (...,3,3)")

    def AxisAngle_to_Rotate6D(self, aa: np.ndarray) -> np.ndarray:
        if aa.ndim == 1:
            return self.Mat_to_Rotate6D(T.quat2mat(T.axisangle2quat(aa)))
        else:
            raise ValueError("Only 1D axis-angle supported here")

    def action_6d_to_axisangle(self, action: np.ndarray) -> np.ndarray:
        """Convert action [..., 3(pos)+6(rot6d)+1(grip)] -> [..., 3(pos)+3(aa)+1(grip)]"""
        if action.ndim == 1:
            final_ori = self.Rotate6D_to_AxisAngle(action[3:9])
            return np.concatenate([action[0:3], final_ori, action[-1:]])
        elif action.ndim == 2:
            final_ori = self.Rotate6D_to_AxisAngle(action[:, 3:9])
            return np.concatenate([action[:, 0:3], final_ori, action[:, -1:]], axis=-1)
        else:
            raise ValueError("Unsupported action shape")


# -----------------------------------------------------------------------------
# HTTP Client Policy with CF Delta Weighting
# -----------------------------------------------------------------------------
class ClientModelCF:
    """HTTP client that queries the /cf_act endpoint with weight_mode support."""

    def __init__(
        self,
        host: str,
        port: int,
        weight_mode: str = "E",
        guidance_scale: float = 0.1,
        effect_threshold: float = 0.5,
        cf_visual_backend: str = "input",
        cf_apply: bool = True,
        save_inputs: bool = True,
    ):
        self.url = f"http://{host}:{port}/cf_act"
        self.weight_mode = weight_mode.upper()
        self.guidance_scale = guidance_scale
        self.effect_threshold = effect_threshold
        self.cf_visual_backend = str(cf_visual_backend).lower()
        self.cf_apply = cf_apply
        self.save_inputs = save_inputs
        self.inputs_dir = None
        self._inputs_rel = ""
        self.processor = LiberoAbsActionProcessor()
        self.reset()

    def reset(self) -> None:
        self.proprio: Optional[np.ndarray] = None
        self.action_plan: Deque[List[float]] = collections.deque()
        self._query_idx: int = -1  # Query counter, incremented on each server call
        self.effects_log: List[Dict[str, float]] = []  # Accumulate effects per query
        self._actions_log: List[Dict] = []  # Accumulate actions per step
        self._query_images = None  # Cache input images for saving

    def _format_query(self, obs: Dict, goal: str) -> Dict:
        main_view = _flip_agentview(obs["agentview_image"])
        wrist_view = obs["robot0_eye_in_hand_image"]

        # Cache input images for saving (before they're serialized)
        if self.save_inputs:
            self._query_images = (main_view, wrist_view)

        closed_loop_proprio = np.concatenate([obs['robo_pos'], obs['robo_ori'], np.array([0.0])], axis=-1)
        closed_loop_proprio = np.concatenate([closed_loop_proprio, np.zeros_like(closed_loop_proprio)], axis=-1)
        if self.proprio is None:
            self.proprio = closed_loop_proprio

        payload = {
            "proprio": json_numpy.dumps(self.proprio),
            "language_instruction": goal,
            "image0": json_numpy.dumps(main_view),
            "image1": json_numpy.dumps(wrist_view),
            "domain_id": 3,
            "steps": 10,
            # CF Delta Weighting parameters
            "weight_mode": self.weight_mode,
            "guidance_scale": self.guidance_scale,
            "effect_threshold": self.effect_threshold,
            "cf_visual_backend": self.cf_visual_backend,
            "cf_apply": self.cf_apply,
        }
        return payload

    def _post(self, payload: Dict) -> np.ndarray:
        try:
            resp = requests.post(self.url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Policy server request failed: {e}") from e

        # Check for errors
        if "error" in data:
            raise RuntimeError(f"Server returned error: {data['error']}")

        action = np.array(data["action"])
        if action.ndim != 2 or action.shape[1] < 10:
            raise RuntimeError(f"Unexpected action shape from server: {action.shape}")

        # Record effects if available (only image_effect and proprio_effect)
        if "effects" in data:
            effects = data["effects"]
            img_paths = []
            if self.save_inputs and self._inputs_rel:
                img_paths = [
                    f"{self._inputs_rel}/query_{self._query_idx}/view0.png",
                    f"{self._inputs_rel}/query_{self._query_idx}/view1.png",
                ]
            self.effects_log.append({
                "image_effect": effects.get("image_effect", 0.0),
                "proprio_effect": effects.get("proprio_effect", 0.0),
                "proprio_input": self.proprio.tolist(),
                "image_paths": img_paths,
            })
            print(f"[CF] effects: image={effects.get('image_effect', 0.0):.4f}, proprio={effects.get('proprio_effect', 0.0):.4f}")

        return action

    def get_effects_log(self) -> List[Dict[str, float]]:
        """Return the accumulated effects log."""
        return self.effects_log

    def clear_effects_log(self) -> None:
        """Clear the effects and actions log."""
        self._query_idx = -1
        self.effects_log = []
        self._actions_log = []

    def get_actions_log(self) -> List[Dict]:
        """Return the accumulated actions log."""
        return self._actions_log

    def set_inputs_dir(self, abs_dir: str, rel_prefix: str = ""):
        """Configure input image saving for the current episode."""
        self.inputs_dir = abs_dir
        self._inputs_rel = rel_prefix

    def step(self, obs: Dict, goal: str) -> np.ndarray:
        if not self.action_plan:
            self._query_idx += 1  # Track which query produced the following actions
            payload = self._format_query(obs, goal)

            # Save input images for this query
            if self.save_inputs and self.inputs_dir is not None and self._query_images is not None:
                query_dir = ensure_query_dir(self.inputs_dir, self._query_idx)
                for i, img in enumerate(self._query_images):
                    save_query_image(img, os.path.join(query_dir, f"view{i}.png"))
                self._query_images = None

            action = self._post(payload)
            self.proprio[:9] = action[-1, :9].copy()

            target_eef = action[:, :3]
            target_axis = self.processor.Rotate6D_to_AxisAngle(action[:, 3:9])
            target_act = action[:, 9:10]
            final_action = np.concatenate([target_eef, target_axis, target_act], axis=-1)

            for row in final_action.tolist():
                self.action_plan.append(row)

        action_predict = np.array(self.action_plan.popleft(), dtype=np.float32)
        gripper = 1.0 if action_predict[-1] > 0.5 else -1.0
        action_predict[-1] = gripper
        # Record per-step action
        self._actions_log.append({
            "query_idx": self._query_idx,
            "action": action_predict.tolist(),
            "gripper": float(gripper),
        })
        return action_predict


# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------
class LIBEROEvalCF:
    """Evaluator for Libero with CF Delta Weighting support."""

    def __init__(
        self,
        task_suite_name: str,
        eval_horizon: int = 600,
        act_type: str = "abs",
        num_episodes: int = 10,
        init_seed: int = 42,
    ) -> None:
        self.task_suite_name = task_suite_name
        self.task_list = LIBERO_DATASETS[task_suite_name]
        self.task_suite_list = [benchmark_dict[task]() for task in self.task_list]
        self.eval_horizon = eval_horizon
        self.num_episodes = num_episodes
        self.init_seed = init_seed
        self.act_type = act_type
        self.processor = LiberoAbsActionProcessor()
        self.base_dir: Path = Path('.')

    def _make_dir(self, save_path: Path) -> None:
        path = save_path / self.task_suite_name
        _ensure_dir(path)
        self.base_dir = path

    def _init_env(self, task_suite, task_id: int = 0, ep: int = 0):
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        print(f"[info] task {task_id}: {task_description}")

        env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 256, "camera_widths": 256}
        env = OffScreenRenderEnv(**env_args)

        env.seed(self.init_seed + ep + 100)
        obs = env.reset()
        init_states = task_suite.get_task_init_states(task_id)
        init_state_id = ep % init_states.shape[0]
        obs = env.set_init_state(init_states[init_state_id])

        for _ in range(10):
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            obs, reward, done, info = env.step(action)

        if self.act_type == 'abs':
            for robot in env.env.robots:
                robot.controller.use_delta = False

        return env, task_description, obs

    def _log_results(self, metrics: Dict) -> None:
        print(metrics)
        save_name = self.base_dir / 'results.json'
        with open(save_name, 'a+', encoding='utf-8') as f:
            f.write(json.dumps(metrics) + "\n")

    def _save_video(self, save_path: Path, images: List[np.ndarray], fps: int = 30) -> None:
        imageio.mimsave(save_path.as_posix(), images, fps=fps)

    def _rollout(self, task_suite, policy: ClientModelCF, task_id: int, ep: int) -> float:
        env, lang, obs = self._init_env(task_suite, task_id, ep)
        images: List[np.ndarray] = []

        # Clear effects log at start of episode
        policy.clear_effects_log()
        # Setup input recording for this episode
        if policy.save_inputs:
            policy.set_inputs_dir(
                os.path.join(self.base_dir, "inputs"),
                os.path.join(self.task_suite_name, "inputs")
            )

        done_flag = False
        step_count = 0
        for _ in tqdm(range(self.eval_horizon), desc=f'{lang}'):
            robo_ori = self.processor.Mat_to_Rotate6D(env.env.robots[0].controller.ee_ori_mat)
            robo_pos = env.env.robots[0].controller.ee_pos
            obs['robo_ori'] = robo_ori
            obs['robo_pos'] = robo_pos

            action = policy.step(obs, lang)

            images.append(_flip_agentview(obs['agentview_image']))
            obs, reward, done, info = env.step(action)
            step_count += 1
            if done:
                done_flag = True
                break

        save_path = self.base_dir / f"{lang}_{ep}.mp4"
        self._save_video(save_path, images, fps=30)

        success = 1.0 if done_flag else 0.0
        metrics = {f'sim/{self.task_suite_name}/{lang}': success}
        self._log_results(metrics)

        # Save effects log to JSON file
        effects_data = {
            "task": lang,
            "task_suite": self.task_suite_name,
            "episode": ep,
            "success": bool(done_flag),
            "total_steps": step_count,
            "effects": [{"step": i, **e} for i, e in enumerate(policy.get_effects_log())],
            "actions": [{"step": i, **a} for i, a in enumerate(policy.get_actions_log())],
        }
        # Compute summary statistics
        effects_log = policy.get_effects_log()
        if effects_log:
            image_effects = [e["image_effect"] for e in effects_log]
            proprio_effects = [e["proprio_effect"] for e in effects_log]
            effects_data["summary"] = {
                "avg_image_effect": float(np.mean(image_effects)),
                "avg_proprio_effect": float(np.mean(proprio_effects)),
                "max_image_effect": float(np.max(image_effects)),
                "max_proprio_effect": float(np.max(proprio_effects)),
                "num_inferences": len(effects_log),
            }
        effects_save_path = self.base_dir / f"effects_{lang}_{ep}.json"
        with open(effects_save_path, 'w', encoding='utf-8') as f:
            json.dump(effects_data, f, indent=2)

        env.close()
        return success

    def eval_episodes(self, policy: ClientModelCF, save_path: Path) -> float:
        self._make_dir(save_path)

        rews: List[float] = []
        for task_suite in self.task_suite_list:
            for task_id in tqdm(range(len(task_suite.tasks)), desc="Evaluating tasks"):
                for ep in range(self.num_episodes):
                    policy.reset()
                    rew = self._rollout(task_suite, policy, task_id, ep)
                    rews.append(rew)

        eval_rewards = float(sum(rews) / max(len(rews), 1))
        metrics = {f'sim_summary/{self.task_suite_name}/all': eval_rewards}
        self._log_results(metrics)
        return eval_rewards


# -----------------------------------------------------------------------------
# Batch evaluator
# -----------------------------------------------------------------------------

def eval_libero_cf(
    agent: ClientModelCF,
    save_path: Path,
    num_episodes: int = 10,
    init_seed: int = 42,
    act_type: str = 'abs',
    task_suites: Iterable[str] = ("libero_goal", "libero_spatial", "libero_10"),
) -> Dict[str, float]:
    result_dict: Dict[str, float] = {}
    for suite_name in task_suites:
        horizon = LIBERO_DATASETS_HORIZON[suite_name]
        evaluator = LIBEROEvalCF(
            task_suite_name=suite_name,
            eval_horizon=horizon,
            act_type=act_type,
            num_episodes=num_episodes,
            init_seed=init_seed,
        )
        eval_rewards = evaluator.eval_episodes(agent, save_path=save_path)
        result_dict[suite_name] = eval_rewards

    with open((save_path / "results.json").as_posix(), "a+", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)
        f.write("\n")
    return result_dict


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("LIBERO CF Evaluation Client")
    
    # Connection options
    parser.add_argument("--connection_info", type=str, default=None,
                        help="Path to server info.json")
    parser.add_argument("--server_ip", type=str, default=None)
    parser.add_argument("--server_port", type=int, default=None)

    # CF options
    parser.add_argument("--weight_mode", type=str, default="E",
                        choices=["A", "B", "C", "D", "E", "F", "G", "H"],
                        help="CF delta weighting mode")
    parser.add_argument("--guidance_scale", type=float, default=0.1,
                        help="CF guidance scale")
    parser.add_argument("--effect_threshold", type=float, default=0.5,
                        help="Effect threshold for fallback")
    parser.add_argument("--cf_visual_backend", type=str, default="hybrid",
                        choices=["input", "mask", "hybrid"],
                        help="Visual CF backend selection")
    parser.add_argument("--cf_apply", type=bool, default=True,
                        help="Apply CF delta to actions (True) or use baseline (False)")
    parser.add_argument("--save_inputs", type=bool, default=True,
                        help="Save input images and proprio for each query")

    # Eval options
    parser.add_argument("--output_dir", type=str, default="logs_cf/",
                        help="Directory for saving evaluation videos and logs")
    parser.add_argument("--task_suites", nargs='+', 
                        default=["libero_10", "libero_spatial", "libero_goal", "libero_object"],
                        help="Libero suites to evaluate")
    parser.add_argument("--eval_time", type=int, default=50, help="Episodes per task")
    parser.add_argument("--init_seed", type=int, default=42)
    parser.add_argument("--act_type", type=str, default="abs", choices=["abs", "rel"])

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    _ensure_dir(out_dir)

    print("🚀 [CF Client] Starting LIBERO CF evaluation client...")
    print(
        f"📊 CF Mode: weight_mode={args.weight_mode}, "
        f"guidance_scale={args.guidance_scale}, cf_visual_backend={args.cf_visual_backend}"
    )

    # Load connection info
    if args.connection_info is not None:
        info_path = Path(args.connection_info)
        print(f"🔍 Waiting for connection info file: {info_path}")
        spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while not info_path.exists():
            sys.stdout.write(f"\r{spinner[i % len(spinner)]} Waiting for server...")
            sys.stdout.flush()
            time.sleep(0.5)
            i += 1
        print("\n✅ Connection info found!")
        with open(info_path, "r", encoding="utf-8") as f:
            infos = json.load(f)
        host, port = infos["host"], int(infos["port"])
    else:
        if not args.server_ip or not args.server_port:
            print("❌ Must specify either --connection_info or both --server_ip and --server_port.")
            sys.exit(1)
        host, port = args.server_ip, args.server_port

    # Initialize CF client
    print(f"🔗 Connecting to server at {host}:{port} ...")
    client = ClientModelCF(
        host=host,
        port=port,
        weight_mode=args.weight_mode,
        guidance_scale=args.guidance_scale,
        effect_threshold=args.effect_threshold,
        cf_visual_backend=args.cf_visual_backend,
        cf_apply=args.cf_apply,
        save_inputs=args.save_inputs,
    )
    print("✅ CF Client initialized!")

    # Run evaluation
    print("🎯 Starting LIBERO CF evaluation...")
    print(f"📁 Results saved to: {out_dir.resolve()}")
    print("-" * 88)
    print(f"weight_mode: {args.weight_mode}")
    print(f"cf_visual_backend: {args.cf_visual_backend}")
    print(f"task_suites: {args.task_suites}")
    print(f"episodes: {args.eval_time}")
    print("-" * 88)

    try:
        eval_results = eval_libero_cf(
            agent=client,
            save_path=out_dir,
            init_seed=args.init_seed,
            num_episodes=args.eval_time,
            task_suites=args.task_suites,
            act_type=args.act_type,
        )
    except KeyboardInterrupt:
        print("\n🛑 Interrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(2)

    print("\n✅ All evaluations completed!")
    print(f"📊 Summary: {json.dumps(eval_results, indent=2)}")