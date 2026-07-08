from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        use_cf_sampling: bool = False, ## modified
        use_cf_feature_sampling: bool = False, ## modified: feature-level CF
        use_cf_token_sampling: bool = False, ## modified: token-level CF
        use_cf_pixel_sampling: bool = False, ## modified: pixel-level CF
        cf_attn_layer_index: int = 16, ## modified: attention-guided CF layer index
        cf_attn_time: float = 1.0, ## modified: attention-guided CF time
        cf_attn_topk_ratio: float = 0.2, ## modified: attention-guided CF topk ratio
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            use_cf_sampling: Whether to use model.sample_actions_with_cf (input-level CF) if available.
            use_cf_feature_sampling: Whether to use model.sample_actions_with_feature_cf (feature-level CF) if available.
            use_cf_token_sampling: Whether to use model.sample_actions_with_single_step_token_cf (token-level CF).
                                   Blocks high-attention image tokens in cf_attn_mask.
            use_cf_pixel_sampling: Whether to use model.sample_actions_with_pixel_level_cf (pixel-level CF).
                                   Zeroes high-attention pixel patches in raw images and re-encodes.
            cf_attn_layer_index: Transformer layer index for attention-guided CF (default 16).
            cf_attn_time: Diffusion time point for attention extraction (default 1.0).
            cf_attn_topk_ratio: Ratio of top-k high-attention tokens/pixels to intervene (default 0.2).

        Note:
            CF mode priority: pixel-level > token-level > feature-level > input-level > baseline.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._use_cf_sampling = use_cf_sampling ## modified
        self._use_cf_feature_sampling = use_cf_feature_sampling ## modified
        self._use_cf_token_sampling = use_cf_token_sampling ## modified
        self._use_cf_pixel_sampling = use_cf_pixel_sampling ## modified

        ## modified: CF sampling 默认不强制返回 metrics；仅在外部显式传入时启用
        use_any_cf = use_cf_feature_sampling or use_cf_sampling or use_cf_token_sampling or use_cf_pixel_sampling
        if use_any_cf:
            self._sample_kwargs.setdefault("return_cf_metrics", False)

        ## modified: select appropriate sample function (priority: pixel > token > feature > input)
        sample_fn = None
        if use_cf_pixel_sampling and hasattr(model, "sample_actions_with_pixel_level_cf"):
            sample_fn = model.sample_actions_with_pixel_level_cf
            logging.info("Using pixel-level counterfactual sampling (attention-guided pixel zeroing)")
        elif use_cf_token_sampling and hasattr(model, "sample_actions_with_single_step_token_cf"):
            sample_fn = model.sample_actions_with_single_step_token_cf
            logging.info("Using token-level counterfactual sampling (attention-guided token masking)")
        elif use_cf_feature_sampling and hasattr(model, "sample_actions_with_feature_cf"):
            sample_fn = model.sample_actions_with_feature_cf
            logging.info("Using feature-level counterfactual sampling")
        elif use_cf_sampling and hasattr(model, "sample_actions_with_cf"):
            sample_fn = model.sample_actions_with_cf
            logging.info("Using input-level counterfactual sampling")
        else:
            if use_any_cf:
                logging.warning(
                    "CF sampling requested but model has no compatible CF method; "
                    "fallback to sample_actions"
                )
            sample_fn = model.sample_actions

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = sample_fn ## modified
        else:
            # JAX model setup
            # 收集所有静态参数名（包括 sample_kwargs 中的和在 infer 中动态添加的）
            # 注意: attention-guided CF 的 layer_index 等参数必须静态，避免在
            # decode_with_last_query_attention 中被追踪为 weak_i32[]。
            cf_static_numeric_keys = {
                "layer_index",
                "attention_time",
                "token_topk_ratio",
                "pixel_topk_ratio",
                "effect_threshold",
                "cf_guidance_scale",
                "visualization_frequency",
            }
            static_argnames = tuple(
                key
                for key, value in self._sample_kwargs.items()
                if isinstance(value, (str, bool)) or key in cf_static_numeric_keys
            )
            # CF sampling 时，return_cf_metrics 也是静态参数
            if use_any_cf:
                # 确保 return_cf_metrics 被包含在静态参数中
                if "return_cf_metrics" not in static_argnames:
                    static_argnames = tuple(list(static_argnames) + ["return_cf_metrics"])
            if static_argnames:
                self._sample_actions = nnx_utils.module_jit(sample_fn, static_argnames=static_argnames) ## modified
            else:
                self._sample_actions = nnx_utils.module_jit(sample_fn) ## modified

            self._rng = rng or jax.random.key(0)

        # Track step and episode indices for visualization
        self._step_idx = 0
        self._episode_idx = 0

    def _maybe_visualize_attention(
        self,
        sample_rng: at.KeyArrayLike,
        observation: _model.Observation,
        sample_kwargs: dict[str, Any],
    ) -> None:
        """Run attention visualization outside the JIT-compiled sampling path."""
        if self._is_pytorch_model or not sample_kwargs.get("visualize_attention", False):
            return

        if not (self._use_cf_token_sampling or self._use_cf_pixel_sampling):
            return

        if not hasattr(self._model, "extract_single_step_attention_map") or not hasattr(self._model, "_visualize_attention"):
            return

        frequency = int(sample_kwargs.get("visualization_frequency", 10))
        if frequency <= 0 or self._step_idx % frequency != 0:
            return

        layer_index = int(sample_kwargs.get("layer_index", 16))
        attention_time = float(sample_kwargs.get("attention_time", 1.0))
        visualization_dir = str(sample_kwargs.get("visualization_dir", "results/attention_vis"))
        cf_mode = str(sample_kwargs.get("cf_mode", "E"))
        noise = sample_kwargs.get("noise")

        try:
            attn_data = self._model.extract_single_step_attention_map(
                sample_rng,
                observation,
                layer_index=layer_index,
                time=attention_time,
                noise=noise,
            )
            self._model._visualize_attention(
                attn_data,
                observation,
                visualization_dir,
                self._step_idx,
                self._episode_idx,
                layer_index,
                cf_mode,
            )
        except Exception as exc:
            logging.warning("Attention visualization skipped due to error: %s", exc)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)

        # Run visualization outside the jitted sampling function to avoid traced Python control flow.
        self._maybe_visualize_attention(sample_rng_or_pytorch_device, observation, sample_kwargs)

        # Increment step counter
        self._step_idx += 1

        # return_cf_metrics 由 sample_kwargs 控制

        start_time = time.monotonic()

        # 调用采样函数，可能返回 tuple (actions, metrics)
        sample_result = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)

        # 处理返回值：可能是 tuple (actions, metrics) 或仅 actions
        if isinstance(sample_result, tuple):
            actions, cf_metrics = sample_result
            outputs = {
                "state": inputs["state"],
                "actions": actions,
            }
            # 将 CF metrics 转换为 Python 标量/列表，避免后续序列化问题
            cf_metrics_np = {}
            for k, v in cf_metrics.items():
                if isinstance(v, torch.Tensor):
                    v_cpu = v.detach().cpu()
                    cf_metrics_np[k] = v_cpu.item() if v_cpu.ndim == 0 else v_cpu.numpy().tolist()
                else:
                    v_np = np.asarray(v)
                    cf_metrics_np[k] = v_np.item() if v_np.shape == () else v_np.tolist()

            # 在非 JIT 路径补充可读元信息
            if self._use_cf_pixel_sampling:
                cf_metrics_np.setdefault("cf_level", "pixel")
            elif self._use_cf_token_sampling:
                cf_metrics_np.setdefault("cf_level", "token")
            elif self._use_cf_feature_sampling:
                cf_metrics_np.setdefault("cf_level", "feature")
            elif self._use_cf_sampling:
                cf_metrics_np.setdefault("cf_level", "input")

            if "cf_mode" in sample_kwargs:
                cf_metrics_np.setdefault("cf_mode", sample_kwargs["cf_mode"])

            if "cf_guidance_scale" in sample_kwargs:
                cf_metrics_np.setdefault("cf_guidance_scale", float(sample_kwargs["cf_guidance_scale"]))

            if "vlm_effect_upper_threshold" in sample_kwargs:
                cf_metrics_np.setdefault("vlm_effect_threshold", float(sample_kwargs["vlm_effect_upper_threshold"]))

            # Attention-guided CF specific metadata
            if self._use_cf_token_sampling or self._use_cf_pixel_sampling:
                cf_metrics_np.setdefault("cf_attn_layer_index", sample_kwargs.get("layer_index", 16))
                cf_metrics_np.setdefault("cf_attn_time", sample_kwargs.get("attention_time", 1.0))
                cf_metrics_np.setdefault("cf_attn_topk_ratio", sample_kwargs.get("token_topk_ratio", sample_kwargs.get("pixel_topk_ratio", 0.2)))
                cf_metrics_np.setdefault("effect_threshold", sample_kwargs.get("effect_threshold", 0.5))

            outputs["cf_metrics"] = cf_metrics_np
        else:
            outputs = {
                "state": inputs["state"],
                "actions": sample_result,
            }

        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs["state"] = np.asarray(outputs["state"][0, ...].detach().cpu())
            outputs["actions"] = np.asarray(outputs["actions"][0, ...].detach().cpu())
        else:
            outputs["state"] = np.asarray(outputs["state"][0, ...])
            outputs["actions"] = np.asarray(outputs["actions"][0, ...])

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def reset_step_counter(self) -> None:
        """Reset step counter for a new episode."""
        self._step_idx = 0

    def set_episode_idx(self, episode_idx: int) -> None:
        """Set current episode index for visualization."""
        self._episode_idx = episode_idx

    def reset_counters(self) -> None:
        """Reset both step and episode counters."""
        self._step_idx = 0
        self._episode_idx = 0


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
