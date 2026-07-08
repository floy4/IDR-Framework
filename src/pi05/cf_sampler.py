"""Counterfactual sampler for Pi0/Pi05 models.

This module provides a wrapper class that enables counterfactual sampling
from Pi0/Pi05 models without modifying the original model code.
"""

from typing import Optional, Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.pi0 import Pi0
from openpi.models import model as _model
Observation = _model.Observation
from openpi.shared import nnx_utils
from .modality_bounds import ModalityBounds
from .attention_mask import (
    CfAttnMode,
    compute_modality_effect,
    get_cf_modes_for_analysis,
)


@dataclass
class CFAnalysisResult:
    """Result container for counterfactual analysis.
    
    Contains actions from different CF modes and computed modality effects.
    """
    
    # Actions from each mode
    actions_by_mode: dict[str, jnp.ndarray]
    
    # Computed modality effects
    effects: dict[str, jnp.ndarray]
    
    # Modality bounds used for the analysis
    modality_bounds: ModalityBounds
    
    # Metadata
    num_steps: int
    action_dim: int
    action_horizon: int
    
    def get_baseline_actions(self) -> jnp.ndarray:
        """Get actions from baseline (normal) mode."""
        return self.actions_by_mode.get("base", None)
    
    def get_effect(self, modality: str) -> Optional[jnp.ndarray]:
        """Get the effect size for a specific modality.
        
        Args:
            modality: One of "image", "language", "state".
        
        Returns:
            Effect size array or None if not computed.
        """
        return self.effects.get(f"{modality}_effect", None)


class CfSampler:
    """Counterfactual sampler wrapper for Pi0 models.
    
    This class wraps a Pi0 model and provides methods for sampling actions
    with different attention mask configurations to analyze modality contributions.
    
    The key idea is to:
    1. Use the model's embed_prefix to get token embeddings
    2. Create custom attention masks that block specific modalities
    3. Run the model's sampling process with these modified masks
    
    Example:
        >>> # Create sampler from existing model
        >>> sampler = CfSampler(model)
        >>> 
        >>> # Run CF analysis
        >>> result = sampler.sample_with_cf(
        ...     rng, observation,
        ...     cf_modes=[CfAttnMode.BASE, CfAttnMode.NO_IMAGE],
        ... )
        >>> 
        >>> # Compare actions
        >>> baseline = result.get_baseline_actions()
        >>> no_image = result.actions_by_mode["no_image"]
        >>> image_effect = result.get_effect("image")
    """
    
    def __init__(
        self,
        model: Pi0,
        *,
        default_num_steps: int = 10,
        default_action_dim: Optional[int] = None,
        default_action_horizon: Optional[int] = None,
    ):
        """Initialize the CF sampler.
        
        Args:
            model: Pi0 model instance to wrap.
            default_num_steps: Default number of flow matching steps.
            default_action_dim: Default action dimension (inferred from model if None).
            default_action_horizon: Default action horizon (inferred from model if None).
        """
        self._model = model

        # Reuse the same jitted sampling path as the main policy code path when possible.
        # Falling back to the raw method keeps compatibility for non-NNX models.
        self._sample_actions = self._model.sample_actions
        try:
            self._sample_actions = nnx_utils.module_jit(self._model.sample_actions)
        except Exception:
            pass

        self._prepare_prefix_for_sampling = None
        if hasattr(self._model, "prepare_prefix_for_sampling"):
            self._prepare_prefix_for_sampling = self._model.prepare_prefix_for_sampling

        self._sample_actions_from_precomputed_prefix = None
        if hasattr(self._model, "sample_actions_from_precomputed_prefix"):
            self._sample_actions_from_precomputed_prefix = self._model.sample_actions_from_precomputed_prefix
            try:
                self._sample_actions_from_precomputed_prefix = nnx_utils.module_jit(
                    self._model.sample_actions_from_precomputed_prefix
                )
            except Exception:
                pass

        self._sample_actions_triplet_shared_prefix = None
        if hasattr(self._model, "sample_actions_triplet_shared_prefix"):
            self._sample_actions_triplet_shared_prefix = self._model.sample_actions_triplet_shared_prefix
            try:
                self._sample_actions_triplet_shared_prefix = nnx_utils.module_jit(
                    self._model.sample_actions_triplet_shared_prefix
                )
            except Exception:
                pass
        
        # Infer dimensions from model if not provided
        self._action_dim = default_action_dim or getattr(model, "action_dim", 7)
        self._action_horizon = default_action_horizon or getattr(model, "action_horizon", 16)
        self._default_num_steps = default_num_steps

        # Cache bounds derived from embed_prefix to avoid repeated expensive prefix
        # embedding calls for the same observation schema.
        self._modality_bounds_cache: dict[tuple, tuple[ModalityBounds, int]] = {}
        
        # Get image token count per camera (SigLIP produces fixed-size outputs)
        # SigLIP So400m/14 with 224x224 input produces 256 tokens
        # (224/14)^2 = 16^2 = 256 patches
        self._image_tokens_per_camera = 256
    
    def sample_with_cf(
        self,
        rng: jax.Array,
        observation: Observation,
        *,
        cf_modes: Optional[list[CfAttnMode]] = None,
        num_steps: Optional[int] = None,
        modality_bounds: Optional[ModalityBounds] = None,
        return_effects: bool = True,
        effect_metric: str = "l2",
    ) -> CFAnalysisResult:
        """Sample actions with counterfactual attention mask modifications.
        
        This is the main entry point for CF analysis. It samples actions
        using different attention mask configurations and computes the
        effect of each modality.
        
        Args:
            rng: JAX random key.
            observation: Observation object containing images, state, and prompt.
            cf_modes: List of CF modes to evaluate. If None, uses recommended modes.
            num_steps: Number of flow matching steps. If None, uses default.
            modality_bounds: Pre-computed modality bounds. If None, computed from observation.
            return_effects: Whether to compute modality effect sizes.
            effect_metric: Metric for effect computation ("l2", "l1", "cosine").
        
        Returns:
            CFAnalysisResult containing actions and effects.
        
        Note:
            This method does NOT modify the original model. It creates
            modified attention masks externally and uses them during sampling.
        """
        cf_modes = cf_modes or get_cf_modes_for_analysis()
        num_steps = num_steps or self._default_num_steps
        
        # Compute modality bounds if not provided
        if modality_bounds is None:
            modality_bounds = self._compute_modality_bounds(observation)

        prefix_context = self._maybe_prepare_prefix_context(observation)
        if prefix_context is not None:
            observation_for_sampling = prefix_context[0]
        else:
            observation_for_sampling = observation
        
        # Sample actions for each mode
        actions_by_mode = {}
        
        for mode in cf_modes:
            rng, sub_rng = jax.random.split(rng)
            actions = self._sample_single_mode(
                sub_rng,
                observation_for_sampling,
                mode,
                modality_bounds,
                num_steps,
                prefix_context=prefix_context,
            )
            actions_by_mode[mode.value] = actions
        
        # Compute effects
        effects = {}
        if return_effects and "base" in actions_by_mode:
            baseline = actions_by_mode["base"]
            
            # Compute individual modality effects
            if "no_image" in actions_by_mode:
                effects["image_effect"] = compute_modality_effect(
                    baseline, actions_by_mode["no_image"], metric=effect_metric
                )
            if "no_lang" in actions_by_mode:
                effects["language_effect"] = compute_modality_effect(
                    baseline, actions_by_mode["no_lang"], metric=effect_metric
                )
            if "no_state" in actions_by_mode:
                effects["state_effect"] = compute_modality_effect(
                    baseline, actions_by_mode["no_state"], metric=effect_metric
                )
            
            # Compute single-modality-only effects
            if "image_only" in actions_by_mode:
                effects["image_only_effect"] = compute_modality_effect(
                    baseline, actions_by_mode["image_only"], metric=effect_metric
                )
            if "lang_only" in actions_by_mode:
                effects["language_only_effect"] = compute_modality_effect(
                    baseline, actions_by_mode["lang_only"], metric=effect_metric
                )
            if "state_only" in actions_by_mode:
                effects["state_only_effect"] = compute_modality_effect(
                    baseline, actions_by_mode["state_only"], metric=effect_metric
                )
        
        return CFAnalysisResult(
            actions_by_mode=actions_by_mode,
            effects=effects,
            modality_bounds=modality_bounds,
            num_steps=num_steps,
            action_dim=self._action_dim,
            action_horizon=self._action_horizon,
        )

    def sample_with_cf_reweight(
        self,
        rng: jax.Array,
        observation: Observation,
        *,
        num_steps: Optional[int] = None,
        modality_bounds: Optional[ModalityBounds] = None,
        cf_guidance_scale: float = 0.1,
        state_weight_base: float = 0.05,
        use_state_adaptive: bool = True,
        effect_threshold: float = 0.5,
        return_metrics: bool = False,
    ) -> jnp.ndarray:
        """Sample actions with attention-level CF reweighting.

        This method implements the attention-level counterfactual reweighting:
        1. Run three forward passes: BASE, NO_IMAGE, NO_STATE
        2. Compute effects: effect_image, effect_state
        3. Reweight baseline with computed deltas

        Args:
            rng: JAX random key.
            observation: Observation input.
            num_steps: Number of sampling steps.
            modality_bounds: Modality position bounds.
            cf_guidance_scale: Guidance scale for image effect (default 0.1).
            state_weight_base: Base weight for state effect (default 0.05).
            use_state_adaptive: Whether to use adaptive state weight based on effect ratio.
            effect_threshold: Threshold for VLM effect to fallback to baseline.
            return_metrics: Whether to return CF metrics along with actions.

        Returns:
            Reweighted actions, or baseline if effects are too large.
            If return_metrics=True, returns (actions, metrics_dict).
        """
        num_steps = num_steps or self._default_num_steps

        if modality_bounds is None:
            modality_bounds = self._compute_modality_bounds(observation)

        # Three forward passes with attention-level CF
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        if self._sample_actions_triplet_shared_prefix is not None:
            from .attention_mask import create_cf_attn_mask_for_sampling

            actual_modality_bounds, suffix_len = self._get_or_compute_modality_bounds(observation)
            batch_size = observation.state.shape[0]

            cf_attn_mask_no_image = create_cf_attn_mask_for_sampling(
                prefix_len=actual_modality_bounds.prefix_len,
                suffix_len=suffix_len,
                modality_bounds=actual_modality_bounds,
                cf_mode=CfAttnMode.NO_IMAGE,
                batch_size=batch_size,
            )
            cf_attn_mask_no_state = create_cf_attn_mask_for_sampling(
                prefix_len=actual_modality_bounds.prefix_len,
                suffix_len=suffix_len,
                modality_bounds=actual_modality_bounds,
                cf_mode=CfAttnMode.NO_STATE,
                batch_size=batch_size,
            )

            actions_base, actions_no_image, actions_no_state = self._sample_actions_triplet_shared_prefix(
                rng1,
                rng2,
                rng3,
                observation,
                cf_attn_mask_no_image,
                cf_attn_mask_no_state,
                num_steps=num_steps,
            )
        else:
            # Compatibility fallback for models without shared-prefix support.
            actions_base = self._sample_single_mode(rng1, observation, CfAttnMode.BASE, modality_bounds, num_steps)
            actions_no_image = self._sample_single_mode(
                rng2,
                observation,
                CfAttnMode.NO_IMAGE,
                modality_bounds,
                num_steps,
            )
            actions_no_state = self._sample_single_mode(
                rng3,
                observation,
                CfAttnMode.NO_STATE,
                modality_bounds,
                num_steps,
            )

        # Compute effects (L2 distance from baseline)
        def compute_effect(base, cf):
            diff = base - cf
            batch = diff.shape[0]
            diff_flat = jnp.reshape(diff, (batch, -1))
            return jnp.mean(jnp.linalg.norm(diff_flat, axis=-1))

        effect_image = compute_effect(actions_base, actions_no_image)
        effect_state = compute_effect(actions_base, actions_no_state)

        # Compute deltas (direction to move from CF to baseline)
        delta_image = actions_base - actions_no_image
        delta_state = actions_base - actions_no_state

        # Adaptive state weight based on effect ratio
        eps = 1e-6
        if use_state_adaptive:
            state_ratio = effect_state / (effect_image + eps)
            state_weight = state_weight_base * jnp.minimum(1.0, state_ratio)
        else:
            state_weight = state_weight_base

        # Clip state delta to prevent over-correction
        delta_state_clipped = jnp.clip(delta_state, -0.1, 0.1)

        # Reweighted actions
        actions_reweighted = actions_base + cf_guidance_scale * delta_image + state_weight * delta_state_clipped

        # Fallback to baseline if image effect is too large
        use_baseline = effect_image > effect_threshold
        actions_final = jnp.where(use_baseline, actions_base, actions_reweighted)

        if return_metrics:
            metrics = {
                "effect_image": float(effect_image),
                "effect_state": float(effect_state),
                "state_weight": float(state_weight),
                "use_baseline": bool(use_baseline),
                "cf_guidance_scale": cf_guidance_scale,
            }
            return actions_final, metrics

        return actions_final

    def sample_baseline(
        self,
        rng: jax.Array,
        observation: Observation,
        *,
        num_steps: Optional[int] = None,
    ) -> jnp.ndarray:
        """Sample baseline actions without any CF intervention."""
        num_steps = num_steps or self._default_num_steps
        return self._sample_actions(rng, observation, num_steps=num_steps)

    def _maybe_prepare_prefix_context(
        self,
        observation: Observation,
    ) -> tuple[Observation, jnp.ndarray, Any] | None:
        """Prepare reusable prefix context if the wrapped model supports it."""
        if self._prepare_prefix_for_sampling is None:
            return None
        return self._prepare_prefix_for_sampling(observation)

    def _sample_single_mode(
        self,
        rng: jax.Array,
        observation: Observation,
        cf_mode: CfAttnMode,
        modality_bounds: ModalityBounds,
        num_steps: int,
        *,
        prefix_context: tuple[Observation, jnp.ndarray, Any] | None = None,
    ) -> jnp.ndarray:
        """Sample actions for a single CF mode.
        
        This method:
        1. Embeds the prefix using the model
        2. Creates appropriate attention masks
        3. Runs flow matching to sample actions
        
        Args:
            rng: JAX random key.
            observation: Observation input.
            cf_mode: Counterfactual attention mode.
            modality_bounds: Modality position bounds.
            num_steps: Number of sampling steps.
        
        Returns:
            Sampled actions as jnp.ndarray.
        """
        # For BASE mode, use the original model's sample_actions method
        if cf_mode == CfAttnMode.BASE:
            if (
                prefix_context is not None
                and self._sample_actions_from_precomputed_prefix is not None
            ):
                prepared_observation, prefix_mask, kv_cache = prefix_context
                return self._sample_actions_from_precomputed_prefix(
                    rng,
                    prepared_observation,
                    prefix_mask,
                    kv_cache,
                    num_steps=num_steps,
                )
            return self._sample_actions(rng, observation, num_steps=num_steps)
        
        # For other modes, we need to intercept and modify the attention mask
        # This requires a more involved approach since we can't directly modify
        # the model's internal attention mask creation
        
        # Strategy: Create a hook/wrapper that modifies the attention mask
        # during the model's forward pass
        
        # Since we can't modify the original model, we'll use a workaround:
        # 1. Extract the prefix embeddings using embed_prefix
        # 2. Create our own attention mask
        # 3. Implement our own sampling loop that uses these
        
        return self._sample_with_custom_attn(
            rng,
            observation,
            cf_mode,
            modality_bounds,
            num_steps,
            prefix_context=prefix_context,
        )

    def _make_observation_signature(self, observation: Observation) -> tuple:
        """Build a lightweight cache key for modality-bound inference."""
        image_signature = ()
        if observation.images is not None:
            image_signature = tuple(
                (name, tuple(observation.images[name].shape[-3:]))
                for name in sorted(observation.images.keys())
            )

        prompt_len = int(observation.tokenized_prompt.shape[-1]) if observation.tokenized_prompt is not None else -1
        state_dim = int(observation.state.shape[-1]) if observation.state is not None else -1
        is_pi05 = bool(getattr(self._model, "pi05", False))

        return (is_pi05, image_signature, prompt_len, state_dim)

    def _get_or_compute_modality_bounds(
        self,
        observation: Observation,
        *,
        actual_prefix_len: Optional[int] = None,
    ) -> tuple[ModalityBounds, int]:
        """Get cached modality bounds and suffix length, or compute once per schema."""
        cache_key = self._make_observation_signature(observation)
        cached = self._modality_bounds_cache.get(cache_key)
        if cached is not None:
            return cached

        # Get actual prefix length by calling embed_prefix once for this schema if needed.
        if actual_prefix_len is None:
            prefix_tokens, _, _ = self._model.embed_prefix(observation)
            actual_prefix_len = int(prefix_tokens.shape[1])

        # Count number of images in observation.
        num_images = len(observation.images) if observation.images else 0

        # SigLIP So400m/14 produces 256 tokens per 224x224 image.
        image_tokens_per_camera = 256

        # Compute image bounds - images come first in the token sequence.
        image_bounds = {}
        current_pos = 0
        if observation.images is not None:
            for name in observation.images:
                image_bounds[name] = (current_pos, current_pos + image_tokens_per_camera)
                current_pos += image_tokens_per_camera

        image_end_pos = num_images * image_tokens_per_camera

        # Language and state bounds.
        if hasattr(self._model, "pi05") and self._model.pi05:
            # Pi05: state is discretized in prefix tokens.
            state_dim = observation.state.shape[-1] if observation.state is not None else 14
            estimated_state_tokens = state_dim + 10

            lang_start = image_end_pos
            lang_end = max(lang_start, actual_prefix_len - estimated_state_tokens)

            state_start = lang_end
            state_end = actual_prefix_len
        else:
            # Pi0: state is NOT in prefix (continuous input in suffix).
            lang_start = image_end_pos
            lang_end = actual_prefix_len
            state_start = actual_prefix_len
            state_end = actual_prefix_len

        actual_modality_bounds = ModalityBounds(
            image_bounds=image_bounds,
            language_bounds=(lang_start, lang_end),
            state_bounds=(state_start, state_end),
            prefix_len=actual_prefix_len,
            suffix_start=actual_prefix_len,
        )

        actual_action_horizon = getattr(self._model, "action_horizon", 16)
        if hasattr(self._model, "pi05") and self._model.pi05:
            suffix_len = actual_action_horizon
        else:
            suffix_len = actual_action_horizon + 1

        result = (actual_modality_bounds, suffix_len)
        self._modality_bounds_cache[cache_key] = result
        return result

    def _sample_with_custom_attn(
        self,
        rng: jax.Array,
        observation: Observation,
        cf_mode: CfAttnMode,
        modality_bounds: ModalityBounds,
        num_steps: int,
        *,
        prefix_context: tuple[Observation, jnp.ndarray, Any] | None = None,
    ) -> jnp.ndarray:
        """Sample with custom attention mask (full implementation).

        This method uses the modified sample_actions method that accepts
        a cf_attn_mask parameter, enabling attention-level counterfactual
        analysis without modifying the model's internal code.

        Args:
            rng: JAX random key.
            observation: Observation input.
            cf_mode: Counterfactual attention mode.
            modality_bounds: Modality position bounds (may be updated with actual lengths).
            num_steps: Number of sampling steps.

        Returns:
            Sampled actions with CF attention mask applied.
        """
        from .attention_mask import create_cf_attn_mask_for_sampling

        if prefix_context is not None:
            prepared_observation, prefix_mask, kv_cache = prefix_context
            observation_for_sampling = prepared_observation
            known_prefix_len = int(prefix_mask.shape[1])
        else:
            prepared_observation = None
            prefix_mask = None
            kv_cache = None
            observation_for_sampling = observation
            known_prefix_len = None

        actual_modality_bounds, suffix_len = self._get_or_compute_modality_bounds(
            observation_for_sampling,
            actual_prefix_len=known_prefix_len,
        )
        actual_prefix_len = actual_modality_bounds.prefix_len

        batch_size = observation_for_sampling.state.shape[0]

        # Create CF attention mask for sampling with actual lengths
        cf_attn_mask = create_cf_attn_mask_for_sampling(
            prefix_len=actual_prefix_len,
            suffix_len=suffix_len,
            modality_bounds=actual_modality_bounds,
            cf_mode=cf_mode,
            batch_size=batch_size,
        )

        # Debug: print sequence dimensions
        import logging
        logger = logging.getLogger("openpi.cf_attention")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("CF Attention sampling debug info:")
            logger.debug("  actual_prefix_len: %s", actual_prefix_len)
            logger.debug("  suffix_len: %s", suffix_len)
            logger.debug("  total_seq_len: %s", actual_prefix_len + suffix_len)
            logger.debug("  cf_attn_mask shape: %s", cf_attn_mask.shape)
            logger.debug("  image_bounds: %s", actual_modality_bounds.image_bounds)
            logger.debug("  language_bounds: %s", actual_modality_bounds.language_bounds)
            logger.debug("  state_bounds: %s", actual_modality_bounds.state_bounds)
            logger.debug("  cf_mode: %s", cf_mode.value)

        # Use precomputed prefix context when available.
        if (
            prepared_observation is not None
            and prefix_mask is not None
            and kv_cache is not None
            and self._sample_actions_from_precomputed_prefix is not None
        ):
            return self._sample_actions_from_precomputed_prefix(
                rng,
                prepared_observation,
                prefix_mask,
                kv_cache,
                num_steps=num_steps,
                cf_attn_mask=cf_attn_mask,
            )

        # Fallback: regular sample_actions path.
        return self._sample_actions(
            rng,
            observation,
            num_steps=num_steps,
            cf_attn_mask=cf_attn_mask,
        )

    def _compute_modality_bounds(
        self,
        observation: Observation,
    ) -> ModalityBounds:
        """Compute modality bounds from observation.
        
        Args:
            observation: Observation object.
        
        Returns:
            ModalityBounds with computed positions.
        """
        # Count image tokens
        image_bounds = {}
        current_pos = 0
        
        if observation.images is not None:
            for name in observation.images:
                # Each image produces ~100 tokens (SigLIP output)
                # The actual count depends on image size and model config
                count = self._get_image_token_count(name, observation.images[name])
                image_bounds[name] = (current_pos, current_pos + count)
                current_pos += count
        
        image_offset = current_pos
        
        # Compute text bounds (depends on tokenized prompt)
        # For Pi05, this includes both language and state
        if observation.tokenized_prompt is not None:
            prompt_len = observation.tokenized_prompt.shape[1]
            
            # Estimate language and state bounds
            # This is approximate; exact bounds require tokenizer info
            # For Pi05: "Task: {lang}, State: {state};\nAction: "
            if observation.state is not None:
                # Pi05 format
                state_dim = observation.state.shape[-1]
                
                # Rough estimate: language is about 70% of prompt, state is 30%
                # Actual implementation should use tokenizer
                lang_end = image_offset + int(prompt_len * 0.7)
                state_start = lang_end
                state_end = state_start + state_dim
                
                prefix_len = image_offset + prompt_len
                
                return ModalityBounds(
                    image_bounds=image_bounds,
                    language_bounds=(image_offset, lang_end),
                    state_bounds=(state_start, state_end),
                    prefix_len=prefix_len,
                    suffix_start=prefix_len,
                )
            else:
                # Pi0 format (state is continuous, not in tokens)
                prefix_len = image_offset + prompt_len
                
                return ModalityBounds(
                    image_bounds=image_bounds,
                    language_bounds=(image_offset, prefix_len),
                    state_bounds=(prefix_len, prefix_len),  # Empty
                    prefix_len=prefix_len,
                    suffix_start=prefix_len,
                )
        
        # No text, just images
        return ModalityBounds(
            image_bounds=image_bounds,
            language_bounds=(current_pos, current_pos),
            state_bounds=(current_pos, current_pos),
            prefix_len=current_pos,
            suffix_start=current_pos,
        )
    
    def _get_image_token_count(
        self,
        name: str,
        image: jnp.ndarray,
    ) -> int:
        """Get the number of tokens an image produces.
        
        Args:
            name: Camera name.
            image: Image array.
        
        Returns:
            Number of tokens.
        """
        # SigLIP typically produces 100 tokens for 224x224 images
        # This can vary based on model configuration
        return self._image_tokens_per_camera
    
    def analyze_modality_importance(
        self,
        rng: jax.Array,
        observation: Observation,
        *,
        num_samples: int = 5,
        num_steps: Optional[int] = None,
    ) -> dict:
        """Analyze modality importance through multiple CF samples.
        
        Runs multiple samples for each CF mode and computes statistics
        on modality effects.
        
        Args:
            rng: JAX random key.
            observation: Observation input.
            num_samples: Number of samples per mode for statistical analysis.
            num_steps: Flow matching steps.
        
        Returns:
            Dictionary with mean and std effects for each modality.
        """
        all_effects = {
            "image": [],
            "language": [],
            "state": [],
        }
        
        for i in range(num_samples):
            rng, sub_rng = jax.random.split(rng)
            result = self.sample_with_cf(
                sub_rng, observation,
                cf_modes=[
                    CfAttnMode.BASE,
                    CfAttnMode.NO_IMAGE,
                    CfAttnMode.NO_LANGUAGE,
                    CfAttnMode.NO_STATE,
                ],
                num_steps=num_steps,
                return_effects=True,
            )
            
            for modality in ["image", "language", "state"]:
                effect = result.get_effect(modality)
                if effect is not None:
                    all_effects[modality].append(float(effect.mean()))
        
        # Compute statistics
        stats = {}
        for modality, effects in all_effects.items():
            if effects:
                effects_array = np.array(effects)
                stats[modality] = {
                    "mean": float(effects_array.mean()),
                    "std": float(effects_array.std()),
                    "min": float(effects_array.min()),
                    "max": float(effects_array.max()),
                }
        
        return stats


def create_cf_sampler(model: Pi0, **kwargs) -> CfSampler:
    """Factory function to create a CF sampler.
    
    Args:
        model: Pi0 model instance.
        **kwargs: Additional arguments for CfSampler.
    
    Returns:
        CfSampler instance.
    """
    return CfSampler(model, **kwargs)