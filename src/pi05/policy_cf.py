"""Counterfactual Policy wrapper for OpenPI policies.

This module provides a wrapper class that adds counterfactual analysis
capabilities to existing OpenPI policies without modifying original code.
"""

from typing import Optional, Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from openpi.policies.policy import Policy
from openpi.models.pi0 import Pi0
from openpi.models import model as _model
Observation = _model.Observation
from .cf_sampler import CfSampler, CFAnalysisResult
from .modality_bounds import ModalityBounds
from .attention_mask import CfAttnMode, get_cf_modes_for_analysis
from .tokenizer_ext import ExtendedPaligemmaTokenizer


@dataclass
class CfPolicyResult:
    """Result from CfPolicy inference.
    
    Contains both the standard policy result and CF analysis data.
    """
    
    # Standard policy output (actions, etc.)
    policy_result: dict
    
    # CF analysis results (if requested)
    cf_analysis: Optional[CFAnalysisResult]
    
    # Modality importance statistics (if requested)
    modality_importance: Optional[dict]
    
    def get_actions(self) -> Any:
        """Get the baseline actions from policy result."""
        return self.policy_result.get("actions", None)
    
    def get_cf_effects(self) -> Optional[dict]:
        """Get modality effects if CF analysis was performed."""
        if self.cf_analysis is None:
            return None
        return self.cf_analysis.effects


class CfPolicy:
    """Counterfactual Policy wrapper.
    
    This class wraps an existing Policy and adds counterfactual analysis
    capabilities. It can be used to:
    1. Perform normal inference (same as original policy)
    2. Perform CF analysis to understand modality contributions
    3. Get statistical importance of each modality
    
    The wrapper does NOT modify the original policy - it uses the policy's
    existing methods and adds CF functionality as a layer on top.
    
    Example:
        >>> # Create original policy
        >>> original_policy = create_trained_policy(config, checkpoint)
        >>> 
        >>> # Wrap with CF capabilities
        >>> cf_policy = CfPolicy(original_policy)
        >>> 
        >>> # Normal inference (same as original)
        >>> result = cf_policy.infer(obs)
        >>> 
        >>> # Inference with CF analysis
        >>> result = cf_policy.infer_with_cf(obs)
        >>> print(result.get_cf_effects())
        >>> 
        >>> # Statistical analysis
        >>> importance = cf_policy.analyze_modality_importance(obs, num_samples=10)
    
    Attributes:
        _policy: The wrapped Policy instance.
        _cf_sampler: CfSampler instance for CF analysis.
        _tokenizer: Extended tokenizer for boundary computation.
    """
    
    def __init__(
        self,
        policy: Policy,
        *,
        tokenizer_max_len: int = 48,
        default_cf_modes: Optional[list[CfAttnMode]] = None,
        default_num_steps: int = 10,
    ):
        """Initialize the CF policy wrapper.
        
        Args:
            policy: Policy instance to wrap.
            tokenizer_max_len: Max token length for tokenizer.
            default_cf_modes: Default CF modes for analysis.
            default_num_steps: Default flow matching steps.
        """
        self._policy = policy
        self._default_cf_modes = default_cf_modes or [
            CfAttnMode.BASE,
            CfAttnMode.NO_IMAGE,
            CfAttnMode.NO_LANGUAGE,
            CfAttnMode.NO_STATE,
        ]
        self._default_num_steps = default_num_steps
        
        # Extract model from policy
        # Policy typically has a _model attribute
        self._model = getattr(policy, "_model", None)
        
        # Create CF sampler if model is available
        if self._model is not None:
            self._cf_sampler = CfSampler(
                self._model,
                default_num_steps=default_num_steps,
            )
        else:
            self._cf_sampler = None
        
        # Create extended tokenizer
        self._tokenizer = ExtendedPaligemmaTokenizer(max_len=tokenizer_max_len)
    
    def infer(self, obs: dict) -> dict:
        """Standard inference (same as original policy).
        
        This method delegates directly to the wrapped policy's infer method.
        
        Args:
            obs: Observation dictionary.
        
        Returns:
            Policy result dictionary (same format as original policy).
        """
        return self._policy.infer(obs)
    
    def infer_with_cf(
        self,
        obs: dict,
        *,
        cf_modes: Optional[list[CfAttnMode]] = None,
        num_steps: Optional[int] = None,
        return_importance: bool = False,
        importance_samples: int = 5,
        rng: Optional[jax.Array] = None,
    ) -> CfPolicyResult:
        """Inference with counterfactual analysis.
        
        This method performs standard inference and adds CF analysis
        to understand how each modality contributes to the action.
        
        Args:
            obs: Observation dictionary.
            cf_modes: CF modes to evaluate. If None, uses defaults.
            num_steps: Flow matching steps. If None, uses default.
            return_importance: Whether to compute statistical importance.
            importance_samples: Number of samples for importance analysis.
            rng: JAX random key. If None, creates a new one.
        
        Returns:
            CfPolicyResult with actions and CF analysis.
        
        Note:
            The observation should contain:
            - Images (e.g., "base_0_rgb", "left_wrist_0_rgb")
            - State (robot proprioceptive state)
            - Prompt (task description)
            
            These are typically provided through the policy's input transform.
        """
        # First, get standard policy result
        policy_result = self._policy.infer(obs)
        
        # Initialize RNG if not provided
        if rng is None:
            rng = jax.random.PRNGKey(42)
        
        # Create observation object for CF analysis
        # This requires converting the dict to Observation format
        observation = self._create_observation_from_dict(obs)
        
        # Perform CF analysis if sampler is available
        cf_analysis = None
        if self._cf_sampler is not None and observation is not None:
            cf_analysis = self._cf_sampler.sample_with_cf(
                rng,
                observation,
                cf_modes=cf_modes or self._default_cf_modes,
                num_steps=num_steps or self._default_num_steps,
                return_effects=True,
            )
        
        # Compute modality importance if requested
        modality_importance = None
        if return_importance and self._cf_sampler is not None and observation is not None:
            modality_importance = self._cf_sampler.analyze_modality_importance(
                rng,
                observation,
                num_samples=importance_samples,
                num_steps=num_steps or self._default_num_steps,
            )
        
        return CfPolicyResult(
            policy_result=policy_result,
            cf_analysis=cf_analysis,
            modality_importance=modality_importance,
        )
    
    def analyze_modality_importance(
        self,
        obs: dict,
        *,
        num_samples: int = 10,
        num_steps: Optional[int] = None,
        rng: Optional[jax.Array] = None,
    ) -> dict:
        """Analyze statistical importance of each modality.
        
        Runs multiple CF samples and computes mean/std effects for
        each modality, giving a robust estimate of modality importance.
        
        Args:
            obs: Observation dictionary.
            num_samples: Number of samples for statistical analysis.
            num_steps: Flow matching steps.
            rng: JAX random key.
        
        Returns:
            Dictionary with importance statistics:
            {
                "image": {"mean": X, "std": Y, "min": Z, "max": W},
                "language": {...},
                "state": {...},
            }
        """
        if rng is None:
            rng = jax.random.PRNGKey(42)
        
        observation = self._create_observation_from_dict(obs)
        
        if self._cf_sampler is None or observation is None:
            return {}
        
        return self._cf_sampler.analyze_modality_importance(
            rng,
            observation,
            num_samples=num_samples,
            num_steps=num_steps or self._default_num_steps,
        )
    
    def _create_observation_from_dict(
        self,
        obs: dict,
    ) -> Optional[Observation]:
        """Create Observation object from observation dictionary.
        
        This method extracts relevant information from the policy's
        input format and creates an Observation object for CF analysis.
        
        Args:
            obs: Observation dictionary from policy input.
        
        Returns:
            Observation object or None if conversion fails.
        
        Note:
            The exact fields depend on the policy's input transform.
            Typical fields include:
            - Images: "base_0_rgb", "left_wrist_0_rgb", etc.
            - State: Joint positions, gripper state, etc.
            - Prompt: Task description (via internal transform)
        """
        try:
            # Extract images
            images = {}
            image_masks = {}
            
            # Common image field names in OpenPI policies
            image_keys = [
                "base_0_rgb",
                "left_wrist_0_rgb",
                "right_wrist_0_rgb",
                "base_1_rgb",
                "base_2_rgb",
            ]
            
            for key in image_keys:
                if key in obs:
                    img = obs[key]
                    if isinstance(img, np.ndarray):
                        img = jnp.array(img)
                    images[key] = img
                    # Create mask (True for valid tokens)
                    image_masks[key] = jnp.ones(1)  # Batch dimension
            
            # Extract state
            state = None
            state_keys = ["state", "joint_positions", "proprio"]
            for key in state_keys:
                if key in obs:
                    state_val = obs[key]
                    if isinstance(state_val, np.ndarray):
                        state_val = jnp.array(state_val)
                    state = state_val
                    break
            
            # Get prompt (might be internal to policy)
            # Try to extract it if available
            prompt = obs.get("prompt", None) or obs.get("task_description", None)
            
            # Tokenize prompt if available
            tokenized_prompt = None
            tokenized_prompt_mask = None
            
            if prompt is not None:
                tokens, mask, bounds = self._tokenizer.tokenize_with_bounds(
                    prompt=prompt,
                    state=np.array(state) if state is not None else None,
                    image_token_counts={k: 100 for k in images.keys()},  # Approximate
                )
                tokenized_prompt = jnp.array(tokens)[None, :]  # Add batch dim
                tokenized_prompt_mask = jnp.array(mask)[None, :]
            
            # Create Observation object
            # Note: This might need adjustment based on actual Observation class structure
            observation = Observation(
                images=images,
                image_masks=image_masks,
                state=state,
                tokenized_prompt=tokenized_prompt,
                tokenized_prompt_mask=tokenized_prompt_mask,
            )
            
            return observation
            
        except Exception as e:
            import warnings
            warnings.warn(
                f"Failed to create Observation from dict: {e}. "
                "CF analysis may not work correctly.",
                UserWarning,
            )
            return None
    
    def get_modality_bounds(
        self,
        obs: dict,
    ) -> Optional[ModalityBounds]:
        """Get modality bounds for an observation.
        
        This is useful for understanding the token structure before
        performing CF analysis.
        
        Args:
            obs: Observation dictionary.
        
        Returns:
            ModalityBounds or None if computation fails.
        """
        prompt = obs.get("prompt", None) or obs.get("task_description", None)
        state = None
        
        state_keys = ["state", "joint_positions", "proprio"]
        for key in state_keys:
            if key in obs:
                state = obs[key]
                break
        
        if prompt is None:
            return None
        
        # Get image token counts
        image_token_counts = {}
        image_keys = [
            "base_0_rgb",
            "left_wrist_0_rgb",
            "right_wrist_0_rgb",
        ]
        for key in image_keys:
            if key in obs:
                image_token_counts[key] = 100  # Approximate
        
        return self._tokenizer.compute_bounds_from_token_counts(
            prompt=prompt,
            state=np.array(state) if state is not None else None,
            image_token_counts=image_token_counts,
        )
    
    # Delegate other Policy methods to wrapped policy
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped policy."""
        return getattr(self._policy, name)
    
    def __repr__(self) -> str:
        return f"CfPolicy(wrapped={self._policy.__class__.__name__})"


def wrap_policy_with_cf(
    policy: Policy,
    **kwargs,
) -> CfPolicy:
    """Factory function to wrap a policy with CF capabilities.
    
    Args:
        policy: Policy instance to wrap.
        **kwargs: Additional arguments for CfPolicy.
    
    Returns:
        CfPolicy instance wrapping the original policy.
    
    Example:
        >>> from openpi.policies.policy import create_trained_policy
        >>> from openpi.models.cf_attention import wrap_policy_with_cf
        >>> 
        >>> # Create and wrap policy
        >>> policy = create_trained_policy(config, checkpoint)
        >>> cf_policy = wrap_policy_with_cf(policy)
        >>> 
        >>> # Use with CF analysis
        >>> result = cf_policy.infer_with_cf(obs)
        >>> print(result.get_cf_effects())
    """
    return CfPolicy(policy, **kwargs)


# Convenience function for one-shot CF analysis
def analyze_observation_cf(
    policy: Policy,
    obs: dict,
    *,
    cf_modes: Optional[list[CfAttnMode]] = None,
    num_samples: int = 1,
    rng: Optional[jax.Array] = None,
) -> CfPolicyResult:
    """Perform one-shot counterfactual analysis on an observation.
    
    This is a convenience function that wraps the policy temporarily
    and performs CF analysis.
    
    Args:
        policy: Policy instance.
        obs: Observation dictionary.
        cf_modes: CF modes to evaluate.
        num_samples: Number of samples.
        rng: JAX random key.
    
    Returns:
        CfPolicyResult with analysis results.
    """
    cf_policy = CfPolicy(policy)
    return cf_policy.infer_with_cf(
        obs,
        cf_modes=cf_modes,
        return_importance=num_samples > 1,
        importance_samples=num_samples,
        rng=rng,
    )