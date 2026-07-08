"""
Utility functions for Counterfactual Attention Module

Provides core functions for:
- Computing CF effects (effect_vlm, effect_prop)
- Applying CF reweighting formulas (modes A-F)
- Action manipulation utilities

Reference: OpenPI's pi0.py implementation
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union

try:
    from prismatic.cf_attention.config import CfMode
except ImportError:
    # Direct import fallback
    from config import CfMode


def compute_cf_effect(
    actions_baseline: np.ndarray,
    actions_cf: np.ndarray,
    method: str = "l2"
) -> float:
    """
    Compute scalar effect for counterfactual analysis.
    
    Args:
        actions_baseline: Baseline action array [action_horizon, action_dim]
        actions_cf: Counterfactual action array [action_horizon, action_dim]
        method: Computation method ("l2", "l1", "cosine")
    
    Returns:
        Scalar effect value
    
    Reference: OpenPI's _compute_cf_action_diff in pi0.py
    """
    diff = actions_baseline - actions_cf
    
    if method == "l2":
        # OpenPI formula: mean(||diff||_2)
        effect = np.mean(np.linalg.norm(diff.reshape(-1), axis=-1))
    elif method == "l1":
        effect = np.mean(np.abs(diff).sum(axis=-1))
    elif method == "cosine":
        # Cosine similarity-based effect
        baseline_flat = actions_baseline.flatten()
        cf_flat = actions_cf.flatten()
        cosine_sim = np.dot(baseline_flat, cf_flat) / (
            np.linalg.norm(baseline_flat) * np.linalg.norm(cf_flat) + 1e-6
        )
        effect = 1.0 - cosine_sim  # Effect = 1 - similarity
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(effect)


def compute_delta(
    actions_baseline: np.ndarray,
    actions_cf: np.ndarray
) -> np.ndarray:
    """
    Compute action delta for counterfactual reweighting.
    
    Args:
        actions_baseline: Baseline action array
        actions_cf: Counterfactual action array
    
    Returns:
        Delta array (baseline - cf)
    """
    return actions_baseline - actions_cf


def apply_cf_reweighting(
    actions_baseline: np.ndarray,
    delta_img: np.ndarray,
    delta_prop: np.ndarray,
    effect_vlm: float,
    effect_prop: float,
    mode: CfMode,
    cf_guidance_scale: float = 0.1,
    vlm_effect_upper_threshold: float = 0.5,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, bool]:
    """
    Apply counterfactual reweighting formula.
    
    Args:
        actions_baseline: Baseline action array
        delta_img: Image delta (baseline - img0)
        delta_prop: Proprio delta (baseline - prop0)
        effect_vlm: VLM effect magnitude
        effect_prop: Proprio effect magnitude
        mode: CF reweighting mode (A-F)
        cf_guidance_scale: Guidance scale factor
        vlm_effect_upper_threshold: Threshold for fallback
        eps: Small constant for numerical stability
    
    Returns:
        Tuple of (reweighted_actions, use_baseline_flag)
    
    Reference: OpenPI's sample_actions_with_cf in pi0.py (lines 400-436)
    """
    # Check if VLM effect exceeds threshold
    use_baseline = effect_vlm > vlm_effect_upper_threshold
    
    # Compute adaptive weights
    total_effect = effect_vlm + effect_prop + eps
    w_vlm = effect_vlm / total_effect
    w_prop = effect_prop / total_effect
    
    # Apply mode-specific formula
    if mode == CfMode.BASE:
        actions_cf = actions_baseline
    
    elif mode == CfMode.A:
        # Mode A: Image-only reweighting
        # Formula: actions_cf = baseline + scale × delta_img
        actions_cf = actions_baseline + cf_guidance_scale * delta_img
    
    elif mode == CfMode.B:
        # Mode B: Image + fixed proprio weight
        # Formula: actions_cf = baseline + scale × delta_img + 0.05 × delta_prop
        prop_scale = 0.05
        actions_cf = actions_baseline + cf_guidance_scale * delta_img + prop_scale * delta_prop
    
    elif mode == CfMode.C:
        # Mode C: Adaptive weights + normalized proprio delta
        # Formula: actions_cf = baseline + scale × (w_vlm × delta_img + w_prop × 0.1 × norm_delta_prop)
        abs_max = np.max(np.abs(delta_prop), axis=(-2, -1), keepdims=True)
        delta_prop_norm = np.where(abs_max > 0, delta_prop / (abs_max + eps), delta_prop)
        actions_cf = actions_baseline + cf_guidance_scale * (
            w_vlm * delta_img + w_prop * 0.1 * delta_prop_norm
        )
    
    elif mode == CfMode.D:
        # Mode D: Image + clipped proprio delta
        # Formula: actions_cf = baseline + scale × delta_img + 0.05 × clip(delta_prop, -0.1, 0.1)
        delta_prop_clipped = np.clip(delta_prop, -0.1, 0.1)
        actions_cf = actions_baseline + cf_guidance_scale * delta_img + 0.05 * delta_prop_clipped
    
    elif mode == CfMode.F:
        # Mode F: Image + soft-clipped proprio (tanh)
        # Formula: actions_cf = baseline + scale × delta_img + 0.05 × tanh(delta_prop/0.1) × 0.1
        delta_prop_soft = np.tanh(delta_prop / 0.1) * 0.1
        actions_cf = actions_baseline + cf_guidance_scale * delta_img + 0.05 * delta_prop_soft
    
    else:  # Mode E (default/recommended)
        # Mode E: Adaptive proprio weight + clipped proprio delta
        # Formula: prop_weight = 0.05 × min(1.0, prop_ratio)
        #          actions_cf = baseline + scale × delta_img + prop_weight × clip(delta_prop)
        prop_ratio = effect_prop / (effect_vlm + eps)
        prop_weight = 0.05 * min(1.0, prop_ratio)
        delta_prop_clipped = np.clip(delta_prop, -0.1, 0.1)
        actions_cf = actions_baseline + cf_guidance_scale * delta_img + prop_weight * delta_prop_clipped
    
    # Apply fallback if VLM effect too high
    actions_final = np.where(use_baseline, actions_baseline, actions_cf)
    
    return actions_final, use_baseline


def create_zero_image(image: Union[np.ndarray, "PIL.Image.Image"]) -> Union[np.ndarray, "PIL.Image.Image"]:
    """
    Create a zero/black image for counterfactual analysis.
    
    Args:
        image: Original image (numpy array or PIL Image)
    
    Returns:
        Zero image of same shape/type
    """
    if hasattr(image, 'mode'):  # PIL Image
        from PIL import Image
        return Image.new(image.mode, image.size, (0, 0, 0))
    else:  # numpy array
        return np.zeros_like(image)


def create_zero_state(state: np.ndarray) -> np.ndarray:
    """
    Create a zero state for counterfactual analysis.
    
    Args:
        state: Original proprioceptive state array
    
    Returns:
        Zero state array
    """
    return np.zeros_like(state)


def normalize_action(
    action: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Normalize action to [-1, 1] range.
    
    Args:
        action: Action array to normalize
        action_low: Lower bounds
        action_high: Upper bounds
        mask: Optional mask for valid dimensions
    
    Returns:
        Normalized action array
    """
    if mask is None:
        mask = np.ones_like(action_low, dtype=bool)
    
    normalized = np.where(
        mask,
        2 * (action - action_low) / (action_high - action_low) - 1,
        action
    )
    return normalized


def unnormalize_action(
    normalized_action: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Unnormalize action from [-1, 1] range to original range.
    
    Args:
        normalized_action: Normalized action array
        action_low: Lower bounds
        action_high: Upper bounds
        mask: Optional mask for valid dimensions
    
    Returns:
        Unnormalized action array
    """
    if mask is None:
        mask = np.ones_like(action_low, dtype=bool)
    
    action = np.where(
        mask,
        0.5 * (normalized_action + 1) * (action_high - action_low) + action_low,
        normalized_action
    )
    return action


def compute_kl_divergence(
    probs_p: np.ndarray,
    probs_q: np.ndarray,
    eps: float = 1e-10
) -> float:
    """
    Compute KL divergence for probability distributions.
    
    Used in Method C (Output Reweighting) for effect estimation.
    
    Args:
        probs_p: First probability distribution
        probs_q: Second probability distribution
        eps: Small constant for numerical stability
    
    Returns:
        KL divergence value
    """
    # Ensure probabilities are valid
    probs_p = np.clip(probs_p, eps, 1.0)
    probs_q = np.clip(probs_q, eps, 1.0)
    
    # Compute KL(P || Q)
    kl_div = np.sum(probs_p * np.log(probs_p / probs_q))
    
    return float(kl_div)


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert logits to probabilities using softmax.
    
    Args:
        logits: Raw logits tensor
    
    Returns:
        Probability tensor
    """
    return torch.softmax(logits, dim=-1)


def apply_cfg_reweighting(
    logits_conditioned: torch.Tensor,
    logits_unconditioned: torch.Tensor,
    cfg_scale: float = 1.0,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Apply classifier-free guidance reweighting to logits.
    
    Formula: logits_cf = logits_cond + scale × (logits_cond - logits_uncond)
    
    Args:
        logits_conditioned: Logits with full conditioning
        logits_unconditioned: Logits without specific conditioning
        cfg_scale: CFG scale factor
        eps: Small constant for numerical stability
    
    Returns:
        Reweighted logits
    
    Reference: CFG technique in language models
    """
    # CFG formula
    logits_cf = logits_conditioned + cfg_scale * (logits_conditioned - logits_unconditioned)
    
    return logits_cf


def get_cf_metrics(
    effect_vlm: float,
    effect_prop: float,
    use_baseline: bool,
    mode: CfMode,
    delta_img_norm: Optional[float] = None,
    delta_prop_norm: Optional[float] = None,
) -> Dict:
    """
    Create CF metrics dictionary for logging.
    
    Args:
        effect_vlm: VLM effect magnitude
        effect_prop: Proprio effect magnitude
        use_baseline: Whether fallback to baseline was used
        mode: CF mode used
        delta_img_norm: Optional norm of image delta
        delta_prop_norm: Optional norm of proprio delta
    
    Returns:
        Metrics dictionary
    """
    metrics = {
        "effect_vlm": effect_vlm,
        "effect_prop": effect_prop,
        "effect_ratio": effect_vlm / (effect_prop + 1e-6),
        "use_baseline": use_baseline,
        "cf_mode": mode.value,
    }
    
    if delta_img_norm is not None:
        metrics["delta_img_norm"] = delta_img_norm
    if delta_prop_norm is not None:
        metrics["delta_prop_norm"] = delta_prop_norm
    
    return metrics