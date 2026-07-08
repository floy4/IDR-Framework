# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

"""Counterfactual attention mode definitions for X-VLA.

This module defines the enumeration of counterfactual modes and provides
functions to determine which modalities should be visible/weighted in each mode.

The CF modes control which modalities contribute to action generation:
- BASE: Normal inference (all modalities)
- NO_IMAGE: Block all visual inputs (VLM + aux_visual)
- NO_PROPRIO: Block proprioceptive input
- IMAGE_ONLY: Only visual inputs visible
- PROPRIO_ONLY: Only proprio input visible
- NO_VLM_PRIMARY: Block only primary VLM features (main view)
- NO_AUX_VISUAL: Block only auxiliary visual inputs (additional views)

Delta weighting modes (CfWeightMode):
- A-F: Compatible with original X-VLA CF implementation
- G-H: Extended modes for fine-grained three-modality weighting
"""

from enum import Enum
from typing import Dict, Any, Union


class CfAttnMode(str, Enum):
    """Counterfactual attention mode enumeration for X-VLA.
    
    Each mode defines which modalities are visible/weighted during action generation.
    
    Modalities in X-VLA:
    - vlm: VLM features (main camera view + language instruction)
    - aux_visual: Additional camera views
    - proprio: Proprioceptive state (joint positions, end-effector pose)
    
    Modes:
    BASE: Normal inference (all modalities visible)
    
    Single modality blocking:
    - NO_IMAGE: Block all visual (vlm + aux_visual)
    - NO_VLM: Block VLM features only
    - NO_AUX_VISUAL: Block auxiliary visual only
    - NO_PROPRIO: Block proprioceptive input
    
    Single modality focus:
    - IMAGE_ONLY: Only visual inputs (no proprio)
    - VLM_ONLY: Only VLM features (no aux_visual, no proprio)
    - PROPRIO_ONLY: Only proprio (no visual)
    
    Dual modality combinations:
    - IMAGE_PROPRIO: Visual + proprio (block language implicitly in VLM)
    - VLM_PROPRIO: VLM + proprio (block aux_visual)
    
    Note: The actual implementation uses feature weighting/reweighting,
    not attention mask modification, to achieve these effects.
    """
    
    # Normal inference
    BASE = "base"
    
    # Block single modality
    NO_IMAGE = "no_image"          # Block all visual
    NO_VLM = "no_vlm"              # Block primary VLM only
    NO_AUX_VISUAL = "no_aux_visual"  # Block auxiliary visual only
    NO_PROPRIO = "no_proprio"      # Block proprioceptive
    
    # Single modality focus
    IMAGE_ONLY = "image_only"      # Only visual (block proprio)
    VLM_ONLY = "vlm_only"          # Only VLM (block aux_visual + proprio)
    PROPRIO_ONLY = "proprio_only"  # Only proprio (block all visual)
    
    # Dual modality combinations
    IMAGE_PROPRIO = "image_proprio"  # Visual + proprio (standard)
    VLM_PROPRIO = "vlm_proprio"      # VLM + proprio (no aux_visual)


def get_modality_visibility(mode: CfAttnMode) -> dict[str, bool]:
    """Get visibility configuration for each modality based on CF mode.
    
    Args:
        mode: Counterfactual attention mode.
    
    Returns:
        Dictionary with keys "vlm", "aux_visual", "proprio" and boolean values
        indicating whether each modality should contribute to action generation.
    """
    # Default: all visible
    config = {"vlm": True, "aux_visual": True, "proprio": True}
    
    # Apply mode-specific visibility
    if mode == CfAttnMode.NO_IMAGE:
        config["vlm"] = False
        config["aux_visual"] = False
    elif mode == CfAttnMode.NO_VLM:
        config["vlm"] = False
    elif mode == CfAttnMode.NO_AUX_VISUAL:
        config["aux_visual"] = False
    elif mode == CfAttnMode.NO_PROPRIO:
        config["proprio"] = False
    elif mode == CfAttnMode.IMAGE_ONLY:
        config["proprio"] = False
    elif mode == CfAttnMode.VLM_ONLY:
        config["aux_visual"] = False
        config["proprio"] = False
    elif mode == CfAttnMode.PROPRIO_ONLY:
        config["vlm"] = False
        config["aux_visual"] = False
    elif mode == CfAttnMode.IMAGE_PROPRIO:
        # All visible (same as BASE, but explicit)
        pass
    elif mode == CfAttnMode.VLM_PROPRIO:
        config["aux_visual"] = False
    
    return config


def get_cf_mode_description(mode: CfAttnMode) -> str:
    """Get human-readable description for a CF mode.
    
    Args:
        mode: Counterfactual attention mode.
    
    Returns:
        Description string.
    """
    descriptions = {
        CfAttnMode.BASE: "Normal inference with all modalities",
        CfAttnMode.NO_IMAGE: "Block all visual inputs (VLM + aux_visual)",
        CfAttnMode.NO_VLM: "Block primary VLM features only",
        CfAttnMode.NO_AUX_VISUAL: "Block auxiliary visual inputs only",
        CfAttnMode.NO_PROPRIO: "Block proprioceptive input",
        CfAttnMode.IMAGE_ONLY: "Only visual inputs, block proprio",
        CfAttnMode.VLM_ONLY: "Only VLM features, block aux_visual and proprio",
        CfAttnMode.PROPRIO_ONLY: "Only proprio, block all visual",
        CfAttnMode.IMAGE_PROPRIO: "Visual + proprio (all modalities)",
        CfAttnMode.VLM_PROPRIO: "VLM + proprio, block aux_visual",
    }
    return descriptions.get(mode, f"Unknown mode: {mode}")


def get_cf_modes_for_analysis() -> list[CfAttnMode]:
    """Get recommended CF modes for comprehensive modality analysis.
    
    Returns a list of modes that can be used to analyze the contribution
    of each modality to the action generation.
    
    Returns:
        List of CfAttnMode values for comprehensive analysis.
    """
    return [
        CfAttnMode.BASE,          # Baseline (all modalities)
        CfAttnMode.NO_IMAGE,      # Image contribution (all visual)
        CfAttnMode.NO_VLM,        # Primary VLM contribution
        CfAttnMode.NO_AUX_VISUAL, # Auxiliary visual contribution
        CfAttnMode.NO_PROPRIO,    # Proprio contribution
        CfAttnMode.IMAGE_ONLY,    # Visual-only baseline
        CfAttnMode.PROPRIO_ONLY,  # Proprio-only baseline
    ]


def compute_modality_effect(
    baseline_actions,
    cf_actions,
    metric: str = "l2",
):
    """Compute the effect size between baseline and counterfactual actions.
    
    This function measures how much the actions change when a modality is blocked,
    indicating the importance of that modality.
    
    Args:
        baseline_actions: Actions from baseline mode (all modalities).
                          Shape: [B, T, D] or [T, D]
        cf_actions: Actions from counterfactual mode.
                    Shape: [B, T, D] or [T, D]
        metric: Distance metric to use. Options: "l2", "l1", "cosine", "max".
    
    Returns:
        Effect size as a scalar (mean over all dimensions).
    
    Note:
        This is a PyTorch implementation adapted from OpenPI's JAX version.
    """
    import torch
    
    if baseline_actions is None or cf_actions is None:
        return 0.0
    
    # Ensure same shape
    if baseline_actions.shape != cf_actions.shape:
        # Try to align shapes
        min_batch = min(baseline_actions.shape[0], cf_actions.shape[0])
        baseline_actions = baseline_actions[:min_batch]
        cf_actions = cf_actions[:min_batch]
    
    if metric == "l2":
        diff = baseline_actions - cf_actions
        # L2 norm per action, then mean
        per_action_norm = torch.norm(diff, p=2, dim=-1)  # [B, T]
        return float(per_action_norm.mean().item())
    elif metric == "l1":
        diff = baseline_actions - cf_actions
        per_action_norm = torch.norm(diff, p=1, dim=-1)  # [B, T]
        return float(per_action_norm.mean().item())
    elif metric == "cosine":
        # Cosine distance = 1 - cosine_similarity
        # Flatten to [B, T*D] for overall similarity
        baseline_flat = baseline_actions.reshape(baseline_actions.shape[0], -1)
        cf_flat = cf_actions.reshape(cf_actions.shape[0], -1)
        
        cosine_sim = torch.nn.functional.cosine_similarity(baseline_flat, cf_flat, dim=-1)
        return float((1.0 - cosine_sim).mean().item())
    elif metric == "max":
        diff = baseline_actions - cf_actions
        return float(torch.abs(diff).max().item())
    else:
        raise ValueError(f"Unknown metric: {metric}")


# Legacy mode mapping for backward compatibility
# Maps the old A-F modes from the existing X-VLA CF implementation
LEGACY_MODE_MAP = {
    "A": CfAttnMode.NO_PROPRIO,     # Image-only guidance (ignore proprio delta)
    "B": CfAttnMode.BASE,           # Small fixed proprio weight (essentially BASE)
    "C": CfAttnMode.BASE,           # Normalized proprio (not a true CF mode)
    "D": CfAttnMode.BASE,           # Hard clip proprio (not a true CF mode)
    "E": CfAttnMode.NO_PROPRIO,     # Adaptive proprio weight (essentially blocks proprio)
    "F": CfAttnMode.BASE,           # Smooth tanh proprio (not a true CF mode)
}


def map_legacy_mode(legacy_mode: str) -> CfAttnMode:
    """Map legacy A-F mode to new CfAttnMode.
    
    Args:
        legacy_mode: Legacy mode letter ("A", "B", "C", "D", "E", "F").
    
    Returns:
        Corresponding CfAttnMode value.
    """
    return LEGACY_MODE_MAP.get(legacy_mode.upper(), CfAttnMode.BASE)


class CfWeightMode(str, Enum):
    """Delta weighting mode for counterfactual action refinement.
    
    These modes define how to combine the deltas (contributions) from different
    modalities to produce a refined action output.
    
    The delta for each modality is computed as:
        delta = baseline_actions - actions_with_modality_blocked
    
    Then the final action is:
        final = baseline + guidance_scale * weighted_delta_sum
    
    Modes A-F: Compatible with original X-VLA CF implementation
    - A: Only VLM delta, threshold-based fallback
    - B: VLM delta + fixed proprio weight (0.05)
    - C: VLM delta + normalized proprio delta
    - D: VLM delta + clipped proprio delta (-0.1, 0.1)
    - E: Adaptive - threshold fallback + adaptive proprio weight
    - F: VLM delta + tanh-smoothed proprio delta
    
    Extended modes G-H: Fine-grained three-modality weighting
    - G: VLM + aux_visual + proprio with fixed weights
    - H: Fully adaptive three-modality weighting based on effects
    """
    
    # Base mode (no weighting)
    BASE = "base"
    
    # Original A-F modes (compatible with existing implementation)
    A = "A"  # VLM only, threshold fallback
    B = "B"  # VLM + fixed proprio (0.05)
    C = "C"  # VLM + normalized proprio
    D = "D"  # VLM + clipped proprio (-0.1, 0.1)
    E = "E"  # Adaptive: threshold + adaptive proprio
    F = "F"  # VLM + tanh smoothed proprio
    
    # Extended modes for fine-grained control
    G = "G"  # VLM + aux_visual + proprio (fixed weights)
    H = "H"  # Fully adaptive three-modality


# Weight configuration type: can be float or special string tokens
WeightValue = Union[float, str]

# Default weight configurations for each mode
DEFAULT_WEIGHT_CONFIGS: Dict[CfWeightMode, Dict[str, WeightValue]] = {
    CfWeightMode.BASE: {
        "vlm": 0.0,
        "aux_visual": 0.0,
        "proprio": 0.0,
    },
    CfWeightMode.A: {
        "vlm": 1.0,
        "aux_visual": 0.0,
        "proprio": 0.0,
        # Special: threshold fallback
        "use_threshold": True,
    },
    CfWeightMode.B: {
        "vlm": 1.0,
        "aux_visual": 0.0,
        "proprio": 0.05,  # Fixed weight
    },
    CfWeightMode.C: {
        "vlm": 1.0,
        "aux_visual": 0.0,
        "proprio": "normalized",  # Normalize by effect
    },
    CfWeightMode.D: {
        "vlm": 1.0,
        "aux_visual": 0.0,
        "proprio": "clipped",  # Clamp to [-0.1, 0.1]
    },
    CfWeightMode.E: {
        "vlm": 1.0,
        "aux_visual": 0.0,
        "proprio": "adaptive",  # Adaptive based on effect
        "use_threshold": True,
    },
    CfWeightMode.F: {
        "vlm": 1.0,
        "aux_visual": 0.0,
        "proprio": "tanh",  # Tanh smoothing
    },
    CfWeightMode.G: {
        "vlm": 1.0,
        "aux_visual": 0.5,
        "proprio": 0.05,  # Three-modality fixed
    },
    CfWeightMode.H: {
        "vlm": "adaptive",
        "aux_visual": "adaptive",
        "proprio": "adaptive",  # Fully adaptive
    },
}


def get_cf_weight_config(mode: CfWeightMode) -> Dict[str, WeightValue]:
    """Get weight configuration for a delta weighting mode.
    
    Args:
        mode: Delta weighting mode.
    
    Returns:
        Dictionary with keys "vlm", "aux_visual", "proprio" and weight values.
        Weight values can be:
        - float: Fixed weight multiplier
        - "normalized": Normalize delta by its effect magnitude
        - "clipped": Clamp delta values to [-0.1, 0.1]
        - "adaptive": Weight based on relative effect importance
        - "tanh": Apply tanh smoothing to delta
    """
    return DEFAULT_WEIGHT_CONFIGS.get(mode, DEFAULT_WEIGHT_CONFIGS[CfWeightMode.BASE])


def get_cf_weight_mode_description(mode: CfWeightMode) -> str:
    """Get human-readable description for a delta weighting mode.
    
    Args:
        mode: Delta weighting mode.
    
    Returns:
        Description string.
    """
    descriptions = {
        CfWeightMode.BASE: "No weighting, return baseline actions",
        CfWeightMode.A: "Visual delta only, threshold fallback (if image effect > threshold, return baseline)",
        CfWeightMode.B: "Visual delta + fixed proprio weight (0.05)",
        CfWeightMode.C: "Visual delta + normalized proprio delta (scale by effect)",
        CfWeightMode.D: "Visual delta + clipped proprio delta (clamp to [-0.1, 0.1])",
        CfWeightMode.E: "Adaptive: threshold fallback + adaptive proprio weight based on effect ratio",
        CfWeightMode.F: "Visual delta + tanh smoothed proprio delta",
        CfWeightMode.G: "Three-modality fixed: VLM(1.0) + aux(0.5) + proprio(0.05)",
        CfWeightMode.H: "Fully adaptive three-modality weighting based on effects",
    }
    return descriptions.get(mode, f"Unknown mode: {mode}")
