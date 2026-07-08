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

"""Counterfactual attention mask utilities for X-VLA.

This module provides functions to create and modify attention masks
for counterfactual analysis, allowing control over which modalities
can attend to action tokens.

Key features:
- Create prefix-suffix attention masks (prefix bidirectional, suffix causal)
- Apply CF modes to block specific modality attention
- Support modality reweighting (A-F modes from OpenPI)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import logging

import torch
import numpy as np

from .modality_bounds import ModalityBounds
from .cf_mode import CfAttnMode, get_modality_visibility


def create_prefix_suffix_attn_mask(
    seq_len: int,
    prefix_len: int,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> torch.Tensor:
    """Create a prefix-suffix attention mask.
    
    The mask structure:
    - Prefix tokens: bidirectional attention (can attend to all prefix)
    - Suffix tokens: causal attention (can attend to prefix + previous suffix)
    
    Args:
        seq_len: Total sequence length (prefix + suffix).
        prefix_len: Length of prefix (bidirectional part).
        batch_size: Batch size.
        dtype: Data type for mask.
        device: Device for mask tensor.
    
    Returns:
        Attention mask tensor [B, seq_len, seq_len] with:
        - 1.0 for allowed attention
        - 0.0 for blocked attention
    """
    device = device or torch.device("cpu")
    
    # Create causal mask for suffix
    # suffix_pos i can attend to suffix_pos j if j <= i
    suffix_len = seq_len - prefix_len
    causal_mask = torch.tril(torch.ones(suffix_len, suffix_len, device=device))
    
    # Create full mask
    mask = torch.zeros(batch_size, seq_len, seq_len, dtype=dtype, device=device)
    
    # Prefix -> Prefix: bidirectional (all can attend to all)
    mask[:, :prefix_len, :prefix_len] = 1.0
    
    # Suffix -> Prefix: all suffix can attend to all prefix
    mask[:, prefix_len:, :prefix_len] = 1.0
    
    # Suffix -> Suffix: causal
    mask[:, prefix_len:, prefix_len:] = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    return mask


def make_cf_attn_mask(
    base_attn_mask: torch.Tensor,
    modality_bounds: ModalityBounds,
    cf_mode: CfAttnMode,
) -> torch.Tensor:
    """Apply counterfactual mode to attention mask.
    
    Modifies the attention mask so that suffix (action) tokens cannot
    attend to blocked modalities in the prefix.
    
    Args:
        base_attn_mask: Base attention mask [B, seq_len, seq_len].
        modality_bounds: ModalityBounds with token position info.
        cf_mode: Counterfactual mode to apply.
    
    Returns:
        Modified attention mask with CF mode applied.
    """
    visibility = get_modality_visibility(cf_mode)
    mask = base_attn_mask.clone()
    
    B, seq_len, _ = mask.shape
    suffix_start = modality_bounds.action_bounds[0]

    # Get modality ranges
    vlm_start, vlm_end = modality_bounds.vlm_bounds
    aux_start, aux_end = modality_bounds.aux_visual_bounds
    proprio_start, proprio_end = modality_bounds.action_bounds
    
    # Block VLM (primary visual + language features)
    if not visibility["vlm"]:
        # Zero attention from suffix to VLM tokens
        mask[:, suffix_start:, vlm_start:vlm_end] = 0.0
        logging.debug(f"CF mask: blocked VLM tokens [{vlm_start}:{vlm_end}]")
    
    # Block aux visual (additional camera views)
    if not visibility["aux_visual"]:
        if aux_start < aux_end:  # Has aux visual tokens
            mask[:, suffix_start:, aux_start:aux_end] = 0.0
            logging.debug(f"CF mask: blocked aux_visual tokens [{aux_start}:{aux_end}]")
    
    # Block proprio (proprioceptive state)
    if not visibility["proprio"]:
        if proprio_start < proprio_end:  # Has proprio tokens
            mask[:, suffix_start:, proprio_start:proprio_end] = 0.0
            logging.debug(f"CF mask: blocked proprio tokens [{proprio_start}:{proprio_end}]")
    
    return mask


def apply_cf_attn_mask_to_model(
    attention_weights: torch.Tensor,
    modality_bounds: ModalityBounds,
    cf_mode: CfAttnMode,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply CF mode to attention weights directly (for reweighting).
    
    This is used for the A-F modes where we reweight (not block) attention
    to specific modalities.
    
    Args:
        attention_weights: Raw attention weights [B, heads, seq_len, seq_len].
        modality_bounds: ModalityBounds with token position info.
        cf_mode: CF mode (A-F for reweighting).
        scale: Reweighting scale (>1 boosts, <1 dampens).
    
    Returns:
        Modified attention weights.
    """
    weights = attention_weights.clone()
    suffix_start = modality_bounds.action_bounds[0]

    vlm_start, vlm_end = modality_bounds.vlm_bounds
    aux_start, aux_end = modality_bounds.aux_visual_bounds
    proprio_start, proprio_end = modality_bounds.action_bounds

    # A: Reweight VLM tokens
    if cf_mode == CfAttnMode.VLM_ONLY:
        # Boost VLM, dampen others
        weights[:, :, suffix_start:, vlm_start:vlm_end] *= scale
        if aux_start < aux_end:
            weights[:, :, suffix_start:, aux_start:aux_end] *= (1.0 / scale)
        if proprio_start < proprio_end:
            weights[:, :, suffix_start:, proprio_start:proprio_end] *= (1.0 / scale)
    
    # C: Reweight proprio tokens
    elif cf_mode == CfAttnMode.PROPRIO_ONLY:
        if proprio_start < proprio_end:
            weights[:, :, suffix_start:, proprio_start:proprio_end] *= scale
        weights[:, :, suffix_start:, vlm_start:vlm_end] *= (1.0 / scale)
        if aux_start < aux_end:
            weights[:, :, suffix_start:, aux_start:aux_end] *= (1.0 / scale)

    # D: Reweight VLM + aux visual (all visual)
    elif cf_mode == CfAttnMode.IMAGE_ONLY:
        weights[:, :, suffix_start:, vlm_start:vlm_end] *= scale
        if aux_start < aux_end:
            weights[:, :, suffix_start:, aux_start:aux_end] *= scale
        if proprio_start < proprio_end:
            weights[:, :, suffix_start:, proprio_start:proprio_end] *= (1.0 / scale)

    return weights


def get_cf_weighting_config(mode: str) -> Dict[str, float]:
    """Get modality weighting configuration for A-F modes.
    
    These modes define which modalities to boost/dampen in attention.
    
    | Mode | Description |
    |------|-------------|
    | A    | Reweight VLM (primary visual+language) |
    | B    | Reweight aux visual (additional cameras) |
    | C    | Reweight proprio (state) |
    | D    | Reweight all visual (VLM + aux) |
    | E    | Reweight VLM + proprio |
    | F    | Reweight aux visual + proprio |
    
    Args:
        mode: Mode letter (A-F).
    
    Returns:
        Dictionary with modality weights:
        {"vlm": 1.0, "aux_visual": 1.0, "proprio": 1.0}
    """
    mode = mode.upper()
    
    # Default: all modalities equal weight
    config = {"vlm": 1.0, "aux_visual": 1.0, "proprio": 1.0}
    
    if mode == "A":
        # Boost VLM, dampen others
        config["vlm"] = 2.0
        config["aux_visual"] = 0.5
        config["proprio"] = 0.5
    elif mode == "B":
        # Boost aux visual, dampen others
        config["vlm"] = 0.5
        config["aux_visual"] = 2.0
        config["proprio"] = 0.5
    elif mode == "C":
        # Boost proprio, dampen others
        config["vlm"] = 0.5
        config["aux_visual"] = 0.5
        config["proprio"] = 2.0
    elif mode == "D":
        # Boost all visual, dampen proprio
        config["vlm"] = 2.0
        config["aux_visual"] = 2.0
        config["proprio"] = 0.5
    elif mode == "E":
        # Boost VLM + proprio, dampen aux
        config["vlm"] = 2.0
        config["aux_visual"] = 0.5
        config["proprio"] = 2.0
    elif mode == "F":
        # Boost aux + proprio, dampen VLM
        config["vlm"] = 0.5
        config["aux_visual"] = 2.0
        config["proprio"] = 2.0
    
    return config


def visualize_attn_mask(
    mask: torch.Tensor,
    prefix_len: int,
    modality_bounds: Optional[ModalityBounds] = None,
) -> str:
    """Visualize attention mask as ASCII art for debugging.
    
    Args:
        mask: Attention mask tensor [B, seq_len, seq_len].
        prefix_len: Prefix length.
        modality_bounds: Optional ModalityBounds for modality labels.
    
    Returns:
        ASCII visualization string.
    """
    # Take first batch
    m = mask[0].cpu().numpy()
    seq_len = m.shape[0]
    
    lines = []
    lines.append(f"Attention Mask Visualization (seq_len={seq_len}, prefix={prefix_len})")
    lines.append("=" * 60)
    
    # Header with modality labels if available
    if modality_bounds:
        header = "Position: "
        for i in range(seq_len):
            mod = modality_bounds.get_modality_at_position(i)
            if mod == "vlm":
                header += "V"
            elif mod == "aux_visual":
                header += "A"
            elif mod == "action":
                header += "X"
            elif mod == "soft_prompt":
                header += "S"
            else:
                header += "?"
        lines.append(header[:60])
    
    # Suffix start marker
    marker = "          " + "." * prefix_len + "|" + "." * (seq_len - prefix_len)
    lines.append(marker[:60])
    
    # Matrix visualization (condensed)
    lines.append("Suffix rows (action tokens attending to positions):")
    for i in range(prefix_len, min(prefix_len + 5, seq_len)):
        row = m[i]
        row_str = f"  Row {i}: "
        for j in range(min(seq_len, 40)):
            row_str += "1" if row[j] > 0.5 else "0"
        if seq_len > 40:
            row_str += "..."
        lines.append(row_str)
    
    return "\n".join(lines)


# ------------------------------------------------------------------------------
# Utility functions for CF guidance computation
# ------------------------------------------------------------------------------

def compute_cf_weighted_actions(
    baseline_actions: torch.Tensor,
    cf_actions: torch.Tensor,
    guidance_scale: float = 0.1,
    weighting_mode: str = "E",
) -> torch.Tensor:
    """Compute CF-weighted actions using modality reweighting.
    
    This implements the A-F mode guidance from OpenPI:
    baseline + guidance_scale * (baseline - cf_actions)
    
    The weighting_mode determines which CF actions to use:
    - A: Use cf_actions from NO_VLM mode (boost VLM contribution)
    - B: Use cf_actions from NO_AUX_VISUAL mode
    - C: Use cf_actions from NO_PROPRIO mode
    - D: Use cf_actions from NO_PROPRIO (boost all visual)
    - E: Use cf_actions from NO_AUX_VISUAL (boost VLM + proprio)
    - F: Use cf_actions from NO_VLM (boost aux + proprio)
    
    Args:
        baseline_actions: Actions from full-modal inference.
        cf_actions: Actions from CF inference (blocked modality).
        guidance_scale: Scale for guidance delta.
        weighting_mode: Mode A-F.
    
    Returns:
        Weighted action tensor.
    """
    # Compute delta (contribution of blocked modality)
    delta = baseline_actions - cf_actions
    
    # Apply guidance
    weighted_actions = baseline_actions + guidance_scale * delta
    
    return weighted_actions


def compute_cf_guidance_with_threshold(
    baseline_actions: torch.Tensor,
    cf_actions: torch.Tensor,
    effect_threshold: float = 0.5,
    guidance_scale: float = 0.1,
) -> Tuple[torch.Tensor, float, bool]:
    """Compute CF guidance with effect threshold check.
    
    If the CF effect exceeds threshold, the blocked modality is
    considered important and we return baseline unchanged.
    
    Args:
        baseline_actions: Actions from full-modal inference.
        cf_actions: Actions from CF inference.
        effect_threshold: Threshold for deciding if guidance is needed.
        guidance_scale: Scale for guidance.
    
    Returns:
        Tuple of (guided_actions, effect_value, guidance_applied).
    """
    from .cf_mode import compute_modality_effect
    
    effect = compute_modality_effect(baseline_actions, cf_actions)
    
    if effect > effect_threshold:
        # Modality is important, don't apply guidance
        return baseline_actions, effect, False
    
    # Apply guidance
    delta = baseline_actions - cf_actions
    guided = baseline_actions + guidance_scale * delta
    
    return guided, effect, True