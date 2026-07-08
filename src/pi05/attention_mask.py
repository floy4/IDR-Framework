"""Counterfactual attention mask generation.

This module provides functions to create attention masks for counterfactual
analysis, allowing selective blocking or enabling of attention between
modalities.
"""

from enum import Enum
from typing import Optional

import jax.numpy as jnp
import numpy as np

from .modality_bounds import ModalityBounds


class CfAttnMode(str, Enum):
    """Counterfactual attention mode enumeration.
    
    Each mode defines which modalities the action tokens can attend to.
    
    BASE mode: Normal attention (all modalities visible to actions)
    
    Single modality isolation (block that modality → action):
    - NO_IMAGE: Actions cannot attend to image tokens
    - NO_LANGUAGE: Actions cannot attend to language tokens
    - NO_STATE: Actions cannot attend to state tokens
    
    Single modality focus (only that modality → action):
    - IMAGE_ONLY: Actions can only attend to images
    - LANGUAGE_ONLY: Actions can only attend to language
    - STATE_ONLY: Actions can only attend to state
    
    Dual modality combinations:
    - IMAGE_LANG: Actions attend to images + language (block state)
    - IMAGE_STATE: Actions attend to images + state (block language)
    - LANG_STATE: Actions attend to language + state (block images)
    
    Note: All modes maintain bidirectional attention within prefix,
    and causal attention within suffix (action tokens).
    """
    
    # Normal attention
    BASE = "base"
    
    # Block single modality
    NO_IMAGE = "no_image"
    NO_LANGUAGE = "no_lang"
    NO_STATE = "no_state"
    
    # Single modality only
    IMAGE_ONLY = "image_only"
    LANGUAGE_ONLY = "lang_only"
    STATE_ONLY = "state_only"
    
    # Dual modality combinations
    IMAGE_LANG = "image_lang"
    IMAGE_STATE = "image_state"
    LANG_STATE = "lang_state"


def get_modality_visibility(mode: CfAttnMode) -> dict[str, bool]:
    """Get visibility configuration for each modality based on CF mode.
    
    Args:
        mode: Counterfactual attention mode.
    
    Returns:
        Dictionary with keys "image", "language", "state" and boolean values
        indicating whether actions can attend to that modality.
    """
    # Default: all visible
    config = {"image": True, "language": True, "state": True}
    
    # Apply mode-specific visibility
    if mode == CfAttnMode.NO_IMAGE:
        config["image"] = False
    elif mode == CfAttnMode.NO_LANGUAGE:
        config["language"] = False
    elif mode == CfAttnMode.NO_STATE:
        config["state"] = False
    elif mode == CfAttnMode.IMAGE_ONLY:
        config["language"] = False
        config["state"] = False
    elif mode == CfAttnMode.LANGUAGE_ONLY:
        config["image"] = False
        config["state"] = False
    elif mode == CfAttnMode.STATE_ONLY:
        config["image"] = False
        config["language"] = False
    elif mode == CfAttnMode.IMAGE_LANG:
        config["state"] = False
    elif mode == CfAttnMode.IMAGE_STATE:
        config["language"] = False
    elif mode == CfAttnMode.LANG_STATE:
        config["image"] = False
    
    return config


def make_cf_attn_mask(
    base_attn_mask: jnp.ndarray,
    modality_bounds: ModalityBounds,
    cf_mode: CfAttnMode,
    *,
    sequence_length: Optional[int] = None,
) -> jnp.ndarray:
    """Create counterfactual attention mask by modifying base mask.
    
    This function takes a base attention mask and modifies it according to
    the specified counterfactual mode, blocking attention from action tokens
    to specific modalities.
    
    Args:
        base_attn_mask: Base attention mask of shape [batch, seq_len, seq_len]
                        or [seq_len, seq_len]. This is typically the mask
                        produced by the model's make_attn_mask function.
        modality_bounds: ModalityBounds object with position information.
        cf_mode: Counterfactual attention mode to apply.
        sequence_length: Total sequence length. If None, inferred from mask.
    
    Returns:
        Modified attention mask with same shape as base_attn_mask.
    
    Example:
        >>> # Create base mask from model
        >>> base_mask = make_attn_mask(input_mask, ar_mask)
        >>> # Apply counterfactual mode
        >>> cf_mask = make_cf_attn_mask(base_mask, bounds, CfAttnMode.NO_IMAGE)
        >>> # Use cf_mask in model forward pass
    """
    if cf_mode == CfAttnMode.BASE:
        return base_attn_mask
    
    # Get visibility configuration
    visibility = get_modality_visibility(cf_mode)
    
    # Infer sequence length
    if sequence_length is None:
        if base_attn_mask.ndim == 3:
            sequence_length = base_attn_mask.shape[1]
        else:
            sequence_length = base_attn_mask.shape[0]
    
    # Copy base mask (don't modify original)
    cf_mask = base_attn_mask.copy()
    
    # Get suffix start position
    suffix_start = modality_bounds.suffix_start
    
    # Handle batch dimension
    if cf_mask.ndim == 2:
        cf_mask = cf_mask[None, ...]  # Add batch dimension temporarily
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size = cf_mask.shape[0]
    
    # Block image → action attention
    if not visibility["image"]:
        for name, (start, end) in modality_bounds.image_bounds.items():
            # Action tokens (suffix) cannot attend to these image tokens
            # cf_mask[:, suffix_start:, start:end] = False
            cf_mask = cf_mask.at[:, suffix_start:sequence_length, start:end].set(False)
    
    # Block language → action attention
    if not visibility["language"]:
        lang_start, lang_end = modality_bounds.language_bounds
        if lang_start < lang_end:  # Only if language tokens exist
            cf_mask = cf_mask.at[:, suffix_start:sequence_length, lang_start:lang_end].set(False)
    
    # Block state → action attention
    if not visibility["state"]:
        state_start, state_end = modality_bounds.state_bounds
        if state_start < state_end:  # Only if state tokens exist
            cf_mask = cf_mask.at[:, suffix_start:sequence_length, state_start:state_end].set(False)
    
    # Remove batch dimension if we added it
    if squeeze_output:
        cf_mask = cf_mask[0]
    
    return cf_mask


def make_cf_attn_mask_from_positions(
    seq_len: int,
    prefix_len: int,
    image_positions: list[tuple[int, int]],
    language_positions: tuple[int, int],
    state_positions: tuple[int, int],
    cf_mode: CfAttnMode,
    *,
    batch_size: int = 1,
    dtype: jnp.dtype = jnp.bool_,
) -> jnp.ndarray:
    """Create counterfactual attention mask from position specifications.
    
    This is a convenience function that creates both the base attention mask
    and applies the counterfactual modification in one step.
    
    Args:
        seq_len: Total sequence length.
        prefix_len: Length of prefix tokens (bidirectional attention).
        image_positions: List of (start, end) tuples for image token positions.
        language_positions: (start, end) tuple for language token positions.
        state_positions: (start, end) tuple for state token positions.
        cf_mode: Counterfactual attention mode.
        batch_size: Batch size for the mask.
        dtype: Data type for the mask.
    
    Returns:
        Attention mask of shape [batch_size, seq_len, seq_len].
    
    Note:
        This creates a standard prefix-suffix attention pattern:
        - Prefix tokens: bidirectional attention within prefix
        - Suffix tokens: causal attention to all previous tokens
        Then applies counterfactual modifications.
    """
    # Create base attention mask
    base_mask = create_prefix_suffix_attn_mask(
        seq_len, prefix_len, batch_size=batch_size, dtype=dtype
    )
    
    # Create modality bounds
    image_bounds = {}
    for i, (start, end) in enumerate(image_positions):
        image_bounds[f"camera_{i}"] = (start, end)
    
    modality_bounds = ModalityBounds(
        image_bounds=image_bounds,
        language_bounds=language_positions,
        state_bounds=state_positions,
        prefix_len=prefix_len,
        suffix_start=prefix_len,
    )
    
    # Apply counterfactual modification
    return make_cf_attn_mask(base_mask, modality_bounds, cf_mode, sequence_length=seq_len)


def create_prefix_suffix_attn_mask(
    seq_len: int,
    prefix_len: int,
    *,
    batch_size: int = 1,
    dtype: jnp.dtype = jnp.bool_,
) -> jnp.ndarray:
    """Create standard prefix-suffix attention mask.
    
    This creates the base attention mask used in Pi0/Pi05 models:
    - Prefix tokens have bidirectional attention within the prefix
    - Suffix tokens have causal attention (can attend to all previous tokens)
    
    Args:
        seq_len: Total sequence length.
        prefix_len: Length of prefix tokens.
        batch_size: Batch size.
        dtype: Data type for the mask.
    
    Returns:
        Attention mask of shape [batch_size, seq_len, seq_len].
    
    Visual representation:
        ┌───────────────────────────────────────────────────────┐
        │  For prefix_len=10, seq_len=20:                        │
        │                                                        │
        │  Prefix (0-9):                                        │
        │  - Can attend to all prefix tokens (bidirectional)     │
        │  - Cannot attend to suffix tokens                     │
        │                                                        │
        │  Suffix (10-19):                                      │
        │  - Can attend to all prefix tokens                    │
        │  - Causal attention within suffix                     │
        │                                                        │
        │  Mask shape [20, 20]:                                  │
        │  ┌──────────┬──────────┐                              │
        │  │ 1s block │ 0s block │  (prefix → prefix: all 1s)  │
        │  │          │          │  (prefix → suffix: all 0s)  │
        │  ├──────────┼──────────┤                              │
        │  │ 1s row   │ causal   │  (suffix → prefix: all 1s)  │
        │  │          │ lower    │  (suffix → suffix: causal) │
        │  └──────────┴──────────┘                              │
        └───────────────────────────────────────────────────────┘
    """
    # Create mask using numpy (easier to manipulate), then convert to JAX
    mask = np.zeros((seq_len, seq_len), dtype=dtype)
    
    # Prefix: bidirectional attention within prefix
    mask[:prefix_len, :prefix_len] = True
    
    # Suffix: can attend to all prefix tokens
    mask[prefix_len:, :prefix_len] = True
    
    # Suffix: causal attention within suffix
    for i in range(prefix_len, seq_len):
        mask[i, prefix_len:i + 1] = True
    
    # Expand to batch
    if batch_size > 1:
        mask = np.tile(mask[None, ...], (batch_size, 1, 1))
    
    return jnp.array(mask)


def visualize_attn_mask(mask: jnp.ndarray, prefix_len: int = 0) -> str:
    """Create a string visualization of an attention mask.
    
    Useful for debugging and understanding attention patterns.
    
    Args:
        mask: Attention mask of shape [seq_len, seq_len] or [batch, seq_len, seq_len].
        prefix_len: Prefix length for visualization markers.
    
    Returns:
        String representation of the mask.
    """
    if mask.ndim == 3:
        mask = mask[0]  # Take first batch
    
    mask_np = np.array(mask)
    seq_len = mask_np.shape[0]
    
    lines = []
    lines.append(f"Attention Mask ({seq_len} x {seq_len})")
    if prefix_len > 0:
        lines.append(f"Prefix length: {prefix_len}, Suffix starts at: {prefix_len}")
    lines.append("")
    
    # Create visualization
    for i in range(seq_len):
        row = ""
        for j in range(seq_len):
            if mask_np[i, j]:
                row += "1"
            else:
                row += "0"
        
        # Add marker for prefix/suffix boundary
        if prefix_len > 0 and i == prefix_len - 1:
            row += " | (prefix end)"
        elif prefix_len > 0 and i == prefix_len:
            row += " | (suffix start)"
        
        lines.append(row)
    
    return "\n".join(lines)


def get_cf_modes_for_analysis() -> list[CfAttnMode]:
    """Get recommended CF modes for comprehensive modality analysis.
    
    Returns a list of modes that can be used to analyze the contribution
    of each modality to the action generation.
    
    Returns:
        List of CfAttnMode values for comprehensive analysis.
    """
    return [
        CfAttnMode.BASE,       # Baseline (all modalities)
        CfAttnMode.NO_IMAGE,   # Image contribution
        CfAttnMode.NO_LANGUAGE, # Language contribution
        CfAttnMode.NO_STATE,   # State contribution (if applicable)
        CfAttnMode.IMAGE_ONLY, # Image-only baseline
        CfAttnMode.LANGUAGE_ONLY, # Language-only baseline
        CfAttnMode.STATE_ONLY, # State-only baseline (if applicable)
    ]


def compute_modality_effect(
    baseline_actions: jnp.ndarray,
    cf_actions: jnp.ndarray,
    *,
    metric: str = "l2",
) -> jnp.ndarray:
    """Compute the effect size between baseline and counterfactual actions.
    
    Args:
        baseline_actions: Actions from baseline mode (all modalities).
        cf_actions: Actions from counterfactual mode.
        metric: Distance metric to use. Options: "l2", "l1", "cosine".
    
    Returns:
        Effect size as a scalar or per-action-dimension array.
    """
    if metric == "l2":
        diff = baseline_actions - cf_actions
        return jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
    elif metric == "l1":
        diff = baseline_actions - cf_actions
        return jnp.sum(jnp.abs(diff), axis=-1)
    elif metric == "cosine":
        # Cosine distance = 1 - cosine_similarity
        dot_product = jnp.sum(baseline_actions * cf_actions, axis=-1)
        norm_baseline = jnp.sqrt(jnp.sum(baseline_actions ** 2, axis=-1))
        norm_cf = jnp.sqrt(jnp.sum(cf_actions ** 2, axis=-1))
        cosine_sim = dot_product / (norm_baseline * norm_cf + 1e-8)
        return 1.0 - cosine_sim
    else:
        raise ValueError(f"Unknown metric: {metric}")


def create_cf_attn_mask_for_sampling(
    prefix_len: int,
    suffix_len: int,
    modality_bounds: ModalityBounds,
    cf_mode: CfAttnMode,
    *,
    batch_size: int = 1,
) -> jnp.ndarray:
    """Create CF attention mask for use in sample_actions.

    This mask blocks action tokens (suffix) from attending to specific modalities
    in the prefix, enabling counterfactual analysis at the attention level.

    Mask shape: [batch_size, suffix_len, prefix_len + suffix_len]

    The mask is applied to the full_attn_mask during flow matching sampling,
    blocking attention from suffix (action) positions to specific prefix positions.

    Args:
        prefix_len: Number of prefix tokens (image + language + state).
        suffix_len: Number of suffix tokens (state token + action tokens for Pi0,
                    or just action tokens for Pi05).
        modality_bounds: ModalityBounds with position information for each modality.
        cf_mode: Counterfactual attention mode to apply.
        batch_size: Batch size for the mask.

    Returns:
        CF attention mask of shape [batch_size, suffix_len, prefix_len + suffix_len].
        True indicates attention is allowed, False indicates blocked.

    Note:
        This mask only affects suffix → prefix attention. The suffix → suffix
        (causal) attention is preserved by the base mask.

    Example:
        >>> bounds = ModalityBounds(
        ...     image_bounds={"cam": (0, 100)},
        ...     language_bounds=(100, 150),
        ...     state_bounds=(150, 151),
        ...     prefix_len=151,
        ...     suffix_start=151,
        ... )
        >>> mask = create_cf_attn_mask_for_sampling(
        ...     prefix_len=151, suffix_len=17, bounds, CfAttnMode.NO_IMAGE
        ... )
        >>> # mask[:, :, 0:100] will be False (blocked image attention)
        >>> # mask[:, :, 100:151] will be True (language + state visible)
        >>> # mask[:, :, 151:168] will be True (suffix → suffix preserved)
    """
    if cf_mode == CfAttnMode.BASE:
        return jnp.ones((batch_size, suffix_len, prefix_len + suffix_len), dtype=jnp.bool_)

    visibility = get_modality_visibility(cf_mode)

    # Create mask with all True (no blocking initially)
    # Use numpy for creation, then convert to JAX array
    total_len = prefix_len + suffix_len
    cf_mask = np.ones((batch_size, suffix_len, total_len), dtype=np.bool_)

    # Block image → action attention
    if not visibility["image"]:
        for name, (start, end) in modality_bounds.image_bounds.items():
            # Ensure bounds are within valid range
            start = min(max(start, 0), total_len)
            end = min(max(end, 0), total_len)
            if start < end:
                # Action tokens (suffix positions 0 to suffix_len) cannot attend to image tokens
                cf_mask[:, :, start:end] = False

    # Block language → action attention
    if not visibility["language"]:
        lang_start, lang_end = modality_bounds.language_bounds
        # Ensure bounds are within valid range
        lang_start = min(max(lang_start, 0), total_len)
        lang_end = min(max(lang_end, 0), total_len)
        if lang_start < lang_end:
            cf_mask[:, :, lang_start:lang_end] = False

    # Block state → action attention
    if not visibility["state"]:
        state_start, state_end = modality_bounds.state_bounds
        # Ensure bounds are within valid range
        state_start = min(max(state_start, 0), total_len)
        state_end = min(max(state_end, 0), total_len)
        if state_start < state_end:
            cf_mask[:, :, state_start:state_end] = False

    # Note: suffix → suffix portion (prefix_len to prefix_len + suffix_len) is not modified
    # This preserves the causal attention within action tokens

    return jnp.array(cf_mask)