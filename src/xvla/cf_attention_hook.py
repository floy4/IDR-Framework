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

"""Counterfactual Attention Hook Controller for X-VLA.

Non-invasive attention mask injection for CF analysis.

CF Blocking Strategy:
- VLM:        Attention mask (precise, sequence-level)
- Aux Visual: Attention mask (precise, sequence-level)
- Proprio:    Input zeroing (approximate, feature-level)

Sequence: [action_tokens(with_proprio) | vlm | aux_visual | soft_prompts]
Proprio is embedded in action tokens, not a separate sequence position.
"""

from dataclasses import dataclass
from typing import List, Optional, Callable
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modality_bounds import ModalityBounds
from .cf_mode import CfAttnMode, get_modality_visibility


@dataclass
class CFAttentionState:
    """State for CF attention hook controller."""
    active: bool = False
    attn_mask: Optional[torch.Tensor] = None
    cf_mode: Optional[CfAttnMode] = None
    modality_bounds: Optional[ModalityBounds] = None


class CFAttentionHookController:
    """Non-invasive CF attention controller using monkey-patch.

    Temporarily modifies Attention.forward() to inject attention masks,
    then restores original methods after inference.

    Usage:
        controller = CFAttentionHookController(model.transformer)
        controller.activate_cf_mode(CfAttnMode.NO_VLM, modality_bounds)
        actions = model.generate_actions(...)  # Mask automatically applied
        controller.deactivate()  # Restore original behavior

    CF Blocking:
        - VLM/Aux: Blocked via attention mask (precise, sequence-level)
        - Proprio: Cannot be blocked via mask (embedded in action tokens)
                   Use input zeroing instead (infer_single_cf_mode)
    """

    def __init__(self, transformer: nn.Module):
        """Initialize hook controller.

        Args:
            transformer: The SoftPromptedTransformer module.
        """
        self.transformer = transformer
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._state = CFAttentionState()

        # Find all Attention modules
        self._attention_modules: List[nn.Module] = []
        for block in transformer.blocks:
            if hasattr(block, 'attn'):
                self._attention_modules.append(block.attn)

        logging.info(f"CFAttentionHookController: found {len(self._attention_modules)} attention modules")

    def _create_cf_attn_mask(
        self,
        seq_len: int,
        num_actions: int,
        modality_bounds: ModalityBounds,
        cf_mode: CfAttnMode,
        batch_size: int = 1,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Create attention mask for CF mode.

        Mask structure:
        - Sequence: [action_tokens | vlm | aux_visual | soft_prompts]
        - Action tokens should NOT attend to blocked modalities
        - Blocked modalities: VLM, Aux Visual (precise blocking)
        - Proprio: Cannot be blocked at sequence level (use input zeroing)

        Args:
            seq_len: Total sequence length.
            num_actions: Number of action tokens.
            modality_bounds: ModalityBounds with position info.
            cf_mode: Counterfactual mode.
            batch_size: Batch size.
            device: Device for tensor.
            dtype: Data type.

        Returns:
            Attention mask [B, 1, seq_len, seq_len] for SDPA.
            Value 1.0 = attend, 0.0 = blocked.
        """
        visibility = get_modality_visibility(cf_mode)

        # Create full attention mask (all positions can attend to all)
        mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=dtype, device=device)

        # Get position bounds
        action_start, action_end = 0, num_actions
        vlm_start, vlm_end = modality_bounds.vlm_bounds
        aux_start, aux_end = modality_bounds.aux_visual_bounds

        # Block VLM tokens from action's perspective (precise blocking)
        if not visibility["vlm"]:
            mask[:, :, action_start:action_end, vlm_start:vlm_end] = 0.0
            logging.debug(f"CF mask: blocked VLM [{vlm_start}:{vlm_end}] for action [{action_start}:{action_end}]")

        # Block aux visual tokens from action's perspective (precise blocking)
        if not visibility["aux_visual"]:
            if aux_start < aux_end:
                mask[:, :, action_start:action_end, aux_start:aux_end] = 0.0
                logging.debug(f"CF mask: blocked aux_visual [{aux_start}:{aux_end}]")

        # Proprio blocking: NOT possible at sequence level
        # Proprio is embedded in action tokens (feature dimension)
        # Use input zeroing approach instead (infer_single_cf_mode)
        if not visibility["proprio"]:
            logging.warning(
                "CF mask: proprio blocking not possible at sequence level "
                "(embedded in action tokens). Use input zeroing approach instead."
            )

        return mask

    def _monkey_patch_attention_forward(self, attn_module: nn.Module) -> tuple:
        """Create patched forward method with mask support.

        Returns:
            Tuple of (original_forward, patched_forward).
        """
        original_forward = attn_module.forward
        state = self._state

        def patched_forward(x: torch.Tensor) -> torch.Tensor:
            """Patched forward with attention mask support."""
            if not state.active or state.attn_mask is None:
                return original_forward(x)

            B, T, C = x.shape
            mask = state.attn_mask

            # Ensure mask shape matches
            if mask.shape[-1] != T or mask.shape[-2] != T:
                return original_forward(x)

            # Compute QKV
            qkv = (
                attn_module.qkv(x)
                .reshape(B, T, 3, attn_module.num_heads, attn_module.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)
            q = attn_module.q_norm(q)
            k = attn_module.k_norm(k)

            # Prepare mask for SDPA
            if mask.dim() == 4 and mask.shape[1] == 1:
                mask_expanded = mask.expand(B, attn_module.num_heads, T, T)
            else:
                mask_expanded = mask

            # Convert to boolean mask for SDPA
            bool_mask = mask_expanded > 0.5

            # Call SDPA with mask
            x_attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=bool_mask,
                dropout_p=attn_module.attn_drop.p if attn_module.training else 0.0,
            )

            # Reshape and project
            x_attn = x_attn.transpose(1, 2).reshape(B, T, C)
            x_attn = attn_module.proj(x_attn)
            x_attn = attn_module.proj_drop(x_attn)

            return x_attn

        return original_forward, patched_forward

    def activate_cf_mode(
        self,
        cf_mode: CfAttnMode,
        modality_bounds: ModalityBounds,
        batch_size: int = 1,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        """Activate CF attention mode.

        Args:
            cf_mode: Counterfactual mode.
            modality_bounds: ModalityBounds.
            batch_size: Batch size.
            device: Device.
            dtype: Data type.
        """
        if device is None:
            device = next(self.transformer.parameters()).device
        if dtype is None:
            dtype = next(self.transformer.parameters()).dtype

        seq_len = modality_bounds.total_seq_len
        num_actions = modality_bounds.num_actions

        # Create attention mask
        attn_mask = self._create_cf_attn_mask(
            seq_len=seq_len,
            num_actions=num_actions,
            modality_bounds=modality_bounds,
            cf_mode=cf_mode,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        # Update state
        self._state.active = True
        self._state.attn_mask = attn_mask
        self._state.cf_mode = cf_mode
        self._state.modality_bounds = modality_bounds

        # Monkey-patch forward methods
        self._original_forwards: List[Callable] = []
        for attn_module in self._attention_modules:
            original, patched = self._monkey_patch_attention_forward(attn_module)
            self._original_forwards.append(original)
            attn_module.forward = patched

        logging.info(f"CFAttentionHookController: activated CF mode {cf_mode}")

    def deactivate(self):
        """Deactivate and restore original forward methods."""
        if hasattr(self, '_original_forwards'):
            for i, attn_module in enumerate(self._attention_modules):
                if i < len(self._original_forwards):
                    attn_module.forward = self._original_forwards[i]
            self._original_forwards.clear()

        self._state.active = False
        self._state.attn_mask = None
        self._state.cf_mode = None

        logging.info("CFAttentionHookController: deactivated")

    def is_active(self) -> bool:
        """Check if CF mode is active."""
        return self._state.active

    def get_current_mode(self) -> Optional[CfAttnMode]:
        """Get current CF mode."""
        return self._state.cf_mode

    def __del__(self):
        """Cleanup."""
        self.deactivate()


class CFAttentionMaskContext:
    """Context manager for CF attention mask injection.

    Usage:
        controller = CFAttentionHookController(model.transformer)
        with CFAttentionMaskContext(controller, CfAttnMode.NO_VLM, bounds):
            actions = model.generate_actions(...)
    """

    def __init__(
        self,
        controller: CFAttentionHookController,
        cf_mode: CfAttnMode,
        modality_bounds: ModalityBounds,
        batch_size: int = 1,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        self.controller = controller
        self.cf_mode = cf_mode
        self.modality_bounds = modality_bounds
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

    def __enter__(self):
        self.controller.activate_cf_mode(
            self.cf_mode,
            self.modality_bounds,
            self.batch_size,
            self.device,
            self.dtype,
        )
        return self.controller

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.controller.deactivate()
        return False


def create_batched_cf_masks(
    modality_bounds: ModalityBounds,
    cf_modes: List[CfAttnMode],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create batched attention masks for parallel CF inference.

    Args:
        modality_bounds: ModalityBounds.
        cf_modes: List of CF modes.
        batch_size: Batch size.
        device: Device.
        dtype: Data type.

    Returns:
        Batched masks [B, 1, seq_len, seq_len].
    """
    seq_len = modality_bounds.total_seq_len
    num_actions = modality_bounds.num_actions

    masks = []
    for mode in cf_modes:
        mask = torch.ones(1, 1, seq_len, seq_len, dtype=dtype, device=device)
        visibility = get_modality_visibility(mode)

        action_start, action_end = 0, num_actions
        vlm_start, vlm_end = modality_bounds.vlm_bounds
        aux_start, aux_end = modality_bounds.aux_visual_bounds

        # Block VLM
        if not visibility["vlm"]:
            mask[:, :, action_start:action_end, vlm_start:vlm_end] = 0.0

        # Block aux visual
        if not visibility["aux_visual"] and aux_start < aux_end:
            mask[:, :, action_start:action_end, aux_start:aux_end] = 0.0

        # Proprio: warning logged, but not blocked at sequence level
        if not visibility["proprio"]:
            logging.warning(f"Batched CF mask: proprio blocking not possible for mode {mode.value}")

        masks.append(mask)

    batched_mask = torch.cat(masks, dim=0)

    if batch_size > len(cf_modes):
        extra = batch_size - len(cf_modes)
        batched_mask = torch.cat([batched_mask, batched_mask[-1:].expand(extra, -1, -1, -1)], dim=0)

    return batched_mask


def wrap_transformer_with_cf_hooks(transformer: nn.Module) -> CFAttentionHookController:
    """Create CF attention hook controller."""
    return CFAttentionHookController(transformer)


def run_inference_with_cf_mask(
    model: nn.Module,
    controller: CFAttentionHookController,
    cf_mode: CfAttnMode,
    modality_bounds: ModalityBounds,
    inference_func: Callable,
    *args,
    **kwargs,
) -> torch.Tensor:
    """Run inference with CF mask."""
    with CFAttentionMaskContext(controller, cf_mode, modality_bounds):
        return inference_func(*args, **kwargs)