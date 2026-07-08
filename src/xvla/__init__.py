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

"""Counterfactual attention module for X-VLA.

Two CF blocking approaches:
1. Input Zeroing (infer_single_cf_mode): Modify input features
   - VLM: Zero image_input[:, 0], mark mask invalid
   - Aux Visual: Zero image_input[:, 1:], mark mask invalid
   - Proprio: Zero proprio tensor (feature-level blocking)

2. Attention Mask (infer_single_cf_mode_with_mask): Control attention via mask
   - VLM: Block action→vlm attention (precise, sequence-level)
   - Aux Visual: Block action→aux attention (precise, sequence-level)
   - Proprio: Cannot block via mask (embedded in action tokens)
              Falls back to input zeroing with warning

Recommended:
- For VLM/Aux blocking: Use attention mask (more principled)
- For Proprio blocking: Use input zeroing (only available option)

Sequence: [action_tokens(with_proprio) | vlm | aux_visual | soft_prompts]
"""

from .modality_bounds import ModalityBounds, create_modality_bounds
from .cf_mode import (
    CfAttnMode,
    CfWeightMode,
    get_modality_visibility,
    get_cf_mode_description,
    get_cf_modes_for_analysis,
    compute_modality_effect,
    map_legacy_mode,
    get_cf_weight_config,
    get_cf_weight_mode_description,
)
from .cf_policy import CfXVLAPolicy, CfPolicyResult, wrap_model_with_cf
from .attention_mask import (
    create_prefix_suffix_attn_mask,
    make_cf_attn_mask,
    apply_cf_attn_mask_to_model,
    get_cf_weighting_config,
    visualize_attn_mask,
    compute_cf_weighted_actions,
    compute_cf_guidance_with_threshold,
)
from .cf_attention_hook import (
    CFAttentionHookController,
    CFAttentionMaskContext,
    create_batched_cf_masks,
    wrap_transformer_with_cf_hooks,
    run_inference_with_cf_mask,
)

__all__ = [
    "ModalityBounds",
    "create_modality_bounds",
    # CF Attention Modes
    "CfAttnMode",
    "get_modality_visibility",
    "get_cf_mode_description",
    "get_cf_modes_for_analysis",
    "compute_modality_effect",
    "map_legacy_mode",
    # Delta Weighting Modes
    "CfWeightMode",
    "get_cf_weight_config",
    "get_cf_weight_mode_description",
    # CF Policy
    "CfXVLAPolicy",
    "CfPolicyResult",
    "wrap_model_with_cf",
    # Attention mask functions
    "create_prefix_suffix_attn_mask",
    "make_cf_attn_mask",
    "apply_cf_attn_mask_to_model",
    "get_cf_weighting_config",
    "visualize_attn_mask",
    "compute_cf_weighted_actions",
    "compute_cf_guidance_with_threshold",
    # Hook-based CF attention
    "CFAttentionHookController",
    "CFAttentionMaskContext",
    "create_batched_cf_masks",
    "wrap_transformer_with_cf_hooks",
    "run_inference_with_cf_mask",
]
