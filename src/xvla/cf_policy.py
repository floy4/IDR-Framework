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

"""Counterfactual policy wrapper for X-VLA.

Two CF approaches:
1. Input Zeroing (infer_single_cf_mode): Modify input features
   - VLM: Zero image_input[:, 0] + mark mask invalid
   - Aux Visual: Zero image_input[:, 1:] + mark mask invalid
   - Proprio: Zero proprio tensor (feature-level blocking)

2. Attention Mask (infer_single_cf_mode_with_mask): Control attention via mask
   - VLM: Block action→vlm attention (precise, sequence-level)
   - Aux Visual: Block action→aux attention (precise, sequence-level)
   - Proprio: Cannot block via mask (embedded in action tokens)
              Falls back to input zeroing with warning

Recommended usage:
- For VLM/Aux blocking: Use attention mask (more principled)
- For Proprio blocking: Use input zeroing (only available option)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

import torch
import numpy as np

from .modality_bounds import ModalityBounds, create_modality_bounds
from .cf_mode import (
    CfAttnMode,
    CfWeightMode,
    get_modality_visibility,
    compute_modality_effect,
    get_cf_mode_description,
    get_cf_weight_config,
    get_cf_weight_mode_description,
    WeightValue,
)


@dataclass
class CfPolicyResult:
    """Result from counterfactual policy inference."""

    actions_by_mode: Dict[str, torch.Tensor] = field(default_factory=dict)
    modality_bounds: Optional[ModalityBounds] = None
    effects: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_actions(self, mode: str = "base") -> Optional[torch.Tensor]:
        return self.actions_by_mode.get(mode)

    def get_baseline_actions(self) -> Optional[torch.Tensor]:
        return self.get_actions("base")

    def get_cf_effects(self) -> Dict[str, float]:
        return self.effects

    def to_dict(self) -> dict:
        return {
            "actions_by_mode": {
                k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v
                for k, v in self.actions_by_mode.items()
            },
            "modality_bounds": self.modality_bounds.to_dict() if self.modality_bounds else None,
            "effects": self.effects,
            "metadata": self.metadata,
        }


class CfXVLAPolicy:
    """Counterfactual policy wrapper for X-VLA.

    Two CF blocking approaches:
    1. Input Zeroing (infer_single_cf_mode): All modalities can be blocked
    2. Attention Mask (infer_single_cf_mode_with_mask): Only VLM/Aux can be blocked
       Proprio will fall back to input zeroing with warning
    """

    def __init__(
        self,
        model,
        cf_guidance_scale: float = 0.1,
        visual_effect_threshold: float = 0.5,
        default_steps: int = 10,
    ):
        self.model = model
        self.cf_guidance_scale = cf_guidance_scale
        self.visual_effect_threshold = visual_effect_threshold
        self.default_steps = default_steps

    def _get_modality_bounds_from_inputs(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: Optional[torch.Tensor],
        proprio: torch.Tensor,
        num_actions: int,
    ) -> ModalityBounds:
        """Compute ModalityBounds from input tensors."""
        # Prefer exact token counts from the encoder output.
        vlm_token_count = 128
        num_views = image_input.shape[1] if image_input.dim() == 5 else 1
        aux_visual_token_count = (num_views - 1) * 64 if num_views > 1 else 0
        if image_mask is not None:
            try:
                enc = self.model.forward_vlm(input_ids, image_input, image_mask)
                vlm_token_count = int(enc["vlm_features"].shape[1])
                aux_visual_token_count = int(enc["aux_visual_inputs"].shape[1])
            except Exception as exc:
                logging.warning(
                    "Failed to infer exact modality token counts; falling back to estimates. "
                    f"Reason: {exc}"
                )

        # Soft prompt count
        soft_prompt_count = getattr(self.model.transformer, "len_soft_prompts", 0)

        # Proprio position in action tokens (feature dimension)
        dim_action = self.model.action_space.dim_action
        dim_proprio = getattr(self.model.action_space, "dim_proprio", dim_action)

        proprio_offset = dim_action
        proprio_len = dim_proprio

        return create_modality_bounds(
            num_actions=num_actions,
            vlm_token_count=vlm_token_count,
            aux_visual_token_count=aux_visual_token_count,
            soft_prompt_count=soft_prompt_count,
            proprio_offset=proprio_offset,
            proprio_len=proprio_len,
        )

    @torch.no_grad()
    def infer_single_cf_mode(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        cf_mode: CfAttnMode,
        steps: int = None,
    ) -> torch.Tensor:
        """Run inference with CF mode using input zeroing.

        Blocks modalities by zeroing input features:
        - VLM: Zero image_input[:, 0], mark mask invalid
        - Aux Visual: Zero image_input[:, 1:], mark mask invalid
        - Proprio: Zero proprio tensor

        This is the most general approach - all modalities can be blocked.
        """
        steps = steps or self.default_steps
        visibility = get_modality_visibility(cf_mode)

        modified_image_input = image_input.clone()
        modified_image_mask = image_mask.clone()
        modified_proprio = proprio.clone()

        # Block VLM (primary view)
        if not visibility["vlm"]:
            modified_image_input[:, 0] = 0.0
            modified_image_mask[:, 0] = 0

        # Block aux visual
        if not visibility["aux_visual"]:
            if image_input.shape[1] > 1:
                modified_image_input[:, 1:] = 0.0
                modified_image_mask[:, 1:] = 0

        # Block all visual
        if not visibility["vlm"] and not visibility["aux_visual"]:
            modified_image_input = torch.zeros_like(image_input)
            modified_image_mask = torch.zeros_like(image_mask)

        # Block proprio (feature-level)
        if not visibility["proprio"]:
            modified_proprio = torch.zeros_like(proprio)

        actions = self.model.generate_actions(
            input_ids=input_ids,
            image_input=modified_image_input,
            image_mask=modified_image_mask,
            domain_id=domain_id,
            proprio=modified_proprio,
            steps=steps,
        )

        return actions

    @torch.no_grad()
    def infer_single_cf_mode_with_mask(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        cf_mode: CfAttnMode,
        steps: int = None,
    ) -> torch.Tensor:
        """Run inference with CF mode using attention mask.

        Blocks modalities via attention mask injection:
        - VLM: Block action→vlm attention (precise, sequence-level)
        - Aux Visual: Block action→aux attention (precise, sequence-level)
        - Proprio: CANNOT be blocked via mask (embedded in action tokens)
                   Falls back to input zeroing with warning

        More principled than input zeroing for VLM/Aux blocking.
        """
        from .cf_attention_hook import CFAttentionHookController, CFAttentionMaskContext

        steps = steps or self.default_steps

        # Get modality bounds
        modality_bounds = self._get_modality_bounds_from_inputs(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            proprio=proprio,
            num_actions=self.model.num_actions,
        )

        # Check if proprio blocking requested (cannot be done via mask)
        visibility = get_modality_visibility(cf_mode)
        if not visibility["proprio"]:
            logging.warning(
                f"CF mode {cf_mode.value}: proprio cannot be blocked via attention mask "
                "(embedded in action tokens). Falling back to input zeroing."
            )
            # Fall back to input zeroing for proprio
            return self.infer_single_cf_mode(
                input_ids=input_ids,
                image_input=image_input,
                image_mask=image_mask,
                domain_id=domain_id,
                proprio=torch.zeros_like(proprio),  # Zero proprio
                cf_mode=cf_mode,
                steps=steps,
            )

        # Create hook controller
        controller = CFAttentionHookController(self.model.transformer)

        # Run with attention mask context
        with CFAttentionMaskContext(
            controller,
            cf_mode,
            modality_bounds,
            batch_size=input_ids.shape[0],
            device=input_ids.device,
            dtype=self.model.transformer.pos_emb.dtype,
        ):
            actions = self.model.generate_actions(
                input_ids=input_ids,
                image_input=image_input,
                image_mask=image_mask,
                domain_id=domain_id,
                proprio=proprio,
                steps=steps,
            )

        return actions

    @torch.no_grad()
    def infer_with_cf_mask(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        cf_modes: List[CfAttnMode] = None,
        steps: int = None,
        compute_effects: bool = True,
    ) -> CfPolicyResult:
        """Run multi-mode CF analysis using attention mask.

        For modes that include proprio blocking, falls back to input zeroing.
        """
        from .cf_attention_hook import CFAttentionHookController, CFAttentionMaskContext

        steps = steps or self.default_steps
        if cf_modes is None:
            cf_modes = [CfAttnMode.BASE, CfAttnMode.NO_IMAGE, CfAttnMode.NO_VLM,
                        CfAttnMode.NO_AUX_VISUAL, CfAttnMode.NO_PROPRIO]

        result = CfPolicyResult()
        result.metadata["steps"] = steps
        result.metadata["cf_modes"] = [m.value for m in cf_modes]
        result.metadata["cf_method"] = "attention_mask"

        modality_bounds = self._get_modality_bounds_from_inputs(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            proprio=proprio,
            num_actions=self.model.num_actions,
        )
        result.modality_bounds = modality_bounds

        controller = CFAttentionHookController(self.model.transformer)

        for mode in cf_modes:
            mode_name = mode.value
            logging.info(f"Running CF inference (mask-based) with mode: {mode_name}")

            # NO_PROPRIO falls back to input zeroing
            if mode == CfAttnMode.NO_PROPRIO:
                actions = self.infer_single_cf_mode(
                    input_ids=input_ids,
                    image_input=image_input,
                    image_mask=image_mask,
                    domain_id=domain_id,
                    proprio=proprio,
                    cf_mode=mode,
                    steps=steps,
                )
            else:
                with CFAttentionMaskContext(
                    controller,
                    mode,
                    modality_bounds,
                    batch_size=input_ids.shape[0],
                    device=input_ids.device,
                    dtype=self.model.transformer.pos_emb.dtype,
                ):
                    actions = self.model.generate_actions(
                        input_ids=input_ids,
                        image_input=image_input,
                        image_mask=image_mask,
                        domain_id=domain_id,
                        proprio=proprio,
                        steps=steps,
                    )

            result.actions_by_mode[mode_name] = actions

        # Compute effects
        if compute_effects:
            baseline = result.actions_by_mode.get("base")
            if baseline is not None:
                for mode_name, cf_actions in result.actions_by_mode.items():
                    if mode_name != "base":
                        effect_key = f"{mode_name.replace('no_', '')}_effect"
                        result.effects[effect_key] = compute_modality_effect(baseline, cf_actions)

        return result

    @torch.no_grad()
    def compare_cf_methods(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        cf_mode: CfAttnMode,
        steps: int = None,
    ) -> Dict[str, Any]:
        """Compare input zeroing vs attention mask CF approaches."""
        steps = steps or self.default_steps

        baseline_actions = self.model.generate_actions(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            domain_id=domain_id,
            proprio=proprio,
            steps=steps,
        )

        actions_input_zeroing = self.infer_single_cf_mode(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            domain_id=domain_id,
            proprio=proprio,
            cf_mode=cf_mode,
            steps=steps,
        )

        actions_attention_mask = self.infer_single_cf_mode_with_mask(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            domain_id=domain_id,
            proprio=proprio,
            cf_mode=cf_mode,
            steps=steps,
        )

        diff = compute_modality_effect(actions_input_zeroing, actions_attention_mask)

        return {
            "baseline": baseline_actions,
            "actions_input_zeroing": actions_input_zeroing,
            "actions_attention_mask": actions_attention_mask,
            "difference": diff,
            "cf_mode": cf_mode.value,
            "steps": steps,
        }

    @torch.no_grad()
    def infer_with_cf(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        cf_modes: List[CfAttnMode] = None,
        steps: int = None,
        compute_effects: bool = True,
    ) -> CfPolicyResult:
        """Run inference with multiple CF modes using input zeroing."""
        steps = steps or self.default_steps
        if cf_modes is None:
            cf_modes = [CfAttnMode.BASE, CfAttnMode.NO_IMAGE, CfAttnMode.NO_PROPRIO]

        result = CfPolicyResult()
        result.metadata["steps"] = steps
        result.metadata["cf_modes"] = [m.value for m in cf_modes]
        result.metadata["cf_guidance_scale"] = self.cf_guidance_scale

        for mode in cf_modes:
            mode_name = mode.value
            logging.info(f"Running CF inference with mode: {mode_name}")

            actions = self.infer_single_cf_mode(
                input_ids=input_ids,
                image_input=image_input,
                image_mask=image_mask,
                domain_id=domain_id,
                proprio=proprio,
                cf_mode=mode,
                steps=steps,
            )
            result.actions_by_mode[mode_name] = actions

        result.modality_bounds = self._get_modality_bounds_from_inputs(
            input_ids,
            image_input,
            image_mask,
            proprio,
            self.model.num_actions,
        )

        if compute_effects:
            baseline = result.get_baseline_actions()
            if baseline is not None:
                for mode_name, cf_actions in result.actions_by_mode.items():
                    if mode_name != "base":
                        effect_key = f"{mode_name.replace('no_', '')}_effect"
                        result.effects[effect_key] = compute_modality_effect(baseline, cf_actions)

        return result

    @torch.no_grad()
    def _infer_cf_actions_with_backend(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        cf_mode: CfAttnMode,
        steps: int,
        cf_visual_backend: str,
    ) -> tuple[torch.Tensor, str]:
        """Run single CF mode with selected backend.

        Args:
            cf_visual_backend: One of "input", "mask", "hybrid".
                - "input": use input-zeroing for all modalities.
                - "mask": use attention mask for visual modalities.
                - "hybrid": same as "mask" for visual modalities.
                  Proprio is always blocked by input-zeroing.

        Returns:
            Tuple of (actions, method_name), where method_name is "input" or "mask".
        """
        backend = str(cf_visual_backend).lower()
        if backend not in {"input", "mask", "hybrid"}:
            raise ValueError(
                f"Unknown cf_visual_backend='{cf_visual_backend}'. "
                "Expected one of ['input', 'mask', 'hybrid']."
            )

        # Proprio cannot be blocked via sequence-level attention mask.
        if cf_mode == CfAttnMode.NO_PROPRIO:
            return (
                self.infer_single_cf_mode(
                    input_ids=input_ids,
                    image_input=image_input,
                    image_mask=image_mask,
                    domain_id=domain_id,
                    proprio=proprio,
                    cf_mode=cf_mode,
                    steps=steps,
                ),
                "input",
            )

        if backend == "input":
            return (
                self.infer_single_cf_mode(
                    input_ids=input_ids,
                    image_input=image_input,
                    image_mask=image_mask,
                    domain_id=domain_id,
                    proprio=proprio,
                    cf_mode=cf_mode,
                    steps=steps,
                ),
                "input",
            )

        return (
            self.infer_single_cf_mode_with_mask(
                input_ids=input_ids,
                image_input=image_input,
                image_mask=image_mask,
                domain_id=domain_id,
                proprio=proprio,
                cf_mode=cf_mode,
                steps=steps,
            ),
            "mask",
        )

    @torch.no_grad()
    def infer_with_delta_weighting(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        weight_mode: CfWeightMode = CfWeightMode.E,
        steps: int = None,
        guidance_scale: float = None,
        effect_threshold: float = None,
        base_prop_scale: float = 0.05,
        clip_range: float = 0.1,
        cf_visual_backend: str = "hybrid",
    ) -> Dict[str, Any]:
        """Run inference with delta weighting."""
        steps = steps or self.default_steps
        guidance_scale = guidance_scale if guidance_scale is not None else self.cf_guidance_scale
        effect_threshold = effect_threshold if effect_threshold is not None else self.visual_effect_threshold
        cf_visual_backend = str(cf_visual_backend).lower()
        if cf_visual_backend not in {"input", "mask", "hybrid"}:
            raise ValueError(
                f"Unknown cf_visual_backend='{cf_visual_backend}'. "
                "Expected one of ['input', 'mask', 'hybrid']."
            )

        # Baseline
        baseline_actions = self.model.generate_actions(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            domain_id=domain_id,
            proprio=proprio,
            steps=steps,
        )

        # CF inference
        actions_no_vlm, method_no_vlm = self._infer_cf_actions_with_backend(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            domain_id=domain_id,
            proprio=proprio,
            cf_mode=CfAttnMode.NO_VLM,
            steps=steps,
            cf_visual_backend=cf_visual_backend,
        )
        actions_no_aux, method_no_aux = self._infer_cf_actions_with_backend(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            domain_id=domain_id,
            proprio=proprio,
            cf_mode=CfAttnMode.NO_AUX_VISUAL,
            steps=steps,
            cf_visual_backend=cf_visual_backend,
        )
        actions_no_proprio = self.infer_single_cf_mode(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            domain_id=domain_id,
            proprio=proprio,
            cf_mode=CfAttnMode.NO_PROPRIO,
            steps=steps,
        )
        actions_no_image, method_no_image = self._infer_cf_actions_with_backend(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            domain_id=domain_id,
            proprio=proprio,
            cf_mode=CfAttnMode.NO_IMAGE,
            steps=steps,
            cf_visual_backend=cf_visual_backend,
        )

        # Deltas
        delta_vlm = baseline_actions - actions_no_vlm
        delta_aux = baseline_actions - actions_no_aux
        delta_proprio = baseline_actions - actions_no_proprio
        delta_image = baseline_actions - actions_no_image

        # Effects
        effect_vlm = compute_modality_effect(baseline_actions, actions_no_vlm)
        effect_aux = compute_modality_effect(baseline_actions, actions_no_aux)
        effect_proprio = compute_modality_effect(baseline_actions, actions_no_proprio)
        effect_image = compute_modality_effect(baseline_actions, actions_no_image)

        effects = {
            "vlm_effect": effect_vlm,
            "aux_visual_effect": effect_aux,
            "proprio_effect": effect_proprio,
            "image_effect": effect_image,
        }

        deltas = {
            "delta_vlm": delta_vlm,
            "delta_aux": delta_aux,
            "delta_proprio": delta_proprio,
            "delta_image": delta_image,
        }

        refined_actions, threshold_applied = self._apply_weight_mode(
            baseline_actions, delta_vlm, delta_aux, delta_proprio, delta_image,
            weight_mode, guidance_scale, effect_threshold, base_prop_scale, clip_range, effects,
        )

        return {
            "actions": refined_actions,
            "baseline_actions": baseline_actions,
            "effects": effects,
            "deltas": deltas,
            "weight_mode": weight_mode.value,
            "threshold_applied": threshold_applied,
            "cf_visual_backend": cf_visual_backend,
            "cf_methods": {
                "no_vlm": method_no_vlm,
                "no_aux_visual": method_no_aux,
                "no_image": method_no_image,
                "no_proprio": "input",
            },
        }

    def _apply_weight_mode(
        self,
        baseline_actions, delta_vlm, delta_aux, delta_proprio, delta_image,
        weight_mode, guidance_scale, effect_threshold, base_prop_scale, clip_range, effects,
    ):
        threshold_applied = False
        config = get_cf_weight_config(weight_mode)

        if weight_mode in [CfWeightMode.A, CfWeightMode.E]:
            image_effect = effects.get("image_effect", 0.0)
            if image_effect > effect_threshold:
                threshold_applied = True
                return baseline_actions, threshold_applied

        weighted_delta = torch.zeros_like(baseline_actions)

        if weight_mode in [CfWeightMode.A, CfWeightMode.B, CfWeightMode.C,
                          CfWeightMode.D, CfWeightMode.E, CfWeightMode.F]:
            weighted_delta = weighted_delta + guidance_scale * delta_image

            if weight_mode == CfWeightMode.B:
                weighted_delta = weighted_delta + base_prop_scale * delta_proprio
            elif weight_mode == CfWeightMode.C:
                effect_proprio = effects.get("proprio_effect", 1.0)
                effect_image = effects.get("image_effect", 1.0)
                if effect_image > 0:
                    normalized_weight = base_prop_scale * (effect_proprio / effect_image)
                else:
                    normalized_weight = base_prop_scale
                weighted_delta = weighted_delta + normalized_weight * delta_proprio
            elif weight_mode == CfWeightMode.D:
                clipped_delta = torch.clamp(delta_proprio, -clip_range, clip_range)
                weighted_delta = weighted_delta + base_prop_scale * clipped_delta
            elif weight_mode == CfWeightMode.E:
                effect_proprio = effects.get("proprio_effect", 0.0)
                if effect_proprio > 0:
                    adaptive_weight = base_prop_scale * (1.0 / (1.0 + effect_proprio))
                else:
                    adaptive_weight = base_prop_scale
                clipped_delta = torch.clamp(delta_proprio, -clip_range, clip_range)
                weighted_delta = weighted_delta + adaptive_weight * clipped_delta
            elif weight_mode == CfWeightMode.F:
                smoothed_delta = torch.tanh(delta_proprio / clip_range) * clip_range
                weighted_delta = weighted_delta + base_prop_scale * smoothed_delta

        elif weight_mode == CfWeightMode.G:
            vlm_weight = float(config.get("vlm", 1.0))
            aux_weight = float(config.get("aux_visual", 0.5))
            proprio_weight = float(config.get("proprio", base_prop_scale))
            weighted_delta = weighted_delta + guidance_scale * (
                vlm_weight * delta_vlm + aux_weight * delta_aux + proprio_weight * delta_proprio
            )

        elif weight_mode == CfWeightMode.H:
            effect_vlm = effects.get("vlm_effect", 1.0)
            effect_aux = effects.get("aux_visual_effect", 0.5)
            effect_proprio = effects.get("proprio_effect", 0.5)
            total_effect = effect_vlm + effect_aux + effect_proprio

            if total_effect > 0:
                vlm_weight = (1.0 / effect_vlm) if effect_vlm > 0 else 1.0
                aux_weight = (1.0 / effect_aux) if effect_aux > 0 else 0.5
                proprio_weight = (1.0 / effect_proprio) if effect_proprio > 0 else base_prop_scale
                total_weight = vlm_weight + aux_weight + proprio_weight
                vlm_weight /= total_weight
                aux_weight /= total_weight
                proprio_weight /= total_weight
            else:
                vlm_weight, aux_weight, proprio_weight = 1.0, 0.5, base_prop_scale

            weighted_delta = weighted_delta + guidance_scale * (
                vlm_weight * delta_vlm + aux_weight * delta_aux + proprio_weight * delta_proprio
            )

        refined_actions = baseline_actions + weighted_delta
        return refined_actions, threshold_applied

    @torch.no_grad()
    def infer_input_level_cf(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        steps: int = None,
        processor: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run input-level CF: zero images + prompt (input_ids).

        Args:
            processor: Optional XVLAProcessor for tokenizing empty string.
                       If None, input_ids will be replaced with padding tokens.

        Returns:
            Dict with:
                - "actions_no_input": actions with all inputs zeroed
                - "actions_no_proprio": actions with proprio zeroed only
        """
        steps = steps or self.default_steps

        # 1. Zero all images
        modified_image_input = torch.zeros_like(image_input)
        modified_image_mask = torch.zeros_like(image_mask)

        # 2. Zero prompt (input_ids)
        # If processor provided, use it to tokenize empty string
        if processor is not None:
            empty_input_ids = processor.encode_language("")["input_ids"]
            # Ensure batch size matches
            if empty_input_ids.shape[0] != input_ids.shape[0]:
                empty_input_ids = empty_input_ids.expand(input_ids.shape[0], -1)
            modified_input_ids = empty_input_ids.to(input_ids.device)
        else:
            # Use padding token from existing input_ids or default
            # Find pad_token_id from VLM tokenizer if available
            pad_token_id = self._get_pad_token_id()
            modified_input_ids = torch.full_like(input_ids, pad_token_id)

        # Run inference with all inputs zeroed
        actions_no_input = self.model.generate_actions(
            input_ids=modified_input_ids,
            image_input=modified_image_input,
            image_mask=modified_image_mask,
            domain_id=domain_id,
            proprio=proprio,  # proprio unchanged
            steps=steps,
        )

        # 3. Zero proprio only (using existing method)
        actions_no_proprio = self.infer_single_cf_mode(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            domain_id=domain_id,
            proprio=proprio,
            cf_mode=CfAttnMode.NO_PROPRIO,
            steps=steps,
        )

        return {
            "actions_no_input": actions_no_input,
            "actions_no_proprio": actions_no_proprio,
        }

    def _get_pad_token_id(self) -> int:
        """Get padding token ID from VLM tokenizer or return default."""
        try:
            tokenizer = getattr(self.model.vlm, "tokenizer", None)
            if tokenizer is None:
                # Try to get from config
                tokenizer = getattr(self.model, "_tokenizer", None)
            if tokenizer is not None:
                pad_id = getattr(tokenizer, "pad_token_id", None)
                if pad_id is not None and pad_id >= 0:
                    return pad_id
                eos_id = getattr(tokenizer, "eos_token_id", None)
                if eos_id is not None and eos_id >= 0:
                    return eos_id
        except Exception:
            pass
        # Default padding token ID for BART-like models
        return 1

    @torch.no_grad()
    def infer_feature_level_cf(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        steps: int = None,
    ) -> Dict[str, torch.Tensor]:
        """Run feature-level CF: zero VLM output features (vlm_features).

        Args:
            steps: Number of denoising steps.

        Returns:
            Dict with:
                - "actions_no_vlm_feature": actions with vlm_features zeroed
                - "actions_no_proprio": actions with proprio zeroed only
        """
        steps = steps or self.default_steps

        # 1. Get VLM features normally
        enc = self.model.forward_vlm(input_ids, image_input, image_mask)

        # 2. Zero vlm_features (all VLM encoder output)
        enc["vlm_features"] = torch.zeros_like(enc["vlm_features"])

        # 3. Run transformer with zeroed vlm_features
        actions_no_vlm_feature = self._generate_actions_with_enc(
            enc=enc,
            domain_id=domain_id,
            proprio=proprio,
            steps=steps,
        )

        # 4. Zero proprio only (using existing method)
        actions_no_proprio = self.infer_single_cf_mode(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            domain_id=domain_id,
            proprio=proprio,
            cf_mode=CfAttnMode.NO_PROPRIO,
            steps=steps,
        )

        return {
            "actions_no_vlm_feature": actions_no_vlm_feature,
            "actions_no_proprio": actions_no_proprio,
        }

    @torch.no_grad()
    def _generate_actions_with_enc(
        self,
        enc: Dict[str, torch.Tensor],
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        """Generate actions using provided encoder outputs.

        This allows feature-level intervention by modifying enc before calling.

        Args:
            enc: Dict with "vlm_features" and "aux_visual_inputs"
            domain_id: Domain ID tensor
            proprio: Proprioception tensor
            steps: Number of denoising steps

        Returns:
            Generated actions tensor
        """
        self.model.eval()
        B = domain_id.shape[0]
        D = self.model.action_space.dim_action

        x1 = torch.randn(B, self.model.num_actions, D, device=proprio.device, dtype=proprio.dtype)
        action = torch.zeros_like(x1)

        steps = max(1, int(steps))
        for i in range(steps, 0, -1):
            t = torch.full((B,), i / steps, device=proprio.device, dtype=proprio.dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.model.action_space.preprocess(proprio, x_t)
            action = self.model.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                **enc,
            )
        return self.model.action_space.postprocess(action)

    @torch.no_grad()
    def infer_with_level_cf(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        cf_level: str = "input",
        steps: int = None,
        guidance_scale: float = None,
        effect_threshold: float = None,
        base_prop_scale: float = 0.05,
        clip_range: float = 0.1,
        processor: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run level-based CF with delta weighting (E-mode style).

        Args:
            cf_level: "input" for input-level CF, "feature" for feature-level CF
            guidance_scale: Scale for primary delta (input/vlm_feature)
            effect_threshold: Threshold for fallback to baseline
            base_prop_scale: Scale for proprio delta
            clip_range: Clip range for proprio delta
            processor: Optional processor for input-level CF empty string tokenization

        Returns:
            Dict with refined actions, effects, deltas, etc.
        """
        steps = steps or self.default_steps
        guidance_scale = guidance_scale if guidance_scale is not None else self.cf_guidance_scale
        effect_threshold = effect_threshold if effect_threshold is not None else self.visual_effect_threshold

        # 1. Baseline
        baseline_actions = self.model.generate_actions(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            domain_id=domain_id,
            proprio=proprio,
            steps=steps,
        )

        # 2. Run CF based on level
        if cf_level == "input":
            cf_results = self.infer_input_level_cf(
                input_ids=input_ids,
                image_input=image_input,
                image_mask=image_mask,
                domain_id=domain_id,
                proprio=proprio,
                steps=steps,
                processor=processor,
            )
            actions_no_primary = cf_results["actions_no_input"]
            primary_name = "input"
        elif cf_level == "feature":
            cf_results = self.infer_feature_level_cf(
                input_ids=input_ids,
                image_input=image_input,
                image_mask=image_mask,
                domain_id=domain_id,
                proprio=proprio,
                steps=steps,
            )
            actions_no_primary = cf_results["actions_no_vlm_feature"]
            primary_name = "vlm_feature"
        else:
            raise ValueError(f"Unknown cf_level: {cf_level}. Expected 'input' or 'feature'.")

        actions_no_proprio = cf_results["actions_no_proprio"]

        # 3. Compute deltas
        delta_primary = baseline_actions - actions_no_primary
        delta_proprio = baseline_actions - actions_no_proprio

        # 4. Compute effects (L2 distance)
        effect_primary = compute_modality_effect(baseline_actions, actions_no_primary)
        effect_proprio = compute_modality_effect(baseline_actions, actions_no_proprio)

        effects = {
            f"{primary_name}_effect": effect_primary,
            "proprio_effect": effect_proprio,
        }

        deltas = {
            f"delta_{primary_name}": delta_primary,
            "delta_proprio": delta_proprio,
        }

        # 5. Apply E-mode weighting
        refined_actions, threshold_applied, weights_info = self._apply_level_weight_mode(
            baseline_actions=baseline_actions,
            delta_primary=delta_primary,
            delta_proprio=delta_proprio,
            effect_primary=effect_primary,
            effect_proprio=effect_proprio,
            guidance_scale=guidance_scale,
            effect_threshold=effect_threshold,
            base_prop_scale=base_prop_scale,
            clip_range=clip_range,
        )

        return {
            "actions": refined_actions,
            "baseline_actions": baseline_actions,
            "effects": effects,
            "deltas": deltas,
            "weights": weights_info,
            "cf_level": cf_level,
            "threshold_applied": threshold_applied,
            "cf_methods": {
                f"no_{primary_name}": "input" if cf_level == "input" else "feature",
                "no_proprio": "input",
            },
        }

    def _apply_level_weight_mode(
        self,
        baseline_actions: torch.Tensor,
        delta_primary: torch.Tensor,
        delta_proprio: torch.Tensor,
        effect_primary: float,
        effect_proprio: float,
        guidance_scale: float,
        effect_threshold: float,
        base_prop_scale: float,
        clip_range: float,
    ) -> tuple[torch.Tensor, bool, Dict[str, Any]]:
        """Apply E-mode weighting for level-based CF.

        Logic (same as CfWeightMode.E):
        1. If effect_primary > threshold, return baseline (threshold fallback)
        2. weighted_delta = guidance_scale * delta_primary
        3. proprio adaptive weight: base_prop_scale * (1.0 / (1.0 + effect_proprio))
        4. clipped proprio delta added to weighted_delta

        Returns:
            Tuple of (refined_actions, threshold_applied, weights_info)
        """
        threshold_applied = False
        weights_info = {
            "primary_weight": guidance_scale,
            "proprio_weight": 0.0,
            "proprio_adaptive_weight": 0.0,
            "proprio_clipped": False,
            "effect_threshold": effect_threshold,
            "base_prop_scale": base_prop_scale,
            "clip_range": clip_range,
        }

        # 1. Threshold check
        if effect_primary > effect_threshold:
            threshold_applied = True
            weights_info["threshold_fallback"] = True
            weights_info["reason"] = f"effect_primary ({effect_primary:.4f}) > threshold ({effect_threshold:.4f})"
            return baseline_actions, threshold_applied, weights_info

        weights_info["threshold_fallback"] = False

        # 2. Primary delta
        weighted_delta = guidance_scale * delta_primary

        # 3. Proprio adaptive weight (E-mode)
        if effect_proprio > 0:
            adaptive_weight = base_prop_scale * (1.0 / (1.0 + effect_proprio))
        else:
            adaptive_weight = base_prop_scale

        weights_info["proprio_adaptive_weight"] = adaptive_weight

        clipped_delta_proprio = torch.clamp(delta_proprio, -clip_range, clip_range)

        # Check if clipping was applied
        proprio_clipped = not torch.allclose(delta_proprio, clipped_delta_proprio)
        weights_info["proprio_clipped"] = proprio_clipped

        weights_info["proprio_weight"] = adaptive_weight
        weighted_delta = weighted_delta + adaptive_weight * clipped_delta_proprio

        # 4. Refined actions
        refined_actions = baseline_actions + weighted_delta
        return refined_actions, threshold_applied, weights_info


def wrap_model_with_cf(
    model,
    cf_guidance_scale: float = 0.1,
    vlm_effect_threshold: float = 0.5,
    default_steps: int = 10,
) -> CfXVLAPolicy:
    """Wrap X-VLA model with CF policy."""
    return CfXVLAPolicy(
        model=model,
        cf_guidance_scale=cf_guidance_scale,
        visual_effect_threshold=vlm_effect_threshold,
        default_steps=default_steps,
    )