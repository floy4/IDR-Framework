"""
Input Zeroing Strategy (Method A)

This strategy implements counterfactual analysis by zeroing out specific modalities
in the input. It follows OpenPI's approach in pi0.py:
- Create CF observation by zeroing proprio state
- Create CF observation by zeroing image (and optionally clearing prompt)
- Compute effect magnitudes and apply reweighting formula

Reference: OpenPI's _make_cf_observation_state_zero and _make_cf_observation_image_zero
"""

from typing import Dict, Optional, Tuple, Union, Any
import numpy as np
import torch
from PIL import Image as PILImage

from prismatic.cf_attention.config import CfConfig, CfMode, CfModality
from prismatic.cf_attention.utils import (
    compute_cf_effect,
    compute_delta,
    apply_cf_reweighting,
    create_zero_image,
    create_zero_state,
    get_cf_metrics,
)


class InputZeroingStrategy:
    """
    Counterfactual strategy using input feature zeroing.
    
    This approach:
    1. Runs baseline inference with original inputs
    2. Runs CF inference with zeroed proprio state
    3. Runs CF inference with zeroed images
    4. Computes effect magnitudes
    5. Applies reweighting formula (modes A-F)
    
    Advantages:
    - Simple implementation, no model modification needed
    - Directly follows OpenPI methodology
    - Easy to debug and understand
    
    Disadvantages:
    - Multiple inference passes (3x computation)
    - Zeroing inputs may cause out-of-distribution behavior
    """
    
    def __init__(self, vla_model: Any, config: CfConfig):
        """
        Initialize InputZeroingStrategy.
        
        Args:
            vla_model: The VLA model (OpenVLA or PrismaticVLM)
            config: CF configuration
        """
        self.vla_model = vla_model
        self.config = config
    
    def predict_action_with_cf(
        self,
        image: Union[PILImage.Image, np.ndarray],
        instruction: str,
        proprio_state: Optional[np.ndarray] = None,
        unnorm_key: Optional[str] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Predict action with counterfactual analysis using input zeroing.
        
        Args:
            image: Input image (PIL Image or numpy array)
            instruction: Task instruction string
            proprio_state: Optional proprioceptive state
            unnorm_key: Dataset key for unnormalization
            **kwargs: Additional arguments for predict_action
        
        Returns:
            Tuple of (actions, cf_metrics)
        """
        if not self.config.enable_cf:
            # Direct inference without CF
            actions = self.vla_model.predict_action(image, instruction, unnorm_key=unnorm_key, **kwargs)
            return actions, None
        
        # Step 1: Baseline inference with original inputs
        actions_baseline = self.vla_model.predict_action(image, instruction, unnorm_key=unnorm_key, **kwargs)
        
        if self.config.verbose:
            print(f"[CF] Baseline action shape: {actions_baseline.shape}")
        
        # Step 2: CF-Proprio inference (zero proprio state)
        # Note: OpenVLA doesn't directly support proprio input
        # For proprio CF, we use the same input since VLA-Adapter doesn't embed proprio
        if proprio_state is not None and self.config.zero_proprio_for_cf:
            # Try to remove proprio from instruction if embedded
            cf_instruction_prop = self._remove_proprio_from_instruction(instruction, proprio_state)
            actions_prop0 = self.vla_model.predict_action(image, cf_instruction_prop, unnorm_key=unnorm_key, **kwargs)
        else:
            # No proprio CF available, use baseline as proxy
            actions_prop0 = actions_baseline.copy()
        
        # Step 3: CF-Image inference (zero image)
        if self.config.zero_image_for_vlm_cf:
            zero_image = create_zero_image(image)
            
            # Optionally clear prompt for image CF
            if self.config.clear_prompt_for_image_cf:
                # Use empty/minimal instruction to estimate pure VLM effect
                cf_instruction_img = ""  # Or use a neutral prompt
            else:
                cf_instruction_img = instruction
            
            actions_img0 = self.vla_model.predict_action(zero_image, cf_instruction_img, unnorm_key=unnorm_key, **kwargs)
        else:
            actions_img0 = actions_baseline.copy()
        
        # Step 4: Compute effect magnitudes
        effect_prop = compute_cf_effect(actions_baseline, actions_prop0, method="l2")
        effect_vlm = compute_cf_effect(actions_baseline, actions_img0, method="l2")
        
        if self.config.verbose:
            print(f"[CF] Effect VLM: {effect_vlm:.4f}")
            print(f"[CF] Effect Prop: {effect_prop:.4f}")
        
        # Step 5: Compute deltas
        delta_img = compute_delta(actions_baseline, actions_img0)
        delta_prop = compute_delta(actions_baseline, actions_prop0)
        
        # Step 6: Apply CF reweighting
        actions_final, use_baseline = apply_cf_reweighting(
            actions_baseline=actions_baseline,
            delta_img=delta_img,
            delta_prop=delta_prop,
            effect_vlm=effect_vlm,
            effect_prop=effect_prop,
            mode=self.config.mode,
            cf_guidance_scale=self.config.cf_guidance_scale,
            vlm_effect_upper_threshold=self.config.vlm_effect_upper_threshold,
        )
        
        if self.config.verbose:
            print(f"[CF] Use baseline: {use_baseline}")
            print(f"[CF] CF mode: {self.config.mode.value}")
        
        # Step 7: Build metrics
        if self.config.return_cf_metrics:
            metrics = get_cf_metrics(
                effect_vlm=effect_vlm,
                effect_prop=effect_prop,
                use_baseline=use_baseline,
                mode=self.config.mode,
                delta_img_norm=np.linalg.norm(delta_img),
                delta_prop_norm=np.linalg.norm(delta_prop),
            )
            return actions_final, metrics
        
        return actions_final, None
    
    def _remove_proprio_from_instruction(self, instruction: str, proprio_state: np.ndarray) -> str:
        """
        Remove proprioceptive information from instruction.
        
        This is a placeholder implementation. In practice, if proprio is embedded
        in the instruction, you would need to parse and remove it.
        
        Args:
            instruction: Original instruction string
            proprio_state: Proprioceptive state array
        
        Returns:
            Modified instruction without proprio information
        """
        # OpenVLA doesn't embed proprio in instruction by default
        # This method is for potential future extensions
        return instruction
    
    def run_cf_analysis(
        self,
        image: Union[PILImage.Image, np.ndarray],
        instruction: str,
        proprio_state: Optional[np.ndarray] = None,
        modes_to_test: Optional[list] = None,
        unnorm_key: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Run comprehensive CF analysis across all modes.
        
        Useful for debugging and understanding CF effects.
        
        Args:
            image: Input image
            instruction: Task instruction
            proprio_state: Optional proprioceptive state
            modes_to_test: List of CfMode values to test (default: all)
            unnorm_key: Dataset key
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with analysis results for each mode
        """
        if modes_to_test is None:
            modes_to_test = [CfMode.BASE, CfMode.A, CfMode.B, CfMode.C, CfMode.D, CfMode.E, CfMode.F]
        
        results = {}
        original_mode = self.config.mode
        
        # Get base outputs first
        actions_baseline = self.vla_model.predict_action(image, instruction, unnorm_key=unnorm_key, **kwargs)
        
        zero_image = create_zero_image(image)
        cf_instruction = "" if self.config.clear_prompt_for_image_cf else instruction
        actions_img0 = self.vla_model.predict_action(zero_image, cf_instruction, unnorm_key=unnorm_key, **kwargs)
        
        if proprio_state is not None:
            cf_instruction_prop = self._remove_proprio_from_instruction(instruction, proprio_state)
            actions_prop0 = self.vla_model.predict_action(image, cf_instruction_prop, unnorm_key=unnorm_key, **kwargs)
        else:
            actions_prop0 = actions_baseline.copy()
        
        # Compute effects (shared across modes)
        effect_prop = compute_cf_effect(actions_baseline, actions_prop0)
        effect_vlm = compute_cf_effect(actions_baseline, actions_img0)
        delta_img = compute_delta(actions_baseline, actions_img0)
        delta_prop = compute_delta(actions_baseline, actions_prop0)
        
        for mode in modes_to_test:
            self.config.mode = mode
            actions, metrics = self.predict_action_with_cf(
                image=image,
                instruction=instruction,
                proprio_state=proprio_state,
                unnorm_key=unnorm_key,
                **kwargs
            )
            
            results[mode.value] = {
                "actions": actions,
                "metrics": metrics,
                "effect_vlm": effect_vlm,
                "effect_prop": effect_prop,
            }
        
        # Restore original mode
        self.config.mode = original_mode
        
        return results