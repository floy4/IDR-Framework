"""
Counterfactual VLA Wrapper

This module provides a unified wrapper that integrates all three CF strategies:
- InputZeroingStrategy (Method A)
- AttentionMaskStrategy (Method B)
- OutputReweightingStrategy (Method C)

The wrapper allows easy switching between strategies via configuration.
"""

from typing import Dict, Optional, Tuple, Union, Any
import numpy as np
from PIL import Image as PILImage

from prismatic.cf_attention.config import CfConfig, CfMethod, CfMode
from prismatic.cf_attention.strategies.input_zeroing import InputZeroingStrategy
from prismatic.cf_attention.strategies.attention_mask import AttentionMaskStrategy
from prismatic.cf_attention.strategies.output_reweighting import OutputReweightingStrategy


class CfVLAWrapper:
    """
    Unified wrapper for counterfactual VLA inference.
    
    This wrapper:
    1. Wraps any VLA model (OpenVLA, PrismaticVLM)
    2. Provides a unified interface for CF inference
    3. Supports switching between three CF methods via config
    4. Returns both actions and CF metrics
    
    Usage:
        ```python
        from prismatic.cf_attention import CfVLAWrapper, CfConfig, CfMethod, CfMode
        
        # Load VLA model
        vla_model = OpenVLA.from_pretrained(...)
        
        # Create CF wrapper
        config = CfConfig(
            method=CfMethod.INPUT_ZEROING,  # or ATTENTION_MASK, OUTPUT_REWEIGHTING
            mode=CfMode.E,  # A-F modes
            cf_guidance_scale=0.1,
        )
        cf_wrapper = CfVLAWrapper(vla_model, config)
        
        # Run CF inference
        actions, metrics = cf_wrapper.predict_action_with_cf(
            image=image,
            instruction="pick up the cup",
        )
        ```
    
    The wrapper does NOT modify the original VLA model code.
    All CF logic is implemented in separate strategy classes.
    """
    
    def __init__(self, vla_model: Any, config: Optional[CfConfig] = None):
        """
        Initialize CfVLAWrapper.
        
        Args:
            vla_model: The VLA model (OpenVLA or PrismaticVLM)
            config: CF configuration (default: CfConfig())
        """
        self.vla_model = vla_model
        self.config = config or CfConfig()
        
        # Initialize strategies
        self._strategies = {
            CfMethod.INPUT_ZEROING: InputZeroingStrategy(vla_model, self.config),
            CfMethod.ATTENTION_MASK: AttentionMaskStrategy(vla_model, self.config),
            CfMethod.OUTPUT_REWEIGHTING: OutputReweightingStrategy(vla_model, self.config),
        }
        
        # Set current strategy
        self._current_strategy = self._strategies[self.config.method]
    
    def predict_action_with_cf(
        self,
        image: Union[PILImage.Image, np.ndarray],
        instruction: str,
        proprio_state: Optional[np.ndarray] = None,
        unnorm_key: Optional[str] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Predict action with counterfactual analysis.
        
        This method delegates to the appropriate strategy based on config.method.
        
        Args:
            image: Input image (PIL Image or numpy array)
            instruction: Task instruction string
            proprio_state: Optional proprioceptive state
            unnorm_key: Dataset key for unnormalization
            **kwargs: Additional arguments for the underlying predict_action
        
        Returns:
            Tuple of (actions, cf_metrics)
            - actions: numpy array of shape [action_horizon, action_dim]
            - cf_metrics: Optional dict with effect_vlm, effect_prop, use_baseline, etc.
        """
        return self._current_strategy.predict_action_with_cf(
            image=image,
            instruction=instruction,
            proprio_state=proprio_state,
            unnorm_key=unnorm_key,
            **kwargs
        )
    
    def predict_action(
        self,
        image: Union[PILImage.Image, np.ndarray],
        instruction: str,
        unnorm_key: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Standard predict_action without CF (passthrough to original model).
        
        Args:
            image: Input image
            instruction: Task instruction
            unnorm_key: Dataset key
            **kwargs: Additional arguments
        
        Returns:
            Actions array
        """
        return self.vla_model.predict_action(image, instruction, unnorm_key=unnorm_key, **kwargs)
    
    def set_method(self, method: CfMethod):
        """
        Switch to a different CF method.
        
        Args:
            method: The CF method to use
        """
        if isinstance(method, str):
            method = CfMethod(method)
        
        self.config.method = method
        self._current_strategy = self._strategies[method]
    
    def set_mode(self, mode: CfMode):
        """
        Switch to a different CF reweighting mode.
        
        Args:
            mode: The CF mode (A-F)
        """
        if isinstance(mode, str):
            mode = CfMode(mode.upper())
        
        self.config.mode = mode
    
    def set_config(self, config: CfConfig):
        """
        Update the entire configuration.
        
        Args:
            config: New configuration
        """
        self.config = config
        self._current_strategy = self._strategies[config.method]
        
        # Update strategy configs
        for strategy in self._strategies.values():
            strategy.config = config
    
    def get_config(self) -> CfConfig:
        """
        Get current configuration.
        
        Returns:
            Current CfConfig
        """
        return self.config
    
    def get_strategy(self) -> Any:
        """
        Get the current strategy instance.
        
        Returns:
            Current strategy object
        """
        return self._current_strategy
    
    def run_comparison_analysis(
        self,
        image: Union[PILImage.Image, np.ndarray],
        instruction: str,
        proprio_state: Optional[np.ndarray] = None,
        unnorm_key: Optional[str] = None,
        methods: Optional[list] = None,
        modes: Optional[list] = None,
        **kwargs
    ) -> Dict:
        """
        Run comprehensive comparison across all methods and modes.
        
        Useful for research and debugging.
        
        Args:
            image: Input image
            instruction: Task instruction
            proprio_state: Optional proprioceptive state
            unnorm_key: Dataset key
            methods: List of methods to compare (default: all three)
            modes: List of modes to compare (default: A-F)
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with results for each method/mode combination
        """
        if methods is None:
            methods = [CfMethod.INPUT_ZEROING, CfMethod.ATTENTION_MASK, CfMethod.OUTPUT_REWEIGHTING]
        
        if modes is None:
            modes = [CfMode.BASE, CfMode.A, CfMode.B, CfMode.C, CfMode.D, CfMode.E, CfMode.F]
        
        results = {}
        original_method = self.config.method
        original_mode = self.config.mode
        
        for method in methods:
            method_key = method.value
            results[method_key] = {}
            
            self.set_method(method)
            
            for mode in modes:
                self.set_mode(mode)
                
                actions, metrics = self.predict_action_with_cf(
                    image=image,
                    instruction=instruction,
                    proprio_state=proprio_state,
                    unnorm_key=unnorm_key,
                    **kwargs
                )
                
                results[method_key][mode.value] = {
                    "actions": actions,
                    "metrics": metrics,
                }
        
        # Restore original config
        self.set_method(original_method)
        self.set_mode(original_mode)
        
        return results
    
    def analyze_cf_effects(
        self,
        image: Union[PILImage.Image, np.ndarray],
        instruction: str,
        unnorm_key: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Analyze CF effects without applying reweighting.
        
        Returns effect magnitudes and deltas for analysis.
        
        Args:
            image: Input image
            instruction: Task instruction
            unnorm_key: Dataset key
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with effect analysis
        """
        # Get baseline action
        actions_baseline = self.vla_model.predict_action(image, instruction, unnorm_key=unnorm_key, **kwargs)
        
        # Get image CF (using input zeroing for reliable comparison)
        from prismatic.cf_attention.utils import create_zero_image, compute_cf_effect, compute_delta
        
        zero_image = create_zero_image(image)
        cf_instruction = "" if self.config.clear_prompt_for_image_cf else instruction
        actions_img0 = self.vla_model.predict_action(zero_image, cf_instruction, unnorm_key=unnorm_key, **kwargs)
        
        # Compute effects
        effect_vlm = compute_cf_effect(actions_baseline, actions_img0)
        delta_img = compute_delta(actions_baseline, actions_img0)
        
        analysis = {
            "baseline_actions": actions_baseline,
            "image_cf_actions": actions_img0,
            "effect_vlm": effect_vlm,
            "delta_img": delta_img,
            "delta_img_norm": np.linalg.norm(delta_img),
        }
        
        # Add threshold check
        analysis["exceeds_threshold"] = effect_vlm > self.config.vlm_effect_upper_threshold
        analysis["threshold"] = self.config.vlm_effect_upper_threshold
        
        return analysis
    
    @property
    def device(self):
        """Get model device."""
        return self.vla_model.device
    
    @property
    def action_tokenizer(self):
        """Get action tokenizer."""
        return self.vla_model.action_tokenizer
    
    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get action dimension."""
        return self.vla_model.get_action_dim(unnorm_key)
    
    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict:
        """Get action normalization statistics."""
        return self.vla_model.get_action_stats(unnorm_key)
    
    # Delegate vision backbone methods
    def get_prompt_builder(self):
        """Get prompt builder."""
        return self.vla_model.get_prompt_builder()
    
    # Expose underlying model for advanced usage
    def get_model(self) -> Any:
        """
        Get the underlying VLA model.
        
        Use this for direct access to the model when needed.
        
        Returns:
            The wrapped VLA model
        """
        return self.vla_model


def create_cf_wrapper(
    vla_model: Any,
    method: str = "input_zeroing",
    mode: str = "E",
    cf_guidance_scale: float = 0.1,
    **kwargs
) -> CfVLAWrapper:
    """
    Convenience function to create a CF wrapper with simple parameters.
    
    Args:
        vla_model: The VLA model
        method: CF method name ("input_zeroing", "attention_mask", "output_reweighting")
        mode: CF mode letter ("A", "B", "C", "D", "E", "F")
        cf_guidance_scale: Guidance scale
        **kwargs: Additional config parameters
    
    Returns:
        CfVLAWrapper instance
    """
    config = CfConfig(
        method=CfMethod(method),
        mode=CfMode(mode.upper()),
        cf_guidance_scale=cf_guidance_scale,
        **kwargs
    )
    
    return CfVLAWrapper(vla_model, config)