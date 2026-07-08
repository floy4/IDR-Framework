"""
Configuration for Counterfactual Attention Module

Defines CfMode (counterfactual modes) and CfConfig (configuration parameters).
Reference: OpenPI's CfMode in pi0.py
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class CfMode(str, Enum):
    """
    Counterfactual reweighting mode.
    
    Modes align with the experimental schemes in OpenPI's pi0.py:
    - BASE: No counterfactual reweighting (passthrough)
    - A: Image-only reweighting
    - B: Image + fixed proprio weight
    - C: Adaptive weights + normalized proprio delta
    - D: Image + clipped proprio delta
    - E: Adaptive proprio weight + clipped proprio delta (recommended)
    - F: Image + soft-clipped proprio (tanh)
    """
    
    BASE = "base"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


class CfMethod(str, Enum):
    """
    Counterfactual implementation method.
    
    - INPUT_ZEROING: Zero out specific modalities in input (Method A)
    - ATTENTION_MASK: Modify attention masks to block information flow (Method B)
    - OUTPUT_REWEIGHTING: Reweight output probability distributions (Method C)
    """
    
    INPUT_ZEROING = "input_zeroing"
    ATTENTION_MASK = "attention_mask"
    OUTPUT_REWEIGHTING = "output_reweighting"


class CfModality(str, Enum):
    """
    Modalities that can be counterfactually modified.
    
    - IMAGE: Visual input (RGB images)
    - LANGUAGE: Text instruction/prompt
    - PROPRIO: Proprioceptive state (joint positions, etc.)
    """
    
    IMAGE = "image"
    LANGUAGE = "language"
    PROPRIO = "proprio"


@dataclass
class CfConfig:
    """
    Configuration for counterfactual inference.
    
    Attributes:
        enable_cf: Whether to enable counterfactual inference
        method: Which CF implementation method to use
        mode: Which CF reweighting mode to use (A-F)
        cf_guidance_scale: Scale factor for CF guidance (default: 0.1)
        vlm_effect_upper_threshold: Threshold for falling back to baseline (default: 0.5)
        
        # Modality-specific settings
        zero_image_for_vlm_cf: Whether to zero image for VLM CF analysis
        clear_prompt_for_image_cf: Whether to clear prompt for image CF analysis
        zero_proprio_for_cf: Whether to zero proprio for CF analysis
        
        # Method B specific settings
        attention_mask_image: Whether to mask image tokens
        attention_mask_language: Whether to mask language tokens
        attention_mask_proprio: Whether to mask proprio tokens
        
        # Method C specific settings
        cfg_scale: Classifier-free guidance scale (default: 1.0)
        
        # Debug settings
        return_cf_metrics: Whether to return CF metrics
        verbose: Whether to print verbose output
    """
    
    enable_cf: bool = True
    method: CfMethod = CfMethod.INPUT_ZEROING
    mode: CfMode = CfMode.E
    
    # Reweighting parameters (from OpenPI)
    cf_guidance_scale: float = 0.1
    vlm_effect_upper_threshold: float = 0.5
    
    # Method A: Input Zeroing settings
    zero_image_for_vlm_cf: bool = True
    clear_prompt_for_image_cf: bool = True
    zero_proprio_for_cf: bool = True
    
    # Method B: Attention Mask settings
    attention_mask_image: bool = True
    attention_mask_language: bool = False
    attention_mask_proprio: bool = False
    
    # Method C: Output Reweighting settings
    cfg_scale: float = 1.0
    
    # Debug settings
    return_cf_metrics: bool = True
    verbose: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if isinstance(self.method, str):
            self.method = CfMethod(self.method)
        if isinstance(self.mode, str):
            self.mode = CfMode(self.mode.upper())
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "CfConfig":
        """Create CfConfig from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert CfConfig to dictionary."""
        return {
            "enable_cf": self.enable_cf,
            "method": self.method.value,
            "mode": self.mode.value,
            "cf_guidance_scale": self.cf_guidance_scale,
            "vlm_effect_upper_threshold": self.vlm_effect_upper_threshold,
            "zero_image_for_vlm_cf": self.zero_image_for_vlm_cf,
            "clear_prompt_for_image_cf": self.clear_prompt_for_image_cf,
            "zero_proprio_for_cf": self.zero_proprio_for_cf,
            "attention_mask_image": self.attention_mask_image,
            "attention_mask_language": self.attention_mask_language,
            "attention_mask_proprio": self.attention_mask_proprio,
            "cfg_scale": self.cfg_scale,
            "return_cf_metrics": self.return_cf_metrics,
            "verbose": self.verbose,
        }