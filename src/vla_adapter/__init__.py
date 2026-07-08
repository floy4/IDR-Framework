"""
Counterfactual Attention Module for VLA-Adapter

This module provides three counterfactual inference strategies:
- Method A (Input Zeroing): Zero out specific modalities in input
- Method B (Attention Mask): Modify attention masks to block information flow
- Method C (Output Reweighting): Reweight output probability distributions

Reference: OpenPI's counterfactual attention mechanism
"""

from prismatic.cf_attention.config import CfConfig, CfMode, CfMethod
from prismatic.cf_attention.wrapper import CfVLAWrapper
from prismatic.cf_attention.strategies.input_zeroing import InputZeroingStrategy
from prismatic.cf_attention.strategies.attention_mask import AttentionMaskStrategy
from prismatic.cf_attention.strategies.output_reweighting import OutputReweightingStrategy
from prismatic.cf_attention.utils import compute_cf_effect, apply_cf_reweighting

__all__ = [
    "CfConfig",
    "CfMode",
    "CfMethod",
    "CfVLAWrapper",
    "InputZeroingStrategy",
    "AttentionMaskStrategy",
    "OutputReweightingStrategy",
    "compute_cf_effect",
    "apply_cf_reweighting",
]