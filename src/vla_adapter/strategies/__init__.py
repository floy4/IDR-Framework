"""
Counterfactual Strategies Module

This module provides three counterfactual inference strategies:
- InputZeroingStrategy: Zero out specific modalities in input
- AttentionMaskStrategy: Modify attention masks to block information flow  
- OutputReweightingStrategy: Reweight output probability distributions
"""

from prismatic.cf_attention.strategies.input_zeroing import InputZeroingStrategy
from prismatic.cf_attention.strategies.attention_mask import AttentionMaskStrategy
from prismatic.cf_attention.strategies.output_reweighting import OutputReweightingStrategy

__all__ = [
    "InputZeroingStrategy",
    "AttentionMaskStrategy",
    "OutputReweightingStrategy",
]