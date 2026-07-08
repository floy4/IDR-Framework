"""Counterfactual Attention Module for Pi0/Pi05 models.

This module provides tools for analyzing modality contributions through
attention mask manipulation, without modifying the original model files.

Key components:
- ModalityBounds: Data structure for tracking token boundaries of each modality
- ExtendedPaligemmaTokenizer: Extended tokenizer with boundary tracking
- CfAttnMode: Enumeration of counterfactual attention modes
- make_cf_attn_mask: Function to create counterfactual attention masks
- CfSampler: Wrapper for Pi0 model with CF sampling capability
- CfPolicy: Policy wrapper with CF analysis support
"""

from .modality_bounds import ModalityBounds
from .attention_mask import CfAttnMode, make_cf_attn_mask, create_cf_attn_mask_for_sampling
from .tokenizer_ext import ExtendedPaligemmaTokenizer
from .cf_sampler import CfSampler
from .policy_cf import CfPolicy

__all__ = [
    "ModalityBounds",
    "CfAttnMode",
    "make_cf_attn_mask",
    "create_cf_attn_mask_for_sampling",
    "ExtendedPaligemmaTokenizer",
    "CfSampler",
    "CfPolicy",
]