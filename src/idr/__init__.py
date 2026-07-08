"""IDR Framework - Infer-Diagnose-Refine for VLA Models."""

from .refiner import (
    IDRActionRefiner,
    IDRConfig,
    create_idr_refiner,
    compute_effect_magnitude,
    MODEL_DEFAULTS,
)

__all__ = [
    "IDRActionRefiner",
    "IDRConfig",
    "create_idr_refiner",
    "compute_effect_magnitude",
    "MODEL_DEFAULTS",
]
