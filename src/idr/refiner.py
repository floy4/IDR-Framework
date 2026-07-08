"""IDR Core Implementation - Infer-Diagnose-Refine Framework.

This module provides the framework-agnostic IDR refiner that diagnoses
visual importance by comparing factual and counterfactual action predictions,
then applies gated residual fusion for action refinement.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np


@dataclass
class IDRConfig:
    """Configuration for IDR action refiner.

    Attributes:
        alpha: Visual correction scale. Controls the strength of visual
               refinement. Higher values increase correction, but excessive
               values (>0.5) can destabilize action generation.
        tau: Intervention threshold. The gate activates when E_img < tau,
             indicating underutilization of visual cues.
        beta: Proprioceptive regularization scale. Controls the influence
              of bounded proprioceptive correction for stability.
        lambda_clip: Clip bound for proprioceptive residual. Bounds the
                     proprioceptive correction within [-lambda, lambda].
        epsilon: Small constant for numerical stability.
    """

    alpha: float = 0.08
    tau: float = 7.0
    beta: float = 0.05
    lambda_clip: float = 0.1
    epsilon: float = 1e-8


# Default hyperparameters for each VLA model
MODEL_DEFAULTS = {
    "pi05": IDRConfig(alpha=0.08, tau=7.0, beta=0.05, lambda_clip=0.1),
    "xvla": IDRConfig(alpha=0.10, tau=0.5, beta=0.05, lambda_clip=0.1),
    "openvla_oft": IDRConfig(alpha=0.10, tau=0.5, beta=0.05, lambda_clip=0.1),
    "vla_adapter": IDRConfig(alpha=0.10, tau=0.5, beta=0.05, lambda_clip=0.1),
}


class IDRActionRefiner:
    """Causality-aware action refiner implementing the IDR framework.

    The refiner diagnoses visual importance by comparing factual and counterfactual
    action predictions, then applies gated residual fusion for action refinement.

    The three stages are:
    1. Infer: Zero-padding interventions to generate counterfactual predictions
    2. Diagnose: Norm-based quantification of causal effects
    3. Refine: Gated residual fusion with bounded proprioceptive regularization
    """

    def __init__(
        self,
        alpha: float = 0.08,
        tau: float = 7.0,
        beta: float = 0.05,
        lambda_clip: float = 0.1,
        epsilon: float = 1e-8,
    ):
        """Initialize the IDR action refiner.

        Args:
            alpha: Visual correction scale.
            tau: Intervention threshold.
            beta: Proprioceptive regularization scale.
            lambda_clip: Clip bound for proprioception.
            epsilon: Numerical stability constant.
        """
        self.alpha = alpha
        self.tau = tau
        self.beta = beta
        self.lambda_clip = lambda_clip
        self.epsilon = epsilon

    def compute_causal_effects(
        self,
        a_base: np.ndarray,
        a_no_img: np.ndarray,
        a_no_prop: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Compute causal effects from factual and counterfactual predictions.

        Args:
            a_base: Base action prediction (factual).
            a_no_img: Action prediction without visual input (counterfactual).
            a_no_prop: Action prediction without proprioceptive input (counterfactual).

        Returns:
            Tuple of (E_img, E_prop, R_img):
                - E_img: Visual effect magnitude (L2 norm of visual deviation)
                - E_prop: Proprioceptive effect magnitude (L2 norm of proprio deviation)
                - R_img: Visual effect ratio (normalized visual importance)
        """
        # Compute action deviations (directional causal effects)
        delta_img = a_base - a_no_img
        delta_prop = a_base - a_no_prop

        # Compute effect magnitudes (L2 norm)
        E_img = float(np.linalg.norm(delta_img))
        E_prop = float(np.linalg.norm(delta_prop))

        # Compute normalized visual effect ratio
        R_img = E_img / (E_img + E_prop + self.epsilon)

        return E_img, E_prop, R_img

    def compute_gate(self, E_img: float) -> int:
        """Compute the intervention gate.

        The gate activates when E_img < tau, indicating the model underutilizes
        visual cues at the current timestep.

        Args:
            E_img: Visual effect magnitude.

        Returns:
            Gate value (1 if activated, 0 otherwise).
        """
        return 1 if E_img < self.tau else 0

    def refine_action(
        self,
        a_base: np.ndarray,
        delta_img: np.ndarray,
        E_img: float,
        E_prop: float,
        delta_prop: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply gated residual fusion to refine the base action.

        The refined action is computed as:
            a_final = a_base + g * (alpha * delta_img + w_prop * u_prop)

        where:
            g = indicator[E_img < tau]
            w_prop = beta * min(1, E_prop / (E_img + epsilon))
            u_prop = clip(delta_prop, -lambda, lambda)

        Args:
            a_base: Base action prediction.
            delta_img: Visual deviation (a_base - a_no_img).
            E_img: Visual effect magnitude.
            E_prop: Proprioceptive effect magnitude.
            delta_prop: Proprioceptive deviation (optional).

        Returns:
            Refined action prediction.
        """
        # Compute gate
        g = self.compute_gate(E_img)

        if g == 0:
            return a_base

        # Visual correction
        visual_correction = self.alpha * delta_img

        # Bounded proprioceptive regularizer
        if delta_prop is not None:
            w_prop = self.beta * min(1.0, E_prop / (E_img + self.epsilon))
            u_prop = np.clip(delta_prop, -self.lambda_clip, self.lambda_clip)
            visual_correction = visual_correction + w_prop * u_prop

        # Apply gated residual fusion
        a_final = a_base + g * visual_correction

        return a_final

    def __call__(
        self,
        a_base: np.ndarray,
        a_no_img: np.ndarray,
        a_no_prop: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Apply IDR refinement to a base action prediction.

        Args:
            a_base: Base action prediction (factual).
            a_no_img: Action prediction without visual input (counterfactual).
            a_no_prop: Action prediction without proprioceptive input (counterfactual).

        Returns:
            Tuple of (refined_action, metrics):
                - refined_action: Refined action prediction.
                - metrics: Dictionary containing causal effect metrics.
        """
        if a_no_prop is None:
            a_no_prop = np.zeros_like(a_base)

        # Compute causal effects
        E_img, E_prop, R_img = self.compute_causal_effects(a_base, a_no_img, a_no_prop)

        # Compute visual deviation
        delta_img = a_base - a_no_img
        delta_prop = a_base - a_no_prop

        # Apply refinement
        a_final = self.refine_action(a_base, delta_img, E_img, E_prop, delta_prop)

        # Compute gate status
        g = self.compute_gate(E_img)

        # Collect metrics
        metrics = {
            "E_img": E_img,
            "E_prop": E_prop,
            "R_img": R_img,
            "gate": g,
            "alpha": self.alpha,
            "tau": self.tau,
        }

        return a_final, metrics


def create_idr_refiner(
    model_name: str,
    alpha: Optional[float] = None,
    tau: Optional[float] = None,
    beta: Optional[float] = None,
    lambda_clip: Optional[float] = None,
) -> IDRActionRefiner:
    """Create an IDR action refiner with default hyperparameters for a given model.

    Args:
        model_name: Name of the VLA model ('pi05', 'xvla', 'openvla_oft', 'vla_adapter').
        alpha: Override visual correction scale.
        tau: Override intervention threshold.
        beta: Override proprioceptive regularization scale.
        lambda_clip: Override clip bound.

    Returns:
        IDRActionRefiner instance configured for the specified model.

    Raises:
        ValueError: If model_name is not recognized.
    """
    if model_name not in MODEL_DEFAULTS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_DEFAULTS.keys())}"
        )

    config = MODEL_DEFAULTS[model_name]

    return IDRActionRefiner(
        alpha=alpha if alpha is not None else config.alpha,
        tau=tau if tau is not None else config.tau,
        beta=beta if beta is not None else config.beta,
        lambda_clip=lambda_clip if lambda_clip is not None else config.lambda_clip,
    )


def compute_effect_magnitude(
    actions_a: np.ndarray,
    actions_b: np.ndarray,
    metric: str = "l2",
) -> float:
    """Compute effect magnitude between two action sequences.

    Args:
        actions_a: First action sequence.
        actions_b: Second action sequence.
        metric: Metric to use ('l2', 'l1', 'cosine').

    Returns:
        Effect magnitude as a scalar.
    """
    diff = actions_a - actions_b

    if metric == "l2":
        return float(np.linalg.norm(diff))
    elif metric == "l1":
        return float(np.sum(np.abs(diff)))
    elif metric == "cosine":
        norm_a = np.linalg.norm(actions_a) + 1e-8
        norm_b = np.linalg.norm(actions_b) + 1e-8
        cosine_sim = np.dot(actions_a.flatten(), actions_b.flatten()) / (norm_a * norm_b)
        return float(1.0 - cosine_sim)
    else:
        raise ValueError(f"Unknown metric: {metric}")
