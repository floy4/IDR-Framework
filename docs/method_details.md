# Method Details

## Overview

The Infer-Diagnose-Refine (IDR) framework is a model-agnostic approach for test-time adaptation in Vision-Language-Action (VLA) models. It operates entirely at inference time without requiring any retraining or parameter updates.

## Core Intuition

Visual observations have varying importance during robot manipulation:
- **Long-distance movements**: Visual cues may be less informative
- **Close-range interactions**: Visual cues become critical

Existing VLA models use static multimodal fusion with fixed weights, limiting adaptability. IDR treats visual importance as a dynamic causal effect and dynamically adjusts it during execution.

## Three-Stage Pipeline

### 1. Infer: Zero-Padding Interventions

Given a VLA model $\pi_{\theta^*}$ with frozen parameters, we first compute the factual base action:
$$a_{\text{base},t} = \pi_{\theta^*}(v_t, s_t, l)$$

We then construct counterfactual scenarios by applying zero-padding interventions:
- $do(V \leftarrow 0_v)$: Zero out visual observation
- $do(S \leftarrow 0_s)$: Zero out proprioceptive state

Counterfactual actions:
$$a_{\text{no\_img},t} = \pi_{\theta^*}(0_v, s_t, l)$$
$$a_{\text{no\_prop},t} = \pi_{\theta^*}(v_t, 0_s, l)$$

### 2. Diagnose: Norm-Based Quantification

We quantify the causal effects by measuring distributional shifts:

**Action deviations:**
$$\Delta_{\text{img},t} = a_{\text{base},t} - a_{\text{no\_img},t}$$
$$\Delta_{\text{prop},t} = a_{\text{base},t} - a_{\text{no\_prop},t}$$

**Effect magnitudes (L2 norm):**
$$E_{\text{img},t} = \|\Delta_{\text{img},t}\|_2$$
$$E_{\text{prop},t} = \|\Delta_{\text{prop},t}\|_2$$

**Visual effect ratio (normalized):**
$$R_{\text{img}} = \frac{E_{\text{img}}}{E_{\text{img}} + E_{\text{prop}} + \epsilon}$$

### 3. Refine: Gated Residual Fusion

The refined action is computed as:
$$a_{\text{final},t} = a_{\text{base},t} + g_t \cdot \left( \alpha \Delta_{\text{img},t} + w_{\text{prop},t} \cdot u_{\text{prop},t} \right)$$

Where:
- **Gate**: $g_t = \mathbb{I}[E_{\text{img},t} < \tau]$ — activates when visual effect is below threshold
- **Visual correction**: $\alpha \Delta_{\text{img},t}$ — scales the visual deviation
- **Bounded proprioceptive regularizer**:
  - $w_{\text{prop},t} = \beta \cdot \min(1, \frac{E_{\text{prop},t}}{E_{\text{img},t} + \epsilon})$
  - $u_{\text{prop},t} = \text{clip}(\Delta_{\text{prop},t}, -\lambda, \lambda)$

## Hyperparameters

| Parameter | Description | Default (π₀.₅) | Default (Others) |
|-----------|-------------|----------------|------------------|
| α (alpha) | Visual correction scale | 0.08 | 0.10 |
| τ (tau) | Intervention threshold | 7.0 | 0.5 |
| β (beta) | Proprioceptive regularization | 0.05 | 0.05 |
| λ (lambda) | Clip bound | 0.1 | 0.1 |

## Implementation Details

### π₀.₅ Implementation

The π₀.₅ implementation uses attention-level counterfactual analysis in the OpenPI framework (JAX):

- **cf_sampler.py**: Counterfactual sampler with attention-level interventions
- **attention_mask.py**: Attention mask generation for modality blocking
- **modality_bounds.py**: Track modality positions in the token sequence
- **policy.py**: Policy wrapper with CF sampling support

Key functions:
```python
# Create CF sampler
sampler = CfSampler(model)

# Sample with reweighting (Mode E)
actions, metrics = sampler.sample_with_cf_reweight(
    rng, observation,
    cf_guidance_scale=0.08,
    effect_threshold=7.0,
    return_metrics=True
)
```

### X-VLA Implementation

The X-VLA implementation uses input-level zeroing and attention masking in PyTorch:

- **cf_policy.py**: Counterfactual policy wrapper with delta weighting
- **cf_mode.py**: CF mode definitions and weight configurations
- **attention_mask.py**: Attention hook controller for mask injection
- **modality_bounds.py**: Modality position tracking

Key functions:
```python
# Create CF policy
cf_policy = CfXVLAPolicy(model, cf_guidance_scale=0.1, visual_effect_threshold=0.5)

# Run with delta weighting (Mode E)
result = cf_policy.infer_with_delta_weighting(
    input_ids, image_input, image_mask, domain_id, proprio,
    weight_mode=CfWeightMode.E,
    guidance_scale=0.1,
    effect_threshold=0.5
)
```

## Module Ablation

| Mode | Configuration | Avg Success |
|------|---------------|-------------|
| Baseline | No intervention | 96.25% |
| A | Without proprioceptive term | 96.30% |
| B | Without clip bound | 96.65% |
| C | Without adaptive $w_{\text{prop}}$ | 96.95% |
| D | Without gate | 96.20% |
| E (IDR) | Full model | **97.50%** |
| F | Negated α, β | 95.05% |

Key findings:
- All components contribute to performance
- Direction matters: negating correction direction drops below baseline
- Selective intervention is crucial: uniform correction (Mode D) underperforms

## Observations

### Observation 1: Environment shifts causal patterns
The same model shifts from proprioception-primary (LIBERO) to vision-dominant (SIMPLER) across environments.

### Observation 2: Architecture shapes patterns
- π₀.₅, OpenVLA-OFT: Vision-heavy
- X-VLA: Proprioception-primary
- VLA-Adapter: Vision-dominant

### Observation 3: Phase-dependent patterns
Visual effect ratio fluctuates with manipulation phases, increasing during localization/placement and decreasing during motion transitions.
