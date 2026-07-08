# IDR Framework

**Infer-Diagnose-Refine Framework for Test-time Adaptation in Vision-Language-Action Models**

This repository contains the implementation of IDR, a model-agnostic framework for test-time action refinement in VLA models. IDR diagnoses the dynamic importance of visual observations through counterfactual inference and refines action predictions without any retraining.

## Method Overview

IDR operates in three stages at each timestep:

1. **Infer**: Construct counterfactual scenarios via zero-padding interventions on visual and proprioceptive inputs
2. **Diagnose**: Quantify causal effects using norm-based measurement
3. **Refine**: Apply gated residual fusion to refine the base action prediction

## Supported VLA Models

| Model | Size | Framework |
|-------|------|-----------|
| π₀.₅ | Small (<4B) | OpenPI (JAX/PyTorch) |
| X-VLA | Tiny (<1B) | PyTorch |
| OpenVLA-OFT | Large (≥4B) | Prismatic (PyTorch) |
| VLA-Adapter | Tiny (<1B) | Transformers |

## Environment Setup

### Common Dependencies

```bash
conda create -n idr python=3.10
conda activate idr
pip install numpy
```

### π₀.₅ (OpenPI)

Follow the official [OpenPI installation guide](https://github.com/Physical-Intelligence/openpi):

```bash
# Clone OpenPI repository
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi

# Install with uv
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Additional dependencies for IDR
pip install libero

# Download model checkpoints
# By default, checkpoints are auto-downloaded from gs://openpi-assets
# π₀.₅-LIBERO: gs://openpi-assets/checkpoints/pi05_libero
```

### X-VLA

Follow the official [X-VLA installation guide](https://github.com/nvidia/X-VLA):

```bash
# Clone X-VLA repository
git clone https://github.com/nvidia/X-VLA.git
cd X-VLA

# Install dependencies
pip install torch torchvision
pip install transformers accelerate

# Download model checkpoints (see X-VLA documentation)
```

### OpenVLA-OFT

Follow the official OpenVLA installation guide (see your organization's documentation for OpenVLA-OFT specific setup):

```bash
# Clone OpenVLA repository
git clone https://github.com/openvla/openvla.git
cd openvla

# Install dependencies
pip install torch torchvision
pip install prismatic-vla
```

### VLA-Adapter

Follow the official [VLA-Adapter installation guide](https://github.com/nvidia/VLA-Adapter):

```bash
# Clone VLA-Adapter repository
git clone https://github.com/nvidia/VLA-Adapter.git
cd VLA-Adapter

# Install dependencies
pip install torch torchvision transformers
pip install dlimp
```

## Model Checkpoints

After installing the respective frameworks, download the model checkpoints:

| Model | Checkpoint | Description |
|-------|------------|-------------|
| π₀.₅ | `gs://openpi-assets/checkpoints/pi05_libero` | π₀.₅ fine-tuned for LIBERO |
| X-VLA-LIBERO | `<X-VLA-CKPT>/X-VLA-Libero` | X-VLA for LIBERO benchmark |
| X-VLA-Calvin | `<X-VLA-CKPT>/X-VLA-Calvin-ABC_D` | X-VLA for CALVIN benchmark |
| X-VLA-SIMPLER | `<X-VLA-CKPT>/X-VLA-SIMPLER` | X-VLA for SIMPLER benchmark |

Set checkpoint paths in the scripts or environment variables:
```bash
export OPENPI_CKPT_DIR=~/.cache/openpi  # For OpenPI models
export XVLA_CKPT_DIR=/path/to/xvla/checkpoints  # For X-VLA models
```

## Project Structure

```
IDR-framework/
├── README.md
├── docs/
│   └── method_details.md      # Detailed method documentation
├── src/
│   ├── idr/                   # Core IDR implementation (framework-agnostic)
│   │   ├── __init__.py
│   │   └── refiner.py         # Main IDR refiner
│   ├── pi05/                  # π₀.₅ implementation (OpenPI/JAX)
│   │   ├── cf_sampler.py      # Counterfactual sampler
│   │   ├── attention_mask.py  # Attention mask generation
│   │   ├── modality_bounds.py  # Modality position tracking
│   │   └── policy.py          # Policy with CF support
│   ├── xvla/                  # X-VLA implementation (PyTorch)
│   │   ├── cf_policy.py       # Counterfactual policy wrapper
│   │   ├── cf_mode.py        # CF mode definitions
│   │   └── modality_bounds.py # Modality position tracking
│   ├── vla_adapter/            # VLA-Adapter implementation
│   │   ├── wrapper.py         # CF wrapper
│   │   ├── config.py         # Configuration
│   │   ├── utils.py          # Utilities
│   │   └── strategies/        # CF strategies
│   └── openvla_oft/          # OpenVLA-OFT implementation
│       └── run_libero_eval_cf.py  # Evaluation script
└── scripts/                   # Evaluation scripts
    ├── pi05/                  # π₀.₅ scripts
    ├── xvla/                  # X-VLA scripts
    ├── openvla_oft/           # OpenVLA-OFT scripts
    └── vla_adapter/            # VLA-Adapter scripts
```

## Usage

### π₀.₅ on LIBERO

```bash
cd scripts/pi05

# Set checkpoint directory
export CHECKPOINT_DIR=~/.cache/openpi/checkpoints/pi05_libero

# Run baseline evaluation
./run_libero_idr.sh --cf_mode BASE

# Run IDR evaluation (Mode E)
./run_libero_idr.sh --cf_mode E
```

### X-VLA on LIBERO

```bash
cd scripts/xvla

# Set model path
export MODEL_PATH=/path/to/X-VLA-Libero

# Terminal 1: Start server
./start_server.sh

# Terminal 2: Run evaluation
./run_libero_idr.sh --weight_mode E
```

## Hyperparameters

| Parameter | Description | Default (π₀.₅) | Default (Others) |
|-----------|-------------|----------------|------------------|
| α (alpha) | Visual correction scale | 0.08 | 0.10 |
| τ (tau) | Intervention threshold | 7.0 | 0.5 |
| β (beta) | Proprioceptive regularization | 0.05 | 0.05 |
| λ (lambda) | Clip bound | 0.1 | 0.1 |

## Citation

```bibtex
@article{idr2026,
  title={Causality-driven Infer-Diagnose-Refine Framework for Test-time Adaptation in Vision-Language-Action Models},
  author={Anonymous Authors},
  journal={Manuscript},
  year={2026}
}
```

## Acknowledgments

This project is built upon the following open-source repositories:
- [OpenPI](https://github.com/Physical-Intelligence/openpi) - π₀.₅ implementation
- [X-VLA](https://github.com/nvidia/X-VLA) - X-VLA implementation
- [OpenVLA](https://github.com/openvla/openvla) - OpenVLA base implementation
- [VLA-Adapter](https://github.com/nvidia/VLA-Adapter) - VLA-Adapter implementation
