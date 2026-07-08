#!/bin/bash

# Run LIBERO-10 evaluation with Counterfactual reasoning
# IMPORTANT: Update the following paths before running:
#   - MODEL_DIR: Path to your OpenVLA-OFT model checkpoint
#   - OPENVLA_ROOT: Path to your OpenVLA-OFT repository
#   - CONDA_ENV: Your conda environment name
#
# IMPORTANT: This evaluation uses ONLY baseline actions for execution.
#            CF effects are computed and logged but DO NOT modify executed actions.

# Configuration - UPDATE THESE PATHS
MODEL_DIR="/path/to/openvla-7b-oft-finetuned-libero-10"
OPENVLA_ROOT="/path/to/openvla-oft"
CONDA_ENV="openvla"

# Set GPU
export CUDA_VISIBLE_DEVICES=0

# Run evaluation
cd $OPENVLA_ROOT

conda run -n $CONDA_ENV python experiments/robot/libero/run_libero_eval_cf.py \
    --model_family openvla \
    --pretrained_checkpoint $MODEL_DIR \
    --task_suite_name libero_10 \
    --use_l1_regression True \
    --use_proprio True \
    --num_images_in_input 2 \
    --center_crop True \
    --num_open_loop_steps 8 \
    --num_trials_per_task 50 \
    --enable_cf_eval True \
    --lora_rank 32 \
    --seed 7 \
    --run_id_note "baseline_cf_eval"