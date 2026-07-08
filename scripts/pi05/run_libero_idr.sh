#!/bin/bash
# IDR Evaluation Script for π₀.₅ on LIBERO Benchmark
#
# Usage:
#   ./run_libero_idr.sh --cf_mode E --task_suites libero_spatial
#
# Environment variables:
#   CHECKPOINT_DIR: Model checkpoint directory (default: ~/.cache/openpi/checkpoints/pi05_libero)
#   CF_MODE: CF mode (BASE or E)
#   TASK_SUITES: Comma-separated task suites
#   NUM_TRIALS: Number of trials per task
#   GPU_ID: GPU device ID

set -e

# Default paths (use environment variables to override)
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/.cache/openpi/checkpoints/pi05_libero}"
CF_MODE="${CF_MODE:-E}"
TASK_SUITES="${TASK_SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"
NUM_TRIALS="${NUM_TRIALS:-50}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-data/cf_attn_libero}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"

echo "=========================================="
echo "IDR Evaluation on LIBERO (π₀.₅)"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "CF Mode:    $CF_MODE"
echo "Suites:     $TASK_SUITES"
echo "Trials:     $NUM_TRIALS"
echo "GPU ID:     $GPU_ID"
echo "=========================================="

python scripts/pi05/eval_cf_attn_libero.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --cf_mode "$CF_MODE" \
    --task_suite_names "$TASK_SUITES" \
    --num_trials_per_task "$NUM_TRIALS" \
    --gpu_id "$GPU_ID" \
    --video_out_path "$OUTPUT_DIR"
