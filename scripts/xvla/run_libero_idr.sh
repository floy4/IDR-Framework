#!/bin/bash
# IDR Evaluation Script for X-VLA on LIBERO Benchmark
#
# Usage:
#   # Terminal 1: Start server
#   ./start_server.sh
#
#   # Terminal 2: Run evaluation
#   ./run_libero_idr.sh --weight_mode E
#
# Environment variables:
#   MODEL_PATH: Path to X-VLA model checkpoint (e.g., /path/to/X-VLA-Libero)
#   GPU_ID: GPU device ID
#   WEIGHT_MODE: CF mode (E or BASE)
#   TASK_SUITES: Task suites to evaluate

set -e

# Default paths (use environment variables to override)
MODEL_PATH="${MODEL_PATH:-/path/to/X-VLA-Libero}"
GPU_ID="${GPU_ID:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9998}"
WEIGHT_MODE="${WEIGHT_MODE:-E}"
TASK_SUITES="${TASK_SUITES:-libero_spatial}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-0.1}"
EFFECT_THRESHOLD="${EFFECT_THRESHOLD:-0.5}"
OUTPUT_DIR="${OUTPUT_DIR:-logs_idr_${WEIGHT_MODE}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
EVAL_DIR="$REPO_DIR/scripts/xvla"

cd "$EVAL_DIR"

echo "=========================================="
echo "IDR Evaluation on LIBERO (X-VLA)"
echo "=========================================="
echo "Model:       $MODEL_PATH"
echo "Weight Mode: $WEIGHT_MODE"
echo "Suites:     $TASK_SUITES"
echo "GPU ID:     $GPU_ID"
echo "Output:     $OUTPUT_DIR"
echo "=========================================="

python libero_client_cf.py \
    --server_ip "$HOST" \
    --server_port "$PORT" \
    --output_dir "$OUTPUT_DIR" \
    --weight_mode "$WEIGHT_MODE" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --effect_threshold "$EFFECT_THRESHOLD" \
    --task_suites "$TASK_SUITES" \
    --eval_time 10
