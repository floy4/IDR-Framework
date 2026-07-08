#!/bin/bash
# Start X-VLA Server for IDR Evaluation
#
# Usage:
#   ./start_server.sh
#   GPU_ID=1 ./start_server.sh
#
# Environment variables:
#   MODEL_PATH: Path to X-VLA model checkpoint (e.g., /path/to/X-VLA-Libero)
#   GPU_ID: GPU device ID
#   HOST: Server host (default: 127.0.0.1)
#   PORT: Server port (default: 9998)

set -e

MODEL_PATH="${MODEL_PATH:-/path/to/X-VLA-Libero}"
GPU_ID="${GPU_ID:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9998}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Starting X-VLA Server for IDR Evaluation"
echo "=========================================="
echo "Model:  $MODEL_PATH"
echo "GPU ID: $GPU_ID"
echo "Host:   $HOST:$PORT"
echo "=========================================="

cd "$REPO_DIR"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m deploy \
    --model_path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --output_dir "logs_server"
