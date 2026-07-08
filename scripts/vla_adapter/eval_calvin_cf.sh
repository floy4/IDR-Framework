#!/bin/bash
#
# eval_calvin_cf.sh
#
# Evaluate VLA on CALVIN benchmark with Counterfactual Inference.
# Supports zeroing visual and proprioceptive features separately with reweighting modes A-F.
#
# Usage:
#   bash vla-scripts/eval_calvin_cf.sh                     # 默认运行 CF (visual + proprio)
#   bash vla-scripts/eval_calvin_cf.sh --no-cf             # 关闭 CF，运行 baseline
#   bash vla-scripts/eval_calvin_cf.sh --no-proprio-cf     # 只做 visual CF，不做 proprio CF
#   bash vla-scripts/eval_calvin_cf.sh --cf-mode B         # 使用 mode B
#   bash vla-scripts/eval_calvin_cf.sh --gpu 0             # 使用 GPU 0
#   bash vla-scripts/eval_calvin_cf.sh --no-cf --gpu 2     # 组合使用

set -e

# =============================================
# Default values
# IMPORTANT: Update these paths before running
# =============================================
GPU=0
MODEL_PATH="/path/to/VLA-Adapter/CALVIN-ABC"
LOG_DIR="./experiments/logs-calvin-cf"
CALVIN_ROOT="/path/to/calvin"
CALVIN_DATASET="/path/to/calvin/task_ABC_D"

USE_CF=true
CF_METHOD="input_zeroing"
CF_MODE="E"
CF_GUIDANCE_SCALE=0.1
VLM_EFFECT_THRESHOLD=0.5
CFG_SCALE=1.0

USE_PROPRIO_CF=true
CF_METHOD_PROPRIO="input_zeroing"
CFG_SCALE_PROPRIO=1.0

CF_VERBOSE=false

# =============================================
# Parse command-line arguments
# =============================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cf)
            USE_CF=false
            shift
            ;;
        --no-proprio-cf)
            USE_PROPRIO_CF=false
            shift
            ;;
        --proprio-cf)
            USE_PROPRIO_CF=true
            shift
            ;;
        --cf-method)
            CF_METHOD="$2"
            shift 2
            ;;
        --cf-mode)
            CF_MODE="$2"
            shift 2
            ;;
        --cf-scale)
            CF_GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --vlm-threshold)
            VLM_EFFECT_THRESHOLD="$2"
            shift 2
            ;;
        --cfg-scale)
            CFG_SCALE="$2"
            shift 2
            ;;
        --proprio-cf-method)
            CF_METHOD_PROPRIO="$2"
            shift 2
            ;;
        --proprio-cfg-scale)
            CFG_SCALE_PROPRIO="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --calvin-root)
            CALVIN_ROOT="$2"
            shift 2
            ;;
        --calvin-dataset)
            CALVIN_DATASET="$2"
            shift 2
            ;;
        --verbose)
            CF_VERBOSE=true
            shift
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash eval_calvin_cf.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-cf               Disable CF (run baseline)"
            echo "  --no-proprio-cf       Disable proprio CF (visual CF only)"
            echo "  --proprio-cf          Enable proprio CF (default: on)"
            echo "  --cf-method METHOD    CF method: input_zeroing, attention_mask, output_reweighting (default: input_zeroing)"
            echo "  --cf-mode MODE        CF mode: BASE, A, B, C, D, E, F (default: E)"
            echo "  --cf-scale FLOAT      CF guidance scale (default: 0.1)"
            echo "  --vlm-threshold FLOAT VLM effect upper threshold (default: 0.5)"
            echo "  --cfg-scale FLOAT     CFG scale for output_reweighting (default: 1.0)"
            echo "  --proprio-cf-method M Proprio CF method (default: input_zeroing)"
            echo "  --proprio-cfg-scale F Proprio CFG scale (default: 1.0)"
            echo "  --gpu ID              GPU device ID (default: 1)"
            echo "  --model PATH          Model checkpoint path"
            echo "  --log-dir DIR         Log directory"
            echo "  --calvin-root PATH    CALVIN repo path (default: /path/to/calvin)"
            echo "  --calvin-dataset PATH CALVIN dataset path (default: /path/to/calvin/task_ABC_D)"
            echo "  --verbose             Enable CF verbose output"
            echo "  --seed N              Random seed (default: 7)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

export CUDA_VISIBLE_DEVICES=$GPU

mkdir -p "$LOG_DIR"

echo "============================================"
echo "CALVIN Counterfactual Evaluation"
echo "============================================"
echo "GPU: $GPU"
echo "Model: $MODEL_PATH"
echo "Use CF: $USE_CF"
echo "CF Method: $CF_METHOD"
echo "CF Mode: $CF_MODE"
echo "CF Guidance Scale: $CF_GUIDANCE_SCALE"
echo "VLM Effect Threshold: $VLM_EFFECT_THRESHOLD"
echo "Use Proprio CF: $USE_PROPRIO_CF"
if [ "$USE_PROPRIO_CF" = true ]; then
    echo "Proprio CF Method: $CF_METHOD_PROPRIO"
fi
echo "CALVIN Root: $CALVIN_ROOT"
echo "CALVIN Dataset: $CALVIN_DATASET"
echo "============================================"

cd "$(dirname "$0")/.."

python vla-scripts/evaluate_calvin_cf.py \
    --model_family openvla \
    --pretrained_checkpoint "$MODEL_PATH" \
    --use_cf "$USE_CF" \
    --cf_method "$CF_METHOD" \
    --cf_mode "$CF_MODE" \
    --cf_guidance_scale "$CF_GUIDANCE_SCALE" \
    --vlm_effect_threshold "$VLM_EFFECT_THRESHOLD" \
    --cfg_scale "$CFG_SCALE" \
    --use_proprio_cf "$USE_PROPRIO_CF" \
    --cf_method_proprio "$CF_METHOD_PROPRIO" \
    --cfg_scale_proprio "$CFG_SCALE_PROPRIO" \
    --cf_verbose "$CF_VERBOSE" \
    --calvin_root "$CALVIN_ROOT" \
    --calvin_dataset "$CALVIN_DATASET" \
    --local_log_dir "$LOG_DIR" \
    --use_l1_regression true \
    --use_proprio true \
    --num_images_in_input 2 \
    --seed "${SEED:-7}"
