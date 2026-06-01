#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Stage 2 worker: run the winning seed for VARIANT on ONE CSV (Surface A test
# or a Surface B diagnostic: fpr / gc_control / fnr). Reads winners.json to find
# the checkpoint, then calls the UNCHANGED inference_gena_lm.py.
#
# Required env:
#   REPO_ROOT, REPL_OUTPUT_DIR, WINDOW, VARIANT, INPUT_CSV, OUTPUT_FILENAME
# Optional env:
#   MAX_LENGTH (auto from WINDOW), INF_BATCH_SIZE (16), THRESHOLD (0.5),
#   PRECISION (bf16|fp16|fp32)

set -euo pipefail

echo "=== inference ${VARIANT} ${WINDOW}  input=${INPUT_CSV}  out=${OUTPUT_FILENAME} ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

module load conda 2>/dev/null || true
module load CUDA/12.8 2>/dev/null || true
conda activate gena_lm 2>/dev/null || source activate gena_lm 2>/dev/null || true
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-/data/lindseylm/.cache/huggingface}"

if [ -z "${CUDA_HOME:-}" ]; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null))) 2>/dev/null || true
fi

if [ -z "${REPO_ROOT:-}" ]; then
    echo "ERROR: REPO_ROOT is not set; the launcher must pass it via --export"; exit 1
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
SCRIPT_DIR_LR="${REPO_ROOT}/slurm_scripts/lambda_replication"

if [ -z "${MAX_LENGTH:-}" ]; then
    case "${WINDOW}" in
        2k) MAX_LENGTH=512  ;;
        4k) MAX_LENGTH=1024 ;;
        8k) MAX_LENGTH=2048 ;;
        *)  echo "ERROR: unknown WINDOW=${WINDOW}"; exit 1 ;;
    esac
fi
INF_BATCH_SIZE=${INF_BATCH_SIZE:-16}
THRESHOLD=${THRESHOLD:-0.5}
PRECISION=${PRECISION:-bf16}

PRECISION_FLAG=""
case "${PRECISION}" in
    bf16) PRECISION_FLAG="--bf16" ;;
    fp16) PRECISION_FLAG="--fp16" ;;
    fp32) PRECISION_FLAG="" ;;
esac

WINNERS_JSON="${REPL_OUTPUT_DIR}/winners.json"
[ -f "${WINNERS_JSON}" ] || { echo "ERROR: ${WINNERS_JSON} not found (run stage 2 selection first)"; exit 1; }

# Extract winner checkpoint path for this variant.
eval "$(python "${SCRIPT_DIR_LR}/print_winner_exports.py" "${WINNERS_JSON}" "${VARIANT}")"
echo "  winner seed:   ${WINNER_SEED}"
echo "  winner path:   ${WINNER_PATH}"

OUTPUT_DIR_JOB="${REPL_OUTPUT_DIR}/inference/${VARIANT}"
mkdir -p "${OUTPUT_DIR_JOB}"
OUTPUT_CSV="${OUTPUT_DIR_JOB}/${OUTPUT_FILENAME}"

python inference_gena_lm.py \
    --input_csv "${INPUT_CSV}" \
    --model_path "${WINNER_PATH}" \
    --output_csv "${OUTPUT_CSV}" \
    --batch_size ${INF_BATCH_SIZE} \
    --max_length ${MAX_LENGTH} \
    --threshold ${THRESHOLD} \
    --save_metrics \
    ${PRECISION_FLAG}

echo "Done: $(date)  -> ${OUTPUT_CSV}"
