#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Stage 1 worker (Surface D): embedding analysis for ONE (variant, window).
# Linear probe + 3-layer NN on embeddings from the PRE-TRAINED base model
# (not the finetuned checkpoint) — independent of the winning seed. Calls the
# UNCHANGED embedding_analysis_gena_lm.py, which already accepts val.csv|dev.csv.
#
# Required env:
#   REPO_ROOT, OUTPUT_DIR, LAMBDA_DIR, WINDOW, VARIANT, EMB_MODEL_PATH
# Optional env:
#   MAX_LENGTH (auto), EMB_BATCH_SIZE (16), POOLING (mean), NN_EPOCHS (100),
#   NN_HIDDEN_DIM (256), NN_LR (0.001), EMB_SEED (42),
#   INCLUDE_RANDOM_BASELINE (true|false)

set -euo pipefail

echo "=== embedding ${VARIANT} ${WINDOW} ==="
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

if [ -z "${MAX_LENGTH:-}" ]; then
    case "${WINDOW}" in
        2k) MAX_LENGTH=512  ;;
        4k) MAX_LENGTH=1024 ;;
        8k) MAX_LENGTH=2048 ;;
        *)  echo "ERROR: unknown WINDOW=${WINDOW}"; exit 1 ;;
    esac
fi

BATCH_SIZE=${EMB_BATCH_SIZE:-16}
POOLING=${POOLING:-mean}
NN_EPOCHS=${NN_EPOCHS:-100}
NN_HIDDEN_DIM=${NN_HIDDEN_DIM:-256}
NN_LR=${NN_LR:-0.001}
EMB_SEED=${EMB_SEED:-42}
INCLUDE_RANDOM_BASELINE=${INCLUDE_RANDOM_BASELINE:-true}

# embedding_analysis_gena_lm.py writes directly into --output_dir (it does NOT
# append the model name), so point it straight at <W>/embedding/<variant>.
OUTPUT_DIR_JOB="${OUTPUT_DIR}/${WINDOW}/embedding/${VARIANT}"
mkdir -p "${OUTPUT_DIR_JOB}"

RANDOM_BASELINE_FLAG=""
if [ "${INCLUDE_RANDOM_BASELINE,,}" = "true" ]; then
    RANDOM_BASELINE_FLAG="--include_random_baseline"
fi

echo "  base model:  ${EMB_MODEL_PATH}"
echo "  lambda dir:  ${LAMBDA_DIR}"
echo "  output:      ${OUTPUT_DIR_JOB}"
echo "  pooling=${POOLING} max_length=${MAX_LENGTH} nn_epochs=${NN_EPOCHS} nn_lr=${NN_LR}"
echo "  include_random_baseline=${INCLUDE_RANDOM_BASELINE}"

python embedding_analysis_gena_lm.py \
    --csv_dir "${LAMBDA_DIR}" \
    --model_path "${EMB_MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR_JOB}" \
    --batch_size ${BATCH_SIZE} \
    --max_length ${MAX_LENGTH} \
    --pooling "${POOLING}" \
    --seed ${EMB_SEED} \
    --nn_epochs ${NN_EPOCHS} \
    --nn_hidden_dim ${NN_HIDDEN_DIM} \
    --nn_lr ${NN_LR} \
    ${RANDOM_BASELINE_FLAG}

echo "Done: $(date)  -> ${OUTPUT_DIR_JOB}/embedding_analysis_results.json"
