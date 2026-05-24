#!/bin/bash
#
# Submit full LAMBDA evaluation (binary + error_bias + genome_wide) for each
# window (2k / 4k / 8k) as separate SLURM jobs. Output layout matches what
# LAMBDA's 03_build_website_data.py expects.
#
# Usage:
#   bash submit_lambda_full_eval.sh                 # default variant: bigbird
#   bash submit_lambda_full_eval.sh bigbird
#   bash submit_lambda_full_eval.sh moderngena
#
# Before running, fill in CHECKPOINT_{2K,4K,8K} for the chosen variant in the
# preset block below. Leave a checkpoint blank ("") to skip that window.

#####################################################################
# COMMON CONFIGURATION
#####################################################################

# Roots (shared across variants)
DATASET_ROOT="/gpfs/gsfs12/users/Irp-jiang/share/lindseylm/lambda_final"
RESULTS_ROOT="/data/lindseylm/GLM_EVALUATIONS/MODELS/FINAL_RESULTS"

# Per-window tokenizer max_length — should match what you trained with.
# Set by training recipe: 2k → 512 (≈3 kb), 4k → 1024 (≈6 kb), 8k → 2048 (≈12 kb).
MAX_LENGTH_2K="512"
MAX_LENGTH_4K="1024"
MAX_LENGTH_8K="2048"

# Inference hyperparameters (apply to every window of every variant)
BATCH_SIZE="16"
THRESHOLD="0.5"
PRECISION="bf16"        # bf16 | fp16 | fp32

#####################################################################
# VARIANT PRESETS — fill in the checkpoint paths for each variant after
# training completes. The MODEL_NAME determines the output subdir under
# RESULTS_ROOT, which becomes the model column in the aggregator and the
# grid-search results.
#####################################################################

VARIANT="${1:-bigbird}"

case "${VARIANT}" in
    bigbird)
        MODEL_NAME="gena_lm_bigbird"
        CHECKPOINT_2K=""
        CHECKPOINT_4K=""
        CHECKPOINT_8K=""
        ;;
    moderngena)
        MODEL_NAME="gena_lm_moderngena"
        CHECKPOINT_2K=""
        CHECKPOINT_4K=""
        CHECKPOINT_8K=""
        ;;
    *)
        echo "ERROR: invalid VARIANT='${VARIANT}' (use bigbird|moderngena)"
        echo "       To add other variants (e.g. bertbase), extend the case block here."
        exit 1
        ;;
esac

#####################################################################
# END CONFIGURATION
#####################################################################

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SBATCH_SCRIPT="${SCRIPT_DIR}/run_lambda_full_eval.sh"

if [ ! -f "${SBATCH_SCRIPT}" ]; then
    echo "ERROR: SLURM script not found: ${SBATCH_SCRIPT}"
    exit 1
fi

if [ ! -d "${DATASET_ROOT}" ]; then
    echo "ERROR: DATASET_ROOT does not exist: ${DATASET_ROOT}"
    exit 1
fi

mkdir -p "${RESULTS_ROOT}"

echo "=========================================="
echo "LAMBDA full evaluation — ${VARIANT} (${MODEL_NAME})"
echo "=========================================="
echo "Dataset root: ${DATASET_ROOT}"
echo "Results root: ${RESULTS_ROOT}"
echo "Batch size:   ${BATCH_SIZE}"
echo "Threshold:    ${THRESHOLD}"
echo "Precision:    ${PRECISION}"
echo ""

submit_window () {
    local window="$1"
    local checkpoint="$2"
    local max_length="$3"

    if [ -z "${checkpoint}" ]; then
        echo "[${window}] SKIP — no checkpoint configured for ${VARIANT}"
        return
    fi

    if [ ! -d "${checkpoint}" ]; then
        echo "[${window}] ERROR — checkpoint does not exist: ${checkpoint}"
        return
    fi

    if [ ! -f "${checkpoint}/config.json" ]; then
        echo "[${window}] ERROR — checkpoint missing config.json: ${checkpoint}"
        return
    fi

    echo "[${window}] Submitting (checkpoint: ${checkpoint}, max_length: ${max_length})"
    sbatch \
        --job-name="${MODEL_NAME}_lambda_${window}" \
        --export=ALL,\
WINDOW="${window}",\
MODEL_PATH="${checkpoint}",\
DATASET_ROOT="${DATASET_ROOT}",\
RESULTS_ROOT="${RESULTS_ROOT}",\
MODEL_NAME="${MODEL_NAME}",\
MAX_LENGTH="${max_length}",\
BATCH_SIZE="${BATCH_SIZE}",\
THRESHOLD="${THRESHOLD}",\
PRECISION="${PRECISION}" \
        "${SBATCH_SCRIPT}"
}

submit_window "2k" "${CHECKPOINT_2K}" "${MAX_LENGTH_2K}"
submit_window "4k" "${CHECKPOINT_4K}" "${MAX_LENGTH_4K}"
submit_window "8k" "${CHECKPOINT_8K}" "${MAX_LENGTH_8K}"

echo ""
echo "Monitor: squeue -u \$USER"
echo ""
echo "When all inference jobs finish for BOTH variants, the post-inference"
echo "workflow is:"
echo "  1. Symlink genome_wide predictions into per_segment layout:"
echo "       bash ${SCRIPT_DIR}/link_for_grid_search.sh"
echo "  2. Run grid search (matches what was done for DNABERT2/NTv2/etc.):"
echo "       bash ${SCRIPT_DIR}/run_grid_search.sh"
echo "  3. Apply best params and aggregate into LAMBDA's metrics_summary.csv."
