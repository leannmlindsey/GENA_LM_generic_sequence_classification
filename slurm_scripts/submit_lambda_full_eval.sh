#!/bin/bash
#
# Submit full LAMBDA evaluation (binary + error_bias + genome_wide) for each
# window (2k / 4k / 8k) as separate SLURM jobs. Output layout matches what
# LAMBDA's 03_build_website_data.py expects.
#
# Usage:
#   1. Fill in the checkpoint paths + MAX_LENGTH for each window below.
#      Leave a checkpoint blank ("") to skip that window.
#   2. bash submit_lambda_full_eval.sh
#

#####################################################################
# CONFIGURATION
#####################################################################

# Output identity — used as ${RESULTS_ROOT}/${MODEL_NAME}/<category>/<window>/
MODEL_NAME="gena_lm"

# Roots
DATASET_ROOT="/gpfs/gsfs12/users/Irp-jiang/share/lindseylm/lambda_final"
RESULTS_ROOT="/data/lindseylm/GLM_EVALUATIONS/MODELS/FINAL_RESULTS"

# Per-window fine-tuned checkpoints (leave blank to skip a window)
CHECKPOINT_2K="/data/lindseylm/GLM_EVALUATIONS/MODELS/GENA-LM/GENA_LM_generic_sequence_classification/output/filtered/2k/gena_lm_lambda_filtered_2k_8_3e-5_20260120_063339/checkpoint-40995"
CHECKPOINT_4K=""
CHECKPOINT_8K=""

# Per-window tokenizer max_length — should match what you trained with.
# GENA-LM BERT base context is 512 tokens (~3 kb). If you fine-tuned on the
# 4k/8k splits with a BigBird variant, raise these accordingly (1024/2048).
MAX_LENGTH_2K="512"
MAX_LENGTH_4K="512"
MAX_LENGTH_8K="512"

# Inference hyperparameters (apply to every window)
BATCH_SIZE="16"
THRESHOLD="0.5"
PRECISION="bf16"        # bf16 | fp16 | fp32

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
echo "LAMBDA full evaluation — ${MODEL_NAME}"
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
        echo "[${window}] SKIP — no checkpoint configured"
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
        --job-name="gena_lm_lambda_${window}" \
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
echo "When ALL jobs finish, aggregate to the LAMBDA dashboard format:"
echo "  cd /data/lindseylm/GLM_EVALUATIONS/NAR_GENOMICS_LAMBDA_REPO/LAMBDA/replication"
echo "  python 03_build_website_data.py \\"
echo "      --predictions ${RESULTS_ROOT} \\"
echo "      --ground-truth \${DATASET_ROOT}/ground_truth.csv \\"
echo "      --taxonomy \${TAXONOMY_CSV} \\"
echo "      --output ${RESULTS_ROOT}/aggregated"
