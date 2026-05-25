#!/bin/bash
#
# Submit embedding-analysis (Linear Probe + 3-layer NN on extracted
# embeddings) for both GENA-LM variants × all 3 LAMBDA windows.
# Produces the Table 2 (binary LP/NN) numbers for the paper.
#
# Embeddings are extracted from the PRE-TRAINED base model (not the
# fine-tuned checkpoint) — this is the "embedding quality" evaluation
# that measures separability in the base model's representation space,
# independent of fine-tuning.
#
# Usage:
#   bash submit_embedding_analysis_all.sh                       # bigbird + moderngena, all 3 windows
#   bash submit_embedding_analysis_all.sh bigbird               # bigbird only
#   bash submit_embedding_analysis_all.sh moderngena "4k 8k"    # moderngena, only 4k and 8k

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SBATCH_SCRIPT="${SCRIPT_DIR}/run_embedding_analysis.sh"
RESULTS_ROOT="/data/lindseylm/GLM_EVALUATIONS/MODELS/FINAL_RESULTS"
DATASET_ROOT="/gpfs/gsfs12/users/Irp-jiang/share/lindseylm/lambda_final"

if [ ! -f "${SBATCH_SCRIPT}" ]; then
    echo "ERROR: SLURM script not found: ${SBATCH_SCRIPT}"
    exit 1
fi

# Single-variant submission helper
submit_for_variant () {
    local variant="$1"
    local windows="$2"

    case "${variant}" in
        bigbird)
            local model_path="AIRI-Institute/gena-lm-bigbird-base-t2t"
            local model_tag="gena_lm_bigbird"
            ;;
        moderngena)
            local model_path="AIRI-Institute/moderngena-base"
            local model_tag="gena_lm_moderngena"
            ;;
        *)
            echo "ERROR: invalid variant '${variant}' (use bigbird|moderngena)"
            return 1
            ;;
    esac

    echo "=========================================="
    echo "Embedding analysis — ${variant} (${model_path})"
    echo "Windows: ${windows}"
    echo "=========================================="

    for window in ${windows}; do
        case "${window}" in
            2k) max_length=512  ;;
            4k) max_length=1024 ;;
            8k) max_length=2048 ;;
            *)  echo "[${window}] SKIP — invalid window"; continue ;;
        esac

        csv_dir="${DATASET_ROOT}/merged_datasets_filtered/${window}"
        output_dir="${RESULTS_ROOT}/${model_tag}/embedding_analysis/${window}"

        if [ ! -d "${csv_dir}" ]; then
            echo "[${window}] SKIP — dataset dir missing: ${csv_dir}"
            continue
        fi

        echo "[${window}] Submitting (model: ${model_path}, max_length: ${max_length})"
        sbatch \
            --job-name="${model_tag}_emb_${window}" \
            --export=ALL,\
CSV_DIR="${csv_dir}",\
MODEL_PATH="${model_path}",\
OUTPUT_DIR="${output_dir}",\
MAX_LENGTH="${max_length}",\
INCLUDE_RANDOM_BASELINE="true",\
POOLING="mean",\
SEED="42" \
            "${SBATCH_SCRIPT}"
    done
    echo ""
}

# ─── Parse args ─────────────────────────────────────────────────────────────
VARIANT="${1:-both}"
WINDOWS="${2:-2k 4k 8k}"

case "${VARIANT}" in
    bigbird|moderngena)
        submit_for_variant "${VARIANT}" "${WINDOWS}"
        ;;
    both)
        submit_for_variant "bigbird"    "${WINDOWS}"
        submit_for_variant "moderngena" "${WINDOWS}"
        ;;
    *)
        echo "ERROR: invalid variant '${VARIANT}' (use bigbird|moderngena|both)"
        exit 1
        ;;
esac

echo "Monitor: squeue -u \$USER"
echo "Outputs land at: ${RESULTS_ROOT}/{gena_lm_bigbird,gena_lm_moderngena}/embedding_analysis/<W>/"
