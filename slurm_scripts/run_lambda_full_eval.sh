#!/bin/bash
#SBATCH --job-name=gena_lm_lambda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=lambda_eval_%x_%j.out
#SBATCH --error=lambda_eval_%x_%j.err

# Run all 3 LAMBDA evaluation categories (binary, error_bias, genome_wide) for
# a single window using a fine-tuned GENA-LM checkpoint. Writes predictions
# under the directory layout the LAMBDA aggregator (03_build_website_data.py)
# expects: ${RESULTS_ROOT}/${MODEL_NAME}/<category>/${WINDOW}/*_predictions.csv
#
# Submit via submit_lambda_full_eval.sh (which sets all the env vars below
# and submits one job per window). Do not edit configuration in this file —
# everything is read from env.
#
# Required env:
#   WINDOW         e.g. 2k | 4k | 8k
#   MODEL_PATH     fine-tuned checkpoint directory for this window
#   DATASET_ROOT   LAMBDA dataset root (contains binary_segments_*,
#                  error_and_bias_*, phoenix/segments/*)
#   RESULTS_ROOT   output root (one dir per model goes here)
#
# Optional env:
#   MODEL_NAME     output subdir name (default: gena_lm)
#   MAX_LENGTH     tokenizer truncation (default: 512 = BERT base ctx)
#   BATCH_SIZE     default: 16
#   THRESHOLD      default: 0.5
#   PRECISION      bf16 | fp16 | fp32 (default: bf16 — A100)

echo "============================================================"
echo "GENA-LM LAMBDA full evaluation — window=${WINDOW}"
echo "============================================================"
echo "Job started: $(date)"
echo "Node:        $(hostname)"
echo "Job ID:      ${SLURM_JOB_ID}"
echo ""

# ─── Required env validation ────────────────────────────────────────────────
for var in WINDOW MODEL_PATH DATASET_ROOT RESULTS_ROOT; do
    if [ -z "${!var}" ]; then
        echo "ERROR: ${var} is not set"
        exit 1
    fi
done

# ─── Defaults ───────────────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-gena_lm}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"
THRESHOLD="${THRESHOLD:-0.5}"
PRECISION="${PRECISION:-bf16}"

PRECISION_FLAG=""
case "${PRECISION}" in
    bf16) PRECISION_FLAG="--bf16" ;;
    fp16) PRECISION_FLAG="--fp16" ;;
    fp32) PRECISION_FLAG="" ;;
    *)    echo "ERROR: invalid PRECISION='${PRECISION}' (use bf16|fp16|fp32)"; exit 1 ;;
esac

# Genome-wide segment-stride convention from replication/baselines/run_baselines.py
case "${WINDOW}" in
    2k) GENOME_STRIDE_DIR="2k_1k" ;;
    4k) GENOME_STRIDE_DIR="4k_2k" ;;
    8k) GENOME_STRIDE_DIR="8k_4k" ;;
    *)  echo "ERROR: invalid WINDOW='${WINDOW}' (use 2k|4k|8k)"; exit 1 ;;
esac

# Load modules (suppress errors for non-Biowulf systems)
module load conda 2>/dev/null || true
module load CUDA/12.8 2>/dev/null || true

# Set CUDA_HOME if not set
if [ -z "${CUDA_HOME:-}" ]; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null))) 2>/dev/null || true
fi

# Activate conda environment (tries modern `conda activate` then falls back
# to legacy `source activate`)
conda activate gena_lm 2>/dev/null || source activate gena_lm 2>/dev/null || true

REPO_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/GENA-LM/GENA_LM_generic_sequence_classification"
cd "${REPO_DIR}" || { echo "ERROR: cannot cd to ${REPO_DIR}"; exit 1; }

echo "GPU:"
nvidia-smi | head -20
echo ""

echo "Configuration:"
echo "  Window:       ${WINDOW}"
echo "  Model name:   ${MODEL_NAME}"
echo "  Model path:   ${MODEL_PATH}"
echo "  Dataset root: ${DATASET_ROOT}"
echo "  Results root: ${RESULTS_ROOT}"
echo "  Max length:   ${MAX_LENGTH}"
echo "  Batch size:   ${BATCH_SIZE}"
echo "  Threshold:    ${THRESHOLD}"
echo "  Precision:    ${PRECISION}"
echo ""

# ─── Per-category configuration ─────────────────────────────────────────────
#
# CATEGORY  | INPUT_DIR                                          | PATTERN
# ----------+----------------------------------------------------+--------
# binary    | ${DATASET_ROOT}/binary_segments_${WINDOW}          | test.csv
# error_bias| ${DATASET_ROOT}/error_and_bias_${WINDOW}           | *.csv
# genome_wd | ${DATASET_ROOT}/phoenix/segments/${stride_dir}     | *.csv
#
# Output category names match what 03_build_website_data.py expects
# (binary | error_bias | genome_wide — note the underscore, not "and_bias").

run_category () {
    local category="$1"
    local input_dir="$2"
    local pattern="$3"

    local output_dir="${RESULTS_ROOT}/${MODEL_NAME}/${category}/${WINDOW}"

    echo ""
    echo "============================================================"
    echo "[${category}] window=${WINDOW}"
    echo "============================================================"
    echo "  Input dir:  ${input_dir}"
    echo "  Pattern:    ${pattern}"
    echo "  Output dir: ${output_dir}"

    if [ ! -d "${input_dir}" ]; then
        echo "  SKIP — input dir not found"
        return 1
    fi

    local n_files
    n_files=$(ls -1 ${input_dir}/${pattern} 2>/dev/null | wc -l)
    if [ "${n_files}" -eq 0 ]; then
        echo "  SKIP — pattern '${pattern}' matched 0 files in ${input_dir}"
        return 1
    fi
    echo "  Files to process: ${n_files}"
    if [ "${category}" = "error_bias" ]; then
        echo "  (listing for sanity — expect 4 files: test, gc_control, bacterial_cds, phage_segments)"
        ls -1 ${input_dir}/${pattern}
    fi

    mkdir -p "${output_dir}"

    python inference_gena_lm_dir.py \
        --input_dir "${input_dir}" \
        --output_dir "${output_dir}" \
        --model_path "${MODEL_PATH}" \
        --batch_size "${BATCH_SIZE}" \
        --max_length "${MAX_LENGTH}" \
        --threshold "${THRESHOLD}" \
        --pattern "${pattern}" \
        --save_metrics \
        ${PRECISION_FLAG}

    local rc=$?
    if [ ${rc} -ne 0 ]; then
        echo "  ERROR — inference_gena_lm_dir.py exited with ${rc}"
    fi
    return ${rc}
}

# ─── Run all 3 categories ───────────────────────────────────────────────────
OVERALL_RC=0

# Binary test lives in the same dir as the training data (the dir training
# reads from for train/dev/test). On this Biowulf setup that's
# merged_datasets_filtered/<W>/test.csv — not binary_segments_<W>/.
run_category "binary" \
    "${DATASET_ROOT}/merged_datasets_filtered/${WINDOW}" \
    "test.csv" \
    || OVERALL_RC=1

run_category "error_bias" \
    "${DATASET_ROOT}/error_and_bias_${WINDOW}" \
    "*.csv" \
    || OVERALL_RC=1

run_category "genome_wide" \
    "${DATASET_ROOT}/phoenix/segments/${GENOME_STRIDE_DIR}" \
    "*.csv" \
    || OVERALL_RC=1

echo ""
echo "============================================================"
echo "Job completed: $(date) — exit ${OVERALL_RC}"
echo "============================================================"
exit ${OVERALL_RC}
