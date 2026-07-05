#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Stage 2 worker (Surface C): genome-wide inference for the winning seed of
# VARIANT over a DIRECTORY of segment CSVs. Calls the UNCHANGED
# inference_gena_lm_dir.py, then renames every output so it starts with
# `genome_wide_` — the harvest pipeline matches genome_wide_<asm>_*_predictions.csv.
#
# Required env:
#   REPO_ROOT, REPL_OUTPUT_DIR, WINDOW, VARIANT, GENOME_WIDE_DIR
# Optional env:
#   MAX_LENGTH (auto), INF_BATCH_SIZE (16), THRESHOLD (0.5), PRECISION (bf16|fp16|fp32)

set -euo pipefail

echo "=== genome-wide inference ${VARIANT} ${WINDOW}  dir=${GENOME_WIDE_DIR} ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

# Delta-AI: initialise conda from the user's miniconda base, then activate the
# env. (Biowulf used `module load conda; module load CUDA/12.8`.) Recent torch
# ships aarch64 CUDA wheels that bundle the runtime, so no CUDA module is needed.
source /u/llindsey1/miniconda3/etc/profile.d/conda.sh
conda activate gena_lm
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-/work/hdd/bfzj/llindsey1/hf_cache}"

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

eval "$(python "${SCRIPT_DIR_LR}/print_winner_exports.py" "${WINNERS_JSON}" "${VARIANT}")"
echo "  winner type:   ${WINNER_TYPE:-finetune}"
echo "  winner path:   ${WINNER_PATH}${WINNER_HEAD_PATH}"

OUTPUT_DIR_JOB="${REPL_OUTPUT_DIR}/inference/${VARIANT}"
mkdir -p "${OUTPUT_DIR_JOB}"

# Write into a staging subdir so genome-wide outputs don't collide with the
# single-file Surface A/B predictions in the same inference/<variant>/ dir.
GW_TMP="${OUTPUT_DIR_JOB}/_gw_tmp"
rm -rf "${GW_TMP}"; mkdir -p "${GW_TMP}"

if [ "${WINNER_TYPE:-finetune}" = "finetune" ]; then
    python inference_gena_lm_dir.py \
        --input_dir "${GENOME_WIDE_DIR}" \
        --output_dir "${GW_TMP}" \
        --model_path "${WINNER_PATH}" \
        --batch_size ${INF_BATCH_SIZE} \
        --max_length ${MAX_LENGTH} \
        --threshold ${THRESHOLD} \
        --pattern "*.csv" \
        --save_metrics \
        ${PRECISION_FLAG}
else
    # Probe winner: loop the genome CSVs, applying the saved probe per file. Output
    # <stem>_predictions.csv into GW_TMP; the relocate step below prepends the
    # genome_wide_ prefix the harvest globs (same as the fine-tuned dir path).
    SCALER_ARG=()
    [ -n "${WINNER_SCALER_PATH:-}" ] && SCALER_ARG=(--scaler_path "${WINNER_SCALER_PATH}")
    shopt -s nullglob
    for csv in "${GENOME_WIDE_DIR}"/*.csv; do
        stem="$(basename "${csv}" .csv)"
        python inference_embedding_head_gena_lm.py \
            --model_path "${BASE_MODEL}" \
            --head_type "${WINNER_TYPE}" \
            --head_path "${WINNER_HEAD_PATH}" \
            "${SCALER_ARG[@]}" \
            --input_csv "${csv}" \
            --output_csv "${GW_TMP}/${stem}_predictions.csv" \
            --batch_size ${INF_BATCH_SIZE} \
            --max_length ${MAX_LENGTH} \
            --threshold ${THRESHOLD} \
            --save_metrics
    done
    shopt -u nullglob
fi

# Move outputs up, guaranteeing a genome_wide_ prefix on predictions/metrics so
# the harvest pipeline matches genome_wide_<asm>_*_predictions.csv. summary.json
# is renamed to avoid clobbering Surface A/B summaries.
echo "--- relocating genome-wide outputs with genome_wide_ prefix ---"
shopt -s nullglob
for f in "${GW_TMP}"/*; do
    base="$(basename "${f}")"
    case "${base}" in
        summary.json) dest="genome_wide_summary.json" ;;
        genome_wide_*) dest="${base}" ;;          # already prefixed
        *) dest="genome_wide_${base}" ;;
    esac
    mv -f "${f}" "${OUTPUT_DIR_JOB}/${dest}"
done
shopt -u nullglob
rmdir "${GW_TMP}" 2>/dev/null || true

echo "Done: $(date)  -> ${OUTPUT_DIR_JOB}/genome_wide_*_predictions.csv"
