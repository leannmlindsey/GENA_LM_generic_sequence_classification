#!/bin/bash
#
# GENA-LM LAMBDA_v1 replication — STAGE 1: fire off all training jobs.
#
# For each window in WINDOWS, submits:
#   - one finetune sbatch job per (variant, seed)          -> finetune_gena_lm_phage.py
#   - one embedding-analysis sbatch job per variant        -> embedding_analysis_gena_lm.py
# All jobs run in parallel (no --dependency chaining). Once they all complete,
# run run_lambda_inference.sh to pick the best seed and run inference.
#
# Mirrors the ProkBERT lambda_replication pipeline. Hyperparameters and the
# per-variant presets are copied verbatim from submit_train_all_windows.sh /
# submit_embedding_analysis_all.sh — the only change is the dataset (LAMBDA_v1).
#
# Usage:
#   1. Edit slurm_scripts/lambda_replication/lambda_replication.conf
#   2. bash slurm_scripts/lambda_replication/run_lambda_training.sh
#   3. Wait for jobs:  squeue -u $USER
#   4. bash slurm_scripts/lambda_replication/run_lambda_inference.sh

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="${SCRIPT_DIR}/../.."
# Resolve REPO_ROOT to an absolute path (jobs cd here; must not be relative).
REPO_ROOT="$( cd "${REPO_ROOT}" && pwd )"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: missing ${CONFIG}"; exit 1
fi
# shellcheck disable=SC1090
source "${CONFIG}"

# ─── variant presets (copied verbatim from submit_train_all_windows.sh) ──────
# Sets MODEL_NAME, LEARNING_RATE, WEIGHT_DECAY, LR_SCHEDULER_TYPE (finetune) and
# EMB_MODEL_PATH (embedding) for the given variant.
variant_preset () {
    case "$1" in
        bigbird)
            MODEL_NAME="AIRI-Institute/gena-lm-bigbird-base-t2t"
            LEARNING_RATE="1e-4"; WEIGHT_DECAY="0.0"
            LR_SCHEDULER_TYPE="constant_with_warmup"
            EMB_MODEL_PATH="AIRI-Institute/gena-lm-bigbird-base-t2t"
            ;;
        moderngena)
            MODEL_NAME="AIRI-Institute/moderngena-base"
            LEARNING_RATE="3e-5"; WEIGHT_DECAY="1e-3"
            LR_SCHEDULER_TYPE="linear"
            EMB_MODEL_PATH="AIRI-Institute/moderngena-base"
            ;;
        bertbase)
            MODEL_NAME="AIRI-Institute/gena-lm-bert-base-t2t"
            LEARNING_RATE="1e-4"; WEIGHT_DECAY="0.0"
            LR_SCHEDULER_TYPE="constant_with_warmup"
            EMB_MODEL_PATH="AIRI-Institute/gena-lm-bert-base-t2t"
            ;;
        *)
            echo "ERROR: unknown variant '$1' (use bigbird|moderngena|bertbase)"; exit 1
            ;;
    esac
}

# ─── validate ────────────────────────────────────────────────────────────────

if [[ "${LAMBDA_BASE}" == /path/to/* ]] || [[ "${OUTPUT_DIR}" == /path/to/* ]]; then
    echo "ERROR: edit ${CONFIG} — LAMBDA_BASE or OUTPUT_DIR still a placeholder"; exit 1
fi
[ -d "${LAMBDA_BASE}/train_val_test" ] || {
    echo "ERROR: ${LAMBDA_BASE}/train_val_test not found (expected LAMBDA_v1 layout)"; exit 1
}
[ -n "${WINDOWS}" ]  || { echo "ERROR: WINDOWS is empty";  exit 1; }
[ -n "${VARIANTS}" ] || { echo "ERROR: VARIANTS is empty"; exit 1; }
[ -n "${SEEDS}" ]    || { echo "ERROR: SEEDS is empty";    exit 1; }

# Validate per-window inputs exist before submitting anything.
for W in ${WINDOWS}; do
    LDIR="${LAMBDA_BASE}/train_val_test/${W}"
    [ -d "${LDIR}" ] || { echo "ERROR: ${LDIR} not found"; exit 1; }
    for f in train.csv test.csv; do
        [ -f "${LDIR}/${f}" ] || { echo "ERROR: ${LDIR}/${f} not found"; exit 1; }
    done
    if [ ! -f "${LDIR}/val.csv" ] && [ ! -f "${LDIR}/dev.csv" ]; then
        echo "ERROR: ${LDIR} must contain val.csv or dev.csv"; exit 1
    fi
done
for V in ${VARIANTS}; do variant_preset "${V}"; done   # fail fast on bad variant

mkdir -p "${OUTPUT_DIR}/logs"
LOGDIR="${OUTPUT_DIR}/logs"

# ─── summary ─────────────────────────────────────────────────────────────────

echo "============================================================"
echo "GENA-LM LAMBDA_v1 replication — Stage 1: training + embedding"
echo "============================================================"
echo "  REPO_ROOT:    ${REPO_ROOT}"
echo "  LAMBDA_BASE:  ${LAMBDA_BASE}"
echo "  OUTPUT_DIR:   ${OUTPUT_DIR}"
echo "  WINDOWS:      ${WINDOWS}"
echo "  VARIANTS:     ${VARIANTS}"
echo "  SEEDS:        ${SEEDS}"
echo "  FT:  epochs=${EPOCHS} batch=${BATCH_SIZE} grad_accum=${GRADIENT_ACCUMULATION_STEPS} precision=${PRECISION}"
echo "  EMB: pooling=${POOLING} nn_epochs=${NN_EPOCHS} nn_lr=${NN_LR} random_baseline=${INCLUDE_RANDOM_BASELINE}"
echo "============================================================"

# ─── common sbatch flags ─────────────────────────────────────────────────────
FT_FLAGS=(--partition=gpu --gres=gpu:a100:1 --mem="${FT_MEM}" --time="${FT_TIME}" --cpus-per-task=8)
EMB_FLAGS=(--partition=gpu --gres=gpu:a100:1 --mem="${EMB_MEM}" --time="${EMB_TIME}" --cpus-per-task=8)

# Shared env passed to every job. REPO_ROOT is propagated explicitly because
# SLURM stages each job script to /var/spool/slurm/... where BASH_SOURCE[0]
# can't recover the original repo location.
COMMON_ENV="REPO_ROOT=${REPO_ROOT},HF_HOME=${HF_HOME}"
FT_ENV="${COMMON_ENV},EPOCHS=${EPOCHS},BATCH_SIZE=${BATCH_SIZE},GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS},WARMUP_RATIO=${WARMUP_RATIO},EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE},EVAL_STEPS=${EVAL_STEPS},SAVE_STEPS=${SAVE_STEPS},SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT},PRECISION=${PRECISION}"
EMB_ENV="${COMMON_ENV},EMB_BATCH_SIZE=${EMB_BATCH_SIZE},POOLING=${POOLING},NN_EPOCHS=${NN_EPOCHS},NN_HIDDEN_DIM=${NN_HIDDEN_DIM},NN_LR=${NN_LR},EMB_SEED=${EMB_SEED},INCLUDE_RANDOM_BASELINE=${INCLUDE_RANDOM_BASELINE}"

NUM_JOBS=0

for W in ${WINDOWS}; do
    LAMBDA_DIR="${LAMBDA_BASE}/train_val_test/${W}"
    echo ""
    echo "--- window: ${W} ---"
    echo "    lambda dir: ${LAMBDA_DIR}"

    for V in ${VARIANTS}; do
        variant_preset "${V}"

        # Finetune jobs (one per seed)
        for SEED in ${SEEDS}; do
            JOB="ft_${W}_${V}_s${SEED}"
            echo "    submitting ${JOB}..."
            sbatch \
                --job-name="${JOB}" \
                --output="${LOGDIR}/${JOB}_%j.out" \
                --error="${LOGDIR}/${JOB}_%j.err" \
                "${FT_FLAGS[@]}" \
                --export="ALL,${FT_ENV},OUTPUT_DIR=${OUTPUT_DIR},LAMBDA_DIR=${LAMBDA_DIR},WINDOW=${W},VARIANT=${V},SEED=${SEED},MODEL_NAME=${MODEL_NAME},LEARNING_RATE=${LEARNING_RATE},WEIGHT_DECAY=${WEIGHT_DECAY},LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE}" \
                "${SCRIPT_DIR}/lambda_finetune_job.sh"
            NUM_JOBS=$((NUM_JOBS + 1))
        done

        # Embedding-analysis job (per variant; pretrained base model)
        JOB="emb_${W}_${V}"
        echo "    submitting ${JOB}..."
        sbatch \
            --job-name="${JOB}" \
            --output="${LOGDIR}/${JOB}_%j.out" \
            --error="${LOGDIR}/${JOB}_%j.err" \
            "${EMB_FLAGS[@]}" \
            --export="ALL,${EMB_ENV},OUTPUT_DIR=${OUTPUT_DIR},LAMBDA_DIR=${LAMBDA_DIR},WINDOW=${W},VARIANT=${V},EMB_MODEL_PATH=${EMB_MODEL_PATH}" \
            "${SCRIPT_DIR}/lambda_embedding_job.sh"
        NUM_JOBS=$((NUM_JOBS + 1))
    done
done

echo ""
echo "Submitted ${NUM_JOBS} jobs. Monitor with: squeue -u \$USER"
echo "When all jobs are done, run:"
echo "  bash ${SCRIPT_DIR}/run_lambda_inference.sh"
