#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Stage 1 worker: finetune ONE (variant, window, seed). Submitted by
# run_lambda_training.sh; all paths/resources come via --export=ALL,<env>.
# Calls the UNCHANGED finetune_gena_lm_phage.py — the only adaptation is a
# small data-staging step that presents LAMBDA_v1's val.csv to the entry point
# as dev.csv (load_data() looks for train/dev/test), so no Python edit needed.
#
# Required env:
#   REPO_ROOT, OUTPUT_DIR, LAMBDA_DIR, WINDOW, VARIANT, SEED, MODEL_NAME,
#   LEARNING_RATE, WEIGHT_DECAY, LR_SCHEDULER_TYPE
# Optional env:
#   MAX_LENGTH (auto from WINDOW), EPOCHS, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
#   WARMUP_RATIO, EARLY_STOPPING_PATIENCE, EVAL_STEPS, SAVE_STEPS,
#   SAVE_TOTAL_LIMIT, PRECISION (bf16|fp16|fp32)

set -euo pipefail

echo "=== finetune ${VARIANT} ${WINDOW} seed=${SEED} ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

module load conda 2>/dev/null || true
module load CUDA/12.8 2>/dev/null || true
conda activate gena_lm 2>/dev/null || source activate gena_lm 2>/dev/null || true
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

# Offline HF: avoid 503s on the Biowulf HTTPS proxy. Cache must be pre-warmed
# from a login node (see lambda_replication.conf).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-/data/lindseylm/.cache/huggingface}"

if [ -z "${CUDA_HOME:-}" ]; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null))) 2>/dev/null || true
fi

# REPO_ROOT is supplied by the launcher via --export. SLURM stages this script
# to /var/spool/slurm/... so never compute REPO_ROOT from BASH_SOURCE[0].
if [ -z "${REPO_ROOT:-}" ]; then
    echo "ERROR: REPO_ROOT is not set; the launcher must pass it via --export"; exit 1
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# Per-window max_length (same mapping as run_train_gena_lm.sh).
if [ -z "${MAX_LENGTH:-}" ]; then
    case "${WINDOW}" in
        2k) MAX_LENGTH=512  ;;
        4k) MAX_LENGTH=1024 ;;
        8k) MAX_LENGTH=2048 ;;
        *)  echo "ERROR: unknown WINDOW=${WINDOW}"; exit 1 ;;
    esac
fi

EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-16}
WARMUP_RATIO=${WARMUP_RATIO:-0.06}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-7}
EVAL_STEPS=${EVAL_STEPS:-100}
SAVE_STEPS=${SAVE_STEPS:-100}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-1}
PRECISION=${PRECISION:-bf16}

PRECISION_FLAG=""
case "${PRECISION}" in
    bf16) PRECISION_FLAG="--bf16" ;;
    fp16) PRECISION_FLAG="--fp16" ;;
    fp32) PRECISION_FLAG="" ;;
    *)    echo "ERROR: invalid PRECISION='${PRECISION}' (bf16|fp16|fp32)"; exit 1 ;;
esac

OUTPUT_DIR_JOB="${OUTPUT_DIR}/${WINDOW}/finetune/${VARIANT}/seed-${SEED}"
mkdir -p "${OUTPUT_DIR_JOB}"

# ─── data staging: present val.csv (or dev.csv) as dev.csv ───────────────────
# finetune_gena_lm_phage.py:load_data() reads train/dev/test.csv. LAMBDA_v1
# ships val.csv. We build a tiny symlink dir so the UNCHANGED entry point sees
# train.csv, test.csv, dev.csv -> (val.csv|dev.csv). No file copies.
STAGE_DIR="${OUTPUT_DIR_JOB}/_staged_data"
mkdir -p "${STAGE_DIR}"
ln -sf "${LAMBDA_DIR}/train.csv" "${STAGE_DIR}/train.csv"
ln -sf "${LAMBDA_DIR}/test.csv"  "${STAGE_DIR}/test.csv"
if [ -f "${LAMBDA_DIR}/dev.csv" ]; then
    ln -sf "${LAMBDA_DIR}/dev.csv" "${STAGE_DIR}/dev.csv"
elif [ -f "${LAMBDA_DIR}/val.csv" ]; then
    ln -sf "${LAMBDA_DIR}/val.csv" "${STAGE_DIR}/dev.csv"
else
    echo "ERROR: no val.csv or dev.csv in ${LAMBDA_DIR}"; exit 1
fi

echo "  model:        ${MODEL_NAME}"
echo "  lambda dir:   ${LAMBDA_DIR}  (staged -> ${STAGE_DIR})"
echo "  output:       ${OUTPUT_DIR_JOB}"
echo "  lr=${LEARNING_RATE} wd=${WEIGHT_DECAY} sched=${LR_SCHEDULER_TYPE} max_length=${MAX_LENGTH}"
echo "  batch=${BATCH_SIZE} grad_accum=${GRADIENT_ACCUMULATION_STEPS} epochs=${EPOCHS} precision=${PRECISION}"

python finetune_gena_lm_phage.py \
    --model_name "${MODEL_NAME}" \
    --dataset_dir "${STAGE_DIR}" \
    --output_dir "${OUTPUT_DIR_JOB}" \
    --max_length ${MAX_LENGTH} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --num_train_epochs ${EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --logging_steps 20 \
    --eval_strategy steps \
    --eval_steps ${EVAL_STEPS} \
    --save_strategy steps \
    --save_steps ${SAVE_STEPS} \
    --load_best_model_at_end \
    --metric_for_best_model eval_f1 \
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --save_only_model \
    --seed ${SEED} \
    ${PRECISION_FLAG}

echo "Done: $(date)  -> ${OUTPUT_DIR_JOB}/test_results.json"
