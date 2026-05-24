#!/bin/bash
#
# Submit GENA-LM / ModernGENA fine-tuning for the three LAMBDA windows
# (2k, 4k, 8k) as separate SLURM jobs. Picks the right MODEL_NAME and
# hyperparameter preset based on the chosen VARIANT, and passes them to
# run_train_gena_lm.sh via env vars (run_train_gena_lm.sh reads each
# hyperparameter as ${VAR:-default}).
#
# Usage:
#   bash submit_train_all_windows.sh                       # bigbird, seed=42, all 3 windows
#   bash submit_train_all_windows.sh 7                     # bigbird, seed=7
#   bash submit_train_all_windows.sh 42 "2k 4k"            # bigbird, only 2k + 4k
#   bash submit_train_all_windows.sh 42 "2k 4k 8k" moderngena
#
# Variant presets:
#
#   bigbird (default) — GENA-LM BigBird HF base
#     MODEL_NAME=AIRI-Institute/gena-lm-bigbird-base-t2t
#     Recipe: downstream_tasks/promoter_prediction/finetune_promoter_16000.sh
#     LR=1e-4, WD=0, constant_with_warmup, optimize_metric=f1
#
#   moderngena — ModernGENA base
#     MODEL_NAME=AIRI-Institute/moderngena-base
#     Recipe: examples/modernGENA/sequence_classification/configs/config.yaml
#     LR=3e-5, WD=1e-3, linear, optimize_metric=pr_auc
#     (we substitute eval_f1 since HF Trainer's default compute_metrics
#     doesn't compute pr_auc without extra wiring — both are logged.)
#
# To run BOTH variants in parallel, just submit twice:
#   bash submit_train_all_windows.sh 42 "2k 4k 8k" bigbird
#   bash submit_train_all_windows.sh 42 "2k 4k 8k" moderngena

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SBATCH_SCRIPT="${SCRIPT_DIR}/run_train_gena_lm.sh"

if [ ! -f "${SBATCH_SCRIPT}" ]; then
    echo "ERROR: Training script not found: ${SBATCH_SCRIPT}"
    exit 1
fi

SEED="${1:-42}"
WINDOWS="${2:-2k 4k 8k}"
VARIANT="${3:-bigbird}"

# ─── Variant → hyperparameter preset ────────────────────────────────────────
case "${VARIANT}" in
    bigbird)
        MODEL_NAME="AIRI-Institute/gena-lm-bigbird-base-t2t"
        LEARNING_RATE="1e-4"
        WEIGHT_DECAY="0.0"
        LR_SCHEDULER_TYPE="constant_with_warmup"
        METRIC_FOR_BEST_MODEL="eval_f1"
        VARIANT_TAG="bigbird"
        ;;
    moderngena)
        MODEL_NAME="AIRI-Institute/moderngena-base"
        LEARNING_RATE="3e-5"
        WEIGHT_DECAY="1e-3"
        LR_SCHEDULER_TYPE="linear"
        METRIC_FOR_BEST_MODEL="eval_f1"
        VARIANT_TAG="moderngena"
        ;;
    *)
        echo "ERROR: invalid VARIANT='${VARIANT}' (use bigbird|moderngena)"
        exit 1
        ;;
esac

echo "=========================================="
echo "GENA-LM fine-tuning — multi-window submit"
echo "=========================================="
echo "Variant:                ${VARIANT}"
echo "Model:                  ${MODEL_NAME}"
echo "Seed:                   ${SEED}"
echo "Windows:                ${WINDOWS}"
echo "Learning rate:          ${LEARNING_RATE}"
echo "Weight decay:           ${WEIGHT_DECAY}"
echo "LR scheduler:           ${LR_SCHEDULER_TYPE}"
echo "Best-model metric:      ${METRIC_FOR_BEST_MODEL}"
echo "Script:                 ${SBATCH_SCRIPT}"
echo ""

for window in ${WINDOWS}; do
    case "${window}" in
        2k|4k|8k) ;;
        *) echo "[${window}] SKIP — invalid window (use 2k|4k|8k)"; continue ;;
    esac

    echo "[${window}] sbatch run_train_gena_lm.sh ${SEED} ${window} (variant=${VARIANT})"
    sbatch \
        --job-name="gena_lm_${VARIANT_TAG}_${window}_s${SEED}" \
        --export=ALL,\
MODEL_NAME="${MODEL_NAME}",\
LEARNING_RATE="${LEARNING_RATE}",\
WEIGHT_DECAY="${WEIGHT_DECAY}",\
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE}",\
METRIC_FOR_BEST_MODEL="${METRIC_FOR_BEST_MODEL}" \
        "${SBATCH_SCRIPT}" "${SEED}" "${window}"
done

echo ""
echo "Monitor:    squeue -u \$USER"
echo "Outputs:    /data/lindseylm/GLM_EVALUATIONS/MODELS/GENA-LM/GENA_LM_generic_sequence_classification/output/filtered/<window>/"
echo ""
echo "When ALL jobs for this variant finish, fill in CHECKPOINT_{2K,4K,8K} in"
echo "  ${SCRIPT_DIR}/submit_lambda_full_eval.sh"
echo "(set MODEL_NAME to e.g. gena_lm_${VARIANT_TAG} so the output dirs don't collide"
echo " across variants) and run it to evaluate against the LAMBDA test sets."
