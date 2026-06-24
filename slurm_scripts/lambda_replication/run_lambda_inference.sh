#!/bin/bash
#
# GENA-LM LAMBDA_v1 replication — STAGE 2: pick the best seed per variant and
# submit all inference jobs.
#
# For each window in WINDOWS:
#   1. select_best_seed.py picks the per-variant winner by test-set eval_mcc and
#      writes <OUTPUT_DIR>/<W>/winners.json.
#   2. Submit one inference job per (variant, diagnostic). Diagnostics:
#        Surface A  test       train_val_test/<W>/test.csv          (always)
#        Surface B  fpr        FPR_<W>                              (optional)
#        Surface B  gc_control GC_<W>                               (optional)
#        Surface B  fnr        FNR_<W>                              (optional)
#        Surface C  genome     GENOME_WIDE_<W>  (a directory of CSVs; optional)
#      Missing optional paths are skipped with a warning so a partial config
#      still produces Surface A results.
#
# Re-running is safe: each job overwrites its own predictions CSV(s).
#
# Usage (after run_lambda_training.sh has FINISHED — verify with `squeue`):
#   bash slurm_scripts/lambda_replication/run_lambda_inference.sh
#
# For an in-progress / smoke run where not every seed finished, allow partial
# winner selection:
#   ALLOW_PARTIAL_TRAINING=true bash slurm_scripts/lambda_replication/run_lambda_inference.sh

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: missing ${CONFIG}"; exit 1
fi
# shellcheck disable=SC1090
source "${CONFIG}"

# ─── validate ────────────────────────────────────────────────────────────────
if [[ "${LAMBDA_BASE}" == /path/to/* ]] || [[ "${OUTPUT_DIR}" == /path/to/* ]]; then
    echo "ERROR: edit ${CONFIG} — LAMBDA_BASE or OUTPUT_DIR still a placeholder"; exit 1
fi
[ -n "${WINDOWS}" ]  || { echo "ERROR: WINDOWS is empty";  exit 1; }
[ -n "${VARIANTS}" ] || { echo "ERROR: VARIANTS is empty"; exit 1; }

mkdir -p "${OUTPUT_DIR}/logs"
LOGDIR="${OUTPUT_DIR}/logs"

# Sanity: every window must have a finetune dir before inference.
for W in ${WINDOWS}; do
    if [ ! -d "${OUTPUT_DIR}/${W}/finetune" ]; then
        echo "ERROR: ${OUTPUT_DIR}/${W}/finetune missing — run run_lambda_training.sh first"; exit 1
    fi
done

# Delta-AI (GH200): one GH200 per node on the ghx4 partition. Biowulf equivalent
# was `--partition=gpu --gres=gpu:a100:1`.
INF_FLAGS=(--account=bfzj-dtai-gh --partition=ghx4 --gpus-per-node=1 --mem="${INF_MEM}" --time="${INF_TIME}" --cpus-per-task=8)

echo "============================================================"
echo "GENA-LM LAMBDA_v1 replication — Stage 2: winners + inference"
echo "============================================================"
echo "  LAMBDA_BASE: ${LAMBDA_BASE}"
echo "  OUTPUT_DIR:  ${OUTPUT_DIR}"
echo "  WINDOWS:     ${WINDOWS}"
echo "  VARIANTS:    ${VARIANTS}"
echo "============================================================"

ALLOW_PARTIAL_FLAG=""
if [ "${ALLOW_PARTIAL_TRAINING:-false}" = "true" ]; then
    ALLOW_PARTIAL_FLAG="--allow-partial"
fi

COMMON_ENV="REPO_ROOT=${REPO_ROOT},HF_HOME=${HF_HOME}"
NUM_JOBS=0
cd "${REPO_ROOT}"

for W in ${WINDOWS}; do
    echo ""
    echo "--- window: ${W} ---"
    REPL_W_DIR="${OUTPUT_DIR}/${W}"

    # 1) select winners (login node; reads test_results.json only)
    echo "  selecting best seed per variant (by eval_mcc)..."
    python3 "${SCRIPT_DIR}/select_best_seed.py" \
        --output_dir "${REPL_W_DIR}" \
        --variants ${VARIANTS} \
        ${ALLOW_PARTIAL_FLAG}

    WINNERS_JSON="${REPL_W_DIR}/winners.json"
    HAVE_VARIANTS=$(python3 -c "import json;print(' '.join(json.load(open('${WINNERS_JSON}')).keys()))")

    # 2) assemble single-file diagnostics (name -> path); test is always present.
    declare -a DIAG_NAMES DIAG_PATHS
    DIAG_NAMES=(test)
    DIAG_PATHS=("${LAMBDA_BASE}/train_val_test/${W}/test.csv")
    for d in fpr gc_control fnr; do
        case "${d}" in
            fpr)        var="FPR_${W}" ;;
            gc_control) var="GC_${W}"  ;;
            fnr)        var="FNR_${W}" ;;
        esac
        path="${!var:-}"
        if [ -z "${path}" ]; then
            echo "  note: ${var} unset — skipping ${d} for ${W}"
            continue
        fi
        if [ ! -f "${path}" ]; then
            echo "  WARNING: ${var}=${path} not found — skipping ${d} for ${W}"
            continue
        fi
        DIAG_NAMES+=("${d}")
        DIAG_PATHS+=("${path}")
    done

    # 3) genome-wide directory (Surface C), optional.
    gw_var="GENOME_WIDE_${W}"
    GW_DIR="${!gw_var:-}"
    if [ -n "${GW_DIR}" ] && [ ! -d "${GW_DIR}" ]; then
        echo "  WARNING: ${gw_var}=${GW_DIR} is not a directory — skipping genome-wide for ${W}"
        GW_DIR=""
    fi

    # 4) submit jobs per variant that has a winner.
    for V in ${VARIANTS}; do
        if [[ " ${HAVE_VARIANTS} " != *" ${V} "* ]]; then
            echo "    skip ${V}: no winner (training incomplete?)"
            continue
        fi

        for i in "${!DIAG_NAMES[@]}"; do
            NAME="${DIAG_NAMES[$i]}"
            CSV="${DIAG_PATHS[$i]}"
            JOB="inf_${W}_${V}_${NAME}"
            echo "    submitting ${JOB}..."
            sbatch \
                --job-name="${JOB}" \
                --output="${LOGDIR}/${JOB}_%j.out" \
                --error="${LOGDIR}/${JOB}_%j.err" \
                "${INF_FLAGS[@]}" \
                --export="ALL,${COMMON_ENV},REPL_OUTPUT_DIR=${REPL_W_DIR},WINDOW=${W},VARIANT=${V},INPUT_CSV=${CSV},OUTPUT_FILENAME=${NAME}_predictions.csv,INF_BATCH_SIZE=${INF_BATCH_SIZE},THRESHOLD=${THRESHOLD},PRECISION=${PRECISION}" \
                "${SCRIPT_DIR}/lambda_inference_job.sh"
            NUM_JOBS=$((NUM_JOBS + 1))
        done

        if [ -n "${GW_DIR}" ]; then
            JOB="gwinf_${W}_${V}"
            echo "    submitting ${JOB} (genome-wide dir)..."
            sbatch \
                --job-name="${JOB}" \
                --output="${LOGDIR}/${JOB}_%j.out" \
                --error="${LOGDIR}/${JOB}_%j.err" \
                "${INF_FLAGS[@]}" \
                --export="ALL,${COMMON_ENV},REPL_OUTPUT_DIR=${REPL_W_DIR},WINDOW=${W},VARIANT=${V},GENOME_WIDE_DIR=${GW_DIR},INF_BATCH_SIZE=${INF_BATCH_SIZE},THRESHOLD=${THRESHOLD},PRECISION=${PRECISION}" \
                "${SCRIPT_DIR}/lambda_genome_inference_job.sh"
            NUM_JOBS=$((NUM_JOBS + 1))
        fi

        # PHROG (Surface B, 2k only): inference on the phrog-annotated phage set
        # for the central per-category PHROG table. Reuses lambda_inference_job.sh
        # (which passes phrog_category/phrog_db_category/label through via
        # df.copy()). Output is model+variant-prefixed so the two GENA-LM variants
        # don't collide: GENA_LM_<variant>_phage_annotated_segments_2k_predictions.csv.
        if [ "${W}" = "2k" ]; then
            PHROG_PATH="${PHROG_2k:-}"
            if [ -z "${PHROG_PATH}" ]; then
                echo "    note: PHROG_2k unset — skipping PHROG for ${V}"
            elif [ ! -f "${PHROG_PATH}" ]; then
                echo "    WARNING: PHROG_2k=${PHROG_PATH} not found — skipping PHROG for ${V}"
            else
                JOB="inf_2k_${V}_phrog"
                PHROG_OUT="GENA_LM_${V}_phage_annotated_segments_2k_predictions.csv"
                echo "    submitting ${JOB} (-> ${PHROG_OUT})..."
                sbatch \
                    --job-name="${JOB}" \
                    --output="${LOGDIR}/${JOB}_%j.out" \
                    --error="${LOGDIR}/${JOB}_%j.err" \
                    "${INF_FLAGS[@]}" \
                    --export="ALL,${COMMON_ENV},REPL_OUTPUT_DIR=${REPL_W_DIR},WINDOW=${W},VARIANT=${V},INPUT_CSV=${PHROG_PATH},OUTPUT_FILENAME=${PHROG_OUT},INF_BATCH_SIZE=${INF_BATCH_SIZE},THRESHOLD=${THRESHOLD},PRECISION=${PRECISION}" \
                    "${SCRIPT_DIR}/lambda_inference_job.sh"
                NUM_JOBS=$((NUM_JOBS + 1))
            fi
        fi
    done

    unset DIAG_NAMES DIAG_PATHS
done

echo ""
echo "Submitted ${NUM_JOBS} jobs. Monitor with: squeue -u \$USER"
echo "Results: ${OUTPUT_DIR}/<W>/inference/<variant>/"
