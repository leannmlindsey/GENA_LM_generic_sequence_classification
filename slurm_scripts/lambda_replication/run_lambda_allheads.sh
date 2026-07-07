#!/bin/bash
#
# GENA-LM — genome-wide predictions for BOTH frozen-embedding probe heads
# (linear probe + 3-layer NN) across all genome-wide CSVs, for bigbird and
# moderngena. Fills the missing LP + NN heads (FT genome-wide already exists) in
# ONE embedding pass per CSV.
#
# Reuses lambda_replication.conf + the variant_preset() from run_lambda_training.sh
# (variant -> base model). max_length per window matches lambda_embedding_job.sh
# (2k=512, 4k=1024, 8k=2048). Submits one lambda_allheads_job.sh per
# (window, variant, genome CSV).
#
# Usage (login node, repo pulled; the job self-activates gena_lm):
#   bash slurm_scripts/lambda_replication/run_lambda_allheads.sh [WINDOW ...]

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"
JOB="${SCRIPT_DIR}/lambda_allheads_job.sh"
[ -f "${CONFIG}" ] || { echo "ERROR: missing ${CONFIG}"; exit 1; }
[ -f "${JOB}" ]    || { echo "ERROR: missing ${JOB}"; exit 1; }
# shellcheck disable=SC1090
source "${CONFIG}"

# variant -> base (EMB_MODEL_PATH), copied verbatim from run_lambda_training.sh.
variant_preset () {
    case "$1" in
        bigbird)    EMB_MODEL_PATH="AIRI-Institute/gena-lm-bigbird-base-t2t" ;;
        moderngena) EMB_MODEL_PATH="AIRI-Institute/moderngena-base" ;;
        bertbase)   EMB_MODEL_PATH="AIRI-Institute/gena-lm-bert-base-t2t" ;;
        *) echo "ERROR: unknown variant '$1'"; exit 1 ;;
    esac
}
# window -> max_length, as in lambda_embedding_job.sh.
maxlen_for () {
    case "$1" in
        2k) echo 512 ;; 4k) echo 1024 ;; 8k) echo 2048 ;;
        *) echo "ERROR: unknown window $1" >&2; exit 1 ;;
    esac
}

WINS=("$@"); [ "${#WINS[@]}" -gt 0 ] || read -ra WINS <<< "${SEGMENT_LENGTHS:-2k 4k 8k}"
BATCH="${INF_BATCH_SIZE:-${EMB_BATCH_SIZE:-16}}"

mkdir -p "${OUTPUT_DIR}/logs"
LOGDIR="${OUTPUT_DIR}/logs"
FLAGS=(--account=bfzj-dtai-gh --partition=ghx4 --gpus-per-node=1 --mem="${INF_MEM}" --time="${INF_TIME}" --cpus-per-task=8)

echo "============================================================"
echo "GENA-LM — all-heads (LP + NN) genome-wide"
echo "  OUTPUT_DIR: ${OUTPUT_DIR}   WINDOWS: ${WINS[*]}   VARIANTS: ${VARIANTS}"
echo "============================================================"

NUM=0
for W in "${WINS[@]}"; do
    REPL_W_DIR="${OUTPUT_DIR}/${W}"
    MAX_LENGTH="$(maxlen_for "${W}")"
    gw_var="GENOME_WIDE_${W}"; GW_PATH="${!gw_var:-}"
    if [ -z "${GW_PATH}" ] || [ ! -d "${GW_PATH}" ]; then
        echo "WARNING: no genome-wide dir for ${W} (${GW_PATH:-unset}) — skipping"; continue
    fi
    for V in ${VARIANTS}; do
        variant_preset "${V}"
        EMB_DIR="${REPL_W_DIR}/embedding/${V}"
        if [ ! -f "${EMB_DIR}/linear_probe_pretrained.pkl" ]; then
            echo "WARNING: no saved LP probe in ${EMB_DIR} — run embedding analysis first; skipping ${W}/${V}"; continue
        fi
        shopt -s nullglob; gw_csvs=("${GW_PATH}"/*.csv); shopt -u nullglob
        [ "${#gw_csvs[@]}" -gt 0 ] || { echo "WARNING: ${GW_PATH} has no *.csv — skipping ${W}/${V}"; continue; }
        echo "--- ${W}/${V} (${EMB_MODEL_PATH}): ${#gw_csvs[@]} genome CSV(s)  max_length=${MAX_LENGTH} ---"
        for csv in "${gw_csvs[@]}"; do
            stem="$(basename "${csv}" .csv)"; J="gwheads_${W}_${V}_${stem}"
            sbatch --job-name="${J}" \
                --output="${LOGDIR}/${J}_%j.out" --error="${LOGDIR}/${J}_%j.err" \
                "${FLAGS[@]}" \
                --export="ALL,REPO_ROOT=${REPO_ROOT},HF_HOME=${HF_HOME},REPL_OUTPUT_DIR=${REPL_W_DIR},VARIANT=${V},BASE_MODEL=${EMB_MODEL_PATH},INPUT_CSV=${csv},MAX_LENGTH=${MAX_LENGTH},BATCH_SIZE=${BATCH},POOLING=${POOLING:-mean},THRESHOLD=${THRESHOLD:-0.5}" \
                "${JOB}"
            NUM=$((NUM+1))
        done
    done
done
echo ""
echo "Submitted ${NUM} all-heads genome-wide jobs. Monitor: squeue -u \$USER"
echo "Output: ${OUTPUT_DIR}/<W>/genome_wide_heads/<variant>/{lp,nn}/"
