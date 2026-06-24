#!/bin/bash
#
# Verify the random-embedding baseline actually ran for every (window × variant)
# embedding cell — don't trust INCLUDE_RANDOM_BASELINE=true alone (the random
# pass can silently fail while the pretrained pass still succeeds).
#
# For GENA-LM the random baseline is produced by the Surface-D embedding stage
# (embedding_analysis_gena_lm.py --include_random_baseline), which runs once per
# (window × variant) and is independent of the finetune seed. So the "cells"
# checked here are WINDOWS × VARIANTS, NOT per-seed.
#
# For each $OUTPUT_DIR/<W>/embedding/<variant>/ this checks that BOTH:
#   1. embeddings_random.npz exists, AND
#   2. embedding_analysis_results.json contains random_linear_probe_mcc and
#      random_nn_mcc (plus the embedding_power_* deltas).
# It prints the random vs pretrained MCC per cell (random should be BELOW
# pretrained) and an "N / total complete" count at the end.
#
# Usage:
#   bash slurm_scripts/lambda_replication/check_random_baseline.sh
#
# Reads OUTPUT_DIR / WINDOWS / VARIANTS from lambda_replication.conf, so it
# matches whatever sweep you configured.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: missing ${CONFIG}"; exit 1
fi
# shellcheck disable=SC1090
source "${CONFIG}"

[ -n "${OUTPUT_DIR:-}" ] || { echo "ERROR: OUTPUT_DIR unset in conf"; exit 1; }
[ -n "${WINDOWS:-}" ]    || { echo "ERROR: WINDOWS unset in conf";    exit 1; }
[ -n "${VARIANTS:-}" ]   || { echo "ERROR: VARIANTS unset in conf";   exit 1; }

echo "============================================================"
echo "Random-baseline check (Surface D embedding cells)"
echo "  OUTPUT_DIR: ${OUTPUT_DIR}"
echo "  WINDOWS:    ${WINDOWS}"
echo "  VARIANTS:   ${VARIANTS}"
echo "============================================================"

TOTAL=0
COMPLETE=0

for W in ${WINDOWS}; do
    for V in ${VARIANTS}; do
        TOTAL=$((TOTAL + 1))
        CELL_DIR="${OUTPUT_DIR}/${W}/embedding/${V}"
        NPZ="${CELL_DIR}/embeddings_random.npz"
        JSON="${CELL_DIR}/embedding_analysis_results.json"
        TAG="${W}/${V}"

        if [ ! -d "${CELL_DIR}" ]; then
            printf "  [MISSING] %-16s no embedding dir (%s)\n" "${TAG}" "${CELL_DIR}"
            continue
        fi
        if [ ! -f "${NPZ}" ]; then
            printf "  [FAIL]    %-16s embeddings_random.npz missing\n" "${TAG}"
            continue
        fi
        if [ ! -f "${JSON}" ]; then
            printf "  [FAIL]    %-16s embedding_analysis_results.json missing\n" "${TAG}"
            continue
        fi

        # Parse the results JSON: require random_* keys, report random vs
        # pretrained MCC. Exit 0 = OK, 2 = keys missing, 3 = unreadable.
        STATUS="$(JSON="${JSON}" python3 - <<'PY'
import json, os, sys
try:
    d = json.load(open(os.environ["JSON"]))
except Exception as e:
    print(f"UNREADABLE {e}"); sys.exit(3)
need = ["random_linear_probe_mcc", "random_nn_mcc"]
missing = [k for k in need if k not in d]
if missing:
    print("MISSING_KEYS " + ",".join(missing)); sys.exit(2)
rl, rn = d["random_linear_probe_mcc"], d["random_nn_mcc"]
pl = d.get("pretrained_linear_probe_mcc", float("nan"))
pn = d.get("pretrained_nn_mcc", float("nan"))
warn = "" if (rl <= pl and rn <= pn) else "  <-- WARN: random >= pretrained MCC"
print(f"OK LP rand={rl:.4f} pre={pl:.4f} | NN rand={rn:.4f} pre={pn:.4f}{warn}")
PY
)" || true

        case "${STATUS}" in
            OK*)          printf "  [OK]      %-16s %s\n" "${TAG}" "${STATUS#OK }"
                          COMPLETE=$((COMPLETE + 1)) ;;
            MISSING_KEYS*) printf "  [FAIL]    %-16s npz present but JSON %s — random pass silently failed; re-run this cell\n" "${TAG}" "${STATUS}" ;;
            UNREADABLE*)  printf "  [FAIL]    %-16s results JSON %s\n" "${TAG}" "${STATUS}" ;;
            *)            printf "  [FAIL]    %-16s unexpected: %s\n" "${TAG}" "${STATUS}" ;;
        esac
    done
done

echo "------------------------------------------------------------"
echo "Random baseline complete: ${COMPLETE} / ${TOTAL} cells"
echo "------------------------------------------------------------"
[ "${COMPLETE}" -eq "${TOTAL}" ] || exit 1
