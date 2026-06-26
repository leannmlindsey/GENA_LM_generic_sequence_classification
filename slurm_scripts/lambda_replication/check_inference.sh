#!/bin/bash
#
# Verify the INFERENCE stage (Stage 2) produced every expected output.
#
# For each window it checks winners.json, then for each variant with a winner
# checks the prediction CSVs under inference/<variant>/, reporting the MCC from
# each *_metrics.json where present:
#   - test        test_predictions.csv            (always)
#   - fpr         fpr_predictions.csv             (if FPR_<W> set)
#   - gc_control  gc_control_predictions.csv      (if GC_<W> set)
#   - fnr         fnr_predictions.csv             (if FNR_<W> set)
#   - genome      genome_wide_*_predictions.csv   (if GENOME_WIDE_<W> set; counts files)
#   - phrog       GENA_LM_<variant>_phage_annotated_segments_2k_predictions.csv
#                                                 (2k only, if PHROG_2k set)
#
# Which diagnostics are "expected" is derived the same way run_lambda_inference.sh
# decides what to submit — from the FPR_/GC_/FNR_/GENOME_WIDE_/PHROG_ conf vars —
# so a partial config is reported correctly (unset = not expected, not a failure).
#
# Reads OUTPUT_DIR / WINDOWS / VARIANTS (+ the diagnostic path vars) from
# lambda_replication.conf. Exits non-zero if an expected output is missing.
#
# Usage:
#   bash slurm_scripts/lambda_replication/check_inference.sh
#
# See also: check_training.sh, check_random_baseline.sh.

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
echo "Inference completeness check"
echo "  OUTPUT_DIR: ${OUTPUT_DIR}"
echo "  WINDOWS:    ${WINDOWS}"
echo "  VARIANTS:   ${VARIANTS}"
echo "============================================================"

# The diagnostic path vars (FPR_<W> etc.) are exported by the sourced conf, so
# the python below reads them straight from the environment.
OUTPUT_DIR="${OUTPUT_DIR}" WINDOWS="${WINDOWS}" VARIANTS="${VARIANTS}" \
python3 - <<'PY'
import json, glob, os, sys

out   = os.environ["OUTPUT_DIR"]
wins  = os.environ["WINDOWS"].split()
varis = os.environ["VARIANTS"].split()

def isset(name):
    return bool(os.environ.get(name, "").strip())

def mcc_of(path):
    try:
        return json.load(open(path)).get("mcc")
    except Exception:
        return None

expected = missing = 0

for w in wins:
    wdir = os.path.join(out, w)
    winners = os.path.join(wdir, "winners.json")
    print(f"\n--- window {w} ---")
    if not os.path.isfile(winners):
        print(f"  [FAIL] winners.json missing — Stage 2 selection didn't run")
    have = {}
    if os.path.isfile(winners):
        try:
            have = json.load(open(winners))
        except Exception:
            have = {}

    # single-file diagnostics expected for this window (name -> conf var gate)
    diags = [("test", True)]
    diags.append(("fpr",        isset(f"FPR_{w}")))
    diags.append(("gc_control", isset(f"GC_{w}")))
    diags.append(("fnr",        isset(f"FNR_{w}")))

    for v in varis:
        idir = os.path.join(wdir, "inference", v)
        tag = f"{w}/{v}"
        if v not in have:
            print(f"  [WARN] {tag}: no winner in winners.json (training incomplete?)")
        # single-file diagnostics
        for name, want in diags:
            if not want:
                continue
            expected += 1
            csv = os.path.join(idir, f"{name}_predictions.csv")
            mj  = os.path.join(idir, f"{name}_predictions_metrics.json")
            if os.path.isfile(csv):
                m = mcc_of(mj)
                mstr = f"mcc={m:.4f}" if isinstance(m, (int, float)) else "(no metrics)"
                print(f"  [OK]   {tag:16} {name:11} {mstr}")
            else:
                print(f"  [MISS] {tag:16} {name:11} {os.path.basename(csv)} not found")
                missing += 1
        # genome-wide (a set of files)
        if isset(f"GENOME_WIDE_{w}"):
            expected += 1
            gw = glob.glob(os.path.join(idir, "genome_wide_*_predictions.csv"))
            if gw:
                print(f"  [OK]   {tag:16} {'genome':11} {len(gw)} prediction file(s)")
            else:
                print(f"  [MISS] {tag:16} {'genome':11} no genome_wide_*_predictions.csv")
                missing += 1
        # PHROG (2k only)
        if w == "2k" and isset("PHROG_2k"):
            expected += 1
            phrog = os.path.join(idir, f"GENA_LM_{v}_phage_annotated_segments_2k_predictions.csv")
            if os.path.isfile(phrog):
                m = mcc_of(phrog.replace(".csv", "_metrics.json"))
                mstr = f"mcc={m:.4f}" if isinstance(m, (int, float)) else ""
                print(f"  [OK]   {tag:16} {'phrog':11} {os.path.basename(phrog)} {mstr}")
            else:
                print(f"  [MISS] {tag:16} {'phrog':11} {os.path.basename(phrog)} not found")
                missing += 1

print("\n" + "-" * 60)
print(f"Expected outputs present: {expected - missing} / {expected}   (missing: {missing})")
print("-" * 60)
sys.exit(0 if missing == 0 else 1)
PY
