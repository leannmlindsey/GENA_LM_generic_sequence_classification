#!/bin/bash
#
# Summarize LAMBDA inference results for the GENA-LM variants currently
# being evaluated. Reads the per-CSV _metrics.json files that
# inference_gena_lm_dir.py writes when --save_metrics is passed, plus the
# per-run summary.json with the mean metrics for each (variant × category × window).
#
# Usage:
#   bash summarize_results.sh
#
# Output structure expected (matches submit_lambda_full_eval.sh layout):
#   ${RESULTS_ROOT}/<variant>/binary/<window>/test_metrics.json
#   ${RESULTS_ROOT}/<variant>/error_bias/<window>/*_metrics.json
#   ${RESULTS_ROOT}/<variant>/genome_wide/<window>/summary.json (+ per-CSV metrics)

set -e

RESULTS_ROOT="/data/lindseylm/GLM_EVALUATIONS/MODELS/FINAL_RESULTS"
VARIANTS=("gena_lm_bigbird" "gena_lm_moderngena")
WINDOWS=("2k" "4k" "8k")

python3 - <<PYTHON
import json
import os
import glob

ROOT = "${RESULTS_ROOT}"
VARIANTS = [$(printf '"%s", ' "${VARIANTS[@]}")]
WINDOWS = [$(printf '"%s", ' "${WINDOWS[@]}")]

# ───────────────────────────────────────────────────────────────────────
# 1. Binary test set (one file per variant × window)
# ───────────────────────────────────────────────────────────────────────
print()
print("=== Binary test ===")
print(f"{'variant':<22} {'window':<6}  {'MCC':>6}  {'F1':>6}  {'AUC':>6}  {'spec':>6}  {'sens':>6}")
print("-" * 70)
for variant in VARIANTS:
    for window in WINDOWS:
        # File written by inference_gena_lm_dir.py is <basename>_metrics.json.
        # For binary the basename is "test" (pattern was "test.csv").
        m_path = f"{ROOT}/{variant}/binary/{window}/test_metrics.json"
        if not os.path.exists(m_path):
            print(f"{variant:<22} {window:<6}  (not yet)")
            continue
        m = json.load(open(m_path))
        print(f"{variant:<22} {window:<6}  {m['mcc']:>6.3f}  {m['f1']:>6.3f}  "
              f"{m['auc']:>6.3f}  {m['specificity']:>6.3f}  {m['sensitivity']:>6.3f}")

# ───────────────────────────────────────────────────────────────────────
# 2. Error & bias diagnostics (~4 files per variant × window)
# ───────────────────────────────────────────────────────────────────────
print()
print("=== Error & bias diagnostics ===")
print(f"{'variant':<22} {'window':<6} {'diagnostic':<32}  {'MCC':>6}  {'spec':>6}  {'sens':>6}")
print("-" * 90)
for variant in VARIANTS:
    for window in WINDOWS:
        eb_dir = f"{ROOT}/{variant}/error_bias/{window}"
        if not os.path.isdir(eb_dir):
            continue
        for m_path in sorted(glob.glob(f"{eb_dir}/*_metrics.json")):
            m = json.load(open(m_path))
            tag = os.path.basename(m_path).replace("_metrics.json", "")
            print(f"{variant:<22} {window:<6} {tag:<32}  "
                  f"{m['mcc']:>6.3f}  {m['specificity']:>6.3f}  {m['sensitivity']:>6.3f}")

# ───────────────────────────────────────────────────────────────────────
# 3. Genome-wide (mean across all assemblies — from summary.json)
# ───────────────────────────────────────────────────────────────────────
print()
print("=== Genome-wide (mean across assemblies) ===")
print(f"{'variant':<22} {'window':<6}  {'files':>5}  {'mean_MCC':>9}  {'mean_F1':>8}  {'mean_AUC':>9}")
print("-" * 75)
for variant in VARIANTS:
    for window in WINDOWS:
        s_path = f"{ROOT}/{variant}/genome_wide/{window}/summary.json"
        if not os.path.exists(s_path):
            print(f"{variant:<22} {window:<6}  (not yet)")
            continue
        s = json.load(open(s_path))
        mm = s.get("mean_metrics", {})
        nf = s.get("files_processed", 0)
        print(f"{variant:<22} {window:<6}  {nf:>5}  "
              f"{mm.get('mcc', 0):>9.3f}  {mm.get('f1', 0):>8.3f}  {mm.get('auc', 0):>9.3f}")

print()
print("Note: 'mean_MCC' here is the mean of per-genome MCCs (macro-average),")
print("not the pooled MCC. Pooled metrics come from the grid search step.")
PYTHON
