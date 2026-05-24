#!/bin/bash
#
# Run the post-inference grid search over clustering hyperparameters for the
# GENA-LM BigBird and ModernGENA models. This matches the sweep methodology
# used for DNABERT2 / NTv2 / Caduceus / ProkBERT / GENERanno / megaDNA /
# EVO2 in grid_search_results_v2_4k8k/.
#
# Sweep (per model × window — 24 combos each):
#   norm           = zscore                             (1)
#   thresholds     = 0.5, 0.75, 1.0, 1.25 (in sigma)    (4)
#   smooth-windows = 3, 5, 7                            (3)
#   min-sizes      = 20000 bp                           (1)
#   merge-gaps     = 1000, 3000 bp                      (2)
#
# Prerequisite: link_for_grid_search.sh has been run so that
# ${GRID_DATA_DIR}/per_segment_<W>/<MODEL>/*.csv symlinks exist.
#
# Usage:
#   bash run_grid_search.sh                # all 3 windows, both variants
#   bash run_grid_search.sh "2k 4k 8k"     # explicit window list
#
# Note: this runs on the login node (CPU only, ~10-30 min for 6 models).
# If you'd rather queue it, prepend `sbatch --mem=32g --cpus-per-task=4
#   --time=2:00:00 --wrap='bash run_grid_search.sh ...'`.

set -e

# ─── Configuration ──────────────────────────────────────────────────────────
GRID_DATA_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/FINAL_RESULTS/grid_search_input"
GROUND_TRUTH="/gpfs/gsfs12/users/Irp-jiang/share/lindseylm/lambda_final/ground_truth.csv"
OUTPUT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/FINAL_RESULTS/grid_search_results_gena_lm"

# Path to grid_search_clustering.py — lives in the LAMBDA repo at
# replication/grid_search_clustering.py (copied there from FINAL_SCRIPTS so
# the full methodology travels with the paper repo). Update if the LAMBDA
# repo lives at a different path on Biowulf.
GRID_SEARCH_SCRIPT="/data/lindseylm/GLM_EVALUATIONS/NAR_GENOMICS_LAMBDA_REPO/LAMBDA/replication/grid_search_clustering.py"

WINDOWS="${1:-2k 4k 8k}"

# ─── Validation ─────────────────────────────────────────────────────────────
for f in "${GRID_SEARCH_SCRIPT}" "${GROUND_TRUTH}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: required file missing: ${f}"
        exit 1
    fi
done

if [ ! -d "${GRID_DATA_DIR}" ]; then
    echo "ERROR: GRID_DATA_DIR missing: ${GRID_DATA_DIR}"
    echo "       Run link_for_grid_search.sh first."
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# ─── Run ────────────────────────────────────────────────────────────────────
echo "=========================================="
echo "Grid search — GENA-LM BigBird + ModernGENA"
echo "=========================================="
echo "Data dir:      ${GRID_DATA_DIR}"
echo "Ground truth:  ${GROUND_TRUTH}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Windows:       ${WINDOWS}"
echo ""
echo "Sweep (matches grid_search_results_v2_4k8k):"
echo "  norm:           zscore"
echo "  thresholds:     0.5 0.75 1.0 1.25"
echo "  smooth-windows: 3 5 7"
echo "  min-sizes:      20000"
echo "  merge-gaps:     1000 3000"
echo "=========================================="

python "${GRID_SEARCH_SCRIPT}" \
    --data-dir "${GRID_DATA_DIR}" \
    --gt "${GROUND_TRUTH}" \
    --output-dir "${OUTPUT_DIR}" \
    --norm-methods zscore \
    --thresholds 0.5 0.75 1.0 1.25 \
    --smooth-windows 3 5 7 \
    --min-sizes 20000 \
    --merge-gaps 1000 3000 \
    --window-sizes ${WINDOWS}

echo ""
echo "Done. Inspect:"
echo "  ${OUTPUT_DIR}/grid_search_best.csv     # one best row per (model, window)"
echo "  ${OUTPUT_DIR}/grid_search_results.csv  # all 24 × n_models × n_windows rows"
echo "  ${OUTPUT_DIR}/grid_search_heatmap_*.png"
echo ""
echo "Next: bundle these results with the published-model rows for the paper table:"
echo "  python /data/lindseylm/.../FINAL_SCRIPTS/compute_all_model_metrics_and_latex.py"
