#!/bin/bash
#
# Bridge the GENA-LM inference output layout to the layout that
# FINAL_SCRIPTS/grid_search_clustering.py expects, via symlinks.
#
# Inference writes:
#   ${RESULTS_ROOT}/${MODEL_NAME}/genome_wide/${WINDOW}/<assembly>_predictions.csv
#
# grid_search_clustering.py reads (per FINAL_SCRIPTS/per_segment_data_paths.md
# and grid_search_clustering.py:334):
#   ${GRID_DATA_DIR}/per_segment_${WINDOW}/${MODEL_NAME}/<assembly>_predictions.csv
#
# This script creates symlinks
#   ${GRID_DATA_DIR}/per_segment_<W>/<MODEL_NAME> ->
#     ${RESULTS_ROOT}/<MODEL_NAME>/genome_wide/<W>
# for each (variant × window) so the grid search picks them up.

set -e

# ─── Configuration ──────────────────────────────────────────────────────────
RESULTS_ROOT="/data/lindseylm/GLM_EVALUATIONS/MODELS/FINAL_RESULTS"
GRID_DATA_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/FINAL_RESULTS/grid_search_input"

# Model names to link — these are the MODEL_NAME values used in
# submit_lambda_full_eval.sh's variant presets. Each must already have
# inference output at ${RESULTS_ROOT}/${MODEL_NAME}/genome_wide/<W>/.
MODEL_NAMES=("gena_lm_bigbird" "gena_lm_moderngena")
WINDOWS=("2k" "4k" "8k")

# ─── Link creation ──────────────────────────────────────────────────────────
echo "Linking inference output into per_segment_<W>/<MODEL>/ layout"
echo "  source: ${RESULTS_ROOT}/<MODEL>/genome_wide/<W>/"
echo "  target: ${GRID_DATA_DIR}/per_segment_<W>/<MODEL>/"
echo ""

mkdir -p "${GRID_DATA_DIR}"

for window in "${WINDOWS[@]}"; do
    seg_dir="${GRID_DATA_DIR}/per_segment_${window}"
    mkdir -p "${seg_dir}"

    for model in "${MODEL_NAMES[@]}"; do
        src="${RESULTS_ROOT}/${model}/genome_wide/${window}"
        dst="${seg_dir}/${model}"

        if [ ! -d "${src}" ]; then
            echo "[${window} / ${model}] SKIP — source dir missing: ${src}"
            continue
        fi

        n=$(ls -1 ${src}/*_predictions.csv 2>/dev/null | wc -l)
        if [ "${n}" -eq 0 ]; then
            echo "[${window} / ${model}] SKIP — no *_predictions.csv in ${src}"
            continue
        fi

        # Replace any existing link/dir at the target
        if [ -L "${dst}" ] || [ -d "${dst}" ]; then
            rm -rf "${dst}"
        fi

        ln -s "${src}" "${dst}"
        echo "[${window} / ${model}] linked (${n} predictions)"
    done
done

echo ""
echo "Done. Layout under ${GRID_DATA_DIR}:"
ls -la "${GRID_DATA_DIR}"/per_segment_*/* 2>/dev/null | grep "^l" | head -20
