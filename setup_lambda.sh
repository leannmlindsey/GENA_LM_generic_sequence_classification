#!/bin/bash
# Setup conda environment for the LAMBDA-evaluation scripts in this fork.
#
# Inherits the upstream modernGENA environment file as the foundation
# (examples/modernGENA/environment.yml) so the torch / transformers / sklearn
# version pins match upstream's tested stack — then pip-installs the extra
# packages the LAMBDA evaluation scripts in this repo need beyond upstream's
# CTCF example (datasets, matplotlib, scipy, tqdm, einops).
#
# Run this once per fresh checkout. The env is named `gena_lm` by default
# (override with ENV_NAME); the SLURM scripts in slurm_scripts/ assume that
# name.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-gena_lm}"
UPSTREAM_ENV_YML="${SCRIPT_DIR}/examples/modernGENA/environment.yml"

if [ ! -f "${UPSTREAM_ENV_YML}" ]; then
    echo "ERROR: upstream env file not found: ${UPSTREAM_ENV_YML}"
    echo "Are you in the GENA_LM_generic_sequence_classification root?"
    exit 1
fi

echo "============================================================"
echo "Setting up GENA-LM LAMBDA-evaluation environment"
echo "  Env name:        ${ENV_NAME}"
echo "  Upstream env:    ${UPSTREAM_ENV_YML}"
echo "============================================================"

# Adjust the module loads to match your cluster — these are the Biowulf names.
module load conda
module load CUDA/12.8

# Step 1: create env from upstream's environment.yml (overrides the `name:`
# field in the yaml via -n). This gives us upstream's tested torch /
# transformers / sklearn pins exactly.
echo
echo "[1/2] Creating env from upstream's environment.yml..."
conda env create -n "${ENV_NAME}" -f "${UPSTREAM_ENV_YML}"

source activate "${ENV_NAME}"

# Step 2: install the extras the LAMBDA evaluation scripts need that upstream's
# env doesn't include (their CTCF example uses a custom PyTorch Dataset, no
# plotting; ours uses HF datasets, matplotlib, scipy KDE).
echo
echo "[2/2] Installing LAMBDA-evaluation extras on top..."
pip install \
    'datasets>=2.14,<4.0' \
    'matplotlib>=3.7,<4.0' \
    'scipy>=1.10,<2.0' \
    'tqdm>=4.65,<5.0' \
    'einops>=0.7,<1.0'

echo
echo "============================================================"
echo "Environment '${ENV_NAME}' is ready."
echo "Activate with:  source activate ${ENV_NAME}"
echo "============================================================"
echo
echo "Quick test:"
echo "  python -c \"from transformers import AutoTokenizer; \\"
echo "      t = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base-t2t'); \\"
echo "      print('GENA-LM tokenizer loaded:', t)\""
