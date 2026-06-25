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

# Delta-AI (GH200): initialise conda from the user's miniconda base. No module
# system is needed — recent torch ships aarch64 CUDA wheels that bundle the CUDA
# runtime, so there is no separate `module load CUDA` step on Delta.
#   (Biowulf equivalent was: module load CUDA/12.8)
CONDA_BASE="${CONDA_BASE:-/u/llindsey1/miniconda3}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Step 1: create env from upstream's environment.yml (overrides the `name:`
# field in the yaml via -n). This gives us upstream's tested torch /
# transformers / sklearn pins exactly.
echo
echo "[1/3] Creating env from upstream's environment.yml..."
conda env create -n "${ENV_NAME}" -f "${UPSTREAM_ENV_YML}"

conda activate "${ENV_NAME}"

# Step 2: pin the aarch64 CUDA torch build. On Delta GH200 (aarch64) the plain
# pip pin in environment.yml resolves to a CPU-only wheel (torch X.Y.Z+cpu), so
# torch.cuda.is_available() is False even on a healthy GPU node. Reinstall the
# matching CUDA wheel from the PyTorch index. torch 2.5.1 / cu124 matches the
# other working Delta GH200 envs (generanno, etc.); override via env vars for a
# different cluster/stack.
echo
echo "[2/3] Installing aarch64 CUDA torch wheel..."
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"
pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_CUDA_INDEX}"
# The CUDA index is a FULL index replacement and does not host torch's
# pure-Python deps (typing_extensions, sympy, ...). If the env-create step
# didn't leave them in place, `import torch` fails with ModuleNotFoundError;
# restore them from PyPI (default index) so the env is usable.
python -c "import torch" 2>/dev/null || \
    pip install typing_extensions sympy networkx jinja2 fsspec filelock mpmath

# Step 3: install the extras the LAMBDA evaluation scripts need that upstream's
# env doesn't include (their CTCF example uses a custom PyTorch Dataset, no
# plotting; ours uses HF datasets, matplotlib, scipy KDE).
echo
echo "[3/3] Installing LAMBDA-evaluation extras on top..."
pip install \
    'datasets>=2.14,<4.0' \
    'matplotlib>=3.7,<4.0' \
    'scipy>=1.10,<2.0' \
    'tqdm>=4.65,<5.0' \
    'einops>=0.7,<1.0'

echo
echo "============================================================"
echo "Environment '${ENV_NAME}' is ready."
echo "Activate with:  source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
echo "============================================================"
echo
echo "Quick test:"
echo "  python -c \"from transformers import AutoTokenizer; \\"
echo "      t = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base-t2t'); \\"
echo "      print('GENA-LM tokenizer loaded:', t)\""
