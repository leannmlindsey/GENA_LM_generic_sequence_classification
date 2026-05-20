#!/bin/bash
# Setup conda environment for the LAMBDA-evaluation scripts in this fork.
# Creates a fresh `gena_lm` env distinct from any environment used by the
# upstream modernGENA / GENA-LM training code. Run once.

set -euo pipefail

echo "Setting up GENA-LM LAMBDA-evaluation environment..."

# Adjust the module loads to match your cluster — these are the Biowulf names.
module load conda
module load CUDA/12.8

ENV_NAME="${ENV_NAME:-gena_lm}"

conda create -n "${ENV_NAME}" python=3.10 -y
source activate "${ENV_NAME}"

# Install PyTorch with CUDA 12.x wheels
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# HuggingFace stack — GENA-LM works with standard AutoTokenizer / AutoModel
pip install -r "$(dirname "${BASH_SOURCE[0]}")/requirements_lambda.txt"

echo ""
echo "============================================================"
echo "Environment '${ENV_NAME}' is ready."
echo "Activate with:  source activate ${ENV_NAME}"
echo "============================================================"
echo ""
echo "Quick test:"
echo "  python -c \"from transformers import AutoTokenizer; \\"
echo "      t = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base-t2t'); \\"
echo "      print('GENA-LM tokenizer loaded:', t)\""
