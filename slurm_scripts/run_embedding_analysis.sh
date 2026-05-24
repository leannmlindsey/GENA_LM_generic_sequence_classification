#!/bin/bash
#SBATCH --job-name=gena_lm_emb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=nt_emb_%j.out
#SBATCH --error=nt_emb_%j.err

# Biowulf batch script for GENA-LM embedding analysis
# Usage: sbatch run_embedding_analysis.sh
#
# Required environment variables:
#   CSV_DIR: Path to directory containing train.csv, dev.csv, test.csv
#   MODEL_PATH: Path to fine-tuned model or HuggingFace model name

echo "============================================================"
echo "GENA-LM Embedding Analysis"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load CUDA/12.8

# Set CUDA_HOME if not set
if [ -z "${CUDA_HOME}" ]; then
    NVCC_PATH=$(which nvcc 2>/dev/null)
    if [ -n "${NVCC_PATH}" ]; then
        export CUDA_HOME=$(dirname $(dirname "${NVCC_PATH}"))
    fi
fi

# Activate conda environment
source activate gena_lm

# Ignore user site-packages
export PYTHONNOUSERSITE=1
SCRIPT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/GENA-LM/GENA_LM_generic_sequence_classification"
# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Set defaults
MODEL_PATH=${MODEL_PATH:-AIRI-Institute/gena-lm-bert-base-t2t}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_LENGTH=${MAX_LENGTH:-2048}
POOLING=${POOLING:-mean}
SEED=${SEED:-42}
NN_EPOCHS=${NN_EPOCHS:-100}
NN_HIDDEN_DIM=${NN_HIDDEN_DIM:-256}
NN_LR=${NN_LR:-0.001}
INCLUDE_RANDOM_BASELINE=${INCLUDE_RANDOM_BASELINE:-false}

# Validate required parameters
if [ -z "${CSV_DIR}" ]; then
    echo "ERROR: CSV_DIR is not set"
    exit 1
fi

# Navigate to repo root
cd "${SCRIPT_DIR}"
echo "Working directory: $(pwd)"

# Set output directory — use absolute path so output survives even if cwd
# changes or SLURM drops the job in /lscratch.
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/results/embedding_analysis/$(basename ${CSV_DIR})}
mkdir -p "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "Configuration:"
echo "============================================================"
echo "  Model: ${MODEL_PATH}"
echo "  CSV dir: ${CSV_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "  Pooling: ${POOLING}"
echo "  Seed: ${SEED}"
echo "  NN epochs: ${NN_EPOCHS}"
echo "  NN hidden dim: ${NN_HIDDEN_DIM}"
echo "  NN learning rate: ${NN_LR}"
echo "  Include random baseline: ${INCLUDE_RANDOM_BASELINE}"
echo "============================================================"
echo ""

# Build random baseline flag
RANDOM_BASELINE_FLAG=""
if [ "${INCLUDE_RANDOM_BASELINE}" == "true" ]; then
    RANDOM_BASELINE_FLAG="--include_random_baseline"
fi

# Run embedding analysis
python embedding_analysis_gena_lm.py \
    --csv_dir="${CSV_DIR}" \
    --model_path="${MODEL_PATH}" \
    --output_dir="${OUTPUT_DIR}" \
    --batch_size=${BATCH_SIZE} \
    --max_length=${MAX_LENGTH} \
    --pooling="${POOLING}" \
    --seed=${SEED} \
    --nn_epochs=${NN_EPOCHS} \
    --nn_hidden_dim=${NN_HIDDEN_DIM} \
    --nn_lr=${NN_LR} \
    ${RANDOM_BASELINE_FLAG}

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"

exit ${EXIT_CODE}
