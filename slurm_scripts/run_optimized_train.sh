#!/bin/bash
#SBATCH --job-name=gena_lm_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=nt_train_%j.out
#SBATCH --error=nt_train_%j.err

# Biowulf batch script for GENA-LM optimized training
# Usage: sbatch run_optimized_train.sh
#
# Required environment variables:
#   DATASET_DIR: Directory containing train.csv, dev.csv, test.csv
#   OUTPUT_DIR: Directory to save model checkpoints
#   MAX_LENGTH: Maximum token length (512, 1024, or 2048)
#   SEED: Random seed for reproducibility

echo "============================================================"
echo "GENA-LM Optimized Training"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load CUDA/12.8

# Activate conda environment
source activate gena_lm

# Ignore user site-packages to avoid conflicts
export PYTHONNOUSERSITE=1

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi

echo ""
echo "Python environment:"
which python
python --version

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Navigate to script directory
SCRIPT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/GENA_LM/GENA_LM_generic_sequence_classification"
cd "${SCRIPT_DIR}" || exit
echo "Working directory: $(pwd)"

# Set defaults for optional parameters
GPU_TYPE=${GPU_TYPE:-"A100"}
LEARNING_RATE=${LEARNING_RATE:-"3e-5"}
NUM_EPOCHS=${NUM_EPOCHS:-"10"}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-"3"}
EVAL_STRATEGY=${EVAL_STRATEGY:-"steps"}
EVAL_STEPS=${EVAL_STEPS:-"500"}

# Validate required parameters
if [ -z "${DATASET_DIR}" ]; then
    echo "ERROR: DATASET_DIR is not set"
    exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: OUTPUT_DIR is not set"
    exit 1
fi

if [ -z "${MAX_LENGTH}" ]; then
    echo "ERROR: MAX_LENGTH is not set"
    exit 1
fi

if [ -z "${SEED}" ]; then
    echo "ERROR: SEED is not set"
    exit 1
fi

# Validate dataset directory
if [ ! -d "${DATASET_DIR}" ]; then
    echo "ERROR: Dataset directory not found: ${DATASET_DIR}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Set precision based on GPU type
if [ "${GPU_TYPE}" == "A100" ] || [ "${GPU_TYPE}" == "H100" ]; then
    PRECISION_FLAGS="--bf16 --tf32"
    OPTIMIZER="adamw_torch_fused"
else
    PRECISION_FLAGS="--fp16"
    OPTIMIZER="adamw_torch"
fi

# Set batch size based on max_length (in tokens) and GPU
# Note: GENA-LM/ESM does NOT support gradient checkpointing
case "${MAX_LENGTH}" in
    512)  # For 2k nucleotide sequences
        if [ "${GPU_TYPE}" == "A100" ] || [ "${GPU_TYPE}" == "H100" ]; then
            BATCH_SIZE="8"
            GRAD_ACCUM="1"
        else
            BATCH_SIZE="4"
            GRAD_ACCUM="2"
        fi
        ;;
    1024)  # For 4k nucleotide sequences
        if [ "${GPU_TYPE}" == "A100" ] || [ "${GPU_TYPE}" == "H100" ]; then
            BATCH_SIZE="1"
            GRAD_ACCUM="1"
        else
            BATCH_SIZE="1"
            GRAD_ACCUM="1"
        fi
        ;;
    2048)  # For 8k nucleotide sequences
        if [ "${GPU_TYPE}" == "A100" ] || [ "${GPU_TYPE}" == "H100" ]; then
            BATCH_SIZE="1"
            GRAD_ACCUM="1"
        else
            BATCH_SIZE="1"
            GRAD_ACCUM="1"
        fi
        ;;
    *)
        echo "ERROR: Unsupported MAX_LENGTH: ${MAX_LENGTH}"
        echo "Supported values: 512 (2k seq), 1024 (4k seq), 2048 (8k seq)"
        exit 1
        ;;
esac

EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM))

echo ""
echo "============================================================"
echo "Training Configuration"
echo "============================================================"
echo "Dataset: ${DATASET_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Sequence length: ${MAX_LENGTH} tokens"
echo "GPU type: ${GPU_TYPE}"
echo "Seed: ${SEED}"
echo ""
echo "Training parameters:"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${GRAD_ACCUM}"
echo "  Effective batch size: ${EFFECTIVE_BATCH_SIZE}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Max epochs: ${NUM_EPOCHS}"
echo "  Early stopping patience: ${EARLY_STOPPING_PATIENCE}"
echo ""
echo "Optimizations:"
echo "  Precision: ${PRECISION_FLAGS}"
echo "  Optimizer: ${OPTIMIZER}"
echo ""
echo "Evaluation:"
echo "  Strategy: ${EVAL_STRATEGY}"
if [ "${EVAL_STRATEGY}" == "steps" ]; then
    echo "  Eval steps: ${EVAL_STEPS}"
fi
echo "============================================================"
echo ""

# Run training
python finetune_gena_lm_phage.py \
    --dataset_dir "${DATASET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_length ${MAX_LENGTH} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --eval_strategy ${EVAL_STRATEGY} \
    --eval_steps ${EVAL_STEPS} \
    --save_strategy ${EVAL_STRATEGY} \
    --save_steps ${EVAL_STEPS} \
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
    --optim ${OPTIMIZER} \
    --seed ${SEED} \
    ${PRECISION_FLAGS}

EXIT_CODE=$?

echo ""
echo "============================================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Training completed successfully"
    echo "Model saved to: ${OUTPUT_DIR}"
else
    echo "Training failed with exit code: ${EXIT_CODE}"
fi
echo "Job finished at: $(date)"
echo "============================================================"

exit ${EXIT_CODE}
