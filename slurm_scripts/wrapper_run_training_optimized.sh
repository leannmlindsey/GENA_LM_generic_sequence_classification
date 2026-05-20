#!/bin/bash

# Optimized Training Wrapper for GENA-LM
#
# This wrapper submits a SLURM job with recommended configurations for
# different sequence lengths and hardware.
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash wrapper_run_training_optimized.sh <SEED>
#   Example: bash wrapper_run_training_optimized.sh 42

#####################################################################
# CONFIGURATION - Edit this section
#####################################################################

# === REQUIRED: Dataset Directory ===
# Directory containing train.csv, dev.csv, test.csv
#DATASET_DIR="/home/lindseylm/lindseylm/lambda_final/merged_datasets_filtered/2k"
DATASET_DIR="/home/lindseylm/lindseylm/lambda_final/merged_datasets_filtered/4k"
#DATASET_DIR="/home/lindseylm/lindseylm/lambda_final/merged_datasets_filtered/8k"

# === REQUIRED: Output Directory ===
OUTPUT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/GENA_LM/GENA_LM_generic_sequence_classification/output/filtered/4k"

# === Sequence Length Configuration ===
# MAX_LENGTH is in TOKENS, not nucleotides!
# GENA-LM uses a 32k BPE tokenizer; tokens ≈ sequence_length / 6 in practice.
# Choose MAX_LENGTH (and a GENA-LM variant) large enough for your sequences:
#   - 2k nucleotides (~333 tokens)  -> MAX_LENGTH=512   (gena-lm-bert-base-t2t)
#   - 4k nucleotides (~667 tokens)  -> MAX_LENGTH=1024  (needs gena-lm-bigbird-base-t2t)
#   - 8k nucleotides (~1333 tokens) -> MAX_LENGTH=2048  (needs gena-lm-bigbird-base-t2t)
# BERT variants are capped at 512; switch MODEL_NAME to a BigBird variant for longer contexts.
MAX_LENGTH="512"

# === Hardware Configuration ===
# Set based on your GPU:
#   - "A100" or "H100": Use bf16, larger batches
#   - "V100" or "older": Use fp16, smaller batches
GPU_TYPE="A100"

# === Training Configuration ===
LEARNING_RATE="3e-5"
NUM_EPOCHS="10"  # Use more epochs with early stopping
EARLY_STOPPING_PATIENCE="3"

# === Evaluation Configuration ===
# "steps" enables early stopping within epochs
# "epoch" only evaluates at end of each epoch
EVAL_STRATEGY="steps"
EVAL_STEPS="500"

#####################################################################
# END CONFIGURATION
#####################################################################

# Get script directory
SCRIPT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/GENA_LM/GENA_LM_generic_sequence_classification/slurm_scripts"
TRAIN_SCRIPT="${SCRIPT_DIR}/run_optimized_train.sh"

# Parse command line arguments
SEED=$1

if [ -z "${SEED}" ]; then
    echo "ERROR: SEED is required as first argument"
    echo "Usage: bash wrapper_run_training_optimized.sh <SEED>"
    echo "Example: bash wrapper_run_training_optimized.sh 42"
    exit 1
fi

# Validate configuration
if [ ! -d "${DATASET_DIR}" ]; then
    echo "ERROR: Dataset directory not found: ${DATASET_DIR}"
    exit 1
fi

if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "ERROR: Training script not found: ${TRAIN_SCRIPT}"
    exit 1
fi

# Create run-specific output directory to avoid overwrites
# Format: gena_lm_lambda_filtered_{seq_len}_{seed}_{lr}_{timestamp}
OUTPUT_DIR_BASE="${OUTPUT_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Determine sequence length label for directory name
case "${MAX_LENGTH}" in
    512)  SEQ_LEN="2k" ;;
    1024) SEQ_LEN="4k" ;;
    2048) SEQ_LEN="8k" ;;
esac

RUN_NAME="gena_lm_lambda_filtered_${SEQ_LEN}_${SEED}_${LEARNING_RATE}_${TIMESTAMP}"
OUTPUT_DIR="${OUTPUT_DIR_BASE}/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

# Calculate effective batch size for display
case "${MAX_LENGTH}" in
    512)
        if [ "${GPU_TYPE}" == "A100" ] || [ "${GPU_TYPE}" == "H100" ]; then
            BATCH_SIZE="8"; GRAD_ACCUM="1"
        else
            BATCH_SIZE="4"; GRAD_ACCUM="2"
        fi
        ;;
    1024|2048)
        BATCH_SIZE="1"; GRAD_ACCUM="1"
        ;;
esac
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM))

# Set precision for display
if [ "${GPU_TYPE}" == "A100" ] || [ "${GPU_TYPE}" == "H100" ]; then
    PRECISION="bf16 + tf32"
else
    PRECISION="fp16"
fi

echo "=========================================="
echo "Submitting GENA-LM Optimized Training Job"
echo "=========================================="
echo ""
echo "Dataset: ${DATASET_DIR}"
echo "Output base: ${OUTPUT_DIR_BASE}"
echo "Run name: ${RUN_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Configuration:"
echo "  Max length: ${MAX_LENGTH} tokens"
echo "  GPU type: ${GPU_TYPE}"
echo "  Seed: ${SEED}"
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
echo "  Precision: ${PRECISION}"
echo ""
echo "Evaluation:"
echo "  Strategy: ${EVAL_STRATEGY}"
if [ "${EVAL_STRATEGY}" == "steps" ]; then
    echo "  Eval steps: ${EVAL_STEPS}"
fi
echo "=========================================="
echo ""

# Submit SLURM job
JOB_ID=$(sbatch \
    --job-name="nt_train_s${SEED}" \
    --output="${OUTPUT_DIR}/slurm_train_s${SEED}_%j.out" \
    --error="${OUTPUT_DIR}/slurm_train_s${SEED}_%j.err" \
    --export=ALL,DATASET_DIR="${DATASET_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",MAX_LENGTH="${MAX_LENGTH}",GPU_TYPE="${GPU_TYPE}",LEARNING_RATE="${LEARNING_RATE}",NUM_EPOCHS="${NUM_EPOCHS}",EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE}",EVAL_STRATEGY="${EVAL_STRATEGY}",EVAL_STEPS="${EVAL_STEPS}",SEED="${SEED}" \
    "${TRAIN_SCRIPT}" | awk '{print $NF}')

echo "=========================================="
echo "Job Submitted"
echo "=========================================="
echo "Job ID: ${JOB_ID}"
echo "Job name: nt_train_s${SEED}"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Output log: ${OUTPUT_DIR}/slurm_train_s${SEED}_${JOB_ID}.out"
echo "Error log: ${OUTPUT_DIR}/slurm_train_s${SEED}_${JOB_ID}.err"
echo "=========================================="
