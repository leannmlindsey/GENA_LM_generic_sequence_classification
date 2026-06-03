#!/bin/bash
#SBATCH --job-name=gena_lm_benchmark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=nt_benchmark_%j.out
#SBATCH --error=nt_benchmark_%j.err

# Benchmark script to compare fp32 vs fp16 training
# Uses EXACT same parameters as original run_train_ntv2.sh, only varying precision
#
# Usage: sbatch run_train_benchmark.sh

echo "============================================================"
echo "GENA-LM Training Benchmark"
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

# ============================================================
# CONFIGURATION - Matches original run_train_ntv2.sh exactly
# ============================================================

MODEL_NAME="AIRI-Institute/gena-lm-bert-base-t2t"

# Small benchmark dataset
DATASET_DIR="/home/lindseylm/lindseylm/lambda_final/merged_datasets_filtered/small"

# Training parameters - SAME as original
SEED=42
LEARNING_RATE=3e-5
BATCH_SIZE=1
EPOCHS=3
MAX_LENGTH=512  # For 2k sequences: 2000/6 = ~341 tokens

# Base output directory
OUTPUT_BASE="/data/lindseylm/GLM_EVALUATIONS/MODELS/GENA-LM/GENA_LM_generic_sequence_classification/output/benchmark"
SCRIPT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/GENA-LM/GENA_LM_generic_sequence_classification"

# Create output directory
mkdir -p "${OUTPUT_BASE}"

# Results file
RESULTS_FILE="${OUTPUT_BASE}/benchmark_results_$(date +%Y%m%d_%H%M%S).txt"

echo ""
echo "============================================================"
echo "Benchmark Configuration (matches original script)"
echo "============================================================"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_DIR"
echo "  Output base: $OUTPUT_BASE"
echo "  Max length: $MAX_LENGTH tokens"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Seed: $SEED"
echo "============================================================"
echo ""

# ============================================================
# TEST 1: Baseline (fp32) - no mixed precision
# ============================================================

echo "============================================================" | tee -a "${RESULTS_FILE}"
echo "TEST 1: Baseline (fp32) - no mixed precision" | tee -a "${RESULTS_FILE}"
echo "============================================================" | tee -a "${RESULTS_FILE}"

OUTPUT_DIR_BASELINE="${OUTPUT_BASE}/baseline_fp32_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR_BASELINE}"

echo "Output: ${OUTPUT_DIR_BASELINE}" | tee -a "${RESULTS_FILE}"
echo "Started at: $(date)" | tee -a "${RESULTS_FILE}"

START_TIME=$(date +%s)

python "${SCRIPT_DIR}/finetune_gena_lm_phage.py" \
    --model_name "$MODEL_NAME" \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR_BASELINE" \
    --max_length $MAX_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --logging_steps 100 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model eval_mcc \
    --early_stopping_patience 3 \
    --save_total_limit 2 \
    --seed $SEED

EXIT_CODE_BASELINE=$?
END_TIME=$(date +%s)
ELAPSED_BASELINE=$((END_TIME - START_TIME))

echo "Finished at: $(date)" | tee -a "${RESULTS_FILE}"
echo "Exit code: ${EXIT_CODE_BASELINE}" | tee -a "${RESULTS_FILE}"
echo "Time: ${ELAPSED_BASELINE} seconds ($(echo "scale=1; ${ELAPSED_BASELINE}/60" | bc) minutes)" | tee -a "${RESULTS_FILE}"

if [ -f "${OUTPUT_DIR_BASELINE}/training_summary.json" ]; then
    MEMORY_BASELINE=$(python3 -c "import json; d=json.load(open('${OUTPUT_DIR_BASELINE}/training_summary.json')); print(f\"{d.get('peak_gpu_memory_mb', 0):.0f}\")")
    echo "Peak GPU memory: ${MEMORY_BASELINE} MB" | tee -a "${RESULTS_FILE}"
fi

echo "" | tee -a "${RESULTS_FILE}"

# ============================================================
# TEST 2: fp16 (matching original working config)
# ============================================================

echo "============================================================" | tee -a "${RESULTS_FILE}"
echo "TEST 2: fp16 (matching original working config)" | tee -a "${RESULTS_FILE}"
echo "============================================================" | tee -a "${RESULTS_FILE}"

OUTPUT_DIR_FP16="${OUTPUT_BASE}/fp16_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR_FP16}"

echo "Output: ${OUTPUT_DIR_FP16}" | tee -a "${RESULTS_FILE}"
echo "Started at: $(date)" | tee -a "${RESULTS_FILE}"

START_TIME=$(date +%s)

python "${SCRIPT_DIR}/finetune_gena_lm_phage.py" \
    --model_name "$MODEL_NAME" \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR_FP16" \
    --max_length $MAX_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --logging_steps 100 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model eval_mcc \
    --early_stopping_patience 3 \
    --save_total_limit 2 \
    --fp16 \
    --seed $SEED

EXIT_CODE_FP16=$?
END_TIME=$(date +%s)
ELAPSED_FP16=$((END_TIME - START_TIME))

echo "Finished at: $(date)" | tee -a "${RESULTS_FILE}"
echo "Exit code: ${EXIT_CODE_FP16}" | tee -a "${RESULTS_FILE}"
echo "Time: ${ELAPSED_FP16} seconds ($(echo "scale=1; ${ELAPSED_FP16}/60" | bc) minutes)" | tee -a "${RESULTS_FILE}"

if [ -f "${OUTPUT_DIR_FP16}/training_summary.json" ]; then
    MEMORY_FP16=$(python3 -c "import json; d=json.load(open('${OUTPUT_DIR_FP16}/training_summary.json')); print(f\"{d.get('peak_gpu_memory_mb', 0):.0f}\")")
    echo "Peak GPU memory: ${MEMORY_FP16} MB" | tee -a "${RESULTS_FILE}"
fi

echo "" | tee -a "${RESULTS_FILE}"

# ============================================================
# SUMMARY
# ============================================================

echo "============================================================" | tee -a "${RESULTS_FILE}"
echo "BENCHMARK SUMMARY" | tee -a "${RESULTS_FILE}"
echo "============================================================" | tee -a "${RESULTS_FILE}"
echo "" | tee -a "${RESULTS_FILE}"
echo "Configuration: ${EPOCHS} epochs, batch_size=${BATCH_SIZE}, max_length=${MAX_LENGTH}" | tee -a "${RESULTS_FILE}"
echo "" | tee -a "${RESULTS_FILE}"
echo "Timing Results:" | tee -a "${RESULTS_FILE}"
echo "  1. Baseline (fp32): ${ELAPSED_BASELINE} seconds ($(echo "scale=1; ${ELAPSED_BASELINE}/60" | bc) min)" | tee -a "${RESULTS_FILE}"
echo "  2. fp16:            ${ELAPSED_FP16} seconds ($(echo "scale=1; ${ELAPSED_FP16}/60" | bc) min)" | tee -a "${RESULTS_FILE}"

# Calculate speedup
if [ ${ELAPSED_FP16} -gt 0 ] && [ ${ELAPSED_BASELINE} -gt 0 ]; then
    SPEEDUP=$(echo "scale=2; ${ELAPSED_BASELINE} / ${ELAPSED_FP16}" | bc)
    echo "" | tee -a "${RESULTS_FILE}"
    echo "Speedup (fp16 vs fp32): ${SPEEDUP}x" | tee -a "${RESULTS_FILE}"
fi

echo "" | tee -a "${RESULTS_FILE}"
echo "Results saved to: ${RESULTS_FILE}" | tee -a "${RESULTS_FILE}"
echo "============================================================" | tee -a "${RESULTS_FILE}"

echo ""
echo "Job finished at: $(date)"

exit 0
