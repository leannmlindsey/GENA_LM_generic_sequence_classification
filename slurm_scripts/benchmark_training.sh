#!/bin/bash

# Benchmark Training Configurations for GENA-LM
#
# This script runs training with different optimization configurations
# to find the fastest setup for your hardware.
#
# Usage:
#   bash benchmark_training.sh
#
# Results are saved to results/training_benchmark/

#####################################################################
# CONFIGURATION - Edit this section
#####################################################################

# Dataset (use a smaller subset for benchmarking)
DATASET_DIR="/path/to/your/benchmark/data"

# Base output directory
OUTPUT_BASE="/path/to/results/training_benchmark"

# Model
MODEL_NAME="AIRI-Institute/gena-lm-bert-base-t2t"

# Fixed parameters for fair comparison
MAX_LENGTH="512"
LEARNING_RATE="3e-5"
SEED="42"

# Benchmark settings (use fewer epochs for quick comparison)
NUM_EPOCHS="1"
EVAL_STRATEGY="steps"
EVAL_STEPS="100"

# Script directory
SCRIPT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/GENA_LM/GENA_LM_generic_sequence_classification"

#####################################################################
# END CONFIGURATION
#####################################################################

# Create output directory
mkdir -p "${OUTPUT_BASE}"

# Log file
LOG_FILE="${OUTPUT_BASE}/benchmark_results.log"

echo "========================================" | tee "${LOG_FILE}"
echo "GENA-LM Training Benchmark" | tee -a "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Function to run a single benchmark
run_benchmark() {
    local config_name="$1"
    local extra_args="$2"
    local batch_size="$3"

    local output_dir="${OUTPUT_BASE}/${config_name}"
    mkdir -p "${output_dir}"

    echo "----------------------------------------" | tee -a "${LOG_FILE}"
    echo "Config: ${config_name}" | tee -a "${LOG_FILE}"
    echo "Batch size: ${batch_size}" | tee -a "${LOG_FILE}"
    echo "Extra args: ${extra_args}" | tee -a "${LOG_FILE}"
    echo "----------------------------------------" | tee -a "${LOG_FILE}"

    # Record start time
    start_time=$(date +%s)

    # Run training
    python "${SCRIPT_DIR}/finetune_gena_lm_phage.py" \
        --model_name "${MODEL_NAME}" \
        --dataset_dir "${DATASET_DIR}" \
        --output_dir "${output_dir}" \
        --max_length ${MAX_LENGTH} \
        --per_device_train_batch_size ${batch_size} \
        --learning_rate ${LEARNING_RATE} \
        --num_train_epochs ${NUM_EPOCHS} \
        --eval_strategy ${EVAL_STRATEGY} \
        --eval_steps ${EVAL_STEPS} \
        --save_strategy "no" \
        --early_stopping_patience 0 \
        --seed ${SEED} \
        ${extra_args} \
        2>&1 | tee "${output_dir}/training.log"

    # Record end time
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    echo "Time: ${elapsed} seconds ($(echo "scale=1; ${elapsed}/60" | bc) minutes)" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"

    # Extract peak memory from log if available
    if grep -q "Peak GPU memory" "${output_dir}/training.log"; then
        peak_mem=$(grep "Peak GPU memory" "${output_dir}/training.log" | tail -1)
        echo "Memory: ${peak_mem}" | tee -a "${LOG_FILE}"
    fi

    echo "" | tee -a "${LOG_FILE}"
}

# ============================================================
# Run benchmarks
# ============================================================

echo "Running benchmarks..." | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# 1. Baseline (fp32, no optimizations)
run_benchmark "baseline_fp32_bs8" "" 8

# 2. fp16 mixed precision
run_benchmark "fp16_bs8" "--fp16" 8
run_benchmark "fp16_bs16" "--fp16" 16

# 3. bf16 mixed precision (recommended for A100)
run_benchmark "bf16_bs8" "--bf16" 8
run_benchmark "bf16_bs16" "--bf16" 16
run_benchmark "bf16_bs32" "--bf16" 32

# 4. bf16 + gradient accumulation (simulate larger batch)
run_benchmark "bf16_bs8_ga4" "--bf16 --gradient_accumulation_steps 4" 8

# 5. bf16 + gradient checkpointing (for longer sequences)
run_benchmark "bf16_gc_bs8" "--bf16 --gradient_checkpointing" 8
run_benchmark "bf16_gc_bs16" "--bf16 --gradient_checkpointing" 16

# 6. bf16 + fused optimizer
run_benchmark "bf16_fused_bs16" "--bf16 --optim adamw_torch_fused" 16

# 7. bf16 + tf32 (Ampere GPUs)
run_benchmark "bf16_tf32_bs16" "--bf16 --tf32" 16

# 8. All optimizations combined
run_benchmark "all_optimized_bs16" "--bf16 --tf32 --optim adamw_torch_fused" 16

# ============================================================
# Summary
# ============================================================

echo "" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "BENCHMARK SUMMARY" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Parse training_summary.json files to create comparison
echo "Config,Time(s),Time(min),Peak_Memory_MB,Batch_Size" | tee -a "${LOG_FILE}"

for dir in "${OUTPUT_BASE}"/*/; do
    config=$(basename "${dir}")
    summary_file="${dir}training_summary.json"

    if [ -f "${summary_file}" ]; then
        time_s=$(python3 -c "import json; d=json.load(open('${summary_file}')); print(f\"{d.get('training_time_seconds', 0):.1f}\")")
        time_m=$(python3 -c "import json; d=json.load(open('${summary_file}')); print(f\"{d.get('training_time_minutes', 0):.1f}\")")
        mem=$(python3 -c "import json; d=json.load(open('${summary_file}')); print(f\"{d.get('peak_gpu_memory_mb', 0):.0f}\")")
        bs=$(python3 -c "import json; d=json.load(open('${summary_file}')); print(d.get('per_device_train_batch_size', 0))")
        echo "${config},${time_s},${time_m},${mem},${bs}" | tee -a "${LOG_FILE}"
    fi
done

echo "" | tee -a "${LOG_FILE}"
echo "Completed: $(date)" | tee -a "${LOG_FILE}"
echo "Results saved to: ${OUTPUT_BASE}" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
