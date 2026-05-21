#!/bin/bash

# Wrapper script for benchmarking GENA-LM inference optimizations
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash wrapper_benchmark_inference.sh
#
# This script runs inference with different optimization settings
# and compares throughput, memory usage, and timing.

#####################################################################
# CONFIGURATION - Edit this section
#####################################################################

# === REQUIRED: Input File ===
# Path to a test CSV file with sequences (use a representative sample)
INPUT_CSV="/data/lindseylm/GLM_EVALUATIONS/MODELS/GENA-LM/GENA_LM_generic_sequence_classification/test_data/benchmark_sample.csv"

# === REQUIRED: Model Configuration ===
# Path to fine-tuned model directory
MODEL_PATH="/data/lindseylm/GLM_EVALUATIONS/MODELS/GENA-LM/GENA_LM_generic_sequence_classification/output/filtered/2k/gena_lm_lambda_filtered_2k_8_3e-5_20260120_063339/checkpoint-40995"

# === OPTIONAL: Inference Parameters ===
BATCH_SIZE="16"
MAX_LENGTH="512"

# === BENCHMARK OPTIONS ===
# Set to "true" to include that configuration in the benchmark
RUN_FP32="true"      # Baseline (no optimization)
RUN_BF16="true"      # bfloat16 mixed precision
RUN_FP16="true"      # float16 mixed precision

# Number of warmup runs before timing (recommended: 1)
WARMUP_RUNS="1"

# Number of timed runs to average (recommended: 3)
TIMED_RUNS="3"

#####################################################################
# END CONFIGURATION
#####################################################################

# Get script directory and project root
SCRIPT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/GENA-LM/GENA_LM_generic_sequence_classification/slurm_scripts"
PROJECT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/GENA-LM/GENA_LM_generic_sequence_classification"

# Validate configuration
if [ ! -f "${INPUT_CSV}" ]; then
    echo "ERROR: Input CSV not found: ${INPUT_CSV}"
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model directory not found: ${MODEL_PATH}"
    exit 1
fi

# Create temp directory for benchmark outputs
BENCHMARK_DIR="${PROJECT_DIR}/benchmark_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BENCHMARK_DIR}"

echo "=========================================="
echo "GENA-LM Inference Optimization Benchmark"
echo "=========================================="
echo "Input CSV: ${INPUT_CSV}"
echo "Model: ${MODEL_PATH}"
echo "Batch size: ${BATCH_SIZE}"
echo "Max length: ${MAX_LENGTH}"
echo "Output dir: ${BENCHMARK_DIR}"
echo "=========================================="
echo ""

# Function to run a single benchmark
run_benchmark() {
    local name="$1"
    local extra_args="$2"
    local output_csv="${BENCHMARK_DIR}/${name}_predictions.csv"

    echo "----------------------------------------"
    echo "Running: ${name}"
    echo "Args: ${extra_args}"
    echo "----------------------------------------"

    # Warmup runs
    for i in $(seq 1 ${WARMUP_RUNS}); do
        echo "  Warmup run ${i}/${WARMUP_RUNS}..."
        python "${PROJECT_DIR}/inference_gena_lm.py" \
            --input_csv "${INPUT_CSV}" \
            --model_path "${MODEL_PATH}" \
            --output_csv "/tmp/warmup_${name}.csv" \
            --batch_size "${BATCH_SIZE}" \
            --max_length "${MAX_LENGTH}" \
            ${extra_args} \
            > /dev/null 2>&1
    done

    # Timed runs
    echo "  Running ${TIMED_RUNS} timed runs..."
    for i in $(seq 1 ${TIMED_RUNS}); do
        echo "  Timed run ${i}/${TIMED_RUNS}..."
        python "${PROJECT_DIR}/inference_gena_lm.py" \
            --input_csv "${INPUT_CSV}" \
            --model_path "${MODEL_PATH}" \
            --output_csv "${output_csv}" \
            --batch_size "${BATCH_SIZE}" \
            --max_length "${MAX_LENGTH}" \
            ${extra_args} \
            2>&1 | tee -a "${BENCHMARK_DIR}/${name}_run${i}.log"
        echo ""
    done
}

# Run benchmarks
if [ "${RUN_FP32}" == "true" ]; then
    run_benchmark "fp32_baseline" ""
fi

if [ "${RUN_BF16}" == "true" ]; then
    run_benchmark "bf16" "--bf16"
fi

if [ "${RUN_FP16}" == "true" ]; then
    run_benchmark "fp16" "--fp16"
fi

# Summary
echo ""
echo "=========================================="
echo "Benchmark Complete"
echo "=========================================="
echo "Results saved to: ${BENCHMARK_DIR}"
echo ""
echo "To compare results, look at the throughput and memory usage in each log:"
echo "  grep -E '(Throughput|Peak GPU memory|Precision)' ${BENCHMARK_DIR}/*.log"
echo ""
