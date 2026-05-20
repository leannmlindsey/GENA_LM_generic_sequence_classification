#!/bin/bash
#SBATCH --job-name=gena_lm_memtest
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=nt_memtest_%j.out
#SBATCH --error=nt_memtest_%j.err

# Memory test for different sequence lengths
# Tests if 4k and 8k sequences fit in GPU memory

echo "============================================================"
echo "GENA-LM Memory Test for Different Sequence Lengths"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

# Load modules
module load conda
module load CUDA/12.8
source activate gena_lm
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

nvidia-smi

SCRIPT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/GENA_LM/GENA_LM_generic_sequence_classification"
OUTPUT_BASE="/data/lindseylm/GLM_EVALUATIONS/MODELS/GENA_LM/GENA_LM_generic_sequence_classification/output/memory_test"
mkdir -p "${OUTPUT_BASE}"

# We'll use the 4k dataset for 4k test, need 8k dataset for 8k test
DATASET_4K="/home/lindseylm/lindseylm/lambda_final/merged_datasets_filtered/4k"
DATASET_8K="/home/lindseylm/lindseylm/lambda_final/merged_datasets_filtered/8k"

echo ""
echo "============================================================"
echo "TEST 1: 4k sequences, batch=1, bf16"
echo "============================================================"
echo "Token count: ~667 tokens (4000/6)"

if [ -d "${DATASET_4K}" ]; then
    python "${SCRIPT_DIR}/finetune_gena_lm_phage.py" \
        --dataset_dir "${DATASET_4K}" \
        --output_dir "${OUTPUT_BASE}/test_4k_bf16" \
        --max_length 1024 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --num_train_epochs 1 \
        --logging_steps 10 \
        --eval_strategy no \
        --save_strategy no \
        --early_stopping_patience 0 \
        --bf16 \
        --seed 42 \
        --dataloader_num_workers 2 \
        2>&1 | head -100

    echo "Exit code: $?"
    echo "Peak GPU memory:"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
else
    echo "Dataset not found: ${DATASET_4K}"
fi

echo ""
echo "============================================================"
echo "TEST 2: 8k sequences, batch=1, bf16"
echo "============================================================"
echo "Token count: ~1333 tokens (8000/6)"
echo "NOTE: Gradient checkpointing NOT supported by GENA-LM/ESM"

if [ -d "${DATASET_8K}" ]; then
    python "${SCRIPT_DIR}/finetune_gena_lm_phage.py" \
        --dataset_dir "${DATASET_8K}" \
        --output_dir "${OUTPUT_BASE}/test_8k_bf16" \
        --max_length 2048 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --num_train_epochs 1 \
        --logging_steps 10 \
        --eval_strategy no \
        --save_strategy no \
        --early_stopping_patience 0 \
        --bf16 \
        --seed 42 \
        --dataloader_num_workers 2 \
        2>&1 | head -100

    echo "Exit code: $?"
    echo "Peak GPU memory:"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
else
    echo "Dataset not found: ${DATASET_8K}"
    echo "Please create or specify the correct path for 8k dataset"
fi

echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Check output above for OOM errors and memory usage"
echo ""
echo "NOTE: GENA-LM/ESM does NOT support gradient checkpointing!"
echo ""
echo "If 8k fails (OOM), options are:"
echo "  - DeepSpeed ZeRO (offload optimizer/params to CPU)"
echo "  - Multiple GPUs with FSDP or tensor parallelism"
echo "  - Sequence chunking (split 8k into overlapping 4k segments)"
echo "============================================================"
echo "Job finished at: $(date)"
