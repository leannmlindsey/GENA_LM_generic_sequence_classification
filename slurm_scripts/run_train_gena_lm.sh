#!/bin/bash
#SBATCH --job-name=gena_lm_phage
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=gena_lm_phage_%j.out
#SBATCH --error=gena_lm_phage_%j.err

# Biowulf batch script for GENA-LM fine-tuning on the LAMBDA benchmark.
#
# Hyperparameter defaults below mirror the upstream modernGENA reference
# config (examples/modernGENA/sequence_classification/configs/config.yaml)
# so that the fine-tuning recipe matches the upstream-published path. The
# only LAMBDA-specific change is `metric_for_best_model = eval_mcc`, since
# the LAMBDA paper uses MCC as the primary metric. Mixed precision (`--bf16`)
# is enabled by default for A100 efficiency; upstream's config has bf16=false.
#
# Usage: sbatch run_train_gena_lm.sh [SEED]
# Example: sbatch run_train_gena_lm.sh 42

echo "============================================================"
echo "GENA-LM Fine-tuning"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules (suppress errors for non-Biowulf systems)
module load conda 2>/dev/null || true
module load CUDA/12.8 2>/dev/null || true

# Set CUDA_HOME if not set
if [ -z "${CUDA_HOME:-}" ]; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null))) 2>/dev/null || true
fi

# Activate conda environment (tries modern `conda activate` then falls back
# to legacy `source activate`)
conda activate gena_lm 2>/dev/null || source activate gena_lm 2>/dev/null || true

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
# CONFIGURATION - MODIFY THESE AS NEEDED
# ============================================================

# Model — GENA-LM / modernGENA variants:
#   AIRI-Institute/gena-lm-bert-base-t2t           BERT, 512 tokens
#   AIRI-Institute/gena-lm-bert-large-t2t          BERT-large, 512 tokens
#   AIRI-Institute/gena-lm-bigbird-base-t2t        BigBird, 4096 tokens
#   AIRI-Institute/gena-lm-bigbird-base-sparse-t2t BigBird-sparse, 4096 tokens
#   AIRI-Institute/moderngena-base                 ModernBERT, long context
#   AIRI-Institute/moderngena-large                ModernBERT-large, long context
MODEL_NAME="AIRI-Institute/gena-lm-bert-base-t2t"

# Dataset directory — must contain train.csv, dev.csv, test.csv
# Each CSV with columns: sequence, label.
# Use the physical /gpfs path rather than the /home/lindseylm/lindseylm symlink
# (compute nodes resolve the physical path more reliably).
DATASET_DIR="/gpfs/gsfs12/users/Irp-jiang/share/lindseylm/lambda_final/merged_datasets_filtered/4k"

# Seed (overridable via the first sbatch positional arg)
SEED=${1:-42}

# === Hyperparameters (defaults match upstream modernGENA reference config) ===
LEARNING_RATE=3e-5                # upstream default
WEIGHT_DECAY=1e-3                 # upstream default (was 0.01 in HF default)
WARMUP_RATIO=0.06                 # upstream default
LR_SCHEDULER_TYPE=linear          # upstream default
BATCH_SIZE=8                      # upstream default
GRADIENT_ACCUMULATION_STEPS=4     # upstream default (effective batch = 32)
EPOCHS=10                         # upstream default
EARLY_STOPPING_PATIENCE=30        # upstream default
EVAL_STEPS=100                    # upstream default (with --eval_strategy=steps)
SAVE_STEPS=100                    # upstream default
SAVE_TOTAL_LIMIT=2                # upstream default

# Max sequence length in tokens. BERT variants = 512, BigBird variants up to 4096.
# GENA-LM uses 32k BPE (~6 bp/token), so 512 tokens ≈ 3 kb of DNA.
MAX_LENGTH=512

f="filtered"
len="4k"

OUTPUT_DIR="./output/${f}/${len}/gena_lm_lambda_${f}_${len}_${SEED}_${LEARNING_RATE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
SCRIPT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/GENA-LM/GENA_LM_generic_sequence_classification"

# ============================================================
# Print configuration
# ============================================================
echo ""
echo "Training configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Max length (tokens): $MAX_LENGTH"
echo "  Per-device batch size: $BATCH_SIZE  (gradient_accum=$GRADIENT_ACCUMULATION_STEPS → effective batch $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)))"
echo "  Epochs (max): $EPOCHS  (early stopping patience: $EARLY_STOPPING_PATIENCE evals)"
echo "  Learning rate: $LEARNING_RATE  (scheduler: $LR_SCHEDULER_TYPE, warmup: $WARMUP_RATIO)"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Eval/save every: $EVAL_STEPS steps"
echo "  Seed: $SEED"
echo ""

# ============================================================
# Run training
# ============================================================
echo "Working directory: $SCRIPT_DIR"

python $SCRIPT_DIR/finetune_gena_lm_phage.py \
    --model_name "$MODEL_NAME" \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_length $MAX_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --num_train_epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --logging_steps 20 \
    --eval_strategy steps \
    --eval_steps $EVAL_STEPS \
    --save_strategy steps \
    --save_steps $SAVE_STEPS \
    --load_best_model_at_end \
    --metric_for_best_model eval_mcc \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --bf16 \
    --seed $SEED

EXIT_CODE=$?

# ============================================================
# Finish
# ============================================================
echo ""
echo "============================================================"
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "Training failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
