#!/bin/bash
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=gena_ft_gw
#SBATCH --output=gena_ft_gw_%j.log
#SBATCH --error=gena_ft_gw_%j.log
#
# GENA-LM FT genome-wide — the FT head was never deployed genome-wide for these
# (a probe won). Deploys each variant's BEST fine-tuned seed across the genome-wide
# CSVs via the EXISTING inference_gena_lm.py, into a dedicated dir (no overwrite).
# bigbird needs all 3 windows; moderngena only 4k (its 2k/8k already deploy FT).
# Best seeds + eval-MCC fingerprints (verify on Delta vs each seed's results):
#   bigbird 2k -> seed 1 (0.6568)   4k -> seed 3 (0.7009)   8k -> seed 2 (0.7386)
#   moderngena 4k -> seed 1 (0.8388)
#
# Submit FROM the GENA-LM repo root on Delta (job self-activates gena_lm):
#   sbatch slurm_scripts/lambda_replication/run_gena_ft_gw.sbatch
set -uo pipefail

source /u/llindsey1/miniconda3/etc/profile.d/conda.sh
conda activate gena_lm
echo "conda env: ${CONDA_DEFAULT_ENV:-none}   python: $(command -v python)  node $(hostname)  $(date)"

export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-/work/hdd/bfzj/llindsey1/hf_cache}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"; export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

OUTROOT=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/GENA_LM_generic_sequence_classification/outputs
GWROOT=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/LAMBDA_v1/genome_wide
maxlen_for () { case "$1" in 2k) echo 512;; 4k) echo 1024;; 8k) echo 2048;; esac; }

# variant:window:seed  (bigbird all windows; moderngena 4k only)
for job in bigbird:2k:1 bigbird:4k:3 bigbird:8k:2 moderngena:4k:1; do
  VARIANT="${job%%:*}"; rest="${job#*:}"; w="${rest%%:*}"; seed="${rest#*:}"
  ML=$(maxlen_for "$w")
  CKPT="$OUTROOT/$w/finetune/$VARIANT/seed-$seed"
  OUT="$OUTROOT/$w/genome_wide_ft/$VARIANT"; mkdir -p "$OUT"
  [ -d "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
  echo "===== $VARIANT $w  seed $seed  max_length $ML  ckpt $CKPT ====="
  n=0
  for csv in "$GWROOT/$w"/*.csv; do
    stem=$(basename "$csv" .csv)
    python inference_gena_lm.py \
      --input_csv "$csv" \
      --model_path "$CKPT" \
      --output_csv "$OUT/genome_wide_${stem}_predictions.csv" \
      --batch_size 16 --max_length $ML --threshold 0.5 --save_metrics
    n=$((n+1))
  done
  echo "  $VARIANT $w done: $n genome(s) -> $OUT"
done
echo "DONE $(date)"
