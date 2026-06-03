# GENA-LM LAMBDA_v1 replication

Two-command Biowulf pipeline that mirrors the ProkBERT `lambda_replication`
layout for GENA-LM. It reuses the **unchanged** GENA-LM Python entry points
(`finetune_gena_lm_phage.py`, `embedding_analysis_gena_lm.py`,
`inference_gena_lm.py`, `inference_gena_lm_dir.py`); the only substantive change
from the previous LAMBDA runs is the **input dataset**, which now reads
`LAMBDA_v1/train_val_test/<W>/{train,val,test}.csv` instead of the old v0
`merged_datasets_filtered/<W>/` path.

## Files

| file | role |
|------|------|
| `lambda_replication.conf`        | all paths + hyperparameters (edit this) |
| `run_lambda_training.sh`         | Stage 1 — submit finetune (variant×seed) + embedding jobs |
| `run_lambda_inference.sh`        | Stage 2 — pick winners + submit inference jobs |
| `lambda_finetune_job.sh`         | worker: one finetune seed → `finetune_gena_lm_phage.py` |
| `lambda_embedding_job.sh`        | worker: Surface D embedding (pretrained base) |
| `lambda_inference_job.sh`        | worker: Surface A/B single-CSV inference |
| `lambda_genome_inference_job.sh` | worker: Surface C genome-wide dir inference |
| `select_best_seed.py`            | writes `winners.json` (best seed per variant by `eval_mcc`) |
| `print_winner_exports.py`        | reads `winners.json[variant]` for the inference jobs |

## Run

```bash
# 0. (once, from a LOGIN node) pre-warm the offline HF cache — see the bottom
#    of lambda_replication.conf for the snippet.

# 1. edit lambda_replication.conf  (LAMBDA_BASE, OUTPUT_DIR, the FPR_/GC_/FNR_/
#    GENOME_WIDE_ diagnostic paths, VARIANTS, SEEDS)

# 2. Stage 1 — training + embedding
bash slurm_scripts/lambda_replication/run_lambda_training.sh

# 3. wait for all jobs
squeue -u $USER

# 4. Stage 2 — winners + inference
bash slurm_scripts/lambda_replication/run_lambda_inference.sh
```

## Output layout (per window `<W>` ∈ {2k,4k,8k})

```
$OUTPUT_DIR/<W>/finetune/<variant>/seed-<N>/   test_results.json + checkpoint
$OUTPUT_DIR/<W>/embedding/<variant>/           embedding_analysis_results.json + .npz + .pkl
$OUTPUT_DIR/<W>/winners.json                   best seed per variant (eval_mcc)
$OUTPUT_DIR/<W>/inference/<variant>/           test_predictions.csv
                                               fpr_predictions.csv
                                               gc_control_predictions.csv
                                               fnr_predictions.csv
                                               genome_wide_<asm>_*_predictions.csv
$OUTPUT_DIR/logs/                              SLURM stdout/stderr (shared)
```

`$OUTPUT_DIR` =
`/data/lindseylm/GLM_EVALUATIONS/NAR_GENOMICS_LAMBDA_REPO/GENA_LM_generic_sequence_classification/outputs`

## Variants

Presets are copied verbatim from `submit_train_all_windows.sh`:

| variant | model | LR | WD | scheduler |
|---------|-------|----|----|-----------|
| `bigbird`    | `AIRI-Institute/gena-lm-bigbird-base-t2t` | 1e-4 | 0    | constant_with_warmup |
| `moderngena` | `AIRI-Institute/moderngena-base`          | 3e-5 | 1e-3 | linear |
| `bertbase`   | `AIRI-Institute/gena-lm-bert-base-t2t`    | 1e-4 | 0    | constant_with_warmup |

`VARIANTS` defaults to `bigbird moderngena` (the two you ran full sweeps for).
Add `bertbase` to `VARIANTS` in the conf to include it (caps at 512 tok, so
4k/8k truncate).

## Smoke test (one seed, one variant, one window)

```bash
cd slurm_scripts/lambda_replication
cp lambda_replication.conf lambda_replication.conf.bak
sed -i 's/^export WINDOWS=.*/export WINDOWS="2k"/'            lambda_replication.conf
sed -i 's/^export VARIANTS=.*/export VARIANTS="bigbird"/'     lambda_replication.conf
sed -i 's/^export SEEDS=.*/export SEEDS="1"/'                 lambda_replication.conf
bash run_lambda_training.sh        # submits 1 finetune + 1 embedding job
# after it finishes:
ALLOW_PARTIAL_TRAINING=true bash run_lambda_inference.sh
# restore the full sweep config:
mv lambda_replication.conf.bak lambda_replication.conf
```

Expected after the smoke test:
`outputs/2k/finetune/bigbird/seed-1/test_results.json`,
`outputs/2k/embedding/bigbird/embedding_analysis_results.json`,
`outputs/2k/winners.json`, and
`outputs/2k/inference/bigbird/test_predictions.csv`.
