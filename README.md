# GENA-LM Generic Sequence Classification

> **Fork of [AIRI-Institute/GENA_LM](https://github.com/AIRI-Institute/GENA_LM)** — adds generic CSV-based binary classification scripts, used to benchmark GENA-LM on the [LAMBDA prophage-detection benchmark](https://github.com/leannmlindsey/LAMBDA).
>
> Original docs preserved verbatim in [`UPSTREAM_README.md`](./UPSTREAM_README.md).

---

## Relationship to the upstream training code

The fine-tune script in this fork (`finetune_gena_lm_phage.py`) is a thin
wrapper around `transformers.Trainer` with `AutoModelForSequenceClassification`
— the HF Trainer path the upstream README sanctions as an alternative to the
custom `lm_experiments_tools.Trainer` + Horovod loop used by the per-task
scripts under [`downstream_tasks/`](./downstream_tasks/). Defaults follow the
upstream GENA-LM BigBird recipe; the modernGENA variant overrides
`learning_rate=3e-5`, `weight_decay=1e-3`,
`lr_scheduler_type=linear` to match the modernGENA reference config. Every CLI
flag can be overridden:

| Parameter | Default (this fork) | Source / rationale |
|-----------|---------------------|--------------------|
| `learning_rate` | 1e-4 | upstream — GENA-LM BigBird recipe |
| `weight_decay` | 0.0 | upstream — BigBird recipe (no L2) |
| `warmup_ratio` | 0.06 | upstream — modernGENA reference default |
| `lr_scheduler_type` | constant_with_warmup | upstream — BigBird recipe |
| `num_train_epochs` | 10 | upstream |
| `per_device_train_batch_size` | 8 | this fork |
| `per_device_eval_batch_size` | 16 | this fork |
| `gradient_accumulation_steps` | 4 | this fork — effective batch 32 |
| `max_length` | 1024 | this fork — matches the 4k window; 512 for 2k (BERT), 2048 for 8k |
| `metric_for_best_model` | `eval_mcc` | **LAMBDA-specific** (the LAMBDA paper reports MCC) |
| `load_best_model_at_end` | True | this fork |
| `early_stopping_patience` | 7 evaluations | upstream — BigBird recipe |
| `save_total_limit` | 1 | this fork |
| `bf16` / `fp16` | opt-in flag | this fork — A100 efficiency |
| `seed` | 42 | HF convention |

`metric_for_best_model` is `eval_mcc` for LAMBDA runs (the LAMBDA paper reports
MCC); `f1` / `mcc` / `pr_auc` / `accuracy` are all computed each eval and only
this flag selects the best checkpoint. The LAMBDA replication picks the best
seed per variant by `eval_mcc`.

The upstream alternative training paths — the per-task scripts under
[`downstream_tasks/`](./downstream_tasks/) (custom Trainer with Horovod for
distributed training and optional RMT memory tokens for long sequences) — are
preserved unchanged. Use them directly for their original training loop.

## What this fork adds

| File | Purpose |
|------|---------|
| `finetune_gena_lm_phage.py` | Fine-tune any GENA-LM / modernGENA checkpoint on a binary CSV dataset (`train.csv` / `dev.csv` (or `val.csv`) / `test.csv` with `sequence,label` columns). |
| `inference_gena_lm.py` | Single-CSV inference — predictions, probabilities, optional metrics. |
| `inference_gena_lm_dir.py` | Directory-mode inference — load the model once, predict over every CSV in a directory. |
| `embedding_analysis_gena_lm.py` | Extract pretrained embeddings; train a linear probe + 3-layer NN; compute silhouette, PCA, and (optionally) a random-init baseline + embedding power. |
| `slurm_scripts/lambda_replication/lambda_replication.conf` | Single config for the LAMBDA replication pipeline (paths, variants, seeds, hyperparameters). |
| `slurm_scripts/lambda_replication/run_lambda_training.sh` | Submit all finetune (variant × seed) + embedding jobs per window. |
| `slurm_scripts/lambda_replication/run_lambda_inference.sh` | Pick the best seed per variant, then submit all diagnostic + genome-wide inference. |
| `slurm_scripts/lambda_replication/lambda_*_job.sh` | SLURM workers: finetune, embedding, single-CSV inference, genome-wide inference. |
| `slurm_scripts/lambda_replication/select_best_seed.py` | Pick the best finetune seed per variant by test-set `eval_mcc`; write `winners.json`. |
| `slurm_scripts/lambda_replication/print_winner_exports.py` | Emit the winning checkpoint path for the inference jobs. |
| `slurm_scripts/lambda_replication/prefetch_models.sh` | Pre-warm the offline HF cache on a login node (compute nodes have no internet). |
| `summarize_inference_results.py` | Aggregate per-CSV `_metrics.json` files into one metrics summary table. |
| `setup_lambda.sh` + `requirements_lambda.txt` | Self-contained `gena_lm` conda env for the LAMBDA scripts, built on upstream's `environment.yml`. |
| `slurm_scripts/wrapper_run_*.sh` | Generic SLURM submission wrappers (finetune, embedding analysis, batch inference). |

## Installation

The fork ships a self-contained env builder; it is the recommended path:

```bash
git clone git@github.com:leannmlindsey/GENA_LM_generic_sequence_classification.git
cd GENA_LM_generic_sequence_classification

# builds the `gena_lm` conda env (the SLURM scripts assume this name)
bash setup_lambda.sh
```

Manual fallback (same env, built by hand):

```bash
conda env create -n gena_lm -f examples/modernGENA/environment.yml
conda activate gena_lm
pip install -r requirements_lambda.txt
```

`environment.yml` pins a CUDA-enabled PyTorch matching upstream's tested stack.
For a different CUDA build, install torch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Using the fork

| If you want to... | Go to |
|---|---|
| Use GENA-LM on **your own** binary classification CSV (finetune, evaluate embeddings, predict) | [Generic classification](#generic-classification) |
| **Replicate** the LAMBDA phage paper — train all variants on the LAMBDA dataset, pick the best seed per variant, run all diagnostic + genome-wide inference | [LAMBDA replication](#lambda-replication) |

### Generic classification

**Inputs:** a directory containing `train.csv`, `dev.csv` (or `val.csv`),
`test.csv`. Each CSV must have a `sequence` column and a `label` column (0/1).

Three sub-steps, each a separate SLURM submission:

```bash
# 1. Embedding analysis — linear probe + 3-layer NN on pretrained embeddings
#    (edit the config block at the top, then run)
bash slurm_scripts/wrapper_run_embedding_analysis.sh

# 2. Fine-tuning — full encoder fine-tune
#    (edit the config block at the top of run_train_gena_lm.sh, then submit)
sbatch slurm_scripts/run_train_gena_lm.sh <SEED> <WINDOW>
#    multi-seed / multi-window / per-variant:
bash slurm_scripts/submit_train_all_windows.sh "<SEEDS>" "<WINDOWS>" <variant>

# 3. Inference — local fine-tuned checkpoint, one job per CSV in INPUT_LIST
bash slurm_scripts/wrapper_run_batch_inference.sh
```

`INPUT_LIST` in the batch-inference wrapper is a text file with one CSV path
per line; one SLURM job per input.

For the full flag list for any script, run `python <script>.py --help`.

### LAMBDA replication

A two-step workflow over a single config file. The pipeline loops over the
LAMBDA_v1 windows (2k / 4k / 8k by default) and for each window submits:
finetune × variants (bigbird, moderngena) × N seeds, embedding analysis ×
variants (on the pretrained base model), automatic best-seed selection (by
test-set `eval_mcc`), inference on the matching-window diagnostics (test, fpr,
gc_control, fnr), and genome-wide inference.

Biowulf compute nodes have no internet, so prefetch the base models into the
offline HF cache from a login node first:

```bash
bash slurm_scripts/lambda_replication/prefetch_models.sh
```

```bash
# 1. Edit the config — LAMBDA_BASE and OUTPUT_DIR are required; VARIANTS, SEEDS,
#    FPR_<W>, GC_<W>, FNR_<W>, GENOME_WIDE_<W> are optional.
$EDITOR slurm_scripts/lambda_replication/lambda_replication.conf

# 2. Launch all training (finetune × N seeds + embedding × variants, per window,
#    in parallel — no dependency chaining)
bash slurm_scripts/lambda_replication/run_lambda_training.sh

# 3. Wait — squeue -u $USER

# 4. Launch all inference (per window: pick the best seed by test-MCC; run
#    inference on test, fpr, gc_control, fnr; run genome-wide inference)
bash slurm_scripts/lambda_replication/run_lambda_inference.sh
```

**Expected LAMBDA_v1 layout** (`LAMBDA_BASE`):

```
LAMBDA_BASE/
└── train_val_test/<W>/{train,val,test}.csv     finetune + embedding + test diagnostic
```

The fpr / gc_control / fnr diagnostics and genome-wide inputs are provided per
window via the optional `FPR_<W>` / `GC_<W>` / `FNR_<W>` / `GENOME_WIDE_<W>`
config variables (`GENOME_WIDE_<W>` is a directory of segment CSVs; its outputs
are renamed `genome_wide_<asm>_*_predictions.csv`). Any unset diagnostic is
skipped with a warning.

**Output layout:**

```
<OUTPUT_DIR>/
├── <W>/                                one subdir per window
│   ├── finetune/<variant>/seed-<N>/    test_results.json + checkpoint
│   ├── embedding/<variant>/            embedding_analysis_results.json, .npz, classifiers
│   ├── winners.json                    best seed per variant (by eval_mcc)
│   └── inference/<variant>/            {test,fpr,gc_control,fnr}_predictions.csv (+ _metrics.json)
│                                       + genome_wide_<asm>_*_predictions.csv
└── logs/                               SLURM stdout/stderr per job (shared)
```

## Available models

All variants share the same 32k BPE tokenizer (~6 bp/token).

| Model | Architecture | Context | HuggingFace |
| --- | --- | --- | --- |
| GENA-LM BERT base | BERT | 512 tokens (~3 kb) | [AIRI-Institute/gena-lm-bert-base-t2t](https://huggingface.co/AIRI-Institute/gena-lm-bert-base-t2t) |
| GENA-LM BERT large | BERT-large | 512 tokens (~3 kb) | [AIRI-Institute/gena-lm-bert-large-t2t](https://huggingface.co/AIRI-Institute/gena-lm-bert-large-t2t) |
| GENA-LM BigBird base | BigBird | 4096 tokens (~24 kb) | [AIRI-Institute/gena-lm-bigbird-base-t2t](https://huggingface.co/AIRI-Institute/gena-lm-bigbird-base-t2t) |
| GENA-LM BigBird sparse base | BigBird (sparse; needs DeepSpeed) | 4096 tokens (~24 kb) | [AIRI-Institute/gena-lm-bigbird-base-sparse-t2t](https://huggingface.co/AIRI-Institute/gena-lm-bigbird-base-sparse-t2t) |
| modernGENA base | ModernBERT | long context (see upstream) | [AIRI-Institute/moderngena-base](https://huggingface.co/AIRI-Institute/moderngena-base) |
| modernGENA large | ModernBERT-large | long context (see upstream) | [AIRI-Institute/moderngena-large](https://huggingface.co/AIRI-Institute/moderngena-large) |

The LAMBDA replication wrappers use three variant presets: `bigbird`
(`gena-lm-bigbird-base-t2t`), `moderngena` (`moderngena-base`), and `bertbase`
(`gena-lm-bert-base-t2t`).

## Citation

If you use GENA-LM itself, cite the original paper:

```bibtex
@article{GENA_LM,
    author  = {Fishman, Veniamin and Kuratov, Yuri and Shmelev, Aleksei and Petrov, Maxim and Penzar, Dmitry and Shepelin, Denis and Chekanov, Nikolay and Kardymon, Olga and Burtsev, Mikhail},
    title   = {GENA-LM: a family of open-source foundational DNA language models for long sequences},
    journal = {Nucleic Acids Research},
    volume  = {53},
    number  = {2},
    pages   = {gkae1310},
    year    = {2025},
    issn    = {0305-1048},
    doi     = {10.1093/nar/gkae1310},
    url     = {https://doi.org/10.1093/nar/gkae1310}
}
```

For modernGENA variants, also cite [Back to BERT in 2026: ModernGENA as a
Strong, Efficient Baseline for DNA Foundation Models](https://www.biorxiv.org/content/10.64898/2026.04.21.719816v1).

If you use this fork as part of the LAMBDA prophage-detection benchmark, also
cite the LAMBDA paper:

```bibtex
@article{LAMBDA2026,
  author  = {Lindsey, LeAnn M. and Pershing, Nicole L. and Dufault-Thompson, Keith and Gwak, Ho-jin and Habib, Anisa and Schindler, Aaron and Rakheja, Arjun and Round, June and Stephens, W. Zac and Blaschke, Anne J. and Sundar, Hari and Jiang, Xiaofang},
  title   = {{LAMBDA}: A Prophage Detection Benchmark for Genomic Language Models},
  year    = {2026},
  doi     = {10.64898/2026.03.26.714501},
  url     = {https://doi.org/10.64898/2026.03.26.714501}
}
```
