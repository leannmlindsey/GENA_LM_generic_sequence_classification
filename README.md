# GENA-LM Generic Sequence Classification

> **Note:** This is a fork of [AIRI-Institute/GENA_LM](https://github.com/AIRI-Institute/GENA_LM) extended with scripts for **generic CSV-based binary classification**, suitable for benchmarking GENA-LM / modernGENA on the [LAMBDA prophage-detection benchmark](https://github.com/leannmlindsey/LAMBDA) or any other binary DNA sequence classification task. The original modernGENA documentation lives in [`UPSTREAM_README.md`](./UPSTREAM_README.md); the previous-generation GENA-LM docs are in [`README_previous_generation.md`](./README_previous_generation.md).

---

## Relationship to the upstream training code

The fine-tune script in this fork (`finetune_gena_lm_phage.py`) is a thin
wrapper around `transformers.Trainer` with `AutoModelForSequenceClassification`
— **the same machinery** the upstream modernGENA reference path uses
(`examples/modernGENA/sequence_classification/train.py`). Hyperparameter
defaults are taken from upstream's reference config
([`examples/modernGENA/sequence_classification/configs/config.yaml`](./examples/modernGENA/sequence_classification/configs/config.yaml)):

| Parameter | Default | Source |
|-----------|---------|--------|
| `learning_rate` | 3e-5 | upstream |
| `weight_decay` | 1e-3 | upstream |
| `warmup_ratio` | 0.06 | upstream |
| `lr_scheduler_type` | linear | upstream |
| `per_device_train_batch_size` | 8 | upstream |
| `gradient_accumulation_steps` | 4 | upstream (effective batch 32) |
| `num_train_epochs` | 10 | upstream |
| `early_stopping_patience` | 30 evaluations | upstream |
| `metric_for_best_model` | `eval_mcc` | **LAMBDA-specific** (the LAMBDA paper reports MCC) |

The two intentional deviations are (1) the best-model selection metric (MCC
rather than upstream's PR-AUC, because the LAMBDA paper uses MCC) and (2)
mixed precision (`--bf16` on for A100 efficiency; upstream defaults to fp32).

The upstream alternative training paths — the per-task scripts under
[`downstream_tasks/`](./downstream_tasks/) (which use a custom
`lm_experiments_tools.Trainer` with Horovod for distributed training and
optional RMT memory tokens for long sequences) — are preserved unchanged.
Use them directly if you want their custom training loop instead of the
HF `Trainer` path used here.

## What this fork adds

| File | Purpose |
|------|---------|
| `finetune_gena_lm_phage.py` | Fine-tune any GENA-LM / modernGENA checkpoint on a binary CSV dataset (`train.csv`/`dev.csv`/`test.csv` with `sequence,label` columns). |
| `inference_gena_lm.py` | Single-CSV inference — predictions + probabilities + (optional) metrics. |
| `inference_gena_lm_dir.py` | Directory-mode inference — loads the model once and processes every CSV in a directory. |
| `embedding_analysis_gena_lm.py` | Extract embeddings; train a linear probe and 3-layer NN; compute silhouette score, PCA, and (optionally) a random-init baseline. |
| `summarize_inference_results.py` | Aggregate per-CSV `_metrics.json` files into a single metrics summary table. |
| `setup_lambda.sh` + `requirements_lambda.txt` | Self-contained conda env for the LAMBDA-evaluation scripts (does not touch the upstream `requirements.txt`). |
| `slurm_scripts/` | SLURM submission scripts for Biowulf / SLURM clusters (training, inference, embedding analysis). |

## Supported models

This fork works with any HuggingFace `AutoModelForSequenceClassification`-compatible checkpoint, including:

| Model | Hugging Face | Context |
|-------|--------------|---------|
| GENA-LM BERT base | [`AIRI-Institute/gena-lm-bert-base-t2t`](https://huggingface.co/AIRI-Institute/gena-lm-bert-base-t2t) | 512 tokens (~3 kb) |
| GENA-LM BERT large | [`AIRI-Institute/gena-lm-bert-large-t2t`](https://huggingface.co/AIRI-Institute/gena-lm-bert-large-t2t) | 512 tokens (~3 kb) |
| GENA-LM BigBird base | [`AIRI-Institute/gena-lm-bigbird-base-t2t`](https://huggingface.co/AIRI-Institute/gena-lm-bigbird-base-t2t) | 4096 tokens (~24 kb) |
| GENA-LM BigBird sparse base | [`AIRI-Institute/gena-lm-bigbird-base-sparse-t2t`](https://huggingface.co/AIRI-Institute/gena-lm-bigbird-base-sparse-t2t) | 4096 tokens (~24 kb) |
| modernGENA base | [`AIRI-Institute/moderngena-base`](https://huggingface.co/AIRI-Institute/moderngena-base) | Long context (see upstream) |
| modernGENA large | [`AIRI-Institute/moderngena-large`](https://huggingface.co/AIRI-Institute/moderngena-large) | Long context (see upstream) |

**Note on context length:** all models in this family use the same 32k BPE tokenizer (~6 bp/token). The default `--max_length 512` works for 2 kb sequences with the BERT variants. For 4 kb / 8 kb sequences, use a BigBird variant and `--max_length 1024` or `2048`.

---

## Quick Start

### 1. Setup environment

```bash
git clone https://github.com/leannmlindsey/GENA_LM_generic_sequence_classification
cd GENA_LM_generic_sequence_classification

bash setup_lambda.sh
# or manually:
conda create -n gena_lm python=3.10 -y
conda activate gena_lm
pip install -r requirements_lambda.txt
```

### 2. Prepare your data

A directory containing three CSV files:

```
my_dataset/
├── train.csv
├── dev.csv     # or val.csv
└── test.csv
```

Each CSV must have:

```csv
sequence,label
ACGTACGT...,0
TGCATGCA...,1
```

- `sequence`: DNA (A/C/G/T/N)
- `label`: integer (0 or 1 for binary classification)

For the LAMBDA benchmark, these CSVs are the `binary_segments_2k/`, `binary_segments_4k/`, and `binary_segments_8k/` subdirectories of the [Zenodo deposit](https://doi.org/10.5281/zenodo.19236553).

---

## Fine-tuning

```bash
python finetune_gena_lm_phage.py \
    --model_name AIRI-Institute/gena-lm-bert-base-t2t \
    --dataset_dir /path/to/LAMBDA/binary_segments_2k \
    --output_dir ./output/gena_lm/2k \
    --max_length 512 \
    --per_device_train_batch_size 16 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --bf16 \
    --early_stopping_patience 3
```

### Long sequences (4 kb / 8 kb)

Use a BigBird variant and gradient checkpointing:

```bash
python finetune_gena_lm_phage.py \
    --model_name AIRI-Institute/gena-lm-bigbird-base-t2t \
    --dataset_dir /path/to/LAMBDA/binary_segments_8k \
    --output_dir ./output/gena_lm_bigbird/8k \
    --max_length 2048 \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4
```

### SLURM (Biowulf)

```bash
# Edit configuration block at the top of slurm_scripts/run_train_gena_lm.sh
sbatch slurm_scripts/run_train_gena_lm.sh
```

After training, `test_results.json` is saved with `eval_accuracy`, `eval_precision`, `eval_recall`, `eval_f1`, `eval_mcc`, `eval_sensitivity`, `eval_specificity`, `eval_auc`.

---

## Inference

### Single CSV

```bash
python inference_gena_lm.py \
    --input_csv /path/to/test.csv \
    --model_path ./output/gena_lm/2k \
    --output_csv predictions.csv \
    --threshold 0.5 \
    --bf16 \
    --save_metrics
```

Outputs `predictions.csv` with `prob_0`, `prob_1`, `pred_label` columns appended. If `--save_metrics` is set and the input has a `label` column, a `_metrics.json` is written alongside.

### Directory mode (recommended for batch)

```bash
python inference_gena_lm_dir.py \
    --input_dir /path/to/csvs \
    --output_dir /path/to/predictions \
    --model_path ./output/gena_lm/2k \
    --bf16 \
    --save_metrics
```

Loads the model once and processes every CSV in `input_dir` — much faster than launching one job per file.

### SLURM batch inference

```bash
# Edit slurm_scripts/wrapper_run_batch_inference.sh, then:
bash slurm_scripts/wrapper_run_batch_inference.sh
```

This submits one SLURM job per input CSV path listed in `INPUT_LIST` (a text file with one path per line).

---

## Embedding analysis

Extract embeddings, train a linear probe + 3-layer NN, compute silhouette and PCA, and (optionally) compare against a randomly initialized baseline:

```bash
python embedding_analysis_gena_lm.py \
    --csv_dir /path/to/csv/data \
    --model_path AIRI-Institute/gena-lm-bert-base-t2t \
    --output_dir ./results/embedding_analysis \
    --pooling mean \
    --include_random_baseline
```

Outputs (in `--output_dir`):

| File | Description |
|------|-------------|
| `embeddings_pretrained.npz` | Cached embeddings for train/val/test sets |
| `pca_visualization_pretrained.png` | PCA plot showing class separation |
| `test_predictions_pretrained.csv` | Per-sequence predictions with probabilities |
| `three_layer_nn_pretrained.pt` | Trained 3-layer NN classifier |
| `embedding_analysis_results.json` | Linear probe + NN metrics, silhouette score, PCA variance, and (with `--include_random_baseline`) embedding-power deltas |

The same files prefixed `_random` are written when `--include_random_baseline` is enabled, plus an `embedding_power_*` metric set for each (= pretrained metric − random metric).

---

## Aggregating across multiple runs

After running directory-mode inference (or a batch of single-file jobs), summarize:

```bash
python summarize_inference_results.py \
    --results_dir /path/to/predictions \
    --output_csv summary.csv
```

Produces one row per `_metrics.json` with all the standard binary-classification metrics.

---

## Reproducibility note

- All scripts default to `seed=42` (`--seed` to change).
- The 3-layer NN in `embedding_analysis_gena_lm.py` uses a fixed seed for weight init and is deterministic given the same embeddings.
- Mixed-precision (`--bf16` / `--fp16`) introduces small numerical differences compared to fp32; metric magnitudes shift by < 0.001 in our testing but the relative ordering is stable.

---

## Citation

If you use this fork, please cite the LAMBDA benchmark and the relevant GENA-LM / modernGENA paper for the model you ran.

**LAMBDA** (bioRxiv preprint — [PMC13041943](https://pmc.ncbi.nlm.nih.gov/articles/PMC13041943/)):
```bibtex
@article{lindsey2026lambda,
  title   = {LAMBDA: A Prophage Detection Benchmark for Genomic Language Models},
  author  = {Lindsey, LeAnn M. and Pershing, Nicole L. and Dufault-Thompson, Keith and
             Gwak, Ho-jin and Habib, Anisa and Schindler, Aaron and Rakheja, Arjun and
             Round, June and Stephens, W. Zac and Blaschke, Anne J. and Sundar, Hari and
             Jiang, Xiaofang},
  journal = {bioRxiv},
  year    = {2026},
  url     = {https://pmc.ncbi.nlm.nih.gov/articles/PMC13041943/}
}
```

**GENA-LM** — original paper:
```bibtex
@article{fishman2025genalm,
  title   = {GENA-LM: a family of open-source foundational DNA language models for long sequences},
  author  = {Fishman, Veniamin and Kuratov, Yuri and Shmelev, Aleksei and Petrov, Maxim and
             Penzar, Dmitry and Shepelin, Denis and Chekanov, Nikolay and Kardymon, Olga and
             Burtsev, Mikhail},
  journal = {Nucleic Acids Research},
  year    = {2025}
}
```

**modernGENA** (bioRxiv preprint):
```bibtex
@article{aspidova2026moderngena,
  title   = {Back to BERT in 2026: ModernGENA as a Strong, Efficient Baseline for DNA Foundation Models},
  author  = {Aspidova, Alena and Kuratov, Yuri and Shadskiy, Artem and Burtsev, Mikhail and
             Fishman, Veniamin},
  journal = {bioRxiv},
  year    = {2026},
  url     = {https://www.biorxiv.org/content/10.64898/2026.04.21.719816v1}
}
```

---

## License

This fork preserves the upstream license — see [`LICENSE`](./LICENSE).
