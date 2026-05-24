#!/usr/bin/env python3
"""
Fine-tuning GENA-LM / ModernGENA for binary sequence classification.

This is an HF Trainer adaptation of the upstream GENA-LM long-context
fine-tuning recipe in
  downstream_tasks/promoter_prediction/finetune_promoter_16000.sh
(BigBird 4096-token training on 16 kb sequences for binary promoter
classification). Upstream's SLURM script uses lm_experiments_tools.Trainer
+ Horovod + APEX + their internal `.pth` checkpoints. We substitute HF
Trainer, which the upstream README explicitly sanctions:
  "you can use HF Transformers Trainer, PyTorch Lightning, or Accelerate
   and PyTorch with custom training loops instead."
  -- README_previous_generation.md L193-194

For BigBird specifically the upstream README also documents the HF idiom:
  from transformers import AutoTokenizer, BigBirdForSequenceClassification
  model = BigBirdForSequenceClassification.from_pretrained(
              'AIRI-Institute/gena-lm-bigbird-base-t2t')

For ModernGENA, the upstream reference is
  examples/modernGENA/sequence_classification/train.py
which already uses HF Trainer; this script's defaults can be flipped to
match its hyperparameters with --lr_scheduler_type linear --weight_decay
1e-3 --learning_rate 3e-5 --metric_for_best_model eval_pr_auc.

Optimization Features:
- Mixed precision training (fp16/bf16) for 2-3x speedup
- Gradient checkpointing for longer sequences (4k/8k)
- Configurable early stopping with step-based evaluation
- TF32 support for Ampere GPUs
- Fused AdamW optimizer option

Usage:
    # Basic training
    python finetune_gena_lm_phage.py \
        --model_name AIRI-Institute/gena-lm-bert-base-t2t \
        --dataset_dir /path/to/data \
        --output_dir ./output \
        --max_length 2048 \
        --per_device_train_batch_size 8 \
        --num_train_epochs 3

    # Optimized training (recommended)
    python finetune_gena_lm_phage.py \
        --dataset_dir /path/to/data \
        --output_dir ./output \
        --bf16 \
        --per_device_train_batch_size 16 \
        --eval_strategy steps \
        --eval_steps 500 \
        --num_train_epochs 10 \
        --early_stopping_patience 3

    # Long sequences (4k/8k) with gradient checkpointing
    python finetune_gena_lm_phage.py \
        --dataset_dir /path/to/data \
        --output_dir ./output \
        --max_length 4096 \
        --bf16 \
        --gradient_checkpointing \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    roc_auc_score,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune GENA-LM for binary classification"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="AIRI-Institute/gena-lm-bigbird-base-t2t",
        help=(
            "HuggingFace model name or path. Default 'gena-lm-bigbird-base-t2t' "
            "is the long-context GENA-LM variant (4096 tokens / ~36 kb context, "
            "per Fishman et al. 2025). Use a BERT base variant only for sequences "
            "<= ~3 kb. For ModernGENA, pass 'AIRI-Institute/moderngena-base' "
            "(switch lr_scheduler_type to 'linear' and weight_decay to 1e-3 "
            "to match the modernGENA reference config)."
        ),
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing train.csv, dev.csv, test.csv",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help=(
            "Maximum sequence length in tokens. GENA-LM uses ~6 bp/token. "
            "BERT variants cap at 512 tokens (~3 kb). BigBird variants accept "
            "up to 4096 tokens (~24 kb in practice, 36 kb in the paper). "
            "For LAMBDA: 2k window → 512; 4k window → 1024; 8k window → 2048."
        ),
    )
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./gena_lm_output")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    # Default hyperparameters follow the upstream GENA-LM BigBird recipe in
    # downstream_tasks/promoter_prediction/finetune_promoter_16000.sh
    # (the upstream long-context binary classification recipe), substituted
    # into HF Trainer per the upstream README's sanctioned alternative:
    # "you can use HF Transformers Trainer ... instead."
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Default 1e-4 matches upstream BigBird promoter recipe.")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Default 0.0 matches upstream BigBird recipe (no L2).")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.06,
                        help="HF-Trainer-friendly equivalent of upstream's 250 warmup steps "
                             "(varies with dataset size; ~0.06 is the modernGENA default).")
    parser.add_argument(
        "--lr_scheduler_type", type=str, default="constant_with_warmup",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                 "constant", "constant_with_warmup", "inverse_sqrt"],
        help="Default 'constant_with_warmup' matches upstream BigBird recipe.",
    )
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_strategy", type=str, default="epoch",
                        help="Evaluation strategy: 'epoch', 'steps', or 'no'")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every N steps (only used if eval_strategy='steps')")
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps (only used if save_strategy='steps')")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--load_best_model_at_end", action="store_true", default=True)
    parser.add_argument("--metric_for_best_model", type=str, default="eval_f1",
                        help="Default 'eval_f1' matches upstream BigBird promoter recipe "
                             "(--optimize_metric f1). All of f1/mcc/pr_auc/accuracy are "
                             "computed and logged each eval; this only controls which one "
                             "drives best-checkpoint selection.")
    parser.add_argument("--early_stopping_patience", type=int, default=7,
                        help="Default 7 matches upstream BigBird recipe. 0 to disable.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    # Precision and optimization arguments
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Use fp16 mixed precision (for V100 and older GPUs)")
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Use bf16 mixed precision (recommended for A100/H100)")
    parser.add_argument("--tf32", action="store_true", default=False,
                        help="Enable TF32 for matmul (Ampere GPUs only, slight speedup)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                        help="Enable gradient checkpointing to reduce memory (allows larger batches/sequences)")
    parser.add_argument("--optim", type=str, default="adamw_torch",
                        choices=["adamw_torch", "adamw_torch_fused", "adamw_apex_fused", "adafactor"],
                        help="Optimizer to use (adamw_torch_fused is faster on newer GPUs)")
    parser.add_argument("--torch_compile", action="store_true", default=False,
                        help="Use torch.compile() for potential speedup (experimental)")
    parser.add_argument("--dataloader_pin_memory", action="store_true", default=True,
                        help="Pin memory for faster GPU transfer")

    return parser.parse_args()


def load_data(dataset_dir: str) -> DatasetDict:
    """
    Load train/dev/test CSV files.
    Expected format: CSV with columns 'sequence' and 'label'
    
    IMPORTANT: Only 'sequence' and 'label' columns are used.
    All other columns are explicitly dropped to prevent data leakage.
    """
    dataset_dir = Path(dataset_dir)
    
    datasets = {}
    for split in ["train", "dev", "test"]:
        filepath = dataset_dir / f"{split}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            
            # Log all columns found
            print(f"\n{split}.csv columns found: {list(df.columns)}")
            
            # Ensure column names are standardized
            if "sequence" not in df.columns:
                # Try common alternatives
                for col in ["seq", "dna", "Sequence", "DNA"]:
                    if col in df.columns:
                        df = df.rename(columns={col: "sequence"})
                        print(f"  Renamed '{col}' -> 'sequence'")
                        break
            if "label" not in df.columns:
                for col in ["Label", "labels", "Labels", "class", "Class"]:
                    if col in df.columns:
                        df = df.rename(columns={col: "label"})
                        print(f"  Renamed '{col}' -> 'label'")
                        break
            
            # Verify required columns exist
            if "sequence" not in df.columns:
                raise ValueError(f"Could not find 'sequence' column in {filepath}")
            if "label" not in df.columns:
                raise ValueError(f"Could not find 'label' column in {filepath}")
            
            # EXPLICITLY select only sequence and label columns
            # This prevents any data leakage from other columns
            used_columns = ["sequence", "label"]
            dropped_columns = [col for col in df.columns if col not in used_columns]
            
            if dropped_columns:
                print(f"  DROPPING columns (not used): {dropped_columns}")
            
            # Create dataset with ONLY the two required columns
            df_clean = df[used_columns].copy()
            
            print(f"  USING columns: {list(df_clean.columns)}")
            print(f"  Loaded {split}: {len(df_clean)} samples")
            
            # Verify label distribution
            label_counts = df_clean["label"].value_counts().to_dict()
            print(f"  Label distribution: {label_counts}")
            
            datasets[split if split != "dev" else "validation"] = Dataset.from_pandas(
                df_clean, preserve_index=False
            )
        else:
            print(f"Warning: {filepath} not found")
    
    return DatasetDict(datasets)


def compute_metrics(eval_pred):
    """Compute classification metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # For binary classification, get probabilities for ROC-AUC
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    mcc = matthews_corrcoef(labels, predictions)
    
    # Calculate sensitivity (recall) and specificity
    # For binary: label 1 = positive, label 0 = negative
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }
    
    # Add ROC-AUC if binary
    try:
        auc = roc_auc_score(labels, probs[:, 1])
        metrics["auc"] = auc
    except Exception:
        pass
    
    return metrics


def main():
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Enable TF32 for Ampere GPUs (slight speedup for matmul)
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("=" * 60)
    print("GENA-LM Fine-tuning")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Max length: {args.max_length}")
    print(f"Batch size: {args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Eval strategy: {args.eval_strategy}" + (f" (every {args.eval_steps} steps)" if args.eval_strategy == "steps" else ""))
    print(f"Early stopping patience: {args.early_stopping_patience}")
    print(f"Seed: {args.seed}")
    print("-" * 60)
    print("Optimizations:")
    print(f"  fp16: {args.fp16}")
    print(f"  bf16: {args.bf16}")
    print(f"  tf32: {args.tf32}")
    print(f"  Gradient checkpointing: {args.gradient_checkpointing}")
    print(f"  Optimizer: {args.optim}")
    print(f"  torch.compile: {args.torch_compile}")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        if args.bf16:
            # Check bf16 support
            if torch.cuda.get_device_capability()[0] >= 8:
                print("  bf16 supported (Ampere or newer)")
            else:
                print("  WARNING: bf16 may not be fully supported on this GPU")
    else:
        print("WARNING: No GPU detected!")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    
    # Load model for sequence classification
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,  # Binary classification
        trust_remote_code=True,
    )

    # Enable gradient checkpointing if requested (saves memory, ~20% slower)
    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        try:
            model.gradient_checkpointing_enable()
            print("  Gradient checkpointing enabled")
        except ValueError as e:
            print(f"  WARNING: Gradient checkpointing not supported by this model: {e}")
            print("  Continuing without gradient checkpointing...")
            args.gradient_checkpointing = False

    # Apply torch.compile if requested (experimental)
    if args.torch_compile:
        print("Applying torch.compile()...")
        try:
            model = torch.compile(model, mode="default")
            print("  torch.compile() applied successfully")
        except Exception as e:
            print(f"  WARNING: torch.compile() failed: {e}")
            print("  Continuing without torch.compile()")

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load data
    print("\nLoading dataset...")
    dataset = load_data(args.dataset_dir)
    
    # Tokenize function
    def tokenize_function(examples):
        """Tokenize DNA sequences."""
        return tokenizer(
            examples["sequence"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors=None,  # Return lists for dataset mapping
        )
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["sequence"],
        desc="Tokenizing",
    )
    
    # Set format for PyTorch
    tokenized_datasets.set_format("torch")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        optim=args.optim,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        report_to="none",  # Disable wandb/tensorboard by default
    )
    
    # Callbacks
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    # Train
    print("\nStarting training...")
    import time
    train_start_time = time.time()
    train_result = trainer.train()
    train_elapsed = time.time() - train_start_time

    # Report training time and memory
    print(f"\nTraining completed in {train_elapsed:.1f} seconds ({train_elapsed/60:.1f} minutes)")
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak GPU memory: {peak_memory_mb:.1f} MB ({peak_memory_mb/1024:.2f} GB)")

    # Save model
    print("\nSaving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Log training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate on validation set
    if "validation" in tokenized_datasets:
        print("\nEvaluating on validation set...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    
    # Evaluate on test set
    if "test" in tokenized_datasets:
        print("\nEvaluating on test set...")
        # Use metric_key_prefix="eval" to match the requested output format
        test_metrics = trainer.evaluate(tokenized_datasets["test"], metric_key_prefix="eval")
        
        # Save test results to test_results.json in the requested format
        test_results_path = os.path.join(args.output_dir, "test_results.json")
        with open(test_results_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"Test results saved to: {test_results_path}")
        
        # Print final test results
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        for key, value in test_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # Save training summary with timing and configuration
    training_summary = {
        "model_name": args.model_name,
        "dataset_dir": args.dataset_dir,
        "output_dir": args.output_dir,
        "max_length": args.max_length,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "actual_epochs": train_result.metrics.get("epoch", args.num_train_epochs),
        "total_steps": train_result.metrics.get("total_flos", 0),
        "training_time_seconds": train_elapsed,
        "training_time_minutes": train_elapsed / 60,
        "seed": args.seed,
        "optimizations": {
            "fp16": args.fp16,
            "bf16": args.bf16,
            "tf32": args.tf32,
            "gradient_checkpointing": args.gradient_checkpointing,
            "optimizer": args.optim,
            "torch_compile": args.torch_compile,
        },
    }
    if torch.cuda.is_available():
        training_summary["gpu_name"] = torch.cuda.get_device_name(0)
        training_summary["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2)
    print(f"\nTraining summary saved to: {summary_path}")

    print("\nTraining complete!")
    print(f"Model saved to: {args.output_dir}")
    print(f"Total training time: {train_elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
