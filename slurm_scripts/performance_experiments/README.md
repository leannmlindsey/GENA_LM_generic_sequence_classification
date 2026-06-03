# Performance experiments (archived)

One-off hardware/precision/throughput benchmarks and memory-fit checks used while
tuning GENA-LM training/inference on Biowulf. **Not part of the LAMBDA pipeline**
(see `../lambda_replication/`) and not required to reproduce any results — kept
here for reference.

| script | what it measures |
|--------|------------------|
| `benchmark_training.sh` | Times `finetune_gena_lm_phage.py` across precision/optimizer configs. |
| `run_train_benchmark.sh` | fp32 vs fp16 training timing (SLURM). |
| `run_memory_test.sh` | Whether 4k/8k sequences fit in GPU memory (1-epoch trial). |
| `wrapper_benchmark_inference.sh` | Inference timing under fp32/bf16/fp16. |
| `run_optimized_train.sh` | GPU-aware "optimized" training run (batch/precision auto-tuned). |
| `wrapper_run_training_optimized.sh` | Config wrapper that submits `run_optimized_train.sh`. |

Note: these scripts hardcode absolute Biowulf paths (repo root and, for the
optimized-training wrapper, this `performance_experiments/` dir). Adjust those
paths if the repo location changes.
