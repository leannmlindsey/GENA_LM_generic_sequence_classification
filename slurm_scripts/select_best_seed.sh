#!/bin/bash
#
# Scan all multi-seed fine-tune outputs and pick the best seed per
# (variant × window) by eval_f1 (the metric we trained with).
#
# Prints:
#   1. A summary table — best seed, eval_f1, eval_mcc, full path for each cell
#   2. The 6 CHECKPOINT_*K= lines ready to paste into submit_lambda_full_eval.sh
#
# Identifies variant by LR in the directory name (matches the pattern
# run_train_gena_lm.sh produces: _<seed>_<lr>_<timestamp>):
#   *_1e-4_* → bigbird     (matches the upstream BigBird recipe)
#   *_3e-5_* → moderngena  (matches the modernGENA reference config)
#
# Usage:
#   bash select_best_seed.sh

OUTPUT_ROOT="/data/lindseylm/GLM_EVALUATIONS/MODELS/GENA-LM/GENA_LM_generic_sequence_classification/output/filtered"

python3 - <<'PYTHON'
import os, json, glob, re

OUTPUT_ROOT = "/data/lindseylm/GLM_EVALUATIONS/MODELS/GENA-LM/GENA_LM_generic_sequence_classification/output/filtered"

# variant tag → LR substring in checkpoint directory name
LR_TO_VARIANT = {
    "1e-4": "bigbird",
    "3e-5": "moderngena",
}

# Collect (variant, window, seed, eval_f1, eval_mcc, path) for every finished cell
rows = []
missing = []

for window in ("2k", "4k", "8k"):
    win_dir = os.path.join(OUTPUT_ROOT, window)
    if not os.path.isdir(win_dir):
        continue
    for d in sorted(os.listdir(win_dir)):
        ckpt_dir = os.path.join(win_dir, d)
        if not os.path.isdir(ckpt_dir):
            continue
        # Parse name: gena_lm_lambda_filtered_<W>_<seed>_<lr>_<ts>
        m = re.match(r"gena_lm_lambda_filtered_(\w+)_(\d+)_([\w\-\.]+)_(\d{8}_\d{6})$", d)
        if not m:
            continue
        _, seed, lr, _ = m.groups()
        variant = LR_TO_VARIANT.get(lr)
        if variant is None:
            continue
        test_json = os.path.join(ckpt_dir, "test_results.json")
        if not os.path.exists(test_json):
            missing.append((variant, window, int(seed), ckpt_dir))
            continue
        with open(test_json) as f:
            t = json.load(f)
        rows.append({
            "variant": variant,
            "window": window,
            "seed": int(seed),
            "eval_f1": t.get("eval_f1", 0.0),
            "eval_mcc": t.get("eval_mcc", 0.0),
            "path": ckpt_dir,
        })

# Group by (variant, window), pick best by eval_f1
by_cell = {}
for r in rows:
    key = (r["variant"], r["window"])
    by_cell.setdefault(key, []).append(r)

print()
print("=== Per-cell summary (all available seeds, ranked by eval_f1) ===")
for (variant, window), cell_rows in sorted(by_cell.items()):
    print(f"\n  {variant} {window}: {len(cell_rows)} seeds available")
    for r in sorted(cell_rows, key=lambda x: -x["eval_f1"]):
        marker = "★ BEST" if r == max(cell_rows, key=lambda x: x["eval_f1"]) else ""
        print(f"    seed={r['seed']:>3}  eval_f1={r['eval_f1']:.4f}  eval_mcc={r['eval_mcc']:.4f}  {marker}")

if missing:
    print(f"\n=== {len(missing)} checkpoints missing test_results.json (still running or failed) ===")
    for v, w, s, p in missing:
        print(f"    {v:<12} {w:<3} seed={s:<3}  {p}")

# Best per cell
best = {key: max(cell_rows, key=lambda x: x["eval_f1"]) for key, cell_rows in by_cell.items()}

print()
print("=" * 70)
print("PASTE THESE INTO submit_lambda_full_eval.sh's case block:")
print("=" * 70)

for variant in ("bigbird", "moderngena"):
    print(f"\n    {variant})")
    print(f'        MODEL_NAME="gena_lm_{variant}"')
    for window in ("2k", "4k", "8k"):
        key = (variant, window)
        if key in best:
            print(f'        CHECKPOINT_{window.upper()}="{best[key]["path"]}"')
        else:
            print(f'        CHECKPOINT_{window.upper()}=""   # no completed seeds found')
    print(f'        ;;')

print()
PYTHON
