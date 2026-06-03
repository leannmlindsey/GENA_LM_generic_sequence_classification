#!/usr/bin/env python3
"""
Diagnostic: does picking the best seed by eval_f1 differ from picking by eval_mcc?

For every (variant x window) under OUTPUT_DIR, read each seed's test_results.json
(which already contains BOTH eval_f1 and eval_mcc for its saved checkpoint) and
report whether the F1-best seed and the MCC-best seed are the same.

If they match everywhere, the metric switch in select_best_seed.py picks the same
winners on the EXISTING checkpoints -> no need to retrain for seed selection.

Usage (on Biowulf, after Stage 1 training finished):
    python3 compare_f1_mcc_winners.py \
        --output_dir /data/.../outputs \
        --windows 2k 4k 8k \
        --variants bigbird moderngena
"""
import argparse
import glob
import json
import os
import sys


def load_seeds(variant_dir):
    """Return [{seed, eval_f1, eval_mcc}, ...] for one variant dir."""
    out = []
    for seed_dir in sorted(glob.glob(os.path.join(variant_dir, "seed-*"))):
        results = os.path.join(seed_dir, "test_results.json")
        if not os.path.isfile(results):
            print(f"  WARN: no test_results.json in {seed_dir}", file=sys.stderr)
            continue
        with open(results) as f:
            m = json.load(f)
        if m.get("eval_f1") is None or m.get("eval_mcc") is None:
            print(f"  WARN: missing eval_f1/eval_mcc in {results}", file=sys.stderr)
            continue
        seed = int(os.path.basename(seed_dir).split("-")[1])
        out.append({"seed": seed,
                    "eval_f1": float(m["eval_f1"]),
                    "eval_mcc": float(m["eval_mcc"])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--windows", nargs="+", default=["2k", "4k", "8k"])
    ap.add_argument("--variants", nargs="+", default=["bigbird", "moderngena"])
    args = ap.parse_args()

    n_match = n_diff = n_missing = 0
    for W in args.windows:
        for V in args.variants:
            vdir = os.path.join(args.output_dir, W, "finetune", V)
            cands = load_seeds(vdir)
            print(f"\n=== {W} / {V} ===")
            if not cands:
                print(f"  (no seed results found under {vdir})")
                n_missing += 1
                continue
            for c in sorted(cands, key=lambda c: c["eval_mcc"], reverse=True):
                print(f"  seed-{c['seed']}: eval_f1={c['eval_f1']:.4f}  eval_mcc={c['eval_mcc']:.4f}")
            best_f1 = max(cands, key=lambda c: c["eval_f1"])
            best_mcc = max(cands, key=lambda c: c["eval_mcc"])
            if best_f1["seed"] == best_mcc["seed"]:
                print(f"  MATCH  -> both pick seed-{best_f1['seed']}")
                n_match += 1
            else:
                print(f"  DIFFER -> F1 picks seed-{best_f1['seed']} (f1={best_f1['eval_f1']:.4f}, "
                      f"mcc={best_f1['eval_mcc']:.4f}); "
                      f"MCC picks seed-{best_mcc['seed']} (f1={best_mcc['eval_f1']:.4f}, "
                      f"mcc={best_mcc['eval_mcc']:.4f})")
                n_diff += 1

    print("\n------------------------------------------------------------")
    print(f"groups: {n_match} match, {n_diff} DIFFER, {n_missing} missing")
    if n_diff == 0 and n_missing == 0:
        print("=> F1-best and MCC-best seed agree everywhere. No retrain needed for "
              "seed selection.")
    elif n_diff:
        print("=> At least one group disagrees; review the DIFFER lines above before "
              "deciding on a retrain.")
    print("Note: this compares CROSS-SEED winner only. The within-run checkpoint in "
          "each seed dir was still chosen by eval_f1; a true MCC checkpoint would "
          "require retraining.")


if __name__ == "__main__":
    main()
