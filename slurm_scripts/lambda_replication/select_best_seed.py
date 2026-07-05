#!/usr/bin/env python3
"""
Per-variant, pick the finetune SEED with the highest test-set eval_mcc (the
metric the LAMBDA paper reports). Writes <output_dir>/winners.json for the
inference stage.

This ranks by eval_mcc (eval_f1 also recorded for info) and emits
machine-readable JSON keyed by variant, matching the
ProkBERT winners.json contract consumed by print_winner_exports.py:

    {
      "bigbird": {
        "type": "finetune",
        "seed": 3,
        "eval_f1": 0.91,
        "eval_mcc": 0.82,
        "path": "<abs path to seed-3 checkpoint dir>",
        "base_model": "AIRI-Institute/gena-lm-bigbird-base-t2t",
        "all_candidates": [{seed, eval_f1, eval_mcc}, ...]
      },
      ...
    }

Reads:  <output_dir>/finetune/<variant>/seed-<N>/test_results.json  (eval_f1, eval_mcc)
"""

import argparse
import glob
import json
import os
import sys


VARIANT_TO_HF = {
    "bigbird":    "AIRI-Institute/gena-lm-bigbird-base-t2t",
    "moderngena": "AIRI-Institute/moderngena-base",
    "bertbase":   "AIRI-Institute/gena-lm-bert-base-t2t",
}


def collect_seed_candidates(variant_dir):
    out = []
    for seed_dir in sorted(glob.glob(os.path.join(variant_dir, "seed-*"))):
        results_path = os.path.join(seed_dir, "test_results.json")
        if not os.path.isfile(results_path):
            print(f"  WARN: missing {results_path}, skipping", file=sys.stderr)
            continue
        with open(results_path) as f:
            metrics = json.load(f)
        mcc = metrics.get("eval_mcc")
        if mcc is None:
            print(f"  WARN: no eval_mcc in {results_path}, skipping", file=sys.stderr)
            continue
        seed = int(os.path.basename(seed_dir).split("-")[1])
        out.append({
            "type": "finetune",
            "seed": seed,
            "eval_mcc": float(mcc),
            "eval_f1": float(metrics.get("eval_f1", 0.0)),
            # finetune_gena_lm_phage.py saves the best checkpoint into the seed
            # dir itself (load_best_model_at_end=True), so the dir IS the model.
            "path": os.path.abspath(seed_dir),
        })
    return out


def read_embedding_scores(embedding_dir):
    """Linear-probe / 3-layer-NN candidates from embedding_analysis_results.json.

    embedding_analysis_gena_lm.py writes FLAT keys (pretrained_linear_probe_mcc /
    pretrained_nn_mcc) — the Table 2 test-MCC numbers — plus the deployable probe
    artifacts saved alongside (linear_probe_pretrained.pkl,
    three_layer_nn_pretrained.pt + three_layer_nn_pretrained_scaler.pkl).
    """
    results_path = os.path.join(embedding_dir, "embedding_analysis_results.json")
    if not os.path.isfile(results_path):
        print(f"  WARN: missing {results_path} (no probe candidates)", file=sys.stderr)
        return []
    with open(results_path) as f:
        r = json.load(f)
    out = []
    lp = r.get("pretrained_linear_probe_mcc")
    if lp is not None:
        out.append({"type": "linear_probe", "eval_mcc": float(lp),
                    "head_path": os.path.abspath(
                        os.path.join(embedding_dir, "linear_probe_pretrained.pkl"))})
    nn = r.get("pretrained_nn_mcc")
    if nn is not None:
        out.append({"type": "three_layer_nn", "eval_mcc": float(nn),
                    "head_path": os.path.abspath(
                        os.path.join(embedding_dir, "three_layer_nn_pretrained.pt")),
                    "scaler_path": os.path.abspath(
                        os.path.join(embedding_dir, "three_layer_nn_pretrained_scaler.pkl"))})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True,
                        help="Per-window replication dir (contains finetune/<variant>/seed-*)")
    parser.add_argument("--variants", nargs="+", required=True,
                        help="Variants to select for (e.g. bigbird moderngena)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Skip variants with no completed seeds instead of aborting. "
                             "Use only for in-progress/smoke runs; a missing variant on the "
                             "reviewer-facing pipeline means a real training failure.")
    args = parser.parse_args()

    winners = {}
    skipped = []
    for variant in args.variants:
        print(f"\n=== {variant} ===")
        variant_dir = os.path.join(args.output_dir, "finetune", variant)
        embedding_dir = os.path.join(args.output_dir, "embedding", variant)
        ft = collect_seed_candidates(variant_dir)
        emb = read_embedding_scores(embedding_dir)

        # Fine-tuning scored by the MEAN eval_mcc of its 5 seeds (deployed via the
        # single best seed); each probe scored by its test MCC.
        sel = []
        if ft:
            ft_avg = sum(c["eval_mcc"] for c in ft) / len(ft)
            best_seed = max(ft, key=lambda c: c["eval_mcc"])
            sel.append({"type": "finetune", "score": float(ft_avg), "seed": best_seed["seed"],
                        "eval_mcc": float(ft_avg), "best_seed_eval_mcc": best_seed["eval_mcc"],
                        "eval_f1": best_seed["eval_f1"], "path": best_seed["path"]})
        for cand in emb:
            c = dict(cand); c["score"] = c["eval_mcc"]
            sel.append(c)

        if not sel:
            if not args.allow_partial:
                print(f"  ERROR: no candidates for {variant} (no finetune seeds AND no "
                      f"embedding_analysis_results.json). Re-run with --allow-partial to skip.",
                      file=sys.stderr)
                sys.exit(1)
            print(f"  SKIP: no candidates for {variant}", file=sys.stderr)
            skipped.append(variant)
            continue

        def _tag(c):
            return c["type"] + (f"/seed-{c['seed']}" if c.get("seed") is not None else "")
        for c in sorted(sel, key=lambda c: c["score"], reverse=True):
            note = "  (mean of 5 seeds)" if c["type"] == "finetune" else ""
            print(f"  score={c['score']:.4f}  {_tag(c)}{note}")

        # Highest score wins; ties prefer finetune (deploy FT only when its 5-seed
        # average is >= both probes), per the LAMBDA design.
        winner = max(sel, key=lambda c: (c["score"], c["type"] == "finetune"))
        winner["base_model"] = VARIANT_TO_HF.get(variant, variant)
        winner["all_candidates"] = [
            {k: v for k, v in c.items() if k in ("type", "seed", "eval_mcc", "score")}
            for c in sel
        ]
        winners[variant] = winner
        print(f"  WINNER: {_tag(winner)} (score={winner['score']:.4f})")

    out_path = os.path.join(args.output_dir, "winners.json")
    with open(out_path, "w") as f:
        json.dump(winners, f, indent=2)
    print(f"\nWrote {out_path}  ({len(winners)} variant(s) with winners"
          f"{'; skipped: ' + ','.join(skipped) if skipped else ''})")

    if not winners:
        print("\nERROR: no variant produced any candidates; nothing to write.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
