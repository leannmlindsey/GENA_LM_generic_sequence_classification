#!/bin/bash
#
# Verify the TRAINING stage is complete before running inference.
#
# For every (window × variant) cell it checks:
#   - finetune: how many of SEEDS produced seed-<N>/test_results.json with an
#     eval_mcc (the metric the LAMBDA paper averages over). Prints each seed's
#     MCC, the mean across completed seeds, and the best seed. Flags any missing
#     seed — a missing seed corrupts the averaged metric, so all SEEDS must be
#     present, not just one.
#   - embedding (Surface D): embedding_analysis_results.json + the pretrained
#     embeddings .npz exist.
#
# Reads OUTPUT_DIR / WINDOWS / VARIANTS / SEEDS from lambda_replication.conf, so
# it matches whatever sweep you configured. Exits non-zero if anything is
# incomplete (so you can gate inference on it).
#
# Usage:
#   bash slurm_scripts/lambda_replication/check_training.sh
#
# See also: check_random_baseline.sh (random-embedding baseline per cell),
#           check_inference.sh (Stage-2 outputs).

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: missing ${CONFIG}"; exit 1
fi
# shellcheck disable=SC1090
source "${CONFIG}"

[ -n "${OUTPUT_DIR:-}" ] || { echo "ERROR: OUTPUT_DIR unset in conf"; exit 1; }
[ -n "${WINDOWS:-}" ]    || { echo "ERROR: WINDOWS unset in conf";    exit 1; }
[ -n "${VARIANTS:-}" ]   || { echo "ERROR: VARIANTS unset in conf";   exit 1; }
[ -n "${SEEDS:-}" ]      || { echo "ERROR: SEEDS unset in conf";      exit 1; }

echo "============================================================"
echo "Training completeness check"
echo "  OUTPUT_DIR: ${OUTPUT_DIR}"
echo "  WINDOWS:    ${WINDOWS}"
echo "  VARIANTS:   ${VARIANTS}"
echo "  SEEDS:      ${SEEDS}"
echo "============================================================"

OUTPUT_DIR="${OUTPUT_DIR}" WINDOWS="${WINDOWS}" VARIANTS="${VARIANTS}" SEEDS="${SEEDS}" \
python3 - <<'PY'
import json, os, sys

out   = os.environ["OUTPUT_DIR"]
wins  = os.environ["WINDOWS"].split()
varis = os.environ["VARIANTS"].split()
seeds = [int(s) for s in os.environ["SEEDS"].split()]

ft_total = ft_complete = 0
emb_total = emb_complete = 0

print("\n--- finetune: seeds with test_results.json (eval_mcc) ---")
for w in wins:
    for v in varis:
        ft_total += 1
        cell = os.path.join(out, w, "finetune", v)
        mccs, missing = {}, []
        for s in seeds:
            rp = os.path.join(cell, f"seed-{s}", "test_results.json")
            if not os.path.isfile(rp):
                missing.append(s); continue
            try:
                m = json.load(open(rp)).get("eval_mcc")
            except Exception:
                m = None
            if m is None:
                missing.append(s)
            else:
                mccs[s] = float(m)
        n = len(mccs)
        tag = f"{w}/{v}"
        if n:
            mean = sum(mccs.values()) / n
            best = max(mccs, key=mccs.get)
            seedstr = " ".join(f"s{s}={mccs[s]:.4f}" for s in sorted(mccs))
            status = "OK " if not missing else "INCOMPLETE"
            print(f"  [{n}/{len(seeds)}] {status:10} {tag:16} {seedstr}")
            print(f"        mean={mean:.4f}  best=seed-{best} ({mccs[best]:.4f})"
                  + (f"  MISSING seeds: {','.join(map(str, missing))}" if missing else ""))
        else:
            print(f"  [0/{len(seeds)}] MISSING    {tag:16} no completed seeds")
        if n == len(seeds):
            ft_complete += 1

print("\n--- embedding (Surface D): results JSON + pretrained .npz ---")
for w in wins:
    for v in varis:
        emb_total += 1
        cell = os.path.join(out, w, "embedding", v)
        js  = os.path.join(cell, "embedding_analysis_results.json")
        npz = os.path.join(cell, "embeddings_pretrained.npz")
        tag = f"{w}/{v}"
        ok_js, ok_npz = os.path.isfile(js), os.path.isfile(npz)
        if ok_js and ok_npz:
            print(f"  [OK]   {tag:16} results + embeddings_pretrained.npz")
            emb_complete += 1
        else:
            miss = []
            if not ok_js:  miss.append("embedding_analysis_results.json")
            if not ok_npz: miss.append("embeddings_pretrained.npz")
            print(f"  [FAIL] {tag:16} missing: {', '.join(miss)}")

print("\n" + "-" * 60)
print(f"Finetune cells fully complete (all {len(seeds)} seeds): {ft_complete} / {ft_total}")
print(f"Embedding cells complete:                            {emb_complete} / {emb_total}")
print("-" * 60)
sys.exit(0 if (ft_complete == ft_total and emb_complete == emb_total) else 1)
PY
