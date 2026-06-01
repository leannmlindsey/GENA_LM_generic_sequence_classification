#!/bin/bash
#
# Prefetch GENA-LM base models into the HF cache the lambda_replication jobs use.
#
# !!! RUN THIS ON A LOGIN NODE !!!  Biowulf COMPUTE nodes have no internet, so
# the finetune/embedding jobs run with HF_HUB_OFFLINE=1 against $HF_HOME. This
# script populates that cache once, ONLINE, from a login node. It reads HF_HOME
# from lambda_replication.conf so the cache location always matches the jobs.
#
# Usage (on a login node):
#   bash slurm_scripts/lambda_replication/prefetch_models.sh
#
# Prefetch a subset instead of the default two:
#   MODELS="AIRI-Institute/gena-lm-bigbird-base-t2t" \
#       bash slurm_scripts/lambda_replication/prefetch_models.sh

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"
# shellcheck disable=SC1090
[ -f "${CONFIG}" ] && source "${CONFIG}"

# HF cache root — MUST match what the job scripts use (jobs default to this too).
export HF_HOME="${HF_HOME:-/data/lindseylm/.cache/huggingface}"
# Allow internet for the prefetch, overriding any offline flags left in the shell.
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

# Models to prefetch. Defaults to the two variants in the standard sweep.
# Add bertbase (AIRI-Institute/gena-lm-bert-base-t2t) if you enable it in VARIANTS.
MODELS="${MODELS:-AIRI-Institute/gena-lm-bigbird-base-t2t AIRI-Institute/moderngena-base}"

echo "============================================================"
echo "Prefetch GENA-LM models into HF cache"
echo "  HF_HOME: ${HF_HOME}"
echo "  MODELS:  ${MODELS}"
echo "  host:    $(hostname)   <-- must be a LOGIN node (has internet)"
echo "============================================================"

case "$(hostname)" in
    cn*) echo "WARNING: this looks like a COMPUTE node (cn*) — it has no internet."
         echo "         Run this on a login node (e.g. biowulf) instead." ;;
esac

module load conda 2>/dev/null || true
conda activate gena_lm 2>/dev/null || source activate gena_lm 2>/dev/null || true

# Unquoted heredoc so ${MODELS} is injected by bash; the Python below has no
# other shell metacharacters.
python - <<PY
from transformers import (AutoConfig, AutoTokenizer,
                          AutoModelForSequenceClassification, AutoModelForMaskedLM)
models = "${MODELS}".split()
for m in models:
    print("=== prefetch", m, flush=True)
    # Cover every class the pipeline loads offline + the custom hub code:
    #   finetune  -> AutoModelForSequenceClassification
    #   embedding -> AutoModelForMaskedLM
    #   trust_remote_code -> pulls GENA-LM's modeling .py into \$HF_HOME/modules
    AutoConfig.from_pretrained(m, trust_remote_code=True)
    AutoTokenizer.from_pretrained(m, trust_remote_code=True)
    AutoModelForSequenceClassification.from_pretrained(m, num_labels=2, trust_remote_code=True)
    AutoModelForMaskedLM.from_pretrained(m, trust_remote_code=True)
    print("   done", m, flush=True)
print("ALL PREFETCH DONE")
PY

echo ""
echo "Cached repos under ${HF_HOME}/hub:"
ls "${HF_HOME}/hub" 2>/dev/null | grep -i -E 'gena|modern' || echo "  (none found — check errors above)"
echo ""
echo "Done. Jobs run offline against HF_HOME=${HF_HOME}."
