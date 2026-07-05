#!/usr/bin/env python3
"""
Inference with a frozen-embedding probe (linear probe or 3-layer NN) for GENA-LM —
the pathway used when select_best_seed.py picks a probe over the fine-tuned
checkpoint. Port of ProkBERT's inference_embedding_head.py, adapted to GENA-LM's
base-model loading (AutoModelForMaskedLM -> .esm) and I/O.

Loads the pretrained GENA-LM base model (per-variant HF name), extracts embeddings
on the input CSV with the SAME extract_embeddings() the probe was trained with, then
applies the saved probe artifacts written by embedding_analysis_gena_lm.py:
    linear_probe_pretrained.pkl              {"classifier", "scaler"}
    three_layer_nn_pretrained.pt             {model_state_dict, input_dim, hidden_dim}
    three_layer_nn_pretrained_scaler.pkl     the NN's StandardScaler

Output matches inference_gena_lm.py (df.copy() + prob_0, prob_1, pred_label; a
sibling _metrics.json when --save_metrics and labels exist).

Usage:
    python inference_embedding_head_gena_lm.py \
        --model_path AIRI-Institute/gena-lm-bigbird-base-t2t \
        --head_type three_layer_nn \
        --head_path   <embedding_dir>/three_layer_nn_pretrained.pt \
        --scaler_path <embedding_dir>/three_layer_nn_pretrained_scaler.pkl \
        --input_csv <csv> --output_csv <out.csv> --max_length 4096 --save_metrics
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from embedding_analysis_gena_lm import ThreeLayerNN, extract_embeddings


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True, help="pretrained base model (HF name or path)")
    p.add_argument("--head_type", required=True, choices=["linear_probe", "three_layer_nn"])
    p.add_argument("--head_path", required=True, help=".pkl (linear_probe) or .pt (three_layer_nn)")
    p.add_argument("--scaler_path", default=None, help="NN scaler .pkl (required for three_layer_nn)")
    p.add_argument("--input_csv", required=True, help="input CSV with a `sequence` column")
    p.add_argument("--output_csv", required=True, help="output predictions CSV path")
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--pooling", default="mean", choices=["mean", "cls", "last"])
    p.add_argument("--threshold", type=float, default=0.5, help="prob_1 threshold for pred_label")
    p.add_argument("--save_metrics", action="store_true")
    args = p.parse_args()
    if args.head_type == "three_layer_nn" and not args.scaler_path:
        p.error("--scaler_path is required when --head_type=three_layer_nn")
    return args


def load_base_model(model_path, device):
    """Load the GENA-LM base encoder for embeddings (mirrors embedding_analysis_gena_lm.py)."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    full_model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(full_model, "esm"):
        model = full_model.esm
    elif hasattr(full_model, "base_model"):
        model = full_model.base_model
    else:
        model = full_model
    model = model.to(device)
    model.eval()
    return model, tokenizer


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"head={args.head_type} device={device}")

    df = pd.read_csv(args.input_csv)
    if "sequence" not in df.columns:
        raise ValueError(f"{args.input_csv} has no `sequence` column")
    sequences = df["sequence"].astype(str).tolist()
    has_labels = "label" in df.columns
    labels = df["label"].astype(int).tolist() if has_labels else [0] * len(df)

    model, tokenizer = load_base_model(args.model_path, device)
    emb, _ = extract_embeddings(
        model, tokenizer, sequences, labels,
        args.batch_size, args.max_length, args.pooling, device,
    )

    if args.head_type == "linear_probe":
        with open(args.head_path, "rb") as f:
            bundle = pickle.load(f)
        clf, scaler = bundle["classifier"], bundle["scaler"]
        probs = clf.predict_proba(scaler.transform(emb))   # [:,0]=neg, [:,1]=pos
    else:
        ckpt = torch.load(args.head_path, map_location=device)
        nn_model = ThreeLayerNN(ckpt["input_dim"], ckpt["hidden_dim"]).to(device)
        nn_model.load_state_dict(ckpt["model_state_dict"])
        nn_model.eval()
        with open(args.scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X = torch.FloatTensor(scaler.transform(emb)).to(device)
        with torch.no_grad():
            probs = torch.softmax(nn_model(X), dim=1).cpu().numpy()

    preds = (probs[:, 1] >= args.threshold).astype(int)

    # Match inference_gena_lm.py: preserve all input columns, add prob_0/prob_1/pred_label.
    out = df.copy()
    out["prob_0"] = probs[:, 0]
    out["prob_1"] = probs[:, 1]
    out["pred_label"] = preds

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Wrote predictions: {args.output_csv}")

    if has_labels and args.save_metrics:
        from sklearn.metrics import matthews_corrcoef
        y = np.asarray(labels, dtype=int)
        tp = int(((y == 1) & (preds == 1)).sum()); tn = int(((y == 0) & (preds == 0)).sum())
        fp = int(((y == 0) & (preds == 1)).sum()); fn = int(((y == 1) & (preds == 0)).sum())
        m = {"mcc": float(matthews_corrcoef(y, preds)) if len(np.unique(y)) > 1 else 0.0,
             "tp": tp, "tn": tn, "fp": fp, "fn": fn,
             "recall": (tp / (tp + fn)) if (tp + fn) else 0.0,
             "specificity": (tn / (tn + fp)) if (tn + fp) else 0.0,
             "fpr": (fp / (fp + tn)) if (fp + tn) else 0.0}
        with open(args.output_csv.replace(".csv", "_metrics.json"), "w") as f:
            json.dump(m, f, indent=2)
        print(f"  MCC={m['mcc']:.4f}  (tp={tp} tn={tn} fp={fp} fn={fn})")


if __name__ == "__main__":
    main()
