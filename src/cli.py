from __future__ import annotations
import argparse, json, os
import pandas as pd
from .data import load_data, normalize_types
from .features import build_links, engineer_features, attach_labels
from .model import train_eval, FEATURE_COLS, save_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    invoices, po_grn, mismatches = load_data(args.data_dir)
    inv_agg, po = normalize_types(invoices, po_grn)

    linked = build_links(inv_agg, po)
    feats = engineer_features(linked)
    labelled = attach_labels(feats, mismatches)

    # Save features for inspection
    labelled.to_csv(os.path.join(args.out_dir, "features.csv"), index=False)

    # Train/eval ML model
    metrics, clf = train_eval(labelled)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save trained model for reuse in D3 or services
    model_path = os.path.join(args.out_dir, "matcher_model.pkl")
    save_model(clf, model_path)

    print("Metrics written to metrics.json")
    print(f"Model saved to {model_path}")
    print("Feature columns:", FEATURE_COLS)

if __name__ == "__main__":
    main()
