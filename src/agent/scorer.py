from __future__ import annotations
import os, sys, pandas as pd
from typing import Dict, Any
from pathlib import Path

src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from data import load_data, normalize_types
from features import build_links, engineer_features
from model import load_model, FEATURE_COLS

def score_invoice(data_dir: str, model_path: str, invoice_id: str, po_number: str) -> Dict[str, Any]:
    """Load CSVs + model, compute features for a single (invoice_id, po_number), return score + facts."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train D2 first to create matcher_model.pkl.")
    
    clf = load_model(model_path)
    invoices, po_grn, mismatches = load_data(data_dir)
    inv_agg, po = normalize_types(invoices, po_grn)
    linked = build_links(inv_agg, po)
    feats = engineer_features(linked)

    row = feats[(feats["invoice_id"] == invoice_id) & (feats["po_number"] == po_number)].copy()
    if row.empty:
        return {"found": False, "message": "No feature row for given (invoice_id, po_number)."}

    if hasattr(clf, '_features_used'):
        feature_cols = clf._features_used
    else:
        feature_cols = []
        for col in FEATURE_COLS:
            if col in feats.columns:
                if feats[col].nunique() > 1:
                    feature_cols.append(col)
    
    # Use only the selected features for prediction
    X = row[feature_cols].astype(float).values
    proba = float(clf.predict_proba(X)[:,1][0])
    pred = int(proba >= 0.5)
    status = "mismatch" if pred == 1 else "match"

    facts = {
        "amount_delta": float(row["amount_delta_abs"].iloc[0]) if "amount_delta_abs" in row else 0.0,
        "vendor_match": bool(int(row["vendor_match"].iloc[0])) if "vendor_match" in row else True,
        "po_missing": bool(int(row["po_missing"].iloc[0])) if "po_missing" in row else False,
        "has_grn": bool(int(row["has_grn"].iloc[0])) if "has_grn" in row else True,
        "days_delta": float(row["days_delta"].iloc[0]) if "days_delta" in row else 0.0,
    }
    return {
        "found": True,
        "status": status,
        "confidence": proba,
        "facts": facts,
    }
