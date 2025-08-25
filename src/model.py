from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

def get_sklearn_modules():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, precision_recall_fscore_support
    return LogisticRegression, train_test_split, classification_report, precision_recall_fscore_support

FEATURE_COLS = [
    "vendor_match",
    "vendor_similarity", 
    "has_grn",
    "amount_delta_abs",
    "amount_delta_pct",
    "amount_over_tolerance",
    "amount_pct_over_tolerance",
    "days_delta",
    "days_since_grn",
    "invoice_before_po",
    "invoice_too_late",
    "invoice_before_grn",
    "po_missing",
    "currency_match",
]

def train_eval(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
              ) -> Tuple[Dict[str, Any], object]:
    # Import sklearn modules only when needed
    LogisticRegression, train_test_split, classification_report, precision_recall_fscore_support = get_sklearn_modules()
    
    df = df.dropna(subset=["is_mismatch"]).copy()

    # Feature selection - remove constant features
    feature_cols = []
    for col in FEATURE_COLS:
        if col in df.columns:
            if df[col].nunique() > 1:
                feature_cols.append(col)
            else:
                print(f"Warning: Dropping constant feature '{col}'")
    
    if len(feature_cols) == 0:
        raise ValueError("No valid features found! All features are constant.")
    
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    
    X = df[feature_cols].astype(float).values
    y = df["is_mismatch"].astype(int).values

    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Reduce false positives with better regularization and calibration
    clf = LogisticRegression(
        max_iter=2000,
        class_weight={0: 1, 1: 2},  # Less aggressive than "balanced"
        random_state=random_state,
        C=10.0,  # Increased regularization to reduce overfitting
        solver='liblinear',
        penalty='l1'  # L1 regularization for feature selection
    )
    clf.fit(X_train, y_train)
    
    # Store the features used for training
    clf._features_used = feature_cols

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]

    report = classification_report(y_test, y_pred, output_dict=True, digits=4)

    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", pos_label=1)
    
    feature_importance = dict(zip(feature_cols, clf.coef_[0]))
    feature_importance = {k: float(v) for k, v in sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)}
    
    metrics = {
        "precision_pos": float(p),
        "recall_pos": float(r),
        "f1_pos": float(f1),
        "report": report,
        "threshold": 0.5,
        "feature_importance": feature_importance,
        "features_used": feature_cols,
        "n_features": len(feature_cols)
    }

    out = pd.DataFrame(X_test, columns=feature_cols)
    out["y_true"] = y_test
    out["y_pred"] = y_pred
    out["y_score"] = y_proba
    
    return metrics, clf

def get_joblib_modules():
    from joblib import dump, load
    return dump, load

def save_model(clf, path: str) -> None:
    dump, _ = get_joblib_modules()
    model_info = {
        'model': clf,
        'features_used': getattr(clf, '_features_used', FEATURE_COLS)
    }
    dump(model_info, path)

def load_model(path: str):
    _, load = get_joblib_modules()
    model_info = load(path)
    if isinstance(model_info, dict):
        model = model_info['model']
        model._features_used = model_info.get('features_used', FEATURE_COLS)
        return model
    else:
        model_info._features_used = FEATURE_COLS
        return model_info
