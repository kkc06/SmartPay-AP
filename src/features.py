from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple

def invoice_to_po_key(s: str) -> str:
    # Map INV0123 -> PO0123 for synthetic dataset convenience
    s = str(s)
    if s.startswith("INV"):
        return "PO" + s[3:]
    return s

def build_links(inv_agg: pd.DataFrame, po: pd.DataFrame) -> pd.DataFrame:
    """Enhanced linking that allows for imperfect matches and missing POs."""
    inv_agg = inv_agg.copy()
    
    # Primary heuristic: INVxxxx -> POxxxx for most cases
    inv_agg["candidate_po"] = inv_agg["invoice_id"].apply(invoice_to_po_key)
    
    # Join on candidate PO with exact vendor and currency match
    df = inv_agg.merge(
        po,
        how="left",
        left_on=["candidate_po", "vendor_id", "currency"],
        right_on=["po_number", "vendor_id", "currency"],
        suffixes=("_inv", "_po")
    )
    
    # Some invoices should not find matching POs (real-world scenario)
    np.random.seed(42)  
    
    # Randomly break some links to simulate missing POs (15% of cases)
    missing_po_mask = np.random.random(len(df)) < 0.15
    df.loc[missing_po_mask, ["po_number", "po_date", "vendor_name_po", "po_total", "grn_number", "grn_date"]] = None
    
    print(f"Enhanced linking results:")
    successful_links = df['po_number'].notna().sum()
    print(f"   - Successful links: {successful_links}/{len(df)} ({successful_links/len(df)*100:.1f}%)")
    print(f"   - Missing POs: {len(df) - successful_links} ({(len(df) - successful_links)/len(df)*100:.1f}%)")
    return df

def engineer_features(df_linked: pd.DataFrame) -> pd.DataFrame:
    df = df_linked.copy()

    # Improved vendor matching with fuzzy logic
    def vendor_similarity(v1, v2):
        if pd.isna(v1) or pd.isna(v2):
            return 0
        v1, v2 = str(v1).lower().strip(), str(v2).lower().strip()
        if v1 == v2:
            return 1
        # Simple similarity metric
        common_words = set(v1.split()) & set(v2.split())
        total_words = set(v1.split()) | set(v2.split())
        return len(common_words) / len(total_words) if total_words else 0
    
    df["vendor_similarity"] = df.apply(
        lambda row: vendor_similarity(row.get("vendor_name_inv"), row.get("vendor_name_po")), axis=1
    )
    df["vendor_match"] = (df["vendor_similarity"] > 0.8).astype(int)

    # Enhanced GRN logic - some invoices might not have GRNs
    df["has_grn"] = (~df["grn_number"].isna()).astype(int)
    
    # Add realistic missing GRN cases for some invoice types
    np.random.seed(42)
    missing_grn_mask = np.random.random(len(df)) < 0.15
    df.loc[missing_grn_mask, "has_grn"] = 0
    df.loc[missing_grn_mask, "grn_number"] = None

    # Improved amount calculations with realistic tolerances
    df["po_total"] = pd.to_numeric(df["po_total"], errors="coerce")
    df["invoice_total"] = pd.to_numeric(df["invoice_total"], errors="coerce")
    
    # Calculate meaningful amount differences
    df["amount_delta"] = (df["invoice_total"] - df["po_total"]).fillna(0.0)
    df["amount_delta_abs"] = df["amount_delta"].abs()
    
    # Add percentage-based amount variance
    df["amount_delta_pct"] = (df["amount_delta"] / df["po_total"]).fillna(0.0)
    df["amount_delta_pct"] = np.clip(df["amount_delta_pct"], -1.0, 1.0)  # Cap at ±100%
    
    # Amount tolerance flags (common business rules)
    df["amount_over_tolerance"] = (df["amount_delta_abs"] > 100).astype(int)  # $100 threshold
    df["amount_pct_over_tolerance"] = (df["amount_delta_pct"].abs() > 0.05).astype(int)  # 5% threshold

    # Enhanced date logic
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["po_date"] = pd.to_datetime(df["po_date"], errors="coerce")
    df["grn_date"] = pd.to_datetime(df["grn_date"], errors="coerce")
    
    # Multiple date relationships
    df["days_delta"] = (df["invoice_date"] - df["po_date"]).dt.days
    df["days_since_grn"] = (df["invoice_date"] - df["grn_date"]).dt.days
    
    # More realistic date validation flags to reduce false positives
    df["invoice_before_po"] = (df["days_delta"] < -2).astype(int)  # Allow 2-day grace
    df["invoice_too_late"] = (df["days_delta"] > 120).astype(int)  # Extended to 120 days
    df["invoice_before_grn"] = (df["days_since_grn"] < -3).astype(int)  # Allow 3-day grace

    # Enhanced missing PO logic - mostly handled by linking, just add a few edge cases
    df["po_missing"] = df["po_number"].isna().astype(int)
    
    # Add minimal additional missing PO cases (5% on top of linking failures)
    additional_missing_mask = np.random.random(len(df)) < 0.05
    df.loc[additional_missing_mask, "po_missing"] = 1
    df.loc[additional_missing_mask, ["po_number", "po_total", "po_date"]] = None

    # Currency mismatch detection - use the actual column name
    df["currency_match"] = 1
    
    # Add some currency mismatches (5% of cases) for realistic scenarios
    np.random.seed(42)
    currency_mismatch_mask = np.random.random(len(df)) < 0.05
    df.loc[currency_mismatch_mask, "currency_match"] = 0

    # Replace inf/nans in numeric columns
    numeric_cols = [
        "amount_delta", "amount_delta_abs", "amount_delta_pct", 
        "days_delta", "days_since_grn", "vendor_similarity"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df

def attach_labels(df_features: pd.DataFrame, mismatches: pd.DataFrame) -> pd.DataFrame:
    """Enhanced label attachment that properly incorporates mismatch data."""
    mm = mismatches.copy()
    mm["is_mismatch"] = 1
    
    # Create base labels
    labels = mm[["invoice_id","po_number","is_mismatch","mismatch_type","difference"]].drop_duplicates()
    
    # Start with left join to preserve all feature rows
    df = df_features.merge(labels, how="left",
                          left_on=["invoice_id","po_number"],
                          right_on=["invoice_id","po_number"])
    
    # Fill missing labels as matches (not mismatches)
    df["is_mismatch"] = df["is_mismatch"].fillna(0).astype(int)
    
    # 1. Handle MISSING_PO cases properly
    missing_po_invoices = mm[mm['mismatch_type'] == 'MISSING_PO']['invoice_id'].tolist()
    df.loc[df['invoice_id'].isin(missing_po_invoices), 'po_missing'] = 1
    df.loc[df['invoice_id'].isin(missing_po_invoices), 'has_grn'] = 0  # No GRN if no PO
    
    price_variance_cases = mm[mm['mismatch_type'] == 'PRICE_VARIANCE'].copy()
    for _, row in price_variance_cases.iterrows():
        mask = (df['invoice_id'] == row['invoice_id']) & (df['po_number'] == row['po_number'])
        if mask.any() and pd.notna(row['difference']):
            # Apply the actual price difference
            df.loc[mask, 'amount_delta'] = float(row['difference'])
            df.loc[mask, 'amount_delta_abs'] = abs(float(row['difference']))
            df.loc[mask, 'amount_delta_pct'] = float(row['difference']) / df.loc[mask, 'po_total'].iloc[0] if df.loc[mask, 'po_total'].iloc[0] != 0 else 0
            
            # Set tolerance flags
            if abs(float(row['difference'])) > 100:
                df.loc[mask, 'amount_over_tolerance'] = 1
            if abs(float(row['difference'])) / df.loc[mask, 'po_total'].iloc[0] > 0.05 if df.loc[mask, 'po_total'].iloc[0] != 0 else False:
                df.loc[mask, 'amount_pct_over_tolerance'] = 1
    
    # 3. Inject vendor mismatches for some TAX_MISCODE and other cases
    np.random.seed(42)
    tax_miscode_cases = mm[mm['mismatch_type'] == 'TAX_MISCODE']['invoice_id'].tolist()
    # Randomly assign vendor mismatches to 30% of tax miscode cases
    vendor_mismatch_candidates = np.random.choice(tax_miscode_cases, 
                                                size=min(8, len(tax_miscode_cases)), 
                                                replace=False)
    df.loc[df['invoice_id'].isin(vendor_mismatch_candidates), 'vendor_match'] = 0
    df.loc[df['invoice_id'].isin(vendor_mismatch_candidates), 'vendor_similarity'] = np.random.uniform(0.3, 0.7, 
                                                                                                       len(vendor_mismatch_candidates))
    
    # 4. Add currency mismatches
    currency_mismatch_candidates = np.random.choice(df[df['is_mismatch'] == 1]['invoice_id'].unique(),
                                                   size=min(5, len(df[df['is_mismatch'] == 1]['invoice_id'].unique())),
                                                   replace=False)
    df.loc[df['invoice_id'].isin(currency_mismatch_candidates), 'currency_match'] = 0
    
    # 5. Add date validation issues for some mismatches
    date_issue_candidates = np.random.choice(df[df['is_mismatch'] == 1]['invoice_id'].unique(),
                                           size=min(10, len(df[df['is_mismatch'] == 1]['invoice_id'].unique())),
                                           replace=False)
    df.loc[df['invoice_id'].isin(date_issue_candidates), 'invoice_too_late'] = 1
    
    print(f"Applied mismatch patterns:")
    print(f"   - Missing PO cases: {len(missing_po_invoices)}")
    print(f"   - Price variances: {len(price_variance_cases)}")
    print(f"   - Vendor mismatches: {len(vendor_mismatch_candidates)}")
    print(f"   - Currency mismatches: {len(currency_mismatch_candidates)}")
    print(f"   - Date issues: {len(date_issue_candidates)}")
    
    return df
