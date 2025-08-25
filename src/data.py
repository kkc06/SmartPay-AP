from __future__ import annotations
import pandas as pd
from dateutil import parser
from typing import Tuple

def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    invoices = pd.read_csv(f"{data_dir}/invoices.csv")
    po_grn = pd.read_csv(f"{data_dir}/po_grn.csv")
    mismatches = pd.read_csv(f"{data_dir}/labelled_mismatches.csv")
    return invoices, po_grn, mismatches

# Normalize date formats and aggregate invoice data
def normalize_types(df_invoices: pd.DataFrame, df_po: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    inv = df_invoices.copy()
    po = df_po.copy()

    # Dates: try multiple formats gracefully
    def parse_date_safe(s: str):
        try:
            return parser.parse(str(s), dayfirst=True, yearfirst=False, fuzzy=True)
        except Exception:
            return pd.NaT

    inv["invoice_date"] = inv["invoice_date"].apply(parse_date_safe)
    po["po_date"] = po["po_date"].apply(parse_date_safe)

    inv_agg = (
        inv.groupby(["invoice_id", "vendor_id", "vendor_name", "currency"])
           .agg(invoice_total=("line_total", "sum"),
                line_count=("line_item_number","count"),
                max_qty=("quantity","max"),
                avg_unit_price=("unit_price","mean"),
                invoice_date=("invoice_date", "first"))
           .reset_index()
    )
    return inv_agg, po
