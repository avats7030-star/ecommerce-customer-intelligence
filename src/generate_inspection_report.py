"""
Detailed column-by-column inspection generator for all 9 Olist CSV files.
Computes null counts, data types, distinct values, and sample values.
"""

import csv
from pathlib import Path
import json

RAW_DIR = Path("c:/Users/DELL/OneDrive/Documents/FDS/ecommerce-customer-intelligence/data/raw")

files = [
    "olist_customers_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv"
]

def analyze_file(fname):
    path = RAW_DIR / fname
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_count = len(header)
        
        row_count = 0
        null_count = [0] * col_count
        sample_vals = [set() for _ in range(col_count)]
        
        for row in reader:
            row_count += 1
            for idx, val in enumerate(row):
                if idx < col_count:
                    v = val.strip()
                    if v == "" or v.lower() in ("null", "none", "nan"):
                        null_count[idx] += 1
                    else:
                        if len(sample_vals[idx]) < 5:
                            sample_vals[idx].add(v)
                            
        col_stats = []
        for i, col in enumerate(header):
            nulls = null_count[i]
            pct = round((nulls / row_count) * 100, 3) if row_count > 0 else 0
            # Infer basic type from sample
            samples = list(sample_vals[i])
            col_stats.append({
                "column": col,
                "null_count": nulls,
                "null_pct": pct,
                "samples": samples[:3]
            })
            
        return {
            "file": fname,
            "rows": row_count,
            "cols": col_count,
            "columns": col_stats
        }

results = []
for f in files:
    print(f"Analyzing {f}...")
    res = analyze_file(f)
    results.append(res)

out_path = Path("c:/Users/DELL/OneDrive/Documents/FDS/ecommerce-customer-intelligence/reports/column_inspection_report.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Saved detailed report to:", out_path)
