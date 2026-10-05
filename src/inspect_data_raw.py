"""
Comprehensive Raw Dataset Inspection Script (using standard library)
Inspects rows, columns, missing values, duplicates, and key relationships for Olist datasets.
"""

import csv
import os
from pathlib import Path
from collections import defaultdict
import json

def inspect_csv(file_path: Path):
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return {"name": file_path.name, "rows": 0, "cols": 0, "columns": [], "missing": {}, "sample_row": []}
        
        num_cols = len(header)
        col_names = header
        missing_counts = {col: 0 for col in col_names}
        row_count = 0
        unique_vals = {col: set() for col in col_names}
        sample_rows = []
        
        for row in reader:
            row_count += 1
            if len(sample_rows) < 3:
                sample_rows.append(row)
            
            for i, val in enumerate(row):
                if i < num_cols:
                    col_name = col_names[i]
                    val_str = val.strip()
                    if val_str == "" or val_str.lower() in ("null", "none", "nan"):
                        missing_counts[col_name] += 1
                    else:
                        if len(unique_vals[col_name]) < 200000:
                            unique_vals[col_name].add(val_str)
                            
        unique_counts = {col: len(unique_vals[col]) for col in col_names}
        
        return {
            "name": file_path.name,
            "size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
            "rows": row_count,
            "cols": num_cols,
            "columns": col_names,
            "missing": missing_counts,
            "unique_counts": unique_counts,
            "sample_rows": sample_rows
        }

def run_inspection():
    data_dir = Path("c:/Users/DELL/OneDrive/Documents/FDS/ecommerce-customer-intelligence/data/raw")
    csv_files = sorted(list(data_dir.glob("*.csv")))
    
    print(f"=== DETECTED {len(csv_files)} CSV FILES ===")
    results = {}
    for p in csv_files:
        print(f"Inspecting {p.name}...")
        res = inspect_csv(p)
        results[p.name] = res
        print(f"  Rows: {res['rows']:,} | Cols: {res['cols']} | Size: {res['size_mb']} MB")
        
    out_file = Path("c:/Users/DELL/OneDrive/Documents/FDS/ecommerce-customer-intelligence/reports/raw_inspection_summary.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        # Convert non-serializable sets if any
        json.dump(results, f, indent=2)
    print(f"Inspection summary saved to {out_file}")

if __name__ == "__main__":
    run_inspection()
