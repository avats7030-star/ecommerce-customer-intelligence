import json

with open('c:/Users/DELL/OneDrive/Documents/FDS/ecommerce-customer-intelligence/reports/column_inspection_report.json') as f:
    data = json.load(f)

for table in data:
    print(f"=== {table['file']} ({table['rows']:,} rows, {table['cols']} cols) ===")
    for c in table['columns']:
        print(f"  - {c['column']}: {c['null_count']:,} nulls ({c['null_pct']}%) | Sample: {c['samples']}")
    print()
