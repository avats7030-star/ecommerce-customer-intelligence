"""
In-depth Dataset Relationship and Quality Inspector.
Computes primary keys, foreign keys, cardinality, duplicates, date ranges, and quality issues.
"""

import csv
import json
from pathlib import Path
from collections import Counter, defaultdict

RAW_DIR = Path("c:/Users/DELL/OneDrive/Documents/FDS/ecommerce-customer-intelligence/data/raw")

def load_csv_dict(file_name):
    path = RAW_DIR / file_name
    rows = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

print("Loading raw files for relational audit...")
customers = load_csv_dict("olist_customers_dataset.csv")
orders = load_csv_dict("olist_orders_dataset.csv")
items = load_csv_dict("olist_order_items_dataset.csv")
payments = load_csv_dict("olist_order_payments_dataset.csv")
reviews = load_csv_dict("olist_order_reviews_dataset.csv")
products = load_csv_dict("olist_products_dataset.csv")
sellers = load_csv_dict("olist_sellers_dataset.csv")
translations = load_csv_dict("product_category_name_translation.csv")

print("Auditing Customers...")
cust_id_set = {r["customer_id"] for r in customers}
cust_unique_id_set = {r["customer_unique_id"] for r in customers}
cust_unique_counts = Counter(r["customer_unique_id"] for r in customers)
repeat_cust_count = sum(1 for c, cnt in cust_unique_counts.items() if cnt > 1)
max_cust_orders = max(cust_unique_counts.values())

print("Auditing Orders...")
order_id_set = {r["order_id"] for r in orders}
order_status_counts = Counter(r["order_status"] for r in orders)
order_dates = [r["order_purchase_timestamp"] for r in orders if r["order_purchase_timestamp"]]
min_order_date = min(order_dates) if order_dates else None
max_order_date = max(order_dates) if order_dates else None

# Check customer_id foreign key in orders
order_cust_ids = {r["customer_id"] for r in orders}
missing_cust_in_orders = order_cust_ids - cust_id_set
orders_missing_customers = cust_id_set - order_cust_ids

print("Auditing Order Items...")
item_order_ids = {r["order_id"] for r in items}
item_prod_ids = {r["product_id"] for r in items}
item_seller_ids = {r["seller_id"] for r in items}
item_per_order = Counter(r["order_id"] for r in items)
max_items_in_order = max(item_per_order.values()) if item_per_order else 0

print("Auditing Payments...")
payment_order_ids = {r["order_id"] for r in payments}
payment_types = Counter(r["payment_type"] for r in payments)
payments_per_order = Counter(r["order_id"] for r in payments)
max_payments_in_order = max(payments_per_order.values()) if payments_per_order else 0

print("Auditing Reviews...")
review_order_ids = {r["order_id"] for r in reviews}
review_scores = Counter(r["review_score"] for r in reviews)
reviews_per_order = Counter(r["order_id"] for r in reviews)
max_reviews_in_order = max(reviews_per_order.values()) if reviews_per_order else 0

print("Auditing Products & Categories...")
prod_id_set = {r["product_id"] for r in products}
prod_categories = Counter(r["product_category_name"] for r in products)
trans_dict = {r["product_category_name"]: r["product_category_name_english"] for r in translations}
categories_in_products = set(prod_categories.keys()) - {""}
categories_without_english = categories_in_products - set(trans_dict.keys())

print("Auditing Sellers...")
seller_id_set = {r["seller_id"] for r in sellers}

print("Auditing Geolocation...")
# Read geolocation lightly without holding 1M dicts
geo_rows = 0
geo_zip_set = set()
geo_coords = set()
with open(RAW_DIR / "olist_geolocation_dataset.csv", "r", encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f)
    for r in reader:
        geo_rows += 1
        geo_zip_set.add(r["geolocation_zip_code_prefix"])

audit_results = {
    "customers": {
        "total_records": len(customers),
        "unique_customer_ids (order session key)": len(cust_id_set),
        "unique_customer_unique_ids (actual human)": len(cust_unique_id_set),
        "repeat_customers_count": repeat_cust_count,
        "repeat_customer_pct": round(repeat_cust_count / len(cust_unique_id_set) * 100, 2),
        "max_orders_by_single_customer": max_cust_orders
    },
    "orders": {
        "total_records": len(orders),
        "unique_order_ids": len(order_id_set),
        "date_range": {
            "earliest_purchase": min_order_date,
            "latest_purchase": max_order_date
        },
        "order_status_distribution": dict(order_status_counts),
        "foreign_key_match_with_customers": {
            "orders_with_valid_customer_id": len(order_cust_ids - missing_cust_in_orders),
            "unmatched_customer_ids": len(missing_cust_in_orders)
        }
    },
    "order_items": {
        "total_records": len(items),
        "unique_orders_represented": len(item_order_ids),
        "orders_in_orders_table_without_items": len(order_id_set - item_order_ids),
        "unique_products_sold": len(item_prod_ids),
        "unique_sellers_represented": len(item_seller_ids),
        "max_items_in_single_order": max_items_in_order,
        "products_in_items_not_in_products_table": len(item_prod_ids - prod_id_set),
        "sellers_in_items_not_in_sellers_table": len(item_seller_ids - seller_id_set)
    },
    "order_payments": {
        "total_records": len(payments),
        "unique_orders_represented": len(payment_order_ids),
        "orders_in_orders_table_without_payments": len(order_id_set - payment_order_ids),
        "max_payment_rows_per_order": max_payments_in_order,
        "payment_type_distribution": dict(payment_types)
    },
    "order_reviews": {
        "total_records": len(reviews),
        "unique_orders_represented": len(review_order_ids),
        "orders_without_reviews": len(order_id_set - review_order_ids),
        "max_reviews_per_order": max_reviews_in_order,
        "review_score_distribution": dict(review_scores)
    },
    "products": {
        "total_records": len(products),
        "unique_products": len(prod_id_set),
        "products_missing_category": prod_categories.get("", 0),
        "unique_portuguese_categories": len(categories_in_products),
        "portuguese_categories_missing_translation": list(categories_without_english)
    },
    "sellers": {
        "total_records": len(sellers),
        "unique_sellers": len(seller_id_set)
    },
    "geolocation": {
        "total_records": geo_rows,
        "unique_zip_prefixes": len(geo_zip_set),
        "duplicate_zip_warning": "1,000,163 rows represent only 19,015 zip prefixes (many lat/lng per zip code)"
    }
}

out_audit = Path("c:/Users/DELL/OneDrive/Documents/FDS/ecommerce-customer-intelligence/reports/relational_audit.json")
with open(out_audit, "w", encoding="utf-8") as f:
    json.dump(audit_results, f, indent=2)

print("Relational audit complete. Results saved.")
print(json.dumps(audit_results, indent=2))
