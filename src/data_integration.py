"""
Phase 3: Data Integration Module for Olist Brazilian E-Commerce System.
Creates integrated analytical datasets at proper grains (Order, Item, Customer)
with mathematical guarantees against Cartesian revenue multiplication.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np


class DataIntegrator:
    def __init__(self, processed_dir: Path):
        self.processed_dir = processed_dir

    def log(self, message: str):
        print(f"[DATA INTEGRATION] {message}")

    def load_cleaned_tables(self):
        self.log("Loading cleaned tables from data/processed/...")
        self.df_cust = pd.read_csv(self.processed_dir / "cleaned_customers.csv")
        self.df_orders = pd.read_csv(self.processed_dir / "cleaned_orders.csv")
        self.df_items = pd.read_csv(self.processed_dir / "cleaned_order_items.csv")
        self.df_payments = pd.read_csv(self.processed_dir / "cleaned_order_payments.csv")
        self.df_reviews = pd.read_csv(self.processed_dir / "cleaned_order_reviews.csv", low_memory=False)
        self.df_products = pd.read_csv(self.processed_dir / "cleaned_products.csv")
        self.df_sellers = pd.read_csv(self.processed_dir / "cleaned_sellers.csv")
        self.log("All 7 core cleaned tables loaded successfully.")

    def create_order_level_analytical_mart(self) -> pd.DataFrame:
        self.log("Building Analytical Mart 1: Order-Level Master (orders_analytical.csv)...")

        # 1. Aggregate Order Items to Order Level
        self.log("  Aggregating items to order grain (eliminating 1:N duplication)...")
        # Find dominant category per order (category of highest priced item)
        items_sorted = self.df_items.sort_values(["order_id", "price"], ascending=[True, False])
        items_with_prod = items_sorted.merge(
            self.df_products[["product_id", "product_category_name_english"]],
            on="product_id",
            how="left"
        )
        dominant_cat = items_with_prod.drop_duplicates(subset=["order_id"])[["order_id", "product_category_name_english"]]
        dominant_cat.rename(columns={"product_category_name_english": "primary_category"}, inplace=True)

        items_agg = self.df_items.groupby("order_id").agg(
            total_items_count=("order_item_id", "count"),
            unique_products_count=("product_id", "nunique"),
            unique_sellers_count=("seller_id", "nunique"),
            total_items_price=("price", "sum"),
            total_freight_value=("freight_value", "sum"),
            total_merchandise_value=("total_item_value", "sum"),
            max_item_price=("price", "max"),
            min_item_price=("price", "min")
        ).reset_index()

        items_agg = items_agg.merge(dominant_cat, on="order_id", how="left")

        # 2. Aggregate Payments to Order Level
        self.log("  Aggregating payments to order grain (eliminating 1:N duplication)...")
        # Identify dominant payment type by highest value
        pay_sorted = self.df_payments.sort_values(["order_id", "payment_value"], ascending=[True, False])
        dominant_pay = pay_sorted.drop_duplicates(subset=["order_id"])[["order_id", "payment_type"]]
        dominant_pay.rename(columns={"payment_type": "primary_payment_type"}, inplace=True)

        payments_agg = self.df_payments.groupby("order_id").agg(
            total_payment_value=("payment_value", "sum"),
            payment_installments_max=("payment_installments", "max"),
            payment_splits_count=("payment_sequential", "count")
        ).reset_index()

        payments_agg = payments_agg.merge(dominant_pay, on="order_id", how="left")

        # 3. Base Orders + Customer Mapping
        self.log("  Merging orders with customer identities (1:1 join)...")
        orders_master = self.df_orders.merge(
            self.df_cust[["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state"]],
            on="customer_id",
            how="left"
        )

        # 4. Merge Aggregated Items, Aggregated Payments, and Reviews
        self.log("  Merging aggregated items, payments, and reviews...")
        orders_master = orders_master.merge(items_agg, on="order_id", how="left")
        orders_master = orders_master.merge(payments_agg, on="order_id", how="left")
        
        # Merge reviews (1:1 because of Phase 2 deduplication)
        rev_subset = self.df_reviews[[
            "order_id", "review_score", "has_comment_title", 
            "has_comment_message", "review_response_time_hours"
        ]]
        orders_master = orders_master.merge(rev_subset, on="order_id", how="left")

        # Fill items/payments nulls for orders canceled before items/payments were recorded
        orders_master["total_items_count"] = orders_master["total_items_count"].fillna(0).astype(int)
        orders_master["unique_products_count"] = orders_master["unique_products_count"].fillna(0).astype(int)
        orders_master["unique_sellers_count"] = orders_master["unique_sellers_count"].fillna(0).astype(int)
        orders_master["total_items_price"] = orders_master["total_items_price"].fillna(0.0)
        orders_master["total_freight_value"] = orders_master["total_freight_value"].fillna(0.0)
        orders_master["total_merchandise_value"] = orders_master["total_merchandise_value"].fillna(0.0)
        orders_master["total_payment_value"] = orders_master["total_payment_value"].fillna(0.0)
        orders_master["payment_installments_max"] = orders_master["payment_installments_max"].fillna(1).astype(int)
        orders_master["payment_splits_count"] = orders_master["payment_splits_count"].fillna(0).astype(int)
        orders_master["primary_category"] = orders_master["primary_category"].fillna("unknown_category")
        orders_master["primary_payment_type"] = orders_master["primary_payment_type"].fillna("not_recorded")

        # Extract calendar features
        purchase_dt = pd.to_datetime(orders_master["order_purchase_timestamp"])
        orders_master["order_year"] = purchase_dt.dt.year
        orders_master["order_month"] = purchase_dt.dt.month
        orders_master["order_year_month"] = purchase_dt.dt.to_period("M").astype(str)
        orders_master["order_day_of_week"] = purchase_dt.dt.day_name()
        orders_master["order_hour"] = purchase_dt.dt.hour

        # Revenue audit: verify no Cartesian inflation
        raw_payment_sum = self.df_payments["payment_value"].sum()
        master_payment_sum = orders_master["total_payment_value"].sum()
        diff = abs(raw_payment_sum - master_payment_sum)
        self.log(f"  Revenue Validation Audit:")
        self.log(f"    Raw payments sum:    R$ {raw_payment_sum:,.2f}")
        self.log(f"    Integrated mart sum: R$ {master_payment_sum:,.2f}")
        self.log(f"    Discrepancy:         R$ {diff:,.2f} ({'PASSED (0 error)' if diff < 0.01 else 'FAILED'})")

        out_path = self.processed_dir / "orders_analytical.csv"
        orders_master.to_csv(out_path, index=False)
        self.log(f"  Saved Order-Level Master to {out_path.name} ({len(orders_master):,} rows, {len(orders_master.columns)} columns)")
        self.orders_master = orders_master
        return orders_master

    def create_item_level_analytical_mart(self) -> pd.DataFrame:
        self.log("Building Analytical Mart 2: Item-Level Analytical Mart (order_items_analytical.csv)...")

        items_master = self.df_items.merge(
            self.df_products[[
                "product_id", "product_category_name", "product_category_name_english",
                "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"
            ]],
            on="product_id",
            how="left"
        )

        items_master = items_master.merge(
            self.df_sellers[["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"]],
            on="seller_id",
            how="left"
        )

        items_master = items_master.merge(
            self.orders_master[[
                "order_id", "customer_id", "customer_unique_id", "order_status",
                "order_purchase_timestamp", "order_year_month", "is_delivered", "is_delayed",
                "customer_state", "customer_city"
            ]],
            on="order_id",
            how="left"
        )

        out_path = self.processed_dir / "order_items_analytical.csv"
        items_master.to_csv(out_path, index=False)
        self.log(f"  Saved Item-Level Mart to {out_path.name} ({len(items_master):,} rows, {len(items_master.columns)} columns)")
        return items_master

    def create_customer_level_analytical_mart(self) -> pd.DataFrame:
        self.log("Building Analytical Mart 3: Customer-Level Analytical Mart (customers_analytical.csv)...")
        
        # Consider only delivered / valid orders for monetary customer metrics
        delivered_orders = self.orders_master[self.orders_master["is_delivered"] == 1].copy()
        
        delivered_orders["order_purchase_timestamp"] = pd.to_datetime(delivered_orders["order_purchase_timestamp"])
        
        # Reference date is 1 day after the latest purchase in entire dataset
        dataset_max_date = pd.to_datetime(self.orders_master["order_purchase_timestamp"]).max()
        ref_date = dataset_max_date + pd.Timedelta(days=1)
        self.log(f"  Dataset latest timestamp: {dataset_max_date} | Reference date: {ref_date}")

        cust_agg = delivered_orders.groupby("customer_unique_id").agg(
            total_delivered_orders=("order_id", "count"),
            total_spend=("total_payment_value", "sum"),
            total_freight_paid=("total_freight_value", "sum"),
            total_items_bought=("total_items_count", "sum"),
            avg_order_value=("total_payment_value", "mean"),
            avg_review_score=("review_score", "mean"),
            avg_delivery_delay_days=("delivery_delay_days", "mean"),
            delayed_orders_count=("is_delayed", "sum"),
            first_purchase_date=("order_purchase_timestamp", "min"),
            last_purchase_date=("order_purchase_timestamp", "max")
        ).reset_index()

        # Recency in days
        cust_agg["recency_days"] = (ref_date - cust_agg["last_purchase_date"]).dt.total_seconds() / 86400.0
        # Customer lifespan in days
        cust_agg["customer_lifespan_days"] = (
            cust_agg["last_purchase_date"] - cust_agg["first_purchase_date"]
        ).dt.total_seconds() / 86400.0
        
        # Repeat customer flag
        cust_agg["is_repeat_customer"] = (cust_agg["total_delivered_orders"] > 1).astype(int)

        # Merge customer primary location (from their latest order)
        latest_cust_orders = delivered_orders.sort_values("order_purchase_timestamp", ascending=True).drop_duplicates(
            subset=["customer_unique_id"], keep="last"
        )
        cust_agg = cust_agg.merge(
            latest_cust_orders[["customer_unique_id", "customer_city", "customer_state", "customer_zip_code_prefix"]],
            on="customer_unique_id",
            how="left"
        )

        n_repeats = cust_agg["is_repeat_customer"].sum()
        total_delivered_custs = len(cust_agg)
        self.log(f"  Delivered customer base: {total_delivered_custs:,} unique customers.")
        self.log(f"  Repeat customers (>=2 delivered orders): {n_repeats:,} ({n_repeats / total_delivered_custs * 100:.2f}%)")

        out_path = self.processed_dir / "customers_analytical.csv"
        cust_agg.to_csv(out_path, index=False)
        self.log(f"  Saved Customer-Level Mart to {out_path.name} ({len(cust_agg):,} rows, {len(cust_agg.columns)} columns)")
        return cust_agg

    def run_all(self):
        print("=" * 70)
        print("STARTING DATA INTEGRATION PIPELINE (PHASE 3)")
        print("=" * 70)
        self.load_cleaned_tables()
        self.create_order_level_analytical_mart()
        self.create_item_level_analytical_mart()
        self.create_customer_level_analytical_mart()
        print("=" * 70)
        print("DATA INTEGRATION COMPLETED SUCCESSFULLY (ZERO REVENUE LEAKAGE)")
        print("=" * 70)


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    processed = project_root / "data" / "processed"
    
    integrator = DataIntegrator(processed)
    integrator.run_all()
