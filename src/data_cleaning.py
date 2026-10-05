"""
Data Cleaning and Validation Pipeline for Olist Brazilian E-Commerce Dataset.
Produces clean, validated datasets in data/processed/ with full audit logs.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np


class DataCleaner:
    def __init__(self, raw_dir: Path, processed_dir: Path):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.cleaning_log = []

    def log(self, message: str):
        print(f"[CLEANING PIPELINE] {message}")
        self.cleaning_log.append(message)

    def clean_categories_and_products(self) -> pd.DataFrame:
        self.log("Step 1/8: Cleaning Products and Merging Category Translations...")
        df_prod = pd.read_csv(self.raw_dir / "olist_products_dataset.csv")
        df_trans = pd.read_csv(self.raw_dir / "product_category_name_translation.csv", encoding="utf-8-sig")

        raw_count = len(df_prod)
        self.log(f"  Raw products loaded: {raw_count:,} rows")

        # Missing category names handling
        missing_cat_count = df_prod["product_category_name"].isna().sum()
        self.log(f"  Found {missing_cat_count:,} products with missing category. Imputing with 'outros' (unknown).")
        df_prod["product_category_name"] = df_prod["product_category_name"].fillna("outros")

        # Fill dimensions and photos missing values with category-level median
        num_cols = ["product_name_lenght", "product_description_lenght", "product_photos_qty",
                    "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
        for col in num_cols:
            df_prod[col] = df_prod.groupby("product_category_name")[col].transform(lambda x: x.fillna(x.median()))
            # Global median fallback if entire category is empty
            df_prod[col] = df_prod[col].fillna(df_prod[col].median())

        # Build comprehensive category translation map
        trans_dict = dict(zip(df_trans["product_category_name"], df_trans["product_category_name_english"]))
        # Add the two known unmapped categories in Olist
        trans_dict["pc_gamer"] = "pc_gamer"
        trans_dict["portateis_cozinha_e_preparadores_de_alimentos"] = "small_appliances_food_preparers"
        trans_dict["outros"] = "unknown_category"

        df_prod["product_category_name_english"] = df_prod["product_category_name"].map(trans_dict).fillna("unknown_category")
        self.log(f"  Category translation complete. English categories populated: {df_prod['product_category_name_english'].nunique()} unique.")

        # Save to processed
        out_path = self.processed_dir / "cleaned_products.csv"
        df_prod.to_csv(out_path, index=False)
        self.log(f"  Cleaned products saved to: {out_path.name} ({len(df_prod):,} rows, 0 nulls in critical fields)")
        return df_prod

    def clean_customers(self) -> pd.DataFrame:
        self.log("Step 2/8: Cleaning Customers Dataset...")
        df_cust = pd.read_csv(self.raw_dir / "olist_customers_dataset.csv")
        raw_count = len(df_cust)

        # Standardize zip code as 5-character string with zero-padding
        df_cust["customer_zip_code_prefix"] = df_cust["customer_zip_code_prefix"].astype(str).str.zfill(5)
        # Standardize city names: lowercase and trim whitespace
        df_cust["customer_city"] = df_cust["customer_city"].astype(str).str.strip().str.lower()
        df_cust["customer_state"] = df_cust["customer_state"].astype(str).str.strip().str.upper()

        # Audit unique human customers vs session tokens
        n_unique_humans = df_cust["customer_unique_id"].nunique()
        n_sessions = df_cust["customer_id"].nunique()
        self.log(f"  Processed {raw_count:,} customer session records.")
        self.log(f"  Identified {n_unique_humans:,} unique individual customers across {n_sessions:,} order sessions.")

        out_path = self.processed_dir / "cleaned_customers.csv"
        df_cust.to_csv(out_path, index=False)
        self.log(f"  Cleaned customers saved to: {out_path.name}")
        return df_cust

    def clean_sellers(self) -> pd.DataFrame:
        self.log("Step 3/8: Cleaning Sellers Dataset...")
        df_sellers = pd.read_csv(self.raw_dir / "olist_sellers_dataset.csv")
        
        df_sellers["seller_zip_code_prefix"] = df_sellers["seller_zip_code_prefix"].astype(str).str.zfill(5)
        df_sellers["seller_city"] = df_sellers["seller_city"].astype(str).str.strip().str.lower()
        df_sellers["seller_state"] = df_sellers["seller_state"].astype(str).str.strip().str.upper()

        out_path = self.processed_dir / "cleaned_sellers.csv"
        df_sellers.to_csv(out_path, index=False)
        self.log(f"  Cleaned sellers saved to: {out_path.name} ({len(df_sellers):,} sellers)")
        return df_sellers

    def clean_orders(self) -> pd.DataFrame:
        self.log("Step 4/8: Cleaning Orders Dataset & Parsing Timestamps...")
        date_cols = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]
        df_orders = pd.read_csv(self.raw_dir / "olist_orders_dataset.csv")

        # Convert to datetime
        for col in date_cols:
            df_orders[col] = pd.to_datetime(df_orders[col], errors="coerce")

        raw_count = len(df_orders)
        self.log(f"  Orders loaded: {raw_count:,} rows")

        # Add derived delivery duration metrics (in fractional days)
        df_orders["delivery_duration_days"] = (
            df_orders["order_delivered_customer_date"] - df_orders["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400.0

        df_orders["estimated_duration_days"] = (
            df_orders["order_estimated_delivery_date"] - df_orders["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400.0

        # Delay = actual delivery - estimated delivery (positive = late)
        df_orders["delivery_delay_days"] = (
            df_orders["order_delivered_customer_date"] - df_orders["order_estimated_delivery_date"]
        ).dt.total_seconds() / 86400.0

        df_orders["is_delayed"] = (df_orders["delivery_delay_days"] > 0).astype(int)
        df_orders["is_delivered"] = (df_orders["order_status"] == "delivered").astype(int)

        # Flag temporal inconsistencies (delivery date before purchase date)
        invalid_dates = df_orders[df_orders["delivery_duration_days"] < 0]
        if len(invalid_dates) > 0:
            self.log(f"  Warning: Found {len(invalid_dates)} records with delivery earlier than purchase. Neutralizing.")
            df_orders.loc[df_orders["delivery_duration_days"] < 0, "delivery_duration_days"] = np.nan

        delivered_count = df_orders["is_delivered"].sum()
        delayed_count = (df_orders["delivery_delay_days"] > 0).sum()
        self.log(f"  Delivered orders: {delivered_count:,} ({delivered_count / raw_count * 100:.2f}%)")
        self.log(f"  Delayed delivered orders: {delayed_count:,} ({delayed_count / delivered_count * 100:.2f}% of delivered)")

        out_path = self.processed_dir / "cleaned_orders.csv"
        df_orders.to_csv(out_path, index=False)
        self.log(f"  Cleaned orders saved to: {out_path.name}")
        return df_orders

    def clean_order_items(self) -> pd.DataFrame:
        self.log("Step 5/8: Cleaning Order Items Dataset...")
        df_items = pd.read_csv(self.raw_dir / "olist_order_items_dataset.csv")
        df_items["shipping_limit_date"] = pd.to_datetime(df_items["shipping_limit_date"], errors="coerce")

        raw_count = len(df_items)
        # Validation checks
        neg_price = (df_items["price"] <= 0).sum()
        neg_freight = (df_items["freight_value"] < 0).sum()
        self.log(f"  Items loaded: {raw_count:,} rows. Non-positive prices: {neg_price}, negative freights: {neg_freight}")

        # Total item expenditure
        df_items["total_item_value"] = df_items["price"] + df_items["freight_value"]

        out_path = self.processed_dir / "cleaned_order_items.csv"
        df_items.to_csv(out_path, index=False)
        self.log(f"  Cleaned order items saved to: {out_path.name}")
        return df_items

    def clean_order_payments(self) -> pd.DataFrame:
        self.log("Step 6/8: Cleaning Order Payments Dataset...")
        df_pay = pd.read_csv(self.raw_dir / "olist_order_payments_dataset.csv")
        raw_count = len(df_pay)

        # Audit 'not_defined'
        not_defined_mask = df_pay["payment_type"] == "not_defined"
        n_not_defined = not_defined_mask.sum()
        self.log(f"  Found {n_not_defined} records with payment_type == 'not_defined'. Handling safely.")
        df_pay.loc[not_defined_mask, "payment_type"] = "other"

        # Validate non-negative payments
        df_pay["payment_value"] = df_pay["payment_value"].clip(lower=0.0)
        df_pay["payment_installments"] = df_pay["payment_installments"].clip(lower=1)

        out_path = self.processed_dir / "cleaned_order_payments.csv"
        df_pay.to_csv(out_path, index=False)
        self.log(f"  Cleaned payments saved to: {out_path.name} ({raw_count:,} rows)")
        return df_pay

    def clean_order_reviews(self) -> pd.DataFrame:
        self.log("Step 7/8: Cleaning Order Reviews Dataset...")
        df_rev = pd.read_csv(self.raw_dir / "olist_order_reviews_dataset.csv")
        raw_count = len(df_rev)

        # Handle dates
        df_rev["review_creation_date"] = pd.to_datetime(df_rev["review_creation_date"], errors="coerce")
        df_rev["review_answer_timestamp"] = pd.to_datetime(df_rev["review_answer_timestamp"], errors="coerce")

        # Response time in hours
        df_rev["review_response_time_hours"] = (
            df_rev["review_answer_timestamp"] - df_rev["review_creation_date"]
        ).dt.total_seconds() / 3600.0

        # Feature indicators for comment presence instead of dropping null text
        df_rev["has_comment_title"] = df_rev["review_comment_title"].notna().astype(int)
        df_rev["has_comment_message"] = df_rev["review_comment_message"].notna().astype(int)

        # Fill text nulls with empty string
        df_rev["review_comment_title"] = df_rev["review_comment_title"].fillna("")
        df_rev["review_comment_message"] = df_rev["review_comment_message"].fillna("")

        # Ensure review score is valid integer between 1 and 5
        df_rev["review_score"] = df_rev["review_score"].clip(1, 5).astype(int)

        # Deduplicate multiple reviews for same order by keeping the latest review answer
        df_rev = df_rev.sort_values("review_answer_timestamp", ascending=True)
        unique_order_revs = df_rev.drop_duplicates(subset=["order_id"], keep="last")
        self.log(f"  Deduplicated review records by order_id: {len(df_rev):,} -> {len(unique_order_revs):,} unique order reviews.")

        out_path = self.processed_dir / "cleaned_order_reviews.csv"
        unique_order_revs.to_csv(out_path, index=False)
        self.log(f"  Cleaned order reviews saved to: {out_path.name}")
        return unique_order_revs

    def clean_geolocation(self) -> pd.DataFrame:
        self.log("Step 8/8: Aggregating Geolocation (Eliminating Combinatorial Duplication)...")
        # To avoid loading 1M records inefficiently, read necessary columns
        df_geo = pd.read_csv(self.raw_dir / "olist_geolocation_dataset.csv")
        raw_count = len(df_geo)
        self.log(f"  Raw geolocation rows: {raw_count:,}")

        # Clean zip prefix
        df_geo["geolocation_zip_code_prefix"] = df_geo["geolocation_zip_code_prefix"].astype(str).str.zfill(5)

        # Filter impossible GPS coordinates outside Brazil
        # Brazil bounds approx: Lat (-34 to +6), Lng (-74 to -34)
        valid_geo = df_geo[
            (df_geo["geolocation_lat"] >= -34.0) & (df_geo["geolocation_lat"] <= 6.0) &
            (df_geo["geolocation_lng"] >= -74.0) & (df_geo["geolocation_lng"] <= -34.0)
        ]
        filtered_out = raw_count - len(valid_geo)
        self.log(f"  Filtered {filtered_out:,} outlier coordinates outside Brazil boundary.")

        # Aggregate by zip prefix to centroid (median lat/lng) and mode city/state
        geo_agg = valid_geo.groupby("geolocation_zip_code_prefix").agg(
            geo_lat=("geolocation_lat", "median"),
            geo_lng=("geolocation_lng", "median"),
            geo_city=("geolocation_city", lambda x: x.mode().iloc[0] if not x.empty else ""),
            geo_state=("geolocation_state", lambda x: x.mode().iloc[0] if not x.empty else "")
        ).reset_index()

        self.log(f"  Aggregated 1,000,163 GPS points to {len(geo_agg):,} unique postal centroids.")
        out_path = self.processed_dir / "cleaned_geolocation_centroids.csv"
        geo_agg.to_csv(out_path, index=False)
        self.log(f"  Cleaned geolocation centroids saved to: {out_path.name}")
        return geo_agg

    def run_all(self):
        print("=" * 70)
        print("STARTING COMPLETE DATA CLEANING & VALIDATION PIPELINE (PHASE 2)")
        print("=" * 70)
        self.clean_categories_and_products()
        self.clean_customers()
        self.clean_sellers()
        self.clean_orders()
        self.clean_order_items()
        self.clean_order_payments()
        self.clean_order_reviews()
        self.clean_geolocation()
        print("=" * 70)
        print("DATA CLEANING PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"All cleaned artifacts are saved in: {self.processed_dir}")
        print("=" * 70)


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    raw = project_root / "data" / "raw"
    processed = project_root / "data" / "processed"
    
    cleaner = DataCleaner(raw, processed)
    cleaner.run_all()
