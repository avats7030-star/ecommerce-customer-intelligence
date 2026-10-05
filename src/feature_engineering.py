"""
Phase 7 & 8: Target Definition & Feature Engineering Module.
Creates an anti-leakage repeat-purchase prediction dataset based on initial customer orders.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self, processed_dir: Path, cutoff_date_str: str = "2018-04-01"):
        self.processed_dir = processed_dir
        self.cutoff_date = pd.to_datetime(cutoff_date_str)

    def log(self, msg: str):
        print(f"[FEATURE ENG] {msg}")

    def build_prediction_dataset(self) -> pd.DataFrame:
        self.log(f"Building prediction dataset with observation cutoff date: {self.cutoff_date.date()}...")
        orders_path = self.processed_dir / "orders_analytical.csv"
        df_orders = pd.read_csv(orders_path, low_memory=False)

        # Focus on realized delivered orders
        delivered = df_orders[df_orders["is_delivered"] == 1].copy()
        delivered["order_purchase_timestamp"] = pd.to_datetime(delivered["order_purchase_timestamp"])

        self.log(f"  Delivered orders loaded: {len(delivered):,} across {delivered['customer_unique_id'].nunique():,} unique customers.")

        # Identify customer order sequence sorted chronologically
        delivered = delivered.sort_values(["customer_unique_id", "order_purchase_timestamp"], ascending=True)
        delivered["order_sequence"] = delivered.groupby("customer_unique_id").cumcount() + 1
        delivered["total_customer_orders"] = delivered.groupby("customer_unique_id")["order_id"].transform("count")

        # First order per customer
        first_orders = delivered[delivered["order_sequence"] == 1].copy()

        # Cohort filtering: First purchase must be before cutoff date to ensure >=5 months observation window
        cohort = first_orders[first_orders["order_purchase_timestamp"] < self.cutoff_date].copy()
        n_cohort = len(cohort)
        self.log(f"  Qualified customer cohort (first order before {self.cutoff_date.date()}): {n_cohort:,} customers.")

        # Define Target Variable: Did customer make ANY subsequent order?
        # repeat_purchase = 1 if total_customer_orders > 1 else 0
        cohort["repeat_purchase"] = (cohort["total_customer_orders"] > 1).astype(int)
        repeats = cohort["repeat_purchase"].sum()
        repeat_rate = repeats / n_cohort * 100.0

        self.log(f"  Target 'repeat_purchase' distribution:")
        self.log(f"    Repeat buyers (1):     {repeats:,} ({repeat_rate:.2f}%)")
        self.log(f"    One-time buyers (0):   {n_cohort - repeats:,} ({100.0 - repeat_rate:.2f}%)")
        self.log("    Anti-Leakage Audit: All features computed strictly from order #1 attributes.")

        # Feature Extraction from Initial Order
        features = pd.DataFrame()
        features["customer_unique_id"] = cohort["customer_unique_id"]
        features["first_order_id"] = cohort["order_id"]
        features["first_order_date"] = cohort["order_purchase_timestamp"]

        # Numerical Features
        features["spend_order1"] = cohort["total_payment_value"].astype(float)
        features["items_count_order1"] = cohort["total_items_count"].astype(int)
        features["unique_products_order1"] = cohort["unique_products_count"].astype(int)
        features["freight_value_order1"] = cohort["total_freight_value"].astype(float)
        
        # Freight burden ratio (freight / total payment)
        features["freight_ratio_order1"] = (
            features["freight_value_order1"] / features["spend_order1"].replace(0, np.nan)
        ).fillna(0.0).clip(0, 1)

        features["installments_order1"] = cohort["payment_installments_max"].astype(int)
        features["payment_splits_order1"] = cohort["payment_splits_count"].astype(int)
        
        # Delivery Experience Features
        features["delivery_days_order1"] = cohort["delivery_duration_days"].fillna(cohort["delivery_duration_days"].median()).astype(float)
        features["delivery_delay_days_order1"] = cohort["delivery_delay_days"].fillna(0.0).astype(float)
        features["is_delayed_order1"] = cohort["is_delayed"].fillna(0).astype(int)

        # Review Feedback Features
        features["review_score_order1"] = cohort["review_score"].fillna(cohort["review_score"].median()).astype(float)
        features["has_comment_title_order1"] = cohort["has_comment_title"].fillna(0).astype(int)
        features["has_comment_message_order1"] = cohort["has_comment_message"].fillna(0).astype(int)

        # Calendar context
        features["purchase_hour_order1"] = cohort["order_hour"].astype(int)
        features["purchase_dow_order1"] = cohort["order_day_of_week"].astype(str)

        # Categorical features (high-level groupers)
        # Simplify payment type
        features["payment_type_order1"] = cohort["primary_payment_type"].astype(str)
        
        # Customer State (keep top 6 states + 'Other')
        top_states = ["SP", "RJ", "MG", "RS", "PR", "SC"]
        features["customer_state_order1"] = cohort["customer_state"].apply(
            lambda s: s if s in top_states else "Other_State"
        )

        # Primary Category (keep top 12 categories + 'other_category')
        top_cats = [
            "health_beauty", "watches_gifts", "bed_bath_table", "sports_leisure",
            "computers_accessories", "furniture_decor", "housewares", "cool_stuff",
            "auto", "toys", "garden_tools", "telephony"
        ]
        features["category_order1"] = cohort["primary_category"].apply(
            lambda c: c if c in top_cats else "other_category"
        )

        # Attach Target
        features["repeat_purchase"] = cohort["repeat_purchase"].values

        # Save to processed
        out_path = self.processed_dir / "customer_prediction_dataset.csv"
        features.to_csv(out_path, index=False)
        self.log(f"  Prediction dataset saved to: {out_path.name} ({len(features):,} rows, {len(features.columns)} columns)")

        return features


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    processed = project_root / "data" / "processed"

    fe = FeatureEngineer(processed)
    fe.build_prediction_dataset()
