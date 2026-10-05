"""
Phase 5: RFM Analysis Module for Olist Brazilian E-Commerce System.
Calculates Recency, Frequency, Monetary metrics and statistical quintile scores (1 to 5).
Safely handles high frequency skewness (97% single-order distribution).
"""

from pathlib import Path
import pandas as pd
import numpy as np


class RFMAnalyzer:
    def __init__(self, processed_dir: Path):
        self.processed_dir = processed_dir
        self.orders_path = self.processed_dir / "orders_analytical.csv"

    def log(self, msg: str):
        print(f"[RFM ANALYSIS] {msg}")

    def compute_rfm_table(self) -> pd.DataFrame:
        self.log("Loading delivered orders for customer-level RFM calculation...")
        df_orders = pd.read_csv(self.orders_path, low_memory=False)
        
        # Filter for realized delivered orders
        delivered = df_orders[df_orders["is_delivered"] == 1].copy()
        delivered["order_purchase_timestamp"] = pd.to_datetime(delivered["order_purchase_timestamp"])
        
        self.log(f"  Delivered orders available: {len(delivered):,} across {delivered['customer_unique_id'].nunique():,} unique customers.")

        # Reference date is 1 day after the maximum purchase timestamp in the dataset
        max_purchase = delivered["order_purchase_timestamp"].max()
        snapshot_date = max_purchase + pd.Timedelta(days=1)
        self.log(f"  Latest purchase timestamp: {max_purchase} | Analysis snapshot date: {snapshot_date}")

        # Aggregate at customer_unique_id grain
        rfm = delivered.groupby("customer_unique_id").agg(
            recency=("order_purchase_timestamp", lambda dates: (snapshot_date - dates.max()).total_seconds() / 86400.0),
            frequency=("order_id", "count"),
            monetary=("total_payment_value", "sum"),
            avg_order_value=("total_payment_value", "mean"),
            first_purchase=("order_purchase_timestamp", "min"),
            last_purchase=("order_purchase_timestamp", "max"),
            total_items=("total_items_count", "sum"),
            avg_review_score=("review_score", "mean"),
            delayed_orders=("is_delayed", "sum")
        ).reset_index()

        rfm["customer_lifespan_days"] = (rfm["last_purchase"] - rfm["first_purchase"]).dt.total_seconds() / 86400.0

        self.log("Computing statistical score quintiles (1 to 5)...")
        # 1. Recency Scoring: lower days = higher recency score (5 = most recent, 1 = oldest)
        rfm["r_score"] = pd.qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1]).astype(int)

        # 2. Monetary Scoring: higher spend = higher score (5 = top spenders, 1 = lowest)
        rfm["m_score"] = pd.qcut(rfm["monetary"], q=5, labels=[1, 2, 3, 4, 5]).astype(int)

        # 3. Frequency Scoring:
        # Note: 97.0% of customers have frequency = 1. Standard qcut fails due to identical bin edges.
        # We use statistically grounded discrete binning for the marketplace:
        # 1 order -> F_Score 1
        # 2 orders -> F_Score 3
        # 3 orders -> F_Score 4
        # >= 4 orders -> F_Score 5
        def score_frequency(f):
            if f == 1:
                return 1
            elif f == 2:
                return 3
            elif f == 3:
                return 4
            else:
                return 5

        rfm["f_score"] = rfm["frequency"].apply(score_frequency)

        # Combined RFM Score string and FM average
        rfm["rfm_score_str"] = rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)
        rfm["fm_score_avg"] = ((rfm["f_score"] + rfm["m_score"]) / 2.0).round(1)

        # Merge primary customer location
        latest_location = delivered.sort_values("order_purchase_timestamp").drop_duplicates(
            subset=["customer_unique_id"], keep="last"
        )[["customer_unique_id", "customer_city", "customer_state", "customer_zip_code_prefix"]]

        rfm = rfm.merge(latest_location, on="customer_unique_id", how="left")

        self.log(f"  RFM table constructed for {len(rfm):,} unique customers.")
        self.log(f"  Recency range: {rfm['recency'].min():.1f} to {rfm['recency'].max():.1f} days (Median: {rfm['recency'].median():.1f})")
        self.log(f"  Monetary range: R$ {rfm['monetary'].min():.2f} to R$ {rfm['monetary'].max():.2f} (Median: R$ {rfm['monetary'].median():.2f})")
        self.log(f"  Frequency range: {rfm['frequency'].min()} to {rfm['frequency'].max()} orders")

        return rfm


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    processed = project_root / "data" / "processed"
    
    analyzer = RFMAnalyzer(processed)
    df_rfm = analyzer.compute_rfm_table()
    out = processed / "customer_rfm_scores.csv"
    df_rfm.to_csv(out, index=False)
    print(f"Saved RFM scores to: {out.name}")
