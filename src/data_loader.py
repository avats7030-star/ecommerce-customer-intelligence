"""
Data Loader Module for Olist Brazilian E-Commerce Dataset.
Provides structured and reproducible loading functions for raw and processed data.
"""

from pathlib import Path
from typing import Dict, Optional
import pandas as pd


def get_data_dir(base_dir: Optional[Path] = None) -> Path:
    """Resolve data directory path robustly from various invocation locations."""
    if base_dir is not None:
        raw_path = base_dir / "data" / "raw"
        if raw_path.exists():
            return raw_path
    
    # Try relative to current script
    current_file_dir = Path(__file__).resolve().parent
    candidates = [
        current_file_dir.parent / "data" / "raw",
        current_file_dir.parent.parent / "data" / "raw",
        Path.cwd() / "data" / "raw",
        Path.cwd() / "ecommerce-customer-intelligence" / "data" / "raw",
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.csv")):
            return candidate

    raise FileNotFoundError("Could not locate data/raw directory with CSV files.")


class DataLoader:
    """Encapsulates loading and validation of Olist e-commerce datasets."""

    def __init__(self, raw_data_dir: Optional[Path] = None):
        self.raw_data_dir = get_data_dir(raw_data_dir)

    def load_customers(self) -> pd.DataFrame:
        """Load olist_customers_dataset.csv"""
        file_path = self.raw_data_dir / "olist_customers_dataset.csv"
        return pd.read_csv(file_path)

    def load_orders(self) -> pd.DataFrame:
        """Load olist_orders_dataset.csv with datetime parsing."""
        file_path = self.raw_data_dir / "olist_orders_dataset.csv"
        date_cols = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
        return pd.read_csv(file_path, parse_dates=date_cols)

    def load_order_items(self) -> pd.DataFrame:
        """Load olist_order_items_dataset.csv with shipping limit date."""
        file_path = self.raw_data_dir / "olist_order_items_dataset.csv"
        return pd.read_csv(file_path, parse_dates=["shipping_limit_date"])

    def load_order_payments(self) -> pd.DataFrame:
        """Load olist_order_payments_dataset.csv"""
        file_path = self.raw_data_dir / "olist_order_payments_dataset.csv"
        return pd.read_csv(file_path)

    def load_order_reviews(self) -> pd.DataFrame:
        """Load olist_order_reviews_dataset.csv with datetime columns."""
        file_path = self.raw_data_dir / "olist_order_reviews_dataset.csv"
        date_cols = ["review_creation_date", "review_answer_timestamp"]
        return pd.read_csv(file_path, parse_dates=date_cols)

    def load_products(self) -> pd.DataFrame:
        """Load olist_products_dataset.csv"""
        file_path = self.raw_data_dir / "olist_products_dataset.csv"
        return pd.read_csv(file_path)

    def load_sellers(self) -> pd.DataFrame:
        """Load olist_sellers_dataset.csv"""
        file_path = self.raw_data_dir / "olist_sellers_dataset.csv"
        return pd.read_csv(file_path)

    def load_geolocation(self) -> pd.DataFrame:
        """Load olist_geolocation_dataset.csv"""
        file_path = self.raw_data_dir / "olist_geolocation_dataset.csv"
        return pd.read_csv(file_path)

    def load_category_translation(self) -> pd.DataFrame:
        """Load product_category_name_translation.csv"""
        file_path = self.raw_data_dir / "product_category_name_translation.csv"
        return pd.read_csv(file_path)

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all 9 datasets into a dictionary."""
        return {
            "customers": self.load_customers(),
            "orders": self.load_orders(),
            "order_items": self.load_order_items(),
            "order_payments": self.load_order_payments(),
            "order_reviews": self.load_order_reviews(),
            "products": self.load_products(),
            "sellers": self.load_sellers(),
            "geolocation": self.load_geolocation(),
            "category_translation": self.load_category_translation(),
        }


if __name__ == "__main__":
    loader = DataLoader()
    print(f"Data directory: {loader.raw_data_dir}")
    dfs = loader.load_all()
    for name, df in dfs.items():
        print(f"Loaded {name}: {df.shape[0]:,} rows, {df.shape[1]} columns")
