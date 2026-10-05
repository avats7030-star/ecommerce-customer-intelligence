"""
Phase 4: Exploratory Data Analysis (EDA) Module.
Analyzes Customers, Orders, Products, Revenue, Payments, Reviews, and Delivery Performance.
Generates comprehensive statistical summaries and high-resolution charts in visualizations/.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure professional plot aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10


class EDAPipeline:
    def __init__(self, processed_dir: Path, reports_dir: Path, viz_dir: Path):
        self.processed_dir = processed_dir
        self.reports_dir = reports_dir
        self.viz_dir = viz_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.eda_metrics = {}

    def log(self, section: str, msg: str):
        print(f"[EDA | {section}] {msg}")

    def load_data(self):
        print("=" * 70)
        print("LOADING INTEGRATED ANALYTICAL MARTS FOR EDA...")
        print("=" * 70)
        self.df_orders = pd.read_csv(self.processed_dir / "orders_analytical.csv", low_memory=False)
        self.df_items = pd.read_csv(self.processed_dir / "order_items_analytical.csv", low_memory=False)
        self.df_cust = pd.read_csv(self.processed_dir / "customers_analytical.csv", low_memory=False)
        print(f"Loaded: orders ({len(self.df_orders):,}), items ({len(self.df_items):,}), customers ({len(self.df_cust):,})")

    def analyze_customers(self):
        self.log("CUSTOMERS", "Analyzing customer spending, frequency, and repeat behavior...")
        total_unique = len(self.df_cust)
        repeats = int(self.df_cust["is_repeat_customer"].sum())
        onetime = total_unique - repeats
        repeat_pct = round(repeats / total_unique * 100, 2)

        spend_onetime = self.df_cust[self.df_cust["is_repeat_customer"] == 0]["total_spend"]
        spend_repeat = self.df_cust[self.df_cust["is_repeat_customer"] == 1]["total_spend"]

        cust_stats = {
            "total_delivered_unique_customers": total_unique,
            "one_time_customers": onetime,
            "one_time_pct": round(onetime / total_unique * 100, 2),
            "repeat_customers": repeats,
            "repeat_pct": repeat_pct,
            "max_orders_by_single_customer": int(self.df_cust["total_delivered_orders"].max()),
            "avg_spend_onetime": round(float(spend_onetime.mean()), 2),
            "median_spend_onetime": round(float(spend_onetime.median()), 2),
            "avg_spend_repeat": round(float(spend_repeat.mean()), 2),
            "median_spend_repeat": round(float(spend_repeat.median()), 2),
            "top_spend_customer": round(float(self.df_cust["total_spend"].max()), 2),
            "top_order_frequency_distribution": {
                str(k): int(v) for k, v in self.df_cust["total_delivered_orders"].value_counts().head(5).to_dict().items()
            }
        }
        self.eda_metrics["customers"] = cust_stats

        self.log("CUSTOMERS", f"Total Delivered Unique Customers: {total_unique:,}")
        self.log("CUSTOMERS", f"One-Time Buyers: {onetime:,} ({cust_stats['one_time_pct']}%) | Repeat Buyers: {repeats:,} ({repeat_pct}%)")
        self.log("CUSTOMERS", f"Mean Spend: One-Time = R${cust_stats['avg_spend_onetime']:.2f} vs Repeat = R${cust_stats['avg_spend_repeat']:.2f}")

        # Visualization: Customer Segmentation Spending & Frequency
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        cust_types = ["One-Time (1 Order)", "Repeat (>=2 Orders)"]
        spends = [cust_stats["avg_spend_onetime"], cust_stats["avg_spend_repeat"]]
        sns.barplot(x=cust_types, y=spends, hue=cust_types, legend=False, ax=axes[0], palette=["#3498db", "#2ecc71"])
        axes[0].set_title("Average Customer Spending: One-Time vs Repeat (BRL)", fontweight="bold")
        axes[0].set_ylabel("Average Spend (R$)")
        for i, val in enumerate(spends):
            axes[0].text(i, val + 5, f"R$ {val:,.2f}", ha="center", fontweight="bold")

        order_counts = self.df_cust["total_delivered_orders"].value_counts().head(4)
        order_labels = [f"{idx} Orders" for idx in order_counts.index]
        sns.barplot(x=order_labels, y=order_counts.values, hue=order_labels, legend=False, ax=axes[1], palette="Blues_r")
        axes[1].set_title("Customer Order Frequency Distribution", fontweight="bold")
        axes[1].set_ylabel("Number of Customers")
        for i, val in enumerate(order_counts.values):
            axes[1].text(i, val + 500, f"{val:,}", ha="center", fontweight="bold")

        plt.tight_layout()
        plt.savefig(self.viz_dir / "06_repeat_vs_onetime_customers.png", dpi=200)
        plt.close()

    def analyze_orders(self):
        self.log("ORDERS", "Analyzing order volume, statuses, and monthly trends...")
        status_dist = self.df_orders["order_status"].value_counts().to_dict()
        total_orders = len(self.df_orders)

        # Monthly order trends (filter complete months: Jan 2017 to Aug 2018)
        monthly_orders = self.df_orders.groupby("order_year_month")["order_id"].count()
        
        orders_stats = {
            "total_orders": total_orders,
            "status_distribution": {k: int(v) for k, v in status_dist.items()},
            "delivered_pct": round(status_dist.get("delivered", 0) / total_orders * 100, 2),
            "canceled_count": int(status_dist.get("canceled", 0)),
            "unavailable_count": int(status_dist.get("unavailable", 0)),
            "busiest_month": str(monthly_orders.idxmax()),
            "busiest_month_order_count": int(monthly_orders.max())
        }
        self.eda_metrics["orders"] = orders_stats
        self.log("ORDERS", f"Delivered Orders: {status_dist.get('delivered', 0):,} ({orders_stats['delivered_pct']}%)")
        self.log("ORDERS", f"Peak Order Month: {orders_stats['busiest_month']} with {orders_stats['busiest_month_order_count']:,} orders")

        # Visualization: Order Status
        plt.figure(figsize=(9, 5))
        ax = sns.barplot(x=list(status_dist.keys()), y=list(status_dist.values()), palette="viridis")
        plt.title("Order Status Distribution Across 99,441 Orders", fontweight="bold")
        plt.ylabel("Order Count")
        plt.yscale("log")
        plt.xticks(rotation=30, ha="right")
        for i, v in enumerate(status_dist.values()):
            ax.text(i, v * 1.15, f"{v:,}", ha="center", fontsize=9, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.viz_dir / "02_order_status_distribution.png", dpi=200)
        plt.close()

    def analyze_revenue(self):
        self.log("REVENUE", "Analyzing revenue, Monthly Recurring Trends, and AOV...")
        delivered = self.df_orders[self.df_orders["is_delivered"] == 1].copy()
        total_realized_rev = delivered["total_payment_value"].sum()
        total_item_rev = delivered["total_items_price"].sum()
        total_freight_rev = delivered["total_freight_value"].sum()
        aov = delivered["total_payment_value"].mean()
        aov_median = delivered["total_payment_value"].median()

        # Monthly revenue
        monthly_rev = delivered.groupby("order_year_month")["total_payment_value"].sum().sort_index()
        monthly_orders = delivered.groupby("order_year_month")["order_id"].count().sort_index()

        # Filter active mature period: 2017-01 to 2018-08
        valid_months = [m for m in monthly_rev.index if "2017-01" <= m <= "2018-08"]
        monthly_rev_filtered = monthly_rev.loc[valid_months]
        monthly_orders_filtered = monthly_orders.loc[valid_months]

        revenue_stats = {
            "total_realized_payment_revenue": round(float(total_realized_rev), 2),
            "total_product_revenue": round(float(total_item_rev), 2),
            "total_freight_revenue": round(float(total_freight_rev), 2),
            "freight_share_pct": round(float(total_freight_rev / (total_item_rev + total_freight_rev) * 100), 2),
            "average_order_value_aov": round(float(aov), 2),
            "median_order_value": round(float(aov_median), 2),
            "highest_revenue_month": str(monthly_rev_filtered.idxmax()),
            "highest_monthly_revenue": round(float(monthly_rev_filtered.max()), 2)
        }
        self.eda_metrics["revenue"] = revenue_stats
        self.log("REVENUE", f"Total Realized Revenue: R$ {revenue_stats['total_realized_payment_revenue']:,.2f}")
        self.log("REVENUE", f"Average Order Value (AOV): R$ {revenue_stats['average_order_value_aov']:.2f} (Median: R$ {revenue_stats['median_order_value']:.2f})")
        self.log("REVENUE", f"Freight Contribution: R$ {revenue_stats['total_freight_revenue']:,.2f} ({revenue_stats['freight_share_pct']}% of merchandise total)")

        # Visualization: Monthly Revenue & Order Volume
        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax2 = ax1.twinx()

        color1 = "#1f77b4"
        color2 = "#ff7f0e"

        ax1.plot(valid_months, monthly_rev_filtered.values / 1000.0, color=color1, marker="o", linewidth=2.5, label="Revenue (k R$)")
        ax2.plot(valid_months, monthly_orders_filtered.values, color=color2, marker="s", linewidth=2, linestyle="--", label="Order Volume")

        ax1.set_title("Monthly Revenue and Order Volume Trends (Jan 2017 – Aug 2018)", fontweight="bold", fontsize=14)
        ax1.set_xlabel("Year-Month", fontweight="bold")
        ax1.set_ylabel("Total Monthly Revenue (Thousands R$)", color=color1, fontweight="bold")
        ax2.set_ylabel("Monthly Order Count", color=color2, fontweight="bold")
        ax1.set_xticklabels(valid_months, rotation=45, ha="right")

        # Highlight Black Friday (Nov 2017)
        if "2017-11" in valid_months:
            bf_idx = valid_months.index("2017-11")
            bf_rev = monthly_rev_filtered.loc["2017-11"] / 1000.0
            ax1.annotate("Black Friday\n(Nov 2017 Peak)", xy=(bf_idx, bf_rev), xytext=(bf_idx - 1.5, bf_rev + 150),
                         arrowprops=dict(facecolor="red", shrink=0.08, width=1.5, headwidth=8),
                         fontweight="bold", color="darkred")

        fig.tight_layout()
        plt.savefig(self.viz_dir / "01_monthly_revenue_and_orders.png", dpi=200)
        plt.close()

    def analyze_products(self):
        self.log("PRODUCTS", "Analyzing top and bottom product categories by revenue & unit volume...")
        cat_agg = self.df_items.groupby("product_category_name_english").agg(
            total_items_sold=("order_item_id", "count"),
            total_revenue=("price", "sum"),
            avg_item_price=("price", "mean")
        ).reset_index()

        top_by_rev = cat_agg.sort_values("total_revenue", ascending=False).head(10)
        bottom_by_rev = cat_agg[cat_agg["product_category_name_english"] != "unknown_category"].sort_values("total_revenue", ascending=True).head(10)

        product_stats = {
            "total_product_categories": int(cat_agg["product_category_name_english"].nunique()),
            "top_10_categories_revenue": {
                r["product_category_name_english"]: round(float(r["total_revenue"]), 2) for _, r in top_by_rev.iterrows()
            },
            "bottom_10_categories_revenue": {
                r["product_category_name_english"]: round(float(r["total_revenue"]), 2) for _, r in bottom_by_rev.iterrows()
            }
        }
        self.eda_metrics["products"] = product_stats
        self.log("PRODUCTS", f"Top Category by Revenue: {top_by_rev.iloc[0]['product_category_name_english']} (R$ {top_by_rev.iloc[0]['total_revenue']:,.2f})")

        # Visualization: Top 10 and Bottom 10 categories
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.barplot(data=top_by_rev, y="product_category_name_english", x="total_revenue", ax=axes[0], palette="Greens_r")
        axes[0].set_title("Top 10 Product Categories by Revenue (BRL)", fontweight="bold")
        axes[0].set_xlabel("Total Sales Revenue (R$)")
        axes[0].set_ylabel("Category")

        sns.barplot(data=bottom_by_rev, y="product_category_name_english", x="total_revenue", ax=axes[1], palette="Reds_r")
        axes[1].set_title("Bottom 10 Product Categories by Revenue (BRL)", fontweight="bold")
        axes[1].set_xlabel("Total Sales Revenue (R$)")
        axes[1].set_ylabel("")

        plt.tight_layout()
        plt.savefig(self.viz_dir / "03_top10_bottom10_categories.png", dpi=200)
        plt.close()

    def analyze_payments(self):
        self.log("PAYMENTS", "Analyzing payment methods, installment distributions, and values...")
        pay_df = pd.read_csv(self.processed_dir / "cleaned_order_payments.csv")
        pay_type_counts = pay_df["payment_type"].value_counts()
        pay_type_val = pay_df.groupby("payment_type")["payment_value"].sum().sort_values(ascending=False)
        installments_dist = pay_df["payment_installments"].value_counts().head(10).to_dict()

        payment_stats = {
            "payment_type_counts": {k: int(v) for k, v in pay_type_counts.items()},
            "payment_type_values": {k: round(float(v), 2) for k, v in pay_type_val.items()},
            "credit_card_revenue_share_pct": round(float(pay_type_val.get("credit_card", 0) / pay_type_val.sum() * 100), 2),
            "max_installments_recorded": int(pay_df["payment_installments"].max()),
            "avg_installments": round(float(pay_df["payment_installments"].mean()), 2)
        }
        self.eda_metrics["payments"] = payment_stats
        self.log("PAYMENTS", f"Credit Card Share: {payment_stats['credit_card_revenue_share_pct']}% of total payments value.")

        # Visualization: Payments
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].pie(pay_type_val.values, labels=pay_type_val.index, autopct="%1.1f%%", startangle=140, 
                   colors=["#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#95a5a6"])
        axes[0].set_title("Revenue Share by Payment Method", fontweight="bold")

        inst_10 = pay_df["payment_installments"].value_counts().head(8).sort_index()
        sns.barplot(x=inst_10.index, y=inst_10.values, ax=axes[1], palette="Blues_r")
        axes[1].set_title("Top Installment Selections (Number of Transactions)", fontweight="bold")
        axes[1].set_xlabel("Number of Installments")
        axes[1].set_ylabel("Transactions Count")
        for i, val in enumerate(inst_10.values):
            axes[1].text(i, val + 1000, f"{val:,}", ha="center", fontsize=8, fontweight="bold")

        plt.tight_layout()
        plt.savefig(self.viz_dir / "04_payment_types_and_installments.png", dpi=200)
        plt.close()

    def analyze_reviews(self):
        self.log("REVIEWS", "Analyzing customer review scores and correlation with delivery delays...")
        rev_scores = self.df_orders["review_score"].dropna().value_counts().sort_index(ascending=False).to_dict()
        avg_score_on_time = self.df_orders[self.df_orders["is_delayed"] == 0]["review_score"].mean()
        avg_score_delayed = self.df_orders[self.df_orders["is_delayed"] == 1]["review_score"].mean()

        review_stats = {
            "score_distribution": {str(k): int(v) for k, v in rev_scores.items()},
            "overall_avg_review_score": round(float(self.df_orders["review_score"].mean()), 2),
            "avg_review_on_time_delivery": round(float(avg_score_on_time), 2),
            "avg_review_delayed_delivery": round(float(avg_score_delayed), 2),
            "review_penalty_from_delay": round(float(avg_score_on_time - avg_score_delayed), 2)
        }
        self.eda_metrics["reviews"] = review_stats
        self.log("REVIEWS", f"Average Review Score: {review_stats['overall_avg_review_score']} / 5.0")
        self.log("REVIEWS", f"On-time delivery score: {review_stats['avg_review_on_time_delivery']} vs Delayed delivery: {review_stats['avg_review_delayed_delivery']} (Severe drop of {review_stats['review_penalty_from_delay']} stars!)")

        # Visualization: Review Scores & Delay Impact
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.barplot(x=list(rev_scores.keys()), y=list(rev_scores.values()), ax=axes[0], palette="RdYlGn")
        axes[0].set_title("Customer Review Score Distribution (1 to 5 Stars)", fontweight="bold")
        axes[0].set_xlabel("Review Score")
        axes[0].set_ylabel("Order Count")
        for i, val in enumerate(rev_scores.values()):
            axes[0].text(i, val + 1000, f"{val:,}", ha="center", fontweight="bold")

        delay_comp = pd.DataFrame({
            "Delivery Status": ["On-Time Delivery", "Delayed Delivery"],
            "Average Rating": [avg_score_on_time, avg_score_delayed]
        })
        sns.barplot(data=delay_comp, x="Delivery Status", y="Average Rating", ax=axes[1], palette=["#2ecc71", "#e74c3c"])
        axes[1].set_title("Customer Rating Penalty Due to Delivery Delays", fontweight="bold")
        axes[1].set_ylim(0, 5)
        for i, val in enumerate([avg_score_on_time, avg_score_delayed]):
            axes[1].text(i, val + 0.15, f"{val:.2f} / 5.0", ha="center", fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.savefig(self.viz_dir / "05_review_scores_and_delivery_delays.png", dpi=200)
        plt.close()

    def analyze_delivery_and_geography(self):
        self.log("DELIVERY & GEO", "Analyzing delivery lead times, carrier latency, and state performance...")
        delivered = self.df_orders[self.df_orders["is_delivered"] == 1].copy()
        
        avg_delivery_time = delivered["delivery_duration_days"].mean()
        median_delivery_time = delivered["delivery_duration_days"].median()
        avg_est_time = delivered["estimated_duration_days"].mean()
        delay_rate = (delivered["is_delayed"].sum() / len(delivered)) * 100.0

        top_states = delivered["customer_state"].value_counts().head(10).to_dict()

        delivery_stats = {
            "avg_actual_delivery_days": round(float(avg_delivery_time), 2),
            "median_actual_delivery_days": round(float(median_delivery_time), 2),
            "avg_estimated_delivery_days": round(float(avg_est_time), 2),
            "delivery_delay_percentage": round(float(delay_rate), 2),
            "top_10_customer_states_by_orders": {k: int(v) for k, v in top_states.items()}
        }
        self.eda_metrics["delivery"] = delivery_stats
        self.log("DELIVERY & GEO", f"Average Actual Delivery Time: {avg_delivery_time:.1f} days (Median: {median_delivery_time:.1f} days)")
        self.log("DELIVERY & GEO", f"Average Quoted SLA: {avg_est_time:.1f} days. Delay rate: {delay_rate:.2f}%")

        # Visualization: Geographic Distribution of Orders
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(top_states.keys()), y=list(top_states.values()), palette="Blues_r")
        plt.title("Top 10 Brazilian States by Order Volume", fontweight="bold")
        plt.xlabel("State (UF)")
        plt.ylabel("Delivered Orders Count")
        for i, val in enumerate(top_states.values()):
            plt.text(i, val + 500, f"{val:,}", ha="center", fontsize=8, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.viz_dir / "07_geographic_distribution.png", dpi=200)
        plt.close()

        # Visualization: Delivery Lead Time Distribution
        plt.figure(figsize=(10, 5))
        sns.histplot(delivered["delivery_duration_days"].dropna().clip(upper=60), bins=60, kde=True, color="#2980b9")
        plt.axvline(avg_delivery_time, color="red", linestyle="--", linewidth=2, label=f"Mean: {avg_delivery_time:.1f} days")
        plt.axvline(median_delivery_time, color="green", linestyle=":", linewidth=2, label=f"Median: {median_delivery_time:.1f} days")
        plt.title("Customer Delivery Lead Time Distribution (Days)", fontweight="bold")
        plt.xlabel("Actual Delivery Time (Days)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.viz_dir / "08_delivery_lead_time_distribution.png", dpi=200)
        plt.close()

    def export_summary(self):
        self.log("EXPORT", "Saving EDA metrics and summary reports...")
        json_path = self.reports_dir / "eda_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.eda_metrics, f, indent=2)

        md_path = self.reports_dir / "eda_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Exploratory Data Analysis (EDA) Executive Summary\n\n")
            f.write(f"- **Total Delivered Orders**: {self.eda_metrics['orders']['total_orders']:,}\n")
            f.write(f"- **Total Unique Human Customers**: {self.eda_metrics['customers']['total_delivered_unique_customers']:,}\n")
            f.write(f"- **Repeat Customer Rate**: {self.eda_metrics['customers']['repeat_pct']}%\n")
            f.write(f"- **Total Realized Payment Revenue**: R$ {self.eda_metrics['revenue']['total_realized_payment_revenue']:,.2f}\n")
            f.write(f"- **Average Order Value (AOV)**: R$ {self.eda_metrics['revenue']['average_order_value_aov']:.2f}\n")
            f.write(f"- **Average Review Score**: {self.eda_metrics['reviews']['overall_avg_review_score']} ★\n")
            f.write(f"- **On-Time Rating**: {self.eda_metrics['reviews']['avg_review_on_time_delivery']} ★ vs **Delayed Rating**: {self.eda_metrics['reviews']['avg_review_delayed_delivery']} ★ (Drop of {self.eda_metrics['reviews']['review_penalty_from_delay']} ★)\n")
            f.write(f"- **Average Delivery Duration**: {self.eda_metrics['delivery']['avg_actual_delivery_days']:.1f} days\n")
            f.write(f"- **Delivery Delay Rate**: {self.eda_metrics['delivery']['delivery_delay_percentage']}%\n")

        print("=" * 70)
        print("EDA COMPLETE! Charts generated in visualizations/ and metrics saved in reports/")
        print("=" * 70)

    def run_all(self):
        self.load_data()
        self.analyze_customers()
        self.analyze_orders()
        self.analyze_revenue()
        self.analyze_products()
        self.analyze_payments()
        self.analyze_reviews()
        self.analyze_delivery_and_geography()
        self.export_summary()


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    processed = project_root / "data" / "processed"
    reports = project_root / "reports"
    viz = project_root / "visualizations"

    eda = EDAPipeline(processed, reports, viz)
    eda.run_all()
