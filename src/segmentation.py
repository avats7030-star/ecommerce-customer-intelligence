"""
Phase 6: Customer Segmentation & Intelligence Module.
Applies statistically robust segment mapping tailored to Olist's behavioral distribution.
Outputs customer-level intelligence features, segment aggregates, and visualization plots.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from rfm import RFMAnalyzer

# Plot aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


class CustomerSegmentation:
    def __init__(self, processed_dir: Path, reports_dir: Path, viz_dir: Path):
        self.processed_dir = processed_dir
        self.reports_dir = reports_dir
        self.viz_dir = viz_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str):
        print(f"[SEGMENTATION] {msg}")

    def assign_segment(self, row) -> str:
        r = row["r_score"]
        f = row["f_score"]
        m = row["m_score"]

        # 1. Can't Lose Them: Top historical repeat spenders who have become inactive
        if r == 1 and f >= 3 and m >= 4:
            return "Can't Lose Them"

        # 2. Champions: High recency, repeat purchases, high monetary spend
        if r >= 4 and f >= 3 and m >= 4:
            return "Champions"

        # 3. Loyal Customers: Steady repeat buyers with decent recency
        if r >= 3 and f >= 3:
            return "Loyal Customers"

        # 4. At Risk: Repeat or high spenders who haven't ordered in a long time
        if r <= 2 and (f >= 3 or m >= 4):
            return "At Risk"

        # 5. Potential Loyalists: High-spend recent buyers (high CLV potential)
        if r >= 4 and m >= 4 and f == 1:
            return "Potential Loyalists"

        # 6. New Customers: Most recent first-time buyers with average spend
        if r == 5 and f == 1 and m <= 3:
            return "New Customers"

        # 7. Promising: Moderately recent single-purchase buyers
        if r == 4 and f == 1 and m <= 3:
            return "Promising"

        # 8. Hibernating: Long inactive buyers with low-to-medium spend
        if r in (2, 3) and f == 1:
            return "Hibernating"

        # 9. Lost: Oldest, lowest spend single-purchase buyers
        return "Lost"

    def run_segmentation(self) -> pd.DataFrame:
        self.log("Step 1/4: Running RFM Analyzer on delivered orders...")
        analyzer = RFMAnalyzer(self.processed_dir)
        df_rfm = analyzer.compute_rfm_table()

        self.log("Step 2/4: Mapping customers to business segments...")
        df_rfm["customer_segment"] = df_rfm.apply(self.assign_segment, axis=1)

        # Generate customer intelligence flags
        self.log("Step 3/4: Creating customer intelligence analytical flags...")
        df_rfm["is_high_value"] = (df_rfm["m_score"] == 5).astype(int)
        df_rfm["is_frequent"] = (df_rfm["frequency"] >= 2).astype(int)
        df_rfm["is_new_customer"] = ((df_rfm["r_score"] == 5) & (df_rfm["frequency"] == 1)).astype(int)
        df_rfm["is_one_time"] = (df_rfm["frequency"] == 1).astype(int)
        # At risk rule: haven't purchased in >270 days and either repeat or spend >= R$ 200
        df_rfm["is_at_risk"] = ((df_rfm["recency"] > 270) & ((df_rfm["frequency"] >= 2) | (df_rfm["monetary"] >= 200))).astype(int)
        df_rfm["is_inactive"] = (df_rfm["recency"] > 365).astype(int)

        # Segment summary analytics
        self.log("Step 4/4: Computing segment benchmarks...")
        segment_summary = df_rfm.groupby("customer_segment").agg(
            customer_count=("customer_unique_id", "count"),
            total_revenue=("monetary", "sum"),
            avg_order_value=("avg_order_value", "mean"),
            avg_recency_days=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_spend_per_customer=("monetary", "mean"),
            avg_review_score=("avg_review_score", "mean")
        ).reset_index()

        total_customers = len(df_rfm)
        total_rev = df_rfm["monetary"].sum()
        segment_summary["customer_share_pct"] = (segment_summary["customer_count"] / total_customers * 100).round(2)
        segment_summary["revenue_share_pct"] = (segment_summary["total_revenue"] / total_rev * 100).round(2)

        # Sort logically by revenue contribution
        segment_summary = segment_summary.sort_values("total_revenue", ascending=False).reset_index(drop=True)

        self.segment_summary = segment_summary
        self.df_rfm = df_rfm

        # Save to processed
        out_rfm = self.processed_dir / "customer_rfm_segments.csv"
        df_rfm.to_csv(out_rfm, index=False)
        self.log(f"  Saved customer RFM segments mart: {out_rfm.name} ({len(df_rfm):,} customers)")

        out_summary = self.reports_dir / "rfm_segment_summary.csv"
        segment_summary.to_csv(out_summary, index=False)
        self.log(f"  Saved segment summary table: {out_summary.name}")

        self.generate_reports_and_charts()
        return df_rfm

    def generate_reports_and_charts(self):
        self.log("Generating RFM and Customer Segmentation Visualizations...")

        # 1. Segment Distribution Barplot (Customers vs Revenue)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.barplot(data=self.segment_summary, y="customer_segment", x="customer_count", ax=axes[0], palette="Blues_r", hue="customer_segment", legend=False)
        axes[0].set_title("Customer Count by Segment", fontweight="bold", fontsize=12)
        axes[0].set_xlabel("Number of Customers")
        axes[0].set_ylabel("Customer Segment")
        for i, val in enumerate(self.segment_summary["customer_count"]):
            pct = self.segment_summary.loc[i, "customer_share_pct"]
            axes[0].text(val + 400, i, f"{val:,} ({pct}%)", va="center", fontsize=9, fontweight="bold")

        sns.barplot(data=self.segment_summary, y="customer_segment", x="total_revenue", ax=axes[1], palette="Greens_r", hue="customer_segment", legend=False)
        axes[1].set_title("Total Revenue by Segment (BRL)", fontweight="bold", fontsize=12)
        axes[1].set_xlabel("Total Spend (R$)")
        axes[1].set_ylabel("")
        for i, val in enumerate(self.segment_summary["total_revenue"]):
            pct = self.segment_summary.loc[i, "revenue_share_pct"]
            axes[1].text(val + 30000, i, f"R$ {val/1e6:.2f}M ({pct}%)", va="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        plt.savefig(self.viz_dir / "09_rfm_segment_distribution.png", dpi=200)
        plt.close()

        # 2. Scatter plot: Recency vs Monetary with Segment Hue
        plt.figure(figsize=(12, 6))
        sample_df = self.df_rfm.sample(min(15000, len(self.df_rfm)), random_state=42)
        sns.scatterplot(
            data=sample_df,
            x="recency",
            y="monetary",
            hue="customer_segment",
            alpha=0.6,
            palette="tab10"
        )
        plt.title("Customer Recency vs Monetary Spend by Segment (Sample of 15k Customers)", fontweight="bold", fontsize=12)
        plt.xlabel("Recency (Days since last purchase)", fontweight="bold")
        plt.ylabel("Total Spend (R$ - Log Scale)", fontweight="bold")
        plt.yscale("log")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Segments")
        plt.tight_layout()
        plt.savefig(self.viz_dir / "10_rfm_scatter_recency_monetary.png", dpi=200)
        plt.close()

        # 3. Heatmap: Recency Score vs FM Score Average Monetary Value
        rfm_pivot = self.df_rfm.pivot_table(
            index="r_score",
            columns="fm_score_avg",
            values="monetary",
            aggfunc="mean"
        ).sort_index(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.heatmap(rfm_pivot, cmap="YlGnBu", annot=True, fmt=".0f", cbar_kws={"label": "Avg Spend (R$)"})
        plt.title("Average Customer Spending by Recency Score vs Frequency-Monetary Tier", fontweight="bold", fontsize=12)
        plt.xlabel("Frequency-Monetary (FM) Score Tier", fontweight="bold")
        plt.ylabel("Recency (R) Score (5=Recent, 1=Old)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.viz_dir / "11_rfm_heatmap_r_vs_fm.png", dpi=200)
        plt.close()

        # Export JSON and Markdown
        report_data = {
            "total_analyzed_customers": len(self.df_rfm),
            "high_value_customers_count": int(self.df_rfm["is_high_value"].sum()),
            "frequent_customers_count": int(self.df_rfm["is_frequent"].sum()),
            "at_risk_customers_count": int(self.df_rfm["is_at_risk"].sum()),
            "inactive_customers_count": int(self.df_rfm["is_inactive"].sum()),
            "segments": self.segment_summary.to_dict(orient="records")
        }
        with open(self.reports_dir / "rfm_segmentation_summary.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        print("\n" + "=" * 70)
        print("CUSTOMER SEGMENTATION SUMMARY TABLE (PHASE 6)")
        print("=" * 70)
        print(self.segment_summary[["customer_segment", "customer_count", "customer_share_pct", "total_revenue", "revenue_share_pct", "avg_spend_per_customer", "avg_recency_days"]].to_string(index=False))
        print("=" * 70)


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    processed = project_root / "data" / "processed"
    reports = project_root / "reports"
    viz = project_root / "visualizations"

    seg = CustomerSegmentation(processed, reports, viz)
    seg.run_segmentation()
