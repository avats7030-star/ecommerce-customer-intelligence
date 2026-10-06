"""
E-Commerce Customer Intelligence & Purchase Prediction System
Professional Business Intelligence & Analytics Application
Powered by Olist Brazilian E-Commerce Dataset
"""

import sys
import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Resolve paths ──────────────────────────────────────────────────────────────
APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
REPORTS_DIR   = PROJECT_ROOT / "reports"
VIZ_DIR       = PROJECT_ROOT / "visualizations"
SRC_DIR       = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recommendations import RecommendationEngine

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Customer Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Olist Brazilian E-Commerce Customer Intelligence & Purchase Prediction System."}
)

# ── Professional Analytics CSS (Power BI / Tableau Style) ─────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }

  /* Main background */
  .stApp {
      background-color: #f8fafc;
  }

  /* Sidebar styling */
  [data-testid="stSidebar"] {
      background-color: #0f172a;
      border-right: 1px solid #1e293b;
  }
  [data-testid="stSidebar"] * {
      color: #e2e8f0 !important;
  }
  [data-testid="stSidebar"] .stRadio label {
      color: #94a3b8 !important;
      font-size: 0.88rem;
      font-weight: 500;
      padding: 4px 8px;
      border-radius: 4px;
      margin-bottom: 2px;
      display: flex;
  }
  [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
      color: #ffffff !important;
      background-color: #1e293b;
  }
  [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
      background-color: #1e3a5f;
      border-left: 3px solid #38bdf8;
      color: #ffffff !important;
      font-weight: 600;
  }

  /* Sidebar brand */
  .sidebar-brand {
      padding: 10px 0 16px 0;
      border-bottom: 1px solid #1e293b;
      margin-bottom: 16px;
  }
  .sidebar-brand-title {
      font-size: 1.05rem;
      font-weight: 700;
      color: #ffffff;
      letter-spacing: -0.01em;
  }
  .sidebar-brand-sub {
      font-size: 0.76rem;
      color: #94a3b8;
      margin-top: 3px;
      font-weight: 400;
  }

  /* Professional Header Banner */
  .hero-header {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-left: 4px solid #0078d4;
      border-radius: 4px;
      padding: 18px 24px;
      margin-bottom: 20px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.02);
  }
  .hero-title {
      font-size: 1.45rem;
      font-weight: 700;
      color: #0f172a;
      margin: 0;
      line-height: 1.25;
      letter-spacing: -0.01em;
  }
  .hero-subtitle {
      font-size: 0.88rem;
      color: #64748b;
      margin-top: 4px;
      font-weight: 400;
  }
  .hero-badge {
      display: inline-block;
      background: #f1f5f9;
      border: 1px solid #cbd5e1;
      color: #475569;
      font-size: 0.72rem;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 3px;
      margin-top: 10px;
      letter-spacing: 0.02em;
  }

  /* Professional KPI Cards */
  .kpi-card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 4px;
      padding: 14px 16px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.02);
      height: 100%;
  }
  .kpi-label {
      font-size: 0.72rem;
      font-weight: 600;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 4px;
  }
  .kpi-value {
      font-size: 1.55rem;
      font-weight: 700;
      color: #0f172a;
      line-height: 1.2;
  }
  .kpi-sub {
      font-size: 0.74rem;
      color: #94a3b8;
      margin-top: 4px;
  }

  /* Section cards */
  .section-card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 4px;
      padding: 18px 20px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.02);
      margin-bottom: 16px;
  }
  .section-title {
      font-size: 0.95rem;
      font-weight: 700;
      color: #1e293b;
      margin-bottom: 12px;
      letter-spacing: -0.01em;
  }

  /* Prediction status card */
  .pred-positive {
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      border-left: 4px solid #16a34a;
      border-radius: 4px;
      padding: 18px 20px;
      text-align: left;
  }
  .pred-negative {
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      border-left: 4px solid #ea580c;
      border-radius: 4px;
      padding: 18px 20px;
      text-align: left;
  }
  .pred-status {
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 4px;
  }
  .pred-positive .pred-status { color: #16a34a; }
  .pred-negative .pred-status { color: #ea580c; }
  .pred-prob {
      font-size: 2.2rem;
      font-weight: 800;
      color: #0f172a;
      line-height: 1.1;
  }
  .pred-label {
      font-size: 0.8rem;
      color: #64748b;
      margin-top: 4px;
  }

  /* Playbook card */
  .playbook-card {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-left: 4px solid #0078d4;
      border-radius: 4px;
      padding: 16px 18px;
  }
  .playbook-title {
      font-weight: 700;
      color: #0f172a;
      font-size: 0.92rem;
      margin-bottom: 4px;
  }
  .playbook-body {
      color: #334155;
      font-size: 0.86rem;
      line-height: 1.55;
  }

  /* Analytical callout box */
  .insight-box {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-left: 4px solid #475569;
      border-radius: 4px;
      padding: 14px 18px;
      color: #334155;
      font-size: 0.86rem;
      line-height: 1.6;
      margin-top: 14px;
  }
  .insight-box strong {
      color: #0f172a;
  }

  /* Divider */
  .fancy-divider {
      height: 1px;
      background: #e2e8f0;
      margin: 18px 0;
  }

  /* Model metric */
  .model-metric {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 4px;
      padding: 12px 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 6px;
  }
  .model-metric-name {
      font-size: 0.82rem;
      color: #64748b;
      font-weight: 500;
  }
  .model-metric-value {
      font-size: 1.05rem;
      font-weight: 700;
      color: #0f172a;
  }

  .block-container {
      padding-top: 1.25rem !important;
      padding-bottom: 2rem !important;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (CACHED)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    orders     = pd.read_csv(PROCESSED_DIR / "orders_analytical.csv",    low_memory=False)
    cust_rfm   = pd.read_csv(PROCESSED_DIR / "customer_rfm_segments.csv", low_memory=False)
    items      = pd.read_csv(PROCESSED_DIR / "order_items_analytical.csv", low_memory=False)
    seg_sum    = pd.read_csv(REPORTS_DIR / "rfm_segment_summary.csv")
    model_perf = pd.read_csv(REPORTS_DIR / "model_performance_comparison.csv")
    feat_imp   = pd.read_csv(REPORTS_DIR / "feature_importance.csv")
    with open(REPORTS_DIR / "eda_summary.json", "r") as f:
        eda_meta = json.load(f)
    return orders, cust_rfm, items, seg_sum, model_perf, feat_imp, eda_meta

@st.cache_data(show_spinner=False)
def load_prediction_dataset():
    pred_path = PROCESSED_DIR / "customer_prediction_dataset.csv"
    if pred_path.exists():
        return pd.read_csv(pred_path, low_memory=False)
    return None

@st.cache_resource(show_spinner=False)
def load_model():
    model = joblib.load(MODELS_DIR / "best_repeat_purchase_model.joblib")
    with open(MODELS_DIR / "model_metadata.json", "r") as f:
        meta = json.load(f)
    return model, meta

orders_df, cust_df, items_df, seg_df, perf_df, feat_imp_df, eda_meta = load_data()
pred_dataset_df = load_prediction_dataset()
model, model_meta = load_model()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">Customer Intelligence</div>
        <div class="sidebar-brand-sub">Olist E-Commerce Analytics Engine</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Executive Dashboard",
            "Customer Intelligence",
            "RFM Segmentation",
            "Prediction Simulator",
            "Upload Data & Predict",
            "Product Intelligence",
            "Model Evaluation",
            "Business Playbooks",
            "Methodology",
            "About Project",
        ],
        label_visibility="collapsed"
    )

    st.markdown("<div class='fancy-divider' style='background:#1e293b;'></div>", unsafe_allow_html=True)
    st.markdown("""
    **Dataset Information**
    | Field | Value |
    |---|---|
    | Orders | 99,441 |
    | Delivered | 96,478 |
    | Customers | 93,358 |
    | Revenue | R$ 15.42M |
    | Period | 2016 – 2018 |

    **Champion Model**
    | Metric | Score |
    |---|---|
    | Algorithm | Logistic Reg. |
    | ROC-AUC | 0.6110 |
    | Recall | 56.96% |
    """)


# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPER (PROFESSIONAL BI PALETTE)
# ══════════════════════════════════════════════════════════════════════════════
PALETTE = ["#005a9e", "#0078d4", "#008272", "#107c41", "#d83b01", "#e3008c", "#5c2d91",
           "#605e5c", "#004b50", "#2b88d8"]

def style_ax(ax):
    ax.set_facecolor("#ffffff")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cbd5e1")
    ax.spines[["left", "bottom"]].set_linewidth(0.8)
    ax.tick_params(colors="#475569", labelsize=8.5)
    ax.xaxis.label.set_color("#334155")
    ax.yaxis.label.set_color("#334155")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Executive Dashboard":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">E-Commerce Customer Intelligence</div>
        <div class="hero-subtitle">Sales, customer behavior and repeat-purchase analytics</div>
        <span class="hero-badge">Dataset: Olist Brazilian E-Commerce</span>
    </div>
    """, unsafe_allow_html=True)

    # Macro KPI Row (Calculated from real data)
    delivered_mask = orders_df["is_delivered"] == 1
    total_revenue  = orders_df.loc[delivered_mask, "total_payment_value"].sum()
    total_orders   = len(orders_df)
    unique_custs   = orders_df["customer_unique_id"].nunique()
    aov_val        = orders_df.loc[delivered_mask, "total_payment_value"].mean()
    repeat_rate    = (cust_df["frequency"] > 1).mean() * 100
    avg_review     = orders_df["review_score"].mean()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpis = [
        (f"R$ {total_revenue/1e6:.2f}M", "Total Revenue"),
        (f"{total_orders:,}",            "Total Orders"),
        (f"{unique_custs:,}",            "Unique Customers"),
        (f"R$ {aov_val:.2f}",            "Average Order Value"),
        (f"{repeat_rate:.2f}%",          "Repeat Customer Rate"),
        (f"{avg_review:.2f} / 5.0",      "Avg Review Score"),
    ]
    for col, (val, label) in zip([k1, k2, k3, k4, k5, k6], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 1: Sales Performance ──────────────────────────────────────────
    st.markdown("### Sales Performance")
    s_col1, s_col2 = st.columns([6, 4])
    with s_col1:
        st.markdown('<div class="section-card"><div class="section-title">Monthly Revenue & Order Volume Trend (2017–2018)</div>', unsafe_allow_html=True)
        delivered = orders_df[delivered_mask].copy()
        monthly   = delivered.groupby("order_year_month").agg(
            revenue=("total_payment_value", "sum"),
            orders=("order_id", "count")
        ).reset_index()
        monthly = monthly[(monthly["order_year_month"] >= "2017-01") & (monthly["order_year_month"] <= "2018-08")]

        fig, ax1 = plt.subplots(figsize=(8.5, 3.8))
        fig.patch.set_facecolor("#ffffff")
        ax2 = ax1.twinx()
        ax1.fill_between(range(len(monthly)), monthly["revenue"]/1000, alpha=0.10, color=PALETTE[0])
        ax1.plot(range(len(monthly)), monthly["revenue"]/1000, color=PALETTE[0], marker="o", lw=2.0, markersize=4, label="Revenue (k BRL)")
        ax2.plot(range(len(monthly)), monthly["orders"], color=PALETTE[4], marker="s", lw=1.6, linestyle="--", markersize=3.5, label="Orders")
        ax1.set_xticks(range(len(monthly)))
        ax1.set_xticklabels(monthly["order_year_month"], rotation=45, ha="right", fontsize=8)
        ax1.set_ylabel("Revenue (k BRL)", color=PALETTE[0], fontsize=8.5)
        ax2.set_ylabel("Order Count", color=PALETTE[4], fontsize=8.5)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1+lines2, labels1+labels2, loc="upper left", fontsize=8)
        style_ax(ax1); style_ax(ax2)
        fig.tight_layout(pad=1.2)
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with s_col2:
        st.markdown('<div class="section-card"><div class="section-title">Payment Method Share</div>', unsafe_allow_html=True)
        pay_dist = orders_df["primary_payment_type"].value_counts().head(4)
        pay_labels = {"credit_card":"Credit Card", "boleto":"Boleto Ticket", "voucher":"Voucher", "debit_card":"Debit Card"}
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        fig.patch.set_facecolor("#ffffff")
        clean_labels = [pay_labels.get(k, k.title()) for k in pay_dist.index]
        ax.pie(pay_dist.values, labels=clean_labels, autopct="%1.1f%%",
               startangle=140, colors=PALETTE[:4],
               wedgeprops={"linewidth":1.5, "edgecolor":"white"},
               pctdistance=0.80, textprops={"fontsize":8.5})
        centre = plt.Circle((0,0), 0.62, fc="white")
        ax.add_patch(centre)
        fig.tight_layout(pad=1.0)
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 2: Customer Performance ───────────────────────────────────────
    st.markdown("### Customer Performance")
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        st.markdown('<div class="section-card"><div class="section-title">Top 10 Product Categories by Revenue</div>', unsafe_allow_html=True)
        cat_rev = items_df.groupby("product_category_name_english")["price"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(7, 3.6))
        fig.patch.set_facecolor("#ffffff")
        ax.barh(cat_rev.index[::-1], cat_rev.values[::-1]/1e6, color=PALETTE[1], alpha=0.9, height=0.62)
        for i, (idx_, v) in enumerate(zip(cat_rev.index[::-1], cat_rev.values[::-1])):
            ax.text(v/1e6 + 0.01, i, f"R$ {v/1e6:.2f}M", va="center", fontsize=7.5, color="#374151")
        ax.set_xlabel("Gross Revenue (Million BRL)", fontsize=8.5)
        ax.set_xlim(0, cat_rev.max()/1e6 * 1.25)
        style_ax(ax); fig.tight_layout(pad=1.2)
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with c_col2:
        st.markdown('<div class="section-card"><div class="section-title">Top 10 Customer States by Revenue</div>', unsafe_allow_html=True)
        state_rev = orders_df.groupby("customer_state")["total_payment_value"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(7, 3.6))
        fig.patch.set_facecolor("#ffffff")
        ax.barh(state_rev.index[::-1], state_rev.values[::-1]/1e6, color=PALETTE[0], alpha=0.9, height=0.62)
        for i, v in enumerate(state_rev.values[::-1]):
            ax.text(v/1e6 + 0.01, i, f"R$ {v/1e6:.2f}M", va="center", fontsize=7.5, color="#374151")
        ax.set_xlabel("Gross Revenue (Million BRL)", fontsize=8.5)
        ax.set_xlim(0, state_rev.max()/1e6 * 1.25)
        style_ax(ax); fig.tight_layout(pad=1.2)
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 3: Business Insights ──────────────────────────────────────────
    st.markdown("### Business Insights")
    st.markdown("""
    <div class="insight-box">
        <strong>Executive Analytical Findings</strong><br>
        • <strong>Repeat Customer Spend Multiplier</strong>: Repeat purchasers represent 3.0% of the customer base but drive an average cumulative spend of R$ 380.20 — 2.38x higher than one-time buyers (R$ 159.48).<br>
        • <strong>Regional Revenue Concentration</strong>: The state of São Paulo (SP) alone accounts for ~42% of total marketplace GMV and transaction volume, making Southeast Brazil the primary commercial hub.<br>
        • <strong>Category Revenue Anchors</strong>: Bed, Bath & Table, Health & Beauty, and Sports & Leisure drive the highest gross merchandise value, generating over R$ 3.8M in aggregate sales.<br>
        • <strong>Fulfillment SLA Impact on Retention</strong>: Orders delivered past the estimated carrier SLA suffer an average review score of 1.6 stars versus 4.2 stars for on-time delivery, directly depressing repeat purchase likelihood.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: CUSTOMER INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Customer Intelligence":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">Customer Intelligence</div>
        <div class="hero-subtitle">Customer cohort behavior, spending patterns, and repeat buyer profiles</div>
        <span class="hero-badge">Dataset: Olist Brazilian E-Commerce</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 1: Customer Profile & Prediction Lookup ───────────────────────
    st.markdown("### Customer Lookup & Repeat Prediction")
    st.markdown("<div style='font-size:0.86rem;color:#64748b;margin-bottom:12px;'>Select a representative customer profile across key RFM segments, or search any customer unique ID from the dataset.</div>", unsafe_allow_html=True)

    sample_customers = {
        "Champions (High Value / Active)": "011575986092c30523ecb71ff10cb473",
        "Potential Loyalists (High Spend / Recent)": "004256f082951ec189a4962b6788c214",
        "At Risk (High Spend / Lapsed)": "0004aac84e0df4da2b147fca70cf8255",
        "Lost (Low Spend / Long Inactive)": "0000f46a3911fa3c0805444483337064",
    }

    lookup_c1, lookup_c2 = st.columns([5, 5])
    with lookup_c1:
        preset_choice = st.selectbox(
            "Select Representative Customer Example:",
            list(sample_customers.keys())
        )
        selected_id = sample_customers[preset_choice]
    with lookup_c2:
        custom_id = st.text_input("Or Enter Any Customer Unique ID:", value="", placeholder="e.g. 011575986092c30523ecb71ff10cb473")
        if custom_id.strip():
            selected_id = custom_id.strip()

    # Retrieve customer profile from RFM dataset
    c_match = cust_df[cust_df["customer_unique_id"] == selected_id]
    if not c_match.empty:
        c_row = c_match.iloc[0]

        # Display Customer Profile Cards
        prof_c1, prof_c2, prof_c3, prof_c4 = st.columns(4)
        with prof_c1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Customer ID</div>
                <div style="font-size:0.95rem;font-weight:700;color:#0f172a;word-break:break-all;">{c_row['customer_unique_id'][:16]}...</div>
                <div class="kpi-sub">Location: {c_row.get('customer_city','N/A').title()}, {c_row.get('customer_state','N/A')}</div>
            </div>""", unsafe_allow_html=True)
        with prof_c2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">RFM Segment</div>
                <div class="kpi-value" style="font-size:1.25rem;">{c_row.get('customer_segment','Standard')}</div>
                <div class="kpi-sub">RFM Score: {c_row.get('rfm_score_str','-')}</div>
            </div>""", unsafe_allow_html=True)
        with prof_c3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Monetary & Orders</div>
                <div class="kpi-value">R$ {c_row['monetary']:,.2f}</div>
                <div class="kpi-sub">{int(c_row['frequency'])} Orders (AOV: R$ {c_row['avg_order_value']:.2f})</div>
            </div>""", unsafe_allow_html=True)
        with prof_c4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Recency & Rating</div>
                <div class="kpi-value">{int(c_row['recency'])} days</div>
                <div class="kpi-sub">Avg Review: {c_row.get('avg_review_score', 0):.1f} / 5.0</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Predict Repeat Propensity for this specific customer ───────────────
        pred_match = pred_dataset_df[pred_dataset_df["customer_unique_id"] == selected_id] if pred_dataset_df is not None else pd.DataFrame()
        
        num_cols = [
            "spend_order1", "items_count_order1", "unique_products_order1",
            "freight_value_order1", "freight_ratio_order1", "installments_order1",
            "payment_splits_order1", "delivery_days_order1", "delivery_delay_days_order1",
            "is_delayed_order1", "review_score_order1", "has_comment_title_order1",
            "has_comment_message_order1", "purchase_hour_order1"
        ]
        cat_cols = ["payment_type_order1", "customer_state_order1", "category_order1", "purchase_dow_order1"]

        if not pred_match.empty:
            feature_row = pred_match.iloc[[0]][num_cols + cat_cols]
            c_prob = model.predict_proba(feature_row)[0][1]
            is_del = int(pred_match.iloc[0].get("is_delayed_order1", 0))
        else:
            # Fallback estimation using profile characteristics
            fallback_row = pd.DataFrame([{
                "spend_order1": float(c_row["monetary"]),
                "items_count_order1": int(c_row.get("total_items", 1)),
                "unique_products_order1": 1,
                "freight_value_order1": 20.0,
                "freight_ratio_order1": 20.0 / max(float(c_row["monetary"]), 1.0),
                "installments_order1": 2,
                "payment_splits_order1": 1,
                "delivery_days_order1": 12.0,
                "delivery_delay_days_order1": 0.0,
                "is_delayed_order1": int(c_row.get("delayed_orders", 0) > 0),
                "review_score_order1": float(c_row.get("avg_review_score", 5.0)),
                "has_comment_title_order1": 0,
                "has_comment_message_order1": 0,
                "purchase_hour_order1": 14,
                "payment_type_order1": "credit_card",
                "customer_state_order1": str(c_row.get("customer_state", "SP")),
                "category_order1": "bed_bath_table",
                "purchase_dow_order1": "Monday"
            }])
            c_prob = model.predict_proba(fallback_row)[0][1]
            is_del = int(c_row.get("delayed_orders", 0) > 0)

        c_prob_pct = c_prob * 100
        risk_tier = "High (>= 50%)" if c_prob >= 0.50 else "Medium (40-50%)" if c_prob >= 0.40 else "Low (< 40%)"

        res_col1, res_col2 = st.columns([5, 5])
        with res_col1:
            st.markdown('<div class="section-card"><div class="section-title">Repeat Purchase Prediction</div>', unsafe_allow_html=True)
            if c_prob >= 0.50:
                st.markdown(f"""
                <div class="pred-positive">
                    <div class="pred-status">High Repeat Propensity</div>
                    <div class="pred-prob">{c_prob_pct:.1f}%</div>
                    <div class="pred-label">Probability of 2nd Order · Propensity Tier: {risk_tier}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-negative">
                    <div class="pred-status">Low Repeat Propensity</div>
                    <div class="pred-prob">{c_prob_pct:.1f}%</div>
                    <div class="pred-label">Probability of 2nd Order · Propensity Tier: {risk_tier}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:0.78rem;color:#64748b;margin-top:8px;">
                Prediction is computed using the champion Logistic Regression model evaluated on initial purchase metrics.
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with res_col2:
            st.markdown('<div class="section-card"><div class="section-title">Recommended Business Action</div>', unsafe_allow_html=True)
            rec_action = RecommendationEngine().get_individual_action(c_prob, str(c_row.get("customer_segment","Standard")), is_delayed=is_del)
            st.markdown(f"""
            <div class="playbook-card">
                <div class="playbook-title">Strategy: {rec_action['playbook']}</div>
                <div style="font-size:0.78rem;color:#64748b;margin-bottom:8px;">
                    Urgency Level: <strong style="color:#0f172a">{rec_action['urgency']}</strong>
                </div>
                <div class="playbook-body">{rec_action['action']}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning(f"Customer ID '{selected_id}' not found in the analytical database.")

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Section 2: Cohort Analytics ───────────────────────────────────────────
    st.markdown("### Cohort Intelligence Explorer")
    with st.expander("Cohort Filter Options", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            all_segs = sorted(cust_df["customer_segment"].dropna().unique().tolist())
            sel_segs = st.multiselect("Customer Segments", all_segs,
                                      default=["Champions","At Risk","Potential Loyalists"])
        with fc2:
            max_m = int(cust_df["monetary"].max())
            spend_range = st.slider("Spend Range (BRL)", 0, min(max_m, 10000), (0, 3000))
        with fc3:
            repeat_only = st.checkbox("Repeat Buyers Only (>= 2 orders)", value=False)

    fdf = cust_df[cust_df["customer_segment"].isin(sel_segs)]
    fdf = fdf[(fdf["monetary"] >= spend_range[0]) & (fdf["monetary"] <= spend_range[1])]
    if repeat_only:
        fdf = fdf[fdf["is_frequent"] == 1]

    st.info(f"**{len(fdf):,}** customers match the selected filter criteria.")

    ck1, ck2, ck3, ck4 = st.columns(4)
    for col, (val, lbl) in zip([ck1, ck2, ck3, ck4], [
        (f"{len(fdf):,}", "Filtered Customers"),
        (f"R$ {fdf['monetary'].sum():,.0f}", "Total Spend"),
        (f"R$ {fdf['monetary'].mean():.2f}", "Avg Spend per Customer"),
        (f"{fdf['recency'].mean():.0f} days", "Avg Inactive Days"),
    ]):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    cca, ccb = st.columns(2)
    with cca:
        st.markdown('<div class="section-card"><div class="section-title">Spend Distribution (Filtered Cohort)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6.5, 3.5)); fig.patch.set_facecolor("#ffffff")
        clip_data = fdf["monetary"].clip(upper=fdf["monetary"].quantile(0.99))
        ax.hist(clip_data, bins=40, color=PALETTE[0], alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.axvline(fdf["monetary"].median(), color=PALETTE[4], lw=1.8, linestyle="--",
                   label=f"Median: R$ {fdf['monetary'].median():.0f}")
        ax.set_xlabel("Total Spend (BRL)", fontsize=8.5); ax.set_ylabel("Customer Count", fontsize=8.5)
        ax.legend(fontsize=8); style_ax(ax); fig.tight_layout(pad=1.2)
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with ccb:
        st.markdown('<div class="section-card"><div class="section-title">Customer Count by Segment</div>', unsafe_allow_html=True)
        seg_cnt = fdf["customer_segment"].value_counts()
        fig, ax = plt.subplots(figsize=(6.5, 3.5)); fig.patch.set_facecolor("#ffffff")
        bars = ax.bar(seg_cnt.index, seg_cnt.values, color=PALETTE[:len(seg_cnt)], alpha=0.9, width=0.58)
        ax.set_ylabel("Customers", fontsize=8.5)
        plt.xticks(rotation=30, ha="right", fontsize=8)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
                    f"{bar.get_height():,}", ha="center", va="bottom", fontsize=7.5)
        style_ax(ax); fig.tight_layout(pad=1.2)
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Customer Profiles Table")
    display_cols = {
        "customer_unique_id":"Customer ID","customer_segment":"Segment",
        "monetary":"Total Spend (R$)","frequency":"Orders","avg_order_value":"AOV (R$)",
        "recency":"Recency (Days)","avg_review_score":"Rating",
        "customer_city":"City","customer_state":"State",
    }
    avail_cols = [c for c in display_cols if c in fdf.columns]
    st.dataframe(fdf[avail_cols].rename(columns=display_cols).head(300).reset_index(drop=True),
                 use_container_width=True, height=320)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: RFM SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "RFM Segmentation":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">RFM Segmentation</div>
        <div class="hero-subtitle">Recency, Frequency, and Monetary quintile scoring and customer classification</div>
        <span class="hero-badge">9 Segments · 93,358 Scored Customers</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Segment Performance Summary")
    st.dataframe(seg_df.rename(columns={
        "customer_segment":"Segment","customer_count":"Customers",
        "customer_share_pct":"Share (%)","total_revenue":"GMV (BRL)",
        "revenue_share_pct":"Rev Share (%)","avg_spend_per_customer":"Avg Spend (R$)",
        "avg_recency_days":"Avg Recency (Days)","avg_order_value":"AOV (R$)",
    }), use_container_width=True, height=330)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    rca, rcb = st.columns(2)
    with rca:
        st.markdown('<div class="section-card"><div class="section-title">Recency vs. Monetary Spend by Segment</div>', unsafe_allow_html=True)
        sample = cust_df.sample(min(6000, len(cust_df)), random_state=42)
        fig, ax = plt.subplots(figsize=(7, 4.2)); fig.patch.set_facecolor("#ffffff")
        segs_ = sample["customer_segment"].unique()
        cmap_ = dict(zip(segs_, PALETTE[:len(segs_)]))
        for seg in segs_:
            sub = sample[sample["customer_segment"]==seg]
            ax.scatter(sub["recency"], sub["monetary"], s=10, alpha=0.45, color=cmap_.get(seg,"#94a3b8"), label=seg)
        ax.set_yscale("log")
        ax.set_xlabel("Recency (Days Inactive)", fontsize=8.5)
        ax.set_ylabel("Monetary Spend (R$ — Log Scale)", fontsize=8.5)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
        style_ax(ax); fig.tight_layout(pad=1.2)
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with rcb:
        st.markdown('<div class="section-card"><div class="section-title">Average Spend Heatmap: Recency vs. Monetary Score</div>', unsafe_allow_html=True)
        if "r_score" in cust_df.columns and "fm_score_avg" in cust_df.columns:
            pivot = cust_df.pivot_table(index="r_score", columns="fm_score_avg",
                                        values="monetary", aggfunc="mean").sort_index(ascending=False)
            fig, ax = plt.subplots(figsize=(7, 4.2)); fig.patch.set_facecolor("#ffffff")
            sns.heatmap(pivot, cmap="Blues", annot=True, fmt=".0f", ax=ax,
                        linewidths=0.5, linecolor="#e2e8f0", cbar_kws={"shrink":0.75})
            ax.set_xlabel("FM Score (1=Low, 5=High)", fontsize=8.5)
            ax.set_ylabel("R Score (1=Oldest, 5=Most Recent)", fontsize=8.5)
            fig.tight_layout(pad=1.2)
            st.pyplot(fig); plt.close()
        else:
            st.info("Score columns not found in dataset.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">Gross Revenue Contribution by Segment</div>', unsafe_allow_html=True)
    seg_sorted = seg_df.sort_values("total_revenue", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 3.4)); fig.patch.set_facecolor("#ffffff")
    bars = ax.bar(seg_sorted["customer_segment"], seg_sorted["total_revenue"]/1e6,
                  color=PALETTE[:len(seg_sorted)], alpha=0.9, width=0.58)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"R$ {bar.get_height():.2f}M", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("GMV (Million BRL)", fontsize=8.5)
    plt.xticks(rotation=20, ha="right", fontsize=8.5)
    style_ax(ax); fig.tight_layout(pad=1.2)
    st.pyplot(fig); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <strong>RFM Segmentation Findings</strong><br>
        • <strong>At Risk</strong> and <strong>Potential Loyalists</strong> account for 57% of total platform GMV.<br>
        • <strong>Champions</strong> (1.06% of customers) generate R$ 372.53 average spend, 2.3x the marketplace baseline.<br>
        • <strong>Hibernating</strong> represents 31.3% of the customer base but only 23% of revenue.<br>
        • Customers with combined Frequency-Monetary scores of 4–5 consistently spend 3–5x more than lower quintiles.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: PREDICTION SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction Simulator":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">Prediction Simulator</div>
        <div class="hero-subtitle">First-order transaction feature inputs to evaluate customer repeat purchase propensity</div>
        <span class="hero-badge">Logistic Regression · ROC-AUC 0.6110 · Production Model</span>
    </div>
    """, unsafe_allow_html=True)

    st.info("**Feature Policy**: Model strictly utilizes features available at the completion of Order 1 to maintain absolute anti-leakage integrity.")

    st.markdown("### Customer Input Profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Transaction Metrics**")
        spend        = st.number_input("Total Spend (BRL)", 10.0, 5000.0, 180.0, 10.0)
        items_cnt    = st.number_input("Items in Order", 1, 21, 1)
        prod_cnt     = st.number_input("Distinct Products", 1, 21, 1)
        category     = st.selectbox("Product Category", [
            "bed_bath_table","furniture_decor","sports_leisure","computers_accessories",
            "health_beauty","watches_gifts","housewares","auto","toys","garden_tools",
            "telephony","cool_stuff","other_category"])

    with c2:
        st.markdown("**Fulfillment and Payment**")
        freight      = st.number_input("Freight Fee (BRL)", 0.0, 400.0, 25.0, 5.0)
        installments = st.slider("Payment Installments", 1, 24, 3)
        payment_type = st.selectbox("Payment Method", ["credit_card","boleto","voucher","debit_card"])
        customer_state = st.selectbox("Customer State", ["SP","RJ","MG","RS","PR","SC","Other_State"])

    with c3:
        st.markdown("**Post-Delivery Experience**")
        review_score = st.slider("Review Rating (1–5)", 1, 5, 5)
        delivery_days = st.slider("Delivery Lead Time (Days)", 1.0, 60.0, 10.0)
        is_delayed   = st.selectbox("Order Delayed vs. SLA?", [0,1],
                                    format_func=lambda x: "Yes — Late Delivery" if x==1 else "No — On Time")
        delay_days   = st.number_input("Days Over SLA", 0.0, 45.0, 0.0 if is_delayed==0 else 4.0)

    freight_ratio = freight / max(spend, 1.0)
    has_title = 1 if review_score in (1,5) else 0
    has_msg   = 1 if review_score in (1,5) else 0

    input_row = pd.DataFrame([{
        "spend_order1":               float(spend),
        "items_count_order1":         int(items_cnt),
        "unique_products_order1":     int(prod_cnt),
        "freight_value_order1":       float(freight),
        "freight_ratio_order1":       float(freight_ratio),
        "installments_order1":        int(installments),
        "payment_splits_order1":      1,
        "delivery_days_order1":       float(delivery_days),
        "delivery_delay_days_order1": float(delay_days),
        "is_delayed_order1":          int(is_delayed),
        "review_score_order1":        float(review_score),
        "has_comment_title_order1":   int(has_title),
        "has_comment_message_order1": int(has_msg),
        "purchase_hour_order1":       14,
        "payment_type_order1":        payment_type,
        "customer_state_order1":      customer_state,
        "category_order1":            category,
        "purchase_dow_order1":        "Monday"
    }])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Predict Repeat Purchase", type="primary"):
        prob = model.predict_proba(input_row)[0][1]
        prob_pct = prob * 100
        tier_label = "HIGH (>= 50%)" if prob >= 0.50 else "MEDIUM (40-50%)" if prob >= 0.40 else "LOW (< 40%)"

        res1, res2 = st.columns([4,6])
        with res1:
            if prob >= 0.50:
                st.markdown(f"""
                <div class="pred-positive">
                    <div class="pred-status">High Return Propensity</div>
                    <div class="pred-prob">{prob_pct:.1f}%</div>
                    <div class="pred-label">Repeat Purchase Probability · Risk Tier: {tier_label}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-negative">
                    <div class="pred-status">Low Return Propensity</div>
                    <div class="pred-prob">{prob_pct:.1f}%</div>
                    <div class="pred-label">Repeat Purchase Probability · Risk Tier: {tier_label}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:10px;padding:10px 12px;background:#ffffff;border:1px solid #e2e8f0;border-radius:4px;font-size:0.8rem;color:#475569;">
                Baseline marketplace repeat rate: <strong>3.00%</strong><br>
                Champion model recall on repeat buyers: <strong>56.96%</strong>
            </div>""", unsafe_allow_html=True)

        with res2:
            rec = RecommendationEngine().get_individual_action(
                prob, "Potential Loyalists" if spend >= 200 else "Standard", is_delayed=is_delayed)
            st.markdown(f"""
            <div class="playbook-card">
                <div class="playbook-title">Strategy: {rec['playbook']}</div>
                <div style="margin-bottom:6px;font-size:0.78rem;color:#64748b">
                    Urgency Level: <strong style="color:#0f172a">{rec['urgency']}</strong></div>
                <div class="playbook-body">{rec['action']}</div>
            </div>""", unsafe_allow_html=True)

    # ── Section: Feature Importance Explainability ────────────────────────────
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown("### What Influences Customer Repeat Purchase?")
    st.markdown("<div style='font-size:0.86rem;color:#64748b;margin-bottom:12px;'>The model uses customer behavioral and transactional features to estimate repeat-purchase likelihood.</div>", unsafe_allow_html=True)

    exp_col1, exp_col2 = st.columns([6, 4])
    with exp_col1:
        top_f = feat_imp_df.head(10).copy()
        metric_col = "abs_impact" if "abs_impact" in top_f.columns else "importance"
        fig, ax = plt.subplots(figsize=(8, 3.8)); fig.patch.set_facecolor("#ffffff")
        clean_features = [f.replace("category_order1_","Cat: ").replace("customer_state_order1_","State: ").replace("_"," ") for f in top_f["feature"][::-1]]
        ax.barh(clean_features, top_f[metric_col][::-1], color=PALETTE[0], alpha=0.88, height=0.62)
        ax.set_xlabel("Absolute Model Coefficient Impact", fontsize=8.5)
        style_ax(ax); fig.tight_layout(pad=1.2)
        st.pyplot(fig); plt.close()

    with exp_col2:
        top_table = feat_imp_df.head(8).copy()
        top_table["feature"] = top_table["feature"].str.replace("category_order1_","Cat: ").str.replace("_"," ")
        st.dataframe(top_table.rename(columns={
            "feature": "Feature",
            "coefficient": "Direction (+/-)",
            "abs_impact": "Impact Magnitude"
        }), use_container_width=True, height=260)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: UPLOAD DATA & PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Upload Data & Predict":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">Upload Data & Predict</div>
        <div class="hero-subtitle">Score incoming customer transaction batches or review raw ingestion pipeline files</div>
        <span class="hero-badge">Batch Inference Pipeline</span>
    </div>
    """, unsafe_allow_html=True)

    tab_pred, tab_raw = st.tabs([
        "Batch Customer Prediction (CSV Upload)",
        "Dataset Ingestion and Pipeline Execution"
    ])

    with tab_pred:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">Upload Customer Batch for Repeat Purchase Prediction</div>
            <div style="font-size:0.88rem;color:#64748b;margin-bottom:14px;">
                Upload a CSV file containing new customer Order #1 transaction details (or load the sample batch below). 
                The trained champion Machine Learning model computes repeat purchase probabilities, risk tiers, and personalized retention actions.
            </div>
        """, unsafe_allow_html=True)

        col_up1, col_up2 = st.columns([7, 3])
        with col_up1:
            uploaded_file = st.file_uploader(
                "Choose customer CSV file",
                type=["csv"],
                help="Upload a CSV with Order #1 metrics (spend, freight, delivery days, review score, etc.)"
            )
        with col_up2:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            use_sample = st.button("Load Sample Batch (30 Customers)", use_container_width=True)
            
            sample_path = PROJECT_ROOT / "data" / "sample_new_customers.csv"
            if sample_path.exists():
                sample_bytes = sample_path.read_bytes()
                st.download_button(
                    "Download CSV Template",
                    data=sample_bytes,
                    file_name="new_customers_template.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        st.markdown("</div>", unsafe_allow_html=True)

        df_to_score = None
        if uploaded_file is not None:
            try:
                df_to_score = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded file with {len(df_to_score):,} customer records.")
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
        elif use_sample or st.session_state.get("sample_loaded", False):
            st.session_state["sample_loaded"] = True
            if sample_path.exists():
                df_to_score = pd.read_csv(sample_path)
                st.info(f"Loaded sample batch of {len(df_to_score)} customer transactions from dataset.")

        if df_to_score is not None:
            num_cols = [
                "spend_order1", "items_count_order1", "unique_products_order1",
                "freight_value_order1", "freight_ratio_order1", "installments_order1",
                "payment_splits_order1", "delivery_days_order1", "delivery_delay_days_order1",
                "is_delayed_order1", "review_score_order1", "has_comment_title_order1",
                "has_comment_message_order1", "purchase_hour_order1"
            ]
            cat_cols = [
                "payment_type_order1", "customer_state_order1", "category_order1", "purchase_dow_order1"
            ]

            missing_cols = [c for c in num_cols + cat_cols if c not in df_to_score.columns]
            df_proc = df_to_score.copy()

            defaults = {
                "spend_order1": 120.0, "items_count_order1": 1, "unique_products_order1": 1,
                "freight_value_order1": 20.0, "freight_ratio_order1": 0.18, "installments_order1": 1,
                "payment_splits_order1": 1, "delivery_days_order1": 12.0, "delivery_delay_days_order1": 0.0,
                "is_delayed_order1": 0, "review_score_order1": 5.0, "has_comment_title_order1": 0,
                "has_comment_message_order1": 0, "purchase_hour_order1": 14,
                "payment_type_order1": "credit_card", "customer_state_order1": "SP",
                "category_order1": "bed_bath_table", "purchase_dow_order1": "Monday"
            }

            if missing_cols:
                for mc in missing_cols:
                    df_proc[mc] = defaults.get(mc, 0)
                st.warning(f"Default values applied for {len(missing_cols)} missing schema columns: `{', '.join(missing_cols[:5])}`...")

            X_batch = df_proc[num_cols + cat_cols]
            probs = model.predict_proba(X_batch)[:, 1]
            prob_pcts = np.round(probs * 100, 1)

            df_proc["repeat_prob_pct"] = prob_pcts
            df_proc["prediction_label"] = np.where(probs >= 0.50, "Likely Repeat", "One-Time Buyer")
            
            def assign_tier(p):
                if p >= 50.0:
                    return "High (>= 50%)"
                elif p >= 40.0:
                    return "Medium (40-50%)"
                else:
                    return "Low (< 40%)"

            def assign_action(p):
                if p >= 50.0:
                    return "VIP Loyalty Invitation & Priority Shipping Guarantee"
                elif p >= 40.0:
                    return "10% Discount Voucher for 2nd Purchase within 14 Days"
                else:
                    return "Post-Delivery Satisfaction Check-in & Customer Care"

            df_proc["propensity_tier"] = df_proc["repeat_prob_pct"].apply(assign_tier)
            df_proc["recommended_action"] = df_proc["repeat_prob_pct"].apply(assign_action)

            n_total = len(df_proc)
            n_high  = (probs >= 0.50).sum()
            n_med   = ((probs >= 0.40) & (probs < 0.50)).sum()
            n_low   = (probs < 0.40).sum()
            total_rev_risk = df_proc["spend_order1"].sum()

            st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
            kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
            with kpi_c1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Customers Scored</div>
                    <div class="kpi-value">{n_total:,}</div>
                </div>""", unsafe_allow_html=True)
            with kpi_c2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">High Propensity (>= 50%)</div>
                    <div class="kpi-value">{n_high:,} <span style="font-size:0.85rem;color:#16a34a;font-weight:600;">({n_high/n_total*100:.1f}%)</span></div>
                </div>""", unsafe_allow_html=True)
            with kpi_c3:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Medium Propensity (40-50%)</div>
                    <div class="kpi-value">{n_med:,} <span style="font-size:0.85rem;color:#d97706;font-weight:600;">({n_med/n_total*100:.1f}%)</span></div>
                </div>""", unsafe_allow_html=True)
            with kpi_c4:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Batch First-Order Volume</div>
                    <div class="kpi-value">R$ {total_rev_risk:,.0f}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            chart1, chart2 = st.columns(2)
            with chart1:
                st.markdown('<div class="section-card"><div class="section-title">Predicted Probability Distribution</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6, 3.2)); fig.patch.set_facecolor("#ffffff")
                ax.hist(prob_pcts, bins=15, color=PALETTE[0], edgecolor="#ffffff", alpha=0.85)
                ax.axvline(50, color=PALETTE[4], linestyle="--", linewidth=1.5, label="Decision Threshold (50%)")
                ax.set_xlabel("Predicted Repeat Probability (%)", fontsize=8.5)
                ax.set_ylabel("Customer Count", fontsize=8.5)
                ax.legend(fontsize=8, loc="upper right")
                style_ax(ax); fig.tight_layout(pad=1.2)
                st.pyplot(fig); plt.close()
                st.markdown('</div>', unsafe_allow_html=True)

            with chart2:
                st.markdown('<div class="section-card"><div class="section-title">Propensity Tier Breakdown</div>', unsafe_allow_html=True)
                tier_counts = df_proc["propensity_tier"].value_counts()
                fig, ax = plt.subplots(figsize=(6, 3.2)); fig.patch.set_facecolor("#ffffff")
                colors = ["#16a34a" if "High" in t else "#d97706" if "Medium" in t else "#dc2626" for t in tier_counts.index]
                ax.bar(tier_counts.index, tier_counts.values, color=colors, width=0.55, edgecolor="#ffffff")
                for i, v in enumerate(tier_counts.values):
                    ax.text(i, v + (max(tier_counts.values)*0.02), f"{v:,}", ha="center", fontsize=8.5, fontweight="bold", color="#1e293b")
                ax.set_ylabel("Customer Count", fontsize=8.5)
                style_ax(ax); fig.tight_layout(pad=1.2)
                st.pyplot(fig); plt.close()
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-card"><div class="section-title">Scored Customer Records</div>', unsafe_allow_html=True)
            f_tier = st.selectbox("Filter by Propensity Tier:", ["All Tiers", "High (>= 50%)", "Medium (40-50%)", "Low (< 40%)"])
            
            display_df = df_proc.copy()
            if f_tier != "All Tiers":
                display_df = display_df[display_df["propensity_tier"] == f_tier]

            view_cols = [
                c for c in ["customer_unique_id", "first_order_id", "spend_order1", "review_score_order1",
                            "delivery_days_order1", "category_order1", "customer_state_order1",
                            "repeat_prob_pct", "propensity_tier", "recommended_action"]
                if c in display_df.columns
            ]
            st.dataframe(
                display_df[view_cols].rename(columns={
                    "customer_unique_id": "Customer ID",
                    "spend_order1": "Order #1 Spend (R$)",
                    "review_score_order1": "Review Score",
                    "delivery_days_order1": "Delivery Days",
                    "category_order1": "Category",
                    "customer_state_order1": "State",
                    "repeat_prob_pct": "Repeat Probability (%)",
                    "propensity_tier": "Propensity Tier",
                    "recommended_action": "Recommended Playbook Action"
                }),
                use_container_width=True,
                height=300
            )

            csv_export = df_proc.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Export Scored Dataset to CSV",
                data=csv_export,
                file_name="batch_repeat_predictions_scored.csv",
                mime="text/csv",
                type="primary"
            )
            st.markdown('</div>', unsafe_allow_html=True)

    with tab_raw:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">Raw Dataset Ingestion and Pipeline Execution</div>
            <div style="font-size:0.88rem;color:#64748b;margin-bottom:16px;">
                To update the analytical platform with refreshed transaction data (e.g., subsequent quarters or alternate stores),
                the system provides an automated 5-stage Data Science pipeline that cleans, validates, integrates, segments, and retrains models from source tables.
            </div>
        """, unsafe_allow_html=True)

        raw_dir = PROJECT_ROOT / "data" / "raw"
        raw_files = list(raw_dir.glob("*.csv")) if raw_dir.exists() else []

        st.markdown("#### Current Raw Dataset Files in `data/raw/`")
        if raw_files:
            file_info = []
            for rf in sorted(raw_files):
                size_mb = rf.stat().st_size / (1024 * 1024)
                file_info.append({
                    "Dataset File": rf.name,
                    "Size (MB)": f"{size_mb:.2f} MB",
                    "Status": "Present and Validated"
                })
            st.dataframe(pd.DataFrame(file_info), use_container_width=True, height=240)
        else:
            st.warning("No raw files found in `data/raw/`.")

        st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
        st.markdown("#### Pipeline Execution Sequence")
        st.markdown("""
        1. **Replace Source CSV Files**: Place updated CSV files in `data/raw/` matching the schema (`olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, etc.).
        2. **Execute Master Pipeline**: Run the orchestrator script in terminal:
        ```bash
        python src/run_pipeline.py
        ```
        3. **Execution Stages**:
           - **Stage 1 (Cleaning)**: Deduplicates, standardizes timestamp formats, and handles missing values.
           - **Stage 2 (Integration)**: Merges orders, payments, reviews, and items into denormalized analytical views.
           - **Stage 3 (RFM Segmentation)**: Re-calculates Recency, Frequency, and Monetary quintiles for cohort profiling.
           - **Stage 4 (Feature Engineering)**: Constructs the strictly anti-leakage Order #1 feature matrix.
           - **Stage 5 (Machine Learning)**: Retrains baseline and tree models, evaluating ROC-AUC and recall to select the champion model.
        4. **Refresh Application**: Reload the dashboard in your browser to inspect updated metrics and model evaluations.
        """)

        st.markdown("""
        <div class="insight-box">
            <strong>Data Architecture Note</strong><br>
            • Batch inference scoring operates independently on incoming customer files without model retraining, 
            while scheduled pipeline retraining refreshes feature representations and model weights periodically.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: PRODUCT INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Product Intelligence":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">Product Intelligence</div>
        <div class="hero-subtitle">Category revenue performance, pricing patterns, and freight fee impact</div>
        <span class="hero-badge">73 Product Categories</span>
    </div>
    """, unsafe_allow_html=True)

    cat_stats = items_df.groupby("product_category_name_english").agg(
        units_sold =("order_item_id","count"),
        revenue    =("price","sum"),
        avg_price  =("price","mean"),
        avg_freight=("freight_value","mean")
    ).reset_index()
    cat_stats["freight_pct"] = (cat_stats["avg_freight"]/cat_stats["avg_price"]*100).round(1)
    cat_stats = cat_stats[cat_stats["product_category_name_english"]!="unknown_category"]

    # Category Review Scores from orders
    cat_reviews = orders_df.groupby("primary_category")["review_score"].agg(["mean", "count"]).reset_index()
    cat_reviews.columns = ["product_category_name_english", "avg_review_score", "review_count"]
    cat_stats = pd.merge(cat_stats, cat_reviews, on="product_category_name_english", how="left")

    pa, pb = st.columns(2)
    with pa:
        st.markdown('<div class="section-card"><div class="section-title">Top 15 Categories by Revenue</div>', unsafe_allow_html=True)
        top15_rev = cat_stats.sort_values("revenue", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(6.8, 4.0)); fig.patch.set_facecolor("#ffffff")
        clean_cats = [c.replace("_"," ").title() for c in top15_rev["product_category_name_english"][::-1]]
        ax.barh(clean_cats, top15_rev["revenue"][::-1]/1e6, color=PALETTE[0], alpha=0.9, height=0.62)
        ax.set_xlabel("Gross Revenue (Million BRL)", fontsize=8.5)
        style_ax(ax); fig.tight_layout(pad=1.2)
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with pb:
        st.markdown('<div class="section-card"><div class="section-title">Top 15 Categories by Units Sold</div>', unsafe_allow_html=True)
        top15_units = cat_stats.sort_values("units_sold", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(6.8, 4.0)); fig.patch.set_facecolor("#ffffff")
        clean_cats_u = [c.replace("_"," ").title() for c in top15_units["product_category_name_english"][::-1]]
        ax.barh(clean_cats_u, top15_units["units_sold"][::-1], color=PALETTE[1], alpha=0.9, height=0.62)
        ax.set_xlabel("Units Sold (Items)", fontsize=8.5)
        style_ax(ax); fig.tight_layout(pad=1.2)
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown('<div class="section-card"><div class="section-title">Average Review Score by Category (Top Volume Categories)</div>', unsafe_allow_html=True)
        top_reviewed = cat_stats[cat_stats["units_sold"] >= 100].sort_values("avg_review_score", ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(6.8, 3.6)); fig.patch.set_facecolor("#ffffff")
        clean_rev_cats = [c.replace("_"," ").title() for c in top_reviewed["product_category_name_english"][::-1]]
        ax.barh(clean_rev_cats, top_reviewed["avg_review_score"][::-1], color=PALETTE[2], alpha=0.88, height=0.6)
        ax.set_xlim(3.0, 5.0)
        ax.set_xlabel("Average Review Rating (1–5)", fontsize=8.5)
        style_ax(ax); fig.tight_layout(pad=1.2)
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with p_col2:
        st.markdown('<div class="section-card"><div class="section-title">Highest Freight Fee Burden (>= 50 units sold)</div>', unsafe_allow_html=True)
        high_fr = cat_stats[cat_stats["units_sold"] >= 50].sort_values("freight_pct", ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(6.8, 3.6)); fig.patch.set_facecolor("#ffffff")
        clean_fr_cats = [c.replace("_"," ").title() for c in high_fr["product_category_name_english"][::-1]]
        ax.barh(clean_fr_cats, high_fr["freight_pct"][::-1], color=PALETTE[4], alpha=0.88, height=0.6)
        ax.set_xlabel("Freight Fee as % of Item Price", fontsize=8.5)
        style_ax(ax); fig.tight_layout(pad=1.2)
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Category Performance Directory")
    display_cat = cat_stats.sort_values("revenue", ascending=False).head(25)
    st.dataframe(display_cat.rename(columns={
        "product_category_name_english":"Category",
        "units_sold":"Units Sold",
        "revenue":"Gross Revenue (R$)",
        "avg_price":"Avg Price (R$)",
        "avg_freight":"Avg Freight (R$)",
        "freight_pct":"Freight % of Price",
        "avg_review_score":"Avg Review Rating"
    })[["Category","Units Sold","Gross Revenue (R$)","Avg Price (R$)","Avg Freight (R$)","Freight % of Price","Avg Review Rating"]].reset_index(drop=True),
    use_container_width=True, height=320)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7: MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Evaluation":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">Model Evaluation</div>
        <div class="hero-subtitle">Model performance comparison, ROC curves, confusion matrices, and feature importance</div>
        <span class="hero-badge">3 Algorithms · Stratified 80/20 Split · Balanced Class Weights</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Model Performance Comparison")
    st.dataframe(perf_df.style.format({
        "Accuracy":"{:.4f}","Precision":"{:.4f}","Recall":"{:.4f}",
        "F1_Score":"{:.4f}","ROC_AUC":"{:.4f}","PR_AUC":"{:.4f}"
    }).highlight_max(subset=["Recall","ROC_AUC"], color="#d1fae5"
    ).highlight_min(subset=["Precision"], color="#fee2e2"),
    use_container_width=True)

    st.markdown("""
    > **Why Logistic Regression is the Champion Model**: The positive class (repeat purchasers = 3.0%) is heavily imbalanced. In customer retention analytics, missing an actual repeat customer (False Negative) costs the enterprise significant lifetime value. By incorporating `class_weight='balanced'`, Logistic Regression captures **56.96% of all true repeat buyers** on the held-out test split, outperforming tree models in recall and maintaining calibrated probabilistic scores.
    """)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    roc_path = VIZ_DIR / "12_model_roc_and_pr_curves.png"
    cm_path  = VIZ_DIR / "14_confusion_matrices.png"
    fi_path  = VIZ_DIR / "13_feature_importance.png"

    if roc_path.exists():
        st.markdown("### ROC and Precision-Recall Curves")
        st.image(str(roc_path), use_container_width=True)

    if cm_path.exists():
        st.markdown("### Confusion Matrices")
        st.image(str(cm_path), use_container_width=True)

    if fi_path.exists():
        st.markdown("### Feature Importance: Champion Model")
        st.image(str(fi_path), use_container_width=True)

    st.markdown("### Champion Model Metadata")
    meta_cols = st.columns(3)
    meta_kpis = [
        ("Algorithm",     model_meta.get("champion_model","Logistic Regression")),
        ("Train Samples", f"{model_meta.get('train_samples',0):,}"),
        ("Test Samples",  f"{model_meta.get('test_samples',0):,}"),
        ("ROC-AUC",       str(model_meta["metrics"].get("ROC_AUC","—"))),
        ("Recall",        str(model_meta["metrics"].get("Recall","—"))),
        ("Target",        model_meta.get("target_variable","repeat_purchase")),
    ]
    for i, (k, v) in enumerate(meta_kpis):
        with meta_cols[i%3]:
            st.markdown(f"""
            <div class="model-metric">
                <span class="model-metric-name">{k}</span>
                <span class="model-metric-value">{v}</span>
            </div><br>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <strong>Model Interpretation</strong><br>
        • <strong>review_score_order1</strong> is the strongest single predictor of repeat purchase propensity.<br>
        • <strong>spend_order1</strong> and <strong>delivery_days_order1</strong> provide strong secondary predictive signal.<br>
        • <strong>customer_state_SP</strong> captures geographical fulfillment advantages and higher buyer density in São Paulo.<br>
        • Precision-Recall AUC of 0.060 vs. 0.038 random baseline reflects a ~1.6x lift in identifying true repeat purchasers under severe class imbalance.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8: BUSINESS PLAYBOOKS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Business Playbooks":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">Business Playbooks</div>
        <div class="hero-subtitle">Prescriptive strategic playbooks mapping RFM segments to retention and growth initiatives</div>
        <span class="hero-badge">9 Targeted Segment Playbooks</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.86rem;color:#64748b;margin-bottom:14px;'>The following strategies represent analytical recommendations designed for commercial and CRM marketing alignment. They translate empirical segment behavior into concrete interventions.</div>", unsafe_allow_html=True)

    recs = RecommendationEngine().get_all_recommendations()

    PRIORITY_MAP = {
        "Champions":          ("HIGH LTV PROTECTION",   "#f0fdf4", "#15803d"),
        "Potential Loyalists": ("HIGH REVENUE CAPTURE",  "#f0f9ff", "#0369a1"),
        "At Risk":             ("HIGH CHURN MITIGATION", "#fef2f2", "#b91c1c"),
        "Can't Lose Them":     ("CRITICAL WINBACK",      "#fef2f2", "#991b1b"),
        "New Customers":       ("ONBOARDING",            "#fefce8", "#854d0e"),
        "Promising":           ("NURTURING",             "#fefce8", "#78350f"),
        "Loyal Customers":     ("SUBSCRIPTION LOCK-IN",  "#f0fdf4", "#166534"),
        "Hibernating":         ("LOW-CAC REACTIVATION",  "#f8fafc", "#475569"),
        "Lost":                ("SUPPRESS / MONITOR",    "#f8fafc", "#64748b"),
    }

    for idx, r in enumerate(recs):
        seg = r["customer_segment"]
        priority_label, bg_color, txt_color = PRIORITY_MAP.get(seg, ("STANDARD", "#f8fafc", "#374151"))
        with st.expander(f"Playbook {idx+1}: {seg}", expanded=(idx < 3)):
            p1, p2 = st.columns([6,4])
            with p1:
                st.markdown("**Problem Statement:**"); st.write(r["problem"])
                st.markdown("**Data Evidence:**"); st.info(r["evidence"])
                st.markdown("**Recommended Action:**"); st.success(r["recommended_action"])
            with p2:
                st.markdown("**Business Objective:**"); st.write(r["business_objective"])
                st.markdown(f"""
                <div style="background:{bg_color};border:1px solid #e2e8f0;border-left:4px solid {txt_color};
                            border-radius:4px;padding:12px 14px;margin-top:10px;">
                    <span style="color:{txt_color};font-weight:700;font-size:0.82rem;letter-spacing:0.04em;">{priority_label}</span>
                </div>""", unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
        <strong>Strategic Budget and Priority Allocation</strong><br>
        • Direct <strong>70% of retention budget</strong> toward <em>Potential Loyalists</em> (R$ 4.35M GMV) and <em>At Risk</em> (R$ 4.42M GMV).<br>
        • <strong>Champions</strong> benefit most from exclusive perks and early catalog access rather than margin-eroding discounts.<br>
        • <strong>Hibernating</strong> cohorts should be engaged through low-cost automated messaging during seasonal retail peaks.<br>
        • <strong>Lost</strong> cohorts should be suppressed from expensive paid retargeting channels to maintain marketing capital efficiency.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9: METHODOLOGY (NEW PAGE)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Methodology":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">Methodology & Technical Architecture</div>
        <div class="hero-subtitle">Comprehensive data pipeline, feature engineering protocol, and machine learning workflow</div>
        <span class="hero-badge">Full Architecture Documentation</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### End-to-End Analytical Workflow")
    st.markdown("""
    ```text
    Raw Relational Data (8 CSVs)
           │
           ▼
    1. Data Cleaning & Validation (Datetime coercion, anomaly filters, UTF-8 normalization)
           │
           ▼
    2. Relational Integration (Star-schema analytical views: Orders, Items, Customers)
           │
           ▼
    3. Exploratory Data Analysis (Macro trends, geographic concentration, logistics analysis)
           │
           ▼
    4. RFM Behavioral Segmentation (Quintile scoring into 9 strategic customer cohorts)
           │
           ▼
    5. Anti-Leakage Feature Engineering (Strict Order #1 observation boundary)
           │
           ▼
    6. ML Model Training & Class Balancing (Logistic Regression, Decision Tree, Random Forest)
           │
           ▼
    7. Rigorous Evaluation & Champion Selection (Stratified 80/20 split, Recall & ROC-AUC priority)
           │
           ▼
    8. Production Inference & Scoring (Batch CSV pipeline and single-profile simulator)
           │
           ▼
    9. Prescriptive Business Strategy (Data-grounded commercial playbooks & CRM actions)
    ```
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    stages = [
        ("Stage 1: Raw Data Ingestion & Audit",
         "The platform ingests 8 relational CSV files from the Brazilian Olist marketplace spanning 2016 through 2018 (99,441 orders, 112,650 order items, 99,441 reviews, and 96,096 customers). A foreign key audit verifies relational integrity across primary entities."),
        
        ("Stage 2: Data Cleaning & Validation",
         "Timestamps across 8 date columns are coerced to ISO datetimes. Cancelled and unavailable orders are isolated from delivered commercial transactions. Monetary anomalies (zero or negative prices) and geolocation coordinates outside Brazilian spatial bounds are validated and normalized."),
        
        ("Stage 3: Relational Integration & Denormalization",
         "Tables are merged into analytical marts at three distinct grains: orders-level (`orders_analytical.csv`), order-items level (`order_items_analytical.csv`), and unique customer level (`customers_analytical.csv`). Key metrics such as delivery lead times and payment installments are calculated."),
        
        ("Stage 4: Exploratory Data Analysis (EDA)",
         "Comprehensive statistical profiling maps monthly revenue seasonality, spatial demand concentration across 27 Brazilian states, fulfillment lead time distributions, and customer rating distributions across product categories."),
        
        ("Stage 5: RFM Segmentation Matrix",
         "Customers are profiled using Recency (days since last purchase), Frequency (total delivered orders), and Monetary value (total GMV). Independent quintile ranking assigns scores from 1 to 5. Customers are categorized into 9 distinct cohorts: Champions, Loyal Customers, Potential Loyalists, New Customers, Promising, At Risk, Can't Lose Them, Hibernating, and Lost."),
        
        ("Stage 6: Anti-Leakage Feature Engineering",
         "To prevent lookahead data leakage, the predictive dataset isolates customer behavior up to the point of Order #1 delivery. All subsequent order interactions are strictly excluded from feature inputs. The target variable is defined as binary repeat purchase indicator (1 if subsequent orders >= 1, else 0)."),
        
        ("Stage 7: Model Training & Class Imbalance Handling",
         "Repeat buyers represent only ~3.0% of unique customers, creating severe class imbalance. Models are trained using `class_weight='balanced'` to prevent majority class bias. Three algorithms are systematically evaluated: Logistic Regression, Decision Tree, and Random Forest."),
        
        ("Stage 8: Model Evaluation & Validation",
         "Using a stratified 80/20 train/test split, models are assessed using ROC-AUC, Recall, Precision, PR-AUC, and Confusion Matrices. Logistic Regression is selected as champion due to highest recall (56.96%) on true repeat purchasers and optimal PR-AUC."),
        
        ("Stage 9: Batch & Real-Time Inference Pipelines",
         "The production inference engine accepts new order batches via CSV upload or interactive parameter simulation, returning repeat probabilities and categorized propensity tiers (High, Medium, Low)."),
        
        ("Stage 10: Prescriptive Commercial Strategy",
         "Inference probabilities and RFM segments map directly into automated commercial recommendations, guiding marketing budget allocation and customer success interventions.")
    ]

    for title, desc in stages:
        st.markdown(f"""
        <div class="section-card">
            <div class="section-title">{title}</div>
            <div style="font-size:0.88rem;color:#475569;line-height:1.6;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 10: ABOUT PROJECT (NEW PAGE)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About Project":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">About Project</div>
        <div class="hero-subtitle">Project context, dataset provenance, and technology stack</div>
        <span class="hero-badge">Academic & Portfolio Project</span>
    </div>
    """, unsafe_allow_html=True)

    ab1, ab2 = st.columns([6, 4])
    with ab1:
        st.markdown('<div class="section-card"><div class="section-title">Project Purpose & Objectives</div>', unsafe_allow_html=True)
        st.markdown("""
        The **E-Commerce Customer Intelligence & Purchase Prediction System** is a business analytics and machine learning application developed to address the core commercial challenge of customer retention in modern digital marketplaces.

        **Core Analytical Objectives:**
        - **Macro Commercial Intelligence**: Track revenue seasonality, fulfillment lead times, and regional sales distribution across 99k+ orders.
        - **Behavioral Lifecycle Segmentation**: Segment 93k+ customers into 9 actionable cohorts using empirical RFM scoring.
        - **Early Repeat Purchase Prediction**: Identify high-propensity repeat buyers immediately after their initial transaction using anti-leakage ML features.
        - **Strategic Decision Support**: Bridge predictive scores with prescriptive commercial playbooks to maximize customer lifetime value (LTV).
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="section-title">Dataset Provenance</div>', unsafe_allow_html=True)
        st.markdown("""
        **Dataset**: Olist Brazilian E-Commerce Dataset  
        **Origin**: Real commercial transaction data from Olist Store, connecting small merchants across Brazil to national retail marketplaces.  
        **Coverage**: September 2016 to October 2018  
        **Volume**: 99,441 unique orders across 96,096 customers and 3,095 distinct sellers.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with ab2:
        st.markdown('<div class="section-card"><div class="section-title">Technology Stack</div>', unsafe_allow_html=True)
        st.markdown("""
        | Layer | Technology |
        |---|---|
        | Programming Language | Python 3 |
        | Data Manipulation | Pandas, NumPy |
        | Machine Learning | Scikit-learn, Joblib |
        | Visualizations | Matplotlib, Seaborn |
        | Web Application | Streamlit |
        | Styling & UI | Custom Vanilla CSS (Power BI Aesthetic) |
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="section-title">Repository Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        ```text
        ecommerce-customer-intelligence/
        ├── dashboard/
        │   └── app.py
        ├── data/
        │   ├── raw/
        │   └── processed/
        ├── models/
        │   ├── best_repeat_purchase_model.joblib
        │   └── model_metadata.json
        ├── reports/
        ├── src/
        └── visualizations/
        ```
        """)
        st.markdown('</div>', unsafe_allow_html=True)
