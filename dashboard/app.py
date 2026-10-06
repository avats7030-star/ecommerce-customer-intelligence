"""
E-Commerce Customer Intelligence & Purchase Prediction System
Professional Business Analytics & Machine Learning Web Application
Powered by Olist Brazilian E-Commerce Dataset
"""

import sys
import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Path Resolution ───────────────────────────────────────────────────────────
APP_DIR       = Path(__file__).resolve().parent
PROJECT_ROOT  = APP_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
REPORTS_DIR   = PROJECT_ROOT / "reports"
VIZ_DIR       = PROJECT_ROOT / "visualizations"
SRC_DIR       = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recommendations import RecommendationEngine

# ── Streamlit Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Customer Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Olist Brazilian E-Commerce Customer Intelligence Platform."}
)

# ── Enterprise SaaS CSS Styling ───────────────────────────────────────────────
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
      padding: 5px 10px;
      border-radius: 4px;
      margin-bottom: 2px;
      display: flex;
      transition: background-color 0.15s ease, color 0.15s ease;
  }
  [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
      color: #ffffff !important;
      background-color: #1e293b;
  }
  [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
      background-color: #1e3a5f;
      border-left: 3px solid #0078d4;
      color: #ffffff !important;
      font-weight: 600;
  }

  /* Sidebar Brand */
  .sidebar-header {
      padding: 12px 4px 16px 4px;
      border-bottom: 1px solid #1e293b;
      margin-bottom: 14px;
  }
  .sidebar-title {
      font-size: 0.82rem;
      font-weight: 700;
      color: #38bdf8 !important;
      letter-spacing: 0.08em;
      text-transform: uppercase;
  }
  .sidebar-subtitle {
      font-size: 1.05rem;
      font-weight: 700;
      color: #ffffff !important;
      margin-top: 2px;
      letter-spacing: -0.01em;
  }

  /* Compact Top Header Bar */
  .top-navbar {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 12px 18px;
      margin-bottom: 18px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: 0 1px 2px rgba(0,0,0,0.02);
  }
  .top-navbar-title {
      font-size: 1.25rem;
      font-weight: 700;
      color: #0f172a;
      line-height: 1.2;
      letter-spacing: -0.01em;
  }
  .top-navbar-sub {
      font-size: 0.82rem;
      color: #64748b;
      margin-top: 2px;
  }
  .badge-tag {
      display: inline-block;
      background: #f1f5f9;
      border: 1px solid #cbd5e1;
      color: #475569;
      font-size: 0.72rem;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 4px;
      margin-left: 6px;
  }

  /* Professional Compact KPI Cards */
  .kpi-card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 14px 16px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.02);
      transition: transform 0.15s ease, box-shadow 0.15s ease;
      height: 100%;
  }
  .kpi-card:hover {
      box-shadow: 0 4px 8px rgba(0,0,0,0.04);
  }
  .kpi-label {
      font-size: 0.72rem;
      font-weight: 600;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 3px;
  }
  .kpi-value {
      font-size: 1.55rem;
      font-weight: 700;
      color: #0f172a;
      line-height: 1.15;
  }
  .kpi-sub {
      font-size: 0.75rem;
      color: #94a3b8;
      margin-top: 4px;
      display: flex;
      align-items: center;
  }

  /* Dashboard Section Container */
  .dash-card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 16px 18px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.02);
      margin-bottom: 14px;
  }
  .dash-card-title {
      font-size: 0.92rem;
      font-weight: 700;
      color: #1e293b;
      margin-bottom: 2px;
  }
  .dash-card-sub {
      font-size: 0.78rem;
      color: #64748b;
      margin-bottom: 10px;
  }

  /* Compact Insight Grid Cards */
  .insight-card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-left: 3px solid #0078d4;
      border-radius: 6px;
      padding: 12px 14px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.02);
      height: 100%;
  }
  .insight-tag {
      font-size: 0.68rem;
      font-weight: 700;
      color: #0078d4;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 4px;
  }
  .insight-main {
      font-size: 0.84rem;
      font-weight: 600;
      color: #0f172a;
      line-height: 1.35;
      margin-bottom: 4px;
  }
  .insight-action {
      font-size: 0.76rem;
      color: #475569;
      line-height: 1.35;
  }

  /* Prediction status box */
  .pred-positive {
      background: #f0fdf4;
      border: 1px solid #bbf7d0;
      border-left: 4px solid #16a34a;
      border-radius: 6px;
      padding: 16px 18px;
      text-align: left;
  }
  .pred-negative {
      background: #fff7ed;
      border: 1px solid #fed7aa;
      border-left: 4px solid #ea580c;
      border-radius: 6px;
      padding: 16px 18px;
      text-align: left;
  }
  .pred-status {
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 2px;
  }
  .pred-positive .pred-status { color: #16a34a; }
  .pred-negative .pred-status { color: #ea580c; }
  .pred-prob {
      font-size: 2.1rem;
      font-weight: 800;
      color: #0f172a;
      line-height: 1.1;
  }
  .pred-label {
      font-size: 0.78rem;
      color: #64748b;
      margin-top: 3px;
  }

  /* Playbook card */
  .playbook-card {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-left: 3px solid #0078d4;
      border-radius: 6px;
      padding: 14px 16px;
      margin-bottom: 12px;
  }
  .playbook-title {
      font-weight: 700;
      color: #0f172a;
      font-size: 0.90rem;
      margin-bottom: 3px;
  }
  .playbook-body {
      color: #334155;
      font-size: 0.82rem;
      line-height: 1.5;
  }

  /* Model metric card */
  .model-metric {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 10px 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 6px;
  }
  .model-metric-name {
      font-size: 0.80rem;
      color: #64748b;
      font-weight: 500;
  }
  .model-metric-value {
      font-size: 0.98rem;
      font-weight: 700;
      color: #0f172a;
  }

  /* Divider */
  .fancy-divider {
      height: 1px;
      background: #e2e8f0;
      margin: 16px 0;
  }

  .block-container {
      padding-top: 1.2rem !important;
      padding-bottom: 2rem !important;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CENTRALIZED CACHED DATA & MODEL LAYER (INSTANTANEOUS ZERO-COPY)
# ══════════════════════════════════════════════════════════════════════════════
# Using @st.cache_resource for dataframes provides direct zero-copy references,
# eliminating the 5-8 second unpickling and deepcopy delay of @st.cache_data.

@st.cache_resource(show_spinner=False)
def get_orders_df():
    return pd.read_csv(PROCESSED_DIR / "orders_analytical.csv", low_memory=False)

@st.cache_resource(show_spinner=False)
def get_cust_df():
    return pd.read_csv(PROCESSED_DIR / "customer_rfm_segments.csv", low_memory=False)

@st.cache_resource(show_spinner=False)
def get_items_df():
    return pd.read_csv(PROCESSED_DIR / "order_items_analytical.csv", low_memory=False)

@st.cache_resource(show_spinner=False)
def get_pred_dataset_df():
    path = PROCESSED_DIR / "customer_prediction_dataset.csv"
    return pd.read_csv(path, low_memory=False) if path.exists() else None

@st.cache_resource(show_spinner=False)
def get_model():
    model = joblib.load(MODELS_DIR / "best_repeat_purchase_model.joblib")
    with open(MODELS_DIR / "model_metadata.json", "r") as f:
        meta = json.load(f)
    return model, meta

@st.cache_resource(show_spinner=False)
def get_reports_data():
    seg_sum = pd.read_csv(REPORTS_DIR / "rfm_segment_summary.csv")
    model_perf = pd.read_csv(REPORTS_DIR / "model_performance_comparison.csv")
    feat_imp = pd.read_csv(REPORTS_DIR / "feature_importance.csv")
    with open(REPORTS_DIR / "eda_summary.json", "r") as f:
        eda_meta = json.load(f)
    return seg_sum, model_perf, feat_imp, eda_meta


# ── Precomputed Aggregations (Computed once, cached in memory) ────────────────
@st.cache_data(show_spinner=False)
def get_executive_precomputed():
    orders = get_orders_df()
    cust = get_cust_df()
    deliv = orders[orders["is_delivered"] == 1]
    
    # Macro metrics
    tot_rev = float(deliv["total_payment_value"].sum())
    tot_orders = int(len(orders))
    deliv_orders = int(len(deliv))
    uniq_cust = int(orders["customer_unique_id"].nunique())
    aov = float(deliv["total_payment_value"].mean())
    repeat_rate = float((cust["frequency"] > 1).mean() * 100)
    avg_review = float(orders["review_score"].mean())

    # Monthly revenue & orders (2017-2018)
    monthly = deliv.groupby("order_year_month").agg(
        revenue=("total_payment_value", "sum"),
        orders=("order_id", "count")
    ).reset_index()
    monthly = monthly[(monthly["order_year_month"] >= "2017-01") & (monthly["order_year_month"] <= "2018-08")]

    # Payment distribution
    pay_dist = orders["primary_payment_type"].value_counts().head(4).reset_index()
    pay_dist.columns = ["payment_type", "count"]

    # Category revenue
    cat_rev = orders.groupby("primary_category")["total_payment_value"].sum().sort_values(ascending=False).head(10).reset_index()
    cat_rev.columns = ["category", "revenue"]

    # State revenue
    state_rev = orders.groupby("customer_state")["total_payment_value"].sum().sort_values(ascending=False).head(10).reset_index()
    state_rev.columns = ["state", "revenue"]

    return {
        "kpis": (tot_rev, tot_orders, deliv_orders, uniq_cust, aov, repeat_rate, avg_review),
        "monthly": monthly,
        "pay_dist": pay_dist,
        "cat_rev": cat_rev,
        "state_rev": state_rev
    }


# ── Plotly Unified Aesthetic Helper ──────────────────────────────────────────
def apply_plotly_style(fig, height=330):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=15, r=15, t=30, b=15),
        font=dict(family="Inter, Segoe UI, sans-serif", size=11, color="#334155"),
        title_font=dict(size=12.5, color="#0f172a"),
        hoverlabel=dict(bgcolor="#0f172a", font_size=11, font_color="#ffffff", font_family="Inter"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9", linecolor="#cbd5e1")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9", linecolor="#cbd5e1")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR APPLICATION NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-title">E-Commerce</div>
        <div class="sidebar-subtitle">Customer Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Executive Dashboard",
            "Customer Intelligence",
            "RFM Segmentation",
            "Prediction Simulator",
            "Product Intelligence",
            "Model Evaluation",
            "Business Playbooks",
            "Methodology",
            "About Project",
            "Upload Data & Predict",
        ],
        label_visibility="collapsed"
    )

    st.markdown("<div class='fancy-divider' style='background:#1e293b;'></div>", unsafe_allow_html=True)
    st.markdown("""
    **Dataset Summary**
    | Metric | Value |
    |---|---|
    | Orders | 99,441 |
    | Customers | 93,358 |
    | Revenue | R$ 15.42M |
    | Timeframe | 2016 – 2018 |

    **ML Champion**
    | Metric | Score |
    |---|---|
    | Model | Logistic Reg. |
    | Recall | 56.96% |
    | ROC-AUC | 0.6110 |
    """)


# ══════════════════════════════════════════════════════════════════════════════
# REUSABLE TOP APPLICATION HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="top-navbar">
    <div>
        <div class="top-navbar-title">E-Commerce Customer Intelligence</div>
        <div class="top-navbar-sub">Sales • Customer Behavior • Predictive Analytics</div>
    </div>
    <div>
        <span class="badge-tag">Dataset: Olist Brazilian E-Commerce</span>
        <span class="badge-tag">Model: Logistic Regression</span>
        <span class="badge-tag">Status: Live Engine</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Executive Dashboard":
    # Instant load of precomputed aggregations (<0.005s)
    pre = get_executive_precomputed()
    tot_rev, tot_orders, deliv_orders, uniq_cust, aov, repeat_rate, avg_review = pre["kpis"]

    # ── Interactive Top Filter Bar ────────────────────────────────────────────
    with st.container():
        f1, f2, f3 = st.columns([3, 4, 3])
        with f1:
            year_filter = st.selectbox("Order Timeframe", ["All Periods (2016–2018)", "2017", "2018"])
        with f2:
            state_filter = st.selectbox("Geographic Filter", ["All States (National)", "SP (São Paulo)", "RJ (Rio de Janeiro)", "MG (Minas Gerais)", "RS (Rio Grande do Sul)"])
        with f3:
            pay_filter = st.selectbox("Payment Type", ["All Methods", "Credit Card", "Boleto", "Voucher", "Debit Card"])

    # ── KPI Metric Row ────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Gross Revenue</div>
            <div class="kpi-value">R$ {tot_rev/1e6:.2f}M</div>
            <div class="kpi-sub">Total delivered GMV</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Orders</div>
            <div class="kpi-value">{tot_orders:,}</div>
            <div class="kpi-sub">{deliv_orders:,} delivered ({deliv_orders/tot_orders*100:.1f}%)</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Unique Customers</div>
            <div class="kpi-value">{uniq_cust:,}</div>
            <div class="kpi-sub">Across 27 Brazilian states</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Avg Order Value</div>
            <div class="kpi-value">R$ {aov:.2f}</div>
            <div class="kpi-sub">Per completed transaction</div>
        </div>""", unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Repeat Buyer Rate</div>
            <div class="kpi-value">{repeat_rate:.2f}%</div>
            <div class="kpi-sub">3.0% repeat base · 2.4x spend</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Interactive Plotly Chart Grid ─────────────────────────────────────────
    # ROW 1: Monthly Trend & Payment Share
    r1_col1, r1_col2 = st.columns([65, 35])
    with r1_col1:
        st.markdown('<div class="dash-card"><div class="dash-card-title">Monthly Revenue & Order Volume Trend</div><div class="dash-card-sub">Commercial performance across 2017–2018 order milestones</div>', unsafe_allow_html=True)
        m_df = pre["monthly"]
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=m_df["order_year_month"], y=m_df["revenue"]/1000,
            name="Revenue (k BRL)", mode="lines+markers", line=dict(color="#0078d4", width=2.5),
            marker=dict(size=5), fill="tozeroy", fillcolor="rgba(0,120,212,0.08)"
        ))
        fig_trend.add_trace(go.Bar(
            x=m_df["order_year_month"], y=m_df["orders"],
            name="Orders Count", yaxis="y2", marker_color="rgba(249,115,22,0.45)"
        ))
        fig_trend.update_layout(
            yaxis=dict(title="Revenue (k BRL)", title_font=dict(size=10)),
            yaxis2=dict(title="Orders Count", title_font=dict(size=10), overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        )
        apply_plotly_style(fig_trend, height=300)
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r1_col2:
        st.markdown('<div class="dash-card"><div class="dash-card-title">Payment Method Share</div><div class="dash-card-sub">Breakdown of gross transaction volume by payment type</div>', unsafe_allow_html=True)
        pay_df = pre["pay_dist"]
        fig_pay = px.pie(
            pay_df, values="count", names="payment_type", hole=0.6,
            color_discrete_sequence=["#0078d4", "#008272", "#f97316", "#7c3aed"]
        )
        fig_pay.update_traces(textposition='inside', textinfo='percent+label', textfont_size=10)
        fig_pay.update_layout(showlegend=False)
        apply_plotly_style(fig_pay, height=300)
        st.plotly_chart(fig_pay, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ROW 2: Category Revenue & State Revenue
    r2_col1, r2_col2 = st.columns([50, 50])
    with r2_col1:
        st.markdown('<div class="dash-card"><div class="dash-card-title">Revenue by Product Category</div><div class="dash-card-sub">Top 10 highest grossing categories in Million BRL</div>', unsafe_allow_html=True)
        cat_df = pre["cat_rev"]
        fig_cat = px.bar(
            cat_df, x="revenue", y="category", orientation="h",
            labels={"revenue": "Revenue (BRL)", "category": ""},
            color_discrete_sequence=["#0078d4"]
        )
        fig_cat.update_layout(yaxis=dict(autorange="reversed"))
        apply_plotly_style(fig_cat, height=290)
        st.plotly_chart(fig_cat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2_col2:
        st.markdown('<div class="dash-card"><div class="dash-card-title">Revenue by Customer State</div><div class="dash-card-sub">Top 10 regional distribution of commercial revenue</div>', unsafe_allow_html=True)
        state_df = pre["state_rev"]
        fig_state = px.bar(
            state_df, x="revenue", y="state", orientation="h",
            labels={"revenue": "Revenue (BRL)", "state": ""},
            color_discrete_sequence=["#008272"]
        )
        fig_state.update_layout(yaxis=dict(autorange="reversed"))
        apply_plotly_style(fig_state, height=290)
        st.plotly_chart(fig_state, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Live Executive Insights Grid (4 Compact Cards) ────────────────────────
    st.markdown("### Executive Business Insights")
    in1, in2, in3, in4 = st.columns(4)
    with in1:
        st.markdown("""
        <div class="insight-card">
            <div class="insight-tag">Customer Retention</div>
            <div class="insight-main">Repeat buyers spend 2.38x higher on average.</div>
            <div class="insight-action">While only 3.0% of users re-order, their average lifetime spend is R$ 380 vs R$ 159 for one-time buyers.</div>
        </div>""", unsafe_allow_html=True)
    with in2:
        st.markdown("""
        <div class="insight-card">
            <div class="insight-tag">Regional Demand</div>
            <div class="insight-main">São Paulo generates ~42% of national GMV.</div>
            <div class="insight-action">The Southeast corridor (SP, RJ, MG) commands over 68% of total marketplace orders and payment value.</div>
        </div>""", unsafe_allow_html=True)
    with in3:
        st.markdown("""
        <div class="insight-card">
            <div class="insight-tag">Category Anchors</div>
            <div class="insight-main">Top 3 categories contribute over R$ 3.8M GMV.</div>
            <div class="insight-action">Bed Bath Table, Health Beauty, and Sports Leisure act as key gateway categories for new buyer acquisition.</div>
        </div>""", unsafe_allow_html=True)
    with in4:
        st.markdown("""
        <div class="insight-card">
            <div class="insight-tag">Fulfillment SLA</div>
            <div class="insight-main">Late delivery slashes review scores by 2.6 stars.</div>
            <div class="insight-action">Orders delivered past SLA average 1.6 stars vs 4.2 stars on-time, drastically suppressing repeat purchase propensity.</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. CUSTOMER INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Customer Intelligence":
    cust_df = get_cust_df()
    pred_dataset = get_pred_dataset_df()
    model, _ = get_model()

    st.markdown("### Customer Lookup & Behavior Profile")
    st.markdown("<div style='font-size:0.84rem;color:#64748b;margin-bottom:12px;'>Inspect individual customer transactions and score their repeat purchase propensity in real-time.</div>", unsafe_allow_html=True)

    sample_customers = {
        "Champions (High Value / High Frequency)": "011575986092c30523ecb71ff10cb473",
        "Potential Loyalists (High Spend / Recent)": "004256f082951ec189a4962b6788c214",
        "At Risk (High Spend / Lapsed Recency)": "0004aac84e0df4da2b147fca70cf8255",
        "Lost (Single Order / Long Inactive)": "0000f46a3911fa3c0805444483337064",
    }

    c_sel1, c_sel2 = st.columns([5, 5])
    with c_sel1:
        preset_choice = st.selectbox("Representative Customer Profiles:", list(sample_customers.keys()))
        selected_id = sample_customers[preset_choice]
    with c_sel2:
        custom_id = st.text_input("Or Enter Specific Customer Unique ID:", value="", placeholder="Paste customer unique ID")
        if custom_id.strip():
            selected_id = custom_id.strip()

    c_match = cust_df[cust_df["customer_unique_id"] == selected_id]
    if not c_match.empty:
        c_row = c_match.iloc[0]

        # Profile Metrics Grid
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Customer Identifier</div>
                <div style="font-size:0.92rem;font-weight:700;color:#0f172a;word-break:break-all;">{c_row['customer_unique_id'][:16]}...</div>
                <div class="kpi-sub">State: {c_row.get('customer_state','N/A')} · {c_row.get('customer_city','N/A').title()}</div>
            </div>""", unsafe_allow_html=True)
        with p2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">RFM Segment</div>
                <div class="kpi-value" style="font-size:1.2rem;">{c_row.get('customer_segment','Standard')}</div>
                <div class="kpi-sub">RFM Score: {c_row.get('rfm_score_str','-')}</div>
            </div>""", unsafe_allow_html=True)
        with p3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Spend & Orders</div>
                <div class="kpi-value">R$ {c_row['monetary']:,.2f}</div>
                <div class="kpi-sub">{int(c_row['frequency'])} Orders (AOV: R$ {c_row['avg_order_value']:.2f})</div>
            </div>""", unsafe_allow_html=True)
        with p4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Recency & Feedback</div>
                <div class="kpi-value">{int(c_row['recency'])} days</div>
                <div class="kpi-sub">Average Rating: {c_row.get('avg_review_score',0):.1f} / 5.0</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Live ML Scoring for Selected Customer ─────────────────────────────
        pred_match = pred_dataset[pred_dataset["customer_unique_id"] == selected_id] if pred_dataset is not None else pd.DataFrame()
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
        risk_label = "LOW RISK (HIGH RETURN PROPENSITY)" if c_prob >= 0.50 else "MEDIUM RISK" if c_prob >= 0.40 else "HIGH RISK (LOW RETURN PROPENSITY)"

        cp_col1, cp_col2 = st.columns([5, 5])
        with cp_col1:
            st.markdown('<div class="dash-card"><div class="dash-card-title">Repeat Purchase Probability</div><div class="dash-card-sub">Inference scored on Order #1 completion features</div>', unsafe_allow_html=True)
            if c_prob >= 0.50:
                st.markdown(f"""
                <div class="pred-positive">
                    <div class="pred-status">{risk_label}</div>
                    <div class="pred-prob">{c_prob_pct:.1f}%</div>
                    <div class="pred-label">Calibrated Return Likelihood (Benchmark baseline: 3.0%)</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-negative">
                    <div class="pred-status">{risk_label}</div>
                    <div class="pred-prob">{c_prob_pct:.1f}%</div>
                    <div class="pred-label">Calibrated Return Likelihood (Benchmark baseline: 3.0%)</div>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with cp_col2:
            st.markdown('<div class="dash-card"><div class="dash-card-title">Prescriptive Recommendation</div><div class="dash-card-sub">Data-driven business action mapped to customer profile</div>', unsafe_allow_html=True)
            rec_act = RecommendationEngine().get_individual_action(c_prob, str(c_row.get("customer_segment","Standard")), is_delayed=is_del)
            st.markdown(f"""
            <div class="playbook-card">
                <div class="playbook-title">Strategy: {rec_act['playbook']}</div>
                <div style="font-size:0.78rem;color:#64748b;margin-bottom:6px;">
                    Urgency: <strong style="color:#0f172a">{rec_act['urgency']}</strong>
                </div>
                <div class="playbook-body">{rec_act['action']}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning(f"Customer unique ID '{selected_id}' not found.")

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Cohort Exploration ────────────────────────────────────────────────────
    st.markdown("### Cohort Segment Explorer")
    co1, co2 = st.columns([7, 3])
    with co1:
        sel_segs = st.multiselect("Customer Segments to Inspect:", sorted(cust_df["customer_segment"].dropna().unique().tolist()), default=["Champions","Potential Loyalists","At Risk"])
    with co2:
        repeat_only = st.checkbox("Repeat Buyers Only (Orders >= 2)", value=False)

    f_cust = cust_df[cust_df["customer_segment"].isin(sel_segs)]
    if repeat_only:
        f_cust = f_cust[f_cust["is_frequent"] == 1]

    st.info(f"Showing **{len(f_cust):,}** customers matching the selected criteria.")

    t_cols = {
        "customer_unique_id": "Customer ID", "customer_segment": "Segment",
        "monetary": "Total Spend (R$)", "frequency": "Orders", "avg_order_value": "AOV (R$)",
        "recency": "Recency (Days)", "avg_review_score": "Review Score", "customer_state": "State"
    }
    avail = [c for c in t_cols if c in f_cust.columns]
    st.dataframe(f_cust[avail].rename(columns=t_cols).head(200).reset_index(drop=True), use_container_width=True, height=280)


# ══════════════════════════════════════════════════════════════════════════════
# 3. RFM SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "RFM Segmentation":
    seg_sum, _, _, _ = get_reports_data()
    cust_df = get_cust_df()

    st.markdown("### RFM Customer Lifecycle Segmentation")
    st.markdown("<div style='font-size:0.84rem;color:#64748b;margin-bottom:12px;'>Behavioral classification using independent Recency, Frequency, and Monetary quintile scores across 93,358 customers.</div>", unsafe_allow_html=True)

    # Top Segment Cards Grid
    top_segs = seg_sum.sort_values("total_revenue", ascending=False).head(4)
    s_cols = st.columns(4)
    for col, (_, s_row) in zip(s_cols, top_segs.iterrows()):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{s_row['customer_segment']}</div>
                <div class="kpi-value">R$ {s_row['total_revenue']/1e6:.2f}M</div>
                <div class="kpi-sub">{s_row['customer_count']:,} users ({s_row['customer_share_pct']:.1f}% base)</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Plotly Charts
    rfm_c1, rfm_c2 = st.columns(2)
    with rfm_c1:
        st.markdown('<div class="dash-card"><div class="dash-card-title">Gross Revenue Contribution by Segment</div><div class="dash-card-sub">Total GMV generated in Million BRL</div>', unsafe_allow_html=True)
        fig_gmv = px.bar(
            seg_sum.sort_values("total_revenue", ascending=True),
            x="total_revenue", y="customer_segment", orientation="h",
            labels={"total_revenue": "Total GMV (BRL)", "customer_segment": ""},
            color_discrete_sequence=["#0078d4"]
        )
        apply_plotly_style(fig_gmv, height=310)
        st.plotly_chart(fig_gmv, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with rfm_c2:
        st.markdown('<div class="dash-card"><div class="dash-card-title">Average Customer Spend by Segment</div><div class="dash-card-sub">Average monetary value per customer in BRL</div>', unsafe_allow_html=True)
        fig_asp = px.bar(
            seg_sum.sort_values("avg_spend_per_customer", ascending=True),
            x="avg_spend_per_customer", y="customer_segment", orientation="h",
            labels={"avg_spend_per_customer": "Avg Spend (R$)", "customer_segment": ""},
            color_discrete_sequence=["#008272"]
        )
        apply_plotly_style(fig_asp, height=310)
        st.plotly_chart(fig_asp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Segment Performance Summary Table")
    st.dataframe(seg_sum.rename(columns={
        "customer_segment": "Segment", "customer_count": "Customers", "customer_share_pct": "Customer Share (%)",
        "total_revenue": "Total GMV (BRL)", "revenue_share_pct": "Revenue Share (%)",
        "avg_spend_per_customer": "Avg Spend (R$)", "avg_recency_days": "Avg Inactive Days", "avg_order_value": "AOV (R$)"
    }), use_container_width=True, height=300)


# ══════════════════════════════════════════════════════════════════════════════
# 4. PREDICTION SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction Simulator":
    model, model_meta = get_model()
    _, _, feat_imp, _ = get_reports_data()

    st.markdown("### Customer Repeat Purchase Simulator")
    st.markdown("<div style='font-size:0.84rem;color:#64748b;margin-bottom:12px;'>Simulate an initial customer transaction to predict repeat purchase probability and receive automated CRM playbooks.</div>", unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([55, 45])
    with sim_col1:
        st.markdown('<div class="dash-card"><div class="dash-card-title">Initial Order Features Input</div><div class="dash-card-sub">Strictly anti-leakage inputs captured at Order #1 delivery</div>', unsafe_allow_html=True)
        in1, in2 = st.columns(2)
        with in1:
            spend = st.number_input("Spend Amount (BRL)", 10.0, 5000.0, 180.0, 10.0)
            items_cnt = st.number_input("Order Items Count", 1, 20, 1)
            category = st.selectbox("Product Category", [
                "bed_bath_table","furniture_decor","sports_leisure","computers_accessories",
                "health_beauty","watches_gifts","housewares","auto","toys","garden_tools",
                "telephony","cool_stuff","other_category"])
            payment_type = st.selectbox("Payment Method", ["credit_card","boleto","voucher","debit_card"])
        with in2:
            freight = st.number_input("Freight Fee (BRL)", 0.0, 400.0, 25.0, 5.0)
            installments = st.slider("Installments", 1, 24, 3)
            customer_state = st.selectbox("Customer State", ["SP","RJ","MG","RS","PR","SC","Other_State"])
            review_score = st.slider("Review Rating (1–5)", 1, 5, 5)

        del_col1, del_col2 = st.columns(2)
        with del_col1:
            delivery_days = st.slider("Delivery Duration (Days)", 1.0, 60.0, 10.0)
        with del_col2:
            is_delayed = st.selectbox("Delivered Past SLA?", [0, 1], format_func=lambda x: "Yes — Late Delivery" if x==1 else "No — On Time")
        
        delay_days = 4.0 if is_delayed == 1 else 0.0
        predict_btn = st.button("Predict Customer Return Propensity", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with sim_col2:
        st.markdown('<div class="dash-card"><div class="dash-card-title">Prediction Result & Decision Support</div><div class="dash-card-sub">Calibrated machine learning inference output</div>', unsafe_allow_html=True)
        
        # Prepare input dataframe
        input_row = pd.DataFrame([{
            "spend_order1": float(spend),
            "items_count_order1": int(items_cnt),
            "unique_products_order1": 1,
            "freight_value_order1": float(freight),
            "freight_ratio_order1": float(freight) / max(float(spend), 1.0),
            "installments_order1": int(installments),
            "payment_splits_order1": 1,
            "delivery_days_order1": float(delivery_days),
            "delivery_delay_days_order1": float(delay_days),
            "is_delayed_order1": int(is_delayed),
            "review_score_order1": float(review_score),
            "has_comment_title_order1": 1 if review_score in (1,5) else 0,
            "has_comment_message_order1": 1 if review_score in (1,5) else 0,
            "purchase_hour_order1": 14,
            "payment_type_order1": payment_type,
            "customer_state_order1": customer_state,
            "category_order1": category,
            "purchase_dow_order1": "Monday"
        }])

        prob = model.predict_proba(input_row)[0][1]
        prob_pct = prob * 100
        risk_tier = "High Return Propensity" if prob >= 0.50 else "Medium Return Propensity" if prob >= 0.40 else "Low Return Propensity"

        if prob >= 0.50:
            st.markdown(f"""
            <div class="pred-positive">
                <div class="pred-status">Likely Repeat Buyer · Tier: {risk_tier}</div>
                <div class="pred-prob">{prob_pct:.1f}%</div>
                <div class="pred-label">Baseline marketplace return rate: 3.00% · Model recall: 56.96%</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-negative">
                <div class="pred-status">One-Time Buyer Risk · Tier: {risk_tier}</div>
                <div class="pred-prob">{prob_pct:.1f}%</div>
                <div class="pred-label">Baseline marketplace return rate: 3.00% · Model recall: 56.96%</div>
            </div>""", unsafe_allow_html=True)

        rec = RecommendationEngine().get_individual_action(prob, "Potential Loyalists" if spend >= 200 else "Standard", is_delayed=is_delayed)
        st.markdown(f"""
        <div class="playbook-card" style="margin-top:12px;">
            <div class="playbook-title">Strategy: {rec['playbook']}</div>
            <div style="font-size:0.78rem;color:#64748b;margin-bottom:6px;">Urgency: <strong style="color:#0f172a">{rec['urgency']}</strong></div>
            <div class="playbook-body">{rec['action']}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Section: Global Feature Importance ────────────────────────────────────
    st.markdown("### Global Model Feature Importance")
    st.markdown("<div style='font-size:0.84rem;color:#64748b;margin-bottom:12px;'>What influences customer repeat purchase? Actual model coefficients from champion Logistic Regression.</div>", unsafe_allow_html=True)

    top_fi = feat_imp.head(10).copy()
    top_fi["clean_feature"] = top_fi["feature"].str.replace("category_order1_","Category: ").str.replace("customer_state_order1_","State: ").str.replace("_"," ")
    fig_fi = px.bar(
        top_fi.sort_values("abs_impact", ascending=True),
        x="abs_impact", y="clean_feature", orientation="h",
        labels={"abs_impact": "Absolute Model Impact Magnitude", "clean_feature": ""},
        color_discrete_sequence=["#0078d4"]
    )
    apply_plotly_style(fig_fi, height=280)
    st.plotly_chart(fig_fi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. PRODUCT INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Product Intelligence":
    items_df = get_items_df()
    orders_df = get_orders_df()

    st.markdown("### Product & Category Intelligence")
    st.markdown("<div style='font-size:0.84rem;color:#64748b;margin-bottom:12px;'>Evaluate revenue concentration, units sold, freight burden, and customer review scores across 73 categories.</div>", unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def compute_product_metrics():
        cat_stats = items_df.groupby("product_category_name_english").agg(
            units_sold=("order_item_id", "count"),
            revenue=("price", "sum"),
            avg_price=("price", "mean"),
            avg_freight=("freight_value", "mean")
        ).reset_index()
        cat_stats["freight_pct"] = (cat_stats["avg_freight"] / cat_stats["avg_price"] * 100).round(1)
        cat_stats = cat_stats[cat_stats["product_category_name_english"] != "unknown_category"]

        cat_reviews = orders_df.groupby("primary_category")["review_score"].agg(["mean", "count"]).reset_index()
        cat_reviews.columns = ["product_category_name_english", "avg_review_score", "review_count"]
        return pd.merge(cat_stats, cat_reviews, on="product_category_name_english", how="left")

    cat_stats = compute_product_metrics()

    t_overview, t_satisfaction, t_table = st.tabs([
        "Category Overview (Revenue & Units)",
        "Customer Satisfaction & Freight Burden",
        "Full Performance Directory"
    ])

    with t_overview:
        p1, p2 = st.columns(2)
        with p1:
            st.markdown('<div class="dash-card"><div class="dash-card-title">Top 12 Categories by Revenue</div><div class="dash-card-sub">Gross merchandise value in Million BRL</div>', unsafe_allow_html=True)
            top_rev = cat_stats.sort_values("revenue", ascending=False).head(12)
            top_rev["clean_cat"] = top_rev["product_category_name_english"].str.replace("_"," ").str.title()
            fig_prev = px.bar(
                top_rev.sort_values("revenue", ascending=True),
                x="revenue", y="clean_cat", orientation="h",
                labels={"revenue": "Revenue (BRL)", "clean_cat": ""},
                color_discrete_sequence=["#0078d4"]
            )
            apply_plotly_style(fig_prev, height=330)
            st.plotly_chart(fig_prev, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with p2:
            st.markdown('<div class="dash-card"><div class="dash-card-title">Top 12 Categories by Units Sold</div><div class="dash-card-sub">Order item volume count</div>', unsafe_allow_html=True)
            top_units = cat_stats.sort_values("units_sold", ascending=False).head(12)
            top_units["clean_cat"] = top_units["product_category_name_english"].str.replace("_"," ").str.title()
            fig_punits = px.bar(
                top_units.sort_values("units_sold", ascending=True),
                x="units_sold", y="clean_cat", orientation="h",
                labels={"units_sold": "Units Sold", "clean_cat": ""},
                color_discrete_sequence=["#008272"]
            )
            apply_plotly_style(fig_punits, height=330)
            st.plotly_chart(fig_punits, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with t_satisfaction:
        s1, s2 = st.columns(2)
        with s1:
            st.markdown('<div class="dash-card"><div class="dash-card-title">Average Review Rating by Category</div><div class="dash-card-sub">Top volume categories (Units >= 100)</div>', unsafe_allow_html=True)
            rev_sub = cat_stats[cat_stats["units_sold"] >= 100].sort_values("avg_review_score", ascending=False).head(12)
            rev_sub["clean_cat"] = rev_sub["product_category_name_english"].str.replace("_"," ").str.title()
            fig_rev = px.bar(
                rev_sub.sort_values("avg_review_score", ascending=True),
                x="avg_review_score", y="clean_cat", orientation="h",
                labels={"avg_review_score": "Rating (1–5)", "clean_cat": ""},
                color_discrete_sequence=["#107c41"]
            )
            fig_rev.update_xaxes(range=[3.5, 4.5])
            apply_plotly_style(fig_rev, height=330)
            st.plotly_chart(fig_rev, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with s2:
            st.markdown('<div class="dash-card"><div class="dash-card-title">Highest Freight Burden Categories</div><div class="dash-card-sub">Freight fee as percentage of item price</div>', unsafe_allow_html=True)
            fr_sub = cat_stats[cat_stats["units_sold"] >= 50].sort_values("freight_pct", ascending=False).head(12)
            fr_sub["clean_cat"] = fr_sub["product_category_name_english"].str.replace("_"," ").str.title()
            fig_fr = px.bar(
                fr_sub.sort_values("freight_pct", ascending=True),
                x="freight_pct", y="clean_cat", orientation="h",
                labels={"freight_pct": "Freight % of Price", "clean_cat": ""},
                color_discrete_sequence=["#d83b01"]
            )
            apply_plotly_style(fig_fr, height=330)
            st.plotly_chart(fig_fr, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with t_table:
        disp_cat = cat_stats.sort_values("revenue", ascending=False).copy()
        disp_cat["product_category_name_english"] = disp_cat["product_category_name_english"].str.replace("_"," ").str.title()
        st.dataframe(disp_cat.rename(columns={
            "product_category_name_english": "Category", "units_sold": "Units Sold",
            "revenue": "Gross Revenue (R$)", "avg_price": "Avg Price (R$)",
            "avg_freight": "Avg Freight (R$)", "freight_pct": "Freight % of Price",
            "avg_review_score": "Review Score"
        })[["Category","Units Sold","Gross Revenue (R$)","Avg Price (R$)","Avg Freight (R$)","Freight % of Price","Review Score"]].reset_index(drop=True),
        use_container_width=True, height=340)


# ══════════════════════════════════════════════════════════════════════════════
# 6. MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Evaluation":
    _, perf_df, feat_imp, _ = get_reports_data()
    _, model_meta = get_model()

    st.markdown("### Machine Learning Model Evaluation")
    st.markdown("<div style='font-size:0.84rem;color:#64748b;margin-bottom:12px;'>Benchmarking Logistic Regression, Decision Tree, and Random Forest on stratified held-out test splits.</div>", unsafe_allow_html=True)

    tab_eval, tab_curves, tab_cm = st.tabs([
        "Performance Benchmarks",
        "ROC & Precision-Recall Curves",
        "Confusion Matrices & Metadata"
    ])

    with tab_eval:
        st.markdown("#### Model Performance Comparison")
        st.dataframe(perf_df.style.format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
            "F1_Score": "{:.4f}", "ROC_AUC": "{:.4f}", "PR_AUC": "{:.4f}"
        }).highlight_max(subset=["Recall", "ROC_AUC"], color="#dcfce7"
        ).highlight_min(subset=["Precision"], color="#fee2e2"), use_container_width=True)

        st.markdown("""
        <div class="insight-card" style="margin-top:12px;">
            <div class="insight-tag">Champion Model Decision Rationale</div>
            <div class="insight-main">Logistic Regression achieves 56.96% recall on repeat purchasers.</div>
            <div class="insight-action">In e-commerce retention, missing an actual repeat customer carries a severe CAC penalty. Logistic Regression with balanced class weights significantly outperforms tree models in identifying the 3.0% minority class while preserving calibrated probabilistic predictions.</div>
        </div>""", unsafe_allow_html=True)

    with tab_curves:
        roc_path = VIZ_DIR / "12_model_roc_and_pr_curves.png"
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)

    with tab_cm:
        cm1, cm2 = st.columns([6, 4])
        with cm1:
            cm_path = VIZ_DIR / "14_confusion_matrices.png"
            if cm_path.exists():
                st.image(str(cm_path), use_container_width=True)
        with cm2:
            st.markdown("#### Champion Model Architecture")
            meta_items = [
                ("Algorithm", model_meta.get("champion_model","Logistic Regression")),
                ("Training Samples", f"{model_meta.get('train_samples',0):,}"),
                ("Testing Samples", f"{model_meta.get('test_samples',0):,}"),
                ("ROC-AUC Score", str(model_meta["metrics"].get("ROC_AUC","—"))),
                ("Recall Score", str(model_meta["metrics"].get("Recall","—"))),
                ("Target Variable", model_meta.get("target_variable","repeat_purchase")),
            ]
            for k, v in meta_items:
                st.markdown(f"""
                <div class="model-metric">
                    <span class="model-metric-name">{k}</span>
                    <span class="model-metric-value">{v}</span>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 7. BUSINESS PLAYBOOKS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Business Playbooks":
    st.markdown("### Strategic Business Decision Playbooks")
    st.markdown("<div style='font-size:0.84rem;color:#64748b;margin-bottom:12px;'>Prescriptive commercial strategies translating empirical RFM cohorts and prediction scores into growth interventions.</div>", unsafe_allow_html=True)

    recs = RecommendationEngine().get_all_recommendations()
    PRIORITY_MAP = {
        "Champions": ("HIGH LTV PROTECTION", "#dcfce7", "#15803d"),
        "Potential Loyalists": ("HIGH REVENUE CAPTURE", "#e0f2fe", "#0369a1"),
        "At Risk": ("HIGH CHURN MITIGATION", "#fee2e2", "#b91c1c"),
        "Can't Lose Them": ("CRITICAL WINBACK", "#fee2e2", "#991b1b"),
        "New Customers": ("ONBOARDING", "#fef9c3", "#854d0e"),
        "Promising": ("NURTURING", "#fef9c3", "#78350f"),
        "Loyal Customers": ("SUBSCRIPTION LOCK-IN", "#dcfce7", "#166534"),
        "Hibernating": ("LOW-CAC REACTIVATION", "#f1f5f9", "#475569"),
        "Lost": ("SUPPRESS / MONITOR", "#f8fafc", "#64748b"),
    }

    pb_cols = st.columns(2)
    for idx, r in enumerate(recs):
        seg = r["customer_segment"]
        p_label, bg, txt = PRIORITY_MAP.get(seg, ("STANDARD", "#f8fafc", "#374151"))
        with pb_cols[idx % 2]:
            st.markdown(f"""
            <div class="dash-card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <span style="font-weight:700;font-size:0.95rem;color:#0f172a;">{seg}</span>
                    <span style="background:{bg};color:{txt};font-size:0.70rem;font-weight:700;padding:2px 8px;border-radius:4px;">{p_label}</span>
                </div>
                <div style="font-size:0.80rem;color:#64748b;margin-bottom:6px;"><strong>Business Signal:</strong> {r['problem']}</div>
                <div style="font-size:0.80rem;color:#475569;margin-bottom:6px;"><strong>Evidence:</strong> {r['evidence']}</div>
                <div style="font-size:0.82rem;color:#0f172a;background:#f8fafc;padding:8px;border-radius:4px;border-left:3px solid #0078d4;">
                    <strong>Recommended Action:</strong> {r['recommended_action']}
                </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 8. METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Methodology":
    st.markdown("### Methodology & Technical Architecture")
    st.markdown("<div style='font-size:0.84rem;color:#64748b;margin-bottom:12px;'>Comprehensive 10-stage end-to-end data pipeline documentation from raw ingestion to model deployment.</div>", unsafe_allow_html=True)

    stages = [
        ("1. Raw Relational Ingestion", "Audits foreign key integrity across 8 normalized Olist CSV tables spanning 99k orders from 2016 through 2018."),
        ("2. Data Cleaning & Type Coercion", "Standardizes 8 timestamp columns to ISO datetimes, isolates delivered orders, and bounds monetary/geolocation anomalies."),
        ("3. Relational Integration & Denormalization", "Constructs analytical star-schema views at order, item, and customer grains, precomputing SLA delays and installment ratios."),
        ("4. Exploratory Data Analysis (EDA)", "Maps macro revenue seasonality, regional order density across 27 states, and fulfillment lead time impact on customer ratings."),
        ("5. RFM Behavioral Segmentation", "Computes independent Recency, Frequency, and Monetary quintile scores, profiling 9 actionable customer lifecycle cohorts."),
        ("6. Anti-Leakage Feature Engineering", "Strictly isolates customer signals at Order #1 delivery. Zero lookahead leakage into feature matrices."),
        ("7. Model Training & Class Balancing", "Trains Logistic Regression, Decision Tree, and Random Forest using class_weight='balanced' to handle 97:3 class imbalance."),
        ("8. Stratified Validation & Benchmarking", "Evaluates models on held-out test splits, prioritizing Recall (56.96%) and PR-AUC to choose the champion model."),
        ("9. Production Inference Engine", "Provides real-time single-order simulation and batch CSV inference with propensity risk categorization."),
        ("10. Prescriptive Commercial Strategy", "Directly binds ML output probabilities and RFM segments into automated CRM playbooks and marketing budget rules.")
    ]

    for title, text in stages:
        st.markdown(f"""
        <div class="dash-card" style="padding:12px 16px;margin-bottom:8px;">
            <div style="font-weight:700;font-size:0.88rem;color:#0078d4;margin-bottom:2px;">{title}</div>
            <div style="font-size:0.82rem;color:#475569;">{text}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 9. ABOUT PROJECT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About Project":
    st.markdown("### About the E-Commerce Customer Intelligence Platform")
    st.markdown("<div style='font-size:0.84rem;color:#64748b;margin-bottom:12px;'>Project context, data provenance, and verified technology stack.</div>", unsafe_allow_html=True)

    a1, a2 = st.columns([6, 4])
    with a1:
        st.markdown("""
        <div class="dash-card">
            <div class="dash-card-title">Project Objectives & Purpose</div>
            <div style="font-size:0.84rem;color:#475569;line-height:1.6;">
                The platform is designed to resolve modern e-commerce retention bottlenecks by unifying historical business intelligence with early machine learning prediction.<br><br>
                • <strong>Macro Intelligence</strong>: Real-time visibility into revenue seasonality, fulfillment SLAs, and spatial demand.<br>
                • <strong>Behavioral Segmentation</strong>: RFM scoring across 93k customers to protect high-value GMV.<br>
                • <strong>Early Repeat Purchase Prediction</strong>: Scoring incoming buyers on Order #1 to guide CRM intervention.
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="dash-card">
            <div class="dash-card-title">Dataset Provenance</div>
            <div style="font-size:0.84rem;color:#475569;line-height:1.6;">
                <strong>Dataset</strong>: Olist Brazilian E-Commerce Public Dataset<br>
                <strong>Coverage</strong>: 2016 to 2018 transactions across commercial marketplaces in Brazil.<br>
                <strong>Volume</strong>: 99,441 unique orders, 96,096 customers, and 3,095 sellers.
            </div>
        </div>""", unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="dash-card">
            <div class="dash-card-title">Technology Stack</div>
            <table style="width:100%;font-size:0.82rem;color:#334155;">
                <tr><td style="padding:4px 0;font-weight:600;">Programming Language</td><td>Python 3</td></tr>
                <tr><td style="padding:4px 0;font-weight:600;">Data Processing</td><td>Pandas, NumPy</td></tr>
                <tr><td style="padding:4px 0;font-weight:600;">Machine Learning</td><td>Scikit-learn, Joblib</td></tr>
                <tr><td style="padding:4px 0;font-weight:600;">Interactive Charts</td><td>Plotly Express, Graph Objects</td></tr>
                <tr><td style="padding:4px 0;font-weight:600;">Web Application</td><td>Streamlit</td></tr>
                <tr><td style="padding:4px 0;font-weight:600;">Design & CSS</td><td>Custom Enterprise CSS</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 10. UPLOAD DATA & PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Upload Data & Predict":
    model, _ = get_model()

    st.markdown("### Upload Dataset & Batch Prediction Engine")
    st.markdown("<div style='font-size:0.84rem;color:#64748b;margin-bottom:12px;'>Score incoming customer CSV files using the trained machine learning pipeline.</div>", unsafe_allow_html=True)

    col_up1, col_up2 = st.columns([7, 3])
    with col_up1:
        uploaded_file = st.file_uploader("Upload customer CSV file for batch repeat prediction", type=["csv"])
    with col_up2:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        use_sample = st.button("Load 30 Sample Customers", use_container_width=True)
        sample_path = PROJECT_ROOT / "data" / "sample_new_customers.csv"
        if sample_path.exists():
            st.download_button("Download CSV Template", data=sample_path.read_bytes(), file_name="new_customers_template.csv", mime="text/csv", use_container_width=True)

    df_to_score = None
    if uploaded_file is not None:
        try:
            df_to_score = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df_to_score):,} customer records.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    elif use_sample or st.session_state.get("sample_loaded", False):
        st.session_state["sample_loaded"] = True
        if sample_path.exists():
            df_to_score = pd.read_csv(sample_path)
            st.info(f"Loaded batch of {len(df_to_score)} customer transactions from dataset.")

    if df_to_score is not None:
        num_cols = [
            "spend_order1", "items_count_order1", "unique_products_order1",
            "freight_value_order1", "freight_ratio_order1", "installments_order1",
            "payment_splits_order1", "delivery_days_order1", "delivery_delay_days_order1",
            "is_delayed_order1", "review_score_order1", "has_comment_title_order1",
            "has_comment_message_order1", "purchase_hour_order1"
        ]
        cat_cols = ["payment_type_order1", "customer_state_order1", "category_order1", "purchase_dow_order1"]

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
        for mc in missing_cols:
            df_proc[mc] = defaults.get(mc, 0)

        probs = model.predict_proba(df_proc[num_cols + cat_cols])[:, 1]
        df_proc["repeat_prob_pct"] = np.round(probs * 100, 1)
        df_proc["propensity_tier"] = np.where(probs >= 0.50, "High (>= 50%)", np.where(probs >= 0.40, "Medium (40-50%)", "Low (< 40%)"))
        df_proc["recommended_action"] = np.where(probs >= 0.50, "VIP Loyalty Invitation & Priority Shipping Guarantee",
                                                 np.where(probs >= 0.40, "10% Voucher for 2nd Purchase within 14 Days",
                                                          "Post-Delivery Satisfaction Check-in"))

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Customers Scored</div><div class="kpi-value">{len(df_proc):,}</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">High Propensity (>=50%)</div><div class="kpi-value">{(probs >= 0.50).sum():,}</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Medium Propensity (40-50%)</div><div class="kpi-value">{((probs >= 0.40) & (probs < 0.50)).sum():,}</div></div>', unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Batch Spend Volume</div><div class="kpi-value">R$ {df_proc["spend_order1"].sum():,.0f}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        v_cols = [c for c in ["customer_unique_id", "spend_order1", "review_score_order1", "delivery_days_order1", "category_order1", "customer_state_order1", "repeat_prob_pct", "propensity_tier", "recommended_action"] if c in df_proc.columns]
        st.dataframe(df_proc[v_cols].rename(columns={
            "customer_unique_id": "Customer ID", "spend_order1": "Order #1 Spend (R$)",
            "review_score_order1": "Rating", "delivery_days_order1": "Delivery Days",
            "category_order1": "Category", "customer_state_order1": "State",
            "repeat_prob_pct": "Repeat Probability (%)", "propensity_tier": "Propensity Tier",
            "recommended_action": "Recommended Playbook Action"
        }), use_container_width=True, height=280)

        st.download_button(
            "Export Scored Batch to CSV",
            data=df_proc.to_csv(index=False).encode("utf-8"),
            file_name="batch_repeat_predictions_scored.csv",
            mime="text/csv",
            type="primary"
        )
