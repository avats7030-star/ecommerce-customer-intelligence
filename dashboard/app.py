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
      padding: 10px 16px;
      margin-top: 0;
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.02);
  }
  .top-navbar-left {
      display: flex;
      flex-direction: column;
  }
  .top-navbar-title {
      font-size: 1.25rem;
      font-weight: 700;
      color: #0f172a;
      line-height: 1.2;
      letter-spacing: -0.015em;
  }
  .top-navbar-sub {
      font-size: 0.78rem;
      color: #64748b;
      font-weight: 500;
      margin-top: 2px;
  }
  .top-navbar-meta {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 6px;
  }
  .badge-tag {
      display: inline-flex;
      align-items: center;
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      color: #475569;
      font-size: 0.70rem;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 4px;
      letter-spacing: 0.01em;
      white-space: nowrap;
  }
  .badge-status {
      background: #f0fdf4;
      border-color: #86efac;
      color: #166534;
  }

  /* Compact Filter Toolbar & Form Controls */
  div[data-testid="stSelectbox"] {
      margin-bottom: 0px !important;
  }
  div[data-testid="stSelectbox"] label p {
      font-size: 0.70rem !important;
      font-weight: 600 !important;
      color: #475569 !important;
      text-transform: uppercase !important;
      letter-spacing: 0.04em !important;
      margin-bottom: 2px !important;
  }
  div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
      min-height: 32px !important;
      height: 32px !important;
      padding-top: 0px !important;
      padding-bottom: 0px !important;
      font-size: 0.80rem !important;
      border-radius: 4px !important;
      border-color: #cbd5e1 !important;
  }
  div[data-testid="stButton"] button {
      height: 32px !important;
      min-height: 32px !important;
      padding: 0 10px !important;
      font-size: 0.76rem !important;
      font-weight: 600 !important;
      border-radius: 4px !important;
      border: 1px solid #cbd5e1 !important;
      color: #334155 !important;
      background: #f8fafc !important;
      transition: all 0.15s ease !important;
  }
  div[data-testid="stButton"] button:hover {
      background: #e2e8f0 !important;
      border-color: #94a3b8 !important;
      color: #0f172a !important;
  }

  /* Professional Compact KPI Cards */
  .kpi-card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 8px 12px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.02);
      min-height: 72px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      margin-bottom: 4px;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
  }
  .kpi-card:hover {
      box-shadow: 0 2px 5px rgba(0,0,0,0.05);
  }
  .kpi-label {
      font-size: 0.65rem;
      font-weight: 700;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 1px;
      line-height: 1.1;
  }
  .kpi-value {
      font-size: 1.30rem;
      font-weight: 700;
      color: #0f172a;
      line-height: 1.15;
  }
  .kpi-sub {
      font-size: 0.68rem;
      color: #94a3b8;
      margin-top: 1px;
      line-height: 1.15;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
  }

  /* Compact Dashboard Section Container */
  .dash-card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 8px 12px 4px 12px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.02);
      margin-bottom: 8px;
  }
  .dash-card-title {
      font-size: 0.85rem;
      font-weight: 700;
      color: #1e293b;
      margin-bottom: 1px;
      line-height: 1.2;
  }
  .dash-card-sub {
      font-size: 0.72rem;
      color: #64748b;
      margin-bottom: 4px;
      line-height: 1.2;
  }

  /* Compact Insight Grid Cards */
  .insight-card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-left: 3px solid #0078d4;
      border-radius: 6px;
      padding: 10px 12px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.02);
      height: 100%;
  }
  .insight-tag {
      font-size: 0.66rem;
      font-weight: 700;
      color: #0078d4;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 3px;
  }
  .insight-main {
      font-size: 0.82rem;
      font-weight: 600;
      color: #0f172a;
      line-height: 1.30;
      margin-bottom: 3px;
  }
  .insight-action {
      font-size: 0.74rem;
      color: #475569;
      line-height: 1.30;
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

  /* Primary Button Styling */
  div[data-testid="stButton"] button[kind="primary"],
  div[data-testid="stButton"] button[data-testid="stBaseButton-primary"] {
      background: #0078d4 !important;
      border: 1px solid #005a9e !important;
      color: #ffffff !important;
      font-weight: 700 !important;
      font-size: 0.82rem !important;
      letter-spacing: 0.02em !important;
      box-shadow: 0 2px 4px rgba(0, 120, 212, 0.25) !important;
  }
  div[data-testid="stButton"] button[kind="primary"]:hover,
  div[data-testid="stButton"] button[data-testid="stBaseButton-primary"]:hover {
      background: #106ebe !important;
      border-color: #004578 !important;
      color: #ffffff !important;
      box-shadow: 0 4px 8px rgba(0, 120, 212, 0.35) !important;
  }

  /* Sliders: Replace harsh red with brand indigo/blue */
  div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
      background-color: #0078d4 !important;
      border: 2px solid #ffffff !important;
      box-shadow: 0 1px 3px rgba(0, 120, 212, 0.4) !important;
  }
  div[data-testid="stSlider"] [data-testid="stThumbValue"] {
      color: #0f172a !important;
      font-weight: 700 !important;
      font-size: 0.74rem !important;
  }
  div[data-testid="stSlider"] label p {
      font-size: 0.70rem !important;
      font-weight: 600 !important;
      color: #475569 !important;
      text-transform: uppercase !important;
      letter-spacing: 0.04em !important;
      margin-bottom: 2px !important;
  }
  div[data-testid="stNumberInput"] label p {
      font-size: 0.70rem !important;
      font-weight: 600 !important;
      color: #475569 !important;
      text-transform: uppercase !important;
      letter-spacing: 0.04em !important;
      margin-bottom: 2px !important;
  }
  div[data-testid="stNumberInput"] div[data-baseweb="input"] {
      min-height: 32px !important;
      height: 32px !important;
      border-radius: 4px !important;
      border-color: #cbd5e1 !important;
      background: #ffffff !important;
  }
  div[data-testid="stNumberInput"] input {
      font-size: 0.82rem !important;
      color: #0f172a !important;
      padding: 2px 8px !important;
  }

  /* Modern Tabs */
  div[data-testid="stTabs"] button[role="tab"] {
      font-size: 0.78rem !important;
      font-weight: 600 !important;
      color: #64748b !important;
      padding: 6px 12px !important;
  }
  div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
      color: #0078d4 !important;
      border-bottom-color: #0078d4 !important;
  }

  /* Breadcrumb & Section Navigation */
  .breadcrumb-bar {
      margin-bottom: 8px;
  }
  .breadcrumb-path {
      font-size: 0.70rem;
      font-weight: 600;
      color: #64748b;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      margin-bottom: 2px;
      display: inline-block;
  }
  .page-main-heading {
      font-size: 1.25rem;
      font-weight: 700;
      color: #0f172a;
      letter-spacing: -0.015em;
      margin: 0 0 2px 0;
      line-height: 1.25;
  }
  .page-main-sub {
      font-size: 0.80rem;
      color: #64748b;
      margin-bottom: 10px;
      line-height: 1.35;
  }

  /* Compact Customer Journey Flow */
  .journey-flow-card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 8px 14px;
      margin-bottom: 12px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.02);
  }
  .journey-header-label {
      font-size: 0.65rem;
      font-weight: 700;
      color: #64748b;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin-bottom: 6px;
  }
  .journey-steps-container {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
  }
  .journey-step {
      display: flex;
      align-items: center;
      gap: 6px;
  }
  .journey-circle {
      width: 20px;
      height: 20px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.68rem;
      font-weight: 700;
  }
  .journey-step.done .journey-circle {
      background: #e0f2fe;
      color: #0284c7;
  }
  .journey-step.current .journey-circle {
      background: #0078d4;
      color: #ffffff;
  }
  .journey-step.next .journey-circle {
      background: #f1f5f9;
      color: #64748b;
  }
  .j-title {
      font-size: 0.70rem;
      font-weight: 700;
      color: #0f172a;
      line-height: 1.1;
  }
  .j-sub {
      font-size: 0.62rem;
      color: #64748b;
      line-height: 1.1;
  }
  .journey-arrow {
      color: #94a3b8;
      font-size: 0.80rem;
      font-weight: 600;
  }

  /* Simulated Customer Scenario Card */
  .scenario-profile-card {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-left: 3px solid #0078d4;
      border-radius: 6px;
      padding: 8px 12px;
      margin-bottom: 10px;
  }
  .scenario-profile-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
  }
  .scenario-tag {
      font-size: 0.68rem;
      font-weight: 700;
      color: #0078d4;
      letter-spacing: 0.05em;
      text-transform: uppercase;
  }
  .scenario-order-badge {
      font-size: 0.64rem;
      font-weight: 600;
      color: #475569;
      background: #ffffff;
      border: 1px solid #cbd5e1;
      padding: 1px 6px;
      border-radius: 4px;
  }
  .scenario-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 4px 14px;
  }
  .scenario-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.74rem;
      border-bottom: 1px dashed #e2e8f0;
      padding-bottom: 2px;
  }
  .scenario-k {
      color: #64748b;
      font-weight: 500;
  }
  .scenario-v {
      color: #0f172a;
      font-weight: 700;
  }

  /* Form Group Headers */
  .input-group-header {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 0.70rem;
      font-weight: 700;
      color: #334155;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-top: 10px;
      margin-bottom: 6px;
      padding-bottom: 3px;
      border-bottom: 1px solid #e2e8f0;
  }

  /* Prediction Hero Card */
  .pred-hero-card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 12px 14px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.03);
      margin-bottom: 10px;
  }
  .pred-status-banner {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 0.74rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      padding: 5px 10px;
      border-radius: 4px;
      margin-bottom: 6px;
  }
  .pred-status-bullet {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      display: inline-block;
  }
  .pred-metrics-strip {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 6px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 8px 10px;
      margin-top: 4px;
  }
  .pred-strip-col {
      display: flex;
      flex-direction: column;
      text-align: center;
  }
  .strip-k {
      font-size: 0.64rem;
      font-weight: 600;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.04em;
  }
  .strip-v {
      font-size: 1.10rem;
      font-weight: 800;
      color: #0f172a;
      line-height: 1.15;
      margin-top: 1px;
  }
  .strip-sub {
      font-size: 0.62rem;
      color: #94a3b8;
  }

  /* Recommended Action Card */
  .rec-action-card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-left: 4px solid #0078d4;
      border-radius: 6px;
      padding: 12px 14px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.02);
      margin-bottom: 10px;
  }
  .rec-action-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 4px;
  }
  .rec-action-badge {
      font-size: 0.66rem;
      font-weight: 700;
      color: #0078d4;
      letter-spacing: 0.06em;
      text-transform: uppercase;
  }
  .rec-urgency-pill {
      font-size: 0.64rem;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 10px;
      background: #f1f5f9;
      color: #475569;
      letter-spacing: 0.02em;
  }
  .rec-strategy-title {
      font-size: 1.05rem;
      font-weight: 700;
      color: #0f172a;
      margin-bottom: 8px;
      line-height: 1.25;
  }
  .rec-section-block {
      margin-bottom: 8px;
  }
  .rec-section-label {
      font-size: 0.65rem;
      font-weight: 700;
      color: #64748b;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      margin-bottom: 2px;
  }
  .rec-section-content {
      font-size: 0.80rem;
      color: #334155;
      line-height: 1.45;
  }
  .rec-footer-grid {
      display: flex;
      gap: 16px;
      padding-top: 6px;
      border-top: 1px solid #f1f5f9;
      margin-top: 6px;
  }
  .rec-footer-item {
      display: flex;
      flex-direction: column;
  }
  .rec-footer-k {
      font-size: 0.62rem;
      font-weight: 600;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.04em;
  }
  .rec-footer-v {
      font-size: 0.80rem;
      font-weight: 700;
      color: #0f172a;
      margin-top: 1px;
  }

  /* Divider */
  .fancy-divider {
      height: 1px;
      background: #e2e8f0;
      margin: 16px 0;
  }

  /* Streamlit Page Content Layout */
  .block-container {
      padding-top: 4.25rem !important;
      padding-bottom: 2rem !important;
      padding-left: 1.75rem !important;
      padding-right: 1.75rem !important;
      max-width: 100% !important;
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
def apply_plotly_style(fig, height=270):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=11, color="#334155"),
        title_font=dict(size=12, color="#0f172a"),
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
st.markdown("""
<div class="top-navbar">
    <div class="top-navbar-left">
        <div style="display: flex; align-items: center; gap: 8px;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect x="2" y="3" width="20" height="18" rx="3" stroke="#0078d4" stroke-width="2"/>
                <path d="M7 16V13M12 16V9M17 16V11" stroke="#0078d4" stroke-width="2" stroke-linecap="round"/>
            </svg>
            <div class="top-navbar-title">E-Commerce Customer Intelligence</div>
        </div>
        <div class="top-navbar-sub">Sales • Customer Behavior • Predictive Analytics</div>
    </div>
    <div class="top-navbar-meta">
        <span class="badge-tag">Dataset: Olist</span>
        <span class="badge-tag">Model: Logistic Regression</span>
        <span class="badge-tag badge-status">Status: Live</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Executive Dashboard":
    # ── Interactive Top Filter Bar ────────────────────────────────────────────
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
        <span style="font-size: 0.70rem; font-weight: 700; color: #64748b; letter-spacing: 0.06em; text-transform: uppercase;">Filters</span>
    </div>
    """, unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns([32, 32, 26, 10])
    with f1:
        year_filter = st.selectbox("Order Timeframe", ["All Periods (2016–2018)", "2017", "2018"], key="sb_year")
    with f2:
        state_filter = st.selectbox("Geographic Filter", ["All States (National)", "SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "GO", "ES"], key="sb_state")
    with f3:
        pay_filter = st.selectbox("Payment Type", ["All Methods", "Credit Card", "Boleto", "Voucher", "Debit Card"], key="sb_pay")
    with f4:
        st.markdown('<div style="height: 18px;"></div>', unsafe_allow_html=True)
        if st.button("Reset", key="btn_reset_filters", use_container_width=True):
            st.session_state["sb_year"] = "All Periods (2016–2018)"
            st.session_state["sb_state"] = "All States (National)"
            st.session_state["sb_pay"] = "All Methods"
            st.rerun()

    # Dynamic Filter Evaluation (Instant precomputed when default, fast sliced in-memory otherwise)
    is_filtered = (
        year_filter != "All Periods (2016–2018)" or
        state_filter != "All States (National)" or
        pay_filter != "All Methods"
    )

    if not is_filtered:
        pre = get_executive_precomputed()
        tot_rev, tot_orders, deliv_orders, uniq_cust, aov, repeat_rate, avg_review = pre["kpis"]
        m_df = pre["monthly"]
        pay_df = pre["pay_dist"]
        cat_df = pre["cat_rev"]
        state_df = pre["state_rev"]
    else:
        orders = get_orders_df()
        cust = get_cust_df()
        f_orders = orders

        if year_filter == "2017":
            f_orders = f_orders[f_orders["order_purchase_timestamp"].str.startswith("2017", na=False)]
        elif year_filter == "2018":
            f_orders = f_orders[f_orders["order_purchase_timestamp"].str.startswith("2018", na=False)]

        if state_filter != "All States (National)":
            f_orders = f_orders[f_orders["customer_state"] == state_filter]

        if pay_filter != "All Methods":
            pay_map = {
                "Credit Card": "credit_card",
                "Boleto": "boleto",
                "Voucher": "voucher",
                "Debit Card": "debit_card"
            }
            target_pay = pay_map.get(pay_filter, pay_filter.lower())
            f_orders = f_orders[f_orders["primary_payment_type"] == target_pay]

        f_deliv = f_orders[f_orders["is_delivered"] == 1]
        tot_rev = float(f_deliv["total_payment_value"].sum()) if len(f_deliv) > 0 else 0.0
        tot_orders = int(len(f_orders))
        deliv_orders = int(len(f_deliv))
        uniq_cust = int(f_orders["customer_unique_id"].nunique()) if len(f_orders) > 0 else 0
        aov = float(f_deliv["total_payment_value"].mean()) if len(f_deliv) > 0 else 0.0
        repeat_rate = float((cust["frequency"] > 1).mean() * 100)

        # Monthly trend
        if len(f_deliv) > 0:
            m_df = f_deliv.groupby("order_year_month").agg(
                revenue=("total_payment_value", "sum"),
                orders=("order_id", "count")
            ).reset_index()
            m_df = m_df[(m_df["order_year_month"] >= "2017-01") & (m_df["order_year_month"] <= "2018-08")]
        else:
            m_df = pd.DataFrame(columns=["order_year_month", "revenue", "orders"])

        # Payment distribution
        if len(f_orders) > 0:
            pay_dist_series = f_orders["primary_payment_type"].value_counts().head(4)
            pay_df = pd.DataFrame({"payment_type": pay_dist_series.index, "count": pay_dist_series.values})
        else:
            pay_df = pd.DataFrame(columns=["payment_type", "count"])

        # Category revenue
        if len(f_orders) > 0:
            cat_series = f_orders.groupby("primary_category")["total_payment_value"].sum().sort_values(ascending=False).head(10)
            cat_df = pd.DataFrame({"category": cat_series.index, "revenue": cat_series.values})
        else:
            cat_df = pd.DataFrame(columns=["category", "revenue"])

        # State revenue
        if len(f_orders) > 0:
            state_series = f_orders.groupby("customer_state")["total_payment_value"].sum().sort_values(ascending=False).head(10)
            state_df = pd.DataFrame({"state": state_series.index, "revenue": state_series.values})
        else:
            state_df = pd.DataFrame(columns=["state", "revenue"])

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
        deliv_pct = (deliv_orders / tot_orders * 100) if tot_orders > 0 else 0.0
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Orders</div>
            <div class="kpi-value">{tot_orders:,}</div>
            <div class="kpi-sub">{deliv_orders:,} delivered ({deliv_pct:.1f}%)</div>
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

    # ── Interactive Plotly Chart Grid ─────────────────────────────────────────
    # ROW 1: Monthly Trend & Payment Share
    r1_col1, r1_col2 = st.columns([65, 35])
    with r1_col1:
        st.markdown('<div class="dash-card"><div class="dash-card-title">Monthly Revenue & Order Volume Trend</div><div class="dash-card-sub">Commercial performance across 2017–2018 order milestones</div>', unsafe_allow_html=True)
        fig_trend = go.Figure()
        if len(m_df) > 0:
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
        apply_plotly_style(fig_trend, height=270)
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r1_col2:
        st.markdown('<div class="dash-card"><div class="dash-card-title">Payment Method Share</div><div class="dash-card-sub">Breakdown of gross transaction volume by payment type</div>', unsafe_allow_html=True)
        if len(pay_df) > 0:
            fig_pay = px.pie(
                pay_df, values="count", names="payment_type", hole=0.6,
                color_discrete_sequence=["#0078d4", "#008272", "#f97316", "#7c3aed"]
            )
            fig_pay.update_traces(textposition='inside', textinfo='percent+label', textfont_size=10)
            fig_pay.update_layout(showlegend=False)
        else:
            fig_pay = go.Figure()
        apply_plotly_style(fig_pay, height=270)
        st.plotly_chart(fig_pay, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ROW 2: Category Revenue & State Revenue
    r2_col1, r2_col2 = st.columns([50, 50])
    with r2_col1:
        st.markdown('<div class="dash-card"><div class="dash-card-title">Revenue by Product Category</div><div class="dash-card-sub">Top 10 highest grossing categories in Million BRL</div>', unsafe_allow_html=True)
        if len(cat_df) > 0:
            fig_cat = px.bar(
                cat_df, x="revenue", y="category", orientation="h",
                labels={"revenue": "Revenue (BRL)", "category": ""},
                color_discrete_sequence=["#0078d4"]
            )
            fig_cat.update_layout(yaxis=dict(autorange="reversed"))
        else:
            fig_cat = go.Figure()
        apply_plotly_style(fig_cat, height=270)
        st.plotly_chart(fig_cat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2_col2:
        st.markdown('<div class="dash-card"><div class="dash-card-title">Revenue by Customer State</div><div class="dash-card-sub">Top 10 regional distribution of commercial revenue</div>', unsafe_allow_html=True)
        if len(state_df) > 0:
            fig_state = px.bar(
                state_df, x="revenue", y="state", orientation="h",
                labels={"revenue": "Revenue (BRL)", "state": ""},
                color_discrete_sequence=["#008272"]
            )
            fig_state.update_layout(yaxis=dict(autorange="reversed"))
        else:
            fig_state = go.Figure()
        apply_plotly_style(fig_state, height=270)
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

    # ── Header & Context Breadcrumb ───────────────────────────────────────────
    st.markdown("""
    <div class="breadcrumb-bar">
        <span class="breadcrumb-path">Customer Intelligence &nbsp;/&nbsp; Prediction Simulator</span>
        <h2 class="page-main-heading">Customer Repeat Purchase</h2>
        <div class="page-main-sub">Estimate the likelihood that a customer will purchase again using calibrated machine learning inference on Order #1 signals.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Customer Journey Visual Flow ──────────────────────────────────────────
    st.markdown("""
    <div class="journey-flow-card">
        <div class="journey-header-label">Customer Intelligence Journey</div>
        <div class="journey-steps-container">
            <div class="journey-step done">
                <div class="journey-circle">1</div>
                <div class="journey-text">
                    <span class="j-title">INITIAL ORDER</span>
                    <span class="j-sub">Order #1 Captured</span>
                </div>
            </div>
            <div class="journey-arrow">→</div>
            <div class="journey-step done">
                <div class="journey-circle">2</div>
                <div class="journey-text">
                    <span class="j-title">EXPERIENCE</span>
                    <span class="j-sub">Fulfillment & SLA</span>
                </div>
            </div>
            <div class="journey-arrow">→</div>
            <div class="journey-step current">
                <div class="journey-circle">3</div>
                <div class="journey-text">
                    <span class="j-title">AI INFERENCE</span>
                    <span class="j-sub">Propensity Scoring</span>
                </div>
            </div>
            <div class="journey-arrow">→</div>
            <div class="journey-step next">
                <div class="journey-circle">4</div>
                <div class="journey-text">
                    <span class="j-title">CRM ACTION</span>
                    <span class="j-sub">Targeted Playbook</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Category and Payment Mapping Dictionaries ─────────────────────────────
    category_options = {
        "Bed Bath Table": "bed_bath_table",
        "Furniture Decor": "furniture_decor",
        "Sports Leisure": "sports_leisure",
        "Computers Accessories": "computers_accessories",
        "Health Beauty": "health_beauty",
        "Watches Gifts": "watches_gifts",
        "Housewares": "housewares",
        "Auto": "auto",
        "Toys": "toys",
        "Garden Tools": "garden_tools",
        "Telephony": "telephony",
        "Cool Stuff": "cool_stuff",
        "Other Category": "other_category"
    }
    payment_options = {
        "Credit Card": "credit_card",
        "Boleto": "boleto",
        "Voucher": "voucher",
        "Debit Card": "debit_card"
    }

    # Initialize Simulator State if not present
    defaults = {
        "sim_spend": 180.0,
        "sim_freight": 25.0,
        "sim_items": 1,
        "sim_installments": 3,
        "sim_category_label": "Bed Bath Table",
        "sim_state": "SP",
        "sim_payment_label": "Credit Card",
        "sim_review": 5,
        "sim_delivery": 10.0,
        "sim_is_delayed": 0
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Two-Sided Experience Layout ───────────────────────────────────────────
    sim_col1, sim_col2 = st.columns([46, 54], gap="medium")

    with sim_col1:
        st.markdown('<div class="dash-card"><div class="dash-card-title">Customer Profile & Inputs</div><div class="dash-card-sub">Configure Order #1 transaction features to evaluate repeat propensity</div>', unsafe_allow_html=True)

        # ── Visual Scenario Summary Card ──────────────────────────────────────
        st.markdown(f"""
        <div class="scenario-profile-card">
            <div class="scenario-profile-head">
                <span class="scenario-tag">Simulated Customer Scenario</span>
                <span class="scenario-order-badge">Order #1 Profile</span>
            </div>
            <div class="scenario-grid">
                <div class="scenario-item">
                    <span class="scenario-k">Category</span>
                    <span class="scenario-v">{st.session_state.get('sim_category_label', 'Bed Bath Table')}</span>
                </div>
                <div class="scenario-item">
                    <span class="scenario-k">State</span>
                    <span class="scenario-v">{st.session_state.get('sim_state', 'SP')} (Brazil)</span>
                </div>
                <div class="scenario-item">
                    <span class="scenario-k">Payment</span>
                    <span class="scenario-v">{st.session_state.get('sim_payment_label', 'Credit Card')}</span>
                </div>
                <div class="scenario-item">
                    <span class="scenario-k">Order Value</span>
                    <span class="scenario-v">R$ {float(st.session_state.get('sim_spend', 180.0)):,.2f}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Group 1: Purchase Details ─────────────────────────────────────────
        st.markdown('<div class="input-group-header">Purchase Details</div>', unsafe_allow_html=True)
        g1_col1, g1_col2 = st.columns(2)
        with g1_col1:
            spend = st.number_input(
                "Spend Amount (BRL)",
                min_value=10.0, max_value=5000.0,
                value=float(st.session_state["sim_spend"]),
                step=10.0,
                help="Amount spent on the initial order.",
                key="input_spend"
            )
            items_cnt = st.number_input(
                "Order Items Count",
                min_value=1, max_value=20,
                value=int(st.session_state["sim_items"]),
                step=1,
                help="Total items included in Order #1.",
                key="input_items"
            )
        with g1_col2:
            freight = st.number_input(
                "Freight Fee (BRL)",
                min_value=0.0, max_value=400.0,
                value=float(st.session_state["sim_freight"]),
                step=5.0,
                help="Shipping cost associated with the order.",
                key="input_freight"
            )
            installments = st.slider(
                "Installments",
                min_value=1, max_value=24,
                value=int(st.session_state["sim_installments"]),
                help="Number of credit payment installments.",
                key="input_installments"
            )

        # ── Group 2: Product & Customer ───────────────────────────────────────
        st.markdown('<div class="input-group-header">Product & Customer Profile</div>', unsafe_allow_html=True)
        g2_col1, g2_col2 = st.columns(2)
        with g2_col1:
            cat_list = list(category_options.keys())
            cur_cat_idx = cat_list.index(st.session_state["sim_category_label"]) if st.session_state["sim_category_label"] in cat_list else 0
            category_label = st.selectbox(
                "Product Category",
                cat_list,
                index=cur_cat_idx,
                help="Commercial category of the primary product purchased.",
                key="input_cat"
            )
            pay_list = list(payment_options.keys())
            cur_pay_idx = pay_list.index(st.session_state["sim_payment_label"]) if st.session_state["sim_payment_label"] in pay_list else 0
            payment_label = st.selectbox(
                "Payment Method",
                pay_list,
                index=cur_pay_idx,
                help="Financial instrument used for Order #1.",
                key="input_pay"
            )
        with g2_col2:
            state_list = ["SP", "RJ", "MG", "RS", "PR", "SC", "Other_State"]
            cur_state_idx = state_list.index(st.session_state["sim_state"]) if st.session_state["sim_state"] in state_list else 0
            customer_state = st.selectbox(
                "Customer State",
                state_list,
                index=cur_state_idx,
                help="Geographic destination state of the customer.",
                key="input_state"
            )

        # ── Group 3: Experience & Fulfillment ─────────────────────────────────
        st.markdown('<div class="input-group-header">Experience & Fulfillment</div>', unsafe_allow_html=True)
        g3_col1, g3_col2 = st.columns(2)
        with g3_col1:
            review_score = st.slider(
                "Review Rating (1–5)",
                min_value=1, max_value=5,
                value=int(st.session_state["sim_review"]),
                help="Customer's post-purchase review score.",
                key="input_review"
            )
            delivery_days = st.slider(
                "Delivery Duration (Days)",
                min_value=1.0, max_value=60.0,
                value=float(st.session_state["sim_delivery"]),
                step=1.0,
                help="Number of days taken to deliver the order.",
                key="input_deliv_days"
            )
        with g3_col2:
            sla_options = [0, 1]
            cur_sla_idx = 1 if st.session_state["sim_is_delayed"] == 1 else 0
            is_delayed = st.selectbox(
                "Delivered Past SLA?",
                sla_options,
                index=cur_sla_idx,
                format_func=lambda x: "Yes — Late Delivery" if x == 1 else "No — On Time (Within SLA)",
                help="Whether order was delivered after the promised delivery deadline.",
                key="input_sla"
            )

        # Update session state values
        st.session_state["sim_spend"] = spend
        st.session_state["sim_freight"] = freight
        st.session_state["sim_items"] = items_cnt
        st.session_state["sim_installments"] = installments
        st.session_state["sim_category_label"] = category_label
        st.session_state["sim_state"] = customer_state
        st.session_state["sim_payment_label"] = payment_label
        st.session_state["sim_review"] = review_score
        st.session_state["sim_delivery"] = delivery_days
        st.session_state["sim_is_delayed"] = is_delayed

        # ── Action Buttons ────────────────────────────────────────────────────
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        btn_c1, btn_c2 = st.columns([7, 3])
        with btn_c1:
            predict_clicked = st.button("Predict Repeat Purchase", type="primary", use_container_width=True, key="btn_run_pred")
        with btn_c2:
            reset_clicked = st.button("Reset Inputs", use_container_width=True, key="btn_reset_sim")
            if reset_clicked:
                for k, v in defaults.items():
                    st.session_state[k] = v
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with sim_col2:
        st.markdown('<div class="dash-card"><div class="dash-card-title">Prediction & Business Intelligence</div><div class="dash-card-sub">Calibrated machine learning inference & automated CRM action playbooks</div>', unsafe_allow_html=True)

        # Prepare input dataframe for model
        cat_key = category_options.get(st.session_state["sim_category_label"], "bed_bath_table")
        pay_key = payment_options.get(st.session_state["sim_payment_label"], "credit_card")
        delay_days = 4.0 if st.session_state["sim_is_delayed"] == 1 else 0.0

        input_row = pd.DataFrame([{
            "spend_order1": float(st.session_state["sim_spend"]),
            "items_count_order1": int(st.session_state["sim_items"]),
            "unique_products_order1": 1,
            "freight_value_order1": float(st.session_state["sim_freight"]),
            "freight_ratio_order1": float(st.session_state["sim_freight"]) / max(float(st.session_state["sim_spend"]), 1.0),
            "installments_order1": int(st.session_state["sim_installments"]),
            "payment_splits_order1": 1,
            "delivery_days_order1": float(st.session_state["sim_delivery"]),
            "delivery_delay_days_order1": float(delay_days),
            "is_delayed_order1": int(st.session_state["sim_is_delayed"]),
            "review_score_order1": float(st.session_state["sim_review"]),
            "has_comment_title_order1": 1 if st.session_state["sim_review"] in (1, 5) else 0,
            "has_comment_message_order1": 1 if st.session_state["sim_review"] in (1, 5) else 0,
            "purchase_hour_order1": 14,
            "payment_type_order1": pay_key,
            "customer_state_order1": st.session_state["sim_state"],
            "category_order1": cat_key,
            "purchase_dow_order1": "Monday"
        }])

        # Predict probability using champion model
        prob = model.predict_proba(input_row)[0][1]
        prob_pct = prob * 100
        lift = prob_pct - 3.0

        # Determine visual state based on prediction
        if prob >= 0.50:
            status_text = "HIGH RETURN PROPENSITY"
            status_bg = "#f0fdf4"
            status_border = "#bbf7d0"
            status_color = "#15803d"
            card_border = "#86efac"
            gauge_color = "#16a34a"
        elif prob >= 0.40:
            status_text = "MEDIUM RETURN PROPENSITY"
            status_bg = "#fffbeb"
            status_border = "#fde68a"
            status_color = "#b45309"
            card_border = "#fcd34d"
            gauge_color = "#d97706"
        else:
            status_text = "LOW RETURN PROPENSITY (CHURN RISK)"
            status_bg = "#fef2f2"
            status_border = "#fecaca"
            status_color = "#b91c1c"
            card_border = "#fca5a5"
            gauge_color = "#dc2626"

        # ── Prediction Hero Card ──────────────────────────────────────────────
        st.markdown(f"""
        <div class="pred-hero-card" style="border-color: {card_border};">
            <div class="pred-status-banner" style="background: {status_bg}; border: 1px solid {status_border}; color: {status_color};">
                <span class="pred-status-bullet" style="background: {status_color};"></span>
                <span>{status_text}</span>
            </div>
        """, unsafe_allow_html=True)

        # Plotly Gauge Indicator
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={
                'suffix': "%",
                'font': {'size': 36, 'family': 'Inter, -apple-system, sans-serif', 'color': '#0f172a', 'weight': 800}
            },
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1", 'ticks': "", 'tickfont': {'size': 9, 'color': '#94a3b8'}},
                'bar': {'color': gauge_color, 'thickness': 0.32},
                'bgcolor': "#f1f5f9",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 40], 'color': "rgba(220, 38, 38, 0.08)"},
                    {'range': [40, 50], 'color': "rgba(217, 119, 6, 0.08)"},
                    {'range': [50, 100], 'color': "rgba(22, 163, 74, 0.08)"}
                ],
                'threshold': {
                    'line': {'color': gauge_color, 'width': 3},
                    'thickness': 0.8,
                    'value': prob_pct
                }
            }
        ))
        fig_gauge.update_layout(
            height=165,
            margin=dict(l=15, r=15, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, -apple-system, Segoe UI, sans-serif")
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Benchmark Metrics Strip
        st.markdown(f"""
            <div class="pred-metrics-strip">
                <div class="pred-strip-col">
                    <span class="strip-k">Baseline Repeat Rate</span>
                    <span class="strip-v">3.00%</span>
                    <span class="strip-sub">Marketplace Avg</span>
                </div>
                <div class="pred-strip-col">
                    <span class="strip-k">Predicted Probability</span>
                    <span class="strip-v" style="color: {status_color};">{prob_pct:.1f}%</span>
                    <span class="strip-sub">Calibrated Output</span>
                </div>
                <div class="pred-strip-col">
                    <span class="strip-k">Probability Lift</span>
                    <span class="strip-v">{'+' if lift >= 0 else ''}{lift:.1f} pp</span>
                    <span class="strip-sub">Above Baseline</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Tabs: Strategic Action, Model Drivers, Audit Vector ──────────────
        t_strat, t_drivers, t_vector = st.tabs([
            "Strategic Recommendation",
            "Why This Result? (Model Drivers)",
            "Model Feature Vector"
        ])

        with t_strat:
            # Business Recommendation from RecommendationEngine
            rec = RecommendationEngine().get_individual_action(
                prob,
                "Potential Loyalists" if float(st.session_state["sim_spend"]) >= 200 else "Standard",
                is_delayed=st.session_state["sim_is_delayed"]
            )

            # Timing mapping
            if "recovery" in rec['playbook'].lower() or st.session_state["sim_is_delayed"] == 1:
                timing = "Immediate (within 48 hours)"
                channel = "Direct Email & CS Ticket"
            elif prob >= 0.50:
                timing = "Within 30 days"
                channel = "Targeted Cross-sell Email / Push"
            else:
                timing = "Within 14 days"
                channel = "Voucher Incentive & Retargeting"

            st.markdown(f"""
            <div class="rec-action-card">
                <div class="rec-action-header">
                    <span class="rec-action-badge">RECOMMENDED ACTION</span>
                    <span class="rec-urgency-pill">{rec['urgency']}</span>
                </div>
                <div class="rec-strategy-title">{rec['playbook']}</div>

                <div class="rec-section-block">
                    <div class="rec-section-label">WHY THIS STRATEGY</div>
                    <div class="rec-section-content">{rec['diagnosis']}</div>
                </div>

                <div class="rec-section-block">
                    <div class="rec-section-label">SPECIFIC CRM ACTION</div>
                    <div class="rec-section-content">{rec['action']}</div>
                </div>

                <div class="rec-footer-grid">
                    <div class="rec-footer-item">
                        <span class="rec-footer-k">RECOMMENDED TIMING</span>
                        <span class="rec-footer-v">{timing}</span>
                    </div>
                    <div class="rec-footer-item">
                        <span class="rec-footer-k">TARGET CHANNEL</span>
                        <span class="rec-footer-v">{channel}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with t_drivers:
            st.markdown("""
            <div style="font-size:0.78rem;color:#475569;margin-bottom:8px;line-height:1.4;">
                <strong>Global Feature Importance</strong> from champion Logistic Regression coefficients.
                Higher spend, prompt delivery, and favorable reviews positively drive repeat propensity, while fulfillment delay and disproportionate shipping cost depress retention.
            </div>
            """, unsafe_allow_html=True)
            top_fi = feat_imp.head(8).copy()
            top_fi["clean_feature"] = (
                top_fi["feature"]
                .str.replace("category_order1_", "Category: ")
                .str.replace("customer_state_order1_", "State: ")
                .str.replace("_", " ")
                .str.title()
            )
            fig_fi = px.bar(
                top_fi.sort_values("abs_impact", ascending=True),
                x="abs_impact", y="clean_feature", orientation="h",
                labels={"abs_impact": "Model Impact Magnitude", "clean_feature": ""},
                color_discrete_sequence=["#0078d4"]
            )
            apply_plotly_style(fig_fi, height=220)
            st.plotly_chart(fig_fi, use_container_width=True)

        with t_vector:
            st.markdown("""
            <div style="font-size:0.76rem;color:#64748b;margin-bottom:6px;">
                Verified 18-feature inference vector evaluated by the Scikit-learn Pipeline (14 numerical + 4 categorical):
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(
                input_row.T.reset_index().rename(columns={"index": "Feature Name", 0: "Feature Value"}),
                use_container_width=True,
                height=220
            )

        st.markdown('</div>', unsafe_allow_html=True)


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
                &bull; <strong>Macro Intelligence</strong>: Real-time visibility into revenue seasonality, fulfillment SLAs, and spatial demand.<br>
                &bull; <strong>Behavioral Segmentation</strong>: RFM scoring across 93k customers to protect high-value GMV.<br>
                &bull; <strong>Early Repeat Purchase Prediction</strong>: Scoring incoming buyers on Order #1 to guide CRM intervention.
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
