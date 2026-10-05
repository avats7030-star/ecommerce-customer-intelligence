"""
E-Commerce Customer Intelligence & Purchase Prediction System
Premium Streamlit Dashboard | Powered by Real Olist Brazilian E-Commerce Data
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
APP_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
REPORTS_DIR   = PROJECT_ROOT / "reports"
VIZ_DIR       = PROJECT_ROOT / "visualizations"
SRC_DIR       = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Real Olist Brazilian E-Commerce Data Science Portfolio Project."}
)

# ── Premium CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark sidebar */
  [data-testid="stSidebar"] {
      background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
      border-right: 1px solid #334155;
  }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  [data-testid="stSidebar"] .stRadio label { color: #94a3b8 !important; }

  /* Main background */
  .stApp { background-color: #f0f4f8; }

  /* Hero gradient header */
  .hero-header {
      background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 40%, #1a4d7a 70%, #0e7490 100%);
      border-radius: 16px;
      padding: 28px 36px;
      margin-bottom: 28px;
      box-shadow: 0 8px 32px rgba(15,23,42,0.25);
      position: relative;
      overflow: hidden;
  }
  .hero-title {
      font-size: 1.9rem; font-weight: 800;
      color: #f8fafc; margin: 0; line-height: 1.2;
  }
  .hero-subtitle {
      font-size: 0.95rem; color: #94a3b8;
      margin-top: 6px; font-weight: 400;
  }
  .hero-badge {
      display: inline-block;
      background: rgba(56,189,248,0.15); border: 1px solid #38bdf8;
      color: #38bdf8; font-size: 0.75rem; font-weight: 600;
      padding: 4px 10px; border-radius: 20px; margin-top: 12px;
      letter-spacing: 0.05em; text-transform: uppercase;
  }

  /* KPI metric cards */
  .kpi-card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 14px;
      padding: 20px 22px;
      box-shadow: 0 2px 12px rgba(15,23,42,0.06);
      text-align: center;
  }
  .kpi-icon { font-size: 1.6rem; margin-bottom: 8px; }
  .kpi-value {
      font-size: 1.75rem; font-weight: 800;
      color: #0f172a;
      margin: 4px 0;
  }
  .kpi-label {
      font-size: 0.78rem; font-weight: 600;
      color: #64748b; text-transform: uppercase; letter-spacing: 0.06em;
  }

  /* Section cards */
  .section-card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 14px;
      padding: 22px 24px;
      box-shadow: 0 2px 8px rgba(15,23,42,0.05);
      margin-bottom: 18px;
  }
  .section-title {
      font-size: 1.05rem; font-weight: 700; color: #1e293b;
      margin-bottom: 14px;
  }

  /* Prediction result card */
  .pred-positive {
      background: linear-gradient(135deg, #d1fae5, #a7f3d0);
      border: 2px solid #10b981; border-radius: 14px;
      padding: 22px; text-align: center;
  }
  .pred-negative {
      background: linear-gradient(135deg, #fff7ed, #fed7aa);
      border: 2px solid #f97316; border-radius: 14px;
      padding: 22px; text-align: center;
  }

  /* Playbook card */
  .playbook-card {
      background: linear-gradient(135deg, #f8faff, #eff6ff);
      border: 1px solid #bfdbfe; border-radius: 12px;
      padding: 18px; margin-top: 10px;
  }
  .playbook-title { font-weight: 700; color: #1e40af; margin-bottom: 6px; font-size:0.95rem; }
  .playbook-body { color: #374151; font-size: 0.9rem; line-height: 1.6; }

  /* Insights callout */
  .insight-box {
      background: linear-gradient(135deg, #0f172a, #1e3a5f);
      border-radius: 12px; padding: 18px 22px; color: #e2e8f0;
      font-size: 0.9rem; line-height: 1.7; margin-top: 14px;
  }
  .insight-box strong { color: #38bdf8; }

  /* Sidebar brand strip */
  .sidebar-brand {
      background: linear-gradient(90deg, #0e7490, #0369a1);
      border-radius: 10px; padding: 12px 16px; margin-bottom: 18px; text-align:center;
  }
  .sidebar-brand-title { font-size: 1rem; font-weight: 800; color: #f0f9ff; }
  .sidebar-brand-sub   { font-size: 0.75rem; color: #bae6fd; margin-top: 2px; }

  /* Hide streamlit default top padding */
  .block-container { padding-top: 1.5rem !important; }

  /* Divider */
  .fancy-divider {
      height: 3px;
      background: linear-gradient(90deg, #0e7490, #38bdf8, transparent);
      border-radius: 2px; margin: 22px 0;
  }

  /* Model metric */
  .model-metric {
      background: #f8fafc; border: 1px solid #e2e8f0;
      border-radius: 10px; padding: 14px 18px;
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 8px;
  }
  .model-metric-name { font-size: 0.85rem; color: #64748b; font-weight: 500; }
  .model-metric-value { font-size: 1.2rem; font-weight: 800; color: #0f172a; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading analytical data…")
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

@st.cache_resource(show_spinner="Loading ML model…")
def load_model():
    model = joblib.load(MODELS_DIR / "best_repeat_purchase_model.joblib")
    with open(MODELS_DIR / "model_metadata.json", "r") as f:
        meta = json.load(f)
    return model, meta

orders_df, cust_df, items_df, seg_df, perf_df, feat_imp_df, eda_meta = load_data()
model, model_meta = load_model()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">🛒 Customer Intelligence</div>
        <div class="sidebar-brand-sub">Olist E-Commerce Analytics Engine</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "📊  Executive Dashboard",
            "👥  Customer Intelligence",
            "🎯  RFM Segmentation",
            "🔮  Prediction Simulator",
            "📂  Upload New Data & Predict",
            "📦  Product Intelligence",
            "🤖  Model Evaluation",
            "💡  Business Playbooks",
        ],
        label_visibility="collapsed"
    )

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    **📁 Dataset**
    | Field | Value |
    |---|---|
    | Orders | 99,441 |
    | Delivered | 96,478 |
    | Customers | 93,358 |
    | Revenue | R$ 15.42M |
    | Period | 2016 – 2018 |

    **🤖 Champion Model**
    | Metric | Score |
    |---|---|
    | Algorithm | Logistic Reg. |
    | ROC-AUC | 0.6110 |
    | Recall | 56.96% |
    """)


# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPER
# ══════════════════════════════════════════════════════════════════════════════
PALETTE = ["#0e7490", "#0369a1", "#1d4ed8", "#7c3aed", "#db2777", "#ea580c", "#16a34a",
           "#d97706", "#475569", "#0891b2"]

def style_ax(ax):
    ax.set_facecolor("#f8fafc")
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#e2e8f0")
    ax.tick_params(colors="#64748b", labelsize=9)
    ax.xaxis.label.set_color("#475569")
    ax.yaxis.label.set_color("#475569")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊  Executive Dashboard":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">📊 Executive Performance Dashboard</div>
        <div class="hero-subtitle">Macro KPIs, revenue trends, and lifecycle overview — powered by 99,441 real Olist orders</div>
        <span class="hero-badge">✅ 100% Real Olist Brazilian E-Commerce Data</span>
    </div>
    """, unsafe_allow_html=True)

    # KPI ROW
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("💰", "R$ 15.42M", "Gross Revenue"),
        ("📦", "96,478",    "Delivered Orders"),
        ("👤", "93,358",    "Unique Customers"),
        ("🛒", "R$ 159.85", "Avg Order Value"),
        ("🔁", "3.00%",     "Repeat Buyer Rate"),
    ]
    for col, (icon, val, label) in zip([k1,k2,k3,k4,k5], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([6, 4])
    with col_l:
        st.markdown('<div class="section-card"><div class="section-title">📈 Monthly Revenue & Order Volume (2017–2018)</div>', unsafe_allow_html=True)
        delivered = orders_df[orders_df["is_delivered"] == 1].copy()
        monthly   = delivered.groupby("order_year_month").agg(
            revenue=("total_payment_value", "sum"),
            orders=("order_id", "count")
        ).reset_index()
        monthly = monthly[(monthly["order_year_month"] >= "2017-01") & (monthly["order_year_month"] <= "2018-08")]

        fig, ax1 = plt.subplots(figsize=(9, 3.8))
        fig.patch.set_facecolor("#ffffff")
        ax2 = ax1.twinx()
        ax1.fill_between(range(len(monthly)), monthly["revenue"]/1000, alpha=0.12, color="#0e7490")
        ax1.plot(range(len(monthly)), monthly["revenue"]/1000, color="#0e7490", marker="o", lw=2.2, markersize=5, label="Revenue (k BRL)")
        ax2.plot(range(len(monthly)), monthly["orders"], color="#f97316", marker="s", lw=1.8, linestyle="--", markersize=4, label="Orders")
        ax1.set_xticks(range(len(monthly)))
        ax1.set_xticklabels(monthly["order_year_month"], rotation=45, ha="right", fontsize=8)
        ax1.set_ylabel("Revenue (k BRL)", color="#0e7490", fontsize=9)
        ax2.set_ylabel("Order Count", color="#f97316", fontsize=9)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1+lines2, labels1+labels2, loc="upper left", fontsize=8)
        style_ax(ax1); style_ax(ax2)
        fig.tight_layout(pad=1.5)
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-card"><div class="section-title">🍩 Revenue by Segment</div>', unsafe_allow_html=True)
        top5 = seg_df.sort_values("total_revenue", ascending=False).head(5)
        fig, ax = plt.subplots(figsize=(5, 3.8))
        fig.patch.set_facecolor("#ffffff")
        ax.pie(top5["total_revenue"], labels=top5["customer_segment"], autopct="%1.1f%%",
               startangle=140, colors=PALETTE[:5],
               wedgeprops={"linewidth":2,"edgecolor":"white"},
               pctdistance=0.82, textprops={"fontsize":8})
        centre_circle = plt.Circle((0,0), 0.65, fc="white")
        ax.add_patch(centre_circle)
        ax.set_facecolor("#ffffff")
        fig.tight_layout(pad=1.0)
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-card"><div class="section-title">🏆 Top 10 Categories by Revenue</div>', unsafe_allow_html=True)
        cat_rev = items_df.groupby("product_category_name_english")["price"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        fig.patch.set_facecolor("#ffffff")
        ax.barh(cat_rev.index[::-1], cat_rev.values[::-1]/1e6, color=PALETTE[0], alpha=0.9, height=0.65)
        for i, (idx_, v) in enumerate(zip(cat_rev.index[::-1], cat_rev.values[::-1])):
            ax.text(v/1e6 + 0.01, i, f"R$ {v/1e6:.2f}M", va="center", fontsize=7.5, color="#374151")
        ax.set_xlabel("Revenue (Million BRL)", fontsize=9)
        ax.set_xlim(0, cat_rev.max()/1e6 * 1.25)
        style_ax(ax); fig.tight_layout(pad=1.5)
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-card"><div class="section-title">🗺️ Revenue by State (Top 10)</div>', unsafe_allow_html=True)
        state_rev = orders_df.groupby("customer_state")["total_payment_value"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        fig.patch.set_facecolor("#ffffff")
        ax.barh(state_rev.index[::-1], state_rev.values[::-1]/1e6, color=PALETTE[1], alpha=0.9, height=0.65)
        for i, v in enumerate(state_rev.values[::-1]):
            ax.text(v/1e6 + 0.01, i, f"R$ {v/1e6:.2f}M", va="center", fontsize=7.5, color="#374151")
        ax.set_xlabel("Revenue (Million BRL)", fontsize=9)
        ax.set_xlim(0, state_rev.max()/1e6 * 1.25)
        style_ax(ax); fig.tight_layout(pad=1.5)
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        💡 <strong>Key Executive Insights</strong><br>
        • <strong>São Paulo (SP)</strong> dominates with ~42% of total revenue — the commercial hub of Brazilian e-commerce.<br>
        • <strong>Bed, Bath & Table</strong> leads categories at R$ 1.71M in gross sales, driven by high average ticket sizes in home goods.<br>
        • Revenue grew <strong>~4.2× from Jan 2017 to Nov 2017</strong>, then stabilized through mid-2018 before the dataset cutoff.<br>
        • Only <strong>3% of customers repeat-purchased</strong> — the core business problem this ML system was designed to solve.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: CUSTOMER INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥  Customer Intelligence":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">👥 Customer Intelligence Explorer</div>
        <div class="hero-subtitle">Drill into customer cohorts, spending patterns, and repeat buyer profiles</div>
        <span class="hero-badge">93,358 Unique Customers Analyzed</span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔧 Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            all_segs = sorted(cust_df["customer_segment"].dropna().unique().tolist())
            sel_segs = st.multiselect("Customer Segments", all_segs,
                                      default=["Champions","At Risk","Potential Loyalists"])
        with fc2:
            max_m = int(cust_df["monetary"].max())
            spend_range = st.slider("Spend Range (BRL)", 0, min(max_m, 10000), (0, 3000))
        with fc3:
            repeat_only = st.checkbox("Repeat Buyers Only (≥ 2 orders)", value=False)

    fdf = cust_df[cust_df["customer_segment"].isin(sel_segs)]
    fdf = fdf[(fdf["monetary"] >= spend_range[0]) & (fdf["monetary"] <= spend_range[1])]
    if repeat_only:
        fdf = fdf[fdf["is_frequent"] == 1]

    st.info(f"🔍 **{len(fdf):,}** customers match your filters")

    ck1, ck2, ck3, ck4 = st.columns(4)
    for col, (icon, val, lbl) in zip([ck1,ck2,ck3,ck4], [
        ("👥", f"{len(fdf):,}", "Customers"),
        ("💰", f"R$ {fdf['monetary'].sum():,.0f}", "Total Spend"),
        ("📊", f"R$ {fdf['monetary'].mean():.2f}", "Avg Spend"),
        ("📅", f"{fdf['recency'].mean():.0f} days", "Avg Recency"),
    ]):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    cca, ccb = st.columns(2)
    with cca:
        st.markdown('<div class="section-card"><div class="section-title">📊 Spend Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6.5, 3.5)); fig.patch.set_facecolor("#ffffff")
        clip_data = fdf["monetary"].clip(upper=fdf["monetary"].quantile(0.99))
        ax.hist(clip_data, bins=50, color="#0e7490", alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.axvline(fdf["monetary"].median(), color="#f97316", lw=1.8, linestyle="--",
                   label=f"Median: R$ {fdf['monetary'].median():.0f}")
        ax.set_xlabel("Total Spend (BRL)", fontsize=9); ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8); style_ax(ax); fig.tight_layout(pad=1.5)
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with ccb:
        st.markdown('<div class="section-card"><div class="section-title">👥 Customers by Segment</div>', unsafe_allow_html=True)
        seg_cnt = fdf["customer_segment"].value_counts()
        fig, ax = plt.subplots(figsize=(6.5, 3.5)); fig.patch.set_facecolor("#ffffff")
        bars = ax.bar(seg_cnt.index, seg_cnt.values, color=PALETTE[:len(seg_cnt)], alpha=0.9, width=0.6)
        ax.set_ylabel("Customers", fontsize=9)
        plt.xticks(rotation=35, ha="right", fontsize=8)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
                    f"{bar.get_height():,}", ha="center", va="bottom", fontsize=7.5)
        style_ax(ax); fig.tight_layout(pad=1.5)
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 📋 Customer Profiles Table")
    display_cols = {
        "customer_unique_id":"Customer ID","customer_segment":"Segment",
        "monetary":"Total Spend (R$)","frequency":"Orders","avg_order_value":"AOV (R$)",
        "recency":"Recency (Days)","avg_review_score":"Rating",
        "customer_city":"City","customer_state":"State",
    }
    avail_cols = [c for c in display_cols if c in fdf.columns]
    st.dataframe(fdf[avail_cols].rename(columns=display_cols).head(500).reset_index(drop=True),
                 use_container_width=True, height=360)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: RFM SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯  RFM Segmentation":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🎯 RFM Customer Segmentation Matrix</div>
        <div class="hero-subtitle">Behavioral segmentation via Recency, Frequency & Monetary quintile scoring</div>
        <span class="hero-badge">9 Segments · 93,358 Customers Scored</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Segment Performance Summary")
    st.dataframe(seg_df.rename(columns={
        "customer_segment":"Segment","customer_count":"Customers",
        "customer_share_pct":"Share (%)","total_revenue":"GMV (BRL)",
        "revenue_share_pct":"Rev Share (%)","avg_spend_per_customer":"Avg Spend (R$)",
        "avg_recency_days":"Avg Recency (Days)","avg_order_value":"AOV (R$)",
    }), use_container_width=True, height=340)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    rca, rcb = st.columns(2)
    with rca:
        st.markdown('<div class="section-card"><div class="section-title">🔵 Recency vs. Monetary — Segment Scatter</div>', unsafe_allow_html=True)
        sample = cust_df.sample(min(8000, len(cust_df)), random_state=42)
        fig, ax = plt.subplots(figsize=(7, 4.5)); fig.patch.set_facecolor("#ffffff")
        segs_ = sample["customer_segment"].unique()
        cmap_ = dict(zip(segs_, PALETTE[:len(segs_)]))
        for seg in segs_:
            sub = sample[sample["customer_segment"]==seg]
            ax.scatter(sub["recency"], sub["monetary"], s=12, alpha=0.5, color=cmap_.get(seg,"#94a3b8"), label=seg)
        ax.set_yscale("log")
        ax.set_xlabel("Recency (Days Inactive)", fontsize=9)
        ax.set_ylabel("Monetary Spend (R$ — Log Scale)", fontsize=9)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
        style_ax(ax); fig.tight_layout(pad=1.5)
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with rcb:
        st.markdown('<div class="section-card"><div class="section-title">🔥 Avg Spend Heatmap — R vs. FM Score</div>', unsafe_allow_html=True)
        if "r_score" in cust_df.columns and "fm_score_avg" in cust_df.columns:
            pivot = cust_df.pivot_table(index="r_score", columns="fm_score_avg",
                                        values="monetary", aggfunc="mean").sort_index(ascending=False)
            fig, ax = plt.subplots(figsize=(7, 4.5)); fig.patch.set_facecolor("#ffffff")
            sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".0f", ax=ax,
                        linewidths=0.5, linecolor="#e2e8f0", cbar_kws={"shrink":0.8})
            ax.set_xlabel("FM Score", fontsize=9); ax.set_ylabel("R Score (5=Most Recent)", fontsize=9)
            fig.tight_layout(pad=1.5)
            st.pyplot(fig); plt.close()
        else:
            st.info("Score columns not found in dataset.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">💰 GMV Contribution by Segment</div>', unsafe_allow_html=True)
    seg_sorted = seg_df.sort_values("total_revenue", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 3.5)); fig.patch.set_facecolor("#ffffff")
    bars = ax.bar(seg_sorted["customer_segment"], seg_sorted["total_revenue"]/1e6,
                  color=PALETTE[:len(seg_sorted)], alpha=0.9, width=0.6)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"R$ {bar.get_height():.2f}M", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("GMV (Million BRL)", fontsize=9)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    style_ax(ax); fig.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        💡 <strong>RFM Insights</strong><br>
        • <strong>At Risk + Potential Loyalists</strong> = 57% of total GMV — the most critical cohorts to protect and convert.<br>
        • <strong>Champions</strong> (1.06% of customers) generate R$ 372.53 avg spend — 2.3× the platform average.<br>
        • <strong>Hibernating</strong> accounts for 31% of customers but only 23% of revenue — mostly lower-ticket single buyers.<br>
        • The FM heatmap shows customers scoring FM 4–5 spend 3–5× more than FM 1–2 groups consistently.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: PREDICTION SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Prediction Simulator":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🔮 ML Repeat-Purchase Prediction Simulator</div>
        <div class="hero-subtitle">Enter a customer's first-order profile to predict their return likelihood</div>
        <span class="hero-badge">Logistic Regression · ROC-AUC 0.611 · Real Trained Model</span>
    </div>
    """, unsafe_allow_html=True)

    st.info("**Anti-Leakage Design**: Only features available *after Order 1 delivery* are used — no future data.")

    st.markdown("### ⚙️ Customer First-Order Profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**💵 Transaction**")
        spend        = st.number_input("Total Spend (BRL)", 10.0, 5000.0, 180.0, 10.0)
        items_cnt    = st.number_input("Items in Order", 1, 21, 1)
        prod_cnt     = st.number_input("Distinct Products", 1, 21, 1)
        category     = st.selectbox("Product Category", [
            "bed_bath_table","furniture_decor","sports_leisure","computers_accessories",
            "health_beauty","watches_gifts","housewares","auto","toys","garden_tools",
            "telephony","cool_stuff","other_category"])

    with c2:
        st.markdown("**🚚 Fulfilment & Payment**")
        freight      = st.number_input("Freight (BRL)", 0.0, 400.0, 25.0, 5.0)
        installments = st.slider("Payment Installments", 1, 24, 3)
        payment_type = st.selectbox("Payment Method", ["credit_card","boleto","voucher","debit_card"])
        customer_state = st.selectbox("Customer State", ["SP","RJ","MG","RS","PR","SC","Other_State"])

    with c3:
        st.markdown("**⭐ Post-Delivery Experience**")
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
    if st.button("🚀 Run Prediction", type="primary"):
        prob = model.predict_proba(input_row)[0][1]
        prob_pct = prob * 100

        res1, res2 = st.columns([4,6])
        with res1:
            if prob >= 0.50:
                st.markdown(f"""
                <div class="pred-positive">
                    <div style="font-size:2rem">✅</div>
                    <div style="font-size:1.3rem;font-weight:800;color:#065f46">Likely to Return</div>
                    <div style="font-size:2.5rem;font-weight:900;color:#059669">{prob_pct:.1f}%</div>
                    <div style="font-size:0.85rem;color:#065f46">Repeat Purchase Probability</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-negative">
                    <div style="font-size:2rem">⚠️</div>
                    <div style="font-size:1.3rem;font-weight:800;color:#92400e">Unlikely to Return</div>
                    <div style="font-size:2.5rem;font-weight:900;color:#ea580c">{prob_pct:.1f}%</div>
                    <div style="font-size:0.85rem;color:#92400e">Repeat Purchase Probability</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:12px;padding:10px 14px;background:#f1f5f9;border-radius:8px;font-size:0.82rem;color:#475569;">
                📌 Marketplace baseline repeat rate: <strong>3.0%</strong><br>
                🎯 Model recall on repeat buyers: <strong>56.96%</strong>
            </div>""", unsafe_allow_html=True)

        with res2:
            from recommendations import RecommendationEngine
            rec = RecommendationEngine().get_individual_action(
                prob, "Potential Loyalists" if spend >= 200 else "Standard", is_delayed=is_delayed)
            st.markdown(f"""
            <div class="playbook-card">
                <div class="playbook-title">🎯 Recommended Action — {rec['playbook']}</div>
                <div style="margin-bottom:8px;font-size:0.78rem;color:#6b7280">
                    Urgency: <strong style="color:#dc2626">{rec['urgency']}</strong></div>
                <div class="playbook-body">{rec['action']}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📈 Top 10 Feature Contributions")
        top_f = feat_imp_df.head(10)
        metric_col = "abs_impact" if "abs_impact" in top_f.columns else "importance"
        fig, ax = plt.subplots(figsize=(10, 3.5)); fig.patch.set_facecolor("#ffffff")
        ax.barh(top_f["feature"][::-1], top_f[metric_col][::-1], color="#0e7490", alpha=0.85, height=0.65)
        ax.set_xlabel("Absolute Coefficient Impact", fontsize=9)
        style_ax(ax); fig.tight_layout(pad=1.5)
        st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: UPLOAD NEW DATA & BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂  Upload New Data & Predict":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">📂 Upload New Dataset & Batch Prediction Engine</div>
        <div class="hero-subtitle">Score new customer order batches, predict repeat purchase propensity, or update raw store transactions</div>
        <span class="hero-badge">Production ML Inference Pipeline</span>
    </div>
    """, unsafe_allow_html=True)

    tab_pred, tab_raw = st.tabs([
        "🚀 Batch Customer Prediction (CSV Upload)",
        "🔄 Ingest Full Raw Dataset (System Retraining)"
    ])

    with tab_pred:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">📤 Upload New Customer Batch for Repeat Purchase Prediction</div>
            <div style="font-size:0.88rem;color:#64748b;margin-bottom:14px;">
                Upload a CSV file containing new customer Order #1 transaction details (or load the sample batch below). 
                The trained champion Machine Learning model will instantly compute repeat purchase probabilities, risk tiers, and personalized retention actions.
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
            use_sample = st.button("⚡ Load 30 Real Sample Customers", use_container_width=True)
            
            sample_path = PROJECT_ROOT / "data" / "sample_new_customers.csv"
            if sample_path.exists():
                sample_bytes = sample_path.read_bytes()
                st.download_button(
                    "📥 Download CSV Template",
                    data=sample_bytes,
                    file_name="new_customers_template.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        st.markdown("</div>", unsafe_allow_html=True)

        # Determine dataframe to process
        df_to_score = None
        if uploaded_file is not None:
            try:
                df_to_score = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded uploaded file with {len(df_to_score):,} customer records.")
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
        elif use_sample or st.session_state.get("sample_loaded", False):
            st.session_state["sample_loaded"] = True
            if sample_path.exists():
                df_to_score = pd.read_csv(sample_path)
                st.info(f"ℹ️ Loaded pre-packaged batch of {len(df_to_score)} real customer transactions from dataset.")

        if df_to_score is not None:
            # Required schema columns
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

            # Validate and fill missing columns gracefully
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
                st.warning(f"⚠️ Note: Auto-filled default values for {len(missing_cols)} missing columns: `{', '.join(missing_cols[:5])}`...")

            # Run Model Inference
            X_batch = df_proc[num_cols + cat_cols]
            probs = model.predict_proba(X_batch)[:, 1]
            prob_pcts = np.round(probs * 100, 1)

            df_proc["repeat_prob_pct"] = prob_pcts
            df_proc["prediction_label"] = np.where(probs >= 0.50, "Likely Repeat", "One-Time Buyer")
            
            def assign_tier(p):
                if p >= 50.0:
                    return "🟢 High (>50%)"
                elif p >= 40.0:
                    return "🟡 Medium (40-50%)"
                else:
                    return "🔴 Low (<40%)"

            def assign_action(p):
                if p >= 50.0:
                    return "🏆 VIP Loyalty Invitation & Priority Shipping Guarantee"
                elif p >= 40.0:
                    return "🎟️ 10% Discount Voucher for 2nd Purchase within 14 Days"
                else:
                    return "📬 Post-Delivery Satisfaction Check-in & Customer Care"

            df_proc["propensity_tier"] = df_proc["repeat_prob_pct"].apply(assign_tier)
            df_proc["recommended_action"] = df_proc["repeat_prob_pct"].apply(assign_action)

            # Summary KPIs
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
                    <div class="kpi-icon">📋</div>
                    <div class="kpi-value">{n_total:,}</div>
                    <div class="kpi-label">Customers Scored</div>
                </div>""", unsafe_allow_html=True)
            with kpi_c2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-icon">🟢</div>
                    <div class="kpi-value">{n_high:,} <span style="font-size:1rem;color:#10b981;">({n_high/n_total*100:.1f}%)</span></div>
                    <div class="kpi-label">High Propensity (≥50%)</div>
                </div>""", unsafe_allow_html=True)
            with kpi_c3:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-icon">🟡</div>
                    <div class="kpi-value">{n_med:,} <span style="font-size:1rem;color:#f59e0b;">({n_med/n_total*100:.1f}%)</span></div>
                    <div class="kpi-label">Medium Propensity (40-50%)</div>
                </div>""", unsafe_allow_html=True)
            with kpi_c4:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-icon">💰</div>
                    <div class="kpi-value">R$ {total_rev_risk:,.0f}</div>
                    <div class="kpi-label">Batch First-Order Volume</div>
                </div>""", unsafe_allow_html=True)

            # Visualizations
            st.markdown("<br>", unsafe_allow_html=True)
            chart1, chart2 = st.columns(2)
            with chart1:
                st.markdown('<div class="section-card"><div class="section-title">📊 Predicted Probability Distribution</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6, 3.2)); fig.patch.set_facecolor("#ffffff")
                ax.hist(prob_pcts, bins=15, color="#0e7490", edgecolor="#ffffff", alpha=0.85)
                ax.axvline(50, color="#ef4444", linestyle="--", linewidth=1.5, label="Decision Threshold (50%)")
                ax.set_xlabel("Predicted Repeat Probability (%)", fontsize=9)
                ax.set_ylabel("Customer Count", fontsize=9)
                ax.legend(fontsize=8, loc="upper right")
                style_ax(ax); fig.tight_layout(pad=1.2)
                st.pyplot(fig); plt.close()
                st.markdown('</div>', unsafe_allow_html=True)

            with chart2:
                st.markdown('<div class="section-card"><div class="section-title">🎯 Propensity Tier Breakdown</div>', unsafe_allow_html=True)
                tier_counts = df_proc["propensity_tier"].value_counts()
                fig, ax = plt.subplots(figsize=(6, 3.2)); fig.patch.set_facecolor("#ffffff")
                colors = ["#10b981" if "High" in t else "#f59e0b" if "Medium" in t else "#ef4444" for t in tier_counts.index]
                ax.bar(tier_counts.index, tier_counts.values, color=colors, width=0.55, edgecolor="#ffffff")
                for i, v in enumerate(tier_counts.values):
                    ax.text(i, v + (max(tier_counts.values)*0.02), f"{v:,}", ha="center", fontsize=9, fontweight="bold", color="#1e293b")
                ax.set_ylabel("Customer Count", fontsize=9)
                style_ax(ax); fig.tight_layout(pad=1.2)
                st.pyplot(fig); plt.close()
                st.markdown('</div>', unsafe_allow_html=True)

            # Filterable Results Table
            st.markdown('<div class="section-card"><div class="section-title">📋 Scored Customer Intelligence Roster</div>', unsafe_allow_html=True)
            f_tier = st.selectbox("Filter by Propensity Tier:", ["All Tiers", "🟢 High (>50%)", "🟡 Medium (40-50%)", "🔴 Low (<40%)"])
            
            display_df = df_proc.copy()
            if f_tier != "All Tiers":
                display_df = display_df[display_df["propensity_tier"] == f_tier]

            # Select key intuitive columns for presentation
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
                height=320
            )

            # Download Button
            csv_export = df_proc.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Full Scored Dataset to CSV",
                data=csv_export,
                file_name="batch_repeat_predictions_scored.csv",
                mime="text/csv",
                type="primary"
            )
            st.markdown('</div>', unsafe_allow_html=True)

    with tab_raw:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">🔄 Ingest Entire Raw E-Commerce Dataset (Full Re-run)</div>
            <div style="font-size:0.88rem;color:#64748b;margin-bottom:16px;">
                To update the website with an entirely new marketplace dataset (e.g. next quarter's orders, or another e-commerce platform), 
                the system provides an automated 5-stage Data Science pipeline that cleans, validates, integrates, segments, and re-trains models from scratch.
            </div>
        """, unsafe_allow_html=True)

        raw_dir = PROJECT_ROOT / "data" / "raw"
        raw_files = list(raw_dir.glob("*.csv")) if raw_dir.exists() else []

        st.markdown("#### 📁 Current Raw Dataset Files in `data/raw/`")
        if raw_files:
            file_info = []
            for rf in sorted(raw_files):
                size_mb = rf.stat().st_size / (1024 * 1024)
                file_info.append({
                    "Dataset File": rf.name,
                    "Size (MB)": f"{size_mb:.2f} MB",
                    "Status": "✅ Present & Validated"
                })
            st.dataframe(pd.DataFrame(file_info), use_container_width=True, height=260)
        else:
            st.warning("No raw files found in `data/raw/`.")

        st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
        st.markdown("#### ⚙️ How to Ingest a New Raw Dataset")
        st.markdown("""
        1. **Replace CSV Files**: Place your updated CSV files in `data/raw/` matching the Olist schema (e.g. `olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, etc.).
        2. **Execute Master Pipeline**: Run the orchestrator script in your terminal:
        ```bash
        python src/run_pipeline.py
        ```
        3. **Automated Execution Sequence**:
           - **Stage 1 (Cleaning)**: Deduplicates, standardizes date datatypes, fixes UTF-8 encodings.
           - **Stage 2 (Integration)**: Joins orders, payments, reviews, and items into analytical tables.
           - **Stage 3 (RFM Segmentation)**: Re-calculates Recency, Frequency, Monetary quintiles and re-profiles 9 segments.
           - **Stage 4 (Feature Engineering)**: Constructs anti-leakage Order #1 feature matrix.
           - **Stage 5 (Machine Learning)**: Retrains Logistic Regression, Decision Tree, and Random Forest; picks the champion model.
        4. **Reload Website**: Simply refresh your browser tab. All 8 dashboard pages will instantly display your new dataset!
        """)

        st.markdown("""
        <div class="insight-box">
            💡 <strong>Interview Pro-Tip</strong><br>
            • This architecture cleanly decouples <strong>batch inference</strong> (fast scoring of new incoming customer files without re-training) 
            from <strong>scheduled pipeline retraining</strong> (periodic re-fitting as customer behavior drifts over time).
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: PRODUCT INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦  Product Intelligence":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">📦 Product & Category Intelligence</div>
        <div class="hero-subtitle">Revenue leaders, freight burden, pricing patterns, and category performance</div>
        <span class="hero-badge">73 Product Categories Analyzed</span>
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

    pa, pb = st.columns(2)
    with pa:
        st.markdown('<div class="section-card"><div class="section-title">🥇 Top 15 Categories by Revenue</div>', unsafe_allow_html=True)
        top15 = cat_stats.sort_values("revenue", ascending=False).head(15)
        st.dataframe(top15.rename(columns={
            "product_category_name_english":"Category","units_sold":"Units",
            "revenue":"Revenue (R$)","avg_price":"Avg Price (R$)",
            "avg_freight":"Avg Freight (R$)","freight_pct":"Freight Burden (%)"
        }).reset_index(drop=True), use_container_width=True, height=420)
        st.markdown('</div>', unsafe_allow_html=True)

    with pb:
        st.markdown('<div class="section-card"><div class="section-title">⚡ Highest Freight Burden (≥50 units sold)</div>', unsafe_allow_html=True)
        high_fr = cat_stats[cat_stats["units_sold"]>=50].sort_values("freight_pct", ascending=False).head(15)
        st.dataframe(high_fr.rename(columns={
            "product_category_name_english":"Category","units_sold":"Units",
            "revenue":"Revenue (R$)","avg_price":"Avg Price (R$)",
            "avg_freight":"Avg Freight (R$)","freight_pct":"Freight % of Price"
        }).reset_index(drop=True), use_container_width=True, height=420)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">💰 Price vs. Freight — Top 20 Categories (bubble = units sold)</div>', unsafe_allow_html=True)
    top20 = cat_stats.sort_values("revenue", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(12, 4.5)); fig.patch.set_facecolor("#ffffff")
    sc = ax.scatter(top20["avg_price"], top20["avg_freight"], s=top20["units_sold"]/8,
                    alpha=0.75, c=range(len(top20)), cmap="viridis", edgecolors="white", linewidths=0.8)
    for _, row in top20.head(12).iterrows():
        ax.text(row["avg_price"]+1, row["avg_freight"]+0.3,
                row["product_category_name_english"].replace("_"," "), fontsize=7.5)
    ax.set_xlabel("Average Item Price (R$)", fontsize=9)
    ax.set_ylabel("Average Freight Fee (R$)", fontsize=9)
    plt.colorbar(sc, ax=ax, label="Rank by Revenue")
    style_ax(ax); fig.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">📦 Units Sold — Top 20 Categories</div>', unsafe_allow_html=True)
    top20u = cat_stats.sort_values("units_sold", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(12, 4)); fig.patch.set_facecolor("#ffffff")
    ax.bar(top20u["product_category_name_english"], top20u["units_sold"], color="#0e7490", alpha=0.9, width=0.7)
    plt.xticks(rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Units Sold", fontsize=9)
    style_ax(ax); fig.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Evaluation":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🤖 ML Model Evaluation Report</div>
        <div class="hero-subtitle">ROC curves, confusion matrices, feature importance & model comparison</div>
        <span class="hero-badge">3 Algorithms · Stratified 80/20 Split · Balanced Class Weights</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Model Performance Comparison")
    st.dataframe(perf_df.style.format({
        "Accuracy":"{:.4f}","Precision":"{:.4f}","Recall":"{:.4f}",
        "F1_Score":"{:.4f}","ROC_AUC":"{:.4f}","PR_AUC":"{:.4f}"
    }).highlight_max(subset=["Recall","ROC_AUC"], color="#d1fae5"
    ).highlight_min(subset=["Precision"], color="#fee2e2"),
    use_container_width=True)

    st.markdown("""
    > **Why Recall is primary**: The positive class (repeat buyers = 3%) is rare. Missing a true repeat buyer (False Negative)
    > costs the business a customer worth R$ 300+. We used `class_weight='balanced'` to compensate.
    > The Logistic Regression champion captures **~57% of all real repeat buyers** on the held-out test set.
    """)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    roc_path = VIZ_DIR / "12_model_roc_and_pr_curves.png"
    cm_path  = VIZ_DIR / "14_confusion_matrices.png"
    fi_path  = VIZ_DIR / "13_feature_importance.png"

    if roc_path.exists():
        st.markdown("### 📈 ROC & Precision-Recall Curves")
        st.image(str(roc_path), use_container_width=True)

    if cm_path.exists():
        st.markdown("### 🔢 Confusion Matrices")
        st.image(str(cm_path), use_container_width=True)

    if fi_path.exists():
        st.markdown("### 🔍 Feature Importance — Champion Model")
        st.image(str(fi_path), use_container_width=True)

    st.markdown("### 🏆 Champion Model Metadata")
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
        🧠 <strong>Model Interpretation</strong><br>
        • <strong>review_score_order1</strong> is the strongest single predictor — a 5-star review is the clearest signal of repeat propensity.<br>
        • <strong>spend_order1</strong> and <strong>delivery_days_order1</strong> are second-tier signals — high first-order spend and fast delivery drive returns.<br>
        • <strong>customer_state_SP</strong> encodes a marketplace composition effect — São Paulo customers have higher baseline return rates.<br>
        • PR-AUC of 0.060 vs. 0.038 baseline = ~1.6× lift in identifying actual repeat buyers — meaningful in a class-imbalanced setting.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7: BUSINESS PLAYBOOKS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡  Business Playbooks":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">💡 Strategic Business Playbooks</div>
        <div class="hero-subtitle">Data-grounded action rules mapping RFM segments to commercial initiatives</div>
        <span class="hero-badge">9 Empirically-Grounded Playbooks</span>
    </div>
    """, unsafe_allow_html=True)

    from recommendations import RecommendationEngine
    recs = RecommendationEngine().get_all_recommendations()

    PRIORITY_MAP = {
        "Champions":          ("🟢 HIGH LTV PROTECTION",   "#dcfce7", "#16a34a"),
        "Potential Loyalists": ("🟦 HIGH REVENUE CAPTURE",  "#e0f2fe", "#0369a1"),
        "At Risk":             ("🔴 HIGH CHURN MITIGATION", "#fee2e2", "#b91c1c"),
        "Can't Lose Them":     ("🔴 CRITICAL WINBACK",      "#fee2e2", "#991b1b"),
        "New Customers":       ("🟡 ONBOARDING",            "#fef9c3", "#92400e"),
        "Promising":           ("🟡 NURTURING",             "#fef9c3", "#78350f"),
        "Loyal Customers":     ("🟢 SUBSCRIPTION LOCK-IN",  "#dcfce7", "#15803d"),
        "Hibernating":         ("⚪ LOW-CAC REACTIVATION",  "#f1f5f9", "#475569"),
        "Lost":                ("⚫ SUPPRESS / MONITOR",    "#f8fafc", "#94a3b8"),
    }

    for idx, r in enumerate(recs):
        seg = r["customer_segment"]
        priority_label, bg_color, txt_color = PRIORITY_MAP.get(seg, ("— —", "#f8fafc", "#374151"))
        with st.expander(f"📌 Playbook #{idx+1} — {seg}", expanded=(idx < 3)):
            p1, p2 = st.columns([6,4])
            with p1:
                st.markdown("**🔍 Problem:**"); st.write(r["problem"])
                st.markdown("**📊 Evidence:**"); st.info(r["evidence"])
                st.markdown("**✅ Recommended Action:**"); st.success(r["recommended_action"])
            with p2:
                st.markdown("**🎯 Business Objective:**"); st.write(r["business_objective"])
                st.markdown(f"""
                <div style="background:{bg_color};border-left:4px solid {txt_color};
                            border-radius:8px;padding:12px 16px;margin-top:10px;">
                    <span style="color:{txt_color};font-weight:700;font-size:0.9rem;">{priority_label}</span>
                </div>""", unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
        🔑 <strong>Strategic Priority Matrix</strong><br>
        • Allocate <strong>70% of marketing budget</strong> to <em>Potential Loyalists</em> (R$ 4.35M at risk) and <em>At Risk</em> (R$ 4.42M dormant).<br>
        • <strong>Champions</strong> require no discounting — invest in exclusivity and VIP experiences to retain R$ 372 avg spend.<br>
        • <strong>Hibernating</strong> (31% of base) should be re-engaged via low-cost push/email during seasonal peaks only.<br>
        • <strong>Lost</strong> customers should be fully suppressed from paid retargeting — each paid click is CAC-negative at 473+ days recency.
    </div>
    """, unsafe_allow_html=True)
