# E-Commerce Customer Intelligence & Purchase Prediction System

> **Portfolio-Grade Data Science Project** | Olist Brazilian E-Commerce Public Dataset | Python · Scikit-learn · Streamlit

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65-red.svg)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange.svg)](https://scikit-learn.org)
[![Data](https://img.shields.io/badge/Data-100%25%20Real%20Olist-green.svg)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

---

## Project Overview

This is an end-to-end **Customer Intelligence and Machine Learning** platform built on the real **Olist Brazilian E-Commerce Public Dataset** (99,441 orders, 2016–2018). The system performs:

- **Customer Segmentation** via RFM (Recency, Frequency, Monetary) analysis
- **Repeat Purchase Prediction** using a trained ML model (Logistic Regression, Decision Tree, Random Forest)
- **Interactive Business Analytics** via a multi-page Streamlit dashboard
- **Actionable Business Recommendations** mapped to each customer segment

All results, model metrics, business insights, and data figures in this project are derived **100% from the real Olist dataset** — nothing is fabricated.

---

## Key Results (Real, Verified)

| KPI | Value |
|-----|-------|
| Total Orders Analyzed | 99,441 |
| Delivered Orders | 96,478 |
| Unique Customers | 93,358 |
| Gross Revenue (Delivered) | R$ 15.42M |
| Average Order Value | R$ 159.85 |
| Repeat Buyer Rate | 3.00% |
| Dataset Time Span | Sep 2016 – Aug 2018 |

### ML Model Performance (Test Set — 12,460 samples)

| Model | ROC-AUC | Recall | Precision | F1-Score |
|-------|---------|--------|-----------|----------|
| **Logistic Regression** ✅ Champion | **0.6110** | **56.96%** | 5.31% | 9.71% |
| Decision Tree | 0.5485 | 50.00% | 4.73% | 8.64% |
| Random Forest | 0.5989 | 44.73% | 5.68% | 10.08% |

> **Note on metrics**: The dataset is severely class-imbalanced (97% non-repeat buyers). Recall is the primary optimization metric — missing a true repeat buyer is far more costly than a false positive. All models use `class_weight='balanced'` to address this.

### RFM Segment Summary

| Segment | Customers | Share | GMV | Avg Spend |
|---------|-----------|-------|-----|-----------|
| Hibernating | 29,202 | 31.3% | R$ 3.55M | R$ 121.53 |
| At Risk | 14,457 | 15.5% | R$ 4.42M | R$ 305.81 |
| Potential Loyalists | 14,405 | 15.4% | R$ 4.35M | R$ 302.21 |
| New Customers | 10,886 | 11.7% | R$ 791K | R$ 72.70 |
| Promising | 10,849 | 11.6% | R$ 794K | R$ 73.16 |
| Lost | 11,431 | 12.2% | R$ 829K | R$ 72.49 |
| Champions | 985 | 1.1% | R$ 367K | R$ 372.53 |
| Loyal Customers | 820 | 0.9% | R$ 205K | R$ 249.57 |
| Can't Lose Them | 323 | 0.3% | R$ 114K | R$ 352.25 |

---

## Project Architecture

```
ecommerce-customer-intelligence/
├── data/
│   ├── raw/                         # Original 9 Olist CSV files (unchanged)
│   └── processed/                   # Cleaned + feature-engineered outputs
│       ├── cleaned_orders.csv
│       ├── cleaned_customers.csv
│       ├── orders_analytical.csv    # Wide analytical table (orders + payments + reviews)
│       ├── customers_analytical.csv # Customer-level aggregates
│       ├── customer_rfm_segments.csv# RFM scored + segmented customers
│       └── customer_prediction_dataset.csv  # ML-ready feature matrix
├── src/
│   ├── data_loader.py               # Structured CSV loading utilities
│   ├── data_cleaning.py             # Phase 2: Data cleaning pipeline
│   ├── data_integration.py          # Phase 3: Dataset merging & integration
│   ├── eda.py                       # Phase 4: Exploratory data analysis
│   ├── feature_engineering.py       # Phase 5: Feature creation for ML
│   ├── rfm.py                       # Phase 6: RFM scoring + segmentation
│   ├── segmentation.py              # Phase 7: Customer segment profiling
│   ├── modeling.py                  # Phase 9-10: ML training & evaluation
│   ├── recommendations.py           # Phase 12: Business recommendation engine
│   └── run_pipeline.py              # Master pipeline orchestrator
├── dashboard/
│   └── app.py                       # Streamlit multi-page dashboard
├── models/
│   ├── best_repeat_purchase_model.joblib
│   └── model_metadata.json
├── reports/
│   ├── dataset_inspection_report.md
│   ├── eda_report.md
│   ├── rfm_segment_summary.csv
│   ├── model_performance_comparison.csv
│   └── feature_importance.csv
├── visualizations/                  # 14 publication-quality charts
└── requirements.txt
```

---

## Dashboard Pages

The Streamlit dashboard (`dashboard/app.py`) includes **8 pages**:

| Page | Description |
|------|-------------|
| 📊 Executive Dashboard | KPIs, monthly revenue trends, category & state revenue charts |
| 👥 Customer Intelligence | Filterable customer profile explorer with cohort analytics |
| 🎯 RFM Segmentation | Segment performance matrix, scatter plot, FM heatmap |
| 🔮 Prediction Simulator | Interactive ML predictor — enter Order #1 details, get repeat probability |
| 📂 Upload New Data & Predict | Drag-and-drop CSV batch inference engine & raw dataset ingestion architecture |
| 📦 Product Intelligence | Category revenue, freight burden, price vs. freight bubble chart |
| 🤖 Model Evaluation | ROC/PR curves, confusion matrices, feature importance, model comparison |
| 💡 Business Playbooks | 9 data-grounded segment playbooks with commercial action rules |

---

## Project Phases

| Phase | Description | Script |
|-------|-------------|--------|
| 1 | Dataset Inspection & EDA | `inspect_data_raw.py`, `audit_relations.py` |
| 2 | Data Cleaning | `data_cleaning.py` |
| 3 | Data Integration & Merging | `data_integration.py` |
| 4 | Exploratory Data Analysis | `eda.py` |
| 5 | Feature Engineering | `feature_engineering.py` |
| 6 | RFM Analysis | `rfm.py` |
| 7 | Customer Segmentation | `segmentation.py` |
| 8–10 | ML Modeling, Training, Evaluation | `modeling.py` |
| 11 | Business Recommendations | `recommendations.py` |
| 12 | Interactive Dashboard | `dashboard/app.py` |

---

## Installation & Running

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib scipy
```

### 2. Prepare dataset

Place the 9 raw Olist CSV files in `data/raw/`:
- `olist_customers_dataset.csv`
- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_products_dataset.csv`
- `olist_sellers_dataset.csv`
- `olist_geolocation_dataset.csv`
- `product_category_name_translation.csv`

### 3. Run the full ML pipeline (if starting fresh)

```bash
cd ecommerce-customer-intelligence
python src/run_pipeline.py
```

### 4. Launch the dashboard

```bash
python -m streamlit run dashboard/app.py
```

Then open **http://localhost:8501** in your browser.

---

## Technical Design Decisions

### Anti-Leakage Feature Engineering
The repeat-purchase prediction model uses **only features available immediately after Order #1 delivery**:
- First-order spend, items count, freight ratio
- Payment method & installments
- Delivery lead time & delay days
- Review score given after Order #1
- Customer geography (state)
- Purchase timing (hour, day of week)

This prevents any data leakage from future orders contaminating the model.

### Class Imbalance Handling
With only 3.0% positive class (repeat buyers), we used:
- `class_weight='balanced'` on all classifiers
- Stratified train/test split to preserve class proportions
- ROC-AUC and Recall as primary evaluation metrics (not Accuracy)

### RFM Methodology
- Quintile-based scoring (1–5 for R, F, M independently)
- Composite FM score averaged from F and M quintiles
- Rule-based segment assignment using business-relevant thresholds
- 9 segments aligned with standard RFM naming conventions

---

## Key Business Insights

1. **Repeat purchase rate is only 3%** — the marketplace is heavily acquisition-driven. Customer retention is the single largest growth lever.

2. **At Risk + Potential Loyalists control 57% of GMV** — losing either cohort would be catastrophic. Win-back and nurturing campaigns for these two segments should receive the majority of marketing budget.

3. **Champions (1% of customers) spend 2.3× the platform average** — they require zero-discount VIP treatment, not margin-burning promotions.

4. **São Paulo (SP) generates 42% of all revenue** — geographic concentration creates both risk and opportunity for regional promotions.

5. **Review score is the strongest ML predictor of repeat purchase** — quality of post-delivery experience is the most actionable lever to improve retention.

6. **Freight cost is a major AOV drag** — categories with freight burden >60% of item price (e.g. furniture, large appliances) need free-shipping thresholds or consolidation strategies.

---

## Dataset Source

**Olist Brazilian E-Commerce Public Dataset**  
Available on Kaggle: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

---

## Author

Built as a portfolio-grade Data Science project demonstrating:
- End-to-end data pipeline construction
- Real-world EDA with 99K+ rows and multi-table joins
- RFM customer segmentation methodology
- Machine learning for imbalanced classification
- Interactive business analytics with Streamlit
- Data-driven business storytelling and recommendation frameworks
