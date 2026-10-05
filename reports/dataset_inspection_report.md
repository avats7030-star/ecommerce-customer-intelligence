# Olist Brazilian E-Commerce Dataset — Comprehensive Dataset Inspection Report

**Project**: E-Commerce Customer Intelligence & Purchase Prediction System  
**Dataset**: Olist Brazilian E-Commerce Public Dataset  
**Inspection Date**: October 2026  
**Status**: Real Data Verified (Zero Synthetic/Fabricated Content)

---

## 1. Files Detected
All 9 official Olist CSV files were detected and loaded from `data/raw/`:

| File Name | File Size (MB) | Raw Row Count | Column Count | Primary Role |
| :--- | :---: | :---: | :---: | :--- |
| `olist_customers_dataset.csv` | 8.62 MB | 99,441 | 5 | Customer session keys & actual customer IDs |
| `olist_orders_dataset.csv` | 16.84 MB | 99,441 | 8 | Core order statuses & lifecycle timestamps |
| `olist_order_items_dataset.csv` | 14.72 MB | 112,650 | 7 | Item-level transactions, prices, freight |
| `olist_order_payments_dataset.csv` | 5.51 MB | 103,886 | 5 | Payment methods, installments, amounts |
| `olist_order_reviews_dataset.csv` | 13.78 MB | 99,224 | 7 | Customer ratings (1-5) and textual feedback |
| `olist_products_dataset.csv` | 2.27 MB | 32,951 | 9 | Product dimensions, categories, and metrics |
| `olist_sellers_dataset.csv` | 0.17 MB | 3,095 | 4 | Seller registry and location |
| `olist_geolocation_dataset.csv` | 58.44 MB | 1,000,163 | 5 | Zip code prefix geographic coordinates |
| `product_category_name_translation.csv` | < 0.01 MB | 71 | 2 | Portuguese-to-English category translation |

**Grand Total Raw Volume**: **1,450,922 records** across 9 relational tables.

---

## 2. Table-by-Table Schema, Data Types & Missing Values

### 2.1 Customers (`olist_customers_dataset.csv`)
* **Total Rows**: 99,441 | **Total Columns**: 5 | **Duplicates**: 0
* **Unique `customer_id`**: 99,441 (Order-level surrogate key)
* **Unique `customer_unique_id`**: 96,096 (Actual human customer entity)
* **Repeat Customers**: 2,997 (3.12%) | **One-time Customers**: 93,099 (96.88%)
* **Max Orders by Single Customer**: 17 orders

| Column Name | Inferred / Actual Data Type | Missing Count | Missing % | Distinct Values | Description / Key Semantic |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `customer_id` | string (32-char hex UUID) | 0 | 0.0% | 99,441 | Order-specific customer session key (1:1 with orders) |
| `customer_unique_id` | string (32-char hex UUID) | 0 | 0.0% | 96,096 | **True customer identifier** for RFM and lifetime analysis |
| `customer_zip_code_prefix` | string (5-digit postal) | 0 | 0.0% | 14,994 | Postal code prefix |
| `customer_city` | string | 0 | 0.0% | 4,119 | Customer residence city |
| `customer_state` | string (2-char UF) | 0 | 0.0% | 27 | Brazilian federative state (e.g., SP, RJ, MG) |

---

### 2.2 Orders (`olist_orders_dataset.csv`)
* **Total Rows**: 99,441 | **Total Columns**: 8 | **Duplicates**: 0
* **Date Range**: `2016-09-04 21:15:19` to `2018-10-17 17:30:18` (773 days / ~25 months)

| Column Name | Inferred / Actual Data Type | Missing Count | Missing % | Distinct Values | Description / Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `order_id` | string (32-char hex UUID) | 0 | 0.0% | 99,441 | **Primary Key** |
| `customer_id` | string (32-char hex UUID) | 0 | 0.0% | 99,441 | Foreign key -> `customers.customer_id` (100% match) |
| `order_status` | category / string | 0 | 0.0% | 8 | delivered (96,478), shipped (1,107), canceled (625), unavailable (609), invoiced (314), processing (301), created (5), approved (2) |
| `order_purchase_timestamp` | datetime64[ns] | 0 | 0.0% | 98,875 | Purchase checkout timestamp |
| `order_approved_at` | datetime64[ns] | 160 | 0.161% | 90,733 | Payment clearance timestamp (missing if canceled/unpaid) |
| `order_delivered_carrier_date` | datetime64[ns] | 1,783 | 1.793% | 81,018 | Handoff to logistics carrier |
| `order_delivered_customer_date` | datetime64[ns] | 2,965 | 2.982% | 95,664 | Actual customer delivery timestamp |
| `order_estimated_delivery_date` | datetime64[ns] | 0 | 0.0% | 459 | Promised delivery date provided at purchase |

---

### 2.3 Order Items (`olist_order_items_dataset.csv`)
* **Total Rows**: 112,650 | **Total Columns**: 7 | **Orders Represented**: 98,666
* **Orders without Items**: 775 (Orders canceled/unavailable prior to fulfillment)
* **Max Items in Single Order**: 21 items

| Column Name | Inferred / Actual Data Type | Missing Count | Missing % | Distinct Values | Range / Semantic |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `order_id` | string (32-char hex UUID) | 0 | 0.0% | 98,666 | Foreign key -> `orders.order_id` |
| `order_item_id` | int64 | 0 | 0.0% | 21 | Sequential item index within the order (1, 2, 3...) |
| `product_id` | string (32-char hex UUID) | 0 | 0.0% | 32,951 | Foreign key -> `products.product_id` (100% matched) |
| `seller_id` | string (32-char hex UUID) | 0 | 0.0% | 3,095 | Foreign key -> `sellers.seller_id` (100% matched) |
| `shipping_limit_date` | datetime64[ns] | 0 | 0.0% | 93,318 | Seller dispatch deadline |
| `price` | float64 | 0 | 0.0% | 5,968 | Item selling price (Min: R$0.85, Max: R$6,735.00) |
| `freight_value` | float64 | 0 | 0.0% | 6,999 | Shipping fee (Min: R$0.00, Max: R$409.68) |

---

### 2.4 Order Payments (`olist_order_payments_dataset.csv`)
* **Total Rows**: 103,886 | **Total Columns**: 5 | **Orders Represented**: 99,440
* **Max Payment Splits per Order**: 29 payment records (Voucher/multi-card splits)

| Column Name | Inferred / Actual Data Type | Missing Count | Missing % | Distinct Values | Distribution / Range |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `order_id` | string (32-char hex UUID) | 0 | 0.0% | 99,440 | Foreign key -> `orders.order_id` |
| `payment_sequential` | int64 | 0 | 0.0% | 29 | Sequential number of payment method applied (1 to 29) |
| `payment_type` | category / string | 0 | 0.0% | 5 | credit_card: 76,795, boleto: 19,784, voucher: 5,775, debit_card: 1,529, not_defined: 3 |
| `payment_installments` | int64 | 0 | 0.0% | 24 | Number of installments chosen (0 to 24) |
| `payment_value` | float64 | 0 | 0.0% | 29,077 | Total amount charged (Min: R$0.00, Max: R$13,664.08) |

---

### 2.5 Order Reviews (`olist_order_reviews_dataset.csv`)
* **Total Rows**: 99,224 | **Total Columns**: 7 | **Orders Represented**: 98,673
* **Orders without Reviews**: 768 | **Multiple Reviews**: Up to 3 reviews per order

| Column Name | Inferred / Actual Data Type | Missing Count | Missing % | Distinct Values | Distribution / Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `review_id` | string (32-char hex UUID) | 0 | 0.0% | 98,410 | Review identifier |
| `order_id` | string (32-char hex UUID) | 0 | 0.0% | 98,673 | Foreign key -> `orders.order_id` |
| `review_score` | int64 | 0 | 0.0% | 5 | 5-star: 57,328 (57.8%), 4-star: 19,142 (19.3%), 3-star: 8,179 (8.2%), 2-star: 3,151 (3.2%), 1-star: 11,424 (11.5%) |
| `review_comment_title` | string | 87,658 | 88.344% | 4,527 | Optional review headline |
| `review_comment_message` | string | 58,274 | 58.730% | 36,159 | Optional customer written review feedback |
| `review_creation_date` | datetime64[ns] | 0 | 0.0% | 687 | Date review survey was generated |
| `review_answer_timestamp` | datetime64[ns] | 0 | 0.0% | 98,248 | Timestamp review was completed |

---

### 2.6 Products (`olist_products_dataset.csv`)
* **Total Rows**: 32,951 | **Total Columns**: 9 | **Unique Products**: 32,951 (100% Unique)

| Column Name | Inferred / Actual Data Type | Missing Count | Missing % | Notes & Distribution |
| :--- | :--- | :---: | :---: | :--- |
| `product_id` | string (32-char hex UUID) | 0 | 0.0% | **Primary Key** |
| `product_category_name` | string / category | 610 | 1.851% | Portuguese category name (73 distinct categories) |
| `product_name_lenght` | float64 | 610 | 1.851% | Character length of product title |
| `product_description_lenght`| float64 | 610 | 1.851% | Character length of product description |
| `product_photos_qty` | float64 | 610 | 1.851% | Number of uploaded product photos |
| `product_weight_g` | float64 | 2 | 0.006% | Product shipping weight in grams |
| `product_length_cm` | float64 | 2 | 0.006% | Product physical dimension (length) |
| `product_height_cm` | float64 | 2 | 0.006% | Product physical dimension (height) |
| `product_width_cm` | float64 | 2 | 0.006% | Product physical dimension (width) |

---

### 2.7 Sellers (`olist_sellers_dataset.csv`)
* **Total Rows**: 3,095 | **Total Columns**: 4 | **Unique Sellers**: 3,095 (100% Unique)

| Column Name | Inferred / Actual Data Type | Missing Count | Missing % | Notes |
| :--- | :--- | :---: | :---: | :--- |
| `seller_id` | string (32-char hex UUID) | 0 | 0.0% | **Primary Key** |
| `seller_zip_code_prefix` | string (5-digit postal) | 0 | 0.0% | 2,246 unique postal prefixes |
| `seller_city` | string | 0 | 0.0% | 611 distinct seller cities |
| `seller_state` | string (2-char UF) | 0 | 0.0% | 23 Brazilian states |

---

### 2.8 Geolocation (`olist_geolocation_dataset.csv`)
* **Total Rows**: 1,000,163 | **Total Columns**: 5 | **Unique Zip Code Prefixes**: 19,015

| Column Name | Inferred / Actual Data Type | Missing Count | Missing % | Notes |
| :--- | :--- | :---: | :---: | :--- |
| `geolocation_zip_code_prefix` | string (5-digit postal) | 0 | 0.0% | 19,015 unique prefixes (Massive multi-point duplication!) |
| `geolocation_lat` | float64 | 0 | 0.0% | Latitude coordinates |
| `geolocation_lng` | float64 | 0 | 0.0% | Longitude coordinates |
| `geolocation_city` | string | 0 | 0.0% | City name variations (contains accents and raw casing) |
| `geolocation_state` | string (2-char UF) | 0 | 0.0% | 27 Brazilian states |

---

### 2.9 Product Category Name Translation (`product_category_name_translation.csv`)
* **Total Rows**: 71 | **Total Columns**: 2 | **Duplicates**: 0

| Column Name | Inferred / Actual Data Type | Missing Count | Missing % | Notes |
| :--- | :--- | :---: | :---: | :--- |
| `product_category_name` | string | 0 | 0.0% | Portuguese category string (71 categories) |
| `product_category_name_english` | string | 0 | 0.0% | English translation string (71 categories) |

---

## 3. Entity Relationships & Cardinality Map

```
  [ customers ] 1 : 1 [ orders ] 1 : N [ order_items ] N : 1 [ products ]
         |                    |                  |                |
         |                    |                  +--- N : 1       +--- N : 1
         |                    |                         |              |
         |                    |                   [ sellers ]   [ category_translation ]
         |                    |
         |                    +-- 1 : N [ order_payments ]
         |                    |
         |                    +-- 1 : N [ order_reviews ]
         |
         +-- N : 1 [ customer_unique_id ] (True Customer Entity)
```

1. **`customers` ↔ `orders`**:
   - `orders.customer_id` → `customers.customer_id` is **1-to-1** (every order has exactly 1 customer session key).
   - `customers.customer_unique_id` → `orders` is **1-to-Many** (96,096 unique individuals generated 99,441 orders).
2. **`orders` ↔ `order_items`**:
   - `order_items.order_id` → `orders.order_id` is **Many-to-1** (up to 21 items in a single order).
   - 775 orders have 0 items (canceled or unavailable before order line allocation).
3. **`orders` ↔ `order_payments`**:
   - `order_payments.order_id` → `orders.order_id` is **Many-to-1** (up to 29 payment records per order).
4. **`orders` ↔ `order_reviews`**:
   - `order_reviews.order_id` → `orders.order_id` is **Many-to-1** (up to 3 reviews recorded for a single order).
5. **`order_items` ↔ `products`**:
   - 100% referential integrity: All 32,951 product IDs in order items exist in `olist_products_dataset.csv`.
6. **`order_items` ↔ `sellers`**:
   - 100% referential integrity: All 3,095 seller IDs in order items exist in `olist_sellers_dataset.csv`.

---

## 4. Critical Data Quality Issues & Architectural Safeguards

1. **The Cartesian Join Trap (Revenue Inflation Risk)**:
   - Orders can have multiple items (1 to 21) AND multiple payment lines (1 to 29).
   - Joining `order_items` and `order_payments` directly on `order_id` creates a cross-product (e.g., 3 items × 4 payments = 12 joined rows). Summing values across this joined table causes severe artificial revenue multiplication.
   - *Safeguard*: Aggregate to the `order_id` grain before joining, or calculate item-level metrics and payment-level metrics on their native tables.
2. **Customer Identity Distinction (`customer_id` vs `customer_unique_id`)**:
   - `customer_id` is an order transaction token (99,441 unique values for 99,441 orders).
   - Grouping on `customer_id` would mistakenly show a 0.0% repeat purchase rate.
   - *Safeguard*: All customer analytics, RFM, and repeat-purchase modeling MUST group on `customer_unique_id`.
3. **Severe Class Imbalance in Repeat Purchase Target**:
   - Exactly 2,997 customers out of 96,096 made repeated orders (3.12% repeat rate).
   - Raw accuracy is misleading (a naive dummy classifier predicting all 0s achieves 96.88% accuracy).
   - *Safeguard*: Evaluation must rely on ROC-AUC, PR-AUC, Precision, Recall, and F1-Score with balanced class weighting.
4. **Translation Gaps in Product Categories**:
   - 2 categories present in `olist_products_dataset.csv` are absent in `product_category_name_translation.csv`:
     - `pc_gamer`
     - `portateis_cozinha_e_preparadores_de_alimentos`
   - *Safeguard*: Implement an explicit mapping dictionary during cleaning:
     - `pc_gamer` → `pc_gamer`
     - `portateis_cozinha_e_preparadores_de_alimentos` → `small_appliances_food_preparers`
5. **Missing Product Metadata (610 Products)**:
   - 610 products lack categories, titles, descriptions, and photo counts.
   - *Safeguard*: Impute missing category with `'unknown'` / `'uncategorized'` to preserve historical sales records.
6. **Undefined Payment Types & Zero-Value Payments**:
   - 3 rows have `payment_type == 'not_defined'`, and voucher records contain zero values.
   - *Safeguard*: Impute or filter non-defined payments; audit voucher deductions.
7. **Geolocation Combinatorial Explosion**:
   - `olist_geolocation_dataset.csv` contains 1,000,163 rows for only 19,015 zip prefixes.
   - *Safeguard*: Pre-aggregate geolocation to the zip prefix level (centroid coordinates: mean/median latitude and longitude) prior to joining.
8. **Time-Based Splitting & Anti-Leakage Protocol**:
   - Historical dataset spans September 2016 to October 2018.
   - Customer features must be computed strictly prior to the observation cutoff date.

---

## 5. Recommended Project Architecture

```
ecommerce-customer-intelligence/
│
├── data/
│   ├── raw/                              # 9 untouched raw Olist CSV files
│   └── processed/                        # Cleaned and integrated analytical tables
│       ├── cleaned_customers.csv
│       ├── cleaned_orders.csv
│       ├── cleaned_order_items.csv
│       ├── cleaned_order_payments.csv
│       ├── cleaned_order_reviews.csv
│       ├── cleaned_products.csv
│       ├── cleaned_sellers.csv
│       ├── customer_rfm_segments.csv
│       ├── customer_prediction_dataset.csv
│       └── analytical_order_master.csv
│
├── src/                                  # Modular, tested production code
│   ├── __init__.py
│   ├── data_loader.py                    # Schema validation and data loader
│   ├── data_cleaning.py                  # Standardized cleaning pipeline
│   ├── data_integration.py               # Safe analytical joins (anti-duplication)
│   ├── rfm.py                            # Customer-level RFM scoring
│   ├── feature_engineering.py            # Pre-cutoff time-based customer features
│   ├── modeling.py                       # ML models (LogReg, Decision Tree, RF)
│   ├── evaluation.py                     # ROC-AUC, PR-AUC, Confusion Matrix
│   ├── interpretability.py               # Feature importance & coefficient analysis
│   └── recommendations.py                # Rule-based business recommendation engine
│
├── models/                               # Serialized models and transformers
│   ├── best_repeat_purchase_model.joblib
│   ├── feature_preprocessor.joblib
│   └── model_metadata.json
│
├── dashboard/                            # Interactive multi-page Streamlit application
│   ├── app.py                            # Executive Dashboard & navigation
│   └── pages/                            # Customer Intelligence, RFM, ML Predictor, Products, Recommendations
│
├── reports/                              # Automated audits, inspection logs & benchmark summaries
├── visualizations/                       # Exported publication-ready plots
├── requirements.txt                      # Project dependencies
├── README.md                             # Comprehensive portfolio documentation
└── .gitignore                            # Standard Python & data exclusions
```
