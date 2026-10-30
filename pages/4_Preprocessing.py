"""
=============================================================================
PAGE: 4_Preprocessing.py
PROJECT: Retail Sales Revenue Prediction System
COURSEWORK: Third-Year B.Tech AI & ML - Predictive Modelling
=============================================================================
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_dataset
from src.preprocessing import detect_outliers_iqr, build_preprocessor, chronological_train_test_split
from src.feature_engineering import prepare_modeling_data, CATEGORICAL_FEATURES, NUMERICAL_FEATURES

st.set_page_config(page_title="Data Preprocessing Pipeline", page_icon="⚙️", layout="wide")

st.title("⚙️ End-to-End Data Preprocessing Pipeline")
st.markdown(
    "A transparent, educational breakdown of every transformation applied to raw retail data "
    "before feeding into Linear Regression and Random Forest Regressors."
)

try:
    df = load_dataset()
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")
    st.stop()

# Step 1: Missing Value Handling
st.markdown("### Step 1: Missing Value Detection & Imputation Strategy")
col_m1, col_m2 = st.columns([5, 7])

missing_series = df.isnull().sum()
missing_df = pd.DataFrame({"Feature": missing_series.index, "Missing Count": missing_series.values})

with col_m1:
    st.dataframe(missing_df, use_container_width=True)

with col_m2:
    st.markdown(
        """
        ##### 📚 Imputation Rationale:
        - **Numerical Features:** Imputed using **Median** (`strategy='median'`).
          *Why?* The median is non-parametric and resistant to extreme skewness or outliers, unlike the sample mean which is pulled toward heavy tails.
        - **Categorical Features:** Imputed using **Most Frequent / Mode** (`strategy='most_frequent'`).
          *Why?* For discrete classes like `Region` or `Category`, the mode represents the most probable baseline category.
        - **Pipeline Integration:** Imputers are fitted strictly on training data inside Scikit-learn's `ColumnTransformer` to prevent leakage.
        """
    )

st.markdown("---")

# Step 2: Duplicate Handling
st.markdown("### Step 2: Duplicate Record Auditing")
num_duplicates = int(df.duplicated().sum())

dup_col1, dup_col2 = st.columns([4, 8])
with dup_col1:
    st.metric("Identified Duplicate Rows", f"{num_duplicates}")
with dup_col2:
    st.markdown(
        """
        ##### 📚 Business Impact of Duplicates:
        - Exact duplicate rows artificially inflate sample weights for specific transactions, biasing model parameters toward over-represented order profiles.
        - In this dataset, transactions have unique order IDs and line-item identifiers; verified clean with 0 duplicate rows.
        """
    )

st.markdown("---")

# Step 3: Outlier Detection via Interquartile Range (IQR)
st.markdown("### Step 3: Outlier Analysis via Interquartile Range (IQR)")
outlier_info = detect_outliers_iqr(df["Sales"])

iqr_c1, iqr_c2, iqr_c3, iqr_c4 = st.columns(4)
iqr_c1.metric("Q1 (25th Percentile)", f"${outlier_info['q1']:.2f}")
iqr_c2.metric("Q3 (75th Percentile)", f"${outlier_info['q3']:.2f}")
iqr_c3.metric("IQR (Q3 - Q1)", f"${outlier_info['iqr']:.2f}")
iqr_c4.metric("Upper Outlier Fence", f"${outlier_info['upper_bound']:.2f}")

st.markdown(
    f"""
    > **Outlier Statistics:** Found **{outlier_info['outlier_count']:,}** observations ({outlier_info['outlier_percentage']}%) exceeding the upper threshold of **${outlier_info['upper_bound']:.2f}**.
    
    ##### ⚠️ Critical Engineering Decision: Why We DO NOT Blindly Discard Outliers:
    In enterprise retail and e-commerce, high-value transactions (e.g. bulk office furnishing orders or enterprise copy machines worth $2,000+) are **authentic, legitimate revenue events**, not measurement sensor errors. Discarding them would blind the forecasting model to high-value revenue streams. Instead, Random Forest naturally isolates extreme values in specialized leaf partitions, while StandardScaler helps Linear Regression maintain gradient stability.
    """
)

st.markdown("---")

# Step 4: Feature Engineering
st.markdown("### Step 4: Predictive Feature Engineering (Zero-Leakage)")
X, y, benchmarks = prepare_modeling_data(df, is_training=True)

st.markdown("##### Engineered Feature Matrix (First 5 Rows):")
st.dataframe(X.head(5), use_container_width=True)

st.markdown(
    """
    ##### 📚 Engineered Feature Breakdown:
    1. **Calendar Periodicity:** `Year`, `Month`, `Day`, `DayOfWeek`, `Quarter`, `IsWeekend`, `Season` (Winter, Spring, Summer, Fall). Enables models to detect annual seasonality, weekend shopping habits, and corporate fiscal closing cycles.
    2. **Historical Sub-Category Demand Benchmark (`Hist_SubCat_Avg_Sales`):**
       - Calculated chronologically using an **expanding mean shifted by 1** (`shift(1)`).
       - At order $t$, the feature represents historical average revenue for that subcategory strictly *prior* to order $t$.
       - **Zero Look-Ahead Leakage:** Order $t$ never sees order $t$'s actual sales!
    3. **Leakage Prevention (`Quantity` and `Unit Price` Exclusion):**
       - In retail accounting: $Sales = Unit\\ Price \\times Quantity \\times (1 - Discount)$.
       - Providing both realized quantity and price simultaneously creates a mathematically trivial identity (target leakage). Thus, our system predicts gross revenue from category, season, customer segment, shipping speed, region, discount, and historical demand.
    """
)

st.markdown("---")

# Step 5: Encoding & Scaling via ColumnTransformer
st.markdown("### Step 5: Categorical Encoding & Feature Scaling Pipeline")
col_enc1, col_enc2 = st.columns(2)

with col_enc1:
    st.markdown(
        """
        ##### 🔠 Categorical One-Hot Encoding:
        - Applied to: `Category`, `Sub-Category`, `Segment`, `Region`, `Ship Mode`, `Season`.
        - Uses `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
        - Converts $k$ discrete string categories into $k$ orthogonal binary columns $[0, 1]$.
        - Avoids imposing an artificial ordinal ranking on unordered nominal labels (e.g., 'West' is not 'greater' than 'East').
        """
    )

with col_enc2:
    st.markdown(
        """
        ##### 📏 Continuous Feature Standardization (StandardScaler):
        - Applied to continuous features: $z = \\frac{x - \\mu}{\\sigma}$.
        - Centers feature values around $\\mu = 0$ with variance $\\sigma^2 = 1$.
        - **Why it matters for Linear Regression:** Ensures the OLS matrix $(X^T X)^{-1} X^T y$ is well-conditioned, avoids numerical instability, and puts coefficients on a standardized scale for direct magnitude comparison.
        - **Why Random Forest is Scale-Invariant:** Decision trees split solely on ranking thresholds ($x_j \\le s$), making them completely invariant to monotonic transformations.
        """
    )

st.markdown("---")

# Step 6: Chronological Train-Test Splitting
st.markdown("### Step 6: Chronological Train-Test Partitioning")
X_train, X_test, y_train, y_test = chronological_train_test_split(X, y, train_ratio=0.8)

split_c1, split_c2, split_c3 = st.columns(3)
split_c1.metric("Training Set (Past 80%)", f"{len(X_train):,} orders")
split_c2.metric("Testing Set (Future 20%)", f"{len(X_test):,} orders")
split_c3.metric("Splitting Method", "Strictly Chronological")

st.markdown(
    """
    ##### ⏳ Why Chronological Splitting is Mandatory in Forecasting:
    - Standard machine learning benchmarks often perform random shuffling (`train_test_split(shuffle=True)`).
    - In sales forecasting, random shuffling causes **temporal data leakage**: the model learns from orders placed in December 2017 to predict orders placed in February 2017 (peeking into the future!).
    - Chronological splitting enforces the realistic business task: **train on historical past quarters, test on unobserved future quarters.**
    """
)
