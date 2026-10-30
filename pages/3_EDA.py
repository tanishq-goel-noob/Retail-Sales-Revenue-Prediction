"""
=============================================================================
PAGE: 3_EDA.py
PROJECT: Retail Sales Revenue Prediction System
COURSEWORK: Third-Year B.Tech AI & ML - Predictive Modelling
=============================================================================
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_dataset
from src.visualization import (
    plot_revenue_distribution,
    plot_monthly_revenue_trend,
    plot_category_revenue,
    plot_subcategory_revenue,
    plot_region_revenue,
    plot_discount_vs_revenue,
    plot_correlation_heatmap,
    plot_outlier_boxplot,
)

st.set_page_config(page_title="Exploratory Data Analysis", page_icon="📉", layout="wide")

st.title("📉 Exploratory Data Analysis (EDA)")
st.markdown(
    "Rigorous visual exploration of sales revenue patterns, promotional discounts, "
    "temporal seasonality, and feature correlations."
)

try:
    df = load_dataset()
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")
    st.stop()

# Section 1: Target Variable Distribution
st.markdown("### 1. Target Revenue Distribution & Skewness")
col_dist1, col_dist2 = st.columns([7, 5])

with col_dist1:
    st.plotly_chart(plot_revenue_distribution(df), use_container_width=True)

with col_dist2:
    st.markdown(
        r"""
        ##### 📊 Statistical & Business Interpretation:
        - **Right-Skewed Distribution:** Retail revenue typically exhibits heavy positive skewness ($mean > median$). The vast majority of sales occur under $\$300$, while high-ticket technology and executive furniture create a long right-tail of high monetary value.
        - **Impact on Modelling:**
          - **Linear Regression (OLS):** Highly sensitive to extreme right-tail values because squaring residuals heavily penalizes large errors, potentially pulling the regression hyperplane toward outliers.
          - **Random Forest:** Partitioning cuts isolate extreme values into specialized leaf nodes without distorting predictions for mainstream orders.
        """
    )

st.markdown("---")

# Section 2: Temporal Revenue Trends
st.markdown("### 2. Chronological Monthly Revenue Trends & Seasonality")
st.plotly_chart(plot_monthly_revenue_trend(df), use_container_width=True)
st.markdown(
    """
    ##### 📊 Statistical & Business Interpretation:
    - **Q4 Surge (Annual Seasonality):** Notice consistent revenue spikes peaking in November and December of every historical year. This aligns with corporate budget flush cycles and holiday retail demand (Black Friday, Cyber Monday).
    - **Long-term Growth:** The overall trend across consecutive years shows compound top-line expansion.
    - **Feature Engineering Decision:** Month, Quarter, and Season features are essential to allow both Linear Regression and Random Forest to learn recurring seasonal shifts.
    """
)

st.markdown("---")

# Section 3: Product Category Breakdown
st.markdown("### 3. Product Taxonomy: Category & Sub-Category Performance")
col_cat1, col_cat2 = st.columns([5, 7])

with col_cat1:
    st.plotly_chart(plot_category_revenue(df), use_container_width=True)
    st.caption("Technology and Furniture generate the highest aggregate revenues due to high unit price tags.")

with col_cat2:
    st.plotly_chart(plot_subcategory_revenue(df), use_container_width=True)
    st.caption("Phones, Chairs, Storage, and Tables constitute over 50% of total commercial gross sales.")

st.markdown("---")

# Section 4: Geographic & Promotional Dynamics
st.markdown("### 4. Geographic Distribution & Promotional Discount Elasticity")
col_geo, col_disc = st.columns([5, 7])

with col_geo:
    st.plotly_chart(plot_region_revenue(df), use_container_width=True)
    st.markdown(
        """
        ##### Regional Dynamics:
        - West and East regions dominate corporate revenue capture (~60% combined).
        - South and Central exhibit lower transaction volumes and varying average order values.
        """
    )

with col_disc:
    st.plotly_chart(plot_discount_vs_revenue(df), use_container_width=True)
    st.markdown(
        r"""
        ##### Promotional Discount vs Revenue:
        - Deep discounts ($> 40\%$) are concentrated on lower-revenue transactions or clearance items.
        - High-value technology items are rarely discounted heavily ($0\% - 20\%$), preserving dollar margins.
        """
    )

st.markdown("---")

# Section 5: Feature Correlations & Outliers
st.markdown("### 5. Correlation Heatmap & Outlier Analysis")
col_corr, col_box = st.columns([6, 6])

with col_corr:
    st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)
    st.caption("Linear correlations between continuous transactional variables.")

with col_box:
    st.plotly_chart(plot_outlier_boxplot(df, column="Sales"), use_container_width=True)
    st.caption("IQR Outlier visualization for gross Sales revenue ($ USD).")
