"""
=============================================================================
PAGE: 1_Dashboard.py
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

from src.data_loader import load_dataset, get_dataset_summary, DATASET_METADATA
from src.visualization import (
    plot_monthly_revenue_trend,
    plot_category_revenue,
    plot_region_revenue,
    plot_subcategory_revenue,
)

st.set_page_config(page_title="Executive Dashboard", page_icon="📊", layout="wide")

st.title("📊 Retail Executive Dashboard")
st.markdown("High-level performance monitoring, revenue trends, and operational breakdown.")

try:
    df = load_dataset()
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")
    st.stop()

summary = get_dataset_summary(df)

# Top KPI Metric Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Transactions", f"{summary['num_rows']:,}")
kpi2.metric("Total Gross Revenue", f"${summary['total_revenue']:,.2f}")
kpi3.metric("Average Order Value", f"${summary['average_revenue']:.2f}")
kpi4.metric("Product Categories", f"{df['Category'].nunique()} Cats / {df['Sub-Category'].nunique()} Subcats")

st.markdown("---")

col_left, col_right = st.columns([7, 5])
with col_left:
    st.plotly_chart(plot_monthly_revenue_trend(df), use_container_width=True)
with col_right:
    st.plotly_chart(plot_region_revenue(df), use_container_width=True)

st.markdown("---")

col_sub1, col_sub2 = st.columns([6, 6])
with col_sub1:
    st.plotly_chart(plot_category_revenue(df), use_container_width=True)
with col_sub2:
    st.plotly_chart(plot_subcategory_revenue(df), use_container_width=True)

st.markdown("---")
st.markdown("### 📋 Recent Retail Transactions")
st.dataframe(
    df[["Order Date", "Category", "Sub-Category", "Region", "Segment", "Discount", "Sales"]].head(15),
    use_container_width=True,
)
