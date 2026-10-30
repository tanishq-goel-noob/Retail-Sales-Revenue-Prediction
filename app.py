"""
=============================================================================
APPLICATION: app.py
PROJECT: Retail Sales Revenue Prediction System
COURSEWORK: Third-Year B.Tech AI & ML - Predictive Modelling
=============================================================================

This is the main entry point and Executive Dashboard for the Retail Sales
Revenue Prediction System. It provides an intuitive, high-level overview of
business performance, historical revenue trends, and machine learning workflow
powered by Linear Regression (Ordinary Least Squares).
=============================================================================
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Add project root to sys.path for robust module discovery
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_dataset, get_dataset_summary
from src.train_models import run_full_training_pipeline, MODELS_DIR
from src.visualization import (
    plot_monthly_revenue_trend,
    plot_category_revenue,
)

# Configure Streamlit page settings
st.set_page_config(
    page_title="Retail Sales Revenue Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished, executive appearance
st.markdown(
    """
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #3B82F6;
        margin-top: 0.3rem;
        font-weight: 500;
    }
    /* Banner Styling */
    .dashboard-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #1E40AF 100%);
        color: white;
        padding: 2rem;
        border-radius: 14px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.15);
    }
    .dashboard-header h1 {
        color: white !important;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .dashboard-header p {
        color: #E0E7FF;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading retail sales dataset...")
def get_data():
    """Loads and caches the retail sales dataset."""
    return load_dataset()


def ensure_models_trained(df: pd.DataFrame):
    """Ensures serialized Linear Regression model exists on disk, training if not present."""
    lr_file = MODELS_DIR / "linear_regression.joblib"
    meta_file = MODELS_DIR / "model_metadata.joblib"

    if not (lr_file.exists() and meta_file.exists()):
        with st.spinner("Initializing pipeline... Fitting Linear Regression model..."):
            run_full_training_pipeline(df)


def main():
    # Load dataset
    try:
        df = get_data()
    except Exception as e:
        st.error(f"Failed to load dataset: {str(e)}")
        st.stop()

    # Ensure model is trained
    ensure_models_trained(df)

    # Main Dashboard Header
    st.markdown(
        """
        <div class="dashboard-header">
            <h1>Retail Sales Revenue Prediction System</h1>
            <p>Predictive Modelling and Revenue Forecasting Using Ordinary Least Squares (OLS) Linear Regression</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dataset Summary Metrics
    summary = get_dataset_summary(df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Transactions</div>
                <div class="metric-value">{summary['num_rows']:,}</div>
                <div class="metric-sub">✓ Valid line-item orders</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Realized Revenue</div>
                <div class="metric-value">${summary['total_revenue']:,.0f}</div>
                <div class="metric-sub">✓ Across 4 US Regions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Average Order Revenue</div>
                <div class="metric-value">${summary['average_revenue']:.2f}</div>
                <div class="metric-sub">Median: ${summary['median_revenue']:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        subcats_count = df["Sub-Category"].nunique() if "Sub-Category" in df.columns else 0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Product Categories</div>
                <div class="metric-value">{subcats_count}</div>
                <div class="metric-sub">Across 3 Super-Categories</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Key Analytics Visualizations
    st.markdown("### 📈 Historical Revenue Dynamics")
    row1_col1, row1_col2 = st.columns([7, 5])

    with row1_col1:
        st.plotly_chart(plot_monthly_revenue_trend(df), use_container_width=True)

    with row1_col2:
        st.plotly_chart(plot_category_revenue(df), use_container_width=True)

    st.markdown("---")




if __name__ == "__main__":
    main()
