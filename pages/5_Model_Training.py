"""
=============================================================================
PAGE: 5_Model_Training.py
PROJECT: Retail Sales Revenue Prediction System
COURSEWORK: Third-Year B.Tech AI & ML - Predictive Modelling
=============================================================================
"""

import sys
import time
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_dataset
from src.train_models import run_full_training_pipeline
from src.visualization import plot_linear_coefficients

st.set_page_config(page_title="Model Training", page_icon="🧠", layout="wide")

st.title("🧠 Linear Regression Model Training & OLS Diagnostics")
st.markdown(
    "Configure train/test partitions, train the **Ordinary Least Squares (OLS) Linear Regression** "
    "pipeline, and inspect the learned mathematical weights (Beta coefficients)."
)

try:
    df = load_dataset()
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")
    st.stop()

# Training Configuration Sidebar
st.sidebar.markdown("### ⚙️ Training Parameters")
train_split = st.sidebar.slider("Chronological Train Split Ratio", min_value=0.65, max_value=0.85, value=0.80, step=0.05)
fit_intercept = st.sidebar.checkbox("Fit Intercept (β₀)", value=True)

train_btn = st.sidebar.button("🚀 Train Linear Regression Pipeline", use_container_width=True)

# Session state initialization for trained results
if "training_results" not in st.session_state or train_btn:
    with st.spinner("Training Ordinary Least Squares Linear Regression pipeline..."):
        start_t = time.time()
        results = run_full_training_pipeline(
            df=df,
            train_ratio=train_split,
        )
        elapsed = round(time.time() - start_t, 2)
        st.session_state["training_results"] = results
        st.session_state["training_time"] = elapsed
        st.success(f"✓ Linear Regression pipeline successfully trained in {elapsed} seconds!")

results = st.session_state["training_results"]
lr_diag = results["lr_diagnostics"]
test_metrics = results["test_metrics"]
train_metrics = results["train_metrics"]

# Section 1: Mathematical Foundations
st.markdown("### 1. Mathematical Formulation & Normal Equation")
st.markdown(
    r"""
    ##### 📐 Ordinary Least Squares (OLS) Regression:
    Linear Regression models sales revenue $Y$ as a linear combination of $p$ standardized input features:
    $$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p + \epsilon$$
    
    In matrix form:
    $$\mathbf{y} = \mathbf{X}\mathbf{\beta} + \mathbf{\epsilon}$$

    - **Intercept ($\beta_0$):** Baseline expected revenue when all standardized continuous features equal 0.
    - **Regression Coefficients ($\beta_j$):** The rate of change in predicted revenue per one-standard-deviation change in feature $X_j$, holding all other features strictly constant (*ceteris paribus*).
    - **Closed-Form Analytical Solution (Normal Equation):** OLS minimizes the Residual Sum of Squares (RSS):
      $$\hat{\mathbf{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$
    """
)

st.markdown(f"**Fitted Intercept ($\beta_0$):** `${lr_diag['intercept']:.2f}`")

st.markdown("---")

# Section 2: Model Performance Metrics
st.markdown("### 2. Empirical Performance Metrics on Unseen Test Split")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Test MAE", f"${test_metrics['MAE']:,.2f}", help="Average absolute dollar error")
m2.metric("Test MSE", f"{test_metrics['MSE']:,.1f}", help="Mean squared error")
m3.metric("Test RMSE", f"${test_metrics['RMSE']:,.2f}", help="Root mean squared dollar error (Primary)")
m4.metric("Test R²", f"{test_metrics['R2']:.4f}", help="Variance explained relative to mean baseline")
m5.metric("Test MAPE", f"{test_metrics['MAPE']:.1f}%", help="Mean absolute percentage error")

st.markdown("---")

# Section 3: Learned Coefficients Inspection
st.markdown("### 3. Learned Regression Feature Weights (Beta Coefficients)")
col_c1, col_c2 = st.columns([7, 5])

with col_c1:
    st.plotly_chart(plot_linear_coefficients(lr_diag["coefficients_df"], top_n=15), use_container_width=True)

with col_c2:
    st.markdown("##### Top Significant Regression Weights:")
    st.dataframe(lr_diag["coefficients_df"].head(12)[["Feature", "Coefficient", "Abs_Coefficient"]], use_container_width=True)
    st.markdown(
        """
        ##### 💡 Interpretation of Coefficients:
        - **Positive Coefficients (Blue):** Features such as high-revenue subcategories (`Copiers`, `Machines`, `Phones`) and historical demand benchmarks increase predicted sales.
        - **Negative Coefficients (Red):** Promotional `Discount` and lower-value office supply categories exhibit negative coefficients, exerting downward pressure on per-order dollar revenue.
        """
    )
