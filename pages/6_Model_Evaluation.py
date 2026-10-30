"""
=============================================================================
PAGE: 6_Model_Evaluation.py
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
from src.train_models import run_full_training_pipeline
from src.evaluation import create_train_test_comparison_table
from src.visualization import (
    plot_train_test_metrics_comparison,
    plot_actual_vs_predicted,
    plot_residuals_distribution,
    plot_residuals_vs_predicted,
)

st.set_page_config(page_title="Model Evaluation & Diagnostics", page_icon="⚖️", layout="wide")

st.title("⚖️ Model Evaluation & Statistical Diagnostics")
st.markdown(
    "Comprehensive empirical evaluation of the **Linear Regression** model: auditing "
    "Train vs. Test generalization, goodness-of-fit, and classical regression assumptions."
)

try:
    df = load_dataset()
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")
    st.stop()

# Ensure trained results exist
if "training_results" not in st.session_state:
    with st.spinner("Executing training and evaluation pipeline on chronological splits..."):
        st.session_state["training_results"] = run_full_training_pipeline(df)

results = st.session_state["training_results"]
train_metrics = results["train_metrics"]
test_metrics = results["test_metrics"]
y_test = results["y_test"]
lr_diag = results["lr_diagnostics"]

# Section 1: Train vs Test Generalization Matrix
st.subheader("📊 Generalization Audit: Training vs. Unseen Testing Performance")
st.markdown(
    """
    Comparing training set metrics ($N_{\text{train}} = 8,640$) against unseen test set metrics 
    ($N_{\text{test}} = 2,160$) allows us to objectively audit the bias-variance tradeoff:
    - **Close Train and Test RMSE:** Indicates that the linear model generalizes consistently without overfitting to historical training noise.
    """
)

comp_df = create_train_test_comparison_table(train_metrics, test_metrics)
st.dataframe(comp_df, use_container_width=True)

st.markdown("---")

# Section 2: Visual Comparison Chart
col_bar, col_diag_summary = st.columns([7, 5])

with col_bar:
    st.plotly_chart(plot_train_test_metrics_comparison(train_metrics, test_metrics), use_container_width=True)

with col_diag_summary:
    st.markdown(
        r"""
        ##### 🔍 Bias-Variance Assessment:
        - **Training RMSE:** \${:,.2f} | **Testing RMSE:** \${:,.2f}
        - **Training R²:** {:.4f} | **Testing R²:** {:.4f}
        - **Observation:** Notice that test RMSE is very close to training RMSE. This confirms that Ordinary Least Squares Linear Regression achieves **low variance** (it does not memorize individual training anomalies) but operates with **moderate bias** (the true retail demand function has non-linear pricing effects).
        """.format(
            train_metrics["RMSE"], test_metrics["RMSE"],
            train_metrics["R2"], test_metrics["R2"]
        )
    )

st.markdown("---")

# Section 3: Actual vs Predicted Revenue (Fit Diagnostics)
st.subheader("🎯 Actual vs. Predicted Revenue (Goodness-of-Fit)")
st.markdown(
    """
    Points falling directly on the red dashed line ($y = x$) represent perfect predictions.
    Points above the line indicate under-prediction; points below indicate over-prediction.
    """
)

col_avp1, col_avp2 = st.columns([7, 5])
with col_avp1:
    st.plotly_chart(
        plot_actual_vs_predicted(y_test, lr_diag["y_test_pred"], "Linear Regression"),
        use_container_width=True,
    )

with col_avp2:
    st.markdown(
        r"""
        ##### 📊 Goodness-of-Fit Takeaways:
        - **Linear Projection:** The model successfully learns the central revenue regime across standard retail transactions ($\le \$500$).
        - **Right-Tail Compression:** Because OLS minimizes squared errors globally, extreme high-value transactions (\$2,000+) are pulled toward the mean to avoid destabilizing the global hyperplane.
        """
    )

st.markdown("---")

# Section 4: Residual Diagnostics & Assumption Verification
st.subheader("📉 Residual Error Diagnostics ($e_i = y_i - \\hat{y}_i$)")
st.markdown(
    """
    Classical Linear Regression assumes that error terms are normally distributed with zero mean 
    and constant variance (**Homoscedasticity**).
    """
)

res_col1, res_col2 = st.columns(2)

with res_col1:
    st.plotly_chart(
        plot_residuals_distribution(lr_diag["test_residuals"], "Linear Regression"),
        use_container_width=True,
    )
    st.caption("Distribution of test residuals. Notice the sharp centering around zero ($e \approx 0$).")

with res_col2:
    st.plotly_chart(
        plot_residuals_vs_predicted(lr_diag["y_test_pred"], lr_diag["test_residuals"]),
        use_container_width=True,
    )
    st.caption("Residuals vs. Fitted Values. Tests homoscedasticity (constant error spread across prediction magnitudes).")

st.markdown("---")

# Section 5: In-Depth Metric Glossary & Formula Reference
with st.expander("📚 In-Depth Academic Regression Formulas & Viva Concepts"):
    st.markdown(
        r"""
        ##### 1. Mean Absolute Error (MAE):
        $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
        - Measures average absolute deviation in original dollar units ($).
        - Robust against extreme outliers because error growth is strictly linear ($|e|$).

        ##### 2. Mean Squared Error (MSE):
        $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
        - Heavily penalizes large forecasting blunders because errors are squared ($e^2$).
        - Expressed in squared dollars ($\$^2$).

        ##### 3. Root Mean Squared Error (RMSE):
        $$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
        - Expressed back in dollar units ($) while preserving the quadratic penalty of MSE.
        - **Primary evaluation metric** in retail revenue forecasting where large misses cause severe supply-chain disruptions.

        ##### 4. R-Squared ($R^2$) Score:
        $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
        - Proportion of target variance explained by the model relative to a naive mean predictor.
        - *Warning:* $R^2$ is **NOT** percentage accuracy. On test data, $R^2$ can be negative if the model predicts worse than the simple horizontal mean line $\bar{y}$.

        ##### 5. Mean Absolute Percentage Error (MAPE):
        $$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$
        - Intuitive business metric representing average percentage error.
        - *Limitation:* Undefined or explosive when actual revenue $y_i = 0$ or approaches zero (handled with numerical epsilon safeguards).
        """
    )
