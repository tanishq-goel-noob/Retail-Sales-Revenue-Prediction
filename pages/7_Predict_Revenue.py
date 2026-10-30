"""
=============================================================================
PAGE: 7_Predict_Revenue.py
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
from src.prediction import RetailSalesPredictor
from src.train_models import run_full_training_pipeline

st.set_page_config(page_title="Predict Revenue", page_icon="🔮", layout="wide")

st.title("🔮 Interactive Retail Sales Revenue Predictor")
st.markdown(
    "Forecast future retail transaction revenue for a planned order using our "
    "pre-trained, leakage-free **Linear Regression (Ordinary Least Squares)** pipeline."
)

# Initialize predictor wrapper
predictor = RetailSalesPredictor()

# Auto-train if model does not exist yet
if not predictor.is_ready():
    with st.spinner("Training Linear Regression pipeline..."):
        df_raw = load_dataset()
        run_full_training_pipeline(df_raw)
        predictor = RetailSalesPredictor()

# Preset Sub-Category to Super-Category Mapping
SUBCAT_CATEGORY_MAP = {
    "Bookcases": "Furniture",
    "Chairs": "Furniture",
    "Furnishings": "Furniture",
    "Tables": "Furniture",
    "Appliances": "Office Supplies",
    "Art": "Office Supplies",
    "Binders": "Office Supplies",
    "Envelopes": "Office Supplies",
    "Fasteners": "Office Supplies",
    "Labels": "Office Supplies",
    "Paper": "Office Supplies",
    "Storage": "Office Supplies",
    "Supplies": "Office Supplies",
    "Accessories": "Technology",
    "Copiers": "Technology",
    "Machines": "Technology",
    "Phones": "Technology",
}

# Prediction Form
st.subheader("📝 Order & Market Specifications")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        order_date = st.date_input("Target Order / Delivery Date", value=pd.Timestamp.now().date())
        subcat_list = sorted(list(SUBCAT_CATEGORY_MAP.keys()))
        selected_subcat = st.selectbox("Product Sub-Category", options=subcat_list, index=subcat_list.index("Phones"))
        inferred_category = SUBCAT_CATEGORY_MAP[selected_subcat]
        st.info(f"📁 Inferred Super-Category: **{inferred_category}**")

        segment = st.selectbox("Customer Segment", ["Consumer", "Corporate", "Home Office"], index=0)

    with col2:
        region = st.selectbox("Geographic Region", ["West", "East", "Central", "South"], index=0)
        ship_mode = st.selectbox("Shipping Mode", ["Standard Class", "Second Class", "First Class", "Same Day"], index=0)
        discount = st.slider("Planned Promotional Discount (%)", min_value=0, max_value=80, value=10, step=5) / 100.0

        # Historical benchmark lookup
        default_bm = predictor.benchmarks.get(selected_subcat, 250.0)
        use_custom_bm = st.checkbox("Override historical subcategory revenue benchmark?")
        if use_custom_bm:
            custom_bm = st.number_input(
                "Historical Subcategory Benchmark Revenue ($ USD)",
                min_value=1.0,
                max_value=10000.0,
                value=float(default_bm),
            )
        else:
            custom_bm = float(default_bm)
            st.caption(f"Historical Demand Benchmark for {selected_subcat}: **${default_bm:.2f}**")

    submit_btn = st.form_submit_button("🚀 Generate Linear Regression Revenue Forecast", use_container_width=True)

if submit_btn:
    payload = {
        "order_date": order_date,
        "category": inferred_category,
        "subcategory": selected_subcat,
        "segment": segment,
        "region": region,
        "ship_mode": ship_mode,
        "discount": discount,
        "custom_benchmark": custom_bm,
    }

    try:
        prediction_result = predictor.predict(payload)

        st.markdown("---")
        st.subheader("🎯 Linear Regression Forecast Results")

        res_col1, res_col2 = st.columns([6, 6])

        with res_col1:
            st.markdown(
                f"""
                <div style="background-color: #FFFFFF; border: 2px solid #3B82F6;
                            border-radius: 12px; padding: 1.5rem; text-align: center;
                            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
                    <div style="font-size: 0.95rem; font-weight: 700; color: #1E3A8A; text-transform: uppercase;">
                        Forecasted Order Revenue
                    </div>
                    <div style="font-size: 2.5rem; font-weight: 800; color: #0F172A; margin: 0.5rem 0;">
                        ${prediction_result['predicted_revenue']:,.2f}
                    </div>
                    <div style="font-size: 0.85rem; color: #64748B;">
                        Raw OLS Solution: ${prediction_result['raw_ols_prediction']:,.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col2:
            st.markdown(
                f"""
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0;
                            border-radius: 12px; padding: 1.5rem; text-align: center;">
                    <div style="font-size: 0.95rem; font-weight: 600; color: #64748B; text-transform: uppercase;">
                        Expected 1-Standard-Error Range (±RMSE)
                    </div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #10B981; margin: 0.7rem 0;">
                        ${prediction_result['lower_bound_estimate']:,.2f} — ${prediction_result['upper_bound_estimate']:,.2f}
                    </div>
                    <div style="font-size: 0.85rem; color: #64748B;">
                        Based on Test RMSE of ${prediction_result['test_rmse']:,.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Executive Recommendation Box
        st.success(
            f"**Forecast Summary:** {prediction_result['summary']}\n\n"
            f"💡 **Promotional Pricing Insight:** {prediction_result['discount_note']}"
        )

        with st.expander("🔍 Inspect Full Engineered Feature Vector Delivered to Linear Model"):
            st.json(prediction_result["feature_vector"])

    except Exception as e:
        st.error(f"Inference error: {str(e)}")
