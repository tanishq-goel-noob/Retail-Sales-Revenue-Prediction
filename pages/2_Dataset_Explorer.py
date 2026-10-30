"""
=============================================================================
PAGE: 2_Dataset_Explorer.py
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

st.set_page_config(page_title="Dataset Explorer", page_icon="📁", layout="wide")

st.title("📁 Retail Sales Dataset Explorer")
st.markdown("Inspect raw records, audit schemas, examine descriptive statistics, and upload custom datasets.")

# Tabbed Interface for Clean Organization
tab_default, tab_upload, tab_doc = st.tabs(["📊 Default Superstore Dataset", "📤 Custom CSV Upload", "📖 Dataset Documentation"])

with tab_default:
    try:
        df = load_dataset()
    except Exception as e:
        st.error(f"Error loading default dataset: {str(e)}")
        st.stop()

    summary = get_dataset_summary(df)

    # Dimensional summary
    dim_col1, dim_col2, dim_col3, dim_col4 = st.columns(4)
    dim_col1.metric("Row Count (N)", f"{summary['num_rows']:,}")
    dim_col2.metric("Column Count (P)", f"{summary['num_cols']}")
    dim_col3.metric("Missing Values", f"{summary['total_missing']}")
    dim_col4.metric("Duplicate Rows", f"{summary['duplicate_rows']}")

    st.markdown("---")

    # Data Preview Selection
    st.subheader("🔍 Data Inspection")
    preview_mode = st.radio("Select preview mode:", ["First 5 Rows (Head)", "Last 5 Rows (Tail)", "Random Sample (10 Rows)"], horizontal=True)

    if preview_mode == "First 5 Rows (Head)":
        st.dataframe(df.head(5), use_container_width=True)
    elif preview_mode == "Last 5 Rows (Tail)":
        st.dataframe(df.tail(5), use_container_width=True)
    else:
        st.dataframe(df.sample(min(10, len(df))), use_container_width=True)

    st.markdown("---")

    # Schema & Column Types
    st.subheader("📋 Schema & Feature Attributes")
    schema_data = []
    for col in df.columns:
        schema_data.append({
            "Column Name": col,
            "Data Type": str(df[col].dtype),
            "Non-Null Count": int(df[col].count()),
            "Null Count": int(df[col].isnull().sum()),
            "Null Percentage": f"{(df[col].isnull().sum() / len(df)) * 100:.2f}%",
            "Unique Values": int(df[col].nunique()),
            "Example Value": str(df[col].dropna().iloc[0]) if df[col].count() > 0 else "N/A"
        })
    st.dataframe(pd.DataFrame(schema_data), use_container_width=True)

    st.markdown("---")

    # Statistical Summaries
    st.subheader("📈 Descriptive Summary Statistics")
    col_num, col_cat = st.columns(2)

    with col_num:
        st.markdown("##### Numerical Features (Continuous)")
        st.dataframe(df.describe().round(2), use_container_width=True)

    with col_cat:
        st.markdown("##### Categorical Features (Discrete)")
        cat_df = df.select_dtypes(include=["object", "category"])
        if not cat_df.empty:
            st.dataframe(cat_df.describe(), use_container_width=True)
        else:
            st.info("No string-based categorical features detected.")


with tab_upload:
    st.subheader("📤 Upload Custom Retail Sales CSV")
    st.info(
        "Upload a custom retail dataset for analysis. To ensure compatibility with the "
        "forecasting pipeline, the file should contain a chronological date column (e.g., 'Order Date'), "
        "product or category descriptions, and the target monetary revenue column ('Sales' or 'Revenue')."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
            st.success(f"✓ Successfully uploaded `{uploaded_file.name}` ({len(custom_df):,} rows, {len(custom_df.columns)} columns)")

            st.write("##### Custom Dataset Preview:")
            st.dataframe(custom_df.head(5), use_container_width=True)

            # Check target variable presence
            has_sales = any(c in custom_df.columns for c in ["Sales", "sales", "Revenue", "revenue"])
            if has_sales:
                st.success("✓ Target revenue column detected.")
            else:
                st.warning("⚠️ No standard 'Sales' or 'Revenue' column found. Please verify column naming.")

        except Exception as err:
            st.error(f"Failed to read uploaded CSV: {str(err)}")


with tab_doc:
    st.subheader("📖 Dataset Technical Documentation")
    st.markdown(
        f"""
        - **Dataset Name:** {DATASET_METADATA['name']}
        - **Source:** {DATASET_METADATA['source']}
        - **Documentation URL:** [{DATASET_METADATA['source_url']}]({DATASET_METADATA['source_url']})
        - **Target Variable:** `{DATASET_METADATA['target_variable']}`
        - **Temporal Span:** {DATASET_METADATA['temporal_span']}
        - **Description:** {DATASET_METADATA['description']}
        
        ##### Feature Glossary:
        | Feature Name | Type | Description | Predictive Role |
        |---|---|---|---|
        | `Order Date` | Datetime | Date of purchase transaction | Temporal seasonality (month, quarter, day of week) |
        | `Category` | Categorical | Broad product hierarchy (Furniture, Tech, etc.) | High-level pricing regime |
        | `Sub-Category` | Categorical | Specific product group (17 categories) | Critical granular demand predictor |
        | `Region` | Categorical | US territory (West, East, Central, South) | Geographic demand propensity |
        | `Segment` | Categorical | Customer group (Consumer, Corporate, Home Office) | Basket size and ordering profile |
        | `Ship Mode` | Categorical | Delivery tier (Standard, Second, First, Same Day) | Urgency and order priority |
        | `Discount` | Continuous | Promotional markdown percentage (0.0 to 0.8) | Price elasticity and realized basket total |
        | `Sales` | Continuous | Gross transaction revenue ($ USD) | **Target Variable (y)** |
        
        ##### Limitations & Academic Boundary Conditions:
        1. **Episodic Transactions:** Orders arrive non-uniformly across days; addressed via calendar feature engineering rather than discrete ARIMA state-space models.
        2. **Target Leakage Prevention:** `Quantity` and `Unit Price` are intentionally omitted from simultaneous model inputs because $Sales = Price \\times Quantity$ represents an exact accounting identity.
        3. **Discount Formulation:** Represents transaction-level markdown rather than top-of-funnel marketing acquisition spend.
        """
    )
