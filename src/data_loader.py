"""
=============================================================================
MODULE: data_loader.py
PROJECT: Retail Sales Revenue Prediction System
COURSEWORK: Third-Year B.Tech AI & ML - Predictive Modelling
=============================================================================

1. WHAT THIS FILE DOES:
   This module handles robust ingestion, verification, schema inspection,
   and standard metadata extraction for retail sales datasets (both the default
   real-world Tableau Superstore dataset and custom user-uploaded CSV files).

2. WHY THIS FILE IS NEEDED:
   In production Machine Learning pipelines, raw data ingestion must be decoupled
   from downstream preprocessing and modeling. A dedicated data loader guarantees
   that data integrity checks, file path resolution, encoding detection, and
   initial schema validations occur consistently across both CLI scripts and
   the Streamlit web interface.

3. LIBRARIES USED:
   - pandas: Primary library for high-performance tabular data manipulation.
   - pathlib.Path: Object-oriented filesystem path handling across Windows/Linux.
   - typing (Optional, Tuple, Dict, Any): Strong type hinting for code reliability.

4. MACHINE LEARNING & STATISTICAL CONCEPTS INVOLVED:
   - Data Ingestion & Sanitization: Ensuring input tensors/matrices receive valid types.
   - Schema Validation: Ensuring expected predictive features and targets exist.
   - Temporal Integrity: Identifying and parsing chronological timestamp fields.
=============================================================================
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd


# Default dataset location relative to project root
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "retail_sales.csv"

# Dataset Documentation Metadata
DATASET_METADATA: Dict[str, Any] = {
    "name": "Tableau Sample Superstore Retail Sales Dataset",
    "source": "Tableau Public & Kaggle Open Datasets Repository",
    "source_url": "https://community.tableau.com/s/article/sample-superstore-sales-excel",
    "description": (
        "A benchmark real-world retail transactions dataset capturing 4 consecutive "
        "years of commercial retail sales across 4 geographic regions in the United States. "
        "Each record represents an individual line-item order with customer demographics, "
        "product taxonomy, promotional discounts, and realized gross monetary sales."
    ),
    "target_variable": "Sales (Monetary Revenue in USD)",
    "temporal_span": "Multi-year historical transactions (Chronological)",
    "limitations": [
        "Dataset represents an office/furniture/tech retailer; consumer behavior may vary in grocery or fashion.",
        "Promotional discount is recorded at transaction time, not long-term campaign budget.",
        "Transactions are episodic rather than strictly equidistant time series (handled via tabular temporal feature engineering)."
    ]
}

# Standard column mappings to allow flexible column names if user uploads similar CSV
EXPECTED_COLUMNS = [
    "Order Date", "Category", "Sub-Category", "Region", "Segment", "Ship Mode", "Discount", "Sales"
]


def load_dataset(file_path_or_buffer: Optional[Any] = None) -> pd.DataFrame:
    """
    Loads the retail sales dataset from a local path or a file-like buffer (e.g., Streamlit upload).

    Parameters:
    -----------
    file_path_or_buffer : Optional[Any]
        File path (str/Path) or BytesIO/StringIO buffer from Streamlit file uploader.
        If None, attempts to load the default dataset at `data/retail_sales.csv`.

    Returns:
    --------
    pd.DataFrame
        Loaded and verified pandas DataFrame with trimmed column headers and parsed dates.

    Raises:
    -------
    FileNotFoundError:
        If default dataset path does not exist on disk.
    ValueError:
        If the file cannot be parsed or lacks the mandatory target variable ('Sales' or 'Revenue').
    """
    target_path = file_path_or_buffer if file_path_or_buffer is not None else DEFAULT_DATASET_PATH

    # Check existence if a filesystem path is provided
    if isinstance(target_path, (str, Path)) and not os.path.exists(str(target_path)):
        raise FileNotFoundError(
            f"Dataset file not found at: {target_path}. "
            "Please ensure 'data/retail_sales.csv' exists or upload a valid CSV file."
        )

    try:
        # Load CSV with fallback encodings for real-world enterprise CSV files
        try:
            df = pd.read_csv(target_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(target_path, encoding="latin1")

    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {str(e)}")

    # Clean whitespace from column names
    df.columns = [col.strip() for col in df.columns]

    # Validate presence of target variable (or common aliases)
    sales_col = None
    for candidate in ["Sales", "sales", "Revenue", "revenue", "Total_Amount", "Total Amount"]:
        if candidate in df.columns:
            sales_col = candidate
            break

    if sales_col is None:
        raise ValueError(
            f"Dataset must contain a revenue column such as 'Sales' or 'Revenue'. Found columns: {list(df.columns)}"
        )
    elif sales_col != "Sales":
        df.rename(columns={sales_col: "Sales"}, inplace=True)

    # Normalize Date column if present
    date_col = None
    for candidate in ["Order Date", "Order_Date", "Date", "date", "Transaction Date"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is not None and date_col != "Order Date":
        df.rename(columns={date_col: "Order Date"}, inplace=True)

    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes a comprehensive statistical and structural overview of the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        The loaded retail sales DataFrame.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing shape, memory usage, missing counts, duplicate counts,
        numerical/categorical column lists, and summary statistics.
    """
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    total_revenue = float(df["Sales"].sum()) if "Sales" in df.columns else 0.0
    avg_revenue = float(df["Sales"].mean()) if "Sales" in df.columns else 0.0
    median_revenue = float(df["Sales"].median()) if "Sales" in df.columns else 0.0

    return {
        "num_rows": int(df.shape[0]),
        "num_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_revenue": total_revenue,
        "average_revenue": avg_revenue,
        "median_revenue": median_revenue,
        "date_min": str(df["Order Date"].min()) if "Order Date" in df.columns else None,
        "date_max": str(df["Order Date"].max()) if "Order Date" in df.columns else None,
    }
