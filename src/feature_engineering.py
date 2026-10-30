"""
=============================================================================
MODULE: feature_engineering.py
PROJECT: Retail Sales Revenue Prediction System
COURSEWORK: Third-Year B.Tech AI & ML - Predictive Modelling
=============================================================================

1. WHAT THIS FILE DOES:
   Constructs predictive, leakage-free temporal and historical benchmark features
   from raw retail transactions. It creates seasonal, calendar, and historical
   demand indicators while strictly preventing forward-looking target leakage.

2. WHY THIS FILE IS NEEDED:
   Raw retail data contains dates and transactional attributes that machine
   learning algorithms cannot ingest directly. Furthermore, in real-world retail
   revenue forecasting:
   - Using 'Quantity' or 'Unit Price' simultaneously with 'Sales' creates
     mathematical target leakage (accounting identity: Sales = Price * Quantity).
   - In a forecasting setting, we predict expected sales revenue based on
     the product category, market segment, region, shipping mode, planned discount,
     calendar seasonality, and historical category benchmark performance.

3. LIBRARIES USED:
   - pandas & numpy: High-performance vectorized time and numerical transforms.
   - typing: Strict typing for inputs and outputs.

4. MACHINE LEARNING & STATISTICAL CONCEPTS INVOLVED:
   - Temporal Feature Extraction: Calendar periodicity (month, quarter, day-of-week).
   - Cyclical / Seasonal Grouping: Mapping months to retail business quarters and seasons.
   - Leakage Prevention: Historical subcategory running averages computed strictly
     using past records via shifting (`shift(1)`), ensuring zero future look-ahead.
=============================================================================
"""

from typing import Tuple, List, Dict
import pandas as pd
import numpy as np


# Categorical and numerical feature definitions used across the modeling pipeline
CATEGORICAL_FEATURES: List[str] = [
    "Category",
    "Sub-Category",
    "Segment",
    "Region",
    "Ship Mode",
    "Season",
]

NUMERICAL_FEATURES: List[str] = [
    "Discount",
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "Quarter",
    "IsWeekend",
    "Hist_SubCat_Avg_Sales",
]

TARGET_COLUMN: str = "Sales"


def get_season(month: int) -> str:
    """
    Maps a calendar month (1-12) to its meteorological/retail season.

    Parameters:
    -----------
    month : int
        Calendar month integer (1 to 12).

    Returns:
    --------
    str
        One of 'Winter', 'Spring', 'Summer', 'Fall'.
    """
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"


def engineer_features(df: pd.DataFrame, is_training: bool = True, subcat_benchmarks: Dict[str, float] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Transforms raw retail dataframe into engineered feature dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing raw retail columns.
    is_training : bool
        If True, calculates historical category benchmarks from training data
        using shift(1) to prevent leakage.
        If False, uses provided `subcat_benchmarks` to transform validation/test/inference data.
    subcat_benchmarks : Dict[str, float], optional
        Historical subcategory revenue benchmarks computed from training data.

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, float]]
        - Transformed DataFrame containing engineered features and target.
        - Dictionary of historical subcategory average sales benchmarks.
    """
    data = df.copy()

    # 1. Ensure Chronological Sorting
    if "Order Date" in data.columns:
        data["Order Date"] = pd.to_datetime(data["Order Date"], errors="coerce")
        data = data.sort_values(by="Order Date").reset_index(drop=True)
    else:
        # Fallback if no date column exists
        data["Order Date"] = pd.date_range(start="2020-01-01", periods=len(data), freq="D")

    # 2. Extract Calendar & Temporal Features
    data["Year"] = data["Order Date"].dt.year.fillna(2020).astype(int)
    data["Month"] = data["Order Date"].dt.month.fillna(1).astype(int)
    data["Day"] = data["Order Date"].dt.day.fillna(1).astype(int)
    data["DayOfWeek"] = data["Order Date"].dt.dayofweek.fillna(0).astype(int)
    data["Quarter"] = data["Order Date"].dt.quarter.fillna(1).astype(int)
    data["IsWeekend"] = data["DayOfWeek"].isin([5, 6]).astype(int)
    data["Season"] = data["Month"].apply(get_season)

    # 3. Ensure 'Discount' exists and is numeric
    if "Discount" in data.columns:
        data["Discount"] = pd.to_numeric(data["Discount"], errors="coerce").fillna(0.0)
    else:
        data["Discount"] = 0.0

    # 4. Create Historical Category Benchmark Feature (Leakage-Free)
    # At inference time, retailers rely on historical category sales performance
    if is_training:
        # Group by Sub-Category, shift by 1 to exclude current transaction, and compute expanding mean
        if "Sub-Category" in data.columns and TARGET_COLUMN in data.columns:
            data["Sales"] = pd.to_numeric(data["Sales"], errors="coerce")
            
            # Shift by 1 within each subcategory so row t never sees row t's Sales
            shifted_sales = data.groupby("Sub-Category")["Sales"].shift(1)
            hist_avg = shifted_sales.groupby(data["Sub-Category"]).expanding().mean().reset_index(level=0, drop=True)
            
            # For the very first transactions in a subcategory, fill with overall expanding mean
            global_expanding_mean = data["Sales"].shift(1).expanding().mean()
            data["Hist_SubCat_Avg_Sales"] = hist_avg.fillna(global_expanding_mean).fillna(data["Sales"].median())

            # Store the final benchmark averages computed across training data for inference
            final_benchmarks = data.groupby("Sub-Category")["Sales"].mean().to_dict()
            global_benchmark = float(data["Sales"].mean())
            final_benchmarks["__GLOBAL__"] = global_benchmark
        else:
            data["Hist_SubCat_Avg_Sales"] = 200.0
            final_benchmarks = {"__GLOBAL__": 200.0}
    else:
        # Use frozen training benchmarks for test/inference
        if subcat_benchmarks is None:
            subcat_benchmarks = {"__GLOBAL__": 200.0}
        global_avg = subcat_benchmarks.get("__GLOBAL__", 200.0)

        if "Sub-Category" in data.columns:
            data["Hist_SubCat_Avg_Sales"] = data["Sub-Category"].map(subcat_benchmarks).fillna(global_avg)
        else:
            data["Hist_SubCat_Avg_Sales"] = global_avg
        final_benchmarks = subcat_benchmarks

    # 5. Ensure Categorical Columns are strings
    for col in CATEGORICAL_FEATURES:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip()
        else:
            data[col] = "Unknown"

    return data, final_benchmarks


def prepare_modeling_data(df: pd.DataFrame, subcat_benchmarks: Dict[str, float] = None, is_training: bool = True) -> Tuple[pd.DataFrame, pd.Series, Dict[str, float]]:
    """
    Full feature pipeline that prepares feature matrix X and target vector y.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe.
    subcat_benchmarks : Dict[str, float], optional
        Historical benchmarks dictionary.
    is_training : bool
        Whether this is the training dataset.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series, Dict[str, float]]
        - X (pd.DataFrame): Clean feature matrix with defined features.
        - y (pd.Series): Target vector (Sales).
        - benchmarks (dict): Benchmark lookup dictionary.
    """
    engineered_df, benchmarks = engineer_features(df, is_training=is_training, subcat_benchmarks=subcat_benchmarks)

    # Select only legitimate predictive columns
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X = engineered_df[feature_cols].copy()

    if TARGET_COLUMN in engineered_df.columns:
        y = pd.to_numeric(engineered_df[TARGET_COLUMN], errors="coerce").fillna(0.0)
    else:
        y = pd.Series([0.0] * len(X), name=TARGET_COLUMN)

    return X, y, benchmarks
