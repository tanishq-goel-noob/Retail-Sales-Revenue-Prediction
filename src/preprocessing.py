"""
=============================================================================
MODULE: preprocessing.py
PROJECT: Retail Sales Revenue Prediction System
COURSEWORK: Third-Year B.Tech AI & ML - Predictive Modelling
=============================================================================

1. WHAT THIS FILE DOES:
   Constructs modular, leak-free Scikit-learn Preprocessing Pipelines,
   ColumnTransformers, IQR-based outlier detectors, and chronological
   train-test splitters for retail sales forecasting.

2. WHY THIS FILE IS NEEDED:
   Machine learning models cannot reliably learn from raw mixed-type data:
   - Categorical attributes (e.g. 'Furniture', 'West') must be numerically encoded.
   - Linear Regression requires scaled features for numerical stability and coefficient interpretability.
   - Preprocessing leakage must be strictly prevented: transformers (imputers, scalers, encoders)
     must be fitted ONLY on the training split, and applied to test and inference data.
   - Time-dependent retail observations must be chronologically partitioned (past -> train, future -> test)
     to prevent look-ahead bias.

3. LIBRARIES USED:
   - scikit-learn:
     - Pipeline: Chains data transformation steps sequentially.
     - ColumnTransformer: Applies distinct preprocessing to numerical vs categorical features.
     - SimpleImputer: Handles missing values using statistical strategies.
     - OneHotEncoder: Converts discrete categorical variables into binary dummy columns.
     - StandardScaler: Centers features to mean=0 and variance=1.
   - numpy & pandas: Numerical matrix operations and dataframe slicing.

4. MACHINE LEARNING & STATISTICAL CONCEPTS INVOLVED:
   - Imputation: Median imputation for continuous variables (robust against outliers),
     most-frequent imputation for categorical strings.
   - Standardization: z = (x - mu) / sigma. Ensures equal gradient descent step sizes
     and avoids numerical conditioning errors in Matrix Inversion (X^T X)^-1.
   - One-Hot Encoding: Converts k categories into k orthogonal binary features, avoiding
     false ordinal assumptions.
   - Interquartile Range (IQR): Non-parametric outlier boundary calculation:
     IQR = Q3 - Q1, Lower Bound = Q1 - 1.5 * IQR, Upper Bound = Q3 + 1.5 * IQR.
   - Chronological Splitting: Preserves the arrow of time, preventing look-ahead leakage.
=============================================================================
"""

from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that handles numerical and categorical columns.

    Parameters:
    -----------
    scale_numeric : bool, default=True
        Whether to apply StandardScaler to numerical features.
        - True for Linear Regression (enhances matrix inversion stability and coefficient comparability).
        - Can be False or True for Random Forest (tree splits are monotonic and scale-invariant).

    Returns:
    --------
    ColumnTransformer
        An unfitted Scikit-learn ColumnTransformer instance.
    """
    # 1. Numerical Pipeline
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=num_steps)

    # 2. Categorical Pipeline
    # handle_unknown='ignore' ensures unseen categories at test/inference time become all zeros
    # sparse_output=False ensures dense array output compatible with all estimators
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # 3. Combine into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def detect_outliers_iqr(series: pd.Series) -> Dict[str, Any]:
    """
    Performs Interquartile Range (IQR) outlier detection on a numerical series.

    Mathematical Formulation:
    - Q1 = 25th percentile
    - Q3 = 75th percentile
    - IQR = Q3 - Q1
    - Lower Bound = Q1 - 1.5 * IQR
    - Upper Bound = Q3 + 1.5 * IQR
    - Outlier if: value < Lower Bound OR value > Upper Bound

    Parameters:
    -----------
    series : pd.Series
        Numerical column (e.g. Sales).

    Returns:
    --------
    Dict[str, Any]
        Dictionary with Q1, Q3, IQR, bounds, outlier count, and percentage.
    """
    clean_series = pd.to_numeric(series, errors="coerce").dropna()
    q1 = float(np.percentile(clean_series, 25))
    q3 = float(np.percentile(clean_series, 75))
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (clean_series < lower_bound) | (clean_series > upper_bound)
    outlier_count = int(outlier_mask.sum())
    total_count = len(clean_series)
    outlier_pct = float((outlier_count / total_count) * 100) if total_count > 0 else 0.0

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": outlier_count,
        "total_count": total_count,
        "outlier_percentage": round(outlier_pct, 2),
        "min_value": float(clean_series.min()) if total_count > 0 else 0.0,
        "max_value": float(clean_series.max()) if total_count > 0 else 0.0,
    }


def chronological_train_test_split(
    X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits features and target chronologically into training and testing sets.

    Parameters:
    -----------
    X : pd.DataFrame
        Chronologically sorted feature matrix.
    y : pd.Series
        Chronologically sorted target vector.
    train_ratio : float, default=0.8
        Fraction of past observations allocated to training (e.g. 0.80 = 80%).

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    n = len(X)
    split_index = int(n * train_ratio)

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()
    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    return X_train, X_test, y_train, y_test
