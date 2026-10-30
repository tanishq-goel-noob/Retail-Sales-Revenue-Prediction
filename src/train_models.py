"""
=============================================================================
MODULE: train_models.py
PROJECT: Retail Sales Revenue Prediction System
COURSEWORK: Third-Year B.Tech AI & ML - Predictive Modelling
=============================================================================

1. WHAT THIS FILE DOES:
   Constructs, trains, evaluates, and serializes the Ordinary Least Squares (OLS)
   Linear Regression Pipeline using chronological training and testing data splits.

2. WHY THIS FILE IS NEEDED:
   In academic machine learning and production deployment:
   - Encapsulating preprocessing (scaling, imputation, one-hot encoding) inside
     a Scikit-learn Pipeline guarantees that all feature transformations are fitted
     strictly on training data and can be serialized cleanly for inference.
   - Diagnostic statistics (coefficients, intercept, residuals, train vs test metrics)
     must be calculated rigorously to evaluate model fit and assumptions.

3. LIBRARIES USED:
   - sklearn.linear_model.LinearRegression: Fits Ordinary Least Squares regression.
   - sklearn.pipeline.Pipeline: Chains preprocessor and estimator.
   - joblib: Efficient serialization of the trained pipeline and metadata.
   - pandas & numpy: Array and matrix operations.

4. MACHINE LEARNING CONCEPTS INVOLVED:
   - Ordinary Least Squares (OLS): Solves for weights beta by minimizing sum of squared residuals:
     min sum (y_i - x_i^T beta)^2.
   - Closed-Form Solution: beta = (X^T X)^(-1) X^T y.
   - Generalization: Benchmarking Train vs Test metrics to identify overfitting or underfitting.
=============================================================================
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from src.preprocessing import build_preprocessor, chronological_train_test_split
from src.feature_engineering import prepare_modeling_data
from src.evaluation import calculate_metrics


# Storage path for serialized model artifacts
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def get_feature_names_from_preprocessor(preprocessor) -> list:
    """
    Safely extracts transformed feature names from a fitted ColumnTransformer.

    Parameters:
    -----------
    preprocessor : ColumnTransformer
        Fitted Scikit-learn ColumnTransformer.

    Returns:
    --------
    list
        List of generated feature names (numerical + one-hot encoded categories).
    """
    try:
        raw_names = preprocessor.get_feature_names_out()
        clean_names = [col.replace("num__", "").replace("cat__", "") for col in raw_names]
        return clean_names
    except Exception:
        return [f"Feature_{i}" for i in range(100)]


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Pipeline, Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """
    Builds, trains, and evaluates a Linear Regression Pipeline.

    Parameters:
    -----------
    X_train, y_train : pd.DataFrame, pd.Series
        Training features and target.
    X_test, y_test : pd.DataFrame, pd.Series
        Testing features and target.

    Returns:
    --------
    Tuple[Pipeline, Dict[str, float], Dict[str, float], Dict[str, Any]]
        - Fitted Scikit-learn Pipeline
        - Train evaluation metrics (MAE, MSE, RMSE, R2, MAPE)
        - Test evaluation metrics (MAE, MSE, RMSE, R2, MAPE)
        - Diagnostics dictionary (coefficients, intercept, predictions, residuals)
    """
    # 1. Build pipeline: Preprocessor (with StandardScaler) + LinearRegression
    preprocessor = build_preprocessor(scale_numeric=True)
    lr_model = LinearRegression(fit_intercept=True)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", lr_model),
    ])

    # 2. Fit pipeline strictly on training data
    pipeline.fit(X_train, y_train)

    # 3. Generate predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # 4. Compute metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)

    # 5. Extract coefficients and diagnostic statistics
    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    fitted_regressor = pipeline.named_steps["regressor"]
    feature_names = get_feature_names_from_preprocessor(fitted_preprocessor)

    coef_df = pd.DataFrame({
        "Feature": feature_names[: len(fitted_regressor.coef_)],
        "Coefficient": fitted_regressor.coef_,
        "Abs_Coefficient": np.abs(fitted_regressor.coef_),
    }).sort_values(by="Abs_Coefficient", ascending=False).reset_index(drop=True)

    train_residuals = np.asarray(y_train) - y_train_pred
    test_residuals = np.asarray(y_test) - y_test_pred

    diagnostics = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "intercept": float(fitted_regressor.intercept_),
        "coefficients_df": coef_df,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "train_residuals": train_residuals,
        "test_residuals": test_residuals,
    }

    return pipeline, train_metrics, test_metrics, diagnostics


def run_full_training_pipeline(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> Dict[str, Any]:
    """
    Executes the end-to-end training and evaluation workflow for Linear Regression.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset.
    train_ratio : float, default=0.8
        Fraction of past chronological observations allocated to training.

    Returns:
    --------
    Dict[str, Any]
        Complete results dictionary with trained pipeline, train/test metrics,
        diagnostics, and partitioned datasets.
    """
    # 1. Feature Engineering
    X, y, benchmarks = prepare_modeling_data(df, is_training=True)

    # 2. Chronological Split
    X_train, X_test, y_train, y_test = chronological_train_test_split(X, y, train_ratio=train_ratio)

    # 3. Train Linear Regression
    lr_pipeline, train_metrics, test_metrics, lr_diag = train_linear_regression(
        X_train, y_train, X_test, y_test
    )

    # 4. Serialize model and metadata
    os.makedirs(MODELS_DIR, exist_ok=True)
    lr_path = MODELS_DIR / "linear_regression.joblib"
    meta_path = MODELS_DIR / "model_metadata.joblib"

    joblib.dump(lr_pipeline, lr_path)

    metadata = {
        "benchmarks": benchmarks,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "model_name": "Linear Regression (Ordinary Least Squares)",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "target_col": "Sales",
    }
    joblib.dump(metadata, meta_path)

    return {
        "lr_pipeline": lr_pipeline,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "lr_diagnostics": lr_diag,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "benchmarks": benchmarks,
    }
