"""
=============================================================================
MODULE: evaluation.py
PROJECT: Retail Sales Revenue Prediction System
COURSEWORK: Third-Year B.Tech AI & ML - Predictive Modelling
=============================================================================

1. WHAT THIS FILE DOES:
   Computes standard regression evaluation metrics (MAE, MSE, RMSE, R2, MAPE)
   for the Linear Regression model on both training and unseen testing partitions.
   It provides structured diagnostic matrices to audit generalization.

2. WHY THIS FILE IS NEEDED:
   Model evaluation in machine learning must be rigorous, objective, and mathematically
   sound:
   - RMSE and MSE penalize large catastrophic misses heavily.
   - MAE offers an intuitive magnitude of median/average error in original dollar units.
   - R-squared indicates variance explained compared to a naive mean predictor.
   - MAPE conveys relative percentage error to business stakeholders.
   - Comparing Train vs Test performance verifies whether the linear model generalizes
     without severe underfitting or overfitting.

3. LIBRARIES USED:
   - scikit-learn.metrics: mean_absolute_error, mean_squared_error, r2_score.
   - numpy: Vectorized square roots and absolute percentage calculations.
   - pandas: Structured tabular display of metrics.

4. MATHEMATICAL FORMULAS IMPLEMENTED:
   A. MAE  = (1 / n) * sum(|y_i - y_hat_i|)
   B. MSE  = (1 / n) * sum((y_i - y_hat_i)^2)
   C. RMSE = sqrt(MSE)
   D. R2   = 1 - [sum((y_i - y_hat_i)^2) / sum((y_i - y_bar)^2)]
   E. MAPE = (100 / n) * sum(|(y_i - y_hat_i) / y_i|) [with epsilon protection for y_i near 0]
=============================================================================
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes all standard regression evaluation metrics between ground truth and predictions.

    Parameters:
    -----------
    y_true : np.ndarray or pd.Series
        True actual target values (Sales).
    y_pred : np.ndarray or pd.Series
        Model predicted values (Predicted Sales).

    Returns:
    --------
    Dict[str, float]
        Dictionary with keys: 'MAE', 'MSE', 'RMSE', 'R2', 'MAPE'.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    # 1. Mean Absolute Error (MAE)
    mae = float(mean_absolute_error(y_t, y_p))

    # 2. Mean Squared Error (MSE)
    mse = float(mean_squared_error(y_t, y_p))

    # 3. Root Mean Squared Error (RMSE)
    rmse = float(np.sqrt(mse))

    # 4. R-Squared Score (R2)
    r2 = float(r2_score(y_t, y_p))

    # 5. Mean Absolute Percentage Error (MAPE)
    epsilon = 1e-6
    non_zero_mask = np.abs(y_t) > epsilon
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs((y_t[non_zero_mask] - y_p[non_zero_mask]) / y_t[non_zero_mask])) * 100.0)
    else:
        mape = float(np.nan)

    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2) if not np.isnan(mape) else 0.0,
    }


def create_train_test_comparison_table(
    train_metrics: Dict[str, float], test_metrics: Dict[str, float]
) -> pd.DataFrame:
    """
    Constructs a structured DataFrame comparing Train vs Test performance
    for Linear Regression to inspect generalization gap.

    Parameters:
    -----------
    train_metrics : Dict[str, float]
        Metrics on chronological training data.
    test_metrics : Dict[str, float]
        Metrics on chronological testing data.

    Returns:
    --------
    pd.DataFrame
        Formatted comparison table.
    """
    metrics_list = ["MAE", "MSE", "RMSE", "R2", "MAPE"]
    descriptions = [
        "Mean Absolute Error ($ USD)",
        "Mean Squared Error ($² USD)",
        "Root Mean Squared Error ($ USD)",
        "Coefficient of Determination (Variance Explained)",
        "Mean Absolute Percentage Error (%)",
    ]
    preferred_directions = [
        "Lower is better",
        "Lower is better",
        "Lower is better",
        "Higher is better",
        "Lower is better",
    ]

    records = []
    for m, desc, pref in zip(metrics_list, descriptions, preferred_directions):
        tr_val = train_metrics.get(m, 0.0)
        te_val = test_metrics.get(m, 0.0)

        # Compute percentage difference relative to train
        if tr_val != 0:
            diff_pct = round(((te_val - tr_val) / abs(tr_val)) * 100, 2)
            diff_str = f"{diff_pct:+.2f}%"
        else:
            diff_str = "N/A"

        records.append({
            "Metric": m,
            "Description": desc,
            "Training Split (Past 80%)": tr_val,
            "Testing Split (Future 20%)": te_val,
            "Generalization Gap (Test vs Train)": diff_str,
            "Preferred Direction": pref,
        })

    return pd.DataFrame(records)
