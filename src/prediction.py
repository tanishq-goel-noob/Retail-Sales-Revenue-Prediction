"""
=============================================================================
MODULE: prediction.py
PROJECT: Retail Sales Revenue Prediction System
COURSEWORK: Third-Year B.Tech AI & ML - Predictive Modelling
=============================================================================

1. WHAT THIS FILE DOES:
   Provides a clean, validated inference interface to generate retail sales
   revenue forecasts using the trained Ordinary Least Squares (OLS) Linear
   Regression pipeline.

2. WHY THIS FILE IS NEEDED:
   In production machine learning systems, inference must:
   - Accept realistic forward-looking inputs (e.g., target date, product category,
     customer segment, regional store location, planned promotional discount).
   - Compute the exact engineered features used during training (e.g., season,
     calendar flags, historical subcategory benchmark) without exposing the model
     to target leakage.
   - Enforce schema conformity and type validation before passing tensors into the pipeline.
   - Present predicted revenue, expected confidence range, and actionable retail pricing insights.

3. LIBRARIES USED:
   - joblib: Deserialization of the saved Linear Regression pipeline and metadata.
   - pandas & numpy: Inference payload construction and array handling.

4. MACHINE LEARNING CONCEPTS INVOLVED:
   - Model Serving / Inference: Passing unobserved feature vectors through the
     frozen preprocessing and regression estimator.
   - Domain-Constrained Output: Linear Regression can theoretically extrapolate
     negative sales when discounts are steep or baseline inputs are extreme; we provide
     both raw OLS predictions and clipped non-negative business interpretations.
=============================================================================
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib

from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    get_season,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class RetailSalesPredictor:
    """
    Inference wrapper for the Linear Regression Retail Sales Revenue Prediction System.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initializes the predictor by loading the serialized Linear Regression pipeline and metadata.
        """
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.lr_pipeline = None
        self.metadata = None
        self.benchmarks = {}
        self.test_rmse = 450.0
        self._load_models()

    def _load_models(self) -> None:
        """Loads serialized pipeline and metadata from the models directory."""
        lr_path = self.models_dir / "linear_regression.joblib"
        meta_path = self.models_dir / "model_metadata.joblib"

        if not lr_path.exists():
            return

        try:
            self.lr_pipeline = joblib.load(lr_path)
            if meta_path.exists():
                self.metadata = joblib.load(meta_path)
                self.benchmarks = self.metadata.get("benchmarks", {})
                test_metrics = self.metadata.get("test_metrics", {})
                self.test_rmse = float(test_metrics.get("RMSE", 450.0))
        except Exception as e:
            print(f"Warning: Failed to load models: {str(e)}")

    def is_ready(self) -> bool:
        """Checks if the Linear Regression pipeline is loaded and ready for predictions."""
        return self.lr_pipeline is not None

    def prepare_input_dataframe(self, input_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Converts user inputs into the exact feature dataframe schema
        expected by the trained Linear Regression pipeline.

        Parameters:
        -----------
        input_dict : Dict[str, Any]
            Dictionary containing order attributes.

        Returns:
        --------
        pd.DataFrame
            Single-row DataFrame matching training feature columns.
        """
        date_val = pd.to_datetime(input_dict.get("order_date", pd.Timestamp.now()))
        month = int(date_val.month)
        day = int(date_val.day)
        year = int(date_val.year)
        day_of_week = int(date_val.dayofweek)
        quarter = int(date_val.quarter)
        is_weekend = int(day_of_week in [5, 6])
        season = get_season(month)

        subcategory = str(input_dict.get("subcategory", "Chairs")).strip()
        category = str(input_dict.get("category", "Furniture")).strip()
        segment = str(input_dict.get("segment", "Consumer")).strip()
        region = str(input_dict.get("region", "West")).strip()
        ship_mode = str(input_dict.get("ship_mode", "Standard Class")).strip()
        discount = float(input_dict.get("discount", 0.0))

        # Retrieve historical benchmark
        global_avg = self.benchmarks.get("__GLOBAL__", 230.0) if self.benchmarks else 230.0
        custom_bm = input_dict.get("custom_benchmark")
        if custom_bm is not None and float(custom_bm) > 0:
            hist_subcat_avg = float(custom_bm)
        elif self.benchmarks and subcategory in self.benchmarks:
            hist_subcat_avg = float(self.benchmarks[subcategory])
        else:
            hist_subcat_avg = global_avg

        row = {
            # Categorical
            "Category": category,
            "Sub-Category": subcategory,
            "Segment": segment,
            "Region": region,
            "Ship Mode": ship_mode,
            "Season": season,
            # Numerical
            "Discount": discount,
            "Year": year,
            "Month": month,
            "Day": day,
            "DayOfWeek": day_of_week,
            "Quarter": quarter,
            "IsWeekend": is_weekend,
            "Hist_SubCat_Avg_Sales": hist_subcat_avg,
        }

        # Order columns exactly matching CATEGORICAL_FEATURES + NUMERICAL_FEATURES
        feature_order = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
        df_input = pd.DataFrame([row])[feature_order]
        return df_input

    def predict(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates sales prediction using the trained Linear Regression pipeline.

        Parameters:
        -----------
        input_dict : Dict[str, Any]
            User input specifications.

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing predicted revenue, raw OLS value, expected
            error bounds, and business interpretations.
        """
        if not self.is_ready():
            raise RuntimeError(
                "Linear Regression model is not loaded. Please train or load the model first."
            )

        X_input = self.prepare_input_dataframe(input_dict)

        # Generate model prediction
        raw_lr_pred = float(self.lr_pipeline.predict(X_input)[0])

        # Domain constraint: Gross Sales cannot physically be negative in retail
        lr_pred = max(0.0, raw_lr_pred)

        discount_val = float(input_dict.get("discount", 0.0))
        subcat = input_dict.get("subcategory", "Product")

        # Confidence interval approximation using test RMSE (empirical standard error of regression)
        lower_bound = max(0.0, lr_pred - self.test_rmse)
        upper_bound = lr_pred + self.test_rmse

        # Actionable business insights
        if discount_val > 0.3:
            discount_note = (
                f"A high promotional discount of {discount_val * 100:.0f}% is applied. While this stimulates "
                "order velocity, the linear model reflects significant downward pressure on gross dollar realization."
            )
        elif discount_val > 0:
            discount_note = (
                f"A moderate promotional discount of {discount_val * 100:.0f}% is configured, "
                "supporting healthy basket size while preserving gross margins."
            )
        else:
            discount_note = "Standard full-price transaction (0% discount) maximizing gross dollar realization."

        return {
            "predicted_revenue": round(lr_pred, 2),
            "raw_ols_prediction": round(raw_lr_pred, 2),
            "lower_bound_estimate": round(lower_bound, 2),
            "upper_bound_estimate": round(upper_bound, 2),
            "test_rmse": round(self.test_rmse, 2),
            "feature_vector": X_input.to_dict(orient="records")[0],
            "discount_note": discount_note,
            "summary": (
                f"Linear Regression forecasts expected revenue of ${lr_pred:,.2f} for {subcat} "
                f"(expected range: ${lower_bound:,.2f} - ${upper_bound:,.2f} based on Test RMSE of ${self.test_rmse:,.2f})."
            ),
        }
