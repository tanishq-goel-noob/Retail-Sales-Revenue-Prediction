"""
=============================================================================
MODULE: visualization.py
PROJECT: Retail Sales Revenue Prediction System
COURSEWORK: Third-Year B.Tech AI & ML - Predictive Modelling
=============================================================================

1. WHAT THIS FILE DOES:
   Provides a suite of statistical, exploratory, and diagnostic plotting functions
   utilizing Plotly (for modern interactive dashboards in Streamlit) and
   Matplotlib / Seaborn.

2. WHY THIS FILE IS NEEDED:
   In predictive modeling and regression analysis:
   - EDA plots uncover skewness, seasonal cyclicality, category dominance, and outliers.
   - Diagnostic plots (Actual vs Predicted, Residual distributions, Residuals vs Fitted)
     validate whether Linear Regression assumptions (homoscedasticity, normality of errors)
     are satisfied.
   - Train vs Test metrics comparison visualizes the generalization gap.
=============================================================================
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go


# Modern corporate color palette
THEME_COLORS = {
    "primary": "#1E3A8A",      # Deep Navy
    "secondary": "#0D9488",    # Teal
    "accent": "#F59E0B",       # Amber
    "danger": "#EF4444",       # Crimson
    "dark": "#1F2937",         # Charcoal
    "light": "#F3F4F6",        # Slate Gray
    "lr_color": "#3B82F6",     # Vibrant Blue
    "train_color": "#6366F1",  # Indigo
    "test_color": "#10B981",   # Emerald
}


# =============================================================================
# EXPLORATORY DATA ANALYSIS (EDA) CHARTS
# =============================================================================

def plot_monthly_revenue_trend(df: pd.DataFrame) -> go.Figure:
    """Renders an interactive monthly revenue time series graph."""
    data = df.copy()
    if "Order Date" not in data.columns:
        return go.Figure()

    data["YearMonth"] = data["Order Date"].dt.to_period("M").astype(str)
    monthly_sales = data.groupby("YearMonth")["Sales"].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly_sales["YearMonth"],
            y=monthly_sales["Sales"],
            mode="lines+markers",
            name="Monthly Sales",
            line=dict(color=THEME_COLORS["primary"], width=3),
            marker=dict(size=7, color=THEME_COLORS["secondary"]),
            hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Monthly Retail Sales Revenue Trend",
        xaxis_title="Transaction Period (Year-Month)",
        yaxis_title="Total Gross Revenue ($ USD)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        height=420,
    )
    return fig


def plot_revenue_distribution(df: pd.DataFrame) -> go.Figure:
    """Renders the target variable (Sales) distribution histogram."""
    sales = df["Sales"].dropna()

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=sales[sales <= 2000],
            nbinsx=60,
            marker_color=THEME_COLORS["primary"],
            opacity=0.85,
            name="Sales <= $2,000",
            hovertemplate="Revenue Bin: $%{x}<br>Count: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Retail Sales Revenue Distribution (Transactions <= $2,000)",
        xaxis_title="Sales Revenue ($ USD)",
        yaxis_title="Frequency / Transaction Count",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_category_revenue(df: pd.DataFrame) -> go.Figure:
    """Renders product category revenue comparison bar chart."""
    if "Category" not in df.columns:
        return go.Figure()

    cat_sales = df.groupby("Category")["Sales"].agg(["sum", "mean"]).reset_index()
    cat_sales = cat_sales.sort_values(by="sum", ascending=False)

    fig = go.Figure(
        data=[
            go.Bar(
                x=cat_sales["Category"],
                y=cat_sales["sum"],
                marker_color=[THEME_COLORS["primary"], THEME_COLORS["secondary"], THEME_COLORS["accent"]],
                text=[f"${v:,.0f}" for v in cat_sales["sum"]],
                textposition="auto",
                hovertemplate="<b>%{x}</b><br>Total Revenue: $%{y:,.2f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Total Revenue by Product Category",
        xaxis_title="Category",
        yaxis_title="Total Gross Revenue ($ USD)",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_subcategory_revenue(df: pd.DataFrame) -> go.Figure:
    """Renders top sub-categories by sales revenue."""
    if "Sub-Category" not in df.columns:
        return go.Figure()

    sub_sales = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=True).reset_index()

    fig = go.Figure(
        data=[
            go.Bar(
                x=sub_sales["Sales"],
                y=sub_sales["Sub-Category"],
                orientation="h",
                marker_color=THEME_COLORS["primary"],
                text=[f"${v:,.0f}" for v in sub_sales["Sales"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Revenue: $%{x:,.2f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Revenue Contribution by Sub-Category",
        xaxis_title="Total Gross Revenue ($ USD)",
        yaxis_title="Sub-Category",
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_region_revenue(df: pd.DataFrame) -> go.Figure:
    """Renders regional revenue share pie/donut chart."""
    if "Region" not in df.columns:
        return go.Figure()

    reg_sales = df.groupby("Region")["Sales"].sum().reset_index()

    fig = px.pie(
        reg_sales,
        names="Region",
        values="Sales",
        hole=0.45,
        color_discrete_sequence=[
            THEME_COLORS["primary"],
            THEME_COLORS["secondary"],
            THEME_COLORS["accent"],
            "#6366F1",
        ],
        title="Geographic Revenue Share by Region",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=40, r=40, t=50, b=40))
    return fig


def plot_discount_vs_revenue(df: pd.DataFrame) -> go.Figure:
    """Renders promotional discount vs revenue scatter plot."""
    if "Discount" not in df.columns or "Sales" not in df.columns:
        return go.Figure()

    sample_df = df.sample(min(1500, len(df)), random_state=42)

    fig = px.scatter(
        sample_df,
        x="Discount",
        y="Sales",
        color="Category" if "Category" in sample_df.columns else None,
        opacity=0.65,
        title="Promotional Discount Rate vs Transaction Sales Revenue",
        labels={"Discount": "Discount Percentage", "Sales": "Sales Revenue ($ USD)"},
        template="plotly_white",
        hover_data=["Sub-Category"] if "Sub-Category" in sample_df.columns else None,
    )
    fig.update_layout(height=420, margin=dict(l=40, r=40, t=50, b=40))
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Renders correlation matrix for numerical features."""
    num_df = df.select_dtypes(include=["number"])
    if num_df.shape[1] < 2:
        return go.Figure()

    corr = num_df.corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Pearson Correlation Heatmap (Numerical Features)",
        aspect="auto",
    )
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=40, r=40, t=50, b=40))
    return fig


def plot_outlier_boxplot(df: pd.DataFrame, column: str = "Sales") -> go.Figure:
    """Renders an IQR boxplot highlighting outliers."""
    if column not in df.columns:
        return go.Figure()

    fig = px.box(
        df,
        y=column,
        points="outliers",
        color_discrete_sequence=[THEME_COLORS["primary"]],
        title=f"Outlier Detection Boxplot: {column}",
        labels={column: f"{column} ($ USD)"},
        template="plotly_white",
    )
    fig.update_layout(height=380, margin=dict(l=40, r=40, t=50, b=40))
    return fig


# =============================================================================
# LINEAR REGRESSION DIAGNOSTIC & EVALUATION CHARTS
# =============================================================================

def plot_linear_coefficients(coef_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Renders Linear Regression coefficient magnitude and direction bar chart."""
    top_df = coef_df.head(top_n).sort_values(by="Coefficient", ascending=True)

    colors = [THEME_COLORS["danger"] if c < 0 else THEME_COLORS["lr_color"] for c in top_df["Coefficient"]]

    fig = go.Figure(
        data=[
            go.Bar(
                x=top_df["Coefficient"],
                y=top_df["Feature"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.2f}" for v in top_df["Coefficient"]],
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title=f"Top {top_n} Linear Regression Weights (Beta Coefficients)",
        xaxis_title="Standardized Beta Coefficient (Red: Negative | Blue: Positive)",
        yaxis_title="Feature Name",
        template="plotly_white",
        height=480,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_train_test_metrics_comparison(
    train_metrics: Dict[str, float], test_metrics: Dict[str, float]
) -> go.Figure:
    """Renders grouped bar chart comparing Train vs Test performance."""
    metric_keys = ["MAE", "RMSE"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Training Set (Past 80%)",
            x=metric_keys,
            y=[train_metrics[m] for m in metric_keys],
            marker_color=THEME_COLORS["train_color"],
            text=[f"${train_metrics[m]:,.1f}" for m in metric_keys],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Testing Set (Future 20%)",
            x=metric_keys,
            y=[test_metrics[m] for m in metric_keys],
            marker_color=THEME_COLORS["test_color"],
            text=[f"${test_metrics[m]:,.1f}" for m in metric_keys],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Generalization Audit: Train vs Test Error Metrics ($ USD)",
        barmode="group",
        xaxis_title="Evaluation Metric",
        yaxis_title="Error Value ($ USD)",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Linear Regression"
) -> go.Figure:
    """Renders actual vs predicted scatter plot with 45-degree diagonal reference line."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if len(y_t) > 1000:
        idx = np.random.RandomState(42).choice(len(y_t), 1000, replace=False)
        y_t = y_t[idx]
        y_p = y_p[idx]

    max_val = min(3000, max(np.max(y_t), np.max(y_p)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_t,
            y=y_p,
            mode="markers",
            marker=dict(color=THEME_COLORS["lr_color"], opacity=0.55, size=6),
            name="Predictions",
            hovertemplate="Actual: $%{x:,.2f}<br>Predicted: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color=THEME_COLORS["danger"], dash="dash", width=2),
            name="Ideal Fit (y = x)",
        )
    )

    fig.update_layout(
        title=f"Actual vs Predicted Revenue: {model_name}",
        xaxis_title="Actual Sales Revenue ($ USD)",
        yaxis_title="Predicted Sales Revenue ($ USD)",
        xaxis=dict(range=[0, max_val]),
        yaxis=dict(range=[0, max_val]),
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_residuals_distribution(
    residuals: np.ndarray, model_name: str = "Linear Regression"
) -> go.Figure:
    """Renders residual distribution histogram to inspect normality and zero-mean bias."""
    res = np.asarray(residuals)
    res_trimmed = res[(res >= np.percentile(res, 2)) & (res <= np.percentile(res, 98))]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=res_trimmed,
            nbinsx=50,
            marker_color=THEME_COLORS["lr_color"],
            opacity=0.75,
            name="Residuals (y - y_hat)",
            hovertemplate="Residual: $%{x:,.1f}<br>Count: %{y}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color=THEME_COLORS["danger"])

    fig.update_layout(
        title=f"Residual Error Distribution: {model_name}",
        xaxis_title="Residual Error ($ USD: Actual - Predicted)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_residuals_vs_predicted(
    y_pred: np.ndarray, residuals: np.ndarray
) -> go.Figure:
    """
    Renders Residuals vs Predicted scatter plot to test the Homoscedasticity assumption
    (constant variance across fitted values).
    """
    y_p = np.asarray(y_pred)
    res = np.asarray(residuals)

    if len(y_p) > 1000:
        idx = np.random.RandomState(42).choice(len(y_p), 1000, replace=False)
        y_p = y_p[idx]
        res = res[idx]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_p,
            y=res,
            mode="markers",
            marker=dict(color=THEME_COLORS["primary"], opacity=0.5, size=5),
            name="Residuals",
            hovertemplate="Fitted: $%{x:,.1f}<br>Residual: $%{y:,.1f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_width=2, line_dash="dash", line_color=THEME_COLORS["danger"])

    fig.update_layout(
        title="Residuals vs Fitted Values (Homoscedasticity Diagnostic)",
        xaxis_title="Predicted Sales Revenue ($ USD)",
        yaxis_title="Residual (Actual - Predicted) ($ USD)",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig
