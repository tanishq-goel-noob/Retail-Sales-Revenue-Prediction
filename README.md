# Retail Sales Revenue Prediction System
### Predictive Modelling and Revenue Forecasting Using Ordinary Least Squares (OLS) Linear Regression

---

## 1. Project Overview
The **Retail Sales Revenue Prediction System** is an end-to-end Machine Learning web application designed to forecast transaction-level retail sales revenue from historical transactional, category, and seasonal features. Built specifically for third-year B.Tech Artificial Intelligence & Machine Learning coursework under **Predictive Modelling**, this project implements, evaluates, and deploys an **Ordinary Least Squares (OLS) Linear Regression** pipeline.

The entire system is deployed via an interactive, executive-grade **Streamlit** dashboard that provides complete pipeline transparency: from exploratory data analysis and leak-free feature engineering to diagnostic residual distributions and live forward-looking revenue forecasting.

---

## 2. Problem Statement
In retail commerce, accurate sales revenue forecasting is fundamental for inventory planning, cash-flow budgeting, dynamic pricing, and warehouse logistics. Overestimating future revenue leads to capital lockup and dead inventory; underestimating revenue causes severe stockouts and lost market share. 

This project addresses the core predictive modeling question:
> **Can historical retail transaction patterns, seasonal calendar indicators, and promotional discounts accurately forecast future sales revenue using Linear Regression without target leakage? How do OLS beta weights reveal the marginal dollar impact of each retail factor?**

---

## 3. Key Coursework Questions Answered
1. **Predictability:** Can we predict future retail sales revenue using historical transaction metadata? *(Yes, Linear Regression captures baseline revenue regimes, category elasticity, and promotional effects).*
2. **Model Accuracy:** How accurately does Linear Regression project revenue? *(Achieved test RMSE \$453.10, test MAE \$209.92, and $R^2$ of 0.2033 on unobserved future orders).*
3. **Generalization:** Does the linear model overfit? *(No; training RMSE is \$375.40 and testing RMSE is \$453.10, demonstrating stable generalization).*
4. **Prediction Errors:** How much error does the model produce across MAE, MSE, RMSE, R², and MAPE?
5. **Feature Attribution:** Which features most heavily influence sales predictions? *(Historical subcategory demand benchmarks, product category, and promotional discount).*
6. **Residual Diagnostics:** Do residuals satisfy classical regression assumptions? *(Residuals center tightly at zero, with slight right-tail variance due to high-value retail orders).*
7. **Business Utility:** How can retail managers use predicted revenue to balance promotional markdown campaigns and inventory allocation?

---

## 4. Dataset Details & Integrity
The application utilizes the authentic, benchmark **Tableau Sample Superstore Retail Sales Dataset** (publicly archived on Tableau Public and Kaggle open-data repositories).

- **Data Source:** Tableau Community Benchmark / Kaggle Open Data
- **Records (N):** 10,800 transactional line items
- **Temporal Span:** Multi-year chronological sequence across 4 US geographic territories
- **Target Variable (y):** `Sales` (Gross monetary transaction revenue in USD)
- **Features Used:**
  - `Order Date` $\rightarrow$ Engineered into `Year`, `Month`, `Day`, `DayOfWeek`, `Quarter`, `IsWeekend`, `Season`
  - `Category` (Furniture, Office Supplies, Technology)
  - `Sub-Category` (17 distinct product taxonomies: Chairs, Phones, Storage, Tables, Copiers, etc.)
  - `Region` (West, East, Central, South)
  - `Segment` (Consumer, Corporate, Home Office)
  - `Ship Mode` (Standard Class, Second Class, First Class, Same Day)
  - `Discount` (Promotional markdown rate, 0.0 to 0.8)
  - `Hist_SubCat_Avg_Sales` (Historical subcategory expanding revenue benchmark)

### ⚠️ Strict Target Leakage Prevention Policy
In commercial retail accounting:
$$\text{Sales} = \text{Unit Price} \times \text{Quantity} \times (1 - \text{Discount})$$
Feeding realized `Quantity` or `Unit Price` simultaneously into the model creates an exact mathematical accounting identity (target leakage). Similarly, using future actual sales or future shipping dates peeks ahead in time. 
Our pipeline **strictly excludes `Quantity` and `Profit`** from model inputs and constructs historical category benchmarks using a `shift(1)` expanding window so that transaction $t$ only uses information available strictly prior to time $t$.

---

## 5. Technology Stack
- **Core Language:** Python 3.12
- **Data Manipulation:** `pandas>=2.0.0`, `numpy>=1.24.0`
- **Machine Learning Harness:** `scikit-learn>=1.3.0` (`Pipeline`, `ColumnTransformer`, `LinearRegression`, `OneHotEncoder`, `StandardScaler`, `SimpleImputer`)
- **Model Serialization:** `joblib>=1.3.0`
- **Web UI & Serving:** `streamlit>=1.30.0`
- **Interactive Visualizations:** `plotly>=5.18.0`, `matplotlib>=3.7.0`, `seaborn>=0.12.0`

---

## 6. End-to-End System Architecture

```text
               ┌──────────────────────────────────────┐
               │  Raw Retail Dataset (Superstore CSV)  │
               └──────────────────┬───────────────────┘
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │    Chronological Sorting (by Date)   │
               └──────────────────┬───────────────────┘
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │  Leakage-Free Feature Engineering    │
               │  - Calendar: Year, Month, Day, DOW   │
               │  - Season: Winter, Spring, etc.      │
               │  - Lag-1 Sub-Category Benchmark      │
               └──────────────────┬───────────────────┘
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │    Chronological 80/20 Partition     │
               │    (Past 80% Train | Future 20% Test)│
               └─────────┬──────────────────┬─────────┘
                         │                  │
                         ▼                  ▼
               ┌──────────────────┐ ┌──────────────────┐
               │  Training Split  │ │   Testing Split  │
               └─────────┬────────┘ └────────┬─────────┘
                         │                   │
                         ▼                   │
               ┌──────────────────┐          │
               │ ColumnTransformer│          │
               │ (Fitted on Train)│          │
               │ - Median Imputer │          │
               │ - StandardScaler │          │
               │ - OneHotEncoder  │          │
               └─────────┬────────┘          │
                         │                   │
                         ▼                   │
               ┌──────────────────┐          │
               │Linear Regression │          │
               │(Ordinary Least   │          │
               │ Squares Fit)     │          │
               └─────────┬────────┘          │
                         │                   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                  ┌──────────────────────────────────┐
                  │ Benchmark on Unseen Testing Set  │
                  │ - Compute MAE, MSE, RMSE, R2     │
                  │ - Protected MAPE Formulation     │
                  │ - Residual Normality Diagnostics │
                  └────────────────┬─────────────────┘
                                   │
                                   ▼
                  ┌──────────────────────────────────┐
                  │  Joblib Pipeline Serialization   │
                  └────────────────┬─────────────────┘
                                   │
                                   ▼
                  ┌──────────────────────────────────┐
                  │ Multi-Page Streamlit Application │
                  │  (Analytics, Auditing & Serving) │
                  └──────────────────────────────────┘
```

---

## 7. Mathematical Foundations of Linear Regression

Linear Regression models target sales revenue as a linear superposition of standardized input features:
$$\hat{Y} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p$$

### Ordinary Least Squares (OLS) Formulation
The coefficient vector $\mathbf{\beta}$ is solved analytically by minimizing the Residual Sum of Squares (RSS):
$$\text{RSS}(\mathbf{\beta}) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \sum_{i=1}^n (y_i - \mathbf{x}_i^T \mathbf{\beta})^2 = (\mathbf{y} - \mathbf{X}\mathbf{\beta})^T (\mathbf{y} - \mathbf{X}\mathbf{\beta})$$

Setting the partial derivative with respect to $\mathbf{\beta}$ to zero:
$$\frac{\partial \text{RSS}}{\partial \mathbf{\beta}} = -2 \mathbf{X}^T (\mathbf{y} - \mathbf{X}\mathbf{\beta}) = \mathbf{0} \implies \mathbf{X}^T \mathbf{X} \mathbf{\beta} = \mathbf{X}^T \mathbf{y}$$

Assuming $\mathbf{X}^T \mathbf{X}$ is non-singular, multiplying both sides by $(\mathbf{X}^T \mathbf{X})^{-1}$ gives the closed-form **Normal Equation**:
$$\hat{\mathbf{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

### Gauss-Markov Theorem & Optimality Guarantee
Under the classical Gauss-Markov assumptions (linearity, strict exogeneity $E[\epsilon|\mathbf{X}] = 0$, spherical errors $\text{Var}(\epsilon|\mathbf{X}) = \sigma^2 \mathbf{I}$, and full column rank), the OLS estimator $\hat{\mathbf{\beta}}$ is the **Best Linear Unbiased Estimator (BLUE)**—possessing minimum variance among all linear unbiased estimators.

---

## 8. Evaluation Metrics Formulations

| Metric | Mathematical Formula | Preferred Direction | Unit |
|---|---|---|---|
| **Mean Absolute Error (MAE)** | $\text{MAE} = \frac{1}{n}\sum_{i=1}^n \|y_i - \hat{y}_i\|$ | Lower is better | Dollars (\$) |
| **Mean Squared Error (MSE)** | $\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$ | Lower is better | Squared Dollars (\$^2) |
| **Root Mean Squared Error (RMSE)** | $\text{RMSE} = \sqrt{\text{MSE}}$ | Lower is better *(Primary)* | Dollars (\$) |
| **R-Squared ($R^2$) Score** | $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Higher is better | Unitless $(-\infty, 1]$ |
| **Mean Absolute Percentage Error (MAPE)** | $\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^n \left\|\frac{y_i - \hat{y}_i}{y_i}\right\|$ | Lower is better | Percentage (%) |

---

## 9. Empirical Generalization Results

| Metric | Training Split (Past 80%) | Testing Split (Future 20%) | Generalization Gap | Interpretation |
|---|---|---|---|---|
| **MAE** | \$183.18 | \$209.92 | +14.59% | Robust average absolute deviation in dollars |
| **MSE** | 140,928.3 | 205,296.1 | +45.67% | Average squared error (punishes large misses) |
| **RMSE** | \$375.40 | \$453.10 | +20.69% | Primary dollar-unit metric; stable generalization |
| **$R^2$ Score** | 0.2312 | 0.2033 | -12.06% | Accounts for 20.33% of test variance vs naive mean |
| **MAPE** | 439.12% | 422.57% | -3.77% | Relative percentage error (stabilized with $\epsilon$) |

---

## 10. Project Directory Structure

```text
SalesPredictionModel/
│
├── app.py                         # Main Streamlit Executive Dashboard entry point
├── requirements.txt               # Locked project dependencies
├── README.md                      # Comprehensive academic & deployment manual
├── PROJECT_LEARNING_GUIDE.md      # In-depth viva voce & theory preparation guide
├── .gitignore                     # Git configuration ignoring cache & artifacts
│
├── data/
│   └── retail_sales.csv           # Authentic 10,800-record Superstore dataset
│
├── models/
│   ├── linear_regression.joblib   # Trained Scikit-learn Pipeline (LR + Preprocessor)
│   └── model_metadata.joblib      # Serialization of benchmarks and evaluation metrics
│
├── src/
│   ├── __init__.py                # Package initialization
│   ├── data_loader.py             # Ingestion, schema validation & dataset metadata
│   ├── preprocessing.py          # ColumnTransformer, IQR outlier analysis & splitting
│   ├── feature_engineering.py     # Calendar periodicity & leakage-free lag benchmarks
│   ├── train_models.py            # OLS fitting, coefficient extraction, and serialization
│   ├── evaluation.py              # MAE, MSE, RMSE, R2, MAPE & generalization tables
│   ├── visualization.py           # Plotly interactive & diagnostic plotting suite
│   └── prediction.py              # Validated inference wrapper for live forecasts
│
├── pages/
│   ├── 1_Dashboard.py             # Executive revenue KPIs & transaction ledger
│   ├── 2_Dataset_Explorer.py      # Schema audit, summary statistics & CSV uploader
│   ├── 3_EDA.py                   # Rigorous exploratory visual data analysis
│   ├── 4_Preprocessing.py         # Step-by-step educational pipeline walkthrough
│   ├── 5_Model_Training.py        # Interactive training & standardized beta weights
│   ├── 6_Model_Evaluation.py      # Generalization matrix, goodness-of-fit & residuals
│   ├── 7_Predict_Revenue.py       # Live interactive order revenue forecasting form
│   └── 8_About.py                 # Academic problem statement, architecture & scope
│
└── notebooks/
    └── exploratory_analysis.ipynb # Interactive Jupyter experimentation notebook
```

---

## 11. Installation & Execution Instructions

### Prerequisites
- Python 3.10+ (Python 3.12 recommended)
- Git & PowerShell / Bash terminal

### Step 1: Open Terminal in Project Directory
```powershell
cd c:\Users\goelt\OneDrive\Desktop\SalesPredictionModel
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Streamlit Application
```powershell
streamlit run app.py
```
Upon execution, your default web browser will automatically open the interactive dashboard at:
```text
http://localhost:8501
```

---
**Author:** Third-Year B.Tech AI & ML Student  
**Course:** Predictive Modelling  
**Supervisor / Mentor:** Senior ML Engineering Faculty  
