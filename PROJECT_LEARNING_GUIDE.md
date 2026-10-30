# Complete Viva Voce & Conceptual Learning Guide
## Retail Sales Revenue Prediction System: Machine Learning & Predictive Modelling

---

> **Purpose of this Guide:**  
> This comprehensive document is crafted specifically for third-year B.Tech Artificial Intelligence and Machine Learning students preparing for academic project presentations, examinations, and viva voce defenses. Every theoretical concept, formula, derivation, and implementation decision is explained from foundational first principles with concrete numerical examples.

---

# Table of Contents
1. [What is Predictive Modelling?](#1-what-is-predictive-modelling)
2. [What is Regression? (Classification vs. Regression)](#2-what-is-regression-classification-vs-regression)
3. [Linear Regression Explained from First Principles](#3-linear-regression-explained-from-first-principles)
   - Mathematical Formulation
   - Ordinary Least Squares (OLS) Optimization
   - Step-by-Step Derivation of the Normal Equation
   - Geometric Interpretation: Orthogonal Projections
   - Core Assumptions of Linear Regression
   - Gauss-Markov Theorem & BLUE Properties
4. [Data Preprocessing Pipeline](#4-data-preprocessing-pipeline)
   - Missing Value Imputation (Median vs. Mean)
   - Outlier Detection via Interquartile Range (IQR)
   - Categorical Encoding (One-Hot vs. Label Encoding)
   - Feature Standardization (StandardScaler & Condition Numbers)
5. [Feature Engineering & Target Leakage Prevention](#5-feature-engineering--target-leakage-prevention)
   - The Dangers of Accounting Identities
   - Leakage-Free Historical Demand Benchmarks
6. [Chronological Train-Test Splitting](#6-chronological-train-test-splitting)
   - Why Random Shuffling Fails in Sales Forecasting
7. [Overfitting vs. Underfitting (The Bias-Variance Tradeoff)](#7-overfitting-vs-underfitting-the-bias-variance-tradeoff)
8. [What are Residuals?](#8-what-are-residuals)
   - Homoscedasticity vs. Heteroscedasticity
9. [Comprehensive Regression Evaluation Metrics (With Numerical Examples)](#9-comprehensive-regression-evaluation-metrics-with-numerical-examples)
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - R-Squared ($R^2$) Score
   - Mean Absolute Percentage Error (MAPE)
   - Step-by-Step Hand-Calculation Table
10. [Interpreting Linear Regression Beta Coefficients](#10-interpreting-linear-regression-beta-coefficients)
11. [How Streamlit Works Internally](#11-how-streamlit-works-internally)
12. [Top 15 Likely Viva Voce Questions & Model Answers](#12-top-15-likely-viva-voce-questions--model-answers)

---

# 1. What is Predictive Modelling?

**Predictive Modelling** is the statistical and mathematical practice of building a mathematical function $f$ using historical observed data that can accurately estimate an unknown outcome variable $Y$ when presented with new, unobserved input features $\mathbf{X}$:

$$Y = f(\mathbf{X}) + \epsilon$$

Where:
- $\mathbf{X} = [X_1, X_2, \dots, X_p]$ represents the vector of **independent features** (e.g., product subcategory, promotional discount, geographic region, month).
- $Y$ is the **dependent target variable** (in our project, gross retail sales revenue in USD).
- $f$ is the mapping function learned by the machine learning algorithm.
- $\epsilon$ is irreducible random noise (uncontrollable real-world factors such as weather, local economic shocks, or individual buyer impulse).

---

# 2. What is Regression? (Classification vs. Regression)

In Supervised Machine Learning, algorithms are broadly divided into two tasks:

| Dimension | Classification | Regression (Our Project) |
|---|---|---|
| **Target Type** | Discrete / Qualitative labels (e.g., Spam vs. Not Spam, Fraud vs. Legitimate) | Continuous / Quantitative real numbers (e.g., Dollar Revenue, House Price, Temperature) |
| **Output Space** | $Y \in \{C_1, C_2, \dots, C_k\}$ | $Y \in \mathbb{R}$ (Real continuous domain) |
| **Loss Functions** | Cross-Entropy Loss, Hinge Loss | Mean Squared Error (MSE), Mean Absolute Error (MAE) |
| **Evaluation Metrics**| Accuracy, Precision, Recall, F1-Score, ROC-AUC | MAE, MSE, RMSE, R², MAPE |

---

# 3. Linear Regression Explained from First Principles

## 3.1 Mathematical Formulation
Linear Regression hypothesizes that the target variable $Y$ is a linear combination of the input variables:

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p + \epsilon$$

In vector notation:
$$Y = \mathbf{x}^T \mathbf{\beta} + \epsilon$$

Where:
- $\beta_0$ is the **intercept** (the predicted revenue when all standardized input features $X_j = 0$).
- $\beta_j$ is the **partial regression coefficient** for feature $X_j$. It represents the expected change in revenue $Y$ for a 1-unit increase in $X_j$, holding all other features strictly constant (*ceteris paribus*).

## 3.2 Ordinary Least Squares (OLS) Optimization
To learn the optimal parameter vector $\hat{\mathbf{\beta}}$, the algorithm minimizes the **Residual Sum of Squares (RSS)**:

$$\text{RSS}(\mathbf{\beta}) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \sum_{i=1}^n (y_i - \mathbf{x}_i^T \mathbf{\beta})^2$$

In matrix form:
$$\text{RSS}(\mathbf{\beta}) = (\mathbf{y} - \mathbf{X}\mathbf{\beta})^T (\mathbf{y} - \mathbf{X}\mathbf{\beta}) = \mathbf{y}^T \mathbf{y} - 2 \mathbf{\beta}^T \mathbf{X}^T \mathbf{y} + \mathbf{\beta}^T \mathbf{X}^T \mathbf{X} \mathbf{\beta}$$

## 3.3 Step-by-Step Derivation of the Normal Equation
To find the vector $\mathbf{\beta}$ that minimizes $\text{RSS}(\mathbf{\beta})$, we take the matrix derivative with respect to $\mathbf{\beta}$ and set it to the zero vector $\mathbf{0}$:

$$\frac{\partial \text{RSS}}{\partial \mathbf{\beta}} = -2 \mathbf{X}^T \mathbf{y} + 2 \mathbf{X}^T \mathbf{X} \mathbf{\beta} = \mathbf{0}$$

Dividing by 2 and rearranging:
$$\mathbf{X}^T \mathbf{X} \mathbf{\beta} = \mathbf{X}^T \mathbf{y}$$

Assuming $\mathbf{X}^T \mathbf{X}$ is invertible (non-singular, which requires full column rank and no perfect multicollinearity), multiplying both sides by $(\mathbf{X}^T \mathbf{X})^{-1}$ gives the closed-form **Normal Equation**:

$$\hat{\mathbf{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

## 3.4 Geometric Interpretation: Orthogonal Projections
Geometrically, the vector of observed targets $\mathbf{y} \in \mathbb{R}^n$ does not generally lie in the subspace spanned by the columns of the feature matrix $\mathbf{X}$ (the column space $\text{col}(\mathbf{X})$).  
Ordinary Least Squares projects $\mathbf{y}$ orthogonally onto $\text{col}(\mathbf{X})$. The prediction vector $\hat{\mathbf{y}} = \mathbf{X}\hat{\mathbf{\beta}} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1}\mathbf{X}^T \mathbf{y} = \mathbf{H}\mathbf{y}$ is the closest point in the column space to $\mathbf{y}$. The matrix $\mathbf{H} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1}\mathbf{X}^T$ is known as the **Hat Matrix** (or orthogonal projection matrix) because it puts the "hat" on $\mathbf{y}$.

## 3.5 Core Assumptions of Linear Regression
1. **Linearity in Parameters:** The relationship between predictors and the mean of the response variable is linear: $E[Y|\mathbf{X}] = \mathbf{X}\mathbf{\beta}$.
2. **Strict Exogeneity:** The expected value of error terms conditional on the predictors is zero: $E[\epsilon_i | \mathbf{X}] = 0$. This implies no omitted variable bias.
3. **No Multicollinearity:** $\mathbf{X}^T \mathbf{X}$ has full column rank $p$. No independent feature is an exact linear combination of other features.
4. **Spherical Errors (Homoscedasticity & No Autocorrelation):** Constant error variance across all observations ($\text{Var}(\epsilon_i) = \sigma^2$) and zero covariance between distinct errors ($\text{Cov}(\epsilon_i, \epsilon_j) = 0$ for $i \ne j$).
5. **Normality of Errors (for Hypothesis Testing):** $\epsilon \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$.

## 3.6 Gauss-Markov Theorem & BLUE Properties
The celebrated **Gauss-Markov Theorem** states that under assumptions 1 through 4, the OLS estimator $\hat{\mathbf{\beta}}$ is the **Best Linear Unbiased Estimator (BLUE)**:
- **Linear:** $\hat{\mathbf{\beta}} = \mathbf{C}\mathbf{y}$, where $\mathbf{C} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T$.
- **Unbiased:** $E[\hat{\mathbf{\beta}}] = \mathbf{\beta}$.
- **Best (Minimum Variance):** For any other linear unbiased estimator $\tilde{\mathbf{\beta}}$, $\text{Var}(\tilde{\mathbf{\beta}}) - \text{Var}(\hat{\mathbf{\beta}})$ is a positive semi-definite matrix.

---

# 4. Data Preprocessing Pipeline

## 4.1 Missing Value Imputation
- **Numerical Features (Continuous):** Imputed with the **Median**.  
  *Why?* The median is non-parametric and resistant to extreme skewness or outliers, unlike the sample mean which is pulled toward heavy tails.
- **Categorical Features (Discrete):** Imputed with the **Mode (Most Frequent)** category.

## 4.2 Outlier Detection via Interquartile Range (IQR)
1. Sort data and find the 25th percentile ($Q_1$) and 75th percentile ($Q_3$).
2. Calculate the spread: $\text{IQR} = Q_3 - Q_1$.
3. Compute outer bounds (Tukey's fences):
   $$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}$$
   $$\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$

*Why do we NOT blindly delete retail sales outliers?*  
In retail, a $\$3,000$ bulk copier sale is **genuine business revenue**, not a sensor glitch. Deleting high-value transactions removes valuable top-line revenue information.

## 4.3 Categorical Encoding (One-Hot vs. Label Encoding)
Machine learning algorithms perform linear algebra and matrix multiplications; they cannot multiply strings like `"Furniture"`.
- **One-Hot Encoding:** Creates binary columns $[0, 1]$ for each category. It creates orthogonal dimensions without imposing an artificial ordinal ranking.
- **Why NOT Label Encoding for nominal categories?** Encoding `[Furniture=1, Office Supplies=2, Technology=3]` implies that `Technology > Furniture` and that `Office Supplies` is the arithmetic midpoint between Furniture and Tech—which is mathematically invalid.

## 4.4 Feature Scaling (StandardScaler)
StandardScaler transforms continuous variables using $z$-scores:
$$z = \frac{x - \mu}{\sigma}$$
- **Linear Regression strongly requires scaling:** Unscaled features with vastly different ranges (e.g. Year $\sim 2017$ vs Discount $\sim 0.2$) cause the matrix $\mathbf{X}^T \mathbf{X}$ to have a very large condition number (ill-conditioned matrix), making numerical inversion $(\mathbf{X}^T \mathbf{X})^{-1}$ computationally unstable.
- Standardizing also puts all coefficients on a standardized scale, enabling direct comparison of feature importance via coefficient magnitude.

---

# 5. Feature Engineering & Target Leakage Prevention

## 5.1 The Danger of Accounting Identities (Target Leakage)
In retail accounting:
$$\text{Sales} = \text{Unit Price} \times \text{Quantity} \times (1 - \text{Discount})$$
If an ML model is given both realized `Quantity` and `Unit Price` to predict `Sales`, the model does not learn a predictive demand pattern; it simply discovers the arithmetic formula!  
Furthermore, at planning time before an order occurs, the manager **does not know how many units the customer will purchase**. Therefore, feeding `Quantity` into the feature set represents severe **Target Leakage**.

## 5.2 Leakage-Free Historical Demand Benchmarks
To give the linear model historical context without peeking into the future:
1. Sort the entire dataset chronologically by `Order Date`.
2. Compute an expanding mean of past sales for each subcategory shifted by 1:
   $$\text{Hist\_Benchmark}_t = \frac{1}{t-1} \sum_{i=1}^{t-1} \text{Sales}_i$$
3. At row $t$, the feature only contains the average of transactions $1$ through $t-1$. It never sees transaction $t$'s actual sales.

---

# 6. Chronological Train-Test Splitting

In standard cross-validation, data is shuffled randomly. In sales forecasting, **random shuffling is an error**:
- Shuffling mixes past and future orders.
- The model trains on transactions from November 2017 to predict transactions from February 2017 (peeking into the future!).
- **Chronological Splitting:** Enforces an authentic temporal split:
  - **Training Set (First 80%):** Past historical years.
  - **Testing Set (Final 20%):** Unobserved future quarters.

---

# 7. Overfitting vs. Underfitting (The Bias-Variance Tradeoff)

- **Underfitting (High Bias):** The model is too simple to capture the underlying structure (e.g., fitting a flat line through highly cyclical seasonal data). It performs poorly on both training and testing data.
- **Overfitting (High Variance):** The model memorizes training noise and outliers rather than generalizing. It achieves near-zero training error but high test error.
- In our project, Linear Regression achieves **low variance** (Training RMSE is \$375.40 and Testing RMSE is \$453.10, showing consistent generalization) with **moderate bias** due to the linear approximation of retail price elasticity.

---

# 8. What are Residuals?

A **residual** $e_i$ is the vertical difference between the actual observed value $y_i$ and the model's predicted value $\hat{y}_i$:

$$e_i = y_i - \hat{y}_i$$

- If $e_i > 0$: The model **under-predicted** (actual revenue was higher than predicted).
- If $e_i < 0$: The model **over-predicted** (actual revenue was lower than predicted).
- If $e_i = 0$: The prediction was perfect.

### Homoscedasticity vs. Heteroscedasticity
- **Homoscedasticity:** Error terms have constant variance across all levels of the predicted outcome ($\text{Var}(\epsilon_i) = \sigma^2$).
- **Heteroscedasticity:** Error variance expands as predicted values grow (common in monetary data where large transactions have larger absolute error spreads). We plot Residuals vs. Fitted values in Page 6 to audit this diagnostic.

---

# 9. Comprehensive Regression Evaluation Metrics (With Numerical Examples)

Let's illustrate each formula using a small numerical example of $n = 4$ test transactions:

| Observation $i$ | Actual Revenue ($y_i$) | Predicted Revenue ($\hat{y}_i$) | Error ($y_i - \hat{y}_i$) | Absolute Error $|y_i - \hat{y}_i|$ | Squared Error $(y_i - \hat{y}_i)^2$ | Absolute Pct Error $|(y_i - \hat{y}_i)/y_i|$ |
|---|---|---|---|---|---|---|
| 1 | \$100 | \$110 | -\$10 | \$10 | 100 | $10/100 = 0.10$ (10%) |
| 2 | \$200 | \$180 | +\$20 | \$20 | 400 | $20/200 = 0.10$ (10%) |
| 3 | \$50 | \$50 | \$0 | \$0 | 0 | $0/50 = 0.00$ (0%) |
| 4 | \$500 | \$400 | +\$100 | \$100 | 10,000 | $100/500 = 0.20$ (20%) |
| **Sum ($\sum$)** | **\$850** | **\$740** | **+\$110** | **\$130** | **10,500** | **0.40** |

Actual mean: $\bar{y} = \frac{850}{4} = \$212.50$

---

### A. Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i| = \frac{130}{4} = \mathbf{\$32.50}$$
**Meaning:** On average, predictions deviate from actual revenue by \$32.50.

---

### B. Mean Squared Error (MSE)
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \frac{10,500}{4} = \mathbf{2,625.00 \text{ \$}^2}$$
**Meaning:** Average squared error. Notice how observation 4 (error of \$100) contributes $10,000$ to the sum ($95\%$ of total MSE!). MSE heavily punishes large misses.

---

### C. Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{2,625.00} \approx \mathbf{\$51.23}$$
**Meaning:** Expresses the squared penalty back in original dollar units. Because RMSE (\$51.23) is higher than MAE (\$32.50), we know the model made several large outlier errors.

---

### D. R-Squared ($R^2$) Score
$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

Let's compute Total Sum of Squares ($\text{SS}_{\text{tot}}$) around mean $\bar{y} = 212.5$:
- $(100 - 212.5)^2 = (-112.5)^2 = 12,656.25$
- $(200 - 212.5)^2 = (-12.5)^2 = 156.25$
- $(50 - 212.5)^2 = (-162.5)^2 = 26,406.25$
- $(500 - 212.5)^2 = (287.5)^2 = 82,656.25$
- $\text{SS}_{\text{tot}} = 12,656.25 + 156.25 + 26,406.25 + 82,656.25 = 121,875.00$

$$R^2 = 1 - \frac{10,500}{121,875} = 1 - 0.08615 = \mathbf{0.9138 \text{ (or } 91.38\%)}$$
**Meaning:** The model accounts for $91.38\%$ of the total variance in sales revenue compared to a simple horizontal line predicting the mean.

---

### E. Mean Absolute Percentage Error (MAPE)
$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right| = \frac{100\%}{4} \times (0.10 + 0.10 + 0.00 + 0.20) = \frac{40\%}{4} = \mathbf{10.0\%}$$
**Meaning:** On average, the model's forecast deviates from the true order value by $10\%$.

---

# 10. Interpreting Linear Regression Beta Coefficients

In our pipeline, continuous features are standardized via `StandardScaler` ($z = (x - \mu)/\sigma$).
- **Magnitude:** A coefficient $\beta_j$ of $+120.50$ means that for every 1-standard-deviation increase in that feature, expected sales revenue increases by $\$120.50$, holding all other features constant.
- **Direction (Sign):**
  - **Positive Coefficients:** Historical Subcategory Demand, Technology category, Q4 Month (November/December holiday rush) increase revenue.
  - **Negative Coefficients:** Promotional Discount has a negative weight, reflecting reduced dollar realization per unit item.

---

# 11. How Streamlit Works Internally

Streamlit operates on an execution-flow paradigm:
1. **Script Re-Run Execution:** Whenever the user interacts with any widget (e.g., changes a slider or clicks a button), Streamlit **re-executes the Python script from top to bottom**.
2. **Session State (`st.session_state`):** A persistent key-value store that survives across re-runs. We use it to retain the trained model object and diagnostic matrices so retraining doesn't execute unnecessarily on every click.
3. **Caching Decorators:**
   - `@st.cache_data`: Caches serializable data loads (e.g., `load_dataset()`). Prevents reading the 2MB CSV from disk repeatedly.
   - `@st.cache_resource`: Caches complex global objects (e.g., ML models, database connections).

---

# 12. Top 15 Likely Viva Voce Questions & Model Answers

### Q1: What is the main objective of your project?
> **Answer:** "The objective is to develop an end-to-end Machine Learning system that predicts future retail sales revenue from historical transactional, category, and seasonal metadata using Ordinary Least Squares (OLS) Linear Regression, and to evaluate its generalization and assumptions on chronological test data."

### Q2: Why did you choose Linear Regression as your predictive model?
> **Answer:** "Linear Regression is the foundational parametric model in predictive modelling. It provides complete transparency through its mathematical formulation, closed-form Normal Equation, and interpretable Beta coefficients that quantify the exact marginal contribution of each retail feature."

### Q3: How does Ordinary Least Squares (OLS) solve for the optimal coefficients?
> **Answer:** "OLS minimizes the Residual Sum of Squares: $\text{RSS} = \sum (y_i - \mathbf{x}_i^T \mathbf{\beta})^2$. By taking the derivative with respect to $\mathbf{\beta}$ and setting it to zero, we obtain the Normal Equation: $\mathbf{X}^T \mathbf{X} \mathbf{\beta} = \mathbf{X}^T \mathbf{y}$. Multiplying by $(\mathbf{X}^T \mathbf{X})^{-1}$ gives the analytical solution $\hat{\mathbf{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$."

### Q4: What is the Gauss-Markov Theorem?
> **Answer:** "The Gauss-Markov Theorem proves that under standard conditions (linearity, strict exogeneity, homoscedasticity, no autocorrelation, and full rank), the OLS estimator is the Best Linear Unbiased Estimator (BLUE), meaning it has the lowest sampling variance among all possible linear unbiased estimators."

### Q5: What is target leakage and how did you prevent it?
> **Answer:** "Target leakage occurs when training data includes features that reveal the target variable or would not be available at inference time. In retail, $\text{Sales} = \text{Price} \times \text{Quantity} \times (1 - \text{Discount})$. If we fed both quantity and price into the model, it would learn a trivial accounting identity. We excluded realized quantity and profit, and engineered historical category demand using an expanding mean shifted by 1 so no future data is leaked."

### Q6: Why did you use chronological train-test splitting instead of random train-test splitting?
> **Answer:** "Sales revenue is a time-dependent sequence. Randomly shuffling transactions allows data from future months to leak into the training set to predict earlier orders. Chronological splitting enforces the realistic business task: train strictly on past quarters, and evaluate strictly on future unseen quarters."

### Q7: Why do we use StandardScaler for Linear Regression?
> **Answer:** "Linear Regression solves the matrix inversion $(\mathbf{X}^T \mathbf{X})^{-1}$. If features have vastly different numerical scales (e.g. Year $\sim 2017$ vs Discount $\sim 0.2$), the matrix becomes ill-conditioned, causing numerical instability. Scaling also makes coefficients directly comparable on a standardized scale."

### Q8: What does the intercept ($\beta_0$) in Linear Regression represent?
> **Answer:** "The intercept ($\beta_0$) represents the expected baseline revenue when all input features equal zero. Because continuous features are standardized to mean zero, $\beta_0$ represents the predicted sales revenue for an order with average continuous values in the baseline reference category."

### Q9: What is the primary metric you used to evaluate your model, and why?
> **Answer:** "We used Root Mean Squared Error (RMSE) as our primary metric, supported by MAE and R². RMSE is measured in original dollar units and squares the errors before averaging, penalizing large forecasting blunders heavily. In retail operations, large forecasting misses cause severe stockouts or excess inventory carrying costs, making RMSE the most business-critical metric."

### Q10: Can R² be negative? If yes, what does it mean?
> **Answer:** "Yes, R² can be negative on test data. R² compares the model's residual sum of squares against the total sum of squares around the mean. If the model's predictions on test data produce greater error than simply predicting the training mean for every order, $\text{SS}_{\text{res}} > \text{SS}_{\text{tot}}$, resulting in a negative R²."

### Q11: What is the difference between MAE and RMSE?
> **Answer:** "MAE takes the absolute value of errors, giving every dollar of error equal linear weight. RMSE squares errors before averaging and taking the square root, giving disproportionately greater weight to large forecasting misses. RMSE is always greater than or equal to MAE; a large difference between them indicates the presence of large outlier errors."

### Q12: Why can MAPE be misleading in sales forecasting?
> **Answer:** "MAPE divides by the actual value: $\frac{|y - \hat{y}|}{y}$. If an order has a very small revenue value (e.g., a \$1.50 stationery item), even a \$5 error produces a $333\%$ percentage error. If $y = 0$, MAPE divides by zero and is undefined. We implemented numerical epsilon protections to stabilize MAPE."

### Q13: What are residuals, and what should their distribution look like?
> **Answer:** "A residual is $y_i - \hat{y}_i$ (actual minus predicted revenue). In a well-fitted model, residuals should be normally distributed around a mean of zero with constant variance (homoscedasticity). In our application, we plot residual distribution histograms and Residuals vs. Fitted scatter plots to visually verify that the model exhibits zero systemic directional bias."

### Q14: How does your application ensure user inputs at prediction time are valid?
> **Answer:** "Our prediction module (`RetailSalesPredictor`) passes inputs through the exact same feature engineering pipeline used during training. It extracts calendar features from the selected date, maps the chosen subcategory to historical expanding benchmarks, formats the data into the identical schema, and clips negative outputs from Linear Regression to zero, ensuring reliable and domain-valid outputs."

### Q15: What are the main limitations of Linear Regression for retail sales forecasting?
> **Answer:** "The main limitation is the linearity constraint: retail consumer reactions to steep discounts or bulk purchases often follow non-linear curves. Additionally, because OLS squares residuals, extreme high-value orders (\$2,000+) can pull the fitted line toward the tail. In future work, log-transforms or regularized regression (Ridge/Lasso) can be explored."

---
*End of Viva Voce & Conceptual Learning Guide.*
