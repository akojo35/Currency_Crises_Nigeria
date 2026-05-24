# Modelling Fiscal Stress of the Ghanaian Economy

### 

### XGBoost Binary Classification Pipeline | 1990–2023

## 

## Overview



This project develops a **machine learning early warning system (EWS)** to identify and predict episodes of fiscal stress in Ghana using annual macroeconomic data spanning 1990–2023. The pipeline frames the problem as a **binary classification task** — Fiscal Stress (1) vs. No Stress (0) — and leverages XGBoost as the primary algorithm, supported by SHAP explainability to ensure interpretability alongside predictive power.

The model is intended to serve as a quantitative diagnostic tool for policymakers, fiscal analysts, and researchers monitoring Ghana's public finance sustainability.

## 

## &#x20;Research Objective

Can macroeconomic and fiscal indicators be used to reliably classify and predict fiscal stress episodes in Ghana, and which drivers are most influential? The project answers this in three parts:



1. **Prediction** — Binary classification of annual fiscal stress status
2. **Comparison** — Benchmark XGBoost against Logistic Regression, Random Forest, and SVM
3. **Interpretation** — SHAP values to explain *why* the model flags stress in specific years



## 

|Attribute|Detail|
|-|-|
|**Coverage**|1990–2023 (annual)|
|**Observations**|34 years (32 usable after lag engineering)|
|**Stress episodes**|\~22 years classified as fiscal stress|
|**Non-stress episodes**|\~10 years|
|**Source**|World Bank, IMF Article IV Reports, Bank of Ghana|

### Raw Features (13)

|Feature|Description|
|-|-|
|`GDP\_Growth`|Annual real GDP growth rate (%)|
|`Inflation`|Annual consumer price inflation (%)|
|`Debt\_to\_GDP`|Total public debt as % of GDP|
|`Primary\_Balance`|Primary fiscal balance as % of GDP|
|`Revenue\_GDP`|Government revenue as % of GDP|
|`Expenditure\_GDP`|Government expenditure as % of GDP|
|`External\_Debt\_GDP`|External debt as % of GDP|
|`Debt\_Service\_Ratio`|Debt service payments as % of revenue|
|`CA\_Balance`|Current account balance as % of GDP|
|`Reserves\_Months`|Gross international reserves (months of import cover)|
|`FX\_Depreciation`|Annual depreciation of the Ghana Cedi (%)|
|`Interest\_Revenue`|Interest payments as % of revenue|
|`IMF\_Program`|Binary flag — active IMF programme (1 = yes)|

### Engineered Features

|Feature|Construction|
|-|-|
|`\*\_lag1`, `\*\_lag2`|One- and two-year lags of 6 key fiscal variables (capturing momentum)|
|`Fiscal\_Space`|`Revenue\_GDP` − `Interest\_Revenue`|
|`Debt\_Acceleration`|Year-on-year change in `Debt\_to\_GDP`|



## &#x20;Pipeline Architecture


Raw Data (1990–2023)
        │
        ▼
 1. Exploratory Data Analysis
    ├── Fiscal stress timeline
    ├── Correlation heatmap
    └── Distribution by stress class
        │
        ▼
 2. Feature Engineering
    ├── Lag features (t-1, t-2)
    ├── Fiscal Space indicator
    └── Debt Acceleration
        │
        ▼
 3. Preprocessing
    ├── StandardScaler (normalisation)
    └── SMOTE (k=3, class imbalance correction)
        │
        ▼
 4. Model Training
    ├── 5.1 Baseline XGBoost
    ├── 5.2 Hyperparameter Tuning (GridSearchCV / RandomizedSearchCV)
    └── 5.3 Final model fit on best params
        │
        ▼
 5. Evaluation \& Comparison
    ├── Walk-forward cross-validation (TimeSeriesSplit, n=3)
    ├── Confusion matrix \& ROC curve
    └── AUC-ROC, F1, Recall, Precision vs. 3 benchmarks
        │
        ▼
 6. SHAP Explainability
    ├── Beeswarm summary plot
    ├── Feature importance bar chart
    └── Force plot (Year 2022 case study)
        │
        ▼
 7. Early Warning Score Output
    ├── Stress probability per year (1992–2023)
    └── Alert levels: 🟢 Low | 🟡 Moderate | 🟠 High | 🔴 Critical




|Model|Notes|
|-|-|
|**XGBoost (Tuned)**|Primary model; tuned via GridSearchCV with TimeSeriesSplit|
|**Random Forest**|200 trees, balanced class weight|
|**Logistic Regression**|Balanced class weight, max\_iter=500|
|**SVM (RBF kernel)**|Calibrated for probability output|

All models are evaluated using **walk-forward (time-series) cross-validation** to respect temporal ordering and avoid data leakage.



Tuning is performed in **Section 5.2** using scikit-learn's search utilities, replacing an earlier Optuna implementation that failed to converge on this small dataset (n=32).

**Option A — GridSearchCV (default)**





param\_grid = {
    'n\_estimators'    : \[100, 200, 300],
    'max\_depth'       : \[2, 3, 4],
    'learning\_rate'   : \[0.01, 0.05, 0.1],
    'subsample'       : \[0.7, 1.0],
    'colsample\_bytree': \[0.7, 1.0],
    'reg\_lambda'      : \[1.0, 3.0],
    'min\_child\_weight': \[1, 3],
}


**Option B — RandomizedSearchCV (faster)**  
Samples 80 random combinations from continuous distributions using `scipy.stats` priors. Uncomment Option B in Section 5.2 and comment out Option A to switch.



Both use `TimeSeriesSplit(n\_splits=3)` and `scoring='roc\_auc'`.

* **EWS probability table** — Fiscal stress probability and alert level for every year 1992–2023
* **SHAP beeswarm plot** — Direction and magnitude of each feature's influence
* **ROC curve** — Model discrimination performance
* **Model comparison chart** — AUC-ROC, F1, Recall, Precision across all four models
* **SHAP force plot (2022)** — Case study of the most recent high-stress episode

## 

## &#x20;Getting Started

### Run on Google Colab (recommended)

Click the badge or open the notebook directly in Colab:

[!\[Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### 

### Run locally




# 1. Clone the repository
git clone https://github.com/<your-username>/ghana-fiscal-stress.git
cd ghana-fiscal-stress

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook Ghana\_Fiscal\_Stress\_XGBoost.ipynb


## 

## Requirements

xgboost>=1.7
scikit-learn>=1.3
imbalanced-learn>=0.11
shap>=0.44
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scipy>=1.11




Install all at once:




!pip install xgboost scikit-learn imbalanced-learn shap pandas numpy matplotlib seaborn scipy


## &#x20;Repository Structure


ghana-fiscal-stress/
│
├── Ghana\_Fiscal\_Stress\_XGBoost.ipynb   # Main notebook (Google Colab)
├── README.md                            # This file
├── requirements.txt                     # Python dependencies
│
├── outputs/
│   ├── shap\_summary.png                 # SHAP beeswarm plot
│   ├── shap\_bar.png                     # Feature importance chart
│   ├── roc\_curve.png                    # ROC curve
│   └── ews\_timeline.png                 # Early warning probability chart
│
└── data/
    └── Ghana\_Fiscal\_Stress\_Dataset.xlsx # Raw dataset (optional upload)


📐 Methodological Notes

* **Class imbalance** is addressed with SMOTE (`k\_neighbors=3`) applied before scaling, and `scale\_pos\_weight` inside XGBoost during tuning.
* **Temporal integrity** is preserved throughout by using `TimeSeriesSplit` (no shuffling) for all cross-validation.
* **Explainability** uses SHAP `TreeExplainer`, which provides exact Shapley values for tree-based models — not approximations.
* **Convergence note**: The original Optuna Bayesian optimiser was replaced with GridSearchCV due to fold starvation on this small dataset (n=32 after lag engineering). With fewer than 10 training samples in early folds, the TPE surrogate model received contradictory signals and failed to converge.



