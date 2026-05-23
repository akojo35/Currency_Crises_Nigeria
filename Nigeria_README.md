# Currency Crisis Early Warning Model — Nigerian Economy

### Random Forest Ensemble Classifier  |  1990–2023

## 

## Overview

This project builds a **machine learning early warning system (EWS)** to identify and predict currency crisis episodes in Nigeria using annual macroeconomic data from 1990–2023. The model frames the problem as a **binary classification task** — Currency Crisis (1) vs. No Crisis (0) — with Random Forest as the primary algorithm, complemented by SHAP explainability to ensure that predictions are transparent and actionable for policymakers.

The EWS is grounded in the **Exchange Market Pressure (EMP) index** methodology, which combines exchange rate depreciation and reserve loss into a single crisis signal. This provides a theoretically defensible, data-driven crisis label rather than relying on subjective episode dating.



## 

## Research Objective



Can macroeconomic, external sector, and oil sector indicators reliably predict currency crisis episodes in Nigeria, and what are the dominant early warning signals? The project addresses this across four dimensions:



1. **Crisis measurement** — Constructing the EMP index and deriving binary crisis labels
2. **Prediction** — Binary classification of annual currency crisis status
3. **Benchmarking** — Comparing Random Forest against four alternative classifiers
4. **Interpretation** — SHAP values to quantify each feature's contribution to crisis predictions

## 

## &#x20;Crisis Definition

A year is labelled a **currency crisis** if any one of the following conditions holds:

|Condition|Rule|
|-|-|
|**EMP threshold**|EMP index > μ + 1.5σ (where μ and σ are the sample mean and standard deviation)|
|**Severe depreciation**|Annual exchange rate depreciation \|ΔE/E\| > 15%|
|**Reserve collapse**|Annual reserve change ΔR/R < −15%|



The EMP index is constructed as:


EMP\_t  =  w₁ · (ΔE/E)  −  w₂ · (ΔR/R)


where weights w₁ and w₂ are the inverse standard deviations of each component (precision weighting), following Kaminsky, Lizondo \& Reinhart (1998).

## 

## &#x20;Dataset

|Attribute|Detail|
|-|-|
|**Coverage**|1990–2023 (annual)|
|**Observations**|34 years (32 usable after lag engineering)|
|**Crisis episodes**|\~14 years classified as currency crisis|
|**Non-crisis episodes**|\~18 years|
|**Primary sources**|CBN Statistical Bulletin, IMF IFS, World Bank WDI, OPEC|

### Raw Features

|Feature|Description|
|-|-|
|`ExRate\_NGBUSD`|Official NGN/USD exchange rate (end of year)|
|`ExRate\_Depr\_pct`|Annual % depreciation of the Naira|
|`Parallel\_Premium`|Parallel market premium over official rate (%)|
|`FX\_Reserves\_USDbn`|Gross foreign exchange reserves (USD billions)|
|`Reserves\_Change\_pct`|Annual % change in FX reserves|
|`Oil\_Price\_USD`|Average annual Brent crude price (USD/barrel)|
|`OilRev\_pct\_TotalRev`|Oil revenue as % of total government revenue|
|`Inflation\_pct`|Annual consumer price inflation (%)|
|`M2\_Growth\_pct`|Broad money supply growth (%)|
|`Trade\_Bal\_USDbn`|Trade balance (USD billions)|
|`EMP\_Index`|Constructed Exchange Market Pressure index|
|`IMF\_Program`|Binary flag — active IMF programme (1 = yes)|

### Engineered Features

|Feature|Construction|
|-|-|
|`\*\_lag1`, `\*\_lag2`|One- and two-year lags of 8 key variables (leading indicators)|
|`Oil\_Vulnerability`|`(OilRev\_pct\_TotalRev / 100) × (1 / Oil\_Price\_USD)` — combined oil dependence signal|
|`Import\_Cover\_Proxy`|`FX\_Reserves\_USDbn /|
|`Currency\_Overvaluation`|Deviation of official rate from estimated equilibrium path|



## &#x20;Pipeline Architecture




Raw Data (1990–2023)
        │
        ▼
 1. Exploratory Data Analysis
    ├── Exchange rate \& EMP timeline
    ├── Correlation heatmap
    └── KDE plots: crisis vs. no-crisis distributions
        │
        ▼
 2. EMP Index Construction \& Crisis Label Verification
    ├── Precision-weighted EMP formula
    ├── Threshold = μ + 1.5σ
    └── Reconcile EMP rule with composite crisis label
        │
        ▼
 3. Feature Engineering
    ├── Lag features (t-1, t-2) for 8 variables
    ├── Oil Vulnerability score
    ├── Import Cover Proxy
    └── Currency Overvaluation indicator
        │
        ▼
 4. Preprocessing
    ├── StandardScaler (normalisation)
    └── SMOTE (k=3, class imbalance correction)
        │
        ▼
 5. Model Training
    ├── 6.1 Baseline Random Forest (n=300, OOB scoring)
    ├── 6.2 Hyperparameter Tuning (GridSearchCV / RandomizedSearchCV)
    └── 6.3 Final model fit on best params
        │
        ▼
 6. Model Evaluation \& Comparison
    ├── Walk-forward CV: 5 models × 4 metrics
    ├── Confusion matrix, ROC curve, Precision-Recall curve
    └── Optimal decision threshold analysis
        │
        ▼
 7. SHAP Explainability
    ├── Beeswarm summary plot (top 18 features)
    ├── Feature importance bar chart (top 15)
    ├── Waterfall plot — 2023 case study
    └── MDI vs. SHAP importance comparison
        │
        ▼
 8. Early Warning Score Output
    ├── Crisis probability per year (1992–2023)
    ├── Alert levels: 🟢 Low | 🟡 Moderate | 🟠 High | 🔴 Critical
    └── Optimal threshold analysis (Precision / Recall / F1 trade-off)


## 

## &#x20;Models Compared

|Model|Notes|
|-|-|
|**Random Forest (Tuned)**|Primary model; 300–600 trees, balanced class weight, OOB scoring|
|**Gradient Boosting**|200 trees, learning rate 0.05, max depth 3|
|**Logistic Regression**|Balanced class weight, max\_iter=500|
|**Decision Tree**|Balanced class weight, max depth 5|
|**K-Nearest Neighbours**|k=5|



All models are evaluated using **walk-forward (time-series) cross-validation** (`TimeSeriesSplit`, n=5) to preserve temporal ordering and prevent future data leakage.

## 

## &#x20;Hyperparameter Tuning

Tuning is performed in **Section 6.2** using scikit-learn's search utilities with `TimeSeriesSplit(n\_splits=3)` and `scoring='roc\_auc'`.

**Option A — GridSearchCV (default)**





param\_grid = {
    'n\_estimators'      : \[200, 300, 500],
    'max\_depth'         : \[3, 5, 10, None],
    'max\_features'      : \['sqrt', 'log2', 0.5],
    'min\_samples\_leaf'  : \[1, 2, 4],
    'min\_samples\_split' : \[2, 5, 10],
}


**Option B — RandomizedSearchCV (faster)**  
Samples 80 random combinations from continuous and discrete distributions using `scipy.stats` priors. Uncomment Option B in Section 6.2 to switch.





## &#x20;Key Outputs

* **EWS probability table** — Crisis probability and alert level for every year 1992–2023
* **Optimal threshold chart** — Precision / Recall / F1 trade-off to support policy calibration
* **SHAP beeswarm plot** — Direction and magnitude of each feature's influence on crisis predictions
* **SHAP waterfall plot (2023)** — Case study of the FX unification crisis episode
* **MDI vs. SHAP comparison** — Validates that the built-in RF importance and model-agnostic SHAP rankings agree
* **ROC \& Precision-Recall curves** — Full discrimination and calibration assessment
* **Model comparison chart** — AUC-ROC, F1, Recall, Precision across all five classifiers





## &#x20;Getting Started

### Run on Google Colab (recommended)

[!\[Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### 

### Run locally



\# 1. Clone the repository
!git clone https://github.com/<your-username>/nigeria-currency-crisis-ews.git
cd nigeria-currency-crisis-ews

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook Nigeria\_Currency\_Crisis\_EWS\_RandomForest.ipynb
```



## &#x20;Requirements


scikit-learn>=1.3
imbalanced-learn>=0.11
shap>=0.44
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scipy>=1.11


Install all at once:



!pip install scikit-learn imbalanced-learn shap pandas numpy matplotlib seaborn scipy


Repository Structure



nigeria-currency-crisis-ews/
│
├── Nigeria\_Currency\_Crisis\_EWS\_RandomForest.ipynb   # Main notebook (Google Colab)
├── README.md                                         # This file
├── requirements.txt                                  # Python dependencies
│
├── outputs/
│   ├── nigeria\_corr\_heatmap.png                     # Correlation heatmap
│   ├── nigeria\_shap\_summary.png                     # SHAP beeswarm plot
│   ├── nigeria\_shap\_waterfall\_2023.png              # SHAP waterfall — 2023
│   ├── nigeria\_ews\_timeline.png                     # EWS probability chart
│   └── nigeria\_roc\_curve.png                        # ROC curve
│
└── data/
    └── Nigeria\_Currency\_Crisis\_Dataset.xlsx          # Raw dataset (optional upload)




## 

## &#x20;Methodological Notes

* **EMP index** follows the precision-weighting approach of Kaminsky et al. (1998), using inverse standard deviations as component weights to normalise for different variances.
* **Class imbalance** is addressed with SMOTE (`k\_neighbors=3`) and `class\_weight='balanced'` inside Random Forest, ensuring the model does not default to predicting the majority class.
* **Temporal integrity** is preserved throughout using `TimeSeriesSplit` for all cross-validation — no shuffling, no future data leakage.
* **OOB score** (Out-of-Bag) is used as an additional unbiased performance estimate, a unique advantage of bootstrap-based ensemble methods.
* **Dual importance analysis** — MDI (Mean Decrease in Impurity, RF built-in) and SHAP values are computed side-by-side to cross-validate feature rankings and detect any MDI bias toward high-cardinality features.
* **Optimal threshold** is derived from the Precision-Recall curve by maximising F1-score, allowing policymakers to tune the EWS sensitivity to their preference for false alarm vs. missed crisis trade-offs.





## Data Sources

|Source|Variables|
|-|-|
|**Central Bank of Nigeria (CBN) Statistical Bulletin**|Exchange rates, FX reserves, M2, credit|
|**IMF International Financial Statistics (IFS)**|BOP, EMP components, external debt|
|**World Bank World Development Indicators (WDI)**|GDP growth, inflation, trade balance|
|**OPEC Annual Statistical Bulletin**|Oil prices, Nigerian crude production|
|**IMF World Economic Outlook (WEO)**|Fiscal aggregates, current account|





## &#x20;

