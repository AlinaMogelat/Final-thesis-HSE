# Final thesis for MDS HSE: Application of ensemble models for forecasting the frequency of insurance claims

Practical component of the final thesis. Application and comparison of GLM and ensemble machine learning methods (bagging, boosting, stacking) for forecasting frequency of insurance claims using auto insurance data.

## Data

Public dataset **freMTPL2freq** was used - a portfolio of auto liability insurance policies (France). It contains 678k policies with information on the driver, vehicle, region, and number of reported claims during the insurance period.

The `data.zip` file contains the source CSV. Unzip it before running.

## Project Structure

```
├── Final_thesis.ipynb     — main notebook with the full pipline
├── data.zip               — raw data (freMTPL2freq.csv)
└── README.md
```
## Notebook Contents

1. **Exploratory Data Analysis (EDA)** - descriptive statistics, distributions, checking for zero-inflation and overdispersion.
2. **Preprocessing** — filtering short exposure periods, clipping outliers, log transformation of density, one-hot encoding of categorical variables.
3. **Base models (GLM):** Poisson, Negative Binomial, ZIP, ZINB.
4. **Base models (Ensembles):** Random Forest and XGBoost (with hyperparameter tuning via Optuna).
5. **Final model - stacking:** out-of-fold predictions using ZIP + RF + XGBoost, meta-model based on Ridge regression.
6. **Interpretation (SHAP)** — global feature importance, summary plot, dependence plots for key variables.
7. **Final comparison** — table and visualization of all models by Poisson deviance, MAE, and Gini coefficient.

## Stack

Python 3.10+, pandas, numpy, scikit-learn, statsmodels, xgboost, optuna, shap, matplotlib, seaborn.

