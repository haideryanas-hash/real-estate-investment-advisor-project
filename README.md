https://haideryanas-hash-real-estate-investment-appstreamlit-app-noftvp.streamlit.app/

# 🏠 Real Estate Investment Advisor

Classify whether a property is a **Good Investment** and predict its **price after 5 years** — with a Streamlit app, MLflow tracking and a full EDA of 35,359 Indian property listings.

| | |
|---|---|
| Classification (XGBoost) | Accuracy 0.992 · F1 0.988 · ROC-AUC 0.9996 |
| Regression (XGBoost) | RMSE 1.73 lakhs · MAE 1.34 lakhs · R² 0.9999 (fixed-8% baseline RMSE: 23.3 lakhs) |

> **Read first:** the dataset has no price history, so both targets are *rule-derived* (documented in `src/config.py`), and the data itself looks simulated (price is unrelated to size, city, amenities…). High scores mean the models learned the rules, not that they forecast the real market. See `docs/PROJECT_DOCUMENTATION.md` §6.

## Quick start
```bash
pip install -r requirements.txt
python src/run_pipeline.py          # clean → features → EDA → train (MLflow) → evaluate  (~6 min on 1 CPU)
streamlit run app/streamlit_app.py  # the app
mlflow ui --backend-store-uri sqlite:///mlflow.db   # experiment tracking UI
```
The repo already contains the processed data, trained models (`models/`) and outputs, so you can run the app immediately without re-training.

## Structure
```
data/raw/india_housing_prices.csv        original data
data/processed/cleaned_housing.csv       after cleaning
data/processed/featured_housing.csv      + engineered features and both targets
src/config.py                            paths + ALL assumptions (growth model, label rule)
src/data_cleaning.py                     step 1a  cleaning, validation, outlier flags
src/feature_engineering.py / features.py step 1b  features, targets (shared with the app)
src/eda_questions.py / eda.py            step 2   24 analyses (20 brief questions + 4 Step-2) → outputs/eda
src/train_models.py                      step 3-4 3 models x 2 tasks, tuning, MLflow runs + registry
src/evaluate.py                          metrics, importance, intervals, honesty checks
src/run_pipeline.py                      runs everything
app/streamlit_app.py                     the Streamlit application (6 pages)
models/                                  champion models (.joblib), city stats, feature importance
outputs/eda | evaluation | reports       charts, EDA findings, metrics, MLflow run export
docs/PROJECT_DOCUMENTATION.md            methodology, findings, use cases
docs/VIDEO_SCRIPT.md                     script + shot list for the explainer video
```

## EDA question → chart
Q1-Q24 charts live in `outputs/eda/qNN_*.png`; `outputs/eda/eda_insights.md` has every question with its tested finding and table.

## Deploy the app (Streamlit Community Cloud)
Push to GitHub → New app → main file `app/streamlit_app.py` (the lean `app/requirements.txt` is used). Models are pinned to scikit-learn 1.8.0 / xgboost 3.4.1.

## MLflow
Training logs 7 runs to `mlflow.db` (SQLite) and registers `GoodInvestmentClassifier` / `FuturePriceRegressor` (alias `champion`). Model artefacts (`mlruns/`, ~176 MB) are not shipped; re-run `python src/train_models.py` to recreate them. A CSV of all runs is in `outputs/reports/mlflow_runs_export.csv`.

*Not financial advice.*
