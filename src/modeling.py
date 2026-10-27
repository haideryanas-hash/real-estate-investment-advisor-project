"""Shared modelling helpers (data split, preprocessing, metrics) used by train_models.py and evaluate.py."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C
import features as F


def load_modelling_data():
    """Returns (df, X, y_clf, y_reg, city_stats). X holds exactly C.MODEL_FEATURES."""
    df = pd.read_csv(C.FEATURED_CSV)
    stats = F.load_city_stats()
    X = F.model_frame(df, stats)
    return df, X, df[C.CLF_TARGET].astype(int), df[C.REG_TARGET].astype(float), stats


def get_split(X, y_clf, y_reg):
    """One shared stratified 80/20 split so classification and regression are evaluated on the same rows."""
    idx_tr, idx_te = train_test_split(
        X.index, test_size=C.TEST_SIZE, stratify=y_clf, random_state=C.RANDOM_STATE)
    return (X.loc[idx_tr], X.loc[idx_te], y_clf.loc[idx_tr], y_clf.loc[idx_te],
            y_reg.loc[idx_tr], y_reg.loc[idx_te])


def make_preprocessor(numeric=None, categorical=None):
    """Scale numeric features, one-hot encode categoricals (unknown categories ignored at predict time)."""
    numeric = C.NUMERIC_FEATURES if numeric is None else numeric
    categorical = C.CATEGORICAL_FEATURES if categorical is None else categorical
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])


def regression_metrics(y_true, y_pred):
    err = y_pred - y_true
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - np.sum(err ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)),
        "mape_pct": float(np.mean(np.abs(err) / y_true) * 100),
    }
