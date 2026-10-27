"""
Step 3 + 4 - MODEL DEVELOPMENT & MLFLOW TRACKING
================================================
Trains and compares, for each task, three model families (as suggested in the brief):

  Classification  "Is this a Good Investment?"   Logistic Regression | Random Forest | XGBoost
  Regression      "Price after 5 years (lakhs)"  Linear Regression   | Random Forest | XGBoost

* One shared stratified 80/20 train/test split.  Test rows are never used for tuning or selection.
* Tuning: RandomizedSearchCV (3-fold) for the tree models; the model with the best CROSS-VALIDATION score is
  selected (not the best test score).
* Every model is one MLflow run (params, CV + test metrics, plots, model artifact + signature).
* The selected models are registered in the MLflow Model Registry (alias "champion") and exported to
  models/*.joblib - the Streamlit app loads those files, so it needs no MLflow server.
* Regression also logs a naive baseline (fixed 8% growth) so the value added by ML is visible.

View the runs:   mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, ConfusionMatrixDisplay, f1_score, precision_score, recall_score,
                             roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import features as F  # noqa: E402
import modeling as M  # noqa: E402

warnings.filterwarnings("ignore")
RS = C.RANDOM_STATE


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ----------------------------------------------------------------------------- candidate models
def clf_candidates():
    pre = M.make_preprocessor
    return {
        "Logistic_Regression": (
            Pipeline([("prep", pre()), ("model", LogisticRegression(max_iter=3000, random_state=RS))]),
            {"model__C": [0.1, 1.0, 10.0]}),
        "Random_Forest": (
            Pipeline([("prep", pre()), ("model", RandomForestClassifier(n_jobs=-1, random_state=RS))]),
            {"model__n_estimators": [100, 150], "model__max_depth": [10, 14],
             "model__min_samples_leaf": [3, 5], "model__max_features": ["sqrt"]}),
        "XGBoost": (
            Pipeline([("prep", pre()), ("model", XGBClassifier(
                tree_method="hist", n_jobs=1, random_state=RS, eval_metric="logloss"))]),
            {"model__n_estimators": [200, 350], "model__max_depth": [3, 5, 7],
             "model__learning_rate": [0.05, 0.1], "model__subsample": [0.8, 1.0],
             "model__colsample_bytree": [0.8, 1.0]}),
    }


def reg_candidates():
    pre = M.make_preprocessor

    def ttr(est):   # model log(price): errors are multiplicative, so this stabilises variance
        return TransformedTargetRegressor(
            regressor=Pipeline([("prep", pre()), ("model", est)]), func=np.log, inverse_func=np.exp)

    return {
        "Linear_Regression": (ttr(LinearRegression()), {}),
        "Random_Forest": (
            ttr(RandomForestRegressor(n_jobs=-1, random_state=RS)),
            {"regressor__model__n_estimators": [100, 150], "regressor__model__max_depth": [12, 16],
             "regressor__model__min_samples_leaf": [3, 5], "regressor__model__max_features": [0.5]}),
        "XGBoost": (
            ttr(XGBRegressor(tree_method="hist", n_jobs=1, random_state=RS)),
            {"regressor__model__n_estimators": [200, 350], "regressor__model__max_depth": [3, 5, 7],
             "regressor__model__learning_rate": [0.05, 0.1], "regressor__model__subsample": [0.8, 1.0],
             "regressor__model__colsample_bytree": [0.8, 1.0]}),
    }


def tune(name, est, space, scoring, cv, X, y, n_iter):
    t0 = time.time()
    if space:
        n_comb = int(np.prod([len(v) for v in space.values()]))
        search = RandomizedSearchCV(est, space, n_iter=min(n_iter, n_comb), scoring=scoring, cv=cv,
                                    random_state=RS, n_jobs=1, refit=True)
        search.fit(X, y)
        best, cv_score, params = search.best_estimator_, float(search.best_score_), search.best_params_
    else:
        best = est.fit(X, y)
        cv_score = float(cross_val_score(est, X, y, scoring=scoring, cv=cv).mean())
        params = {}
    return best, cv_score, params, time.time() - t0


def flat(params):
    return {k.split("__")[-1]: v for k, v in params.items()}


# ----------------------------------------------------------------------------- main
def main():
    C.MODELS_DIR.mkdir(exist_ok=True)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
    client = MlflowClient()

    df, X, y_clf, y_reg, stats = M.load_modelling_data()
    Xtr, Xte, ytr_c, yte_c, ytr_r, yte_r = M.get_split(X, y_clf, y_reg)
    log(f"rows: train={len(Xtr):,} test={len(Xte):,} | features={X.shape[1]} | good-investment rate={y_clf.mean():.1%}")

    common_params = {
        "n_train": len(Xtr), "n_test": len(Xte), "n_features": X.shape[1], "test_size": C.TEST_SIZE,
        "random_state": RS, "growth_base": C.BASE_GROWTH, "growth_cap": C.GROWTH_CAP,
        "good_score_threshold": C.GOOD_SCORE_THRESHOLD, "growth_good_threshold": C.GROWTH_GOOD_THRESHOLD,
    }
    comparison = {"classification": [], "regression": []}

    # ======================================================================= CLASSIFICATION
    log("=== CLASSIFICATION: Good Investment ===")
    cv = StratifiedKFold(3, shuffle=True, random_state=RS)
    best_clf = None
    for name, (est, space) in clf_candidates().items():
        model, cv_auc, params, secs = tune(name, est, space, "roc_auc", cv, Xtr, ytr_c, n_iter=4)
        pred, proba = model.predict(Xte), model.predict_proba(Xte)[:, 1]
        m = {"test_accuracy": accuracy_score(yte_c, pred), "test_precision": precision_score(yte_c, pred),
             "test_recall": recall_score(yte_c, pred), "test_f1": f1_score(yte_c, pred),
             "test_roc_auc": roc_auc_score(yte_c, proba), "cv_roc_auc": cv_auc,
             "train_accuracy": accuracy_score(ytr_c, model.predict(Xtr)), "train_seconds": secs}
        with mlflow.start_run(run_name=f"clf_{name}", tags={"task": "classification", "model": name}) as run:
            mlflow.log_params({**common_params, "model_family": name, **flat(params)})
            mlflow.log_metrics(m)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            ConfusionMatrixDisplay.from_predictions(yte_c, pred, ax=axes[0], cmap="Blues", colorbar=False)
            RocCurveDisplay.from_predictions(yte_c, proba, ax=axes[1], name=name)
            axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
            fig.suptitle(f"{name} - test set")
            mlflow.log_figure(fig, "test_diagnostics.png")
            plt.close(fig)
            sig = infer_signature(Xtr.head(5), model.predict(Xtr.head(5)))
            info = mlflow.sklearn.log_model(model, name="model", signature=sig, input_example=Xtr.head(3),
                                          serialization_format="cloudpickle")
            rid = run.info.run_id
        row = {"model": name, "cv_roc_auc": cv_auc, **{k: v for k, v in m.items() if k.startswith("test_")},
               "train_accuracy": m["train_accuracy"], "train_seconds": round(secs, 1), "run_id": rid,
               "model_uri": info.model_uri, "best_params": json.dumps(flat(params), default=str)}
        comparison["classification"].append(row)
        log(f"  {name:<20} cv_auc={cv_auc:.4f} | test acc={m['test_accuracy']:.4f} f1={m['test_f1']:.4f} "
            f"auc={m['test_roc_auc']:.4f} | {secs:.0f}s")
        if best_clf is None or cv_auc > best_clf["cv_roc_auc"]:
            best_clf = {**row, "model": name, "obj": model}

    # ======================================================================= REGRESSION
    log("=== REGRESSION: Future price after 5 years ===")
    cvr = KFold(3, shuffle=True, random_state=RS)
    # naive baseline: fixed 8% p.a.
    base_pred = Xte["Price_in_Lakhs"].values * (1 + C.BASE_GROWTH) ** C.HORIZON_YEARS
    bm = M.regression_metrics(yte_r.values, base_pred)
    with mlflow.start_run(run_name="reg_Baseline_Fixed_8pct", tags={"task": "regression", "model": "baseline"}):
        mlflow.log_params({**common_params, "model_family": "Baseline_Fixed_8pct"})
        mlflow.log_metrics({f"test_{k}": v for k, v in bm.items()})
    comparison["regression"].append({"model": "Baseline_Fixed_8pct", "cv_rmse": np.nan,
                                     **{f"test_{k}": v for k, v in bm.items()}})
    log(f"  {'Baseline (8% fixed)':<20} test rmse={bm['rmse']:.3f} mae={bm['mae']:.3f} r2={bm['r2']:.4f}")

    best_reg = None
    for name, (est, space) in reg_candidates().items():
        model, cv_score, params, secs = tune(name, est, space, "neg_root_mean_squared_error", cvr, Xtr, ytr_r, n_iter=4)
        cv_rmse = -cv_score
        pred = model.predict(Xte)
        rm = M.regression_metrics(yte_r.values, pred)
        tr = M.regression_metrics(ytr_r.values, model.predict(Xtr))
        m = {**{f"test_{k}": v for k, v in rm.items()}, "cv_rmse": cv_rmse, "train_rmse": tr["rmse"],
             "train_r2": tr["r2"], "train_seconds": secs}
        with mlflow.start_run(run_name=f"reg_{name}", tags={"task": "regression", "model": name}) as run:
            mlflow.log_params({**common_params, "model_family": name, **flat(params)})
            mlflow.log_metrics(m)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].scatter(yte_r, pred, s=3, alpha=0.3)
            lim = [yte_r.min(), yte_r.max()]
            axes[0].plot(lim, lim, "r--", lw=1)
            axes[0].set(xlabel="Actual future price (lakhs)", ylabel="Predicted", title="Predicted vs actual")
            axes[1].hist(pred - yte_r, bins=60, color="#2A6F97")
            axes[1].set(xlabel="Error (lakhs)", title="Residuals")
            fig.suptitle(f"{name} - test set")
            mlflow.log_figure(fig, "test_diagnostics.png")
            plt.close(fig)
            sig = infer_signature(Xtr.head(5), model.predict(Xtr.head(5)))
            info = mlflow.sklearn.log_model(model, name="model", signature=sig, input_example=Xtr.head(3),
                                          serialization_format="cloudpickle")
            rid = run.info.run_id
        row = {"model": name, "cv_rmse": cv_rmse, **{k: v for k, v in m.items() if k.startswith("test_")},
               "train_rmse": tr["rmse"], "train_r2": tr["r2"], "train_seconds": round(secs, 1), "run_id": rid,
               "model_uri": info.model_uri, "best_params": json.dumps(flat(params), default=str)}
        comparison["regression"].append(row)
        log(f"  {name:<20} cv_rmse={cv_rmse:.3f} | test rmse={rm['rmse']:.3f} mae={rm['mae']:.3f} r2={rm['r2']:.4f} | {secs:.0f}s")
        if best_reg is None or cv_rmse < best_reg["cv_rmse"]:
            best_reg = {**row, "model": name, "obj": model}

    # ======================================================================= register + export
    log("=== registering champions ===")
    for best, reg_name, path in [(best_clf, C.REGISTERED_CLF_NAME, C.CLASSIFIER_PATH),
                                 (best_reg, C.REGISTERED_REG_NAME, C.REGRESSOR_PATH)]:
        mv = mlflow.register_model(best["model_uri"], reg_name)
        client.set_registered_model_alias(reg_name, "champion", mv.version)
        joblib.dump(best["obj"], path, compress=3)
        log(f"  {reg_name} v{mv.version} <- {best['model']} | exported {path.name} ({path.stat().st_size/1e6:.1f} MB)")

    for task, rows in comparison.items():
        pd.DataFrame(rows).drop(columns=["model_uri"], errors="ignore").to_csv(
            C.REPORTS_DIR / f"model_comparison_{task}.csv", index=False)

    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": len(Xtr), "n_test": len(Xte),
        "classifier": {k: best_clf[k] for k in ["model", "cv_roc_auc", "test_accuracy", "test_precision",
                                                "test_recall", "test_f1", "test_roc_auc", "best_params", "run_id"]},
        "regressor": {k: best_reg[k] for k in ["model", "cv_rmse", "test_rmse", "test_mae", "test_r2",
                                               "test_mape_pct", "best_params", "run_id"]},
        "baseline_fixed_8pct": {f"test_{k}": v for k, v in bm.items()},
        "features": C.MODEL_FEATURES,
    }
    C.MODEL_META_PATH.write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    log("done.")


if __name__ == "__main__":
    main()
