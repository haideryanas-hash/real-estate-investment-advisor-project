"""
Step 3b - MODEL EVALUATION
==========================
Loads the champion models exported by train_models.py and, on the held-out TEST split, produces

  * classification report, confusion matrix, ROC and precision-recall curves, probability histogram
  * regression metrics (RMSE, MAE, R2, MAPE) vs the naive "fixed 8% growth" baseline, residual plots
  * permutation feature importance on the ORIGINAL columns (-> models/feature_importance_*.csv, shown in the app)
  * empirical 90% prediction interval for the 5-year price (-> models/model_metadata.json, used by the app)
  * TWO HONESTY CHECKS that put the headline numbers in context:
      1. Signal test  - can the features predict the CURRENT price?  (if not, the data has no price signal)
      2. Ablation     - how much does the classifier lean on price-based features that define the label?
Outputs: outputs/evaluation/*.png, outputs/reports/evaluation_report.(json|md)
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score,
                             PrecisionRecallDisplay, roc_auc_score, RocCurveDisplay)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import modeling as M  # noqa: E402

warnings.filterwarnings("ignore")
PRIMARY, ACCENT = "#2A6F97", "#E07A5F"


def perm_importance(model, X, y, scoring, n=3000):
    idx = X.sample(min(n, len(X)), random_state=C.RANDOM_STATE).index
    r = permutation_importance(model, X.loc[idx], y.loc[idx], scoring=scoring, n_repeats=5,
                               random_state=C.RANDOM_STATE, n_jobs=1)
    out = pd.DataFrame({"feature": X.columns, "importance": r.importances_mean, "std": r.importances_std})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def importance_plot(imp, title, path, xlabel):
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color=PRIMARY)
    ax.set(title=title, xlabel=xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    C.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads(C.MODEL_META_PATH.read_text(encoding="utf-8"))
    clf, reg = joblib.load(C.CLASSIFIER_PATH), joblib.load(C.REGRESSOR_PATH)
    df, X, y_c, y_r, stats = M.load_modelling_data()
    Xtr, Xte, ytr_c, yte_c, ytr_r, yte_r = M.get_split(X, y_c, y_r)
    rep = {"classifier": meta["classifier"]["model"], "regressor": meta["regressor"]["model"]}

    # ------------------------------------------------------------------ classification
    pred, proba = clf.predict(Xte), clf.predict_proba(Xte)[:, 1]
    cr = classification_report(yte_c, pred, target_names=["Not good", "Good investment"], output_dict=True)
    rep["classification"] = {
        "accuracy": accuracy_score(yte_c, pred), "roc_auc": roc_auc_score(yte_c, proba), "f1": f1_score(yte_c, pred),
        "precision": cr["Good investment"]["precision"], "recall": cr["Good investment"]["recall"],
        "report": cr, "n_test": int(len(yte_c)), "positive_rate_test": float(yte_c.mean()),
    }
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    ConfusionMatrixDisplay.from_predictions(yte_c, pred, display_labels=["Not good", "Good"], ax=axes[0],
                                            cmap="Blues", colorbar=False)
    axes[0].set_title("Confusion matrix")
    RocCurveDisplay.from_predictions(yte_c, proba, ax=axes[1], color=PRIMARY, name=rep["classifier"])
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[1].set_title("ROC curve")
    PrecisionRecallDisplay.from_predictions(yte_c, proba, ax=axes[2], color=ACCENT)
    axes[2].set_title("Precision-recall curve")
    fp, mp = calibration_curve(yte_c, proba, n_bins=10)
    axes[3].plot(mp, fp, "o-", color=PRIMARY, label="Model")
    axes[3].plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect")
    axes[3].set(xlabel="Mean predicted probability", ylabel="Observed share good", title="Calibration")
    axes[3].legend()
    fig.tight_layout()
    fig.savefig(C.EVAL_DIR / "classification_diagnostics.png", dpi=130)
    plt.close(fig)

    imp_c = perm_importance(clf, Xte, yte_c, "roc_auc")
    imp_c.to_csv(C.IMPORTANCE_CLF_PATH, index=False)
    importance_plot(imp_c, "Classifier - permutation importance", C.EVAL_DIR / "importance_classifier.png",
                    "Drop in ROC-AUC when the feature is shuffled")

    # ------------------------------------------------------------------ regression
    rp = reg.predict(Xte)
    rm = M.regression_metrics(yte_r.values, rp)
    base = Xte["Price_in_Lakhs"].values * (1 + C.BASE_GROWTH) ** C.HORIZON_YEARS
    bm = M.regression_metrics(yte_r.values, base)
    rel = (yte_r.values / rp) - 1                     # relative error -> used for the app's 90% interval
    p05, p95 = np.percentile(rel, [5, 95])
    rep["regression"] = {**rm, "baseline_fixed_8pct": bm, "rel_error_p05": float(p05), "rel_error_p95": float(p95),
                         "rmse_improvement_vs_baseline_pct": float((1 - rm["rmse"] / bm["rmse"]) * 100),
                         "target_mean": float(yte_r.mean()), "target_std": float(yte_r.std())}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].scatter(yte_r, rp, s=3, alpha=0.3, color=PRIMARY)
    lim = [yte_r.min(), yte_r.max()]
    axes[0].plot(lim, lim, "r--", lw=1)
    axes[0].set(xlabel="Actual 5-yr price (lakhs)", ylabel="Predicted", title=f"Predicted vs actual (R\u00b2={rm['r2']:.4f})")
    axes[1].hist(rp - yte_r, bins=60, color=PRIMARY)
    axes[1].set(xlabel="Error (lakhs)", title=f"Residuals (MAE {rm['mae']:.2f} L)")
    axes[2].bar(["Fixed 8% baseline", rep["regressor"]], [bm["rmse"], rm["rmse"]], color=["#9aa5b1", PRIMARY])
    axes[2].set(ylabel="Test RMSE (lakhs)", title="ML vs naive baseline")
    fig.tight_layout()
    fig.savefig(C.EVAL_DIR / "regression_diagnostics.png", dpi=130)
    plt.close(fig)

    imp_r = perm_importance(reg, Xte, yte_r, "neg_root_mean_squared_error")
    imp_r.to_csv(C.IMPORTANCE_REG_PATH, index=False)
    importance_plot(imp_r, "Regressor - permutation importance", C.EVAL_DIR / "importance_regressor.png",
                    "Increase in RMSE (lakhs) when the feature is shuffled")

    # ------------------------------------------------------------------ honesty check 1: signal test
    leak = ["Price_in_Lakhs", "Log_Price", "Price_per_SqFt_INR", "PPSF_vs_City_Median"]
    num = [c for c in C.NUMERIC_FEATURES if c not in leak]
    probe = TransformedTargetRegressor(
        regressor=Pipeline([("prep", M.make_preprocessor(num, C.CATEGORICAL_FEATURES)),
                            ("model", XGBRegressor(n_estimators=250, max_depth=4, learning_rate=0.05, tree_method="hist",
                                                   n_jobs=1, random_state=C.RANDOM_STATE))]),
        func=np.log, inverse_func=np.exp)
    cur_tr, cur_te = df.loc[Xtr.index, "Price_in_Lakhs"], df.loc[Xte.index, "Price_in_Lakhs"]
    probe.fit(Xtr.drop(columns=leak), cur_tr)
    pm = M.regression_metrics(cur_te.values, probe.predict(Xte.drop(columns=leak)))
    rep["signal_test_current_price"] = {
        "description": "XGBoost predicting TODAY's price from every other feature (city, size, BHK, age, amenities ...)",
        "test_r2": pm["r2"], "test_rmse": pm["rmse"], "price_std": float(cur_te.std()),
        "interpretation": "R2 ~ 0 means the features carry no information about price in this dataset."}

    # ------------------------------------------------------------------ honesty check 2: ablation
    num_a = [c for c in C.NUMERIC_FEATURES if c not in leak]
    abl = Pipeline([("prep", M.make_preprocessor(num_a, C.CATEGORICAL_FEATURES)),
                    ("model", XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.1, tree_method="hist",
                                            n_jobs=1, random_state=C.RANDOM_STATE, eval_metric="logloss"))])
    abl.fit(Xtr.drop(columns=leak), ytr_c)
    ap = abl.predict_proba(Xte.drop(columns=leak))[:, 1]
    rep["ablation_without_price_features"] = {
        "description": "Same XGBoost classifier WITHOUT Price_in_Lakhs, Log_Price, Price_per_SqFt_INR, PPSF_vs_City_Median",
        "accuracy": accuracy_score(yte_c, ap > 0.5), "roc_auc": roc_auc_score(yte_c, ap), "f1": f1_score(yte_c, ap > 0.5),
        "full_model_accuracy": rep["classification"]["accuracy"], "full_model_roc_auc": rep["classification"]["roc_auc"]}

    # ------------------------------------------------------------------ persist for the app + docs
    meta["classifier"].update(test_accuracy=rep["classification"]["accuracy"], test_roc_auc=rep["classification"]["roc_auc"],
                              test_f1=rep["classification"]["f1"], test_precision=rep["classification"]["precision"],
                              test_recall=rep["classification"]["recall"])
    meta["regressor"].update(test_rmse=rm["rmse"], test_mae=rm["mae"], test_r2=rm["r2"], test_mape_pct=rm["mape_pct"],
                             rel_error_p05=float(p05), rel_error_p95=float(p95))
    meta["signal_test_r2_current_price"] = pm["r2"]
    meta["ablation_roc_auc"] = rep["ablation_without_price_features"]["roc_auc"]
    C.MODEL_META_PATH.write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    (C.REPORTS_DIR / "evaluation_report.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    c, r, s, a = rep["classification"], rep["regression"], rep["signal_test_current_price"], rep["ablation_without_price_features"]
    md = f"""# Evaluation report (held-out test set, n={c['n_test']:,})

## Classification - {rep['classifier']}
| Metric | Value |
|---|---|
| Accuracy | {c['accuracy']:.4f} |
| Precision (good) | {c['precision']:.4f} |
| Recall (good) | {c['recall']:.4f} |
| F1 (good) | {c['f1']:.4f} |
| ROC-AUC | {c['roc_auc']:.4f} |

## Regression - {rep['regressor']}
| Metric | Model | Fixed-8% baseline |
|---|---|---|
| RMSE (lakhs) | {rm['rmse']:.3f} | {bm['rmse']:.3f} |
| MAE (lakhs) | {rm['mae']:.3f} | {bm['mae']:.3f} |
| R2 | {rm['r2']:.5f} | {bm['r2']:.5f} |
| MAPE % | {rm['mape_pct']:.3f} | {bm['mape_pct']:.3f} |

RMSE improvement over the baseline: {r['rmse_improvement_vs_baseline_pct']:.1f}%.
90% prediction interval (relative error): {p05*100:+.2f}% to {p95*100:+.2f}%.

## Honesty checks
* **Signal test** - predicting TODAY's price from all other features: R2 = {s['test_r2']:.4f}, RMSE = {s['test_rmse']:.1f} L (price std = {s['price_std']:.1f} L). The features contain no usable information about price.
* **Ablation** - classifier without the four price-based features: ROC-AUC {a['roc_auc']:.4f} (full model {a['full_model_roc_auc']:.4f}), accuracy {a['accuracy']:.4f} (full {a['full_model_accuracy']:.4f}).
"""
    (C.REPORTS_DIR / "evaluation_report.md").write_text(md, encoding="utf-8")
    print(md)

    # ------------------------------------------------------------------ export the MLflow run history
    try:
        import mlflow
        mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
        runs = mlflow.search_runs(experiment_names=[C.MLFLOW_EXPERIMENT])
        keep = [c for c in runs.columns if c.startswith(("params.", "metrics.", "tags.task", "tags.model"))
                or c in ("run_id", "tags.mlflow.runName", "start_time")]
        runs[keep].to_csv(C.REPORTS_DIR / "mlflow_runs_export.csv", index=False)
        print(f"Exported {len(runs)} MLflow runs -> outputs/reports/mlflow_runs_export.csv")
    except Exception as e:  # MLflow is optional for evaluation
        print("MLflow export skipped:", e)


if __name__ == "__main__":
    main()
