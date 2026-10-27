"""
Run the whole project end-to-end:  python src/run_pipeline.py [--skip-train] [--skip-eda]

  1. data_cleaning       raw CSV            -> data/processed/cleaned_housing.csv
  2. feature_engineering cleaned CSV        -> data/processed/featured_housing.csv + models/city_stats.json
  3. eda                 featured CSV       -> outputs/eda/*.png + eda_insights.md
  4. train_models        featured CSV       -> MLflow runs/registry + models/*.joblib
  5. evaluate            models + test set  -> outputs/evaluation/*.png, feature importance, report
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def step(title, fn):
    t0 = time.time()
    print(f"\n{'=' * 78}\n>>> {title}\n{'=' * 78}", flush=True)
    fn()
    print(f"<<< {title} finished in {time.time() - t0:.0f}s", flush=True)


def main():
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-eda", action="store_true", help="skip EDA chart export")
    ap.add_argument("--skip-train", action="store_true", help="skip training/evaluation (reuse models/)")
    a = ap.parse_args()

    import data_cleaning, feature_engineering
    step("1/5 Data cleaning", data_cleaning.main)
    step("2/5 Feature engineering", feature_engineering.main)
    if not a.skip_eda:
        import eda
        step("3/5 Exploratory data analysis", eda.main)
    if not a.skip_train:
        import train_models, evaluate
        step("4/5 Model training + MLflow", train_models.main)
        step("5/5 Evaluation", evaluate.main)
    print("\nAll done. Launch the app with:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
