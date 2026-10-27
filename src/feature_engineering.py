"""
Step 1b - FEATURE ENGINEERING & TARGET CREATION
===============================================
Input : data/processed/cleaned_housing.csv
Output: data/processed/featured_housing.csv     (cleaned data + engineered features + 2 targets)
        models/city_stats.json                  (city reference medians reused by the Streamlit app)
        outputs/reports/feature_report.json

New features : amenity flags & count, Has_Parking/Has_Security, School_Density_Score, Hospital_Score,
               Transport_Score, Infrastructure_Score, Floor_Ratio, Size_per_BHK,
               PPSF_vs_City_Median, Annual_Growth_Rate.
Targets      : Future_Price_5Y_Lakhs (regression)  = Price * (1 + r)^5
               Good_Investment       (classification) = multi-factor score >= threshold
Both targets are RULE-DERIVED (the dataset has no price history); see config.py for every assumption.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import features as F  # noqa: E402


def main():
    df = pd.read_csv(C.CLEAN_CSV)
    stats = F.compute_city_stats(df)
    C.MODELS_DIR.mkdir(exist_ok=True)
    C.CITY_STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    out = F.engineer(df, stats)
    out.to_csv(C.FEATURED_CSV, index=False)

    crit = F.investment_criteria(out, out["Annual_Growth_Rate"])
    rep = {
        "rows": int(len(out)),
        "columns": int(out.shape[1]),
        "annual_growth_rate_pct": {k: round(float(v * 100), 3) for k, v in
                                   out["Annual_Growth_Rate"].describe().items() if k != "count"},
        "expected_roi_5y_pct_mean": round(float(out["Expected_ROI_5Y_Pct"].mean()), 2),
        "investment_score_distribution": {int(k): int(v) for k, v in
                                          out["Investment_Score"].value_counts().sort_index().items()},
        "good_investment_threshold_score": C.GOOD_SCORE_THRESHOLD,
        "good_investment_share_pct": round(float(out[C.CLF_TARGET].mean() * 100), 2),
        "criterion_hit_rate_pct": {k: round(float(v * 100), 1) for k, v in crit.mean().items()},
    }
    (C.REPORTS_DIR / "feature_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    print(f"Saved -> {C.FEATURED_CSV}  and  {C.CITY_STATS_PATH}")


if __name__ == "__main__":
    main()
