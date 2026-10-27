"""
Step 1a - DATA CLEANING
=======================
Input : data/raw/india_housing_prices.csv
Output: data/processed/cleaned_housing.csv
        outputs/reports/cleaning_report.json  (+ .md)

What this script does (and why)
-------------------------------
1.  Standardise column names / strip whitespace in text columns.
2.  Missing values  -> median (numeric) / mode (categorical) imputation. (This file has none, but the
    logic is kept so the pipeline is safe on a refreshed extract.)
3.  Duplicates      -> drop exact duplicate rows and duplicate IDs.
4.  Validity checks -> impossible values (size<=0, price<=0, future Year_Built ...) are dropped;
    Age_of_Property is re-derived from Year_Built and compared with the supplied column.
5.  Floor logic     -> Floor_No > Total_Floors is impossible (a flat cannot be above the top floor).
    ~47% of rows have it, so dropping would destroy the dataset; Total_Floors is repaired to
    max(Floor_No, Total_Floors) and the row is flagged (Floor_Data_Repaired = 1).
6.  Price per sq ft -> the supplied column is in *lakhs* rounded to 2 decimals (0.10 = Rs 10,000) and
    is 0.00 for 331 rows. It is recomputed exactly in rupees (Price_per_SqFt_INR).
7.  Amenities       -> 325 spellings of the same 5 amenities (only the ORDER differs). Canonicalised
    to a sorted string; binary flags are created in the feature-engineering step.
8.  Outliers        -> IQR rule (1.5 x IQR) flags on price, size and price-per-sq-ft. Rows are KEPT
    (flagged), because extremes here are valid combinations (e.g. a 500 sq ft flat at 500 lakh).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def iqr_bounds(s: pd.Series, k: float = 1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def clean(df: pd.DataFrame):
    rep = {"rows_raw": int(len(df)), "columns_raw": int(df.shape[1])}

    # 1 ---- standardise text -------------------------------------------------------------
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in text_cols:
        df[c] = df[c].astype(str).str.strip()
    # unify the spelling of known categorical levels (guards against 'high ' / 'HIGH' in new extracts)
    for c in ["Public_Transport_Accessibility"]:
        df[c] = df[c].str.title()
    for c in ["Parking_Space", "Security"]:
        df[c] = df[c].str.title()

    # 2 ---- missing values ----------------------------------------------------------------
    miss_before = df.isna().sum()
    rep["missing_values_before"] = {k: int(v) for k, v in miss_before.items() if v > 0} or "none"
    num_cols = df.select_dtypes(include=[np.number]).columns.drop("ID", errors="ignore")
    cat_cols = [c for c in df.columns if c not in num_cols and c != "ID"]
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    rep["missing_values_after"] = int(df.isna().sum().sum())

    # 3 ---- duplicates ---------------------------------------------------------------------
    rep["duplicate_rows_excl_id"] = int(df.drop(columns="ID").duplicated().sum())
    rep["duplicate_ids"] = int(df["ID"].duplicated().sum())
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "ID"]).drop_duplicates(subset="ID")

    # 4 ---- validity ----------------------------------------------------------------------
    bad = (
        (df["Size_in_SqFt"] <= 0) | (df["Price_in_Lakhs"] <= 0) | (df["BHK"] < 1) |
        (df["Year_Built"] > C.REFERENCE_YEAR) | (df["Floor_No"] < 0) | (df["Total_Floors"] < 1)
    )
    rep["invalid_rows_dropped"] = int(bad.sum())
    df = df[~bad].copy()

    age_calc = C.REFERENCE_YEAR - df["Year_Built"]
    rep["age_mismatch_rows"] = int((age_calc != df["Age_of_Property"]).sum())
    df["Age_of_Property"] = age_calc            # single source of truth (identical here)

    # 5 ---- floor logic ---------------------------------------------------------------------
    inconsistent = df["Floor_No"] > df["Total_Floors"]
    rep["floor_gt_total_floors_rows"] = int(inconsistent.sum())
    rep["floor_gt_total_floors_pct"] = round(float(inconsistent.mean() * 100), 2)
    df["Floor_Data_Repaired"] = inconsistent.astype(int)
    df["Total_Floors"] = np.maximum(df["Total_Floors"], df["Floor_No"])

    # 6 ---- price per sq ft -----------------------------------------------------------------
    rep["supplied_price_per_sqft_zero_rows"] = int((df["Price_per_SqFt"] == 0).sum())
    df["Price_per_SqFt_INR"] = (df["Price_in_Lakhs"] * C.LAKH / df["Size_in_SqFt"]).round(2)
    df = df.drop(columns=["Price_per_SqFt"])

    # 7 ---- amenities -----------------------------------------------------------------------
    rep["amenity_string_variants_before"] = int(df["Amenities"].nunique())
    df["Amenities"] = df["Amenities"].apply(
        lambda s: ", ".join(sorted({a.strip() for a in s.split(",") if a.strip()})) or "None")
    rep["amenity_string_variants_after"] = int(df["Amenities"].nunique())

    # 8 ---- outliers (flag, keep) -------------------------------------------------------------
    out = {}
    for col, flag in [("Price_in_Lakhs", "Outlier_Price"), ("Size_in_SqFt", "Outlier_Size"),
                      ("Price_per_SqFt_INR", "Outlier_PPSF")]:
        lo, hi = iqr_bounds(df[col])
        df[flag] = ((df[col] < lo) | (df[col] > hi)).astype(int)
        out[col] = {"lower_fence": round(float(lo), 2), "upper_fence": round(float(hi), 2),
                    "outliers": int(df[flag].sum()), "pct": round(float(df[flag].mean() * 100), 2)}
    rep["outliers_iqr"] = out
    rep["outlier_treatment"] = "flagged and kept (valid extreme combinations); PPSF winsorised at 99th pct for modelling"

    rep["rows_clean"] = int(len(df))
    rep["columns_clean"] = int(df.shape[1])
    df = df.sort_values("ID").reset_index(drop=True)
    return df, rep


def to_markdown(rep: dict) -> str:
    lines = ["# Data cleaning report", ""]
    for k, v in rep.items():
        if isinstance(v, dict):
            lines.append(f"- **{k}**")
            for kk, vv in v.items():
                lines.append(f"    - {kk}: {vv}")
        else:
            lines.append(f"- **{k}**: {v}")
    return "\n".join(lines) + "\n"


def main():
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(C.RAW_CSV)
    df, rep = clean(raw)
    df.to_csv(C.CLEAN_CSV, index=False)
    (C.REPORTS_DIR / "cleaning_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (C.REPORTS_DIR / "cleaning_report.md").write_text(to_markdown(rep), encoding="utf-8")
    print(to_markdown(rep))
    print(f"Saved -> {C.CLEAN_CSV}")


if __name__ == "__main__":
    main()
