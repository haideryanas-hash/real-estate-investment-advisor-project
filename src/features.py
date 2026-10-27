"""
Shared feature-engineering logic.

The SAME functions are used by the training pipeline and by the Streamlit app, so a single property
typed into the form goes through exactly the transformations the models were trained on.
"""
import json

import numpy as np
import pandas as pd

import config as C

TRANSPORT_SCORE = {"Low": 0.0, "Medium": 0.5, "High": 1.0}


# ----------------------------------------------------------------------------- city reference stats
def compute_city_stats(df: pd.DataFrame) -> dict:
    """Median price-per-sq-ft etc. per city (the 'market median' the brief's label rule refers to)."""
    ppsf = df["Price_in_Lakhs"] * C.LAKH / df["Size_in_SqFt"]
    tmp = df.assign(_ppsf=ppsf)
    g = tmp.groupby("City")
    cities = {}
    for city, sub in g:
        cities[city] = {
            "state": sub["State"].iloc[0],
            "median_ppsf": float(sub["_ppsf"].median()),
            "mean_ppsf": float(sub["_ppsf"].mean()),
            "median_price_lakhs": float(sub["Price_in_Lakhs"].median()),
            "n": int(len(sub)),
        }
    ratio = tmp["_ppsf"] / tmp["City"].map({k: v["median_ppsf"] for k, v in cities.items()})
    glob = {
        "median_ppsf": float(ppsf.median()),
        "ppsf_cap": float(ppsf.quantile(0.99)),       # winsorisation cap used for modelling
        "ratio_cap": float(ratio.quantile(0.99)),
    }
    return {"cities": cities, "global": glob}


def load_city_stats(path=C.CITY_STATS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- row-level features
def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Price_per_SqFt_INR" not in df:
        df["Price_per_SqFt_INR"] = df["Price_in_Lakhs"] * C.LAKH / df["Size_in_SqFt"]
    # amenities -> 5 binary flags + count
    for a in C.AMENITY_LIST:
        df[f"Has_{a}"] = df["Amenities"].fillna("").str.contains(a, regex=False).astype(int)
    df["Amenity_Count"] = df[[f"Has_{a}" for a in C.AMENITY_LIST]].sum(axis=1)
    df["Has_Parking"] = (df["Parking_Space"].astype(str).str.title() == "Yes").astype(int)
    df["Has_Security"] = (df["Security"].astype(str).str.title() == "Yes").astype(int)
    # infrastructure
    df["School_Density_Score"] = df["Nearby_Schools"] / 10.0
    df["Hospital_Score"] = df["Nearby_Hospitals"] / 10.0
    df["Transport_Score"] = df["Public_Transport_Accessibility"].map(TRANSPORT_SCORE)
    df["Infrastructure_Score"] = (df["School_Density_Score"] + df["Hospital_Score"] + df["Transport_Score"]) / 3
    # log of price: the 5-year target is multiplicative (price x growth), so log-price makes it additive
    df["Log_Price"] = np.log(df["Price_in_Lakhs"])
    # structure
    df["Floor_Ratio"] = (df["Floor_No"] / df["Total_Floors"].clip(lower=1)).clip(upper=1)
    df["Size_per_BHK"] = df["Size_in_SqFt"] / df["BHK"].clip(lower=1)
    return df


def add_city_relative(df: pd.DataFrame, city_stats: dict) -> pd.DataFrame:
    df = df.copy()
    med = df["City"].map(lambda c: city_stats["cities"].get(c, {}).get(
        "median_ppsf", city_stats["global"]["median_ppsf"]))
    df["City_Median_PPSF"] = med
    df["PPSF_vs_City_Median"] = df["Price_per_SqFt_INR"] / med
    return df


# ----------------------------------------------------------------------------- growth model & targets
def annual_growth_rate(df: pd.DataFrame) -> pd.Series:
    """Documented ASSUMPTION model (see config.py): base 8% + location/type/infrastructure/lifecycle."""
    r = pd.Series(C.BASE_GROWTH, index=df.index, dtype=float)
    r += df["City"].isin(C.TIER1_CITIES).astype(float) * C.TIER1_BONUS
    r += df["Property_Type"].map(C.PROPERTY_TYPE_ADJ).fillna(0.0)
    r += (df["Infrastructure_Score"] - 0.5) * C.INFRA_GROWTH_WEIGHT
    r += (df["Availability_Status"] == "Under_Construction").astype(float) * C.UNDER_CONSTRUCTION_ADJ
    r += (df["Age_of_Property"] <= C.NEW_BUILD_MAX_AGE).astype(float) * C.NEW_BUILD_ADJ
    r += (df["Age_of_Property"] > C.OLD_BUILD_MIN_AGE).astype(float) * C.OLD_BUILD_ADJ
    return r.clip(C.GROWTH_FLOOR, C.GROWTH_CAP)


def future_price(price_lakhs, rate, years=C.HORIZON_YEARS):
    return price_lakhs * (1.0 + rate) ** years


def investment_criteria(df: pd.DataFrame, rate: pd.Series) -> pd.DataFrame:
    """Boolean table, one column per rule in config.SCORE_WEIGHTS."""
    k = list(C.SCORE_WEIGHTS)
    out = pd.DataFrame(index=df.index)
    out[k[0]] = df["PPSF_vs_City_Median"] <= 1.0
    out[k[1]] = rate >= C.GROWTH_GOOD_THRESHOLD
    out[k[2]] = df["BHK"] >= 3
    out[k[3]] = df["Availability_Status"] == "Ready_to_Move"
    out[k[4]] = df["Public_Transport_Accessibility"] == "High"
    out[k[5]] = df["Infrastructure_Score"] >= C.INFRA_GOOD_THRESHOLD
    out[k[6]] = df["Has_Security"] == 1
    return out


def investment_score(criteria: pd.DataFrame) -> pd.Series:
    return sum(criteria[k].astype(int) * w for k, w in C.SCORE_WEIGHTS.items())


# ----------------------------------------------------------------------------- full pipeline entry points
def engineer(df: pd.DataFrame, city_stats: dict) -> pd.DataFrame:
    """Cleaned data -> all features + growth + targets. Used by feature_engineering.py and the app."""
    df = add_basic_features(df)
    df = add_city_relative(df, city_stats)
    df["Annual_Growth_Rate"] = annual_growth_rate(df)
    df[C.REG_TARGET] = future_price(df["Price_in_Lakhs"], df["Annual_Growth_Rate"]).round(2)
    df["Expected_Gain_Lakhs"] = (df[C.REG_TARGET] - df["Price_in_Lakhs"]).round(2)
    df["Expected_ROI_5Y_Pct"] = ((df[C.REG_TARGET] / df["Price_in_Lakhs"] - 1) * 100).round(2)
    crit = investment_criteria(df, df["Annual_Growth_Rate"])
    df["Investment_Score"] = investment_score(crit)
    df[C.CLF_TARGET] = (df["Investment_Score"] >= C.GOOD_SCORE_THRESHOLD).astype(int)
    return df


def model_frame(df: pd.DataFrame, city_stats: dict) -> pd.DataFrame:
    """Feature matrix in the exact column order the sklearn pipelines expect (with winsorisation)."""
    X = df[C.MODEL_FEATURES].copy()
    X["Price_per_SqFt_INR"] = X["Price_per_SqFt_INR"].clip(upper=city_stats["global"]["ppsf_cap"])
    X["PPSF_vs_City_Median"] = X["PPSF_vs_City_Median"].clip(upper=city_stats["global"]["ratio_cap"])
    return X
