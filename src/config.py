"""
Central configuration for the Real Estate Investment Advisor project.

Everything that is an *assumption* (growth rates, label thresholds) lives here so it can be
reviewed, changed and re-run in one place. The dataset contains no market-return history, so
the growth model below is a documented, transparent business rule - NOT observed market data.
"""
from pathlib import Path

# ----------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "india_housing_prices.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
CLEAN_CSV = PROCESSED_DIR / "cleaned_housing.csv"
FEATURED_CSV = PROCESSED_DIR / "featured_housing.csv"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
EDA_DIR = OUTPUTS_DIR / "eda"
EVAL_DIR = OUTPUTS_DIR / "evaluation"
REPORTS_DIR = OUTPUTS_DIR / "reports"

CLASSIFIER_PATH = MODELS_DIR / "good_investment_classifier.joblib"
REGRESSOR_PATH = MODELS_DIR / "future_price_regressor.joblib"
CITY_STATS_PATH = MODELS_DIR / "city_stats.json"
MODEL_META_PATH = MODELS_DIR / "model_metadata.json"
IMPORTANCE_CLF_PATH = MODELS_DIR / "feature_importance_classifier.csv"
IMPORTANCE_REG_PATH = MODELS_DIR / "feature_importance_regressor.csv"

# MLflow: SQLite backend (needed for the Model Registry)
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"   # forward slashes: safe on Windows
MLFLOW_EXPERIMENT = "Real_Estate_Investment_Advisor"
REGISTERED_CLF_NAME = "GoodInvestmentClassifier"
REGISTERED_REG_NAME = "FuturePriceRegressor"

# ----------------------------------------------------------------------------- general
RANDOM_STATE = 42
TEST_SIZE = 0.20
REFERENCE_YEAR = 2025          # the dataset's Age_of_Property = 2025 - Year_Built
HORIZON_YEARS = 5
LAKH = 100_000                 # 1 lakh = 100,000 INR

# ----------------------------------------------------------------------------- growth model (ASSUMPTIONS)
# Future_Price_5Y = Price * (1 + r) ** 5, with r varying by location / type / features
BASE_GROWTH = 0.08                                   # brief's example fixed rate
TIER1_CITIES = {"Mumbai", "New Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata",
                "Pune", "Ahmedabad", "Gurgaon", "Noida"}
TIER1_BONUS = 0.005
PROPERTY_TYPE_ADJ = {"Apartment": 0.000, "Independent House": 0.003, "Villa": 0.006}
INFRA_GROWTH_WEIGHT = 0.04                           # (infra_score - 0.5) * 0.04  -> about +/-1.2%
UNDER_CONSTRUCTION_ADJ = 0.005                       # completion premium
NEW_BUILD_ADJ, NEW_BUILD_MAX_AGE = 0.005, 5
OLD_BUILD_ADJ, OLD_BUILD_MIN_AGE = -0.005, 25
GROWTH_FLOOR, GROWTH_CAP = 0.05, 0.12

# ----------------------------------------------------------------------------- "Good Investment" rule
# Multi-factor score (brief: "Price vs Median", "Price per Sq Ft", "Multi-factor score").
GROWTH_GOOD_THRESHOLD = 0.085                        # appreciation rate > threshold
INFRA_GOOD_THRESHOLD = 0.60
SCORE_WEIGHTS = {
    "Priced below city median per sq ft": 2,
    "Projected growth above threshold": 2,
    "3+ BHK": 1,
    "Ready to move": 1,
    "High public-transport access": 1,
    "Strong infrastructure score": 1,
    "Security available": 1,
}
GOOD_SCORE_THRESHOLD = 6                              # Good if score >= 6 of 9 (two-thirds of weighted criteria)

# ----------------------------------------------------------------------------- model feature lists
AMENITY_LIST = ["Playground", "Gym", "Garden", "Pool", "Clubhouse"]
NUMERIC_FEATURES = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Log_Price", "Price_per_SqFt_INR", "PPSF_vs_City_Median",
    "Age_of_Property", "Floor_No", "Total_Floors", "Floor_Ratio", "Size_per_BHK",
    "Nearby_Schools", "Nearby_Hospitals", "School_Density_Score", "Infrastructure_Score",
    "Amenity_Count", "Has_Parking", "Has_Security",
] + [f"Has_{a}" for a in AMENITY_LIST]
CATEGORICAL_FEATURES = [
    "City", "Property_Type", "Furnished_Status", "Public_Transport_Accessibility",
    "Facing", "Owner_Type", "Availability_Status",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
CLF_TARGET = "Good_Investment"
REG_TARGET = "Future_Price_5Y_Lakhs"
