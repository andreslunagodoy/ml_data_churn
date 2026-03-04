from pathlib import Path

#BASE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = Path("/Users/andresluna/mlprojects/ml_churn")

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "telco_churn.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "v1"

LOG_DIR = BASE_DIR / "logs"

CONFIG = {
    "target_column": "Churn",
    "target_map" : {'Yes': 1, 'No': 0},
    "test_size": 0.2,
    "random_state": 42,
    "model_type": "logistic_regression"  # or "random_forest"
}

MODELS_FAMILY = {"logistic_regression" : "linear", "random_forest" : "tree", "gradient_boosting" : "tree"}