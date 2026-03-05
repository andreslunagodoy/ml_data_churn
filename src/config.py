from pathlib import Path
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%y%m%d_%H%M%S")
MODEL_VERSION = "v1"

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "telco_churn.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / MODEL_VERSION / TIMESTAMP

INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output" / TIMESTAMP

LOG_DIR = BASE_DIR / "logs"

MODELS_FAMILY = {"logistic_regression" : "linear", "random_forest" : "tree", "gradient_boosting" : "tree"}

CONFIG = {
    "target_column": "Churn",
    "target_map" : {'Yes': 1, 'No': 0},
    "test_size": 0.2,
    "random_state": 42,
    "model_type": "logistic_regression",
    "model_version": "v1",
    "timestamp": TIMESTAMP
}

