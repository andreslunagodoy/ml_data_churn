import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "customer_churn.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "churn_processed.csv"
MODEL_DIR = BASE_DIR / "models"

CONFIG = {
    "random_seed": 42,
    "test_size": 0.2,
    "model_version": "v1",
    "target_col": "churn"
}