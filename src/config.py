from pathlib import Path
from datetime import datetime
import yaml

TIMESTAMP = datetime.now().strftime("%y%m%d_%H%M%S")

BASE_DIR = Path(__file__).resolve().parent.parent

# Load tunable config from YAML
with open(BASE_DIR / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

CONFIG["timestamp"] = TIMESTAMP

MODEL_VERSION = CONFIG["model_version"]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "telco_churn.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / MODEL_VERSION / TIMESTAMP

INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output" / TIMESTAMP

LOG_DIR = BASE_DIR / "logs"

MODELS_FAMILY = {"logistic_regression": "linear", "random_forest": "tree", "gradient_boosting": "tree"}
