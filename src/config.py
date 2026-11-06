from pathlib import Path
from datetime import datetime
from typing import Any
import yaml

TIMESTAMP = datetime.now().strftime("%y%m%d_%H%M%S")

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "telco_churn.csv"

MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR = BASE_DIR / "output" / TIMESTAMP

LOG_DIR = BASE_DIR / "logs"

MODELS_FAMILY = {"logistic_regression": "linear", "random_forest": "tree", "gradient_boosting": "tree"}


def validate_model_type(model_type: str) -> None:
    """Raise a clear error if model_type is not a supported model."""
    if model_type not in MODELS_FAMILY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Valid options: {', '.join(MODELS_FAMILY)}"
        )


def validate_config(config: dict[str, Any]) -> None:
    """Raise a clear error if the config has invalid values."""
    validate_model_type(config["model_type"])
    if not 0 < config["test_size"] < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {config['test_size']}")


def load_config(override_path: Path | str | None = None) -> dict[str, Any]:
    """Load config.yaml, apply an optional override file on top, validate, and return it."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    if override_path:
        with open(override_path) as f:
            overrides = yaml.safe_load(f) or {}
        unknown = sorted(set(overrides) - set(config))
        if unknown:
            raise ValueError(f"Unknown config keys in {override_path}: {unknown}. Valid keys: {sorted(config)}")
        config.update(overrides)

    validate_config(config)
    config["timestamp"] = TIMESTAMP
    return config


def get_model_path(config: dict[str, Any]) -> Path:
    """Folder where a training run saves its artifacts: models/<version>/<timestamp>."""
    return MODEL_DIR / config["model_version"] / config["timestamp"]
