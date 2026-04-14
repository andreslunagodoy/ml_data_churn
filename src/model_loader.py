import joblib
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from src.config import MODEL_DIR

def load_model(
    model_path: Path = MODEL_DIR / "current/model.pkl",
    preprocessor_path: Path = MODEL_DIR / "current/preprocessor.pkl"
) -> tuple[BaseEstimator, Pipeline]:
    """Load model and preprocessor. Paths can be overridden if needed."""

    if model_path.parent != preprocessor_path.parent:
        raise ValueError(
            f"Model and preprocessor must be from the same directory: "
            f"{model_path.parent} != {preprocessor_path.parent}"
        )

    config_path = model_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {model_path.parent} — cannot verify artifact integrity")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor
