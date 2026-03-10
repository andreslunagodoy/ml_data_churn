import joblib
from src.config import MODEL_DIR

def load_model(
    model_path = MODEL_DIR / "current/model.pkl",
    preprocessor_path = MODEL_DIR / "current/preprocessor.pkl"
):
    """Load model and preprocessor. Paths can be overridden if needed."""
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor