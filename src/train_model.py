import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.config import CONFIG, MODEL_DIR
from src.logger import logger
from src.preprocessing import preprocess_data
from src.data_loader import load_processed_data

def train():
    df = load_processed_data()
    y = df[CONFIG["target_col"]]
    X = df.drop(columns=[CONFIG["target_col"]])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_seed"]
    )

    X_train = preprocess_data(X_train, fit=True, scaler_path=MODEL_DIR / "scaler.pkl", encoder_path=MODEL_DIR / "encoder.pkl")
    X_val = preprocess_data(X_val, fit=False, scaler_path=MODEL_DIR / "scaler.pkl", encoder_path=MODEL_DIR / "encoder.pkl")

    model = RandomForestClassifier(random_state=CONFIG["random_seed"])
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_DIR / CONFIG["model_version"] / "model.pkl")
    logger.info(f"Model saved to {MODEL_DIR / CONFIG['model_version'] / 'model.pkl'}")
c