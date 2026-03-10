# scripts/predict.py
import joblib
import pandas as pd
from src.config import MODEL_DIR

def predict(input_data: dict) -> dict:
    # Paths
    model_path = MODEL_DIR / "current/model.pkl"
    preprocessor_path = MODEL_DIR / "current/preprocessor.pkl"

    # Load model and preprocessor
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    # Load new data
    df_new = pd.DataFrame([input_data])

    # Preprocess
    X_new = preprocessor.transform(df_new)

    # Predict
    y_pred = model.predict(X_new)
    y_proba = model.predict_proba(X_new)[:, 1]

    # Save predictions
    prediction_df = pd.DataFrame({
        "customer_id": df_new["customerID"],
        "prediction": y_pred,
        "probability": y_proba
    })

    prediction_dict = prediction_df.to_dict(orient="records")[0]

    return prediction_dict