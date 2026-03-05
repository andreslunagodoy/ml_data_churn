# scripts/predict.py
import joblib
import pandas as pd
from src.config import MODEL_DIR, INPUT_DIR, OUTPUT_DIR

def main():
    # Paths
    model_path = MODEL_DIR / "current/model.pkl"
    preprocessor_path = MODEL_DIR / "current/preprocessor.pkl"
    
    input_data_path = INPUT_DIR / "new_data.csv"
    output_path = OUTPUT_DIR / "predictions.csv"

    # Load model and preprocessor
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    # Load new data
    df_new = pd.read_csv(input_data_path)

    # Preprocess
    X_new = preprocessor.transform(df_new)

    # Predict
    y_pred = model.predict(X_new)
    y_proba = model.predict_proba(X_new)[:, 1]

    # Save predictions
    predictions = pd.DataFrame({
        "customer_id": df_new["customerID"],
        "prediction": y_pred,
        "probability": y_proba
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()