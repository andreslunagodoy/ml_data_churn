import pandas as pd

REQUIRED_COLUMNS = [
    'customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract',
    'PaymentMethod', 'InternetService', 'gender', 'Partner',
    'Dependents', 'PhoneService', 'PaperlessBilling', 'MultipleLines',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
]

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

def predict_df(df: pd.DataFrame, model: BaseEstimator, preprocessor: Pipeline) -> pd.DataFrame:
    """Run predictions on a dataframe."""

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    X = preprocessor.transform(df)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    predictions = pd.DataFrame({
        "customer_id": df["customerID"],
        "prediction": y_pred,
        "probability": y_proba
    })

    return predictions