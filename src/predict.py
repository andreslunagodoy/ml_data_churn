import pandas as pd

def predict_df(df: pd.DataFrame, model, preprocessor) -> pd.DataFrame:
    """Run predictions on a dataframe."""

    X = preprocessor.transform(df)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    predictions = pd.DataFrame({
        "customer_id": df["customerID"],
        "prediction": y_pred,
        "probability": y_proba
    })

    return predictions