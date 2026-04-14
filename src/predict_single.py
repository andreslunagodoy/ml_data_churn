import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from src.predict import predict_df

def predict_single(input_data: dict, model: BaseEstimator, preprocessor: Pipeline) -> dict:
    """Predict a single customer."""
    
    df = pd.DataFrame([input_data])
    predictions = predict_df(df, model, preprocessor)

    return predictions.to_dict(orient="records")[0]