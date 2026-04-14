import pytest
import pandas as pd
from src.predict_single import predict_single
from src.predict import predict_df
from src.model_loader import load_model

SAMPLE_INPUT = {
    "customerID":"3524-WQDSG",
    "gender":"Female",
    "SeniorCitizen":0,
    "Partner":"Yes",
    "Dependents":"Yes",
    "tenure":43,
    "PhoneService":"Yes",
    "MultipleLines":"Yes",
    "InternetService":"Fiber optic",
    "OnlineSecurity":"No",
    "OnlineBackup":"No",
    "DeviceProtection":"Yes",
    "TechSupport":"No",
    "StreamingTV":"Yes",
    "StreamingMovies":"Yes",
    "Contract":"Month-to-month",
    "PaperlessBilling":"Yes",
    "PaymentMethod":"Bank transfer (automatic)",
    "MonthlyCharges":99.3,
    "TotalCharges":"4209.95"
}

def test_predict_customer():
    model, preprocessor = load_model()
    result = predict_single(SAMPLE_INPUT, model, preprocessor)
    assert "prediction" in result
    assert "probability" in result
    assert 0 <= result["probability"] <= 1
    assert isinstance(result["prediction"], int)

def test_predict_df_output_columns():
    model, preprocessor = load_model()
    df = pd.DataFrame([SAMPLE_INPUT])
    result = predict_df(df, model, preprocessor)
    assert list(result.columns) == ["customer_id", "prediction", "probability"]

def test_predict_df_missing_columns():
    model, preprocessor = load_model()
    df = pd.DataFrame([{"customerID": "123", "tenure": 10}])
    with pytest.raises(ValueError, match="missing required columns"):
        predict_df(df, model, preprocessor)

def test_predict_df_probability_range():
    model, preprocessor = load_model()
    df = pd.DataFrame([SAMPLE_INPUT])
    result = predict_df(df, model, preprocessor)
    assert (result["probability"] >= 0).all()
    assert (result["probability"] <= 1).all()
