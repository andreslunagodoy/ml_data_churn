from src.predict_single import predict_single
from src.model_loader import load_model

def test_predict_customer():

    model, preprocessor = load_model()

    sample_input = {
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
    result = predict_single(sample_input, model, preprocessor)
    assert "prediction" in result
    assert "probability" in result
    assert 0 <= result["probability"] <= 1
    assert isinstance(result["prediction"], int)