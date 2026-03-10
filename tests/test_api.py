from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_endpoint():
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
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data

def test_invalid_predict_input():

    invalid_input = {"feature1": "wrong_type"}

    response = client.post("/predict", json=invalid_input)

    assert response.status_code in [400, 422]