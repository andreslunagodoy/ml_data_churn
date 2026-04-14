import pytest
from fastapi.testclient import TestClient
from api.main import app

VALID_INPUT = {
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

@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_endpoint(client):
    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability"] <= 1

def test_invalid_predict_input(client):
    invalid_input = {"feature1": "wrong_type"}
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422

def test_missing_required_field(client):
    incomplete = VALID_INPUT.copy()
    del incomplete["tenure"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422

def test_invalid_gender_value(client):
    bad_input = VALID_INPUT.copy()
    bad_input["gender"] = "Other"
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422

def test_invalid_contract_value(client):
    bad_input = VALID_INPUT.copy()
    bad_input["Contract"] = "Weekly"
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422

def test_negative_tenure(client):
    bad_input = VALID_INPUT.copy()
    bad_input["tenure"] = -1
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422

def test_negative_monthly_charges(client):
    bad_input = VALID_INPUT.copy()
    bad_input["MonthlyCharges"] = -50.0
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422

def test_invalid_senior_citizen(client):
    bad_input = VALID_INPUT.copy()
    bad_input["SeniorCitizen"] = 2
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422
