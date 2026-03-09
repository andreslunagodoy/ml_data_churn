from fastapi import FastAPI
from pydantic import ValidationError
import pandas as pd
from model.predict import load_model, predict
from api.schemas import CustomerFeatures, PredictionResponse

app = FastAPI(title="Customer Churn Prediction API")

model = load_model()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def make_prediction(customer: CustomerFeatures):
    try:
        data = pd.DataFrame([customer.dict()])
        result = predict(model, data)
        return PredictionResponse(
            prediction=int(result["prediction"][0]),
            churn_proba=float(result["churn_proba"][0])
        )
    except ValidationError as e:
        return {"error": str(e)}