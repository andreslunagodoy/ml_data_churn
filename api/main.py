from fastapi import FastAPI
from src.predict import predict
from api.schemas import CustomerFeatures, PredictionResponse

app = FastAPI(title="Customer Churn API", version="1.0")

# Load the model and preprocessor once here
# (optional: you can have predict_single handle loading internally)
# model, preprocessor = load_model_and_preprocessor()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(customer: CustomerFeatures):
    try: 
        input_dict = customer.model_dump()
        result = predict(input_dict)
        return PredictionResponse(
            prediction=result["prediction"],
            probability=result["probability"]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))