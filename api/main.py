from fastapi import FastAPI
from src.predict_single import predict_single
from api.schemas import CustomerFeatures, PredictionResponse
from src.model_loader import load_model

app = FastAPI(title="Customer Churn API", version="1.0")

# Load the model and preprocessor once here
# (optional: you can have predict_single handle loading internally)
# model, preprocessor = load_model_and_preprocessor()

@app.on_event("startup")
def load_artifacts():
    global model, preprocessor
    model, preprocessor = load_model()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(customer: CustomerFeatures):
    try: 
        input_dict = customer.model_dump()
        result = predict_single(input_dict,model,preprocessor)
        return PredictionResponse(
            prediction=result["prediction"],
            probability=result["probability"]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))