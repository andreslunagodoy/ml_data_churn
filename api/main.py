from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from src.predict_single import predict_single
from api.schemas import CustomerFeatures, PredictionResponse
from src.model_loader import load_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.preprocessor = load_model()
    yield

app = FastAPI(title="Customer Churn API", version="1.0", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(customer: CustomerFeatures, request: Request):
    try:
        input_dict = customer.model_dump()
        result = predict_single(input_dict, request.app.state.model, request.app.state.preprocessor)
        return PredictionResponse(
            prediction=result["prediction"],
            probability=result["probability"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal prediction error")
