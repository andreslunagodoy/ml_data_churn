from pydantic import BaseModel
from typing import List

class CustomerFeatures(BaseModel):
    age: int
    tenure: float
    balance: float
    products_number: int
    has_cr_card: int
    is_active_member: int
    estimated_salary: float

class PredictionResponse(BaseModel):
    prediction: int
    churn_proba: float