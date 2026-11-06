import math
from typing import Literal
from pydantic import BaseModel, Field, field_validator

class CustomerFeatures(BaseModel):
    customerID: str
    gender: Literal['Male', 'Female']
    SeniorCitizen: Literal[0, 1]
    Partner: Literal['Yes', 'No']
    Dependents: Literal['Yes', 'No']
    tenure: int = Field(ge=0)
    PhoneService: Literal['Yes', 'No']
    MultipleLines: Literal['Yes', 'No', 'No phone service']
    InternetService: Literal['DSL', 'Fiber optic', 'No']
    OnlineSecurity: Literal['Yes', 'No', 'No internet service']
    OnlineBackup: Literal['Yes', 'No', 'No internet service']
    DeviceProtection: Literal['Yes', 'No', 'No internet service']
    TechSupport: Literal['Yes', 'No', 'No internet service']
    StreamingTV: Literal['Yes', 'No', 'No internet service']
    StreamingMovies: Literal['Yes', 'No', 'No internet service']
    Contract: Literal['Month-to-month', 'One year', 'Two year']
    PaperlessBilling: Literal['Yes', 'No']
    PaymentMethod: Literal[
        'Electronic check', 'Mailed check',
        'Bank transfer (automatic)', 'Credit card (automatic)']
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: str

    @field_validator('TotalCharges')
    @classmethod
    def total_charges_is_blank_or_non_negative_number(cls, v: str) -> str:
        if v.strip() == '':
            return v  # the training data has blanks for brand-new customers
        try:
            number = float(v)
        except ValueError:
            raise ValueError('TotalCharges must be a number or blank')
        if not math.isfinite(number) or number < 0:
            raise ValueError('TotalCharges must be a finite number >= 0')
        return v

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
