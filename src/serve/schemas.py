"""Pydantic schemas for API request/response validation."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class Gender(str, Enum):
    Male = "Male"
    Female = "Female"


class Partner(str, Enum):
    Yes = "Yes"
    No = "No"


class Dependents(str, Enum):
    Yes = "Yes"
    No = "No"


class PhoneService(str, Enum):
    Yes = "Yes"
    No = "No"


class MultipleLines(str, Enum):
    Yes = "Yes"
    No = "No"
    No_phone_service = "No phone service"


class InternetService(str, Enum):
    DSL = "DSL"
    Fiber_optic = "Fiber optic"
    No = "No"


class OnlineSecurity(str, Enum):
    Yes = "Yes"
    No = "No"
    No_internet_service = "No internet service"


class OnlineBackup(str, Enum):
    Yes = "Yes"
    No = "No"
    No_internet_service = "No internet service"


class DeviceProtection(str, Enum):
    Yes = "Yes"
    No = "No"
    No_internet_service = "No internet service"


class TechSupport(str, Enum):
    Yes = "Yes"
    No = "No"
    No_internet_service = "No internet service"


class StreamingTV(str, Enum):
    Yes = "Yes"
    No = "No"
    No_internet_service = "No internet service"


class StreamingMovies(str, Enum):
    Yes = "Yes"
    No = "No"
    No_internet_service = "No internet service"


class Contract(str, Enum):
    Month_to_month = "Month-to-month"
    One_year = "One year"
    Two_year = "Two year"


class PaperlessBilling(str, Enum):
    Yes = "Yes"
    No = "No"


class PaymentMethod(str, Enum):
    Electronic_check = "Electronic check"
    Mailed_check = "Mailed check"
    Credit_card_automatic = "Credit card (automatic)"


class CustomerFeatures(BaseModel):
    """Raw input features for a single customer prediction (before encoding)."""
    # Numeric features
    tenure: int = Field(..., ge=0, le=100, description="Number of months the customer has stayed")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges in dollars")
    TotalCharges: float = Field(..., ge=0, description="Total charges in dollars")
    
    # Categorical features (raw values)
    gender: Gender
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: Partner
    Dependents: Dependents
    PhoneService: PhoneService
    MultipleLines: MultipleLines
    InternetService: InternetService
    OnlineSecurity: OnlineSecurity
    OnlineBackup: OnlineBackup
    DeviceProtection: DeviceProtection
    TechSupport: TechSupport
    StreamingTV: StreamingTV
    StreamingMovies: StreamingMovies
    Contract: Contract
    PaperlessBilling: PaperlessBilling
    PaymentMethod: PaymentMethod
    
    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 12,
                "MonthlyCharges": 70.35,
                "TotalCharges": 844.2,
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
            }
        }


class CustomerFeaturesEncoded(BaseModel):
    """Encoded features for a single customer prediction (internal use)."""
    # Numeric features
    tenure: int = Field(..., ge=0, le=100)
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)
    
    # Categorical features (one-hot encoded)
    gender_Male: int = Field(..., ge=0, le=1)
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner_Yes: int = Field(..., ge=0, le=1)
    Dependents_Yes: int = Field(..., ge=0, le=1)
    PhoneService_Yes: int = Field(..., ge=0, le=1)
    MultipleLines_No_phone_service: int = Field(..., ge=0, le=1)
    MultipleLines_Yes: int = Field(..., ge=0, le=1)
    InternetService_Fiber_optic: int = Field(..., ge=0, le=1)
    InternetService_No: int = Field(..., ge=0, le=1)
    OnlineSecurity_No_internet_service: int = Field(..., ge=0, le=1)
    OnlineSecurity_Yes: int = Field(..., ge=0, le=1)
    OnlineBackup_No_internet_service: int = Field(..., ge=0, le=1)
    OnlineBackup_Yes: int = Field(..., ge=0, le=1)
    DeviceProtection_No_internet_service: int = Field(..., ge=0, le=1)
    DeviceProtection_Yes: int = Field(..., ge=0, le=1)
    TechSupport_No_internet_service: int = Field(..., ge=0, le=1)
    TechSupport_Yes: int = Field(..., ge=0, le=1)
    StreamingTV_No_internet_service: int = Field(..., ge=0, le=1)
    StreamingTV_Yes: int = Field(..., ge=0, le=1)
    StreamingMovies_No_internet_service: int = Field(..., ge=0, le=1)
    StreamingMovies_Yes: int = Field(..., ge=0, le=1)
    Contract_One_year: int = Field(..., ge=0, le=1)
    Contract_Two_year: int = Field(..., ge=0, le=1)
    PaperlessBilling_Yes: int = Field(..., ge=0, le=1)
    PaymentMethod_Credit_card_automatic: int = Field(..., ge=0, le=1)
    PaymentMethod_Electronic_check: int = Field(..., ge=0, le=1)
    PaymentMethod_Mailed_check: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    """Response for a single prediction."""
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    churn_prediction: int = Field(..., ge=0, le=1, description="Binary prediction (1=churn, 0=no churn)")
    model_version: str = Field(..., description="Model version used for prediction")


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    customers: List[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: List[PredictionResponse]
    model_version: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: Optional[str] = None