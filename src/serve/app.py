"""FastAPI application for serving the churn prediction model."""
import mlflow
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import MLFLOW_TRACKING_URI, MODEL_URI, API_HOST, API_PORT
from src.serve.schemas import (
    CustomerFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
)

# Global model variable
model = None
model_version = None


def load_model_from_registry():
    """Load the model from MLflow Model Registry (Production stage)."""
    global model, model_version
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        # Load model from registry
        model = mlflow.pyfunc.load_model(MODEL_URI)
        
        # Get model version info
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions("telco-churn-model", stages=["Production"])
        if latest_version:
            model_version = latest_version[0].version
        else:
            model_version = "unknown"
        
        print(f"Model loaded successfully from {MODEL_URI}")
        print(f"Model version: {model_version}")
        return True
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None
        model_version = None
        return False


def encode_features(raw_features: CustomerFeatures) -> dict:
    """Convert raw customer features to one-hot encoded format expected by the model.
    
    Note: Column names must match the training data exactly (with spaces, not underscores).
    Note: One-hot encoded columns must be boolean type.
    """
    data = raw_features.model_dump()
    
    # Create encoded feature dict with exact column names from training
    encoded = {
        "tenure": int(data["tenure"]),
        "MonthlyCharges": float(data["MonthlyCharges"]),
        "TotalCharges": float(data["TotalCharges"]),
        "SeniorCitizen": int(data["SeniorCitizen"]),
    }
    
    # Gender: Female is reference (0), Male is 1
    encoded["gender_Male"] = bool(data["gender"] == "Male")
    
    # Partner: No is reference (0), Yes is 1
    encoded["Partner_Yes"] = bool(data["Partner"] == "Yes")
    
    # Dependents: No is reference (0), Yes is 1
    encoded["Dependents_Yes"] = bool(data["Dependents"] == "Yes")
    
    # PhoneService: No is reference (0), Yes is 1
    encoded["PhoneService_Yes"] = bool(data["PhoneService"] == "Yes")
    
    # MultipleLines: "No phone service" is reference (0)
    encoded["MultipleLines_No phone service"] = bool(data["MultipleLines"] == "No phone service")
    encoded["MultipleLines_Yes"] = bool(data["MultipleLines"] == "Yes")
    
    # InternetService: "DSL" is reference (0)
    encoded["InternetService_Fiber optic"] = bool(data["InternetService"] == "Fiber optic")
    encoded["InternetService_No"] = bool(data["InternetService"] == "No")
    
    # OnlineSecurity: "No internet service" is reference (0)
    encoded["OnlineSecurity_No internet service"] = bool(data["OnlineSecurity"] == "No internet service")
    encoded["OnlineSecurity_Yes"] = bool(data["OnlineSecurity"] == "Yes")
    
    # OnlineBackup: "No internet service" is reference (0)
    encoded["OnlineBackup_No internet service"] = bool(data["OnlineBackup"] == "No internet service")
    encoded["OnlineBackup_Yes"] = bool(data["OnlineBackup"] == "Yes")
    
    # DeviceProtection: "No internet service" is reference (0)
    encoded["DeviceProtection_No internet service"] = bool(data["DeviceProtection"] == "No internet service")
    encoded["DeviceProtection_Yes"] = bool(data["DeviceProtection"] == "Yes")
    
    # TechSupport: "No internet service" is reference (0)
    encoded["TechSupport_No internet service"] = bool(data["TechSupport"] == "No internet service")
    encoded["TechSupport_Yes"] = bool(data["TechSupport"] == "Yes")
    
    # StreamingTV: "No internet service" is reference (0)
    encoded["StreamingTV_No internet service"] = bool(data["StreamingTV"] == "No internet service")
    encoded["StreamingTV_Yes"] = bool(data["StreamingTV"] == "Yes")
    
    # StreamingMovies: "No internet service" is reference (0)
    encoded["StreamingMovies_No internet service"] = bool(data["StreamingMovies"] == "No internet service")
    encoded["StreamingMovies_Yes"] = bool(data["StreamingMovies"] == "Yes")
    
    # Contract: "Month-to-month" is reference (0)
    encoded["Contract_One year"] = bool(data["Contract"] == "One year")
    encoded["Contract_Two year"] = bool(data["Contract"] == "Two year")
    
    # PaperlessBilling: No is reference (0), Yes is 1
    encoded["PaperlessBilling_Yes"] = bool(data["PaperlessBilling"] == "Yes")
    
    # PaymentMethod: "Electronic check" is reference (0)
    encoded["PaymentMethod_Credit card (automatic)"] = bool(data["PaymentMethod"] == "Credit card (automatic)")
    encoded["PaymentMethod_Electronic check"] = bool(data["PaymentMethod"] == "Electronic check")
    encoded["PaymentMethod_Mailed check"] = bool(data["PaymentMethod"] == "Mailed check")
    
    return encoded


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    load_model_from_registry()
    yield
    # Shutdown (if needed)
    pass


app = FastAPI(
    title="Telco Churn Prediction API",
    description="API for predicting customer churn using the registered MLflow model",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=str(model_version) if model_version else None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerFeatures):
    """Predict churn probability for a single customer."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert raw features to encoded format
        encoded = encode_features(customer)
        
        # Convert to DataFrame for prediction
        import pandas as pd
        input_df = pd.DataFrame([encoded])
        
        # Make prediction
        # MLflow pyfunc model returns probabilities for binary classification
        prediction = model.predict(input_df)
        
        # Handle different return types
        if hasattr(prediction, 'shape') and len(prediction.shape) > 1:
            # Probability output
            churn_prob = float(prediction[0][1]) if prediction.shape[1] > 1 else float(prediction[0][0])
        else:
            # Binary prediction
            churn_prob = float(prediction[0])
        
        churn_pred = 1 if churn_prob >= 0.5 else 0
        
        return PredictionResponse(
            churn_probability=churn_prob,
            churn_prediction=churn_pred,
            model_version=str(model_version) if model_version else "unknown",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_churn_batch(request: BatchPredictionRequest):
    """Predict churn probability for multiple customers."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert all raw features to encoded format
        encoded_list = [encode_features(c) for c in request.customers]
        
        import pandas as pd
        input_df = pd.DataFrame(encoded_list)
        
        predictions = model.predict(input_df)
        
        results = []
        for i, pred in enumerate(predictions):
            if hasattr(pred, '__len__') and len(pred) > 1:
                churn_prob = float(pred[1])
            else:
                churn_prob = float(pred)
            
            churn_pred = 1 if churn_prob >= 0.5 else 0
            
            results.append(PredictionResponse(
                churn_probability=churn_prob,
                churn_prediction=churn_pred,
                model_version=str(model_version) if model_version else "unknown",
            ))
        
        return BatchPredictionResponse(
            predictions=results,
            model_version=str(model_version) if model_version else "unknown",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


def main():
    """Entry point for running the server."""
    import uvicorn
    uvicorn.run("src.serve.app:app", host=API_HOST, port=API_PORT, reload=True)


if __name__ == "__main__":
    main()