import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import joblib
import pandas as pd
from car_feature_engineering import add_features
from sklearn.preprocessing import FunctionTransformer


# Setup logging
logging.basicConfig(
    filename="app.log",           # log file name
    level=logging.INFO,           # log level (INFO, WARNING, ERROR)
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Use it in a transformer
feature_creator = FunctionTransformer(add_features, validate=False)

# Load pipeline (includes preprocessing + model)
pipeline = joblib.load("car_price_prediction.pkl")

# Versioned API
app_v1 = FastAPI(
    title="Car Price Prediction API v1",
    version="1.0"
)

# Main API (with versioning + root endpoints)
app = FastAPI(title="Car Price Prediction API")

# Mount v1 under /v1
app.mount("/v1", app_v1)

# Input schema with validation
class CarFeatures(BaseModel):
    Make: str = Field(..., description="Car manufacturer (e.g., Toyota, Ford)")
    Model: str = Field(..., description="Car model (e.g., Corolla, Civic)")
    Year: int = Field(..., ge=1980, le=2025, description="Year of manufacture")
    Engine_Size: float = Field(..., ge=0.5, le=10, description="Engine size in liters")
    Mileage: float = Field(..., ge=0, description="Car mileage in km")
    Fuel_Type: str = Field(..., description="Fuel type (e.g., Petrol, Diesel, Hybrid, Electric)")
    Transmission: str = Field(..., description="Transmission type (Manual or Automatic)")

    @validator("Fuel_Type")
    def fuel_type_check(cls, v):
        allowed = ["Petrol", "Diesel", "Hybrid", "Electric"]
        if v not in allowed:
            raise ValueError(f"Fuel_Type must be one of {allowed}")
        return v

    @validator("Transmission")
    def transmission_check(cls, v):
        allowed = ["Manual", "Automatic"]
        if v not in allowed:
            raise ValueError(f"Transmission must be one of {allowed}")
        return v

# Root + Health endpoints
@app.get("/")
def home():
    logging.info("Root endpoint accessed")
    return {
        "message": "Welcome to Car Price Prediction API!",
        "available_versions": ["/predict (default)", "/v1/predict"]
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running"}

# Prediction Endpoint (Shared)
def make_prediction(features: CarFeatures):
    input_dict = features.dict()

    # Match training column names
    input_data = pd.DataFrame([{
        "Make": input_dict["Make"],
        "Model": input_dict["Model"],
        "Year": input_dict["Year"],
        "Engine Size": input_dict["Engine_Size"],
        "Mileage": input_dict["Mileage"],
        "Fuel Type": input_dict["Fuel_Type"],
        "Transmission": input_dict["Transmission"]
    }])

    prediction = pipeline.predict(input_data)
    logging.info(f"Prediction successful for input: {input_dict}")
    return {"predicted_price": round(float(prediction[0]), 2)}


# Default /predict (no version)
@app.post("/predict")
def predict_price_default(features: CarFeatures):
    try:
        return make_prediction(features)
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


# Versioned /v1/predict
@app_v1.post("/predict")
def predict_price_v1(features: CarFeatures):
    try:
        return make_prediction(features)
    except Exception as e:
        logging.error(f"[v1] Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"[v1] Prediction failed: {str(e)}")


# Global Error Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.critical(f"Unexpected error at {request.url} - {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Unexpected error: {str(exc)}"}
    )
