from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from car_feature_engineering import add_features
from sklearn.preprocessing import FunctionTransformer


# Use it in a transformer
feature_creator = FunctionTransformer(add_features, validate=False)

# Load pipeline (includes preprocessing + model)
pipeline = joblib.load("car_price_prediction.pkl")

# FastAPI app
app = FastAPI(title="Car Price Prediction API")

# Input schema
class CarFeatures(BaseModel):
    Make: str
    Model: str
    Year: int
    Engine_Size: float   # ✅ use underscore
    Mileage: float
    Fuel_Type: str       # ✅ use underscore
    Transmission: str

@app.get("/")
def home():
    return {"message": "Welcome to Car Price Prediction API!"}

@app.post("/predict")
def predict_price(features: CarFeatures):
    try:
        # Convert input to DataFrame
        input_dict = features.dict()

        # ✅ Rename keys back to match training dataset column names
        input_data = pd.DataFrame([{
            "Make": input_dict["Make"],
            "Model": input_dict["Model"],
            "Year": input_dict["Year"],
            "Engine Size": input_dict["Engine_Size"],   # match training column
            "Mileage": input_dict["Mileage"],
            "Fuel Type": input_dict["Fuel_Type"],       # match training column
            "Transmission": input_dict["Transmission"]
        }])

        # Pipeline handles preprocessing internally
        prediction = pipeline.predict(input_data)

        return {"predicted_price": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
