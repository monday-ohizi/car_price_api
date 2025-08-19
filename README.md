# 🚗 Car Price Prediction API
This project is a **FastAPI-based machine learning API** that predicts the selling price of cars based on various features such as mileage, year, fuel type, transmission, and more.  
It allows users (or other apps) to send car details as input and receive a predicted price in return.
# Project Structure

# ⚙️ Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/monday-ohizi/car_price_api.git
   cd car_price_api
   
2. **Create and activate a virtual environment (optional but recommended)**
python -m venv venv
source venv/bin/activate     # On Mac/Linux
venv\Scripts\activate        # On Windows

3. **Install dependencies**
pip install -r requirements.txt

🚀 **Usage**
1. **Run the FastAPI server**
uvicorn main:app --reload

2. **Access the API**
API Root: http://127.0.0.1:8000
Interactive Docs (Swagger UI): http://127.0.0.1:8000/docs

3. **Example Request (via cURL)**

curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "Make": "Toyota",
  "Model": "Corolla",
  "Year": 2018,
  "Transmission": "Automatic",
  "Fuel_Type": "Petrol",
  "Mileage": 45000,
  "Engine_Size": 1.8
}'


And your API would return something like:
{
  "predicted_price": 12500.75
}


