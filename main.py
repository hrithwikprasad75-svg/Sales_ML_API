
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# Create FastAPI application
app = FastAPI(title="Sales Manager Dashboard API")

# Locate saved models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load all five trained models
m1 = joblib.load(os.path.join(MODEL_DIR, "feature1_product_risk.pkl"))
m2 = joblib.load(os.path.join(MODEL_DIR, "feature2_customer_return.pkl"))
m3 = joblib.load(os.path.join(MODEL_DIR, "feature3_sales_forecast.pkl"))
m4 = joblib.load(os.path.join(MODEL_DIR, "feature4_cancellation_forecast.pkl"))
m6 = joblib.load(os.path.join(MODEL_DIR, "feature6_anomaly.pkl"))


# Input format for product risk
class ProductInput(BaseModel):
    CancellationRowRate: float
    CreditToPurchaseRatio: float
    AvgUnitPrice: float
    TotalInvoices: int
    TotalPurchaseAmount: float
    NegativeQuantityRows: int


# Input format for customer behaviour
class CustomerInput(BaseModel):
    TotalInvoices: int
    CancellationInvoices: int
    CancellationRate: float
    TotalPurchaseAmount: float
    AvgPurchaseValue: float


# Input format for unusual transaction detection
class TransactionInput(BaseModel):
    Quantity: float
    UnitPrice: float
    TotalAmount: float
    Hour: int


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Sales Manager Dashboard API is running",
        "models": 5
    }


# Health check endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}


# Feature 1: Product return risk
@app.post("/predict/product-risk")
def product_risk(data: ProductInput):
    try:
        features = pd.DataFrame([data.model_dump()])
        prediction = m1["model"].predict(features[m1["features"]])
        risk = m1["label_encoder"].inverse_transform(prediction)[0]

        return {"product_risk": str(risk)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Feature 2: Customer return behaviour
@app.post("/predict/customer-return")
def customer_return(data: CustomerInput):
    try:
        features = pd.DataFrame([data.model_dump()])
        model = m2["model"]
        columns = m2["features"]

        probability = float(
            model.predict_proba(features[columns])[0][1]
        )

        threshold = m2["threshold"]

        return {
            "high_risk_probability": round(probability, 4),
            "prediction": int(probability > threshold),
            "risk": "High" if probability > threshold else "Low"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Feature 3: Future sales prediction
@app.get("/forecast/sales")
def sales_forecast(days: int = 28):
    if days < 1 or days > 365:
        raise HTTPException(status_code=400, detail="Days must be 1 to 365")

    try:
        future = m3.make_future_dataframe(periods=days)
        forecast = m3.predict(future)

        result = forecast[
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ].tail(days).copy()

        result["ds"] = result["ds"].astype(str)

        return result.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Feature 4: Cancellation forecasting
@app.get("/forecast/cancellations")
def cancellation_forecast(days: int = 28):
    if days < 1 or days > 365:
        raise HTTPException(status_code=400, detail="Days must be 1 to 365")

    try:
        future = m4.make_future_dataframe(periods=days)
        forecast = m4.predict(future)

        result = forecast[["ds", "yhat"]].tail(days).copy()

        # Convert log predictions back to cancellation counts
        result["predicted_cancellations"] = np.clip(
            np.expm1(result["yhat"]), 0, None
        ).round(2)

        result["ds"] = result["ds"].astype(str)

        return result[
            ["ds", "predicted_cancellations"]
        ].to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Feature 6: Unusual transaction detection
@app.post("/predict/anomaly")
def detect_anomaly(data: TransactionInput):
    try:
        model = m6["model"]
        scaler = m6["scaler"]
        columns = m6["features"]

        features = pd.DataFrame([data.model_dump()])
        scaled_data = scaler.transform(features[columns])

        prediction = int(model.predict(scaled_data)[0])
        score = float(model.score_samples(scaled_data)[0])

        return {
            "anomaly": prediction == -1,
            "result": "Unusual Transaction" if prediction == -1 else "Normal Transaction",
            "anomaly_score": round(score, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Feature 7: Financial impact calculator
class FinancialInput(BaseModel):
    total_credit: float
    reduction_pct: float


@app.post("/financial-impact")
def financial_impact(data: FinancialInput):
    if data.total_credit < 0 or not 0 <= data.reduction_pct <= 1:
        raise HTTPException(status_code=400, detail="Invalid input values")

    saving = data.total_credit * data.reduction_pct

    return {
        "total_credit": data.total_credit,
        "reduction_percentage": data.reduction_pct * 100,
        "potential_saving": round(saving, 2)
    }
