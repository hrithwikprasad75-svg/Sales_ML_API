from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
import numpy as np
import pandas as pd
from pydantic import AliasChoices, BaseModel, ConfigDict, EmailStr, Field, field_validator
from sqlalchemy.orm import Session

from auth import (
    create_access_token,
    get_current_user,
    get_current_user_optional,
    hash_password,
    require_role,
    verify_password,
)
from database import TransactionRecord, User, get_db, init_db
from services import (
    delete_transaction,
    get_analytics_summary,
    get_customer_features_from_db,
    get_daily_series_from_db,
    get_product_features_from_db,
    ingest_transaction,
    ingest_transactions_batch,
    load_ml_models,
    log_prediction,
    parse_transactions_csv_bytes,
    predict_customer_return_from_db,
    predict_product_risk_from_db,
    query_predictions,
    query_transactions,
    serialize_transaction,
    update_transaction,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_ml_models()
    yield


app = FastAPI(
    title="Retail Intelligence & Sales Manager Dashboard API",
    description=(
        "Enterprise retail analytics API featuring JWT Authentication, Role-Based Access Control (Admin, "
        "Sales Manager, Analyst), 3NF Relational Database Schema, Full CRUD, Real-Time Anomaly Detection, "
        "and Machine Learning Decision Support."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# -------------------- PYDANTIC SCHEMAS --------------------
class UserRegister(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., max_length=120)
    password: str = Field(..., min_length=6, max_length=64)
    role: str = Field(default="Sales Manager", pattern="^(Admin|Sales Manager|Analyst)$")

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        clean = v.strip().lower()
        if "@" not in clean or "." not in clean.split("@")[-1]:
            raise ValueError("Invalid email format. Must contain '@' and domain.")
        return clean


class UserLogin(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: int
    name: str
    email: str
    role: str
    is_active: bool
    created_at: Optional[datetime] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class ProductInput(BaseModel):
    CancellationRowRate: float = Field(..., ge=0.0, le=1.0)
    CreditToPurchaseRatio: float = Field(..., ge=0.0)
    AvgUnitPrice: float = Field(..., ge=0.0)
    TotalInvoices: int = Field(..., ge=1)
    TotalPurchaseAmount: float
    NegativeQuantityRows: int = Field(..., ge=0)


class CustomerInput(BaseModel):
    TotalInvoices: int = Field(..., ge=1)
    CancellationInvoices: int = Field(..., ge=0)
    CancellationRate: float = Field(..., ge=0.0, le=1.0)
    TotalPurchaseAmount: float
    AvgPurchaseValue: float


class TransactionInput(BaseModel):
    Quantity: float
    UnitPrice: float = Field(..., ge=0.0)
    TotalAmount: float
    Hour: int = Field(..., ge=0, le=23)


class FinancialInput(BaseModel):
    total_credit: float = Field(..., ge=0.0)
    reduction_pct: float = Field(..., ge=0.0, le=1.0)


class TransactionCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    invoice_no: str = Field(..., min_length=1, validation_alias=AliasChoices("invoice_no", "InvoiceNo"))
    stock_code: str = Field(..., min_length=1, validation_alias=AliasChoices("stock_code", "StockCode"))
    description: Optional[str] = Field(
        default="Unknown",
        validation_alias=AliasChoices("description", "Description"),
    )
    quantity: float = Field(..., validation_alias=AliasChoices("quantity", "Quantity"))
    invoice_date: Optional[datetime] = Field(
        default=None,
        validation_alias=AliasChoices("invoice_date", "InvoiceDate"),
    )
    unit_price: float = Field(..., ge=0.0, validation_alias=AliasChoices("unit_price", "UnitPrice"))
    customer_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("customer_id", "CustomerID"),
    )
    country: Optional[str] = Field(
        default="Unknown",
        validation_alias=AliasChoices("country", "Country"),
    )


class TransactionUpdate(BaseModel):
    """Schema for updating an existing transaction (Full or Partial)."""
    quantity: Optional[float] = None
    unit_price: Optional[float] = Field(default=None, ge=0.0)
    description: Optional[str] = None
    country: Optional[str] = None
    customer_id: Optional[str] = None
    invoice_date: Optional[datetime] = None


class TransactionBatchCreate(BaseModel):
    transactions: List[TransactionCreate]


# -------------------- CORE / HEALTH --------------------
@app.get("/")
def home(db: Session = Depends(get_db)):
    summary = get_analytics_summary(db)
    return {
        "message": "Retail Intelligence & Sales Manager Dashboard API is running",
        "version": "2.0.0",
        "models_loaded": len(load_ml_models()),
        "stored_transactions": summary["total_transactions"],
        "authentication": "JWT Bearer RBAC (Admin, Sales Manager, Analyst)",
    }


@app.get("/health")
def health(db: Session = Depends(get_db)):
    summary = get_analytics_summary(db)
    return {
        "status": "healthy",
        "database": "connected (SQLite 3NF)",
        "models": "all 5 operational",
        "stored_transactions": summary["total_transactions"],
    }


# -------------------- AUTHENTICATION & STAKEHOLDER MANAGEMENT --------------------
@app.post("/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: UserRegister, db: Session = Depends(get_db)):
    """Register a new stakeholder user with designated role (Admin, Sales Manager, or Analyst)."""
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User with email '{payload.email}' already exists.",
        )

    new_user = User(
        name=payload.name.strip(),
        email=payload.email.strip().lower(),
        password_hash=hash_password(payload.password),
        role=payload.role,
        is_active=True,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token_data = {
        "sub": new_user.email,
        "user_id": new_user.id,
        "name": new_user.name,
        "role": new_user.role,
    }
    access_token = create_access_token(token_data)
    user_out = UserOut(
        id=new_user.id,
        name=new_user.name,
        email=new_user.email,
        role=new_user.role,
        is_active=new_user.is_active,
        created_at=new_user.created_at,
    )
    return TokenResponse(access_token=access_token, user=user_out)


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    """Authenticate with email and password to receive a JWT access token."""
    email_clean = payload.email.strip().lower()
    user = db.query(User).filter(User.email == email_clean).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive. Contact an Administrator.",
        )

    token_data = {
        "sub": user.email,
        "user_id": user.id,
        "name": user.name,
        "role": user.role,
    }
    access_token = create_access_token(token_data)
    user_out = UserOut(
        id=user.id,
        name=user.name,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
    )
    return TokenResponse(access_token=access_token, user=user_out)


@app.get("/auth/me", response_model=UserOut)
def get_current_user_profile(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve profile and role information for the currently authenticated user."""
    user = db.query(User).filter(User.id == current_user["user_id"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserOut(
        id=user.id,
        name=user.name,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
    )


@app.get("/auth/users", response_model=List[UserOut])
def list_stakeholders(
    admin_user: Dict[str, Any] = Depends(require_role(["Admin"])),
    db: Session = Depends(get_db),
):
    """Admin-only endpoint to view all registered stakeholders."""
    users = db.query(User).order_by(User.id).all()
    return [
        UserOut(
            id=u.id,
            name=u.name,
            email=u.email,
            role=u.role,
            is_active=u.is_active,
            created_at=u.created_at,
        )
        for u in users
    ]


# -------------------- REAL-TIME DATA INGESTION & CRUD --------------------
@app.post("/transactions", status_code=status.HTTP_201_CREATED)
def create_transaction(
    payload: TransactionCreate,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """Ingest a single transaction, compute derived metrics, run anomaly detection, and sync to 3NF schema."""
    if current_user and current_user.get("role") == "Analyst":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Analysts have read-only access and cannot create transactions.",
        )
    try:
        models = load_ml_models()
        record = ingest_transaction(
            db,
            invoice_no=payload.invoice_no,
            stock_code=payload.stock_code,
            quantity=payload.quantity,
            unit_price=payload.unit_price,
            description=payload.description,
            invoice_date=payload.invoice_date,
            customer_id=payload.customer_id,
            country=payload.country,
            models=models,
        )
        return {
            "status": "stored",
            "message": "Transaction persisted to normalized schema & denormalized view",
            "transaction": serialize_transaction(record),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/transactions/batch", status_code=status.HTTP_201_CREATED)
def create_transactions_batch(
    payload: TransactionBatchCreate,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """Ingest a batch of transactions into normalized tables and denormalized table."""
    if current_user and current_user.get("role") == "Analyst":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Analysts have read-only access and cannot ingest transactions.",
        )
    if not payload.transactions:
        raise HTTPException(status_code=400, detail="transactions list cannot be empty")
    try:
        models = load_ml_models()
        items = [t.model_dump() for t in payload.transactions]
        records = ingest_transactions_batch(db, items, models=models)
        anomalies = sum(1 for r in records if r.is_anomaly)
        return {
            "status": "stored",
            "inserted_count": len(records),
            "anomalies_detected": anomalies,
            "transactions": [serialize_transaction(r) for r in records],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/transactions/upload-csv", status_code=status.HTTP_201_CREATED)
async def upload_transactions_csv(
    request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """Admin-level batch CSV ingestion into normalized schema."""
    if current_user and current_user.get("role") not in ("Admin", "Sales Manager"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only Admins and Sales Managers can upload transaction CSV files.",
        )

    content_type = request.headers.get("content-type", "")
    filename = "uploaded.csv"
    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            upload = form.get("file")
            if upload is None:
                raise HTTPException(status_code=400, detail="Missing 'file' field in multipart upload")
            filename = getattr(upload, "filename", filename) or filename
            raw = await upload.read()
        else:
            raw = await request.body()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty CSV payload")
        items = parse_transactions_csv_bytes(raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV payload: {e}")

    models = load_ml_models()
    records = ingest_transactions_batch(db, items, models=models)
    return {
        "status": "stored",
        "filename": filename,
        "inserted_count": len(records),
        "anomalies_detected": sum(1 for r in records if r.is_anomaly),
    }


@app.get("/transactions")
def list_transactions(
    stock_code: Optional[str] = None,
    customer_id: Optional[str] = None,
    invoice_no: Optional[str] = None,
    country: Optional[str] = None,
    is_cancelled: Optional[bool] = None,
    is_anomaly: Optional[bool] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """Retrieve stored transactions with optional filtering and pagination."""
    total, rows = query_transactions(
        db,
        stock_code=stock_code,
        customer_id=customer_id,
        invoice_no=invoice_no,
        country=country,
        is_cancelled=is_cancelled,
        is_anomaly=is_anomaly,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "transactions": [serialize_transaction(r) for r in rows],
    }


@app.get("/transactions/{transaction_id}")
def get_transaction(transaction_id: int, db: Session = Depends(get_db)):
    """Retrieve a specific transaction by its database ID."""
    record = db.query(TransactionRecord).filter(TransactionRecord.id == transaction_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Transaction with id {transaction_id} not found")
    return serialize_transaction(record)


@app.put("/transactions/{transaction_id}")
@app.patch("/transactions/{transaction_id}")
def update_existing_transaction(
    transaction_id: int,
    payload: TransactionUpdate,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """Full/Partial update of a transaction: recalculates metrics, re-checks anomalies, and syncs normalized tables."""
    if current_user and current_user.get("role") not in ("Admin", "Sales Manager"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only Admins or Sales Managers can modify stored transactions.",
        )
    try:
        models = load_ml_models()
        updated = update_transaction(
            db,
            transaction_id=transaction_id,
            quantity=payload.quantity,
            unit_price=payload.unit_price,
            description=payload.description,
            country=payload.country,
            customer_id=payload.customer_id,
            invoice_date=payload.invoice_date,
            models=models,
        )
        return {
            "status": "updated",
            "message": f"Transaction {transaction_id} successfully updated and re-evaluated.",
            "transaction": serialize_transaction(updated),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/transactions/{transaction_id}")
def delete_transaction_endpoint(
    transaction_id: int,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """Delete a transaction from normalized line items and transactions table (Admin only)."""
    if current_user and current_user.get("role") != "Admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Transaction deletion is restricted to Administrators.",
        )
    success = delete_transaction(db, transaction_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Transaction with id {transaction_id} not found")
    return {
        "status": "deleted",
        "message": f"Transaction {transaction_id} successfully removed from database.",
        "id": transaction_id,
    }


# -------------------- LIVE DATABASE ANALYTICS & PREDICTIONS --------------------
@app.get("/analytics/summary")
def analytics_summary(db: Session = Depends(get_db)):
    """Get live KPIs computed from all stored transactions."""
    return get_analytics_summary(db)


@app.get("/analytics/products")
def analytics_products(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """Get aggregated product metrics from stored transactions."""
    df = get_product_features_from_db(db)
    if df.empty:
        return {"total": 0, "products": []}
    df = df.sort_values("TotalRows", ascending=False)
    total = len(df)
    sliced = df.iloc[offset: offset + limit].round(4)
    return {
        "total": total,
        "products": sliced.to_dict(orient="records"),
    }


@app.get("/analytics/products/{stock_code}/risk")
def live_product_risk(
    stock_code: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """Compute live product features from stored DB records and predict return risk."""
    models = load_ml_models()
    user_id = current_user.get("user_id") if current_user else None
    result = predict_product_risk_from_db(db, stock_code, models, user_id=user_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No transactions found for StockCode '{stock_code}'")
    return result


@app.get("/analytics/customers")
def analytics_customers(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """Get aggregated customer metrics from stored transactions."""
    df = get_customer_features_from_db(db)
    if df.empty:
        return {"total": 0, "customers": []}
    df = df.sort_values("TotalInvoices", ascending=False)
    total = len(df)
    sliced = df.iloc[offset: offset + limit].round(4)
    return {
        "total": total,
        "customers": sliced.to_dict(orient="records"),
    }


@app.get("/analytics/customers/{customer_id}/risk")
def live_customer_risk(
    customer_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """Compute live customer features from stored DB records and predict return risk."""
    models = load_ml_models()
    user_id = current_user.get("user_id") if current_user else None
    result = predict_customer_return_from_db(db, customer_id, models, user_id=user_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No transactions found for CustomerID '{customer_id}'")
    return result


@app.get("/analytics/daily")
def analytics_daily(db: Session = Depends(get_db)):
    """Get daily aggregated revenue, cancellations, and transaction counts from DB."""
    df = get_daily_series_from_db(db)
    if df.empty:
        return {"days": 0, "series": []}
    df["Date"] = df["Date"].astype(str)
    return {"days": len(df), "series": df.round(2).to_dict(orient="records")}


@app.get("/analytics/financial-impact")
def live_financial_impact(
    reduction_pct: float = Query(default=0.20, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
):
    """Calculate potential savings using live refund/credit volume from stored DB data."""
    summary = get_analytics_summary(db)
    total_credit = summary["total_credit_volume"]
    saving = round(total_credit * reduction_pct, 2)
    return {
        "total_credit_volume": total_credit,
        "reduction_percentage": round(reduction_pct * 100, 2),
        "potential_saving": saving,
        "based_on_transactions": summary["total_transactions"],
    }


@app.get("/predictions")
def list_predictions(
    prediction_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """Retrieve stored prediction audit logs with stakeholder attribution."""
    total, items = query_predictions(
        db,
        prediction_type=prediction_type,
        entity_id=entity_id,
        limit=limit,
        offset=offset,
    )
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "predictions": items,
    }


# -------------------- DIRECT ML ENDPOINTS (BACKWARDS COMPATIBLE) --------------------
@app.post("/predict/product-risk")
def product_risk(
    data: ProductInput,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    try:
        m1 = load_ml_models()["m1"]
        features = pd.DataFrame([data.model_dump()])
        prediction = m1["model"].predict(features[m1["features"]])
        risk = str(m1["label_encoder"].inverse_transform(prediction)[0])
        result = {"product_risk": risk}
        user_id = current_user.get("user_id") if current_user else None
        log_prediction(
            db,
            prediction_type="product_risk",
            entity_id=None,
            input_data=data.model_dump(),
            result_data=result,
            user_id=user_id,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/customer-return")
def customer_return(
    data: CustomerInput,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    try:
        m2 = load_ml_models()["m2"]
        features = pd.DataFrame([data.model_dump()])
        probability = float(m2["model"].predict_proba(features[m2["features"]])[0][1])
        threshold = float(m2["threshold"])
        result = {
            "high_risk_probability": round(probability, 4),
            "prediction": int(probability > threshold),
            "risk": "High" if probability > threshold else "Low",
        }
        user_id = current_user.get("user_id") if current_user else None
        log_prediction(
            db,
            prediction_type="customer_return",
            entity_id=None,
            input_data=data.model_dump(),
            result_data=result,
            user_id=user_id,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/forecast/sales")
def sales_forecast(days: int = 28):
    if days < 1 or days > 365:
        raise HTTPException(status_code=400, detail="Days must be 1 to 365")
    try:
        m3 = load_ml_models()["m3"]
        future = m3.make_future_dataframe(periods=days)
        forecast = m3.predict(future)
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days).copy()
        result["ds"] = result["ds"].astype(str)
        return result.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/forecast/cancellations")
def cancellation_forecast(days: int = 28):
    if days < 1 or days > 365:
        raise HTTPException(status_code=400, detail="Days must be 1 to 365")
    try:
        m4 = load_ml_models()["m4"]
        future = m4.make_future_dataframe(periods=days)
        forecast = m4.predict(future)
        result = forecast[["ds", "yhat"]].tail(days).copy()
        result["predicted_cancellations"] = np.clip(np.expm1(result["yhat"]), 0, None).round(2)
        result["ds"] = result["ds"].astype(str)
        return result[["ds", "predicted_cancellations"]].to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/anomaly")
def detect_anomaly(
    data: TransactionInput,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    try:
        m6 = load_ml_models()["m6"]
        features = pd.DataFrame([data.model_dump()])
        scaled_data = m6["scaler"].transform(features[m6["features"]])
        prediction = int(m6["model"].predict(scaled_data)[0])
        score = float(m6["model"].score_samples(scaled_data)[0])
        result = {
            "anomaly": prediction == -1,
            "result": "Unusual Transaction" if prediction == -1 else "Normal Transaction",
            "anomaly_score": round(score, 4),
        }
        user_id = current_user.get("user_id") if current_user else None
        log_prediction(
            db,
            prediction_type="anomaly",
            entity_id=None,
            input_data=data.model_dump(),
            result_data=result,
            user_id=user_id,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/financial-impact")
def financial_impact(
    data: FinancialInput,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    if data.total_credit < 0 or not 0 <= data.reduction_pct <= 1:
        raise HTTPException(status_code=400, detail="Invalid input values")
    saving = data.total_credit * data.reduction_pct
    result = {
        "total_credit": data.total_credit,
        "reduction_percentage": data.reduction_pct * 100,
        "potential_saving": round(saving, 2),
    }
    user_id = current_user.get("user_id") if current_user else None
    log_prediction(
        db,
        prediction_type="financial_impact",
        entity_id=None,
        input_data=data.model_dump(),
        result_data=result,
        user_id=user_id,
    )
    return result
