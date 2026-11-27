import io
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from database import (
    Customer,
    Invoice,
    PredictionLog,
    Product,
    TransactionItem,
    TransactionRecord,
    User,
    utc_now_naive,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

_MODEL_CACHE: Optional[Dict[str, Any]] = None


def _find_model_file(filename: str) -> str:
    """Find a model file either in models/ subdirectory or root BASE_DIR."""
    sub_path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(sub_path):
        return sub_path
    root_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(root_path):
        return root_path
    raise FileNotFoundError(f"Model file '{filename}' not found in '{MODEL_DIR}' or '{BASE_DIR}'.")


def load_ml_models() -> Dict[str, Any]:
    """Load and cache all 5 trained ML models with robust path resolution."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = {
            "m1": joblib.load(_find_model_file("feature1_product_risk.pkl")),
            "m2": joblib.load(_find_model_file("feature2_customer_return.pkl")),
            "m3": joblib.load(_find_model_file("feature3_sales_forecast.pkl")),
            "m4": joblib.load(_find_model_file("feature4_cancellation_forecast.pkl")),
            "m6": joblib.load(_find_model_file("feature6_anomaly.pkl")),
        }
    return _MODEL_CACHE


def normalize_timestamp(dt: Optional[datetime]) -> datetime:
    """Convert optional aware/naive datetime to naive UTC datetime."""
    if dt is None:
        return utc_now_naive()
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def build_transaction_record(
    *,
    invoice_no: str,
    stock_code: str,
    quantity: float,
    unit_price: float,
    description: Optional[str] = None,
    invoice_date: Optional[datetime] = None,
    customer_id: Optional[str] = None,
    country: Optional[str] = None,
    models: Optional[Dict[str, Any]] = None,
) -> TransactionRecord:
    """Compute derived features and run real-time anomaly detection on a transaction."""
    clean_invoice = str(invoice_no).strip()
    clean_stock = str(stock_code).strip()
    clean_desc = str(description).strip() if description and str(description).strip() else "Unknown"
    clean_country = str(country).strip() if country and str(country).strip() else "Unknown"
    clean_customer = (
        str(customer_id).strip()
        if customer_id is not None and str(customer_id).strip() not in ("", "nan", "None")
        else None
    )
    dt = normalize_timestamp(invoice_date)

    qty = float(quantity)
    price = float(unit_price)
    total_amount = round(qty * price, 4)

    is_cancelled = clean_invoice.upper().startswith("C")
    is_negative_qty = qty < 0
    is_return_like = is_negative_qty and not is_cancelled
    purchase_amount = round(total_amount, 4) if (qty > 0 and price > 0) else 0.0
    credit_amount = round(abs(total_amount), 4) if is_negative_qty else 0.0
    hour = int(dt.hour)

    is_anomaly = False
    anomaly_score = None
    if models and "m6" in models:
        m6 = models["m6"]
        feat_df = pd.DataFrame([{
            "Quantity": qty,
            "UnitPrice": price,
            "TotalAmount": total_amount,
            "Hour": hour,
        }])
        scaled = m6["scaler"].transform(feat_df[m6["features"]])
        pred = int(m6["model"].predict(scaled)[0])
        is_anomaly = pred == -1
        anomaly_score = round(float(m6["model"].score_samples(scaled)[0]), 4)

    return TransactionRecord(
        invoice_no=clean_invoice,
        stock_code=clean_stock,
        description=clean_desc,
        quantity=qty,
        invoice_date=dt,
        unit_price=price,
        customer_id=clean_customer,
        country=clean_country,
        is_cancelled=is_cancelled,
        is_negative_quantity=is_negative_qty,
        is_return_like=is_return_like,
        total_amount=total_amount,
        purchase_amount=purchase_amount,
        credit_amount=credit_amount,
        hour=hour,
        is_anomaly=is_anomaly,
        anomaly_score=anomaly_score,
    )


def score_anomaly_batch(df_features: pd.DataFrame, m6: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized anomaly detection for fast batch ingestion."""
    scaled = m6["scaler"].transform(df_features[m6["features"]])
    preds = m6["model"].predict(scaled)
    scores = m6["model"].score_samples(scaled)
    return preds == -1, np.round(scores, 4)


def sync_single_to_normalized_tables(db: Session, record: TransactionRecord) -> None:
    """Ensure customer, product, invoice, and transaction_item exist in normalized schema."""
    # 1. Customer
    if record.customer_id:
        cust = db.query(Customer).filter(Customer.customer_id == record.customer_id).first()
        if not cust:
            cust = Customer(customer_id=record.customer_id, country=record.country)
            db.add(cust)
            db.flush()

    # 2. Product
    prod = db.query(Product).filter(Product.stock_code == record.stock_code).first()
    if not prod:
        prod = Product(stock_code=record.stock_code, description=record.description)
        db.add(prod)
        db.flush()
    elif record.description and prod.description == "Unknown":
        prod.description = record.description

    # 3. Invoice
    inv = db.query(Invoice).filter(Invoice.invoice_no == record.invoice_no).first()
    if not inv:
        inv = Invoice(
            invoice_no=record.invoice_no,
            customer_id=record.customer_id,
            invoice_date=record.invoice_date,
            country=record.country,
            is_cancelled=record.is_cancelled,
        )
        db.add(inv)
        db.flush()

    # 4. Transaction Item
    item = TransactionItem(
        invoice_no=record.invoice_no,
        stock_code=record.stock_code,
        quantity=record.quantity,
        unit_price=record.unit_price,
        total_amount=record.total_amount,
        purchase_amount=record.purchase_amount,
        credit_amount=record.credit_amount,
        hour=record.hour,
        is_negative_quantity=record.is_negative_quantity,
        is_return_like=record.is_return_like,
        is_anomaly=record.is_anomaly,
        anomaly_score=record.anomaly_score,
        created_at=record.created_at or utc_now_naive(),
    )
    db.add(item)
    db.flush()


def ingest_transaction(
    db: Session,
    *,
    invoice_no: str,
    stock_code: str,
    quantity: float,
    unit_price: float,
    description: Optional[str] = None,
    invoice_date: Optional[datetime] = None,
    customer_id: Optional[str] = None,
    country: Optional[str] = None,
    models: Optional[Dict[str, Any]] = None,
) -> TransactionRecord:
    """Persist a single real-time transaction into normalized tables and denormalized table."""
    record = build_transaction_record(
        invoice_no=invoice_no,
        stock_code=stock_code,
        quantity=quantity,
        unit_price=unit_price,
        description=description,
        invoice_date=invoice_date,
        customer_id=customer_id,
        country=country,
        models=models,
    )
    db.add(record)
    sync_single_to_normalized_tables(db, record)
    db.commit()
    db.refresh(record)
    return record


def parse_transactions_csv_bytes(raw: bytes) -> List[Dict[str, Any]]:
    """Parse raw CSV bytes into normalized transaction dicts ready for batch ingestion."""
    df = pd.read_csv(io.BytesIO(raw))
    col_map = {c.lower(): c for c in df.columns}
    required = ["invoiceno", "stockcode", "quantity", "unitprice"]
    missing = [r for r in required if r not in col_map]
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}. Expected InvoiceNo, StockCode, Quantity, UnitPrice."
        )

    items: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        inv_date = None
        if "invoicedate" in col_map and pd.notna(row[col_map["invoicedate"]]):
            inv_date = pd.to_datetime(row[col_map["invoicedate"]]).to_pydatetime()

        cust_id = None
        if "customerid" in col_map and pd.notna(row[col_map["customerid"]]):
            val = row[col_map["customerid"]]
            cust_id = str(int(val)) if isinstance(val, (int, float)) and float(val).is_integer() else str(val)

        desc = (
            str(row[col_map["description"]])
            if "description" in col_map and pd.notna(row[col_map["description"]])
            else "Unknown"
        )
        country = (
            str(row[col_map["country"]])
            if "country" in col_map and pd.notna(row[col_map["country"]])
            else "Unknown"
        )

        items.append({
            "invoice_no": str(row[col_map["invoiceno"]]),
            "stock_code": str(row[col_map["stockcode"]]),
            "quantity": float(row[col_map["quantity"]]),
            "unit_price": float(row[col_map["unitprice"]]),
            "description": desc,
            "invoice_date": inv_date,
            "customer_id": cust_id,
            "country": country,
        })
    return items


def ingest_transactions_batch(
    db: Session,
    items: List[Dict[str, Any]],
    models: Optional[Dict[str, Any]] = None,
) -> List[TransactionRecord]:
    """Persist a batch of transactions in one database transaction across all tables."""
    if not items:
        return []

    records: List[TransactionRecord] = []
    for item in items:
        records.append(
            build_transaction_record(
                invoice_no=item["invoice_no"],
                stock_code=item["stock_code"],
                quantity=item["quantity"],
                unit_price=item["unit_price"],
                description=item.get("description"),
                invoice_date=item.get("invoice_date"),
                customer_id=item.get("customer_id"),
                country=item.get("country"),
                models=None,
            )
        )

    if models and "m6" in models:
        feat_df = pd.DataFrame([
            {
                "Quantity": r.quantity,
                "UnitPrice": r.unit_price,
                "TotalAmount": r.total_amount,
                "Hour": r.hour,
            }
            for r in records
        ])
        flags, scores = score_anomaly_batch(feat_df, models["m6"])
        for rec, flag, score in zip(records, flags, scores):
            rec.is_anomaly = bool(flag)
            rec.anomaly_score = float(score)

    db.add_all(records)
    for rec in records:
        sync_single_to_normalized_tables(db, rec)

    db.commit()
    for rec in records:
        db.refresh(rec)
    return records


def update_transaction(
    db: Session,
    transaction_id: int,
    *,
    quantity: Optional[float] = None,
    unit_price: Optional[float] = None,
    description: Optional[str] = None,
    country: Optional[str] = None,
    customer_id: Optional[str] = None,
    invoice_date: Optional[datetime] = None,
    models: Optional[Dict[str, Any]] = None,
) -> TransactionRecord:
    """Update an existing transaction record, recompute derived values and anomaly flags, and sync."""
    record = db.query(TransactionRecord).filter(TransactionRecord.id == transaction_id).first()
    if not record:
        raise ValueError(f"Transaction with id {transaction_id} not found")

    old_invoice_no = record.invoice_no
    old_stock_code = record.stock_code

    if quantity is not None:
        record.quantity = float(quantity)
    if unit_price is not None:
        record.unit_price = float(unit_price)
    if description is not None:
        record.description = str(description).strip()
    if country is not None:
        record.country = str(country).strip()
    if customer_id is not None:
        clean_cust = str(customer_id).strip()
        record.customer_id = clean_cust if clean_cust not in ("", "nan", "None") else None
    if invoice_date is not None:
        record.invoice_date = normalize_timestamp(invoice_date)

    # Recompute derived metrics
    total_amount = round(record.quantity * record.unit_price, 4)
    record.total_amount = total_amount
    record.is_negative_quantity = record.quantity < 0
    record.is_cancelled = record.invoice_no.upper().startswith("C")
    record.is_return_like = record.is_negative_quantity and not record.is_cancelled
    record.purchase_amount = round(total_amount, 4) if (record.quantity > 0 and record.unit_price > 0) else 0.0
    record.credit_amount = round(abs(total_amount), 4) if record.is_negative_quantity else 0.0
    record.hour = int(record.invoice_date.hour) if record.invoice_date else 0

    # Re-evaluate anomaly detection
    if models and "m6" in models:
        m6 = models["m6"]
        feat_df = pd.DataFrame([{
            "Quantity": record.quantity,
            "UnitPrice": record.unit_price,
            "TotalAmount": total_amount,
            "Hour": record.hour,
        }])
        scaled = m6["scaler"].transform(feat_df[m6["features"]])
        pred = int(m6["model"].predict(scaled)[0])
        record.is_anomaly = pred == -1
        record.anomaly_score = round(float(m6["model"].score_samples(scaled)[0]), 4)

    # Sync to normalized TransactionItem
    item = (
        db.query(TransactionItem)
        .filter(
            TransactionItem.invoice_no == old_invoice_no,
            TransactionItem.stock_code == old_stock_code,
        )
        .first()
    )
    if item:
        item.quantity = record.quantity
        item.unit_price = record.unit_price
        item.total_amount = record.total_amount
        item.purchase_amount = record.purchase_amount
        item.credit_amount = record.credit_amount
        item.hour = record.hour
        item.is_negative_quantity = record.is_negative_quantity
        item.is_return_like = record.is_return_like
        item.is_anomaly = record.is_anomaly
        item.anomaly_score = record.anomaly_score

    db.commit()
    db.refresh(record)
    return record


def delete_transaction(db: Session, transaction_id: int) -> bool:
    """Delete a transaction from database and cleanup line item."""
    record = db.query(TransactionRecord).filter(TransactionRecord.id == transaction_id).first()
    if not record:
        return False

    # Also remove from transaction_items if exists
    item = (
        db.query(TransactionItem)
        .filter(
            TransactionItem.invoice_no == record.invoice_no,
            TransactionItem.stock_code == record.stock_code,
        )
        .first()
    )
    if item:
        db.delete(item)

    db.delete(record)
    db.commit()
    return True


def serialize_transaction(r: TransactionRecord) -> Dict[str, Any]:
    """Convert TransactionRecord ORM row to JSON-serializable dict."""
    return {
        "id": r.id,
        "invoice_no": r.invoice_no,
        "stock_code": r.stock_code,
        "description": r.description,
        "quantity": r.quantity,
        "invoice_date": r.invoice_date.isoformat() if r.invoice_date else None,
        "unit_price": r.unit_price,
        "customer_id": r.customer_id,
        "country": r.country,
        "is_cancelled": r.is_cancelled,
        "is_negative_quantity": r.is_negative_quantity,
        "is_return_like": r.is_return_like,
        "total_amount": r.total_amount,
        "purchase_amount": r.purchase_amount,
        "credit_amount": r.credit_amount,
        "hour": r.hour,
        "is_anomaly": r.is_anomaly,
        "anomaly_score": r.anomaly_score,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


def query_transactions(
    db: Session,
    *,
    stock_code: Optional[str] = None,
    customer_id: Optional[str] = None,
    invoice_no: Optional[str] = None,
    country: Optional[str] = None,
    is_cancelled: Optional[bool] = None,
    is_anomaly: Optional[bool] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100,
    offset: int = 0,
) -> Tuple[int, List[TransactionRecord]]:
    """Filter and paginate stored transactions."""
    q = db.query(TransactionRecord)
    if stock_code:
        q = q.filter(TransactionRecord.stock_code == stock_code.strip())
    if customer_id:
        q = q.filter(TransactionRecord.customer_id == customer_id.strip())
    if invoice_no:
        q = q.filter(TransactionRecord.invoice_no == invoice_no.strip())
    if country:
        q = q.filter(TransactionRecord.country == country.strip())
    if is_cancelled is not None:
        q = q.filter(TransactionRecord.is_cancelled == is_cancelled)
    if is_anomaly is not None:
        q = q.filter(TransactionRecord.is_anomaly == is_anomaly)
    if start_date is not None:
        q = q.filter(TransactionRecord.invoice_date >= normalize_timestamp(start_date))
    if end_date is not None:
        q = q.filter(TransactionRecord.invoice_date <= normalize_timestamp(end_date))

    total = q.count()
    rows = (
        q.order_by(desc(TransactionRecord.invoice_date), desc(TransactionRecord.id))
        .offset(offset)
        .limit(limit)
        .all()
    )
    return total, rows


def get_analytics_summary(db: Session) -> Dict[str, Any]:
    """Compute live summary KPIs across all stored transactions."""
    total_tx = db.query(func.count(TransactionRecord.id)).scalar() or 0
    if total_tx == 0:
        return {
            "total_transactions": 0,
            "total_purchase_revenue": 0.0,
            "total_credit_volume": 0.0,
            "net_revenue": 0.0,
            "cancelled_transactions": 0,
            "cancellation_rate": 0.0,
            "anomalous_transactions": 0,
            "unique_products": 0,
            "unique_customers": 0,
            "earliest_date": None,
            "latest_date": None,
        }

    agg = db.query(
        func.coalesce(func.sum(TransactionRecord.purchase_amount), 0.0),
        func.coalesce(func.sum(TransactionRecord.credit_amount), 0.0),
        func.coalesce(func.sum(TransactionRecord.total_amount), 0.0),
        func.count(func.distinct(TransactionRecord.stock_code)),
        func.count(func.distinct(TransactionRecord.customer_id)),
        func.min(TransactionRecord.invoice_date),
        func.max(TransactionRecord.invoice_date),
    ).one()

    cancelled_count = (
        db.query(func.count(TransactionRecord.id))
        .filter(TransactionRecord.is_cancelled.is_(True))
        .scalar()
        or 0
    )
    anomaly_count = (
        db.query(func.count(TransactionRecord.id))
        .filter(TransactionRecord.is_anomaly.is_(True))
        .scalar()
        or 0
    )

    return {
        "total_transactions": int(total_tx),
        "total_purchase_revenue": round(float(agg[0]), 2),
        "total_credit_volume": round(float(agg[1]), 2),
        "net_revenue": round(float(agg[2]), 2),
        "cancelled_transactions": int(cancelled_count),
        "cancellation_rate": round(cancelled_count / total_tx, 4),
        "anomalous_transactions": int(anomaly_count),
        "unique_products": int(agg[3]),
        "unique_customers": int(agg[4]),
        "earliest_date": agg[5].isoformat() if agg[5] else None,
        "latest_date": agg[6].isoformat() if agg[6] else None,
    }


def _load_transactions_df(
    db: Session,
    *,
    stock_code: Optional[str] = None,
    customer_id: Optional[str] = None,
) -> pd.DataFrame:
    """Load relevant columns from transactions table into a DataFrame."""
    q = db.query(
        TransactionRecord.invoice_no,
        TransactionRecord.stock_code,
        TransactionRecord.description,
        TransactionRecord.quantity,
        TransactionRecord.invoice_date,
        TransactionRecord.unit_price,
        TransactionRecord.customer_id,
        TransactionRecord.country,
        TransactionRecord.is_cancelled,
        TransactionRecord.is_negative_quantity,
        TransactionRecord.total_amount,
        TransactionRecord.purchase_amount,
        TransactionRecord.credit_amount,
    )
    if stock_code:
        q = q.filter(TransactionRecord.stock_code == stock_code.strip())
    if customer_id:
        q = q.filter(TransactionRecord.customer_id == customer_id.strip())

    rows = q.all()
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        rows,
        columns=[
            "InvoiceNo",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "UnitPrice",
            "CustomerID",
            "Country",
            "IsCancelled",
            "IsNegativeQuantity",
            "TotalAmount",
            "PurchaseAmount",
            "CreditAmount",
        ],
    )


def get_product_features_from_db(
    db: Session,
    stock_code: Optional[str] = None,
) -> pd.DataFrame:
    """Compute product-level ML features (matching Feature 1 training pipeline) from DB."""
    df = _load_transactions_df(db, stock_code=stock_code)
    if df.empty:
        return pd.DataFrame()

    g = df.groupby("StockCode").agg(
        Description=("Description", "first"),
        TotalInvoices=("InvoiceNo", "nunique"),
        TotalRows=("InvoiceNo", "size"),
        TotalPurchaseAmount=("TotalAmount", "sum"),
        AvgUnitPrice=("UnitPrice", "mean"),
        NegativeQuantityRows=("IsNegativeQuantity", "sum"),
        CancelRows=("IsCancelled", "sum"),
    ).reset_index()

    g["CancellationRowRate"] = g["CancelRows"] / g["TotalRows"].clip(lower=1)
    cancel_totals = (
        df[df["IsCancelled"]]
        .groupby("StockCode")["TotalAmount"]
        .sum()
        .reindex(g["StockCode"])
        .fillna(0)
        .abs()
        .values
    )
    denom = g["TotalPurchaseAmount"].abs().clip(lower=1)
    g["CreditToPurchaseRatio"] = cancel_totals / denom

    cols = [
        "StockCode",
        "Description",
        "CancellationRowRate",
        "CreditToPurchaseRatio",
        "AvgUnitPrice",
        "TotalInvoices",
        "TotalPurchaseAmount",
        "NegativeQuantityRows",
        "TotalRows",
    ]
    return g[cols]


def predict_product_risk_from_db(
    db: Session,
    stock_code: str,
    models: Dict[str, Any],
    user_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Pull product metrics from DB, run Feature 1 Random Forest model, and audit log."""
    feats_df = get_product_features_from_db(db, stock_code=stock_code)
    if feats_df.empty:
        return None

    row = feats_df.iloc[0]
    m1 = models["m1"]
    feat_names = m1["features"]
    input_df = pd.DataFrame([row[feat_names].to_dict()])
    pred = m1["model"].predict(input_df[feat_names])
    risk = str(m1["label_encoder"].inverse_transform(pred)[0])

    features_dict = row[feat_names].round(4).to_dict()
    result = {
        "stock_code": str(row["StockCode"]),
        "description": str(row["Description"]),
        "features": features_dict,
        "product_risk": risk,
    }
    log_prediction(
        db,
        prediction_type="product_risk",
        entity_id=str(row["StockCode"]),
        input_data=features_dict,
        result_data={"product_risk": risk},
        user_id=user_id,
    )
    return result


def get_customer_features_from_db(
    db: Session,
    customer_id: Optional[str] = None,
) -> pd.DataFrame:
    """Compute customer-level ML features (matching Feature 2 training pipeline) from DB."""
    df = _load_transactions_df(db, customer_id=customer_id)
    if df.empty:
        return pd.DataFrame()

    df = df[df["CustomerID"].notna() & (df["CustomerID"] != "")]
    if df.empty:
        return pd.DataFrame()

    cg = df.groupby("CustomerID").agg(
        TotalInvoices=("InvoiceNo", "nunique"),
        TotalPurchaseAmount=("PurchaseAmount", "sum"),
    ).reset_index()

    canc_inv = (
        df[df["IsCancelled"]]
        .groupby("CustomerID")["InvoiceNo"]
        .nunique()
        .reindex(cg["CustomerID"])
        .fillna(0)
        .values
    )
    cg["CancellationInvoices"] = canc_inv.astype(int)
    cg["CancellationRate"] = cg["CancellationInvoices"] / cg["TotalInvoices"].clip(lower=1)
    cg["AvgPurchaseValue"] = cg["TotalPurchaseAmount"] / cg["TotalInvoices"].clip(lower=1)

    cols = [
        "CustomerID",
        "TotalInvoices",
        "CancellationInvoices",
        "CancellationRate",
        "TotalPurchaseAmount",
        "AvgPurchaseValue",
    ]
    return cg[cols]


def predict_customer_return_from_db(
    db: Session,
    customer_id: str,
    models: Dict[str, Any],
    user_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Pull customer metrics from DB, run Feature 2 Logistic Regression pipeline, and audit log."""
    cust_df = get_customer_features_from_db(db, customer_id=customer_id)
    if cust_df.empty:
        return None

    row = cust_df.iloc[0]
    m2 = models["m2"]
    feat_names = m2["features"]
    input_df = pd.DataFrame([row[feat_names].to_dict()])
    prob = float(m2["model"].predict_proba(input_df[feat_names])[0][1])
    threshold = float(m2["threshold"])
    is_high = prob > threshold

    features_dict = row[feat_names].round(4).to_dict()
    result = {
        "customer_id": str(row["CustomerID"]),
        "features": features_dict,
        "high_risk_probability": round(prob, 4),
        "prediction": int(is_high),
        "risk": "High" if is_high else "Low",
    }
    log_prediction(
        db,
        prediction_type="customer_return",
        entity_id=str(row["CustomerID"]),
        input_data=features_dict,
        result_data={
            "high_risk_probability": result["high_risk_probability"],
            "risk": result["risk"],
        },
        user_id=user_id,
    )
    return result


def get_daily_series_from_db(db: Session) -> pd.DataFrame:
    """Aggregate stored transactions into daily revenue and cancellation counts."""
    df = _load_transactions_df(db)
    if df.empty:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["InvoiceDate"]).dt.date
    daily_rev = (
        df[df["Quantity"] > 0]
        .groupby("Date")["TotalAmount"]
        .sum()
        .rename("Revenue")
    )
    daily_canc = (
        df[df["IsCancelled"]]
        .groupby("Date")
        .size()
        .rename("Cancellations")
    )
    daily_tx = df.groupby("Date").size().rename("Transactions")

    combined = pd.concat([daily_rev, daily_canc, daily_tx], axis=1).fillna(0).reset_index()
    combined = combined.sort_values("Date")
    return combined


def log_prediction(
    db: Session,
    *,
    prediction_type: str,
    entity_id: Optional[str],
    input_data: Dict[str, Any],
    result_data: Dict[str, Any],
    user_id: Optional[int] = None,
) -> PredictionLog:
    """Save an ML prediction evaluation to prediction_logs with optional user_id attribution."""
    entry = PredictionLog(
        prediction_type=prediction_type,
        entity_id=entity_id,
        input_payload=json.dumps(input_data, default=str),
        result_payload=json.dumps(result_data, default=str),
        user_id=user_id,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def query_predictions(
    db: Session,
    *,
    prediction_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Retrieve stored prediction history with user attribution."""
    q = db.query(PredictionLog)
    if prediction_type:
        q = q.filter(PredictionLog.prediction_type == prediction_type.strip())
    if entity_id:
        q = q.filter(PredictionLog.entity_id == entity_id.strip())

    total = q.count()
    rows = (
        q.order_by(desc(PredictionLog.created_at), desc(PredictionLog.id))
        .offset(offset)
        .limit(limit)
        .all()
    )
    items = []
    for r in rows:
        user_email = r.user.email if r.user else "Anonymous / System"
        user_role = r.user.role if r.user else "System"
        items.append({
            "id": r.id,
            "prediction_type": r.prediction_type,
            "entity_id": r.entity_id,
            "input_payload": json.loads(r.input_payload),
            "result_payload": json.loads(r.result_payload),
            "user_id": r.user_id,
            "user_email": user_email,
            "user_role": user_role,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })
    return total, items
