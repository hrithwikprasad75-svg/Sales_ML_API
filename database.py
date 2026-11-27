import os
from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "sales_data.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

is_sqlite = DATABASE_URL.startswith("sqlite")
connect_args = {"check_same_thread": False} if is_sqlite else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
)

if is_sqlite:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def utc_now_naive() -> datetime:
    """Return current UTC timestamp without tzinfo for consistent DB storage."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# -------------------- AUTHENTICATION & USERS --------------------
class User(Base):
    """User account model with role-based access control (RBAC)."""

    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("role IN ('Admin', 'Sales Manager', 'Analyst')", name="check_user_role"),
    )

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(120), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(32), nullable=False, default="Sales Manager", index=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=utc_now_naive, index=True)

    predictions = relationship("PredictionLog", back_populates="user")


# -------------------- NORMALIZED RELATIONAL SCHEMA --------------------
class Customer(Base):
    """Normalized customer entity."""

    __tablename__ = "customers"

    customer_id = Column(String(64), primary_key=True, index=True)
    country = Column(String(128), nullable=False, default="Unknown", index=True)
    created_at = Column(DateTime, nullable=False, default=utc_now_naive)

    invoices = relationship("Invoice", back_populates="customer")


class Product(Base):
    """Normalized product catalog entity."""

    __tablename__ = "products"

    stock_code = Column(String(64), primary_key=True, index=True)
    description = Column(String(255), nullable=False, default="Unknown")
    created_at = Column(DateTime, nullable=False, default=utc_now_naive)

    items = relationship("TransactionItem", back_populates="product")


class Invoice(Base):
    """Normalized sales invoice entity."""

    __tablename__ = "invoices"

    invoice_no = Column(String(64), primary_key=True, index=True)
    customer_id = Column(
        String(64),
        ForeignKey("customers.customer_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    invoice_date = Column(DateTime, nullable=False, index=True)
    country = Column(String(128), nullable=False, default="Unknown", index=True)
    is_cancelled = Column(Boolean, nullable=False, default=False, index=True)
    created_at = Column(DateTime, nullable=False, default=utc_now_naive)

    customer = relationship("Customer", back_populates="invoices")
    items = relationship("TransactionItem", back_populates="invoice", cascade="all, delete-orphan")


class TransactionItem(Base):
    """Normalized transaction line item linked to invoice and product."""

    __tablename__ = "transaction_items"
    __table_args__ = (
        CheckConstraint("unit_price >= 0", name="check_positive_unit_price"),
    )

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    invoice_no = Column(
        String(64),
        ForeignKey("invoices.invoice_no", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stock_code = Column(
        String(64),
        ForeignKey("products.stock_code", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    quantity = Column(Float, nullable=False)
    unit_price = Column(Float, nullable=False)
    total_amount = Column(Float, nullable=False, default=0.0)
    purchase_amount = Column(Float, nullable=False, default=0.0)
    credit_amount = Column(Float, nullable=False, default=0.0)
    hour = Column(Integer, nullable=False, default=0)

    is_negative_quantity = Column(Boolean, nullable=False, default=False)
    is_return_like = Column(Boolean, nullable=False, default=False)
    is_anomaly = Column(Boolean, nullable=False, default=False, index=True)
    anomaly_score = Column(Float, nullable=True)

    created_at = Column(DateTime, nullable=False, default=utc_now_naive, index=True)

    invoice = relationship("Invoice", back_populates="items")
    product = relationship("Product", back_populates="items")


# -------------------- COMPATIBILITY & ML AUDIT TABLES --------------------
class TransactionRecord(Base):
    """Denormalized retail transaction view/table for fast vectorized ML and backwards compatibility."""

    __tablename__ = "transactions"
    __table_args__ = (
        CheckConstraint("unit_price >= 0", name="check_tx_unit_price"),
    )

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    invoice_no = Column(String(64), nullable=False, index=True)
    stock_code = Column(String(64), nullable=False, index=True)
    description = Column(String(255), nullable=False, default="Unknown")
    quantity = Column(Float, nullable=False)
    invoice_date = Column(DateTime, nullable=False, index=True)
    unit_price = Column(Float, nullable=False)
    customer_id = Column(String(64), nullable=True, index=True)
    country = Column(String(128), nullable=False, default="Unknown", index=True)

    # Derived fields matching training pipeline
    is_cancelled = Column(Boolean, nullable=False, default=False, index=True)
    is_negative_quantity = Column(Boolean, nullable=False, default=False)
    is_return_like = Column(Boolean, nullable=False, default=False)
    total_amount = Column(Float, nullable=False, default=0.0)
    purchase_amount = Column(Float, nullable=False, default=0.0)
    credit_amount = Column(Float, nullable=False, default=0.0)
    hour = Column(Integer, nullable=False, default=0)

    # Real-time anomaly detection enrichment
    is_anomaly = Column(Boolean, nullable=False, default=False, index=True)
    anomaly_score = Column(Float, nullable=True)

    created_at = Column(DateTime, nullable=False, default=utc_now_naive, index=True)


class PredictionLog(Base):
    """Audit log of ML predictions and risk evaluations linked to the authenticated user."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    prediction_type = Column(String(64), nullable=False, index=True)
    entity_id = Column(String(128), nullable=True, index=True)
    input_payload = Column(Text, nullable=False)
    result_payload = Column(Text, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at = Column(DateTime, nullable=False, default=utc_now_naive, index=True)

    user = relationship("User", back_populates="predictions")


# -------------------- INITIALIZATION & SEEDING --------------------
def seed_default_users(db) -> None:
    """Seed initial stakeholder accounts if no users exist."""
    from auth import hash_password

    if db.query(User).first() is None:
        default_users = [
            User(
                name="System Administrator",
                email="admin@retail.com",
                password_hash=hash_password("Admin@123"),
                role="Admin",
                is_active=True,
            ),
            User(
                name="Senior Sales Manager",
                email="manager@retail.com",
                password_hash=hash_password("Manager@123"),
                role="Sales Manager",
                is_active=True,
            ),
            User(
                name="Data Analyst",
                email="analyst@retail.com",
                password_hash=hash_password("Analyst@123"),
                role="Analyst",
                is_active=True,
            ),
        ]
        db.add_all(default_users)
        db.commit()


def sync_normalized_from_transactions(db) -> None:
    """Populate normalized tables from existing denormalized transactions if empty."""
    if db.query(Invoice).first() is not None:
        return

    tx_rows = db.query(TransactionRecord).all()
    if not tx_rows:
        return

    customers_map = {}
    products_map = {}
    invoices_map = {}

    for tx in tx_rows:
        # Customer
        if tx.customer_id and tx.customer_id not in customers_map:
            cust = db.query(Customer).filter(Customer.customer_id == tx.customer_id).first()
            if not cust:
                cust = Customer(customer_id=tx.customer_id, country=tx.country)
                db.add(cust)
            customers_map[tx.customer_id] = cust

        # Product
        if tx.stock_code not in products_map:
            prod = db.query(Product).filter(Product.stock_code == tx.stock_code).first()
            if not prod:
                prod = Product(stock_code=tx.stock_code, description=tx.description)
                db.add(prod)
            products_map[tx.stock_code] = prod

        # Invoice
        if tx.invoice_no not in invoices_map:
            inv = db.query(Invoice).filter(Invoice.invoice_no == tx.invoice_no).first()
            if not inv:
                inv = Invoice(
                    invoice_no=tx.invoice_no,
                    customer_id=tx.customer_id,
                    invoice_date=tx.invoice_date,
                    country=tx.country,
                    is_cancelled=tx.is_cancelled,
                )
                db.add(inv)
            invoices_map[tx.invoice_no] = inv

    db.commit()

    # Add items
    for tx in tx_rows:
        item = TransactionItem(
            invoice_no=tx.invoice_no,
            stock_code=tx.stock_code,
            quantity=tx.quantity,
            unit_price=tx.unit_price,
            total_amount=tx.total_amount,
            purchase_amount=tx.purchase_amount,
            credit_amount=tx.credit_amount,
            hour=tx.hour,
            is_negative_quantity=tx.is_negative_quantity,
            is_return_like=tx.is_return_like,
            is_anomaly=tx.is_anomaly,
            anomaly_score=tx.anomaly_score,
            created_at=tx.created_at,
        )
        db.add(item)
    db.commit()


def init_db() -> None:
    """Create all database tables, seed default users, and sync normalized schema."""
    Base.metadata.create_all(bind=engine)

    # Migrate prediction_logs if missing user_id column from older version
    if is_sqlite:
        with engine.connect() as conn:
            cols = [row[1] for row in conn.exec_driver_sql("PRAGMA table_info(prediction_logs);").fetchall()]
            if "user_id" not in cols:
                conn.exec_driver_sql("ALTER TABLE prediction_logs ADD COLUMN user_id INTEGER REFERENCES users(id);")
                conn.commit()

    db = SessionLocal()
    try:
        seed_default_users(db)
        sync_normalized_from_transactions(db)
    finally:
        db.close()


def get_db():
    """FastAPI dependency yielding a scoped SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
