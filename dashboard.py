from contextlib import contextmanager
from datetime import datetime, time, timedelta
import os
import random

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from auth import create_access_token, hash_password, verify_password
from database import Customer, Invoice, Product, SessionLocal, TransactionItem, User, init_db
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

st.set_page_config(
    page_title="Retail Intelligence & Sales Manager AI Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database schema and default accounts
init_db()


@contextmanager
def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@st.cache_resource
def load_cached_models():
    try:
        return load_ml_models()
    except Exception as e:
        return {"error": str(e)}


models = load_cached_models()
has_local_models = "error" not in models

# ----------------- SESSION STATE & AUTHENTICATION -----------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = True
    st.session_state["user"] = {
        "id": 2,
        "name": "Senior Sales Manager",
        "email": "manager@retail.com",
        "role": "Sales Manager",
    }

# ----------------- SIDEBAR: STAKEHOLDER AUTH & NAVIGATION -----------------
st.sidebar.title("🔐 Stakeholder Access")

current_user = st.session_state["user"]
role = current_user["role"]

role_badges = {
    "Admin": "👑 **Administrator**",
    "Sales Manager": "💼 **Sales Manager**",
    "Analyst": "📊 **Data Analyst**",
}

st.sidebar.markdown(f"**Logged in as:** {current_user['name']}")
st.sidebar.markdown(f"**Role:** {role_badges.get(role, role)}")
st.sidebar.caption(f"Email: {current_user['email']}")

# Quick Role Switcher for live presentation demonstration
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎭 Quick Role Switcher *(Live Demo)*")
demo_cols = st.sidebar.columns(3)
if demo_cols[0].button("👑 Admin"):
    st.session_state["user"] = {
        "id": 1,
        "name": "System Administrator",
        "email": "admin@retail.com",
        "role": "Admin",
    }
    st.rerun()

if demo_cols[1].button("💼 Manager"):
    st.session_state["user"] = {
        "id": 2,
        "name": "Senior Sales Manager",
        "email": "manager@retail.com",
        "role": "Sales Manager",
    }
    st.rerun()

if demo_cols[2].button("📊 Analyst"):
    st.session_state["user"] = {
        "id": 3,
        "name": "Data Analyst",
        "email": "analyst@retail.com",
        "role": "Analyst",
    }
    st.rerun()

# Expandable Login / Register modal
with st.sidebar.expander("🔑 Login with Custom Account"):
    login_tab, reg_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("custom_login_form"):
            l_email = st.text_input("Email", value="")
            l_pass = st.text_input("Password", type="password")
            l_submit = st.form_submit_button("Sign In")
            if l_submit:
                with get_db_session() as db:
                    u = db.query(User).filter(User.email == l_email.strip().lower()).first()
                    if u and verify_password(l_pass, u.password_hash):
                        st.session_state["user"] = {
                            "id": u.id,
                            "name": u.name,
                            "email": u.email,
                            "role": u.role,
                        }
                        st.success(f"Welcome back, {u.name}!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")

    with reg_tab:
        with st.form("custom_register_form"):
            r_name = st.text_input("Full Name")
            r_email = st.text_input("Email")
            r_pass = st.text_input("Password", type="password")
            r_role = st.selectbox("Role", ["Sales Manager", "Analyst", "Admin"])
            r_submit = st.form_submit_button("Register")
            if r_submit:
                if len(r_name) < 2 or "@" not in r_email or len(r_pass) < 6:
                    st.error("Please fill all fields (password >= 6 chars).")
                else:
                    with get_db_session() as db:
                        existing = db.query(User).filter(User.email == r_email.strip().lower()).first()
                        if existing:
                            st.error("Account already exists with this email.")
                        else:
                            new_u = User(
                                name=r_name.strip(),
                                email=r_email.strip().lower(),
                                password_hash=hash_password(r_pass),
                                role=r_role,
                            )
                            db.add(new_u)
                            db.commit()
                            db.refresh(new_u)
                            st.session_state["user"] = {
                                "id": new_u.id,
                                "name": new_u.name,
                                "email": new_u.email,
                                "role": new_u.role,
                            }
                            st.success("Registered and logged in!")
                            st.rerun()

st.sidebar.markdown("---")
# System & Database Status
with get_db_session() as db:
    live_summary = get_analytics_summary(db)

if has_local_models:
    st.sidebar.success("🟢 ML Models: 5/5 Operational")
else:
    st.sidebar.error(f"🔴 Models Missing: {models.get('error')}")

st.sidebar.success("🗄️ Database: Connected (3NF SQLite)")
st.sidebar.metric("Transactions Stored", f"{live_summary['total_transactions']:,}")
st.sidebar.metric("Anomalies Flagged", f"{live_summary['anomalous_transactions']:,}")
st.sidebar.metric("Total Purchase Revenue", f"${live_summary['total_purchase_revenue']:,.2f}")
st.sidebar.metric("Refund / Credit Volume", f"${live_summary['total_credit_volume']:,.2f}")

if st.sidebar.button("🔄 Refresh Live Database"):
    st.rerun()


# ----------------- MAIN TITLE & KPI CARDS -----------------
st.title("📈 Retail Intelligence & Decision Support Platform")
st.markdown(
    "Enterprise retail system integrating **JWT Role-Based Access Control**, **3NF Relational Storage**, "
    "**Real-time Anomaly Detection**, and **Predictive ML Decision Engines**."
)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total Records", f"{live_summary['total_transactions']:,}")
kpi2.metric("Purchase Revenue", f"${live_summary['total_purchase_revenue']:,.2f}")
kpi3.metric("Refunds / Credit", f"${live_summary['total_credit_volume']:,.2f}")
kpi4.metric(
    "Cancellations",
    f"{live_summary['cancelled_transactions']:,}",
    delta=f"{live_summary['cancellation_rate'] * 100:.1f}% rate",
    delta_color="inverse",
)
kpi5.metric("Unique SKUs / Customers", f"{live_summary['unique_products']} / {live_summary['unique_customers']}")

# Main Navigation Tabs
tab_names = [
    "⚡ Data Ingestion",
    "🗄️ Database CRUD Explorer",
    "🛡️ Product & Customer Risk",
    "📈 Sales & Cancellation Forecasts",
    "⚠️ Anomaly Detection",
    "💰 Financial Impact Simulator",
    "📜 Prediction Audit History",
]
if role == "Admin":
    tab_names.append("👥 Stakeholder Management")

tabs = st.tabs(tab_names)


# ----------------- TAB 1: DATA INGESTION -----------------
with tabs[0]:
    st.subheader("⚡ Real-Time Data Ingestion & Batch CSV Upload")

    if role == "Analyst":
        st.info("ℹ️ **Analyst Access:** You have read-only access. Switch to **Sales Manager** or **Admin** to ingest live transactions.")

    col_form, col_bulk = st.columns([3, 2])

    with col_form:
        st.markdown("### ➕ Live Single Transaction Entry")
        with st.form("realtime_tx_form", clear_on_submit=False):
            c1, c2, c3 = st.columns(3)
            inv_no = c1.text_input("Invoice No", value=f"INV-{datetime.now().strftime('%H%M%S')}")
            stock_code = c2.text_input("Stock Code", value="85123A")
            cust_id = c3.text_input("Customer ID (optional)", value="17850")

            c4, c5, c6 = st.columns(3)
            description = c4.text_input("Description", value="WHITE HANGING HEART T-LIGHT HOLDER")
            quantity = c5.number_input("Quantity (negative for returns)", value=6.0, step=1.0)
            unit_price = c6.number_input("Unit Price ($)", value=2.55, min_value=0.0, step=0.5)

            c7, c8 = st.columns(2)
            country = c7.text_input("Country", value="United Kingdom")
            inv_date = c8.date_input("Invoice Date", value=datetime.today())

            submit_tx = st.form_submit_button(
                "🚀 Submit & Ingest into Database",
                disabled=(role == "Analyst"),
            )

            if submit_tx:
                with st.spinner("Ingesting transaction, scoring anomalies, and syncing 3NF schema..."):
                    with get_db_session() as db:
                        dt = datetime.combine(inv_date, time(12, 0))
                        record = ingest_transaction(
                            db,
                            invoice_no=inv_no,
                            stock_code=stock_code,
                            quantity=quantity,
                            unit_price=unit_price,
                            description=description,
                            invoice_date=dt,
                            customer_id=cust_id if cust_id else None,
                            country=country,
                            models=models if has_local_models else None,
                        )
                    st.success(
                        f"✅ Transaction #{record.id} stored successfully! "
                        f"(Total: ${record.total_amount:,.2f})"
                    )
                    if record.is_anomaly:
                        st.warning(f"⚠️ Flagged as Anomaly! Score: {record.anomaly_score:.4f}")

    with col_bulk:
        st.markdown("### 📂 Bulk CSV Ingestion")
        if role != "Admin":
            st.warning("🔒 **Restricted:** Bulk CSV file upload requires **Administrator** role.")
        else:
            uploaded_file = st.file_uploader(
                "Upload transactions CSV (InvoiceNo, StockCode, Quantity, UnitPrice)",
                type=["csv"],
            )
            if uploaded_file is not None:
                if st.button("📥 Parse & Ingest Entire CSV"):
                    with st.spinner("Processing CSV and synchronizing normalized tables..."):
                        try:
                            raw = uploaded_file.getvalue()
                            items = parse_transactions_csv_bytes(raw)
                            with get_db_session() as db:
                                recs = ingest_transactions_batch(
                                    db,
                                    items,
                                    models=models if has_local_models else None,
                                )
                            anom_count = sum(1 for r in recs if r.is_anomaly)
                            st.success(f"✅ Ingested {len(recs)} transactions! ({anom_count} anomalies flagged)")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error processing CSV: {e}")


# ----------------- TAB 2: DATABASE CRUD EXPLORER -----------------
with tabs[1]:
    st.subheader("🗄️ Database CRUD Operations (Create, Read, Update, Delete)")
    st.markdown("Inspect stored data in 3NF schema, filter records, update attributes, or delete rows.")

    with get_db_session() as db:
        # Filter controls
        f1, f2, f3, f4 = st.columns(4)
        filter_stock = f1.text_input("Filter by Stock Code", value="")
        filter_cust = f2.text_input("Filter by Customer ID", value="")
        filter_inv = f3.text_input("Filter by Invoice No", value="")
        filter_anomaly = f4.selectbox("Anomaly Filter", ["All", "Anomalies Only", "Normal Only"])

        anom_val = True if filter_anomaly == "Anomalies Only" else (False if filter_anomaly == "Normal Only" else None)

        total_tx, tx_rows = query_transactions(
            db,
            stock_code=filter_stock if filter_stock else None,
            customer_id=filter_cust if filter_cust else None,
            invoice_no=filter_inv if filter_inv else None,
            is_anomaly=anom_val,
            limit=100,
        )

        st.caption(f"Displaying top {len(tx_rows)} of {total_tx} matching transactions in SQLite database:")
        if tx_rows:
            df_display = pd.DataFrame([serialize_transaction(r) for r in tx_rows])
            st.dataframe(df_display, use_container_width=True)

            # CRUD: UPDATE & DELETE Section
            st.markdown("---")
            st.markdown("### ✏️ Live Record Update & Deletion (CRUD)")

            crud_col1, crud_col2 = st.columns(2)

            with crud_col1:
                st.markdown("#### 🔄 Update Transaction (PUT/PATCH)")
                selected_id = st.selectbox(
                    "Select Transaction ID to Edit",
                    options=[r.id for r in tx_rows],
                )
                selected_record = next(r for r in tx_rows if r.id == selected_id)

                with st.form("update_transaction_form"):
                    u_desc = st.text_input("Description", value=selected_record.description)
                    u_qty = st.number_input("Quantity", value=float(selected_record.quantity), step=1.0)
                    u_price = st.number_input("Unit Price ($)", value=float(selected_record.unit_price), min_value=0.0, step=0.5)
                    u_country = st.text_input("Country", value=selected_record.country)
                    u_cust = st.text_input("Customer ID", value=selected_record.customer_id or "")

                    btn_update = st.form_submit_button(
                        "💾 Update Record in Database",
                        disabled=(role == "Analyst"),
                    )

                    if btn_update:
                        with st.spinner("Updating transaction and re-evaluating anomaly metrics..."):
                            up_rec = update_transaction(
                                db,
                                transaction_id=selected_id,
                                quantity=u_qty,
                                unit_price=u_price,
                                description=u_desc,
                                country=u_country,
                                customer_id=u_cust,
                                models=models if has_local_models else None,
                            )
                        st.success(f"✅ Record #{up_rec.id} updated! New total: ${up_rec.total_amount:,.2f}")
                        st.rerun()

            with crud_col2:
                st.markdown("#### 🗑️ Delete Transaction (DELETE)")
                st.markdown(f"**Target ID:** #{selected_id}")
                st.markdown(f"**Invoice:** `{selected_record.invoice_no}` | **SKU:** `{selected_record.stock_code}`")
                st.markdown(f"**Amount:** `${selected_record.total_amount:,.2f}`")

                if role != "Admin":
                    st.info("🔒 Record deletion is restricted to **Administrator** accounts.")
                else:
                    if st.button("🚨 Confirm Delete Record", type="primary"):
                        with st.spinner("Deleting record from database..."):
                            del_ok = delete_transaction(db, selected_id)
                        if del_ok:
                            st.success(f"🗑️ Transaction #{selected_id} deleted permanently!")
                            st.rerun()
                        else:
                            st.error("Deletion failed.")
        else:
            st.info("No transactions found matching your criteria.")


# ----------------- TAB 3: PRODUCT & CUSTOMER RISK -----------------
with tabs[2]:
    st.subheader("🛡️ Product & Customer Return Risk Classification")
    col_p, col_c = st.columns(2)

    with col_p:
        st.markdown("### 📦 Product Return Risk (Feature 1: Random Forest)")
        with get_db_session() as db:
            prod_df = get_product_features_from_db(db)

        if not prod_df.empty:
            sku_list = prod_df["StockCode"].tolist()
            chosen_sku = st.selectbox("Select Product StockCode from Database", sku_list)
            if st.button("Predict Product Risk"):
                if not has_local_models:
                    st.error("Models not available.")
                else:
                    with st.spinner("Computing live product metrics and evaluating Random Forest..."):
                        with get_db_session() as db:
                            res = predict_product_risk_from_db(
                                db,
                                chosen_sku,
                                models,
                                user_id=current_user["id"],
                            )
                    if res:
                        risk_level = res["product_risk"]
                        if risk_level == "High":
                            st.error(f"🚨 **High Return Risk** for SKU: {chosen_sku}")
                        elif risk_level == "Medium":
                            st.warning(f"⚠️ **Medium Return Risk** for SKU: {chosen_sku}")
                        else:
                            st.success(f"✅ **Low Return Risk** for SKU: {chosen_sku}")
                        st.json(res["features"])
        else:
            st.info("No product transactions in database yet.")

    with col_c:
        st.markdown("### 👤 Customer Return Risk (Feature 2: Logistic Regression)")
        with get_db_session() as db:
            cust_df = get_customer_features_from_db(db)

        if not cust_df.empty:
            cust_list = cust_df["CustomerID"].tolist()
            chosen_cust = st.selectbox("Select Customer ID from Database", cust_list)
            if st.button("Predict Customer Risk"):
                if not has_local_models:
                    st.error("Models not available.")
                else:
                    with st.spinner("Computing customer metrics and evaluating Logistic Regression..."):
                        with get_db_session() as db:
                            res = predict_customer_return_from_db(
                                db,
                                chosen_cust,
                                models,
                                user_id=current_user["id"],
                            )
                    if res:
                        prob = res["high_risk_probability"]
                        risk = res["risk"]
                        if risk == "High":
                            st.error(f"🚨 **High Return Probability:** {prob * 100:.2f}%")
                        else:
                            st.success(f"✅ **Low Return Probability:** {prob * 100:.2f}%")
                        st.json(res["features"])
        else:
            st.info("No customer transactions in database yet.")


# ----------------- TAB 4: FORECASTS -----------------
with tabs[3]:
    st.subheader("📈 Time-Series Forecasting (Feature 3 & 4: Prophet Models)")
    forecast_days = st.slider("Forecast Horizon (Days)", min_value=7, max_value=90, value=28)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🛒 Daily Sales Revenue Forecast")
        if st.button("Generate Sales Forecast", key="sales_btn"):
            if not has_local_models:
                st.error("Models not available.")
            else:
                with st.spinner("Forecasting revenue with Prophet..."):
                    m3 = models["m3"]
                    future = m3.make_future_dataframe(periods=forecast_days)
                    forecast = m3.predict(future)
                    df_sales = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_days).copy()

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_sales["ds"], y=df_sales["yhat_upper"],
                        mode="lines", line=dict(width=0), showlegend=False,
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_sales["ds"], y=df_sales["yhat_lower"],
                        mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(0,100,255,0.15)", name="Confidence Interval",
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_sales["ds"], y=df_sales["yhat"],
                        mode="lines+markers", line=dict(color="#1f77b4", width=2),
                        name="Predicted Sales ($)",
                    ))
                    fig.update_layout(title=f"Predicted Sales Revenue ({forecast_days} Days)", xaxis_title="Date", yaxis_title="Sales Value ($)")
                    st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### ❌ Daily Cancellation Forecast")
        if st.button("Generate Cancellation Forecast", key="canc_btn"):
            if not has_local_models:
                st.error("Models not available.")
            else:
                with st.spinner("Forecasting cancellations with log-transformed Prophet..."):
                    m4 = models["m4"]
                    future = m4.make_future_dataframe(periods=forecast_days)
                    forecast = m4.predict(future)
                    df_canc = forecast[["ds", "yhat"]].tail(forecast_days).copy()
                    df_canc["predicted_cancellations"] = np.clip(np.expm1(df_canc["yhat"]), 0, None).round(2)

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df_canc["ds"], y=df_canc["predicted_cancellations"],
                        marker_color="#ff7f0e", name="Cancellations",
                    ))
                    fig.update_layout(title=f"Predicted Cancellations ({forecast_days} Days)", xaxis_title="Date", yaxis_title="Units")
                    st.plotly_chart(fig, use_container_width=True)


# ----------------- TAB 5: ANOMALY DETECTION -----------------
with tabs[4]:
    st.subheader("⚠️ Unusual Transaction Detection (Feature 6: Isolation Forest)")
    st.markdown("### 🚨 Flagged Database Anomalies")
    with get_db_session() as db:
        anom_total, anom_rows = query_transactions(db, is_anomaly=True, limit=50)
    if anom_rows:
        st.warning(f"Found {anom_total:,} anomalous transactions stored in database.")
        st.dataframe(pd.DataFrame([serialize_transaction(r) for r in anom_rows]), use_container_width=True)
    else:
        st.success("No anomalous transactions stored in database yet.")

    st.markdown("---")
    st.markdown("### 🧪 Manual Anomaly Diagnostic Tool")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        qty = st.number_input("Transaction Quantity", value=12.0, step=1.0)
        unit_p = st.number_input("Unit Price ($)", value=25.0, step=5.0)
    with col_a2:
        tot_amt = st.number_input("Total Amount ($)", value=300.0, step=50.0)
        hour = st.slider("Hour of Day (0-23)", min_value=0, max_value=23, value=14)

    if st.button("Evaluate Transaction with Isolation Forest"):
        if not has_local_models:
            st.error("Models not available.")
        else:
            with st.spinner("Scoring transaction anomaly..."):
                m6 = models["m6"]
                payload = {"Quantity": qty, "UnitPrice": unit_p, "TotalAmount": tot_amt, "Hour": hour}
                features = pd.DataFrame([payload])
                scaled = m6["scaler"].transform(features[m6["features"]])
                pred = int(m6["model"].predict(scaled)[0])
                score = float(m6["model"].score_samples(scaled)[0])
                is_anomaly = pred == -1
                with get_db_session() as db:
                    log_prediction(
                        db,
                        prediction_type="anomaly",
                        entity_id=None,
                        input_data=payload,
                        result_data={"anomaly": is_anomaly, "anomaly_score": round(score, 4)},
                        user_id=current_user["id"],
                    )

            if is_anomaly:
                st.error(f"🚨 **Unusual / Anomalous Transaction Detected!** (Anomaly Score: {score:.4f})")
            else:
                st.success(f"✅ **Normal Transaction** (Anomaly Score: {score:.4f})")


# ----------------- TAB 6: FINANCIAL IMPACT -----------------
with tabs[5]:
    st.subheader("💰 Financial Impact & Savings Simulator")
    st.markdown("Calculate potential cost reductions by proactively preventing returns and cancellations.")

    default_credit = live_summary["total_credit_volume"] if live_summary["total_credit_volume"] > 0 else 10000.0
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        credit_input = st.number_input(
            "Total Credit/Refund Volume ($)",
            value=float(default_credit),
            step=500.0,
        )
        reduction_slider = st.slider("Target Return Reduction Rate (%)", min_value=1, max_value=50, value=20)
        st.caption(f"Current Live Refund Volume in Database: ${live_summary['total_credit_volume']:,.2f}")

    with col_f2:
        if st.button("Calculate Projected Cost Savings"):
            pct = reduction_slider / 100.0
            saving = credit_input * pct
            with get_db_session() as db:
                log_prediction(
                    db,
                    prediction_type="financial_impact",
                    entity_id=None,
                    input_data={"total_credit": credit_input, "reduction_pct": pct},
                    result_data={"potential_saving": round(saving, 2)},
                    user_id=current_user["id"],
                )
            st.metric(
                label="Estimated Direct Cost Savings",
                value=f"${saving:,.2f}",
                delta=f"{reduction_slider}% reduction",
            )
            st.info(f"Proactive interventions on top high-risk SKUs can recover up to ${saving:,.2f} annually.")


# ----------------- TAB 7: PREDICTION AUDIT HISTORY -----------------
with tabs[6]:
    st.subheader("📜 Stakeholder Prediction Audit History")
    st.markdown("Audited logs of all ML predictions, timestamps, inputs, outputs, and stakeholder attribution.")

    with get_db_session() as db:
        tot_pred, pred_items = query_predictions(db, limit=100)

    st.caption(f"Total Logged Predictions: {tot_pred}")
    if pred_items:
        df_p = pd.DataFrame(pred_items)
        st.dataframe(df_p, use_container_width=True)
    else:
        st.info("No predictions logged yet. Run a prediction in Tab 3, 4, 5, or 6.")


# ----------------- TAB 8: STAKEHOLDER MANAGEMENT (ADMIN ONLY) -----------------
if role == "Admin" and len(tabs) > 7:
    with tabs[7]:
        st.subheader("👥 Stakeholder Directory & Role Management")
        st.markdown("Administrators can view all registered users and assign organizational roles.")

        with get_db_session() as db:
            all_users = db.query(User).order_by(User.id).all()
            user_records = [
                {
                    "ID": u.id,
                    "Name": u.name,
                    "Email": u.email,
                    "Role": u.role,
                    "Status": "Active" if u.is_active else "Inactive",
                    "Created At": u.created_at.strftime("%Y-%m-%d %H:%M") if u.created_at else "-",
                }
                for u in all_users
            ]
            st.dataframe(pd.DataFrame(user_records), use_container_width=True)
