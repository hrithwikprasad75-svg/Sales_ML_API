import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# Optional API base URL (can be customized via Streamlit secrets or sidebar)
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Sales Manager AI Dashboard",
    page_icon="📈",
    layout="wide"
)

# ----------------- CACHED MODEL LOADER (for standalone/cloud execution) -----------------
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")
    try:
        m1 = joblib.load(os.path.join(model_dir, "feature1_product_risk.pkl"))
        m2 = joblib.load(os.path.join(model_dir, "feature2_customer_return.pkl"))
        m3 = joblib.load(os.path.join(model_dir, "feature3_sales_forecast.pkl"))
        m4 = joblib.load(os.path.join(model_dir, "feature4_cancellation_forecast.pkl"))
        m6 = joblib.load(os.path.join(model_dir, "feature6_anomaly.pkl"))
        return {"m1": m1, "m2": m2, "m3": m3, "m4": m4, "m6": m6}
    except Exception as e:
        return {"error": str(e)}

models = load_models()
has_local_models = "error" not in models

st.title("📈 Sales Manager AI Dashboard")
st.markdown("Real-time ML decision support for Sales & Inventory Managers")

# Sidebar Status
if has_local_models:
    st.sidebar.success("🟢 ML Models: Loaded & Ready")
else:
    st.sidebar.error(f"🔴 Model Loading Error: {models.get('error')}")

st.sidebar.markdown("---")
st.sidebar.caption("Dashboard running in self-contained cloud mode with 5 trained models.")

tabs = st.tabs([
    "📈 Sales & Cancellation Forecasts",
    "🛡️ Product & Customer Risk",
    "⚠️ Anomaly Detection",
    "💰 Financial Impact"
])

# ----------------- TAB 1: FORECASTS -----------------
with tabs[0]:
    st.subheader("Future Forecasts (Prophet Time Series Models)")
    forecast_days = st.slider("Forecast Horizon (Days)", min_value=7, max_value=90, value=28, step=7)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🛒 Sales Forecast")
        if st.button("Generate Sales Forecast", key="sales_btn"):
            if not has_local_models:
                st.error("Models not available.")
            else:
                with st.spinner("Forecasting sales..."):
                    m3 = models["m3"]
                    future = m3.make_future_dataframe(periods=forecast_days)
                    forecast = m3.predict(future)
                    df_sales = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_days).copy()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_sales["ds"], y=df_sales["yhat_upper"],
                        mode="lines", line=dict(width=0), showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_sales["ds"], y=df_sales["yhat_lower"],
                        mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(0,100,255,0.15)", name="Confidence Interval"
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_sales["ds"], y=df_sales["yhat"],
                        mode="lines+markers", line=dict(color="#1f77b4", width=2),
                        name="Predicted Sales"
                    ))
                    fig.update_layout(title=f"Predicted Sales ({forecast_days} Days)", xaxis_title="Date", yaxis_title="Sales Value")
                    st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### ❌ Cancellation Forecast")
        if st.button("Generate Cancellation Forecast", key="canc_btn"):
            if not has_local_models:
                st.error("Models not available.")
            else:
                with st.spinner("Forecasting cancellations..."):
                    m4 = models["m4"]
                    future = m4.make_future_dataframe(periods=forecast_days)
                    forecast = m4.predict(future)
                    df_canc = forecast[["ds", "yhat"]].tail(forecast_days).copy()
                    df_canc["predicted_cancellations"] = np.clip(np.expm1(df_canc["yhat"]), 0, None).round(2)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df_canc["ds"], y=df_canc["predicted_cancellations"],
                        marker_color="#ff7f0e", name="Cancellations"
                    ))
                    fig.update_layout(title=f"Predicted Cancellations ({forecast_days} Days)", xaxis_title="Date", yaxis_title="Units")
                    st.plotly_chart(fig, use_container_width=True)

# ----------------- TAB 2: RISK ASSESSMENTS -----------------
with tabs[1]:
    col_prod, col_cust = st.columns(2)
    
    with col_prod:
        st.subheader("📦 Product Return Risk")
        cancellation_rate = st.number_input("Cancellation Row Rate", value=0.05, step=0.01, format="%.3f")
        credit_ratio = st.number_input("Credit To Purchase Ratio", value=0.02, step=0.01, format="%.3f")
        avg_price = st.number_input("Average Unit Price ($)", value=15.5, step=1.0)
        prod_invoices = st.number_input("Product Total Invoices", value=12, step=1)
        prod_purchase = st.number_input("Product Total Purchase Amount ($)", value=3400.0, step=100.0)
        neg_qty = st.number_input("Negative Quantity Rows", value=1, step=1)
        
        if st.button("Predict Product Risk"):
            if not has_local_models:
                st.error("Models not available.")
            else:
                m1 = models["m1"]
                features = pd.DataFrame([{
                    "CancellationRowRate": cancellation_rate,
                    "CreditToPurchaseRatio": credit_ratio,
                    "AvgUnitPrice": avg_price,
                    "TotalInvoices": prod_invoices,
                    "TotalPurchaseAmount": prod_purchase,
                    "NegativeQuantityRows": neg_qty
                }])
                pred = m1["model"].predict(features[m1["features"]])
                risk = m1["label_encoder"].inverse_transform(pred)[0]
                
                if str(risk).lower() == "low":
                    st.success(f"Product Return Risk: **{risk}**")
                elif str(risk).lower() == "medium":
                    st.warning(f"Product Return Risk: **{risk}**")
                else:
                    st.error(f"Product Return Risk: **{risk}**")
                    
    with col_cust:
        st.subheader("👤 Customer Return Behaviour")
        cust_invoices = st.number_input("Customer Total Invoices", value=10, step=1)
        cust_canc_inv = st.number_input("Customer Cancellation Invoices", value=1, step=1)
        cust_canc_rate = st.number_input("Customer Cancellation Rate", value=0.10, step=0.01, format="%.2f")
        cust_amount = st.number_input("Customer Total Purchase Amount ($)", value=1250.0, step=50.0)
        cust_avg_val = st.number_input("Average Purchase Value ($)", value=125.0, step=10.0)
        
        if st.button("Predict Customer Return"):
            if not has_local_models:
                st.error("Models not available.")
            else:
                m2 = models["m2"]
                features = pd.DataFrame([{
                    "TotalInvoices": cust_invoices,
                    "CancellationInvoices": cust_canc_inv,
                    "CancellationRate": cust_canc_rate,
                    "TotalPurchaseAmount": cust_amount,
                    "AvgPurchaseValue": cust_avg_val
                }])
                prob = float(m2["model"].predict_proba(features[m2["features"]])[0][1])
                threshold = m2["threshold"]
                is_high = prob > threshold
                
                st.metric("High-Risk Probability", f"{prob * 100:.1f}%")
                st.progress(min(prob, 1.0))
                if is_high:
                    st.error("⚠️ High Risk Customer (Likely to Return)")
                else:
                    st.success("✅ Low Risk Customer")

# ----------------- TAB 3: ANOMALY DETECTION -----------------
with tabs[2]:
    st.subheader("🔍 Unusual Transaction Detection")
    st.markdown("Evaluates whether a single transaction deviates from normal purchasing patterns.")
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        qty = st.number_input("Transaction Quantity", value=12.0, step=1.0)
        unit_p = st.number_input("Unit Price ($)", value=25.0, step=5.0)
    with col_a2:
        tot_amt = st.number_input("Total Amount ($)", value=300.0, step=50.0)
        hour = st.slider("Hour of Day (0-23)", min_value=0, max_value=23, value=14)
        
    if st.button("Analyze Transaction"):
        if not has_local_models:
            st.error("Models not available.")
        else:
            m6 = models["m6"]
            features = pd.DataFrame([{
                "Quantity": qty,
                "UnitPrice": unit_p,
                "TotalAmount": tot_amt,
                "Hour": hour
            }])
            scaled = m6["scaler"].transform(features[m6["features"]])
            pred = int(m6["model"].predict(scaled)[0])
            score = float(m6["model"].score_samples(scaled)[0])
            is_anomaly = pred == -1
            
            if is_anomaly:
                st.error(f"🚨 **Unusual Transaction** (Anomaly Score: {score:.4f})")
                st.caption("This transaction deviates significantly from regular invoice distributions.")
            else:
                st.success(f"✅ **Normal Transaction** (Anomaly Score: {score:.4f})")
                st.caption("This transaction matches normal customer purchasing patterns.")

# ----------------- TAB 4: FINANCIAL IMPACT -----------------
with tabs[3]:
    st.subheader("💵 Financial Impact & Potential Savings")
    st.markdown("Calculate how proactive interventions can protect revenue.")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        credit_input = st.number_input("Total Credit/Refund Volume ($)", value=10000.0, step=500.0)
        reduction_slider = st.slider("Target Return Reduction (%)", min_value=1, max_value=50, value=20)
        
    with col_f2:
        if st.button("Calculate Savings"):
            pct = reduction_slider / 100.0
            saving = credit_input * pct
            st.metric(
                label="Estimated Direct Cost Savings",
                value=f"${saving:,.2f}",
                delta=f"{reduction_slider}% reduction"
            )
