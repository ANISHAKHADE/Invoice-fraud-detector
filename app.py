import streamlit as st
import pandas as pd
import joblib

# 1. Set up the UI layout
st.set_page_config(page_title="Razorpay Fraud Engine", layout="wide")
st.title("🛡️ Enterprise Procurement Fraud Detector")
st.write("Upload a CSV of processed invoices to instantly scan for anomalies.")

# 2. Load BOTH saved models
@st.cache_resource
def load_models():
    xgb = joblib.load("fraud_model.pkl")
    iso = joblib.load("isolation_forest.pkl")
    return xgb, iso

xgb_model, iso_model = load_models()

# 3. Create the UI Controls
st.markdown("### System Settings")
use_hybrid = st.toggle("🚨 Activate Strict Hybrid Mode (XGBoost + Isolation Forest)", value=False)
st.caption("When activated, the system casts a wider net to catch Zero-Day anomalies, but may increase False Positives.")

# 4. Create the file upload button
uploaded_file = st.file_uploader("Upload ML-Ready Invoice Data (CSV)", type="csv")

if uploaded_file is not None:
    # Read the data
    df = pd.read_csv(uploaded_file)
    
    # If the user accidentally uploaded the file with the answer key, drop it so the AI can guess
    if 'is_fraud' in df.columns:
        X = df.drop(columns=['is_fraud'])
    else:
        X = df
        
    with st.spinner("Analyzing transactions..."):
        # Step A: Get standard XGBoost predictions (30% Threshold)
        probabilities = xgb_model.predict_proba(X)[:, 1]
        xgb_flags = (probabilities >= 0.30).astype(int)
        
        # Step B: Check if Hybrid Mode is ON
        if use_hybrid:
            # Get Isolation Forest predictions (-1 is anomaly, 1 is normal)
            if_preds = iso_model.predict(X)
            if_flags = (if_preds == -1).astype(int)
            
            # Combine them: Flag if XGBoost OR Isolation Forest triggers
            final_flags = (xgb_flags | if_flags)
        else:
            final_flags = xgb_flags
            
        # Attach results to the dashboard view
        results_df = df.copy()
        results_df.insert(0, 'Fraud_Risk_%', (probabilities * 100).round(2))
        results_df.insert(1, 'System_Action', ['🛑 BLOCKED' if flag == 1 else '✅ APPROVED' for flag in final_flags])
        
        # Display the metrics
        st.subheader("Scan Results")
        col1, col2 = st.columns(2)
        col1.metric("Total Invoices Scanned", len(results_df))
        col2.metric("Anomalies Blocked", sum(final_flags))
        
        # Show the interactive data table
        st.dataframe(results_df, use_container_width=True)