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
    # 1. Wrap the ENTIRE process (reading + predicting) in the spinner
    with st.spinner("⏳ Ingesting data and running AI threat detection... Please wait."):
        
        # Read the data 
        df = pd.read_csv(uploaded_file)
        
        if 'is_fraud' in df.columns:
            X = df.drop(columns=['is_fraud'])
        else:
            X = df
            
        # Get standard XGBoost predictions
        probabilities = xgb_model.predict_proba(X)[:, 1]
        xgb_flags = (probabilities >= 0.30).astype(int)
        
        # Check if Hybrid Mode is ON
        if use_hybrid:
            if_preds = iso_model.predict(X)
            if_flags = (if_preds == -1).astype(int)
            final_flags = (xgb_flags | if_flags)
        else:
            final_flags = xgb_flags
            
        # Attach results
        results_df = df.copy()
        results_df.insert(0, 'Fraud_Risk_%', (probabilities * 100).round(2))
        results_df.insert(1, 'System_Action', ['🛑 BLOCKED' if flag == 1 else '✅ APPROVED' for flag in final_flags])
        
    # 2. Flash a success message once the spinner finishes
    st.success("✅ Scan Complete! Anomalies isolated below.")
    
    # Display the metrics
    st.subheader("Scan Results")
    col1, col2 = st.columns(2)
    col1.metric("Total Invoices Scanned", len(results_df))
    col2.metric("Anomalies Blocked", sum(final_flags))
    
    # Show the interactive data tables using UI Tabs
    tab1, tab2 = st.tabs(["🚨 Action Required (Blocked)", "📋 Full Audit Log"])
    
    with tab1:
        blocked_df = results_df[results_df['System_Action'] == '🛑 BLOCKED']
        st.dataframe(blocked_df, use_container_width=True)
        
    with tab2:
        st.dataframe(results_df, use_container_width=True)