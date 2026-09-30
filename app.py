# import streamlit as st
# import pandas as pd
# import joblib

# # 1. Set up the UI layout
# st.set_page_config(page_title="Razorpay Fraud Engine", layout="wide")
# st.title("🛡️ Enterprise Procurement Fraud Detector")
# st.write("Upload a CSV of processed invoices to instantly scan for anomalies.")

# # 2. Load BOTH saved models
# @st.cache_resource
# def load_models():
#     xgb = joblib.load("fraud_model.pkl")
#     iso = joblib.load("isolation_forest.pkl")
#     return xgb, iso

# xgb_model, iso_model = load_models()

# # 3. Create the UI Controls
# st.markdown("### System Settings")
# use_hybrid = st.toggle("🚨 Activate Strict Hybrid Mode (XGBoost + Isolation Forest)", value=False)
# st.caption("When activated, the system casts a wider net to catch Zero-Day anomalies, but may increase False Positives.")

# # 4. Create the file upload button
# uploaded_file = st.file_uploader("Upload ML-Ready Invoice Data (CSV)", type="csv")

# if uploaded_file is not None:
#     # 1. Wrap the ENTIRE process (reading + predicting) in the spinner
#     with st.spinner("⏳ Ingesting data and running AI threat detection... Please wait."):
        
#         # Read the data 
#         df = pd.read_csv(uploaded_file)
        
#         if 'is_fraud' in df.columns:
#             X = df.drop(columns=['is_fraud'])
#         else:
#             X = df
            
#         # Get standard XGBoost predictions
#         probabilities = xgb_model.predict_proba(X)[:, 1]
#         xgb_flags = (probabilities >= 0.30).astype(int)
        
#         # Check if Hybrid Mode is ON
#         if use_hybrid:
#             if_preds = iso_model.predict(X)
#             if_flags = (if_preds == -1).astype(int)
#             final_flags = (xgb_flags | if_flags)
#         else:
#             final_flags = xgb_flags
            
#         # Attach results
#         results_df = df.copy()
#         results_df.insert(0, 'Fraud_Risk_%', (probabilities * 100).round(2))
#         results_df.insert(1, 'System_Action', ['🛑 BLOCKED' if flag == 1 else '✅ APPROVED' for flag in final_flags])
        
#     # 2. Flash a success message once the spinner finishes
#     st.success("✅ Scan Complete! Anomalies isolated below.")
    
#     # Display the metrics
#     st.subheader("Scan Results")
#     col1, col2 = st.columns(2)
#     col1.metric("Total Invoices Scanned", len(results_df))
#     col2.metric("Anomalies Blocked", sum(final_flags))
    
#     # Show the interactive data tables using UI Tabs
#     tab1, tab2 = st.tabs(["🚨 Action Required (Blocked)", "📋 Full Audit Log"])
    
#     with tab1:
#         blocked_df = results_df[results_df['System_Action'] == '🛑 BLOCKED']
#         st.dataframe(blocked_df, use_container_width=True)
        
#     with tab2:
#         st.dataframe(results_df, use_container_width=True)






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
# 4. Create the file upload button
uploaded_file = st.file_uploader("Upload Raw Invoice Data (CSV)", type="csv")

# --- NEW: External Sample Data Hosting Link ---
st.markdown("""
💡 **Don't have a dataset to test?**  
Since corporate invoice logs contain sensitive information, public datasets are rare. 
We have hosted an anonymized, production-scale sample dataset containing over 300,000 rows to test this engine's capabilities.
""")

sample_data_url = "https://drive.google.com/file/d/1dPG7jN8IlXphkk7VStX0m4gk8navzsIK/view?usp=drive_link"

st.link_button(
    label="📥 Download 300MB Sample Invoice Dataset (External Cloud)", 
    url=sample_data_url,
    use_container_width=False
)
st.caption("Instructions: Click the button above to download the CSV from our cloud storage, then drop it into the file upload zone above.")



if uploaded_file is not None:
    # 1. Wrap the ENTIRE process (reading + predicting) in the spinner
    with st.spinner("⏳ Ingesting data and running AI threat detection... Please wait."):
        
        # Read the raw data 
        raw_df = pd.read_csv(uploaded_file)
        df = raw_df.copy()
        
        # --- Preprocessing to match Training Data Pipeline ---
        num_col = df.select_dtypes(include=["number"]).columns
        df[num_col] = df[num_col].fillna(0)
        text_col = df.select_dtypes(exclude=["number"]).columns
        df[text_col] = df[text_col].fillna("Unknown")

        if "invoice_date" in df.columns:
            df["invoice_date"] = pd.to_datetime(df["invoice_date"])
            df["invoice_month"] = df["invoice_date"].dt.month
            df["invoice_day_of_week"] = df["invoice_date"].dt.dayofweek
            df["is_weekend"] = (df["invoice_day_of_week"] >= 5).astype(int)

        # Drop useless columns just like training
        use_les = ["invoice_id", "supplier_id", "department_id", "image_path", "invoice_date", "fraud_type", "fraud_tags", "explanations", "split", "is_fraud"]
        existing_drop_cols = [col for col in use_les if col in df.columns]
        df.drop(columns=existing_drop_cols, inplace=True)
        
        # One-hot encode the text columns
        df = pd.get_dummies(df)
        
        # --- Align Columns with Model Expected Features ---
        model_features = xgb_model.feature_names_in_
        
        # Add missing columns that the model expects but aren't in this CSV
        for col in model_features:
            if col not in df.columns:
                df[col] = 0
                
        # Reorder columns to match the exact sequence the model expects
        X = df[model_features]
            
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
            
        # Attach results to the original raw dataframe so it's readable by humans
        results_df = raw_df.copy()
        results_df.insert(0, 'Fraud_Risk_%', (probabilities * 100).round(2))
        results_df.insert(1, 'System_Action', ['🛑 BLOCKED' if flag == 1 else '✅ APPROVED' for flag in final_flags])
        
    # 2. Flash a success message once the spinner finishes
    st.success("✅ Scan Complete! Anomalies isolated below.")
    
    # Display the metrics
    st.subheader("Scan Results")
    col1, col2 = st.columns(2)
    col1.metric("Total Invoices Scanned", len(results_df))
    col2.metric("Anomalies Blocked", int(sum(final_flags)))
    
    # Show the interactive data tables using UI Tabs
    tab1, tab2 = st.tabs(["🚨 Action Required (Blocked)", "📋 Full Audit Log"])
    
    with tab1:
        blocked_df = results_df[results_df['System_Action'] == '🛑 BLOCKED']
        st.dataframe(blocked_df, use_container_width=True)
        
    with tab2:
        st.dataframe(results_df, use_container_width=True)
