

# 🛡️ Enterprise Procurement Fraud Detector

An end-to-end machine learning pipeline built to detect zero-day procurement fraud and anomalous invoice submissions. 

## Architecture
* **Supervised Engine**: XGBoost Classifier tuned via RandomizedSearchCV to catch known fraud patterns, optimized for a strict business threshold (30% probability) to balance False Positives and False Negatives.
* **Unsupervised Safety Net**: Isolation Forest algorithm deployed as a "Strict Hybrid Mode" toggle to catch zero-day anomalies and data patterns that bypass standard probability thresholds.
* **Frontend**: Interactive Streamlit dashboard allowing business users to process batch invoices (up to 300,000 rows locally) and isolate blocked threats instantly.

## System in Action

**Standard XGBoost Detection:**
<img width="3168" height="1750" alt="Xgboost" src="https://github.com/user-attachments/assets/92243760-5c10-4c44-a7aa-5157b9c24f93" />

**Strict Hybrid Mode Activated (Isolation Forest catching zero-day threats):**
<img width="3170" height="1726" alt="Hybrid" src="https://github.com/user-attachments/assets/dc72a108-7b40-4421-bd1c-b0ae5ec9db3e" />

## Tech Stack
* **Language/Libraries:** Python, pandas, scikit-learn, XGBoost, joblib
* **Deployment:** Streamlit, Git

