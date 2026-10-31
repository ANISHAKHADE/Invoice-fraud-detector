

# 🛡️ Enterprise Procurement Fraud Detector

An end-to-end machine learning pipeline built to detect procurement fraud and anomalous invoice submissions. 

## Architecture
* **Supervised Engine**: XGBoost classifier tuned via RandomizedSearchCV for high recall on known fraud patterns.
* **Unsupervised Safety Net**: Isolation Forest algorithm deployed as a "Strict Hybrid Mode" toggle to catch anomalies and data patterns that bypass standard probability thresholds.
* **Frontend**: Interactive Streamlit dashboard allowing business users to process batch invoices (up to 300,000 rows locally) and isolate blocked threats instantly.

## System in Action

**Standard XGBoost Detection:**
<img width="3168" height="1750" alt="Xgboost" src="https://github.com/user-attachments/assets/92243760-5c10-4c44-a7aa-5157b9c24f93" />

**Strict Hybrid Mode Activated (Isolation Forest catching  threats):**
<img width="3170" height="1726" alt="Hybrid" src="https://github.com/user-attachments/assets/dc72a108-7b40-4421-bd1c-b0ae5ec9db3e" />

## Tech Stack
* **Language/Libraries:** Python, pandas, scikit-learn, XGBoost, joblib
* **Deployment:** Streamlit, Git

## Results (60,000-invoice held-out test set, synthetic Kaggle data)
|           Mode              | Precision | Recall | False alarms |
| --------------------------- | --------- | ------ | ------------ |
| XGBoost only (default)      | 97.7%     | 93.1%  |    288       |
| Hybrid (+ Isolation Forest) | 88.7%     | 93.3%  |   1,588      |

The hybrid mode catches 23 more fraud cases but adds ~1,300 false alarms, so it is off by default.
Best single rule (split-invoice flag): 62.8% recall. XGBoost ROC-AUC: 0.962.