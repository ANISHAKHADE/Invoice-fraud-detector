

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


## Dataset

**Source:** Procurement Invoice Fraud Dataset(https://www.kaggle.com/datasets/tokelomashile2/procurement-invoice-fraud-dataset) on Kaggle. It is a synthetic / semi-synthetic dataset built for procurement-fraud research and benchmarking.

**What it contains:** about 300,000 invoices linked to suppliers and departments, engineered behavioural features, rule-based fraud labels, invoice-image metadata and a provided train/test split. Roughly 22% of invoices are labelled fraud.

**Features used by the model (174 after encoding):**
- Invoice: amount, submission hour, amount z-score, OCR-extracted total, image-tamper flag
- Behaviour: supplier invoice count (30 days), supplier average amount (90 days), duplicate-invoice flag, split-invoice flag, late-night-submission flag
- Supplier / department: supplier age, supplier risk score, blacklisted flag, average invoice amount, annual budget
- Categorical (one-hot encoded): currency, payment terms, invoice type, supplier country, region
- Date features derived from the invoice date: month, day of week, weekend flag

**How the data flows through this repo**

| Step | Script              | What it does                                                                                                                              | Output                                    |
|------|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|
| 1    | `merge_data.py`     | Joins the dataset tables (invoices, labels, behavioural features, image metadata, suppliers, departments, splits)                         | `master_training_data.csv`                |
| 2    | `preprocess.py`     | Fills missing values, adds date features, drops ID and label-explanation columns, one-hot encodes                                         | `ml_ready_data.csv`                       | 
| 3    | `train_model.py`    | 80/20 split (`random_state=42`), tunes XGBoost with `RandomizedSearchCV` (scored on recall), fits the Isolation Forest, saves both models | `fraud_model.pkl`, `isolation_forest.pkl` |
| 4    | `evaluate_fraud.py` | Reproduces the held-out test metrics for XGBoost only, Isolation Forest only and the hybrid                                               | Printed report                            |
| 5    | `app.py`            | Streamlit dashboard that scans an uploaded CSV                                                                                            | Web app                                   |

The 60,000-invoice test set in the Results table is the 20% hold-out from step 3.

## How to run

Tested on Python 3.12. The pinned versions in `requirements.txt` are the ones the saved models were built with.

### Option A: quick demo (no dataset download needed)

```
git clone https://github.com/ANISHAKHADE/Invoice-fraud-detector.git
cd Invoice-fraud-detector

python -m venv venv
# Windows:      venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501), upload `sample_invoices_2000.csv` (included in this repo), and switch **Strict Hybrid Mode** on and off to compare results. The sample is a 2,000-row slice of the dataset, so it is a smoke test and not the held-out test set.

### Option B: reproduce the training

1. Download the dataset from Kaggle (link above).
2. Put these files, as **CSV**, in the project folder: `invoices.csv`, `labels.csv`, `behavioural_features.csv`, `images_metadata.csv`, `suppliers.csv`, `departments.csv`, `splits.csv`. [Check the format you downloaded. If the files are Parquet, convert each one first with `pd.read_parquet("x.parquet").to_csv("x.csv", index=False)`.]
3. Run the pipeline in order:

```
python merge_data.py       # -> master_training_data.csv
python preprocess.py       # -> ml_ready_data.csv
python train_model.py      # -> fraud_model.pkl, isolation_forest.pkl
python evaluate_fraud.py   # prints precision, recall, false alarms, ROC-AUC
streamlit run app.py
```



## Limitations

- **The data is synthetic and its labels come from rules.** High scores show the model can learn those rules. They do not predict performance on real invoices, where fraud is rarer and messier.
- **Possible label leakage.** Some features (for example the supplier risk score, blacklist flag, duplicate and split flags) may be generated by the same logic that creates the labels. As a check, `evaluate_fraud.py` also scores single rules on their own. The best one, the split-invoice flag, reaches 62.8% recall against 93% for the model.
- **Own split.** The model uses its own random 80/20 split and does not use the dataset's provided `splits.csv`.
- **Hybrid mode trades precision for recall.** It is off by default because it adds about 1,300 false alarms for 23 extra frauds caught.
- **Scale.** The 300,000-row figure refers to local runs, not free hosting.
