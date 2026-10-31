"""
Run this in the SAME folder as ml_ready_data.csv, fraud_model.pkl and isolation_forest.pkl:

    python evaluate_fraud.py

It rebuilds the exact same 80/20 split as train_model.py (random_state=42) and prints
XGBoost-only, Isolation-Forest-only and Hybrid metrics. Paste the full output back to Claude.
"""
import time, joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix,
                             roc_auc_score, average_precision_score)

df = pd.read_csv("ml_ready_data.csv")
X, y = df.drop(columns=["is_fraud"]), df["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb = joblib.load("fraud_model.pkl")
iso = joblib.load("isolation_forest.pkl")
X_test = X_test[xgb.feature_names_in_]          # same column order the model was trained on

print(f"\nTOTAL rows: {len(df):,} | train: {len(X_train):,} | test: {len(X_test):,}")
print(f"Fraud rate overall: {y.mean()*100:.2f}% | in test set: {y_test.mean()*100:.2f}% "
      f"({int(y_test.sum()):,} fraud invoices)")

t0 = time.time()
prob = xgb.predict_proba(X_test)[:, 1]
xgb_time = time.time() - t0
iso_flag = (iso.predict(X_test) == -1).astype(int)

def report(name, pred):
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"\n--- {name} ---")
    print(f"precision={precision_score(y_test,pred):.3f}  recall={recall_score(y_test,pred):.3f}  "
          f"f1={f1_score(y_test,pred):.3f}")
    print(f"TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,} | flagged {pred.mean()*100:.1f}% of invoices "
          f"| false-positive rate {fp/(fp+tn)*100:.2f}%")

print(f"\nXGBoost ROC-AUC={roc_auc_score(y_test, prob):.4f}  PR-AUC={average_precision_score(y_test, prob):.4f}")
for th in (0.5, 0.3):
    report(f"XGBoost only (threshold {th})", (prob >= th).astype(int))
report("Isolation Forest only", iso_flag)
report("HYBRID (XGBoost@0.30 OR Isolation Forest)", ((prob >= 0.3).astype(int) | iso_flag))

print(f"\nScoring speed: {len(X_test):,} invoices in {xgb_time:.2f}s (XGBoost predict_proba)")
print("\nTop 10 features:")
print(pd.Series(xgb.feature_importances_, index=xgb.feature_names_in_).sort_values(ascending=False).head(10).round(3).to_string())

# Sanity check for leakage: does ONE rule alone explain most fraud?
for col in ("split_invoice_flag", "blacklisted_flag", "duplicate_invoice_flag"):
    if col in X_test.columns:
        r = recall_score(y_test, X_test[col].astype(int))
        p = precision_score(y_test, X_test[col].astype(int))
        print(f"Rule-only baseline '{col}': precision={p:.3f} recall={r:.3f}")
