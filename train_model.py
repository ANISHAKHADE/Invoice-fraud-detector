import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest 
from xgboost import XGBClassifier
import joblib
df=pd.read_csv("ml_ready_data.csv")

X=df.drop(columns=["is_fraud"])
y=df["is_fraud"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2 , random_state=42)


param_dis={
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.03, 0.08, 0.15],
    'subsample': [0.7, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.9, 1.0],
    'scale_pos_weight': [3.0, 3.5, 4.5]
}
base_model = XGBClassifier(random_state=42, eval_metric="logloss")
search=RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dis,
    n_iter=6,
    scoring="recall",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)


search.fit(X_train, y_train)
best_model=search.best_estimator_

probabilities = best_model.predict_proba(X_test)[:,1]
cus_threshold = 0.3
y_pred=(probabilities >= cus_threshold).astype(int)

iso_forest=IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X_train)
if_pred=iso_forest.predict(X_test)
is_fraud_flags=(if_pred==-1).astype(int)
hybrid_pred=(y_pred | is_fraud_flags)

print(classification_report(y_test,hybrid_pred))
print(confusion_matrix(y_test,hybrid_pred))

joblib.dump(best_model,"fraud_model.pkl")
joblib.dump(iso_forest,"isolation_forest.pkl")
