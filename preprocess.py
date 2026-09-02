import pandas as pd
df=pd.read_csv("master_training_data.csv")
# df.fillna(0, inplace=True)
# print(df.isnull().sum())
num_col=df.select_dtypes(include=["number"]).columns
df[num_col]=df[num_col].fillna(0)
text_col=df.select_dtypes(exclude=["number"]).columns
df[text_col]=df[text_col].fillna("Unknown")

df["invoice_date"]=pd.to_datetime(df["invoice_date"])

df["invoice_month"]=df["invoice_date"].dt.month
df["invoice_day_of_week"]=df["invoice_date"].dt.dayofweek
df["is_weekend"]=(df["invoice_day_of_week"]>=5).astype(int)


use_les=["invoice_id", "supplier_id", "department_id", "image_path","invoice_date", "fraud_type", "fraud_tags", "explanations", "split"]
df.drop(columns=use_les, inplace=True)
df=pd.get_dummies(df)
df.to_csv("ml_ready_data.csv", index=False)
