"""Legacy training script. Requires the original Kaggle creditcard.csv."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

BASE=Path(__file__).resolve().parent
source=BASE/"creditcard.csv"
if not source.exists():
    raise FileNotFoundError("creditcard.csv is not included. Download the Kaggle dataset separately; do not commit sensitive/raw data.")
df=pd.read_csv(source)
if "Class" not in df.columns:
    raise ValueError("Expected target column 'Class'.")
X=df.drop(columns=["Class"]); y=df["Class"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=42,n_jobs=-1)
model.fit(X_train,y_train)
p=model.predict_proba(X_test)[:,list(model.classes_).index(1)]
print(classification_report(y_test,(p>=.5).astype(int),zero_division=0))
print("ROC-AUC:",roc_auc_score(y_test,p))
joblib.dump(model,BASE/"model.pkl")
print("Saved model.pkl")
