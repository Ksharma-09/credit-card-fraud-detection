"""Legacy batch predictor. Scores the existing sample once; no infinite polling."""
from pathlib import Path
import pandas as pd
import joblib

BASE=Path(__file__).resolve().parent
model=joblib.load(BASE/"model.pkl")
data=pd.read_csv(BASE/"live_data.csv")
features=list(getattr(model,"feature_names_in_",[]))
if not features:
    raise RuntimeError("Model has no feature_names_in_; cannot safely align live_data.csv.")
missing=[c for c in features if c not in data.columns]
if missing:
    raise ValueError(f"Dataset is missing model features: {missing}")
X=data[features].apply(pd.to_numeric,errors="coerce")
valid=X.notna().all(axis=1)
X=X.loc[valid]
if X.empty:
    print("No valid rows to score.")
else:
    classes=list(model.classes_)
    if 1 not in classes:
        raise RuntimeError(f"Model classes {classes} do not contain fraud label 1.")
    p=model.predict_proba(X)[:,classes.index(1)]
    out=data.loc[X.index].copy()
    out["fraud_probability"]=p
    out["prediction"]=(p>=0.30).astype(int)
    out.to_csv(BASE/"prediction.csv",index=False)
    print(f"Scored {len(out)} rows -> prediction.csv")
