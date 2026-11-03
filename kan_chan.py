"""Legacy polling predictor for appended rows in live_data.csv."""
from pathlib import Path
import time
import joblib
import pandas as pd
from datetime import datetime

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "live_data.csv"
OUTPUT_PATH = BASE / "prediction_row.csv"
model = joblib.load(BASE / "model.pkl")
feature_names = list(getattr(model, "feature_names_in_", []))
classes = list(getattr(model, "classes_", []))
if not feature_names:
    raise RuntimeError("Model lacks feature_names_in_; cannot align input safely.")
if 1 not in classes:
    raise RuntimeError(f"Model classes {classes} do not contain fraud label 1.")
fraud_index = classes.index(1)
last_index = 0

print("Polling demo CSV (not a real payment stream)...")
while True:
    try:
        df = pd.read_csv(DATA_PATH)
        missing = [name for name in feature_names if name not in df.columns]
        if missing:
            raise ValueError(f"Input CSV missing model features: {missing}")
        if len(df) > last_index:
            new_rows = df.iloc[last_index:]
            last_index = len(df)
            X = new_rows[feature_names].apply(pd.to_numeric, errors="coerce")
            valid = X.notna().all(axis=1)
            X = X.loc[valid]
            for idx, row in X.iterrows():
                prob = float(model.predict_proba(pd.DataFrame([row], columns=feature_names))[0, fraud_index])
                prediction = int(prob >= 0.30)
                timestamp = datetime.now().strftime("%H:%M:%S")
                amount = new_rows.loc[idx, "Amount"] if "Amount" in new_rows.columns else "N/A"
                print(f"{timestamp} | Amount:{amount} | fraud_probability:{prob:.3f} | prediction:{prediction}")
                pd.DataFrame([[timestamp, amount, prob, prediction]]).to_csv(
                    OUTPUT_PATH, mode="a", header=not OUTPUT_PATH.exists(), index=False,
                    columns=None
                )
        else:
            print("Waiting for new demo rows...")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        print(f"Input read issue: {exc}")
    except Exception as exc:
        print(f"Prediction issue: {exc}")
    time.sleep(2)
