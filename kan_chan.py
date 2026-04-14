import pandas as pd
import time
import joblib
from datetime import datetime

last_index = 0
model = joblib.load("model.pkl")

print("System started...")

while True:
    try:
        # 🔥 real dataset read
        df = pd.read_csv("live_data.csv")

        if df.empty:
            print("Waiting for data...")
            time.sleep(2)
            continue

        
        if len(df) > last_index:

            new_rows = df.iloc[last_index:]
            last_index = len(df)

            for _, row in new_rows.iterrows():

                row_df = pd.DataFrame([row])


                prob = model.predict_proba(row_df)[0][1]
                prediction = 1 if prob > 0.3 else 0

                
                timestamp = datetime.now().strftime("%H:%M:%S")

                
                amount = row.get("Amount", 0)

                print(f"{timestamp} | Amt:{amount} | prob:{prob:.2f} | pred:{prediction}")

            
                pd.DataFrame([[
                    timestamp,
                    amount,
                    prob,
                    prediction
                ]]).to_csv(
                    "prediction_row.csv",
                    mode='a',
                    header=False,
                    index=False
                )

        else:
            print("Waiting for new data...")

    except Exception as e:
        print("Error:", e)

    time.sleep(2)