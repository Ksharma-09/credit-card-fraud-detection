import pandas as pd
import joblib
import time
import datetime
from datetime import datetime
import pandas as pd
import os
# Load mode
model = joblib.load("model.pkl")

print("Model loaded...")

import random
import time
from datetime import datetime

        

while True:
    
    try:
    
        df = pd.read_csv("live_data.csv")

        
        if df.empty:
            print("waiting for  data")
            time.sleep(2)
            continue
        """if len(df)>last_index:
            new_rows=df.iloc[last_index]=len(df)
            for _, row in new_rows.iterrows():
                row_df=pd.DataFrame([row])""" 
        row=df.tail(1)

        
        prob=model.predict_proba(row)[0][1]
        
        
        prediction=1 if prob>0.3 else 0
        print(f"prob:{prob:2f}|prediction;{prediction}")
        # Save result
        pd.DataFrame([[prob,prediction]]).to_csv("output.csv",mode='a',header=False,index=False)
        pd.DataFrame([[prob,prediction]]).to_csv("prediction.csv",mode='a',header=False,index=False)
        log=(f"{datatime.datetime.now()}|prob:{prob:2f}|pred:{prediction}")
        try:
            with open("logs.csv","a")as f:
                f.write(log+"\n")
        except:
            print("log write error")
            
        time.sleep(2)

    except Exception as e:
        print("Waiting for data...")
        time.sleep(2)
        
