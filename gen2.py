import pandas as pd
import time
df=pd.read_csv("creditcard.csv")
df=df.drop("Class",axis=1)
df.head(0).to_csv("live_data.csv",index=False)
for i in range(len(df)):
    row=df.iloc[i:i+1]
    row.to_csv("live_data.csv",mode="a",header=False,index=False)
    print("Real transaction streamed")
    time.sleep(2)