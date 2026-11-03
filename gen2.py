"""Legacy CSV streamer. Requires the original Kaggle creditcard.csv."""
from pathlib import Path
import time
import pandas as as pd

BASE=Path(__file__).resolve().parent
source=BASE/"creditcard.csv"
if not source.exists():
    raise FileNotFoundError("creditcard.csv is not included. This optional simulator needs the original dataset; live_data.csv is a static demo sample.")
df=pd.read_csv(source)
if "Class" not in df.columns:
    raise ValueError("Expected target column 'Class' in creditcard.csv.")
features=df.drop(columns=["Class"])
features.head(0).to_csv(BASE/"live_data.csv",index=False)
for _,row in features.iterrows():
    row.to_frame().T.to_csv(BASE/"live_data.csv",mode="a",header=False,index=False)
    print("Demo row streamed (not a real transaction)")
    time.sleep(2)
