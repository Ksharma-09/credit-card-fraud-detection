import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
.stApp {background: radial-gradient(ellipse at top left,#17243a 0%,#0b1220 48%,#080d16 100%);color:#e8f0ff}
[data-testid="stSidebar"]{background:#101d30;border-right:1px solid #263e5d}
[data-testid="stMetric"]{background:rgba(20,35,55,.72);border:1px solid #263e5d;padding:12px;border-radius:12px}
</style>
""",unsafe_allow_html=True)
st.title("🛡️ Credit Card Fraud Detection")
st.caption("Machine-learning demo for fraud-risk screening. Not a payment authorization system.")

MODEL_PATH=Path(__file__).with_name("model.pkl")
DATA_PATH=Path(__file__).with_name("live_data.csv")
THRESHOLD=st.sidebar.slider("Fraud alert threshold",0.05,0.95,0.30,0.05)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl is missing from the project folder.")
    return joblib.load(MODEL_PATH)

@st.cache_data(ttl=5)
def load_transactions():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(DATA_PATH)
    except (pd.errors.EmptyDataError, OSError, pd.errors.ParserError):
        return pd.DataFrame()

try:
    model=load_model()
except Exception as exc:
    st.error(f"Model load failed: {exc}")
    st.stop()

df=load_transactions()
feature_names=list(getattr(model,"feature_names_in_",[]))
classes=list(getattr(model,"classes_",[]))
if 1 not in classes:
    st.error(f"Loaded model classes are {classes}; class label 1 (fraud) is unavailable.")
    st.stop()
fraud_index=classes.index(1)
if not feature_names:
    st.error("Model metadata does not contain feature names; cannot safely map transaction columns.")
    st.stop()

missing=[name for name in feature_names if name not in df.columns]
if not df.empty and missing:
    st.warning("Input data is missing expected model features: "+", ".join(missing[:8]))
    st.stop()

# Live/demo stream file contains unlabeled transaction features. Predict safely in batch.
if not df.empty:
    X=df[feature_names].apply(pd.to_numeric,errors="coerce")
    valid=X.notna().all(axis=1)
    X=X.loc[valid]
    if not X.empty:
        probabilities=model.predict_proba(X)[:,fraud_index]
        results=df.loc[X.index].copy()
        results["Fraud probability"]=probabilities
        results["Prediction"]=np.where(probabilities>=THRESHOLD,"🚨 Suspicious","✅ Likely normal")
    else:
        results=pd.DataFrame()
else:
    results=pd.DataFrame()

c1,c2,c3=st.columns(3)
c1.metric("Transactions loaded",f"{len(df):,}")
c2.metric("Flagged transactions",f"{int((results['Fraud probability']>=THRESHOLD).sum()):,}" if not results.empty else "0")
c3.metric("Alert threshold",f"{THRESHOLD:.0%}")
st.caption(f"Dashboard refresh: {datetime.now().strftime('%d %b %Y, %H:%M:%S')} · auto-refresh on Streamlit rerun")

tab1,tab2=st.tabs(["Transaction monitor","Manual risk scoring"])
with tab1:
    if results.empty:
        st.info("No transaction rows available yet. Add rows to live_data.csv or run the data simulator.")
    else:
        view=results.copy()
        if "Time" in view.columns:
            view["Time (dataset seconds)"]=view["Time"]
        view=view.sort_values("Fraud probability",ascending=False)
        st.subheader("Highest-risk transactions")
        st.dataframe(view.head(100),use_container_width=True,hide_index=True)
        st.download_button("Download scored transactions (CSV)",view.to_csv(index=False).encode("utf-8"),
                          file_name="scored_transactions.csv",mime="text/csv")
        hist=results["Fraud probability"].clip(0,1)
        st.subheader("Risk probability distribution")
        st.bar_chart(hist.value_counts(bins=10,sort=False).sort_index())
with tab2:
    st.write("Enter the 30 features used by the trained model. Values default to zero; adjust based on a properly transformed transaction.")
    with st.form("manual_transaction"):
        vals={}
        cols=st.columns(3)
        for i,name in enumerate(feature_names):
            with cols[i%3]:
                vals[name]=st.number_input(name,value=0.0,format="%.6f",key=f"manual_{name}")
        submitted=st.form_submit_button("Analyze transaction")
    if submitted:
        sample=pd.DataFrame([vals],columns=feature_names)
        prob=float(model.predict_proba(sample)[0,fraud_index])
        decision="🚨 Suspicious / potential fraud" if prob>=THRESHOLD else "✅ Below alert threshold"
        st.metric("Estimated fraud probability",f"{prob:.2%}")
        st.subheader(decision)
        st.caption("Threshold is configurable. This score is a model output and requires human/operational review.")

st.warning("Dataset/model are historical demonstration artifacts. Predictions can be wrong; do not use this dashboard alone to block real transactions.")
