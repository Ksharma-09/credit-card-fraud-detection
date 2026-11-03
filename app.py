import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

# -------------------- Page setup & visual system --------------------
st.set_page_config(
    page_title="Sentinel AI | Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#eaf1ff; --muted:#91a4c4; --panel:#111d32; --line:#263957; }
.stApp { background:radial-gradient(ellipse at 8% 0%,#183b60 0%,#101b30 38%,#080e1a 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
h1,h2,h3 {font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.035em;}
h1 {font-size:2.35rem!important;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#101d33,#0b1323);border-right:1px solid #263957;}
[data-testid="stMetric"] {background:linear-gradient(135deg,rgba(25,45,73,.95),rgba(15,26,45,.95));border:1px solid #2b4568;padding:16px 18px;border-radius:16px;box-shadow:0 8px 28px rgba(0,0,0,.15);}
[data-testid="stMetricLabel"] p {color:#9bb2d5!important;font-weight:600;}
[data-testid="stMetricValue"] {color:#f4f8ff!important;font-family:'Space Grotesk',sans-serif;}
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button {border:0;border-radius:10px;background:linear-gradient(90deg,#5b7cfa,#9a64f5);color:white;font-weight:700;transition:.2s;}
.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover {filter:brightness(1.12);transform:translateY(-1px);}
div[data-baseweb="tab-list"] {gap:8px;border-bottom:1px solid #293c5b;}
button[data-baseweb="tab"] {background:#121f34;border-radius:10px 10px 0 0;padding:12px 18px;color:#a8bbd8;}
button[data-baseweb="tab"][aria-selected="true"] {background:#243c62;color:#fff;border-bottom:2px solid #8d78ff;}
div[data-testid="stDataFrame"] {border:1px solid #2a3d5c;border-radius:12px;overflow:hidden;}
div[data-testid="stAlert"] {border-radius:12px;}
.hero {padding:24px 26px;border:1px solid #2b4568;border-radius:20px;background:linear-gradient(110deg,rgba(27,58,95,.95),rgba(35,29,72,.78));margin-bottom:18px;}
.hero-kicker {color:#8fafff;text-transform:uppercase;letter-spacing:.16em;font-size:.76rem;font-weight:700;}
.hero-title {font-size:2rem;font-weight:700;line-height:1.2;margin:8px 0;}
.hero-sub {color:#b6c8e5;font-size:1rem;}
.pill {display:inline-block;padding:5px 10px;border-radius:99px;background:#173e3c;color:#8ef0c1;border:1px solid #28685c;font-size:.78rem;font-weight:700;}
.section-note {color:#96aac9;font-size:.88rem;}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = Path(__file__).with_name("model.pkl")
DATA_PATH = Path(__file__).with_name("live_data.csv")

with st.sidebar:
    st.markdown("## 🛡️ Sentinel AI")
    st.caption("FRAUD INTELLIGENCE CONSOLE")
    st.divider()
    threshold = st.slider("🚨 Alert sensitivity", 0.05, 0.95, 0.30, 0.05,
                          help="Lower threshold flags more transactions; higher threshold is stricter.")
    st.markdown("### Navigation")
    st.caption("Use the tabs in the main workspace.")
    st.divider()
    st.markdown("🟢 **Model status:** loaded at runtime")
    st.caption("Demo analytics · Not a payment authorization system")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl is missing from the project folder.")
    return joblib.load(MODEL_PATH)

@st.cache_data(ttl=10)
def load_transactions():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(DATA_PATH)
    except (pd.errors.EmptyDataError, OSError, pd.errors.ParserError):
        return pd.DataFrame()

try:
    model = load_model()
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()

df = load_transactions()
feature_names = list(getattr(model, "feature_names_in_", []))
classes = list(getattr(model, "classes_", []))
if 1 not in classes:
    st.error(f"Fraud class label 1 is unavailable. Model classes: {classes}")
    st.stop()
fraud_index = classes.index(1)
if not feature_names:
    st.error("Model feature metadata is missing; safe feature mapping is not possible.")
    st.stop()

missing = [name for name in feature_names if name not in df.columns]
if not df.empty and missing:
    st.error("Dataset schema mismatch. Missing model features: " + ", ".join(missing[:12]))
    st.stop()

# Score only finite, numeric feature rows. Keep the corresponding source records.
if not df.empty:
    X = df[feature_names].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    X_valid = X.loc[finite_mask]
    source_valid = df.loc[X_valid.index].copy()
    if not X_valid.empty:
        probabilities = model.predict_proba(X_valid)[:, fraud_index]
        results = source_valid.copy()
        results["Fraud probability"] = probabilities
        results["Risk level"] = pd.cut(
            probabilities, bins=[-0.000001, threshold, min(1.000001, threshold + (1-threshold)*0.45), 1.000002],
            labels=["🟢 Low", "🟠 Medium", "🔴 High"], include_lowest=True
        )
        results["Decision"] = np.where(probabilities >= threshold, "🚨 Flag for review", "✓ Below threshold")
    else:
        results = pd.DataFrame()
else:
    results = pd.DataFrame()

total = len(df)
scored = len(results)
flagged = int((results["Fraud probability"] >= threshold).sum()) if scored else 0
flag_rate = flagged / scored * 100 if scored else 0
avg_risk = float(results["Fraud probability"].mean()) if scored else 0.0
normal_count = scored - flagged

# -------------------- Workspace header --------------------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">✦ Intelligent transaction protection</div>
  <div class="hero-title">Fraud Intelligence Dashboard</div>
  <div class="hero-sub">Machine-learning risk signals · Transaction monitoring · Explainable review workflow</div>
  <div style="margin-top:14px"><span class="pill">● DEMO ENVIRONMENT</span></div>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("📦 Transactions ingested", f"{total:,}", help="Rows available in the bundled demonstration CSV.")
k2.metric("🔎 Transactions scored", f"{scored:,}", help="Rows with complete, finite model features.")
k3.metric("🚨 Flagged for review", f"{flagged:,}", delta=f"{flag_rate:.2f}% of scored" if scored else None,
          delta_color="inverse")
k4.metric("📈 Mean model risk", f"{avg_risk:.2%}", help="Average predicted probability for fraud class 1.")

st.caption(f"Last refreshed: {datetime.now().strftime('%d %b %Y · %H:%M:%S')}  |  Current alert threshold: {threshold:.0%}")
overview, monitor, scorer, about = st.tabs(["✨ Overview", "🔍 Transaction Explorer", "🧪 Risk Simulator", "🧠 Model & Insights"])

with overview:
    st.subheader("Your risk snapshot")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("#### 📊 Risk distribution")
        if scored:
            bucket_counts = pd.cut(results["Fraud probability"], bins=[0, .1, .3, .6, 1.000001],
                                 labels=["Very low · 0–10%", "Low · 10–30%", "Elevated · 30–60%", "Critical · 60%+"],
                                 include_lowest=True).value_counts(sort=False)
            st.bar_chart(bucket_counts, color="#8b7cf6", use_container_width=True)
        else:
            st.info("No valid rows have been scored yet.")
    with right:
        st.markdown("#### 🧭 Screening outcome")
        outcome = pd.DataFrame({"Outcome":["Below threshold","Flagged for review"],
                               "Transactions":[normal_count,flagged]}).set_index("Outcome")
        st.bar_chart(outcome, color="#38d6b4", use_container_width=True)
        st.markdown("#### ⚙️ Current screening rule")
        st.write(f"Transactions at or above **{threshold:.0%}** probability are flagged for human review.")
    st.markdown("#### ⚡ Quick insights")
    if scored:
        highest = results["Fraud probability"].max()
        st.info(f"Highest predicted risk in this sample: **{highest:.2%}**. Review flagged rows before drawing conclusions.")
        if "Amount" in results.columns:
            suspicious = results.loc[results["Fraud probability"] >= threshold, "Amount"]
            st.write(f"💳 Flagged transaction amount total (sample): **{suspicious.sum():,.2f}**")
    else:
        st.warning("The dataset is empty or no rows have all required finite features.")

with monitor:
    st.subheader("🔍 Transaction Explorer")
    st.markdown('<p class="section-note">Filter, sort, inspect and export model-scored transactions.</p>', unsafe_allow_html=True)
    if results.empty:
        st.info("No scored transactions available. Check that live_data.csv is present and has the model features.")
    else:
        f1, f2, f3 = st.columns([1,1,1.3])
        with f1:
            risk_filter = st.selectbox("Risk group", ["All", "Flagged only", "Below threshold"])
        with f2:
            min_prob = st.slider("Minimum probability", 0.0, 1.0, 0.0, 0.05)
        with f3:
            sort_order = st.selectbox("Sort by", ["Highest risk first", "Lowest risk first", "Largest amount first"] if "Amount" in results.columns else ["Highest risk first", "Lowest risk first"])
        view = results.copy()
        if risk_filter == "Flagged only":
            view = view[view["Fraud probability"] >= threshold]
        elif risk_filter == "Below threshold":
            view = view[view["Fraud probability"] < threshold]
        view = view[view["Fraud probability"] >= min_prob]
        if sort_order == "Lowest risk first":
            view = view.sort_values("Fraud probability", ascending=True)
        elif sort_order == "Largest amount first" and "Amount" in view.columns:
            view = view.sort_values("Amount", ascending=False)
        else:
            view = view.sort_values("Fraud probability", ascending=False)
        st.write(f"Showing **{len(view):,}** transaction(s)")
        st.dataframe(view.head(500), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export filtered scored CSV", view.to_csv(index=False).encode("utf-8"),
                          file_name="sentinel_scored_transactions.csv", mime="text/csv")
        st.caption("Export is limited to the currently filtered view (up to 500 rows displayed; CSV exports all filtered rows).")

with scorer:
    st.subheader("🧪 Manual transaction risk simulator")
    st.markdown('<p class="section-note">Input the model’s transformed feature values. Defaults are zero; this is a demonstration, not a real card validation.</p>', unsafe_allow_html=True)
    with st.form("manual_transaction"):
        values = {}
        cols = st.columns(3)
        for i, name in enumerate(feature_names):
            with cols[i % 3]:
                values[name] = st.number_input(name, value=0.0, format="%.6f", key=f"sim_{name}")
        submitted = st.form_submit_button("⚡ Run risk analysis", use_container_width=True)
    if submitted:
        sample = pd.DataFrame([values], columns=feature_names)
        try:
            prob = float(model.predict_proba(sample)[0, fraud_index])
            st.progress(min(max(prob, 0.0), 1.0), text=f"Estimated fraud probability: {prob:.2%}")
            if prob >= threshold:
                st.error("🚨 HIGHER RISK — Flag for additional review")
            else:
                st.success("🟢 BELOW CURRENT THRESHOLD — Still review in context")
            st.metric("Predicted fraud probability", f"{prob:.2%}")
            st.caption("Model scores are not proof of fraud. Use independent review and appropriate controls.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with about:
    st.subheader("🧠 Model & data quality")
    m1, m2, m3 = st.columns(3)
    m1.metric("Model type", type(model).__name__)
    m2.metric("Input features", f"{len(feature_names)}")
    m3.metric("Class labels", str(classes))
    st.markdown("#### Feature schema")
    st.dataframe(pd.DataFrame({"Feature": feature_names, "Input type": ["Numeric"]*len(feature_names)}),
                 use_container_width=True, hide_index=True, height=320)
    st.markdown("#### Data quality summary")
    if not df.empty:
        raw_numeric = df[feature_names].apply(pd.to_numeric, errors="coerce")
        missing_cells = int(raw_numeric.isna().sum().sum())
        nonfinite_cells = int((~np.isfinite(raw_numeric.to_numpy(dtype=float))).sum())
        q1, q2, q3 = st.columns(3)
        q1.metric("Missing / non-numeric cells", f"{missing_cells:,}")
        q2.metric("Infinite values", f"{nonfinite_cells:,}")
        q3.metric("Rows excluded from scoring", f"{total-scored:,}")
    else:
        st.warning("No dataset loaded.")

st.divider()
st.markdown('<p class="section-note">⚠️ Educational demonstration using historical/sample artifacts. This is not a live bank feed, payment processor, or production fraud-control system. Model predictions may be inaccurate and must not be used alone to block transactions.</p>', unsafe_allow_html=True)
