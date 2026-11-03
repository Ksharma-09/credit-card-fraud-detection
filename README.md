# 🛡️ Credit Card Fraud Detection Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/Machine%20Learning-Classification-orange)](https://scikit-learn.org/)

A machine-learning dashboard that scores transaction-feature data, surfaces potentially suspicious activity, and helps explore fraud detection thresholds and model performance. Built with Python, Streamlit, Pandas, and scikit-learn.

## Key features
- Batch scoring of transaction feature records
- Adjustable alert threshold for fraud-risk classification
- Risk-ranked transaction table with CSV export
- Fraud-score distribution visualization
- Manual prediction interface for model input features
- Evaluation metrics including Precision, Recall, F1-score, and PR-AUC when labeled data is available
- Dark dashboard interface and data-quality checks

## Tech stack
Python · Streamlit · Pandas · NumPy · scikit-learn · Joblib

## Run locally
Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Sign in at https://share.streamlit.io/
2. Create a new app and select this repository, branch `main`.
3. Set **Main file path** to `app.py`.
4. Deploy; dependencies are installed from `requirements.txt`.

## Model and data
- `model.pkl`: serialized trained classifier
- `live_data.csv`: static sample feature data for demonstration; it is **not live bank/payment data**
- `app.py`: recommended dashboard entry point
- Additional scripts (`kan_chan.py`, `pred.py`, `gen2.py`, `project33.py`) are legacy experiments and are not required to launch the main dashboard

## Validation
GitHub Actions checks Python syntax, verifies model/data assets, and smoke-tests model loading and prediction on sample rows.

## Responsible use and limitations
- A predicted fraud probability is a model estimate, not proof of fraudulent activity.
- False positives and false negatives are possible; threshold choice changes the trade-off.
- The sample dataset is static and does not connect to a bank or payment gateway.
- Manual inputs must follow the feature transformations used during model training.
- Do not use this educational demo alone to approve, block, or investigate real payments.

## Disclaimer
This project is for educational purposes only and is not a production payment-security system.


## Live demo
🚀 **Try the deployed app:** https://credit-card-fraud-detection-5uadfcayhnoptty56ajmmk.streamlit.app/
