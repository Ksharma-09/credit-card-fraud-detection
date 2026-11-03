# Credit Card Fraud Detection Dashboard
A Machine Learning-powered Credit Card Fraud Detection Dashboard built with Python and Streamlit to identify suspicious transactions and assess fraud risk. Features transaction scoring, interactive risk classification, customizable detection thresholds, model evaluation using Precision, Recall, F1-score, and PR-AUC, and data-driven fraud analysis for exploring transaction patterns and detection performance.

## Features
- Batch scoring of the included feature dataset
- Configurable fraud alert threshold
- Risk-ranked transaction table and CSV export
- Probability distribution chart
- Manual scoring form for model input features
- Dark dashboard styling and clear model limitations

## Run locally (Python 3.10/3.11 recommended)
```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# macOS/Linux
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy
Deploy this GitHub repository on Streamlit Community Cloud and set **Main file path** to `app.py`. Dependencies are specified in `requirements.txt`.

## Project files
- `app.py`: recommended dashboard entry point
- `model.pkl`: trained classifier
- `live_data.csv`: sample transaction features (not actual live bank transactions)
- `kan_chan.py`, `pred.py`, `gen2.py`, `project33.py`: legacy experiment scripts

## Important limitations
- The included `live_data.csv` is a static sample dataset. It is **not connected to a bank/payment gateway** and does not stream real transactions.
- The model uses the feature schema embedded in `model.pkl`; manual values should be prepared using the original dataset's transformations.
- Fraud probability is a model estimate, not proof of fraud. False positives and false negatives are possible.
- Do not use this demo alone to authorize, block, or investigate real payments.
