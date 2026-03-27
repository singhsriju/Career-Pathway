# CareerPath AI — Analytics Platform

An AI-powered career guidance analytics platform for Indian students (Grade 8 – Postgraduate).
Built with Streamlit, scikit-learn, XGBoost, and Plotly.

---

## Deployment Fix — Python Version

Streamlit Cloud now provisions **Python 3.14** by default. This causes build failures
for pandas, pillow, xgboost, and mlxtend which lack Python 3.14 wheels.

**Fix applied:** `.python-version` file containing `3.11` forces Python 3.11 on Streamlit Cloud.

---

## Deploy on Streamlit Cloud

### Step 1 — Push to GitHub
Upload **all files** from this folder at **root level** (no sub-folders). Ensure `.python-version` is included.

### Step 2 — Deploy
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app** → select repo → Main file: `app.py` → Deploy

---

## Run Locally

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## File Structure

- `app.py` — 7-page Streamlit dashboard
- `requirements.txt` — dependencies (no version pins)
- `.python-version` — pins Python 3.11 for Streamlit Cloud
- `data_generator.py` — 2000-row synthetic dataset
- `preprocessing.py` — shared ML preprocessing pipeline
- `classification.py` — LR + RF + XGBoost classifiers
- `clustering.py` — K-Means + PCA
- `association_rules.py` — Apriori rule mining
- `regression.py` — Linear + Ridge + RF regressors
- `prediction_engine.py` — new student prediction pipeline
- `career_dataset.csv` — pre-generated dataset

---

## Python Version

Requires **Python 3.11**. Not compatible with Python 3.12+ on Streamlit Cloud.
