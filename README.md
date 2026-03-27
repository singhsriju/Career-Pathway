# CareerPath AI — Analytics Platform

An AI-powered career guidance analytics platform for Indian students (Grade 8 – Postgraduate).  
Built with Streamlit, scikit-learn, XGBoost, and Plotly.

---

## Project Overview

This platform analyses survey responses from 2,000 synthetic Indian students across 5 behavioral personas to derive:
- **Who** the target student base is (Descriptive)
- **Why** confusion and WTP vary across segments (Diagnostic)
- **Which** students will adopt the platform and at what budget (Predictive)
- **How** to acquire, convert, and price for each segment (Prescriptive)

---

## File Structure

```
career_guidance_platform/
├── app.py                  # Main Streamlit app (7-page dashboard)
├── requirements.txt        # All dependencies with versions
├── data_generator.py       # Synthetic 2000-row dataset generator
├── preprocessing.py        # Shared preprocessing pipeline
├── classification.py       # LR + RF + XGBoost classifiers
├── clustering.py           # K-Means clustering + PCA visualization
├── association_rules.py    # Apriori association rule mining
├── regression.py           # Linear + Ridge + RF regressors
├── prediction_engine.py    # New customer upload + prediction pipeline
├── career_dataset.csv      # Pre-generated dataset (auto-generated if missing)
└── README.md               # This file
```

---

## Run Locally

### Requirements
- Python 3.10 or 3.11 (recommended)
- pip

### Steps

```bash
# 1. Clone or unzip the project
cd career_guidance_platform

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate the dataset (optional — auto-generated on first run)
python data_generator.py

# 5. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Deploy on Streamlit Cloud

1. **Push to GitHub**
   - Create a new public GitHub repository
   - Upload all files from this folder (flat — no sub-folders)
   - Ensure `requirements.txt` and `app.py` are at the root level

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click **New app**
   - Select your GitHub repository
   - Set **Main file path**: `app.py`
   - Click **Deploy**

3. **First run note**
   - On first load, models are trained automatically (~30–60 seconds)
   - All subsequent interactions use `@st.cache_resource` — no retraining
   - The dataset is auto-generated if `career_dataset.csv` is not found

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| Platform Overview | KPI cards, dataset preview, persona distribution |
| Descriptive Analysis | 8 interactive Plotly charts — demographics, features, confusion |
| Diagnostic Analysis | Correlation heatmap, Sankey diagram, feature importance |
| Classification Model | LR + RF + XGBoost — ROC curves, confusion matrix, feature importance |
| Clustering & ARM | K-Means elbow/silhouette, PCA scatter, Apriori rules |
| Regression Analysis | WTP prediction — actual vs predicted, residuals, pricing tiers |
| Prediction Engine | CSV upload → adoption prob + WTP + persona + marketing tag |

---

## ML Models

| Model | Task | Target Variable |
|-------|------|-----------------|
| XGBoost Classifier | Will student use platform? | `F_will_use_binary` |
| Random Forest Regressor | What budget will student pay? | `F_wtp_score_continuous` |
| K-Means (k=5) | Which persona cluster? | Unsupervised |
| Apriori / FP-Growth | Interest–career–feature associations | Transactional |

---

## Student Personas

| Persona | Share | Key Trait |
|---------|-------|-----------|
| Confused Explorer | 25% | High confusion, parent-influenced, low WTP |
| Focused Achiever | 20% | High clarity, self-driven, highest WTP |
| Career Switcher | 15% | Mid-journey, dissatisfied, open to pivot |
| Budget Conscious Learner | 25% | Price-sensitive, Tier 2/3, freemium-first |
| Parent Driven Student | 15% | Low agency, parents decide, B2B channel |

---

## Python Version

Tested on Python **3.10** and **3.11**.  
Not compatible with Python 3.12+ (XGBoost dependency constraint).
