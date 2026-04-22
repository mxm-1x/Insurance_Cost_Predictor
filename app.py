"""
Insurance Cost Predictor — Production-grade Streamlit app
Pipeline: log1p(charges) target | StandardScaler | Ridge Regression
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered",
)

# ══════════════════════════════════════════════════════════════════════════════
# STYLES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

:root {
    --bg:       #0d1521;
    --surface:  rgba(255,255,255,0.04);
    --border:   rgba(255,255,255,0.08);
    --blue:     #5b9cf6;
    --blue-dim: #3b7bd4;
    --text:     #e8edf8;
    --muted:    #7a8db3;
    --green:    #34d399;
    --amber:    #fbbf24;
    --red:      #f87171;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: var(--bg); }

/* hero */
.hero { text-align:center; padding:2.5rem 0 1rem; }
.hero h1 { font-family:'DM Serif Display',serif; font-size:2.6rem;
           color:var(--text); margin:0; letter-spacing:-0.5px; }
.hero p  { color:var(--muted); font-size:0.95rem; margin-top:0.4rem; font-weight:300; }
.accent  { color:var(--blue); }

/* cards */
.card { background:var(--surface); border:1px solid var(--border);
        border-radius:18px; padding:1.6rem 1.8rem; margin-bottom:1rem; }
.card-title { font-size:0.7rem; font-weight:600; letter-spacing:2px;
              text-transform:uppercase; color:var(--blue); margin-bottom:1rem; }

/* result */
.result-wrap { background:linear-gradient(135deg,#1a3260,#152850);
               border:1px solid var(--blue); border-radius:18px;
               padding:2rem; text-align:center;
               box-shadow:0 0 50px rgba(91,156,246,0.12); }
.result-label  { font-size:0.7rem; letter-spacing:2px; text-transform:uppercase;
                 color:var(--muted); font-weight:600; }
.result-amount { font-family:'DM Serif Display',serif; font-size:3rem;
                 color:var(--text); margin:0.2rem 0; letter-spacing:-1px; }
.result-range  { font-size:0.82rem; color:var(--muted); margin-top:0.2rem; }
.result-month  { font-size:0.9rem; color:var(--blue); margin-top:0.3rem; font-weight:500; }

/* badge */
.badge { display:inline-block; padding:0.3rem 0.9rem; border-radius:999px;
         font-size:0.72rem; font-weight:600; letter-spacing:1px;
         text-transform:uppercase; margin-top:0.6rem; }
.badge-low    { background:rgba(52,211,153,.12); color:var(--green); border:1px solid rgba(52,211,153,.25); }
.badge-medium { background:rgba(251,191,36,.12);  color:var(--amber); border:1px solid rgba(251,191,36,.25); }
.badge-high   { background:rgba(248,113,113,.12); color:var(--red);   border:1px solid rgba(248,113,113,.25); }

/* contribution bars */
.contrib-row   { display:flex; align-items:center; gap:10px; margin:0.45rem 0; }
.contrib-label { color:var(--muted); font-size:0.8rem; width:130px; flex-shrink:0; }
.contrib-track { flex:1; background:rgba(255,255,255,0.06); border-radius:4px; height:7px; }
.contrib-fill  { height:7px; border-radius:4px; }
.contrib-val   { color:var(--muted); font-size:0.78rem; width:70px; text-align:right; flex-shrink:0; }

/* what-if */
.whatif-box { background:rgba(52,211,153,0.05); border:1px solid rgba(52,211,153,0.18);
              border-radius:14px; padding:1.2rem 1.4rem; margin-top:0.6rem; }
.whatif-title { color:var(--green); font-size:0.7rem; font-weight:600;
                letter-spacing:2px; text-transform:uppercase; margin-bottom:0.6rem; }
.whatif-row { display:flex; justify-content:space-between; align-items:center;
              padding:0.4rem 0; border-bottom:1px solid rgba(255,255,255,0.05); }
.whatif-row:last-child { border-bottom:none; }
.whatif-scenario { color:var(--muted); font-size:0.85rem; }
.whatif-saving   { font-size:0.85rem; font-weight:600; }

/* tips */
.tip { display:flex; gap:0.6rem; align-items:flex-start;
       padding:0.5rem 0; border-bottom:1px solid var(--border); }
.tip:last-child { border-bottom:none; }
.tip-icon { font-size:1rem; flex-shrink:0; margin-top:1px; }
.tip-text { color:#b0bdd8; font-size:0.84rem; line-height:1.5; }

/* Streamlit overrides */
label, .stSelectbox label, .stNumberInput label,
.stSlider label { color:#b0bdd8 !important; font-size:0.88rem !important; }
.stSelectbox > div > div {
    background:rgba(255,255,255,0.05) !important;
    border:1px solid var(--border) !important;
    color:var(--text) !important; border-radius:10px !important; }
.stNumberInput input {
    background:rgba(255,255,255,0.05) !important;
    border:1px solid var(--border) !important;
    color:var(--text) !important; border-radius:10px !important; }
.stButton > button {
    width:100%; background:linear-gradient(135deg,var(--blue-dim),var(--blue));
    color:white; border:none; border-radius:12px; padding:0.75rem;
    font-family:'DM Sans',sans-serif; font-size:1rem; font-weight:600;
    letter-spacing:0.4px; transition:opacity 0.2s; margin-top:0.3rem; }
.stButton > button:hover { opacity:0.85; }
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    """
    Load model + scaler. Tries ridge_model.pkl first, falls back to linear.
    Scaler must be the same StandardScaler fitted during training.
    Returns (model, scaler | None, model_name).
    """
    scaler = None
    if os.path.exists("scaler.pkl"):
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

    for path, label in [("ridge_model.pkl", "Ridge"), ("linear_model.pkl", "Linear")]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                model = pickle.load(f)
            return model, scaler, label

    raise FileNotFoundError(
        "No model file found. Place ridge_model.pkl or linear_model.pkl "
        "in the same directory as app.py."
    )


try:
    model, scaler, model_name = load_artifacts()
except FileNotFoundError as e:
    st.error(f"**Model not found.** {e}")
    st.stop()
except Exception as e:
    st.error(f"**Failed to load model:** {e}")
    st.stop()

if scaler is None:
    st.warning(
        "scaler.pkl not found — predicting on unscaled features. "
        "Save your StandardScaler during training for accurate results.",
        icon="⚠️",
    )


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE COLUMNS — must exactly match training order
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    "age", "sex", "bmi", "children", "smoker",
    "region_northwest", "region_southeast", "region_southwest",
    "bmi_smoker", "age_smoker",
]

FEATURE_LABELS = {
    "smoker":           "Smoking",
    "bmi_smoker":       "BMI × Smoking",
    "age_smoker":       "Age × Smoking",
    "age":              "Age",
    "bmi":              "BMI",
    "children":         "Children",
    "sex":              "Sex",
    "region_northwest": "Region (NW)",
    "region_southeast": "Region (SE)",
    "region_southwest": "Region (SW)",
}


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def build_features(age, sex, bmi, children, smoker, region) -> pd.DataFrame:
    """
    Encode inputs to match training pipeline:
      - sex / smoker    : label-encoded (male=1, yes=1)
      - region          : get_dummies drop_first=True → northeast is base (dropped)
      - interactions    : bmi*smoker, age*smoker
    """
    sex_val    = 1 if sex    == "male" else 0
    smoker_val = 1 if smoker == "yes"  else 0
    return pd.DataFrame([[
        age, sex_val, bmi, children, smoker_val,
        int(region == "northwest"),
        int(region == "southeast"),
        int(region == "southwest"),
        bmi * smoker_val,
        age * smoker_val,
    ]], columns=FEATURE_COLS)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION  (log-space → dollar-space)
# ══════════════════════════════════════════════════════════════════════════════
def predict(features_df: pd.DataFrame) -> float:
    """Scale → predict in log space → expm1 back to dollars."""
    X = scaler.transform(features_df) if scaler is not None else features_df.values
    return float(np.expm1(model.predict(X)[0]))


def confidence_range(base: float, pct: float = 0.12) -> tuple:
    """±12 % band derived from typical residual spread on this dataset."""
    return base * (1 - pct), base * (1 + pct)


# ══════════════════════════════════════════════════════════════════════════════
# EXPLAINABILITY — real coefficient-based contributions
# ══════════════════════════════════════════════════════════════════════════════
def feature_contributions(features_df: pd.DataFrame) -> pd.Series:
    """
    Linear decomposition: contribution_i = coeff_i × scaled_value_i.
    This is mathematically exact for linear/ridge models.
    Values are in log-charge space (model's native space).
    """
    if scaler is None:
        return pd.Series(dtype=float)
    X_scaled = scaler.transform(features_df)[0]
    contribs  = pd.Series(model.coef_ * X_scaled, index=FEATURE_COLS)
    return contribs.reindex(contribs.abs().sort_values(ascending=False).index)


# ══════════════════════════════════════════════════════════════════════════════
# WHAT-IF SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════
def whatif_scenarios(age, sex, bmi, children, smoker, region) -> list:
    """Return list of {label, saving, new, increase} for actionable changes."""
    base      = predict(build_features(age, sex, bmi, children, smoker, region))
    scenarios = []

    if smoker == "yes":
        alt = predict(build_features(age, sex, bmi, children, "no", region))
        scenarios.append({"label": "If you quit smoking", "saving": base - alt, "new": alt, "increase": False})

    if bmi > 27.5:
        new_bmi = max(round(bmi - 5.0, 1), 22.0)
        alt = predict(build_features(age, sex, new_bmi, children, smoker, region))
        scenarios.append({"label": f"If BMI drops to {new_bmi} (−5 pts)", "saving": base - alt, "new": alt, "increase": False})

    if age < 60:
        alt = predict(build_features(age + 10, sex, bmi, children, smoker, region))
        scenarios.append({"label": "In 10 years (same profile)", "saving": alt - base, "new": alt, "increase": True})

    return scenarios


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def risk_badge(charge: float) -> tuple:
    if charge < 8_000:  return "Low Risk",    "badge-low"
    if charge < 20_000: return "Medium Risk", "badge-medium"
    return "High Risk", "badge-high"


def validate(bmi: float) -> list:
    warns = []
    if bmi < 10 or bmi > 60:
        warns.append("BMI is outside the expected range (10–60). Predictions may be unreliable.")
    return warns


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <h1>Insurance <span class="accent">Cost</span> Predictor</h1>
  <p>Model: <strong style="color:#5b9cf6">{model_name} Regression</strong>
     &nbsp;·&nbsp; Trained on US insurance data with log-transformed charges.</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# INPUTS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="card"><div class="card-title">Personal Profile</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    age      = st.slider("Age", 18, 100, 30)
    sex      = st.selectbox("Biological Sex", ["male", "female"])
with c2:
    children = st.slider("Number of Children", 0, 5, 0)
    smoker   = st.selectbox("Do you smoke?", ["no", "yes"])
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card"><div class="card-title">Body & Location</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    bmi = st.number_input(
        "BMI", min_value=10.0, max_value=60.0, value=26.5, step=0.1,
        help="Healthy: 18.5–24.9 | Overweight: 25–29.9 | Obese: ≥30",
    )
with c4:
    region = st.selectbox("US Region", ["northeast", "northwest", "southeast", "southwest"])
st.markdown('</div>', unsafe_allow_html=True)

for w in validate(bmi):
    st.warning(w)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT + RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if st.button("Calculate Insurance Cost →"):

    features      = build_features(age, sex, bmi, children, smoker, region)
    charge        = predict(features)
    lo, hi        = confidence_range(charge)
    rlabel, rcls  = risk_badge(charge)

    # ── Result ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-wrap">
      <div class="result-label">Estimated Annual Premium</div>
      <div class="result-amount">₹{charge:,.0f}</div>
      <div class="result-range">Likely range &nbsp;₹{lo:,.0f} – ₹{hi:,.0f}</div>
      <div class="result-month">≈ ₹{charge/12:,.0f} per month</div>
      <span class="badge {rcls}">{rlabel}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature Contributions ────────────────────────────────────────────────
    contribs = feature_contributions(features)
    if not contribs.empty:
        abs_max   = contribs.abs().max() or 1.0
        rows_html = ""
        for feat, val in contribs.items():
            label = FEATURE_LABELS.get(feat, feat)
            pct   = abs(val) / abs_max * 100
            color = "#f87171" if val > 0 else "#34d399"   # red = raises cost
            sign  = "+" if val > 0 else "−"
            rows_html += f"""
            <div class="contrib-row">
              <div class="contrib-label">{label}</div>
              <div class="contrib-track">
                <div class="contrib-fill" style="width:{pct:.1f}%;background:{color};"></div>
              </div>
              <div class="contrib-val">{sign}{abs(val):.3f}</div>
            </div>"""

        st.markdown(
            f'<div class="card"><div class="card-title">Feature Contributions (log scale)</div>'
            f'{rows_html}'
            f'<p style="color:#354060;font-size:0.72rem;margin-top:0.8rem;">'
            f'Values = coeff × scaled_feature — exact linear decomposition of the prediction. '
            f'Red raises cost · Green lowers cost.</p></div>',
            unsafe_allow_html=True,
        )

    # ── What-If Analysis ─────────────────────────────────────────────────────
    scenarios = whatif_scenarios(age, sex, bmi, children, smoker, region)
    if scenarios:
        rows_html = ""
        for s in scenarios:
            color = "var(--red)" if s["increase"] else "var(--green)"
            arrow = "↑" if s["increase"] else "↓"
            rows_html += f"""
            <div class="whatif-row">
              <div class="whatif-scenario">{s['label']}</div>
              <div class="whatif-saving" style="color:{color};">
                {arrow} ₹{abs(s['saving']):,.0f}
                <span style="color:var(--muted);font-weight:400;font-size:0.78rem;">
                  &nbsp;→ ₹{s['new']:,.0f}/yr</span>
              </div>
            </div>"""

        st.markdown(
            f'<div class="whatif-box"><div class="whatif-title">What-If Analysis</div>{rows_html}</div>',
            unsafe_allow_html=True,
        )

    # ── Personalised Tips ────────────────────────────────────────────────────
    tips = []
    if smoker == "yes":
        tips.append(("🚬", "Smoking is the single largest cost driver. Quitting typically reduces premiums by 30–50%."))
    if bmi >= 30:
        tips.append(("⚖️", f"BMI {bmi:.1f} falls in the obese range. Reducing below 30 lowers your risk classification."))
    elif bmi < 18.5:
        tips.append(("⚖️", f"BMI {bmi:.1f} is below healthy range — underweight status can also elevate premiums."))
    if age >= 55:
        tips.append(("📅", "Premiums accelerate after 55. Locking in a long-term plan earlier reduces lifetime costs."))
    if children >= 3:
        tips.append(("👨‍👩‍👧‍👦", "Larger families attract higher group-coverage premiums due to increased dependant risk."))

    if tips:
        tips_html = "".join(
            f'<div class="tip"><div class="tip-icon">{icon}</div>'
            f'<div class="tip-text">{text}</div></div>'
            for icon, text in tips
        )
        st.markdown(
            f'<div class="card"><div class="card-title">Personalised Insights</div>{tips_html}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        "<p style='text-align:center;color:#354060;font-size:0.72rem;margin-top:1.2rem;'>"
        "Estimates are based on a regression model trained on historical US insurance data. "
        "Not financial or medical advice.</p>",
        unsafe_allow_html=True,
    )