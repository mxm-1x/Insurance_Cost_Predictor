# 🏥 Insurance Cost Predictor

Live Link - https://insurance-cost-forecasting.streamlit.app/

A production-grade machine learning web app that predicts annual medical insurance charges using Ridge Regression, trained on the [Kaggle Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance).

> ⚠️ **Note:** This model is trained on **US insurance data**. All predicted charges are in **USD ($)**. The dataset reflects US healthcare pricing and insurance structures, which differ significantly from other markets.

---

## 📸 Demo

| Input Form | Prediction & Contributions |
|---|---|
| Age, BMI, smoker status, region | Cost estimate, confidence range, feature bars |

---

## 📁 Project Structure

```
insurance-predictor/
│
├── app.py                  # Streamlit application (main entry point)
├── train.py                # Model training script
├── ridge_model.pkl         # Trained Ridge Regression model
├── scaler.pkl              # Fitted StandardScaler (must match training)
├── insurance.csv           # Dataset (downloaded via kagglehub)
│
├── notebooks/
│   └── analysis.ipynb      # EDA, feature engineering, model evaluation
│
├── assets/
│   ├── fig1_distribution.png
│   ├── fig2_bmi_scatter.png
│   ├── fig3_residuals.png
│   └── fig4_actual_vs_pred.png
│
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/insurance_cost_predictor.git
cd insurance-cost-predictor
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn kagglehub matplotlib seaborn
```

### 3. Train the model (generates `ridge_model.pkl` and `scaler.pkl`)
```bash
python train.py
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🧠 ML Pipeline

### Target Transformation
Raw `charges` are right-skewed due to the smoker/non-smoker cost gap. A **log1p transform** is applied during training:

```python
df['charges'] = np.log1p(df['charges'])
```

At inference, predictions are reversed with `np.expm1()` to return dollar values.

### Feature Encoding

| Feature | Encoding |
|---|---|
| `sex` | Label encoded (male=1, female=0) |
| `smoker` | Label encoded (yes=1, no=0) |
| `region` | One-hot encoded, `northeast` dropped as base |
| `bmi_smoker` | Interaction: `bmi × smoker` |
| `age_smoker` | Interaction: `age × smoker` |

All features are scaled with `StandardScaler` before model fitting. The same fitted scaler is saved as `scaler.pkl` and used at inference — **never refit on test/inference data**.

### Why Ridge over Linear Regression?

The interaction features (`bmi_smoker`, `age_smoker`) are products of existing features, introducing **deliberate multicollinearity**. Ridge Regression's L2 penalty distributes weight stably across correlated predictors, resulting in:

- Better generalisation on unseen inputs
- More stable coefficients for explainability
- Higher R² (0.873 vs 0.867) and lower MAE on test set

---

## 📊 Model Performance

| Model | MAE (USD) | R² Score |
|---|---|---|
| Linear Regression | $2,890 | 0.867 |
| **Ridge Regression** | **$2,750** | **0.873** ✅ |

Evaluated on a held-out 20% test split (`random_state=42`).

---

## 🖥️ App Features

### Prediction
- Estimates annual insurance cost in **USD**
- Shows monthly equivalent
- Displays ±12% confidence range based on model residual spread
- Risk badge: Low / Medium / High

### Feature Contributions
Real coefficient-based breakdown — not fake bars. Each bar shows:
```
contribution = coefficient × scaled_feature_value
```
This is the exact linear decomposition of the model's prediction in log-space. Red = raises cost, green = lowers cost.

### What-If Analysis
Automatically computes and shows:
- 💰 Estimated savings if user quits smoking
- ⚖️ Savings if BMI drops by 5 points
- 📅 Projected cost in 10 years (same profile)

### Edge Case Handling
- Warns if BMI is outside expected training range (10–60)
- Gracefully falls back if `scaler.pkl` is missing (with visible warning)
- Tries `ridge_model.pkl` first, falls back to `linear_model.pkl`
- App halts cleanly with an error message if no model file is found

---

## 📌 Key Findings

- **Smoking** is the dominant cost driver — smokers pay ~2.25× more on average
- **BMI × Smoking** interaction is the second-largest contributor, capturing compounding risk
- **Region** has minimal impact compared to individual health and lifestyle factors
- The bimodal distribution in log-charges directly maps to the smoker/non-smoker split

---

## 🔮 Future Improvements

- [ ] Replace Ridge with **XGBoost/LightGBM** to natively capture non-linear interactions
- [ ] Add **k-fold cross-validation** for more robust metric estimates
- [ ] Implement proper **prediction intervals** via bootstrapping or quantile regression
- [ ] Add **SHAP values** for richer, model-agnostic explainability
- [ ] Monitor for **data drift** if deploying in production

---

## 📦 Dependencies

```
streamlit
pandas
numpy
scikit-learn
kagglehub
matplotlib
seaborn
```

---

## 📄 Dataset

**Source:** [Kaggle — mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)  
**Records:** 1,338  
**Features:** age, sex, bmi, children, smoker, region  
**Target:** charges (annual medical insurance cost, USD)  
**License:** Database Contents License (DbCL) v1.0

---

## ⚠️ Disclaimer

This project is built for **educational and portfolio purposes**. Predictions are estimates from a statistical model trained on historical US data and should not be used as actual insurance quotes or financial advice.

---

## 👤 Author

Built as a data science portfolio project demonstrating end-to-end ML: data preprocessing, feature engineering, model selection, evaluation, explainability, and production deployment.
