# 💎 Diamond Price Predictor

A production-ready machine learning pipeline that predicts diamond prices in USD based on physical and quality attributes. The project covers the full ML lifecycle — from EDA and feature engineering to hyperparameter tuning, model evaluation, and deployment via an interactive Streamlit web app.

---

## 📌 Problem Statement

Given a diamond's measurable characteristics, predict its **market price in USD**.  
This is a **supervised regression** problem trained on ~53,000 real-world diamond records.

---

## 🗂️ Project Structure

```
diamond-price-predictor/
│
├── app.py                            # Streamlit web application
├── model/
│    └── best_diamond_model.pkl       # Serialised best-performing model
├── data/
│    └── diamonds.csv                 # Raw dataset (~53k records)
├── notebook/
│    └── diamonds_price_prediction.ipynb   # End-to-end ML notebook
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## 📊 Dataset

| Property       | Detail                        |
|----------------|-------------------------------|
| Source         | [Diamonds dataset](https://www.kaggle.com/datasets/shivam2503/diamonds) |
| Rows           | 53,940                        |
| Features       | 9 (3 categorical, 6 numerical)|
| Target         | `price` (USD)                 |

**Feature descriptions:**

| Feature   | Type        | Description                                              |
|-----------|-------------|----------------------------------------------------------|
| `carat`   | Numerical   | Diamond weight                                           |
| `cut`     | Ordinal     | Cut quality: Fair → Good → Very Good → Premium → Ideal  |
| `color`   | Ordinal     | Color grade: J (worst) → D (best)                       |
| `clarity` | Ordinal     | Clarity grade: I1 → IF (best)                           |
| `depth`   | Numerical   | Total depth % (z / mean(x, y) × 100)                    |
| `table`   | Numerical   | Width of top facet relative to widest point             |
| `x`, `y`, `z` | Numerical | Physical dimensions in mm                           |

---

## 🔬 Methodology

### 1. Data Preprocessing
- Removed **146 duplicate rows**
- Dropped **physically impossible entries** (x / y / z = 0)
- Applied **3×IQR Winsorisation** on numerical features to cap extreme outliers without discarding valid high-value stones

### 2. Feature Engineering
- Engineered `volume = x * y * z` — a single size metric that consolidates three highly collinear dimension features
- Dropped `x`, `y`, `z` to reduce multicollinearity

### 3. Preprocessing Pipelines

| Pipeline              | Encoding          | Scaling       | Applied To                                         |
|-----------------------|-------------------|---------------|----------------------------------------------------|
| `preprocessor_scaled` | OrdinalEncoder    | RobustScaler  | LinearRegression, KNN                              |
| `preprocessor_no_scale` | OrdinalEncoder  | None          | DecisionTree, RandomForest, GradientBoosting, XGBoost, LightGBM |

`RobustScaler` was chosen over `StandardScaler` due to the heavy-tailed price distribution.  
`OrdinalEncoder` preserves the meaningful grade hierarchy of all three categorical features.

### 4. Model Selection
Seven models were benchmarked via **5-fold cross-validation** on the training set:

| Model               | CV R²   |
|---------------------|---------|
| Linear Regression   | ~0.92   |
| KNN Regressor       | ~0.95   |
| Decision Tree       | ~0.96   |
| Random Forest       | ~0.98   |
| Gradient Boosting   | ~0.97   |
| XGBoost             | ~0.98   |
| LightGBM            | ~0.98   |

### 5. Hyperparameter Tuning
`RandomizedSearchCV` (50 iterations, 5-fold CV) was applied to the top-3 performers: **Random Forest**, **XGBoost**, and **LightGBM**.

### 6. Final Evaluation

The best-performing model is **LightGBM (tuned)**, selected by validation R² and evaluated once on the held-out test set:

| Metric | Score     |
|--------|-----------|
| R²     | **> 0.98** |
| RMSE   | ~$535     |
| MAE    | ~$282     |

Residuals are approximately normally distributed with mean ≈ 0, indicating no systematic bias.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+

### Installation

```bash
git clone https://github.com/amr-salah-shaheen/diamond-price-predictor.git
cd diamond-price-predictor
pip install -r requirements.txt
```

### Run the Web App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Re-train the Model

Open and run all cells in the notebook:

```bash
jupyter notebook notebook/diamonds_price_prediction.ipynb
```

This will regenerate `best_diamond_model.pkl`.

---

## 🖥️ Web App

The Streamlit app accepts all diamond characteristics and returns an estimated price in real-time.

**Input fields:**
- Physical dimensions: Carat, Depth %, Table %, x / y / z (mm)
- Quality grades: Cut, Color, Clarity (dropdown selectors)

**Output:** Estimated price in USD with input validation and descriptive error messages.

---

## 🛠️ Tech Stack

| Category          | Libraries                                      |
|-------------------|------------------------------------------------|
| Data Processing   | `pandas`, `numpy`, `scipy`                     |
| Machine Learning  | `scikit-learn`, `xgboost`, `lightgbm`          |
| Web App           | `streamlit`                                    |
| Visualisation     | `matplotlib`, `seaborn`                        |

---
"# diamond-prices-predictor" 
