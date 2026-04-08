# app.py — Diamond Price Prediction Streamlit App
# Run with: streamlit run app.py
import streamlit as st
import pandas as pd
import joblib

# ── Config ──
MODEL_PATH    = "model/best_diamond_model.pkl"
CUT_ORDER     = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER   = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

NUMERIC_FIELDS = {
    "Carat Weight": "carat",
    "Depth %":      "depth",
    "Table %":      "table",
    "Length (x mm)": "x_dim",
    "Width  (y mm)": "y_dim",
    "Depth  (z mm)": "z_dim",
}

# ── Load Model ──
@st.cache_resource
def load_model():
    artifact = joblib.load(MODEL_PATH)
    if isinstance(artifact, dict):
        return artifact
    return {"model": artifact, "bounds": {}}

# ── Page Setup ──
st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="centered"
)
st.title("💎 Diamond Price Predictor")
st.markdown(
    """
    <h3 style='font-size:18px;'>
    Enter the diamond's characteristics below and click <b>Predict Price</b> to get an estimated market price in USD.
    </h3>
    """,
    unsafe_allow_html=True
)
st.divider()

# ── Input Form ──
col1, col2 = st.columns(2)

with col1:
    st.subheader("Physical Properties")
    carat = st.number_input("Carat Weight",   value=None, step=0.01)
    depth = st.number_input("Depth %",        value=None, step=0.1)
    table = st.number_input("Table %",        value=None, step=0.5)
    x_dim = st.number_input("Length (x mm)", value=None, step=0.01)
    y_dim = st.number_input("Width  (y mm)", value=None, step=0.01)
    z_dim = st.number_input("Depth  (z mm)", value=None, step=0.01)

with col2:
    st.subheader("Quality Grades")
    cut = st.selectbox(
        "Cut",
        options=CUT_ORDER,
        index=None,
        placeholder="Choose...",
    )
    color = st.selectbox(
        "Color",
        options=COLOR_ORDER,
        index=None,
        placeholder="Choose...",
    )
    clarity = st.selectbox(
        "Clarity",
        options=CLARITY_ORDER,
        index=None,
        placeholder="Choose...",
    )

st.divider()

# ── Prediction ──
numeric_values = {
    "Carat Weight":  carat,
    "Depth %":       depth,
    "Table %":       table,
    "Length (x mm)": x_dim,
    "Width  (y mm)": y_dim,
    "Depth  (z mm)": z_dim,
}

grades_missing  = [name for name, val in [("Cut", cut), ("Color", color), ("Clarity", clarity)] if val is None]
numeric_missing = [name for name, val in numeric_values.items() if val is None]
zero_fields     = [name for name, val in numeric_values.items() if val is not None and val <= 0]

if st.button("Predict Price", type="primary", use_container_width=True):
    errors = []

    if numeric_missing:
        errors.append(f"Please fill in: **{', '.join(numeric_missing)}**")

    if zero_fields:
        errors.append(f"Values must be greater than zero: **{', '.join(zero_fields)}**")

    if grades_missing:
        errors.append(f"Please select a value for: **{', '.join(grades_missing)}**")

    if errors:
        for msg in errors:
            st.warning(msg)
    else:
        try:
            volume = x_dim * y_dim * z_dim
            input_df = pd.DataFrame([{
                "carat":   carat,
                "cut":     cut,
                "color":   color,
                "clarity": clarity,
                "depth":   depth,
                "table":   table,
                "volume":  volume,
            }])
            artifact = load_model()
            model = artifact["model"]
            bounds = artifact.get("bounds", {})

            if bounds:
                for col, (lo, hi) in bounds.items():
                    if col in input_df.columns:
                        input_df[col] = input_df[col].clip(lo, hi)

            pred_price = max(0.0, float(model.predict(input_df)[0]))
            st.success(f"Estimated Price: **${pred_price:,.0f}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.caption("Best-performing model trained on ~53k diamonds")