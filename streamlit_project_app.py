import pickle
import pandas as pd
import streamlit as st

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Predicted Sales Quantity",
    layout="wide"
)

st.title("📊 Predicted Sales Quantity")
st.write(
    "This app predicts the quantity of products needed for a specific day, month, and year."
)

# -----------------------------
# Load model
# -----------------------------
model_data = pickle.load(open("predicted_sales_model.pkl", "rb"))

model = model_data["model"]
features = model_data["features"]
brand_options = model_data["brand_options"]
material_options = model_data["material_options"]

# -----------------------------
# User inputs
# -----------------------------
st.subheader("Enter prediction details:")

day = st.slider("Day", 1, 31, 1)
month = st.slider("Month", 1, 12, 1)
year = st.number_input("Year", value=2026, step=1)

brand_name = st.selectbox("Brand Name", brand_options)
material_family = st.selectbox("Material Family", material_options)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Quantity"):
    input_df = pd.DataFrame([[
        day,
        month,
        year,
        brand_name,
        material_family
    ]], columns=features)

    prediction = model.predict(input_df)[0]

    st.success(
        f"Predicted product quantity: {round(prediction, 0)} units"
    )

    st.write("Input used for prediction:")
    st.dataframe(input_df)
