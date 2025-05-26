import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

model = tf.keras.models.load_model("model.h5")
scaler = joblib.load("scaler.pkl")

st.title("🔮 Dự đoán khách hàng rời bỏ ngân hàng")

credit_score = st.slider("Credit Score", 300, 900, 650)
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 35)
tenure = st.slider("Tenure (years)", 0, 10, 3)
balance = st.number_input("Balance", value=10000.0)
products = st.slider("Number of Products", 1, 4, 2)
has_card = st.radio("Has Credit Card?", [1, 0])
is_active = st.radio("Is Active Member?", [1, 0])
salary = st.number_input("Estimated Salary", value=50000.0)

gender_encoded = 1 if gender == "Male" else 0

if st.button("Dự đoán"):
    input_data = np.array([[credit_score, gender_encoded, age, tenure, balance,
                            products, has_card, is_active, salary]])
    input_scaled = scaler.transform(input_data)
    prob = model.predict(input_scaled)[0][0]
    st.success(f"Xác suất khách hàng rời bỏ: {prob:.2%}")
    st.warning("⚠️ Có thể rời bỏ." if prob > 0.5 else "✅ Có khả năng giữ chân.")
