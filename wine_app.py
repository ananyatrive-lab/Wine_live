import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Wine Quality Prediction")

model = joblib.load("wine_quality_model.pkl")

st.title("🍷 Wine Quality Prediction")

fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0)
volatile_acidity = st.number_input("Volatile Acidity", 0.0, 5.0)
citric_acidity = st.number_input("Citric Acidity", 0.0, 2.0)
residual_sugar = st.number_input("Residual Sugar", 0.0, 20.0)
chlorides = st.number_input("Chlorides", 0.0, 2.0)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0, 100.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0.0, 300.0)
density = st.number_input("Density", 0.900, 1.100)
pH = st.number_input("pH", 2.0, 5.0)
sulphates = st.number_input("Sulphates", 0.0, 2.0)
alcohol = st.number_input("Alcohol", 0.0, 20.0)

if st.button("Predict"):

    data = np.array([[

        fixed_acidity,
        volatile_acidity,
        citric_acidity,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol

    ]])

    prediction = model.predict(data)


    pred = np.argmax(prediction) + 3

    st.success(f"Predicted Wine Quality : **{pred}**")

    if pred >= 7:
        st.balloons()
        st.success("Excellent Quality Wine 🍷")

    elif pred >= 5:
        st.info("Average Quality Wine 🍇")

    else:
        st.error("Poor Quality Wine ⚠️")

st.markdown("---")

st.markdown(
    "<div class='footer'>Developed using ❤️ with Streamlit</div>",
    unsafe_allow_html=True
)

