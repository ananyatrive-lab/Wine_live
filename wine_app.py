import streamlit as st
import tensorflow as tf
import numpy as np

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Wine Quality Checker",
    page_icon="🍷",
    layout="wide"
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
<style>
.main {
    background-color: #f7f7f7;
}

h1 {
    color: #8B0000;
    text-align: center;
}

.stButton>button {
    background-color: #8B0000;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size:18px;
}

.stButton>button:hover {
    background-color: #b30000;
    color:white;
}

.footer{
text-align:center;
color:grey;
font-size:15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Title
# ---------------------------
st.title("🍷 Wine Quality Checker")

st.markdown("""
Predict the quality of wine using a trained Deep Learning model.

Fill all the details below and click **Predict Quality**.
""")

# Optional image
try:
    st.image("wine.jpg", use_container_width=True)
except:
    pass

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    """
    **Wine Quality Checker**

    Technologies Used:
    - Streamlit
    - TensorFlow
    - NumPy

    This application predicts wine quality from chemical properties.
    """
)

# ---------------------------
# Load Model
# ---------------------------
model = tf.keras.models.load_model("wine_quality_model.h5")

# ---------------------------
# Input Section
# ---------------------------
st.subheader("Enter Wine Parameters")

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input(
        "Fixed Acidity", 0.0, 20.0, 0.0, 0.1)

    volatile_acidity = st.number_input(
        "Volatile Acidity", 0.0, 5.0, 0.0, 0.1)

    citric_acidity = st.number_input(
        "Citric Acidity", 0.0, 1.0, 0.0, 0.1)

    pH = st.number_input(
        "pH", 0.0, 10.0, 3.0, 0.01)

    residual_sugar = st.number_input(
        "Residual Sugar", 0.0, 15.0, 0.0, 0.1)

    chlorides = st.number_input(
        "Chlorides", 0.0, 1.0, 0.0, 0.01)

with col2:
    free_sulfur_dioxide = st.number_input(
        "Free Sulfur Dioxide", 0.0, 72.0, 0.0, 1.0)

    total_sulfur_dioxide = st.number_input(
        "Total Sulfur Dioxide", 0.0, 289.0, 0.0, 1.0)

    density = st.number_input(
        "Density", 0.0, 1.5, 0.0, 0.001)

    sulphates = st.number_input(
        "Sulphates", 0.0, 2.0, 0.0, 0.01)

    alcohol = st.number_input(
        "Alcohol (%)", 0.0, 15.0, 0.0, 0.1)

# ---------------------------
# Prediction
# ---------------------------
if st.button("🍷 Predict Quality"):

    input_data = np.array([[

        fixed_acidity,
        volatile_acidity,
        citric_acidity,
        pH,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        sulphates,
        alcohol

    ]])

    with st.spinner("Predicting..."):
        prediction = model.predict(input_data)

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