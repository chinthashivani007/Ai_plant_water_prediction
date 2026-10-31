import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="AI Plant Water Predictor",
    page_icon="🌱",
    layout="wide"
)

# ---------------- TRAIN AI MODEL ----------------

@st.cache_resource
def train_model():
    np.random.seed(42)

    n = 1000

    moisture = np.random.uniform(5, 95, n)
    temperature = np.random.uniform(15, 42, n)
    humidity = np.random.uniform(20, 95, n)

    # Simulated labels for demonstration only.
    # These rules are not validated plant-water requirements.
    needs_water = (
        (moisture < 30) |
        ((moisture < 45) &
         (temperature > 32) &
         (humidity < 40))
    ).astype(int)

    X = pd.DataFrame({
        "Moisture": moisture,
        "Temperature": temperature,
        "Humidity": humidity
    })

    model = DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    )

    model.fit(X, needs_water)

    return model

model = train_model()

# ---------------- WEBSITE DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background-color: #f2f8f1;
}
h1 {
    color: #176b3a;
}
h2, h3 {
    color: #267348;
}
div.stButton > button {
    background-color: #218838;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.title("🌱 AI-Based Plant Water Requirement Prediction")

st.markdown("""
### Smart plant care using Biology and Artificial Intelligence

Enter your plant's environmental conditions to explore
an AI-generated watering prediction.
""")

st.info(
    "Educational demonstration: the current model uses "
    "simulated data. Its predictions are not validated "
    "watering advice for real plants."
)

# ---------------- SIDEBAR ----------------

st.sidebar.header("🌿 Plant Information")

plant = st.sidebar.selectbox(
    "Choose your plant",
    ["Tomato", "Rose", "Mint", "Basil", "Bean", "Money Plant"]
)

st.sidebar.write("Selected plant:", plant)

st.sidebar.markdown("""
### About this project
This project explores how environmental data and
machine learning can support plant care and water
conservation.
""")

# ---------------- USER INPUTS ----------------

st.subheader("🌤️ Enter Environmental Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    moisture = st.slider(
        "💧 Soil moisture (%)",
        min_value=0,
        max_value=100,
        value=35
    )

with col2:
    temperature = st.slider(
        "🌡️ Temperature (°C)",
        min_value=10,
        max_value=45,
        value=28
    )

with col3:
    humidity = st.slider(
        "☁️ Relative humidity (%)",
        min_value=10,
        max_value=100,
        value=50
    )

# ---------------- PREDICTION ----------------

if st.button("🔍 Predict Water Requirement"):

    sample = pd.DataFrame({
        "Moisture": [moisture],
        "Temperature": [temperature],
        "Humidity": [humidity]
    })

    prediction = int(model.predict(sample)[0])

    st.divider()
    st.subheader("🤖 AI Prediction Result")

    if prediction == 1:
        st.warning(
            "💧 The model predicts that watering may be needed."
        )
        st.write(
            "Check the actual soil condition and the selected "
            "plant's care requirements before watering."
        )
    else:
        st.success(
            "🌿 The model predicts that watering may not "
            "be needed yet."
        )
        st.write(
            "Continue monitoring the soil and plant."
        )

    # Display entered measurements
    st.subheader("📊 Environmental Data Summary")

    a, b, c = st.columns(3)

    a.metric("Soil Moisture", f"{moisture}%")
    b.metric("Temperature", f"{temperature} °C")
    c.metric("Humidity", f"{humidity}%")

    # Create chart
    chart_data = pd.DataFrame({
        "Measurement": [
            "Soil Moisture (%)",
            "Temperature (°C)",
            "Humidit
