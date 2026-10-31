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

    # Simulated demonstration labels.
    # These are not validated real-world watering rules.
    needs_water = (
        (moisture < 30) |
        (
            (moisture < 45) &
            (temperature > 32) &
            (humidity < 40)
        )
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


# ---------------- WEBSITE TITLE ----------------

st.title("🌱 AI-Based Plant Water Requirement Prediction")

st.markdown("""
### Smart Plant Care Using Biology and Artificial Intelligence

Enter your plant name and environmental conditions
to explore a plant-watering prediction.
""")

st.info(
    "Educational demonstration: this model uses "
    "simulated data. Its predictions have not been "
    "validated as watering advice for real plants."
)


# ---------------- SIDEBAR ----------------

st.sidebar.header("🌿 Plant Information")

plant = st.sidebar.text_input(
    "Enter any plant name",
    placeholder="Example: Mango, Tulsi, Rose"
)

if plant.strip():
    st.sidebar.success(
        f"Selected plant: {plant.strip()}"
    )
else:
    st.sidebar.info(
        "Please enter the name of your plant."
    )

st.sidebar.markdown("""
---
### About This Project

This project demonstrates how environmental
measurements and machine learning can support
plant care and water conservation.

### Features

- Enter a plant name
- Monitor soil moisture
- Check temperature
- Check humidity
- View prediction results
- Explore environmental charts
""")


# ---------------- USER INPUTS ----------------

st.subheader("🌤️ Enter Environmental Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    moisture = st.slider(
        "💧 Soil Moisture (%)",
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
        "☁️ Relative Humidity (%)",
        min_value=10,
        max_value=100,
        value=50
    )


# ---------------- PREDICTION ----------------

if st.button("🔍 Predict Water Requirement"):

    if not plant.strip():

        st.warning(
            "Please enter a plant name before predicting."
        )

    else:

        sample = pd.DataFrame({
            "Moisture": [moisture],
            "Temperature": [temperature],
            "Humidity": [humidity]
        })

        prediction = int(model.predict(sample)[0])

        st.divider()

        st.subheader("🤖 AI Prediction Result")

        st.write(
            f"🌱 Plant entered: **{plant.strip()}**"
        )

        if prediction == 1:

            st.warning(
                "💧 The demonstration model predicts "
                "that watering may be needed."
            )

            st.write(
                "Check the soil and the care requirements "
                "of your particular plant before watering."
            )

        else:

            st.success(
                "🌿 The demonstration model predicts "
                "that watering may not be needed yet."
            )

            st.write(
                "Continue monitoring the soil and "
                "your plant's condition."
            )

        st.caption(
            "This is a simulated prediction. "
            "It does not account for the specific "
            "water requirements of each plant species."
        )

        # ---------------- DATA SUMMARY ----------------

        st.subheader("📊 Environmental Data Summary")

        a, b, c = st.columns(3)

        a.metric(
            "Soil Moisture",
            f"{moisture}%"
        )

        b.metric(
            "Temperature",
            f"{temperature} °C"
        )

        c.metric(
            "Humidity",
            f"{humidity}%"
        )

        # ---------------- CHART ----------------

        chart_data = pd.DataFrame({
            "Measurement": [
                "Soil Moisture (%)",
                "Temperature (°C)",
                "Humidity (%)"
            ],

            "Value": [
                moisture,
                temperature,
                humidity
            ]
        })

        st.subheader("📈 Environmental Measurements")

        fig, ax = plt.subplots(figsize=(9, 4))

        ax.bar(
            chart_data["Measurement"],
            chart_data["Value"]
        )

        ax.set_ylabel("Recorded Value")
        ax.set_title(
            f"Environmental Conditions for {plant.strip()}"
        )

        plt.xticks(rotation=10)
        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


# ---------------- FOOTER ----------------

st.divider()

st.markdown(
    "<center>🌱 AI Plant Water Predictor | "
    "Educational Project</center>",
    unsafe_allow_html=True
)
