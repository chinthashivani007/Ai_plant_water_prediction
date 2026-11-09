import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from datetime import date, timedelta


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="SmartPlant AI",
    page_icon="🌱",
    layout="wide"
)


# ==========================================
# AI MODEL
# ==========================================

@st.cache_resource
def train_model():

    np.random.seed(42)

    n = 1000

    moisture = np.random.uniform(5, 95, n)
    temperature = np.random.uniform(15, 42, n)
    humidity = np.random.uniform(20, 95, n)

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


# ==========================================
# SESSION STORAGE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

if "reminders" not in st.session_state:
    st.session_state.reminders = []


# ==========================================
# PLANT DATABASE
# ==========================================

plant_data = {

    "rose": {
        "type": "Flowering Plant",
        "soil": "Well-drained fertile soil",
        "sunlight": "Around 6 hours of direct sunlight",
        "watering": "Water when the upper soil begins to dry",
        "care": "Remove dry leaves and provide good airflow.",
        "small": (100, 250),
        "medium": (250, 600),
        "large": (600, 1200)
    },

    "tulsi": {
        "type": "Medicinal Plant",
        "soil": "Well-drained soil",
        "sunlight": "Bright sunlight",
        "watering": "Water when the upper soil begins to dry",
        "care": "Avoid waterlogging and provide sunlight.",
        "small": (100, 200),
        "medium": (200, 400),
        "large": (400, 800)
    },

    "aloe vera": {
        "type": "Succulent Plant",
        "soil": "Sandy, well-draining soil",
        "sunlight": "Bright indirect or suitable direct light",
        "watering": "Allow soil to dry thoroughly between watering",
        "care": "Avoid overwatering.",
        "small": (30, 100),
        "medium": (75, 200),
        "large": (150, 350)
    },

    "money plant": {
        "type": "Indoor Plant",
        "soil": "Loose, well-draining soil",
        "sunlight": "Bright indirect light",
        "watering": "Water when the upper soil becomes dry",
        "care": "Avoid standing water.",
        "small": (75, 150),
        "medium": (150, 350),
        "large": (300, 700)
    },

    "tomato": {
        "type": "Vegetable Plant",
        "soil": "Fertile, well-drained soil",
        "sunlight": "Full sunlight",
        "watering": "Maintain consistent soil moisture",
        "care": "Avoid irregular watering and monitor pests.",
        "small": (150, 300),
        "medium": (300, 700),
        "large": (700, 1500)
    },

    "mango": {
        "type": "Fruit Tree",
        "soil": "Well-drained soil",
        "sunlight": "Full sunlight",
        "watering": "Depends on tree age, root area and soil moisture",
        "care": "Avoid waterlogging.",
        "small": (250, 500),
        "medium": (500, 1000),
        "large": (1000, 2500)
    },

    "hibiscus": {
        "type": "Flowering Plant",
        "soil": "Fertile, well-drained soil",
        "sunlight": "Several hours of sunlight",
        "watering": "Keep soil appropriately moist",
        "care": "Remove dry flowers and leaves.",
        "small": (100, 250),
        "medium": (250, 600),
        "large": (600, 1200)
    },

    "sunflower": {
        "type": "Flowering Plant",
        "soil": "Well-drained fertile soil",
        "sunlight": "Full sunlight",
        "watering": "Water according to soil moisture",
        "care": "Provide sufficient sunlight and space.",
        "small": (150, 300),
        "medium": (300, 700),
        "large": (700, 1500)
    },

    "neem": {
        "type": "Tree",
        "soil": "Well-drained soil",
        "sunlight": "Full sunlight",
        "watering": "Young plants need more monitoring than established trees",
        "care": "Avoid excessive watering.",
        "small": (200, 400),
        "medium": (400, 900),
        "large": (900, 2000)
    },

    "jasmine": {
        "type": "Flowering Plant",
        "soil": "Well-drained fertile soil",
        "sunlight": "Bright sunlight",
        "watering": "Water when the upper soil starts drying",
        "care": "Prune when needed.",
        "small": (100, 200),
        "medium": (200, 500),
        "large": (500, 1000)
    }
}


# ==========================================
# WATER QUANTITY CALCULATION
# ==========================================

def calculate_water(
    plant,
    pot_size,
    soil_type,
    age,
    temperature
):

    key = plant.strip().lower()

    if key not in plant_data:
        return None

    info = plant_data[key]

    if pot_size == "Small (10-15 cm)":
        low, high = info["small"]

    elif pot_size == "Medium (15-25 cm)":
        low, high = info["medium"]

    elif pot_size == "Large (25-35 cm)":
        low, high = info["large"]

    else:
        return None

    # Illustrative adjustments only.
    # Actual water needs require soil and plant observations.

    if age == "Seedling":
        low *= 0.5
        high *= 0.5

    elif age == "Young Plant":
        low *= 0.75
        high *= 0.75

    if soil_type == "Clay Soil":
        low *= 0.75
        high *= 0.75

    elif soil_type == "Sandy Soil":
        low *= 1.1
        high *= 1.1

    if temperature >= 35:
        st.info(
            "High temperature: check soil moisture "
            "more frequently. Do not automatically "
            "increase watering."
        )

    return round(low), round(high)


# ==========================================
# WEBSITE DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #f1f8f1;
}

h1 {
    color: #176b3a;
    font-weight: bold;
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

[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🌿 SmartPlant AI")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "🌱 Plant Selection",
        "💧 Water Prediction",
        "🌤️ Weather",
        "📅 Watering Schedule",
        "🪴 Plant Care Guide",
        "📊 Analytics",
        "📜 Prediction History",
        "🔔 Reminders"
    ]
)


# ==========================================
# HOME
# ==========================================

if page == "🏠 Home":

    st.title("🌱 SmartPlant AI")

    st.subheader(
        "AI-Based Plant Water Requirement Prediction"
    )

    st.write(
        "A smart plant-care educational application "
        "using biology and artificial intelligence."
    )

    st.info(
        "The machine-learning model is trained using "
        "simulated data. Water quantities are illustrative "
        "starting estimates, not validated irrigation advice."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Predictions",
        len(st.session_state.history)
    )

    c2.metric(
        "Plant Profiles",
        len(plant_data)
    )

    c3.metric(
        "Saved Reminders",
        len(st.session_state.reminders)
    )

    st.divider()

    st.subheader("Application Features")

    st.write("🌱 Plant selection")

    st.write("💧 Water quantity recommendations")

    st.write("🤖 AI water prediction")

    st.write("🌤️ Environmental information")

    st.write("📊 Analytics and prediction history")


# ==========================================
# PLANT SELECTION
# ==========================================

elif page == "🌱 Plant Selection":

    st.title("🌱 Plant Selection")

    plant = st.text_input(
        "Enter Plant Name",
        placeholder="Example: Rose, Tulsi, Mango"
    )

    if plant.strip():

        key = plant.strip().lower()

        if key in plant_data:

            info = plant_data[key]

            st.success(
                f"Plant found: {plant.title()}"
            )

            st.write("**Plant Type:**", info["type"])

            st.write("**Soil:**", info["soil"])

            st.write("**Sunlight:**", info["sunlight"])

            st.write("**Watering:**", info["watering"])

            st.write("**Care:**", info["care"])

        else:

            st.warning(
                "This plant is not yet in our verified "
                "care-information collection. You can still "
                "enter its name in the prediction section, "
                "but plant-specific water estimates are unavailable."
            )


# ==========================================
# WATER PREDICTION
# ==========================================

elif page == "💧 Water Prediction":

    st.title("💧 AI Water Requirement Prediction")

    st.write(
        "Enter plant and environmental information "
        "to explore a demonstration prediction."
    )

    plant = st.text_input(
        "Plant Name",
        placeholder="Example: Rose, Aloe Vera, Tulsi"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        moisture = st.slider(
            "Soil Moisture (%)",
            0, 100, 35
        )

    with col2:

        temperature = st.slider(
            "Temperature (°C)",
            10, 45, 28
        )

    with col3:

        humidity = st.slider(
            "Humidity (%)",
            10, 100, 50
        )

    st.subheader("🪴 Plant Details")

    pot_size = st.selectbox(
        "Pot Size",
        [
            "Small (10-15 cm)",
            "Medium (15-25 cm)",
            "Large (25-35 cm)",
            "Garden Soil"
        ]
    )

    soil_type = st.selectbox(
        "Soil Type",
        [
            "Normal Garden Soil",
            "Sandy Soil",
            "Clay Soil",
            "Potting Mix"
        ]
    )

    age = st.selectbox(
        "Plant Age",
        [
            "Seedling",
            "Young Plant",
            "Mature Plant"
        ]
    )

    location = st.selectbox(
        "Plant Location",
        [
            "Indoor",
            "Outdoor"
        ]
    )

    if st.button("🔍 Predict Water Requirement"):

        if not plant.strip():

            st.warning("Enter a plant name.")

        else:

            sample = pd.DataFrame({

                "Moisture": [moisture],

                "Temperature": [temperature],

                "Humidity": [humidity]

            })

            prediction = int(
                model.predict(sample)[0]
            )

            st.divider()

            st.subheader("🤖 AI Prediction Result")

            st.write(
                f"**Plant:** {plant.title()}"
            )

            if prediction == 1:

                st.warning(
                    "💧 The demonstration model predicts "
                    "that watering may be needed."
                )

            else:

                st.success(
                    "🌿 The demonstration model predicts "
                    "that watering may not be needed yet."
                )

            st.caption(
                "This model is trained on simulated data. "
                "Its output is not a validated real-world "
                "watering decision."
            )

            st.subheader("🚿 Water Quantity Recommendation")

            key = plant.strip().lower()

            if prediction == 1 and moisture < 25:

                if key in plant_data:

                    if pot_size == "Garden Soil":

                        st.info(
                            "Garden plants need a different "
                            "calculation based on root-zone "
                            "area and soil depth. A pot-based "
                            "millilitre estimate is not suitable."
                        )

                    else:

                        result = calculate_water(
                            plant,
                            pot_size,
                            soil_type,
                            age,
                            temperature
                        )

                        if result:

                            low, high = result

                            st.success(
                                f"Estimated starting range: "
                                f"{low}–{high} ml"
                            )

                            st.write(
                                "Apply water slowly and check "
                                "whether the root-zone soil "
                                "becomes appropriately moist."
                            )

                            st.caption(
                                "This is an illustrative estimate "
                                "for a college demonstration, not "
                                "a scientifically validated dose. "
                                "Stop if the soil is already moist "
                                "or water begins draining out."
                            )

                else:

                    st.warning(
                        "No plant-specific water quantity "
                        "is available for this plant yet. "
                        "Please consult a reliable care guide."
                    )

            else:

                st.info(
                    "No additional watering quantity is "
                    "recommended from this demonstration result. "
                    "Check the actual soil condition."
                )

            st.subheader("📊 Environmental Summary")

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

            st.session_state.history.append({

                "Date": str(date.today()),

                "Plant": plant.title(),

                "Moisture": moisture,

                "Temperature": temperature,

                "Humidity": humidity,

                "Pot Size": pot_size,

                "Soil": soil_type,

                "Age": age,

                "Location": location,

                "Prediction": (
                    "Water may be needed"
                    if prediction == 1
                    else "Water may not be needed"
                )

            })


# ==========================================
# WEATHER
# ==========================================

elif page == "🌤️ Weather":

    st.title("🌤️ Weather Information")

    temperature = st.slider(
        "Temperature (°C)",
        10, 45, 28,
        key="weather_temperature"
    )

    humidity = st.slider(
        "Humidity (%)",
        10, 100, 50,
        key="weather_humidity"
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        value=0.0
    )

    if rainfall > 0:

        st.info(
            "Rainfall recorded. Check soil moisture "
            "before watering."
        )

    if temperature > 35:

        st.warning(
            "High temperature. Monitor soil moisture "
            "more frequently."
        )

    if humidity < 35:

        st.warning(
            "Low humidity recorded."
        )

    st.caption(
        "Enter locally observed values. This section "
        "does not fetch live weather data."
    )


# ==========================================
# WATERING SCHEDULE
# ==========================================

elif page == "📅 Watering Schedule":

    st.title("📅 Plant Monitoring Schedule")

    start_date = st.date_input(
        "Start Date",
        date.today()
    )

    activities = [
        "Check soil moisture",
        "Inspect leaves",
        "Check soil moisture",
        "Inspect plant condition",
        "Check soil moisture",
        "Look for pests",
        "Review plant health"
    ]

    schedule = []

    for i in range(7):

        d = start_date + timedelta(days=i)

        schedule.append({

            "Date": d,

            "Day": d.strftime("%A"),

            "Activity": activities[i]

        })

    st.dataframe(
        pd.DataFrame(schedule),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "This is a monitoring schedule. It does not "
        "mean plants must be watered every day."
    )


# ==========================================
# PLANT CARE GUIDE
# ==========================================

elif page == "🪴 Plant Care Guide":

    st.title("🪴 Plant Care Guide")

    selected = st.selectbox(
        "Select Plant",
        list(plant_data.keys())
    )

    info = plant_data[selected]

    st.subheader(selected.title())

    st.write("**Plant Type:**", info["type"])

    st.write("**Soil:**", info["soil"])

    st.write("**Sunlight:**", info["sunlight"])

    st.write("**Watering:**", info["watering"])

    st.write("**Care Tips:**", info["care"])


# ==========================================
# ANALYTICS
# ==========================================

elif page == "📊 Analytics":

    st.title("📊 Analytics Dashboard")

    if st.session_state.history:

        df = pd.DataFrame(
            st.session_state.history
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Predictions",
            len(df)
        )

        c2.metric(
            "Different Plants",
            df["Plant"].nunique()
        )

        c3.metric(
            "Average Moisture",
            f"{df['Moisture'].mean():.1f}%"
        )

        st.subheader("Soil Moisture History")

        fig, ax = plt.subplots()

        ax.plot(
            range(1, len(df) + 1),
            df["Moisture"],
            marker="o"
        )

        ax.set_xlabel("Prediction Number")

        ax.set_ylabel("Moisture (%)")

        ax.set_title("Plant Moisture Measurements")

        st.pyplot(fig)

        plt.close(fig)

        st.dataframe(
            df,
            use_container_width=True
        )

    else:

        st.info(
            "Make predictions to view analytics."
        )


# ==========================================
# HISTORY
# ==========================================

elif page == "📜 Prediction History":

    st.title("📜 Prediction History")

    if st.session_state.history:

        df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        csv = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download History",
            csv,
            "plant_history.csv",
            "text/csv"
        )

        if st.button("Clear History"):

            st.session_state.history = []

            st.rerun()

    else:

        st.info("No prediction history available.")


# ==========================================
# REMINDERS
# ==========================================

elif page == "🔔 Reminders":

    st.title("🔔 Plant Care Reminders")

    reminder_plant = st.text_input
