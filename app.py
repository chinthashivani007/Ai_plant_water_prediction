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
# SESSION HISTORY
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []


# ==========================================
# PLANT INFORMATION
# ==========================================

plant_data = {

    "rose": {
        "soil": "Well-drained, fertile soil",
        "sunlight": "Around 6 hours of direct sunlight",
        "water": "Water when the upper soil begins to dry",
        "care": "Remove dry leaves and maintain good airflow."
    },

    "tulsi": {
        "soil": "Well-drained soil",
        "sunlight": "Bright sunlight",
        "water": "Water when the upper soil begins to dry",
        "care": "Provide sunlight and avoid waterlogging."
    },

    "aloe vera": {
        "soil": "Sandy, well-draining soil",
        "sunlight": "Bright light",
        "water": "Allow soil to dry between watering",
        "care": "Avoid overwatering."
    },

    "money plant": {
        "soil": "Loose, well-draining soil",
        "sunlight": "Indirect bright light",
        "water": "Water when the upper soil becomes dry",
        "care": "Clean leaves and avoid excessive watering."
    },

    "tomato": {
        "soil": "Fertile, well-drained soil",
        "sunlight": "Full sunlight",
        "water": "Maintain consistent soil moisture",
        "care": "Support the plant and monitor pests."
    },

    "mango": {
        "soil": "Well-drained soil",
        "sunlight": "Full sunlight",
        "water": "Water according to soil moisture and growth stage",
        "care": "Avoid waterlogging and maintain healthy soil."
    },

    "hibiscus": {
        "soil": "Fertile, well-drained soil",
        "sunlight": "Several hours of sunlight",
        "water": "Keep soil appropriately moist",
        "care": "Remove dry flowers and leaves."
    },

    "sunflower": {
        "soil": "Well-drained fertile soil",
        "sunlight": "Full sunlight",
        "water": "Water according to soil moisture",
        "care": "Provide enough sunlight and space."
    }
}


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

div.stButton > button:hover {
    background-color: #145c2e;
    color: white;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🌿 SmartPlant AI")

st.sidebar.markdown("### Plant Care System")

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

st.sidebar.divider()

st.sidebar.info(
    "SmartPlant AI helps users explore "
    "plant care using environmental "
    "measurements and machine learning."
)


# ==========================================
# HOME PAGE
# ==========================================

if page == "🏠 Home":

    st.title("🌱 Welcome to SmartPlant AI")

    st.subheader(
        "Intelligent Plant Care & Water Prediction System"
    )

    st.write(
        "Explore plant care, environmental conditions "
        "and educational machine-learning predictions."
    )

    st.info(
        "This application uses simulated training data "
        "for educational demonstration. It does not "
        "provide scientifically validated watering advice."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "🌿 Predictions",
        len(st.session_state.history)
    )

    col2.metric(
        "🪴 Plant Guides",
        len(plant_data)
    )

    col3.metric(
        "📊 Application",
        "Active"
    )

    st.divider()

    st.subheader("✨ Application Features")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 🌱 Plant Selection")
        st.write("Enter plant names and explore care information.")

    with c2:
        st.markdown("### 💧 Water Prediction")
        st.write("Explore environmental-based ML predictions.")

    with c3:
        st.markdown("### 📊 Analytics")
        st.write("Review environmental measurements and history.")

    st.success(
        "Select a section from the sidebar to get started!"
    )


# ==========================================
# PLANT SELECTION
# ==========================================

elif page == "🌱 Plant Selection":

    st.title("🌱 Plant Selection")

    plant = st.text_input(
        "Enter your plant name",
        placeholder="Example: Rose, Mango, Tulsi"
    )

    if plant.strip():

        st.success(
            f"Selected Plant: {plant.strip().title()}"
        )

        key = plant.strip().lower()

        if key in plant_data:

            info = plant_data[key]

            st.subheader("🌿 Plant Information")

            st.write("**Soil:**", info["soil"])

            st.write("**Sunlight:**", info["sunlight"])

            st.write("**Water:**", info["water"])

            st.write("**Care Tips:**", info["care"])

        else:

            st.info(
                "This plant is accepted by the application, "
                "but a verified care profile is not yet available. "
                "Please consult a reliable plant-care source."
            )

    else:

        st.info("Enter a plant name to explore.")


# ==========================================
# WATER PREDICTION
# ==========================================

elif page == "💧 Water Prediction":

    st.title("💧 AI Plant Water Requirement Prediction")

    plant = st.text_input(
        "Plant Name",
        placeholder="Enter any plant name"
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

    if st.button("🔍 Predict Water Requirement"):

        if not plant.strip():

            st.warning("Please enter a plant name.")

        else:

            sample = pd.DataFrame({
                "Moisture": [moisture],
                "Temperature": [temperature],
                "Humidity": [humidity]
            })

            prediction = int(
                model.predict(sample)[0]
            )

            if prediction == 1:

                result = "Water may be needed"

                st.warning(
                    "💧 The demonstration model predicts "
                    "that watering may be needed."
                )

            else:

                result = "Water may not be needed yet"

                st.success(
                    "🌿 The demonstration model predicts "
                    "that watering may not be needed yet."
                )

            st.write(
                f"Selected plant: **{plant.title()}**"
            )

            st.caption(
                "This result comes from simulated training data. "
                "Check actual soil conditions and plant-specific "
                "requirements before watering."
            )

            st.subheader("📊 Environmental Summary")

            a, b, c = st.columns(3)

            a.metric("Soil Moisture", f"{moisture}%")

            b.metric("Temperature", f"{temperature} °C")

            c.metric("Humidity", f"{humidity}%")

            st.session_state.history.append({

                "Date": str(date.today()),

                "Plant": plant.title(),

                "Moisture": moisture,

                "Temperature": temperature,

                "Humidity": humidity,

                "Prediction": result

            })


# ==========================================
# WEATHER SECTION
# ==========================================

elif page == "🌤️ Weather":

    st.title("🌤️ Environmental Conditions")

    st.write(
        "Enter local weather observations to explore "
        "their possible influence on plant watering."
    )

    temperature = st.slider(
        "Temperature (°C)",
        10, 45, 28,
        key="weather_temp"
    )

    humidity = st.slider(
        "Humidity (%)",
        10, 100, 50,
        key="weather_humidity"
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=500.0,
        value=0.0
    )

    if rainfall > 0:

        st.info(
            "Rainfall has been recorded. Check the soil "
            "before adding more water."
        )

    if temperature > 35:

        st.warning(
            "High temperature recorded. Monitor soil "
            "moisture more frequently."
        )

    elif temperature < 20:

        st.info(
            "Lower temperature recorded. Water needs "
            "may differ depending on the plant."
        )

    if humidity < 35:

        st.warning(
            "Low humidity recorded."
        )

    else:

        st.success(
            "Environmental observations recorded."
        )

    st.caption(
        "These are manually entered observations, not live weather data."
    )


# ==========================================
# WATERING SCHEDULE
# ==========================================

elif page == "📅 Watering Schedule":

    st.title("📅 Smart Watering Schedule")

    st.write(
        "Create a weekly plant-monitoring plan."
    )

    plant = st.text_input(
        "Plant Name",
        key="schedule_plant"
    )

    start_date = st.date_input(
        "Start Date",
        date.today()
    )

    st.subheader("🌱 Weekly Monitoring Plan")

    activities = [
        "Check soil moisture",
        "Inspect leaves and soil",
        "Check soil moisture",
        "Review plant condition",
        "Check soil moisture",
        "Inspect for pests",
        "Review plant health"
    ]

    schedule_data = []

    for i in range(7):

        current_date = start_date + timedelta(days=i)

        schedule_data.append({

            "Date": current_date,

            "Day": current_date.strftime("%A"),

            "Activity": activities[i]

        })

    schedule_df = pd.DataFrame(schedule_data)

    st.dataframe(
        schedule_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "This is a monitoring schedule, not a fixed watering "
        "instruction. Water only when the plant's actual "
        "requirements and soil condition indicate it is needed."
    )


# ==========================================
# PLANT CARE GUIDE
# ==========================================

elif page == "🪴 Plant Care Guide":

    st.title("🪴 Plant Care Guide")

    selected = st.selectbox(
        "Choose a plant",
        list(plant_data.keys())
    )

    info = plant_data[selected]

    st.subheader(
        f"🌿 {selected.title()}"
    )

    st.write("### Soil Requirements")

    st.write(info["soil"])

    st.write("### Sunlight Requirements")

    st.write(info["sunlight"])

    st.write("### Water Requirements")

    st.write(info["water"])

    st.write("### Maintenance Tips")

    st.success(info["care"])


# ==========================================
# ANALYTICS
# ==========================================

elif page == "📊 Analytics":

    st.title("📊 Plant Analytics Dashboard")

    if len(st.session_state.history) == 0:

        st.info(
            "Make some predictions to see your analytics."
        )

    else:

        df = pd.DataFrame(
            st.session_state.history
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Predictions",
            len(df)
        )

        col2.metric(
            "Plants Recorded",
            df["Plant"].nunique()
        )

        col3.metric(
            "Average Moisture",
            f"{df['Moisture'].mean():.1f}%"
        )

        st.subheader("📈 Moisture Measurements")

        fig, ax = plt.subplots(figsize=(9, 4))

        ax.plot(
            range(1, len(df) + 1),
            df["Moisture"],
            marker="o"
        )

        ax.set_xlabel("Prediction Number")

        ax.set_ylabel("Soil Moisture (%)")

        ax.set_title("Soil Moisture History")

        st.pyplot(fig)

        plt.close(fig)

        st.subheader("🌡️ Environmental Data")

        st.dataframe(
            df,
            use_container_width=True
        )


# ==========================================
# PREDICTION HISTORY
# ==========================================

elif page == "📜 Prediction History":

    st.title("📜 Prediction History")

    if len(st.session_state.history) == 0:

        st.info(
            "No predictions recorded yet."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Prediction History",
            data=csv,
            file_name="plant_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):

            st.session_state.history = []

            st.rerun()


# ==========================================
# REMINDERS
# ==========================================

elif page == "🔔 Reminders":

    st.title("🔔 Plant Care Reminders")

    st.write(
        "Create a reminder to check your plant."
    )

    reminder_plant = st.text_input(
        "Plant Name",
        key="reminder_plant"
    )

    reminder_date = st.date_input(
        "Reminder Date",
        date.today(),
        key="reminder_date"
    )

    reminder_note = st.text_input(
        "Reminder Activity",
        value="Check soil moisture"
    )

    if st.button("Save Reminder"):

        if reminder_plant.strip():

            if "reminders" not in st.session_state:

                st.session_state.reminders = []

            st.session_state.reminders.append({

                "Plant": reminder_plant.title(),

                "Date": str(reminder_date),

                "Activity": reminder_note

            })

            st.success("Reminder saved!")

        else:

            st.warning("Enter a plant name.")

    if "reminders" in st.session_state:

        st.subheader("📋 Your Reminders")

        for reminder in st.session_state.reminders:

            st.write(
                f"🌱 **{reminder['Plant']}** — "
                f"{reminder['Date']} — "
                f"{reminder['Activity']}"
            )

    st.info(
        "These are in-app reminders. They do not send "
        "automatic phone notifications."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown("""
<center>

🌱 <b>SmartPlant AI</b>

AI-Based Plant Water Requirement Prediction

Educational Biology & Artificial Intelligence Project

</center>
""", unsafe_allow_html=True)
