import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, time, timedelta
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------

st.set_page_config(
    page_title="AI Plant Water Predictor",
    page_icon="🌱",
    layout="wide"
)

# ---------------------------------------
# PLANT DATABASE
# Approximate educational estimates.
# Base quantity is for a medium-sized pot.
# ---------------------------------------

plants = {
    "Rose": ("Flowering", 400, "6-8 hours", "Well-drained soil"),
    "Hibiscus": ("Flowering", 500, "6-8 hours", "Loamy soil"),
    "Jasmine": ("Flowering", 400, "4-6 hours", "Well-drained soil"),
    "Sunflower": ("Flowering", 500, "6-8 hours", "Loamy soil"),
    "Marigold": ("Flowering", 300, "6-8 hours", "Well-drained soil"),
    "Lotus": ("Flowering", 700, "Full sunlight", "Aquatic soil"),
    "Lily": ("Flowering", 400, "4-6 hours", "Rich soil"),
    "Petunia": ("Flowering", 350, "6 hours", "Well-drained soil"),
    "Chrysanthemum": ("Flowering", 400, "5-6 hours", "Loamy soil"),
    "Bougainvillea": ("Flowering", 250, "6-8 hours", "Sandy, well-drained soil"),

    "Money Plant": ("Indoor", 250, "Indirect sunlight", "Well-drained soil"),
    "Snake Plant": ("Indoor", 150, "Indirect sunlight", "Sandy soil"),
    "Spider Plant": ("Indoor", 250, "Indirect sunlight", "Loamy soil"),
    "Peace Lily": ("Indoor", 350, "Indirect sunlight", "Moist, well-drained soil"),
    "Rubber Plant": ("Indoor", 300, "Bright indirect sunlight", "Well-drained soil"),
    "Areca Palm": ("Indoor", 400, "Bright indirect sunlight", "Loamy soil"),
    "ZZ Plant": ("Indoor", 150, "Low to indirect sunlight", "Well-drained soil"),
    "Dracaena": ("Indoor", 250, "Indirect sunlight", "Well-drained soil"),
    "Aglaonema": ("Indoor", 250, "Indirect sunlight", "Loamy soil"),
    "Boston Fern": ("Indoor", 400, "Indirect sunlight", "Moist soil"),

    "Tulsi": ("Herb", 300, "4-6 hours", "Well-drained soil"),
    "Mint": ("Herb", 400, "4-6 hours", "Moist, well-drained soil"),
    "Coriander": ("Herb", 300, "4-5 hours", "Loamy soil"),
    "Basil": ("Herb", 300, "6 hours", "Well-drained soil"),
    "Rosemary": ("Herb", 200, "6-8 hours", "Sandy soil"),
    "Thyme": ("Herb", 200, "6-8 hours", "Well-drained soil"),
    "Oregano": ("Herb", 200, "6-8 hours", "Well-drained soil"),
    "Curry Leaves": ("Herb", 350, "5-6 hours", "Loamy soil"),
    "Lemongrass": ("Herb", 400, "6-8 hours", "Loamy soil"),
    "Parsley": ("Herb", 300, "4-6 hours", "Rich soil"),

    "Tomato": ("Vegetable", 600, "6-8 hours", "Loamy soil"),
    "Potato": ("Vegetable", 500, "6 hours", "Loose, well-drained soil"),
    "Brinjal": ("Vegetable", 500, "6-8 hours", "Loamy soil"),
    "Chilli": ("Vegetable", 400, "6 hours", "Well-drained soil"),
    "Cabbage": ("Vegetable", 500, "6 hours", "Rich soil"),
    "Spinach": ("Vegetable", 350, "4-6 hours", "Moist, rich soil"),
    "Carrot": ("Vegetable", 350, "6 hours", "Loose soil"),
    "Cucumber": ("Vegetable", 600, "6-8 hours", "Rich soil"),
    "Onion": ("Vegetable", 350, "6-8 hours", "Well-drained soil"),
    "Okra": ("Vegetable", 450, "6-8 hours", "Loamy soil"),

    "Mango": ("Fruit", 900, "6-8 hours", "Well-drained soil"),
    "Banana": ("Fruit", 900, "6-8 hours", "Rich soil"),
    "Lemon": ("Fruit", 600, "6-8 hours", "Well-drained soil"),
    "Guava": ("Fruit", 600, "6-8 hours", "Loamy soil"),
    "Papaya": ("Fruit", 700, "6-8 hours", "Well-drained soil"),
    "Pomegranate": ("Fruit", 500, "6-8 hours", "Sandy loam soil"),
    "Orange": ("Fruit", 700, "6-8 hours", "Well-drained soil"),
    "Coconut": ("Fruit", 1000, "Full sunlight", "Sandy loam soil"),
    "Grapes": ("Fruit", 600, "6-8 hours", "Well-drained soil"),
    "Apple": ("Fruit", 700, "6-8 hours", "Loamy soil"),

    "Aloe Vera": ("Succulent", 150, "Bright sunlight", "Sandy soil"),
    "Neem": ("Tree", 500, "Full sunlight", "Well-drained soil"),
    "Bamboo": ("Other", 500, "Bright indirect sunlight", "Moist soil"),
    "Money Tree": ("Indoor", 300, "Bright indirect sunlight", "Well-drained soil"),
    "Rubber Fig": ("Indoor", 300, "Bright indirect sunlight", "Well-drained soil"),
    "Lavender": ("Flowering", 250, "6-8 hours", "Sandy soil"),
    "Geranium": ("Flowering", 300, "5-6 hours", "Well-drained soil"),
    "Ficus": ("Indoor", 300, "Bright indirect sunlight", "Well-drained soil")
}

# ---------------------------------------
# SESSION STATE
# ---------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "reminders" not in st.session_state:
    st.session_state.reminders = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ---------------------------------------
# AI MODEL
# DEMONSTRATION MODEL USING SIMULATED DATA
# ---------------------------------------

np.random.seed(42)

training_data = pd.DataFrame({
    "moisture": np.random.randint(5, 100, 1000),
    "temperature": np.random.randint(15, 45, 1000),
    "humidity": np.random.randint(20, 100, 1000)
})

training_data["need_water"] = (
    (training_data["moisture"] < 30) &
    (training_data["temperature"] > 20)
).astype(int)

model = DecisionTreeClassifier(random_state=42)

model.fit(
    training_data[["moisture", "temperature", "humidity"]],
    training_data["need_water"]
)

# ---------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------

def get_water_quantity(plant, pot_size, age):
    """
    Rough illustrative estimate for one watering.
    Not a scientifically validated recommendation.
    """

    if plant not in plants:
        return None

    base_quantity = plants[plant][1]

    pot_multiplier = {
        "Small": 0.6,
        "Medium": 1.0,
        "Large": 1.8
    }

    age_multiplier = {
        "Seedling": 0.4,
        "Young": 0.7,
        "Mature": 1.0
    }

    quantity = (
        base_quantity
        * pot_multiplier[pot_size]
        * age_multiplier[age]
    )

    minimum = int(quantity * 0.8)
    maximum = int(quantity * 1.2)

    return minimum, maximum


def add_history(plant, moisture, temperature, humidity, result):
    st.session_state.history.append({
        "Date": datetime.now().strftime("%d-%m-%Y %H:%M"),
        "Plant": plant,
        "Soil Moisture": moisture,
        "Temperature": temperature,
        "Humidity": humidity,
        "Prediction": result
    })


# ---------------------------------------
# SIDEBAR
# ---------------------------------------

st.sidebar.title("🌱 Plant Water AI")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Choose a section",
    [
        "🏠 Home",
        "🌿 Plant Selection",
        "💧 Water Prediction",
        "🌤️ Weather Information",
        "📅 Watering Schedule",
        "📖 Plant Care Guide",
        "📊 Analytics",
        "📜 Prediction History",
        "🔔 Reminders"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Smart plant care through AI-based demonstration predictions."
)

# ---------------------------------------
# HOME PAGE
# ---------------------------------------

if page == "🏠 Home":

    st.title("🌱 AI-Based Plant Water Requirement Prediction")

    st.subheader("Smart watering recommendations for your plants")

    st.write(
        "This application demonstrates how artificial intelligence "
        "can help users understand plant watering needs using soil "
        "moisture, temperature and humidity."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Plants Available", len(plants))

    with col2:
        st.metric("Plant Categories", 7)

    with col3:
        st.metric("Main Features", 8)

    st.markdown("---")

    st.subheader("🌿 What can this application do?")

    st.write("""
    - Select from more than 50 plants.
    - Enter soil moisture, temperature and humidity.
    - Get a demonstration AI watering prediction.
    - View approximate water quantity estimates.
    - Learn about plant care and sunlight.
    - View watering schedules.
    - Maintain prediction history.
    - Create watering reminders.
    """)

    st.success(
        "Select a section from the sidebar to explore the application."
    )

    st.info(
        "Note: The machine-learning model uses simulated training data. "
        "Its predictions are for educational demonstration, not "
        "scientifically validated watering decisions."
    )

# ---------------------------------------
# PLANT SELECTION PAGE
# ---------------------------------------

elif page == "🌿 Plant Selection":

    st.title("🌿 Plant Selection")

    category = st.selectbox(
        "Select plant category",
        ["All"] + sorted(set(value[0] for value in plants.values()))
    )

    if category == "All":
        filtered_plants = plants
    else:
        filtered_plants = {
            name: info
            for name, info in plants.items()
            if info[0] == category
        }

    search = st.text_input("Search for a plant")

    for plant, information in filtered_plants.items():

        if search.lower() in plant.lower():

            with st.expander("🌱 " + plant):

                st.write("Category:", information[0])
                st.write("Approximate base water estimate:",
                         information[1], "ml per watering")
                st.write("Sunlight:", information[2])
                st.write("Suitable soil:", information[3])

# ---------------------------------------
# WATER PREDICTION PAGE
# ---------------------------------------

elif page == "💧 Water Prediction":

    st.title("💧 AI Plant Water Prediction")

    st.write("Enter your plant and environmental information.")

    plant_choice = st.selectbox(
        "Choose a plant",
        ["Select a plant"] + sorted(plants.keys()) + ["Other plant"]
    )

    custom_plant = ""

    if plant_choice == "Other plant":
        custom_plant = st.text_input("Enter your plant name")
        selected_plant = custom_plant.strip()
    elif plant_choice == "Select a plant":
        selected_plant = ""
    else:
        selected_plant = plant_choice

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        moisture = st.slider(
            "Soil Moisture (%)",
            0, 100, 40
        )

        temperature = st.slider(
            "Temperature (°C)",
            10, 50, 28
        )

        humidity = st.slider(
            "Air Humidity (%)",
            0, 100, 60
        )

    with col2:

        pot_size = st.selectbox(
            "Pot Size",
            ["Small", "Medium", "Large"]
        )

        plant_age = st.selectbox(
            "Plant Age",
            ["Seedling", "Young", "Mature"]
        )

        soil_type = st.selectbox(
            "Soil Type",
            ["Loamy Soil", "Sandy Soil", "Clay Soil", "Garden Soil"]
        )

        location = st.selectbox(
            "Plant Location",
            ["Indoor", "Outdoor"]
        )

    st.markdown("---")

    if st.button("🔍 Predict Water Requirement", use_container_width=True):

        if not selected_plant:
            st.warning("Please select or enter a plant name.")

        else:

            input_data = pd.DataFrame(
                [[moisture, temperature, humidity]],
                columns=["moisture", "temperature", "humidity"]
            )

            prediction = model.predict(input_data)[0]

            if prediction == 1:
                result = "Watering may be needed"
            else:
                result = "Immediate watering may not be needed"

            st.session_state.last_prediction = result

            add_history(
                selected_plant,
                moisture,
                temperature,
                humidity,
                result
            )

            st.subheader("🌱 Prediction Result")

            if prediction == 1:
                st.warning("💧 " + result)
            else:
                st.success("🌿 " + result)

            if selected_plant in plants:

                category, base_water, sunlight, soil = plants[selected_plant]

                st.markdown("---")
                st.subheader("💧 Approximate Water Quantity")

                st.write("Plant:", selected_plant)
                st.write("Plant category:", category)

                if prediction == 1:

                    quantity = get_water_quantity(
                        selected_plant,
                        pot_size,
                        plant_age
                    )

                    if quantity:
                        st.metric(
                            "Estimated water for one watering",
                            f"{quantity[0]}–{quantity[1]} ml"
                        )

                        st.caption(
                            "This is a rough educational estimate, "
                            "not a validated recommendation. Check the "
                            "soil and pot drainage before watering."
                        )

                else:
                    st.info(
                        "The demonstration model does not indicate "
                        "an immediate watering need. Check the soil "
                        "again later rather than watering automatically."
                    )

                st.markdown("---")

                st.subheader("🌞 Plant Information")

                st.write("Sunlight:", sunlight)
                st.write("Suitable soil:", soil)

            else:

                st.info(
                    "This plant is not in the database yet. "
                    "The general demonstration prediction is available, "
                    "but a plant-specific water estimate and care guide "
                    "are not available."
                )

            if moisture < 20:
                st.warning(
                    "Soil moisture is very low. Check the soil and "
                    "plant condition before deciding how much to water."
                )

            elif moisture > 70:
                st.info(
                    "Soil moisture is high. Check drainage and avoid "
                    "adding water if the soil is already wet."
                )

# ---------------------------------------
# WEATHER INFORMATION
# ---------------------------------------

elif page == "🌤️ Weather Information":

    st.title("🌤️ Weather Information")

    st.write(
        "Enter the weather information manually to understand "
        "how environmental conditions can affect plants."
    )

    temp = st.slider("Current Temperature (°C)", 0, 50, 28)
    humidity_weather = st.slider("Air Humidity (%)", 0, 100, 60)

    condition = st.selectbox(
        "Weather Condition",
        ["Sunny", "Cloudy", "Rainy", "Windy"]
    )

    if st.button("Check Weather Effect"):

        st.subheader("Weather Analysis")

        if condition == "Rainy":
            st.info(
                "Rain may provide water to outdoor plants. "
                "Check soil moisture before watering."
            )

        elif condition == "Sunny" and temp > 35:
            st.warning(
                "Hot weather may increase water loss. "
                "Check soil moisture more frequently."
            )

        elif humidity_weather > 75:
            st.info(
                "High humidity may slow soil drying."
            )

        else:
            st.success(
                "Continue checking soil moisture and plant condition."
            )

        st.write("Temperature:", temp, "°C")
        st.write("Humidity:", humidity_weather, "%")
        st.write("Weather:", condition)

    st.caption("This page uses manually entered weather data, not a live weather service.")

# ---------------------------------------
# WATERING SCHEDULE
# ---------------------------------------

elif page == "📅 Watering Schedule":

    st.title("📅 Watering Schedule")

    st.write(
        "Create a plant-checking schedule. Check the soil before "
        "watering instead of following a fixed quantity every day."
    )

    plant = st.selectbox(
        "Choose plant",
        sorted(plants.keys())
    )

    start_date = st.date_input(
        "Start checking from",
        date.today()
    )

    frequency = st.selectbox(
        "How often would you like to check the soil?",
        ["Daily", "Every 2 days", "Every 3 days", "Weekly"]
    )

    days = {
        "Daily": 1,
        "Every 2 days": 2,
        "Every 3 days": 3,
        "Weekly": 7
    }

    if st.button("Create Schedule"):

        st.subheader("🌱 Your Soil-Checking Schedule")

        for i in range(5):

            next_date = start_date + timedelta(
                days=days[frequency] * i
            )

            st.write(
                f"📅 {next_date.strftime('%d-%m-%Y')} — "
                f"Check soil moisture for {plant}"
            )

        st.info(
            "Water only when the plant and soil conditions indicate "
            "that watering is needed."
        )

# ---------------------------------------
# PLANT CARE GUIDE
# ---------------------------------------

elif page == "📖 Plant Care Guide":

    st.title("📖 Plant Care Guide")

    selected = st.selectbox(
        "Select a plant",
        sorted(plants.keys())
    )

    category, water, sunlight, soil = plants[selected]

    st.subheader("🌿 " + selected)

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Category:**", category)
        st.write("**Sunlight:**", sunlight)

    with col2:
        st.write("**Suitable soil:**", soil)
        st.write("**Approximate base estimate:**", water, "ml")

    st.markdown("---")

    st.subheader("🌱 Basic Care Tips")

    st.write("""
    1. Check the soil before watering.
    2. Make sure the pot has proper drainage.
    3. Provide suitable sunlight.
    4. Remove dry or damaged leaves when appropriate.
    5. Observe changes in plant growth.
    6. Adjust care according to weather and plant condition.
    """)

    st.caption(
        "Care information is general guidance. Requirements vary "
        "by variety, growing stage and local conditions."
    )

# ---------------------------------------
# ANALYTICS PAGE
# ---------------------------------------

elif page == "📊 Analytics":

    st.title("📊 Plant Prediction Analytics")

    if len(st.session_state.history) == 0:

        st.info(
            "No prediction data available. Make some predictions first."
        )

    else:

        df = pd.DataFrame(st.session_state.history)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Predictions", len(df))

        with col2:
            st.metric(
                "Different Plants",
                df["Plant"].nunique()
            )

        st.subheader("🌱 Plant Prediction Records")

        st.dataframe(df, use_container_width=True)

        st.subheader("Soil Moisture Values")

        fig, ax = plt.subplots()

        ax.bar(
            range(len(df)),
            df["Soil Moisture"]
        )

        ax.set_xlabel("Prediction Number")
        ax.set_ylabel("Soil Moisture (%)")
        ax.set_title("Soil Moisture Analysis")

        st.pyplot(fig)

        st.subheader("Prediction Distribution")

        counts = df["Prediction"].value_counts()

        fig2, ax2 = plt.subplots()

        ax2.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%"
        )

        ax2.set_title("Prediction Results")

        st.pyplot(fig2)

# ---------------------------------------
# PREDICTION HISTORY
# ---------------------------------------

elif page == "📜 Prediction History":

    st.title("📜 Prediction History")

    if len(st.session_state.history) == 0:

        st.info("No prediction history available.")
else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

        csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download History as CSV",
            data=csv,
            file_name="plant_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.success("History cleared.")
            st.rerun()

# ---------------------------------------
# REMINDERS PAGE
# ---------------------------------------

else page == "🔔 Reminders":

    st.title("🔔 Plant Care Reminders")

    st.write(
        "Create reminders to check your plants. "
        "These reminders are displayed inside this app; "
        "they are not phone notifications."
    )

    reminder_plant = st.selectbox(
        "Select plant",
        sorted(plants.keys())
    )

    reminder_date = st.date_input(
        "Reminder Date",
        date.today()
    )

    reminder_time = st.time_input(
        "Reminder Time",
        time(9, 0)
    )

    reminder_note = st.text_input(
        "Reminder Message",
        "Check soil moisture"
    )

    if st.button("Add Reminder"):

        st.session_state.reminders.append({
            "Plant": reminder_plant,
            "Date": str(reminder_date),
            "Time": str(reminder_time),
            "Message": reminder_note
        })

        st.success("Reminder added successfully!")

    st.markdown("---")

    st.subheader("📌 Your Reminders")

    if len(st.session_state.reminders) == 0:

        st.info("No reminders created yet.")

    else:

        for reminder in st.session_state.reminders:

            st.write(
                f"🌱 **{reminder['Plant']}** — "
                f"{reminder['Date']} at {reminder['Time']}"
            )

            st.write("Message:", reminder["Message"])
            st.markdown("---")

# ---------------------------------------
# FOOTER
# ---------------------------------------

st.markdown("---")

st.markdown(
    "<center>🌱 AI-Based Plant Water Requirement Prediction "
    "| Smart Plant Care Project 🌱</center>",
    unsafe_allow_html=True
)
