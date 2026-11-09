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
    "
