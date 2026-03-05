import streamlit as st
import pandas as pd
import pickle
import os
import urllib.request

# Hugging Face model URL
MODEL_URL = "https://huggingface.co/adityaseth07/cricket-win-predictor-model/resolve/main/model.pkl"
MODEL_PATH = "model.pkl"

# Load model with caching
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return pickle.load(open(MODEL_PATH, "rb"))

# Load trained model
model = load_model()

st.title("🏏 IPL Win Probability Predictor")

st.write("Enter match situation:")

team_runs = st.number_input("Current Runs", 0, 300, 120)
wickets_left = st.number_input("Wickets Left", 0, 10, 6)
balls_remaining = st.number_input("Balls Remaining", 0, 120, 30)
current_run_rate = st.number_input("Current Run Rate", 0.0, 20.0, 8.5)
target = st.number_input("Target Runs", 0, 300, 160)

if st.button("Predict Win Probability"):
    
    overs_remaining = balls_remaining / 6
    runs_remaining = target - team_runs

    if overs_remaining == 0:
        required_run_rate = 0
    else:
        required_run_rate = runs_remaining / overs_remaining

    sample = pd.DataFrame(
        [[team_runs, wickets_left, balls_remaining, current_run_rate, runs_remaining, required_run_rate]],
        columns=[
            'team_runs',
            'wickets_left',
            'balls_remaining',
            'current_run_rate',
            'runs_remaining',
            'required_run_rate'
        ]
    )

    probability = model.predict_proba(sample)[0][1]

    st.success(f"Win Probability: {round(probability * 100, 2)}%")