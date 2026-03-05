import streamlit as st
import pandas as pd
import pickle
import os
import urllib.request

st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="centered")

# Hugging Face model URL
MODEL_URL = "https://huggingface.co/adityaseth07/cricket-win-predictor-model/resolve/main/model.pkl"
MODEL_PATH = "model.pkl"

# Load model with caching
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return pickle.load(open(MODEL_PATH, "rb"))

model = load_model()

st.title("🏏 IPL Match Win Probability Predictor")
st.markdown("Predict the chances of a team winning while chasing in an IPL match.")

st.divider()

st.subheader("📊 Enter Match Situation")

col1, col2 = st.columns(2)

with col1:
    team_runs = st.number_input("Current Runs", 0, 300, 120)
    wickets_left = st.number_input("Wickets Left", 0, 10, 6)
    current_run_rate = st.number_input("Current Run Rate", 0.0, 20.0, 8.5)

with col2:
    balls_remaining = st.number_input("Balls Remaining", 0, 120, 30)
    target = st.number_input("Target Runs", 0, 300, 160)

st.divider()

if st.button("Predict Win Probability"):

    overs_remaining = balls_remaining / 6
    runs_remaining = target - team_runs

    st.subheader("📌 Match Situation")

    colA, colB = st.columns(2)

    with colA:
        st.metric("Runs Remaining", max(runs_remaining,0))

    with colB:
        st.metric("Overs Remaining", round(overs_remaining,2))

    # Case 1: Already won
    if runs_remaining <= 0:
        st.success("🏆 Chasing team has already won!")
        st.progress(100)
        st.stop()

    # Case 2: No balls left
    if balls_remaining == 0 and runs_remaining > 0:
        st.error("❌ No balls remaining. Chasing team loses.")
        st.progress(0)
        st.stop()

    # Calculate required run rate
    if overs_remaining > 0:
        required_run_rate = runs_remaining / overs_remaining
    else:
        required_run_rate = 0

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
    probability_percent = round(probability * 100, 2)

    st.subheader("🏏 Win Probability")

    st.progress(probability)

    st.success(f"Predicted Win Probability: **{probability_percent}%**")

    st.caption("Model: Random Forest trained on historical IPL match data")