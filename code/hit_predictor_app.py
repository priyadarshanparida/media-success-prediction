import streamlit as st
import pandas as pd
import numpy as np

# STEP 1: Base Streamlit Dashboard Setup

# Page Title & Intro
st.set_page_config(page_title="Hit Predictor Dashboard", layout="wide")

st.title("🎵 Real-Time Hit Predictor Dashboard")
st.markdown("""
This dashboard uses our **Logistic Regression (class_weight='balanced')** model  
to estimate the probability that a track will reach the Billboard Top 10  
based on its **audio** and **sentiment** features.
""")

# Model Details
st.subheader("Model Overview")
st.write("Model: Logistic Regression (AUC ≈ 0.72 | Recall ≈ 0.65)")

# Confirmation message to ensure Streamlit setup works
st.success("✅ Dashboard initialized successfully! Proceed to Step 2 for user input and prediction.")

# STEP 2: User Input & Prediction
import joblib

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("log_reg_model.pkl")

log_reg = load_model()

st.subheader("🔮 Predict Hit Probability")

# Choose input method
input_mode = st.radio(
    "Select input method:",
    ("Enter features manually", "Upload CSV file")
)

# Load feature names expected by the model
feature_names = [
    'danceability', 'energy', 'acousticness', 'speechiness',
    'liveness', 'instrumentalness', 'sentiment_intensity',
    'log_mentions', 'engagement_score'
]

# Manual input form
if input_mode == "Enter features manually":
    user_inputs = {}
    st.markdown("Enter feature values (0–1 scale unless noted):")
    for feat in feature_names:
        user_inputs[feat] = st.slider(feat, 0.0, 1.0, 0.5)
    input_df = pd.DataFrame([user_inputs])

# CSV upload option
else:
    uploaded_file = st.file_uploader("Upload a CSV containing feature columns:", type="csv")
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(input_df.head())
    else:
        st.stop()

# Run prediction when button pressed
if st.button("Predict Hit Probability"):
    try:
        # Ensure all expected columns are present
        expected_cols = log_reg.feature_names_in_
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_cols]

        probs = log_reg.predict_proba(input_df)[:, 1]
        input_df["Hit_Probability"] = np.round(probs * 100, 2)
        st.success("Prediction complete!")

        # Visual Display for First Song
        prob_value = float(probs[0])  # take first prediction
        st.metric(label="🎯 Predicted Hit Probability", value=f"{prob_value*100:.2f}%")
        st.progress(prob_value)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

