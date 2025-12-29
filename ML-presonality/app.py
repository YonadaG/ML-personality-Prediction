import streamlit as st
import numpy as np
import joblib
import json
import os

st.set_page_config(
    page_title="Personality Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("Personality Prediction System")
st.write("Predict whether a person is Introvert or Extrovert using Machine Learning.")


@st.cache_resource
def load_model():
    model = joblib.load("personality_model.pkl")
    scaler = joblib.load("scaler.pkl")
    target_encoder = joblib.load("target_encoder.pkl")
    return model, scaler, target_encoder


model_files_exist = all(os.path.exists(f) for f in ["personality_model.pkl", "scaler.pkl", "target_encoder.pkl"])

if not model_files_exist:
    st.error(" Model files not found! Please run `Ml_train.py` first to train and save the model.")
    st.stop()

model, scaler, target_encoder = load_model()

st.success(" Model loaded successfully!")
st.write(f"Target classes: {list(target_encoder.classes_)}")


if os.path.exists("model_accuracy.json"):
    with open("model_accuracy.json", "r") as f:
        accuracy_data = json.load(f)
    
    st.subheader("📊 Model Accuracy")
    col1, col2 = st.columns(2)
    with col1:
        knn_acc = accuracy_data.get("knn_accuracy", 0) * 100
        st.metric("KNN Accuracy", f"{knn_acc:.2f}%")
    with col2:
        svm_acc = accuracy_data.get("svm_accuracy", 0) * 100
        st.metric("SVC Accuracy", f"{svm_acc:.2f}%")
    
    best_model_name = accuracy_data.get("best_model", "Unknown")
    st.info(f"Currently using: **{best_model_name}** (best performing model)")

# -----------------------------
# User Input Section
# -----------------------------
st.subheader(" Enter User Details")

time_alone = st.slider("Time spent alone (hours)", 0, 10, 6)
stage_fear = st.selectbox("Stage fear", ["No", "Yes"])
social_events = st.slider("Social event attendance", 0, 10, 3)
going_outside = st.slider("Going outside (per week)", 0, 10, 3)
drained = st.selectbox("Drained after socializing", ["No", "Yes"])
friends = st.slider("Friends circle size", 0, 20, 4)
posts = st.slider("Social media post frequency", 0, 10, 3)

# Convert categorical inputs
stage_fear = 1 if stage_fear == "Yes" else 0
drained = 1 if drained == "Yes" else 0

user_input = np.array([[time_alone, stage_fear, social_events, going_outside, drained, friends, posts]])
user_input_scaled = scaler.transform(user_input)


if st.button(" Predict Personality"):
    prediction = model.predict(user_input_scaled)
    # LabelEncoder encodes alphabetically: Extrovert=0, Introvert=1
    if prediction[0] == 0:
        st.success(" Prediction: Extrovert")
    else:
        st.info(" Prediction: Introvert")