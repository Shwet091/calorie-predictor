import streamlit as st
import numpy as np
import pickle
import google.generativeai as genai
import os
from sklearn.preprocessing import StandardScaler
from streamlit_lottie import st_lottie
import json
from supabase import create_client, Client
from datetime import datetime
import pandas as pd

# Load trained XGBoost model
model = pickle.load(open("calories_model.pkl", "rb"))

# Load the StandardScaler used during training
scaler = pickle.load(open("scaling.pkl", "rb"))

# Set up Gemini API Key securely
GEMINI_API_KEY = st.secrets["env"]["GEMINI_API_KEY"]
if not GEMINI_API_KEY:
    st.error("⚠️ Gemini API key not found. Please set it in your secrets.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Supabase connection setup
SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load Lottie animation
@st.cache_data
def load_lottie_animation(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

lottie_fitness = load_lottie_animation("Animation - 1741535941522.json")
lottie_chatbot = load_lottie_animation("Animation - 1741536478823.json")

# Styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
        animation: fadeIn 1.5s ease-in-out;
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    .stButton > button {
        background-color: #ff5e62 !important;
        color: white !important;
        border-radius: 12px;
        font-size: 16px;
        padding: 10px 20px;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.05);
    }
    .stMetric {
        border: 2px solid #ff5e62;
        padding: 12px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.1);
        animation: bounceIn 1s;
    }
    @keyframes bounceIn {
        0% { transform: scale(0.5); opacity: 0; }
        60% { transform: scale(1.1); opacity: 1; }
        100% { transform: scale(1); }
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st_lottie(lottie_fitness, height=180, key="fit")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input validation
def validate_inputs(age, height, weight, duration, heart_rate, body_temp):
    errors = []
    if not (10 <= age <= 100): errors.append("⚠️ Age should be between 10 and 100.")
    if not (100 <= height <= 250): errors.append("⚠️ Height should be between 100 cm and 250 cm.")
    if not (30 <= weight <= 200): errors.append("⚠️ Weight should be between 30 kg and 200 kg.")
    if not (5 <= duration <= 180): errors.append("⚠️ Exercise duration should be between 5 and 180 minutes.")
    if not (50 <= heart_rate <= 200): errors.append("⚠️ Heart rate should be between 50 and 200 bpm.")
    if not (35.0 <= body_temp <= 42.0): errors.append("⚠️ Body temperature should be between 35.0°C and 42.0°C.")
    return errors

# Predict calories
def predict_calories(age, gender, height, weight, duration, heart_rate, body_temp):
    gender_encoded = 1 if gender == "Male" else 0
    features = np.array([[gender_encoded, age, height, weight, duration, heart_rate, body_temp]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    return round(float(prediction), 2)

# Diet Plan
def generate_diet_plan_by_weight(weight):
    if weight < 50:
        return "🥗 **Underweight Diet:** High-protein meals, nuts, dairy, and healthy fats."
    elif 50 <= weight <= 80:
        return "🍽️ **Balanced Diet:** Lean protein, vegetables, whole grains, and fruits."
    else:
        return "🍖 **Weight Loss Diet:** High-protein, low-carb, lots of vegetables, and portion control."

# UI Title
st.title("🔥 **AI-Powered Calorie Burn Predictor & Diet Planner**")

# Inputs
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("🧑 Age", min_value=10, max_value=100, value=None)
    gender = st.selectbox("⚥ Gender", ["Male", "Female"])
    height = st.number_input("🖉 Height (cm)", min_value=100, max_value=250, value=None)
with col2:
    weight = st.number_input("⚖️ Weight (kg)", min_value=30, max_value=200, value=None)
    duration = st.number_input("⏳ Exercise Duration (mins)", min_value=5, max_value=180, value=None)
    heart_rate = st.number_input("❤️ Heart Rate (bpm)", min_value=50, max_value=200, value=None)
    body_temp = st.number_input("🌡️ Body Temperature (°C)", min_value=35.0, max_value=42.0, value=None)

# Predict & Save
if st.button("🚀 Predict Calories Burned"):
    errors = validate_inputs(age, height, weight, duration, heart_rate, body_temp)
    if errors:
        for error in errors:
            st.warning(error)
    else:
        calories_burned = predict_calories(age, gender, height, weight, duration, heart_rate, body_temp)
        st.metric(label="🔥 **Estimated Calories Burned**", value=f"{calories_burned} kcal")
        diet_plan = generate_diet_plan_by_weight(weight)
        st.success(f"🍽️ **Recommended Diet Plan:** {diet_plan}")

        prediction_data = {
            "gender": str(gender),
            "age": int(age),
            "height": float(height),
            "weight": float(weight),
            "duration": float(duration),
            "heart_rate": float(heart_rate),
            "body_temp": float(body_temp),
            "predicted_calories": float(calories_burned),
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            response = supabase.table("Predictions").insert(prediction_data).execute()
            if response.data:
                st.success("✅ Prediction saved to database.")
                st.session_state.prediction_id = response.data[0]['id']
        except Exception as e:
            st.warning(f"⚠️ Failed to save prediction: {e}")

# Chatbot
st.subheader("🤖 AI Fitness Assistant")
chat_col1, chat_col2 = st.columns([2, 1])
with chat_col1:
    predefined_topics = [
        "Workout plans",
        "Nutrition advice",
        "Weight loss tips",
        "Muscle gain strategies",
        "Daily calorie needs",
        "Best exercises for fat loss",
        "Diet plan according to weight",
    ]
    topic = st.selectbox("💡 Choose a fitness topic or type your question:", ["Custom Query"] + predefined_topics)
    user_query = st.text_input("💬 Enter your fitness or nutrition question:")

    if st.button("💡 Get Answer"):
        try:
            g_model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                system_instruction="""
                You are a professional fitness assistant. You help users with diet plans, calorie tracking, workouts, weight loss strategies, and general health advice.
                Be supportive, accurate, and clear in your answers. Always assume the user wants help with their fitness goals.
                """
            )
            query = user_query if topic == "Custom Query" else (
                f"""
                I am a {age}-year-old {gender.lower()} with height {height} cm and weight {weight} kg.
                My exercise duration is {duration} minutes, heart rate is {heart_rate} bpm, and body temperature is {body_temp}°C.
                Please provide guidance on the topic: '{topic}'.
                """
            )
            response = g_model.generate_content(query)
            response_text = response.text
            st.session_state.chat_history.append({"question": query.strip(), "answer": response_text})
            st.write(response_text)

            if "prediction_id" in st.session_state:
                chat_response = supabase.table("Predictions").update({
                    "user_question": query.strip(),
                    "bot_response": response_text
                }).eq("id", st.session_state.prediction_id).execute()
                if chat_response.data:
                    st.info("💬 Chat log saved to database.")

        except Exception as e:
            st.error(f"⚠️ Gemini API Error: {str(e)}")

with chat_col2:
    st_lottie(lottie_chatbot, height=300, key="chat_anim")

# Chat History
st.subheader("📜 Chat History")
for chat in st.session_state.chat_history:
    with st.expander(chat["question"]):
        st.write(chat["answer"])

# Admin: Download all predictions
st.subheader("📤 Download All Predictions")
try:
    result = supabase.table("Predictions").select("*").execute()
    if result.data:
        df = pd.DataFrame(result.data)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📁 Download All Predictions CSV", csv, "all_predictions.csv", "text/csv")

        st.subheader("📊 Prediction Trend")
        st.line_chart(df[['timestamp', 'predicted_calories']].set_index('timestamp'))
except Exception as e:
    st.error(f"⚠️ Could not fetch predictions: {e}")
