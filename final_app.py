# app.py
"""
Streamlit app for NLP Language Detection
This app allows users to upload their trained vectorizer and model,
input text, and get the predicted language.
"""

import streamlit as st
import pickle
import numpy as np

# ---------------- App Configuration ----------------
st.set_page_config(page_title="🌍 NLP Language Detection", layout="centered")
st.title("🌍 Language Detection App")
st.write("Upload your trained **Vectorizer** and **Model**, then type a sentence to identify its language.")

# ---------------- Sidebar: Upload Models ----------------
st.sidebar.header("📂 Upload Model Files")

vector_file = st.sidebar.file_uploader("Upload Vectorizer (.pkl)", type=["pkl"])
model_file = st.sidebar.file_uploader("Upload Model (.pkl)", type=["pkl"])

# Load uploaded model and vectorizer
if vector_file and model_file:
    vectorizer = pickle.load(vector_file)
    model = pickle.load(model_file)
    st.sidebar.success("✅ Model & Vectorizer loaded successfully!")
else:
    st.sidebar.warning("Please upload both the model and vectorizer files to continue.")
    st.stop()

# ---------------- Text Input ----------------
st.header("📝 Enter Text to Detect Language")
user_input = st.text_area("Type or paste a sentence here:", "")

# ---------------- Language Mapping ----------------
LANGUAGE_LABELS = {
    0: 'Arabic',
    1: 'Danish',
    2: 'Dutch',
    3: 'English',
    4: 'French',
    5: 'German',
    6: 'Greek',
    7: 'Hindi',
    8: 'Italian',
    9: 'Kannada',
    10: 'Malayalam',
    11: 'Portuguese',
    12: 'Russian',
    13: 'Spanish',
    14: 'Swedish',
    15: 'Tamil',
    16: 'Turkish'
}

# ---------------- Prediction Button ----------------
if st.button("🔍 Detect Language"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text for prediction.")
    else:
        with st.spinner("Analyzing text..."):
            try:
                # Vectorize text
                X = vectorizer.transform([user_input]).toarray()
                # Predict language
                y_pred = model.predict(X)
                language = LANGUAGE_LABELS.get(int(y_pred[0]), "Unknown")

                st.success(f"**Detected Language:** {language}")
                st.balloons()

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit | NLP | Language Detection")
