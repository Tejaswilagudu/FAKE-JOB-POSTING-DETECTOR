import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# App title
st.title("🕵️ Fake Job Posting Detector")

# User input
job_input = st.text_area("Paste the job description here:")

# Predict button
if st.button("Check if Fake"):
    if job_input.strip() == "":
        st.warning("Please enter some job description text.")
    else:
        # Vectorize and predict
        input_vect = vectorizer.transform([job_input])
        prediction = model.predict(input_vect)[0]
        
        if prediction == 1:
            st.error("🚩 This job posting looks **FAKE**.")
        else:
            st.success("✅ This job posting looks **REAL**.")
