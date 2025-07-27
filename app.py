import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Motion background + style fix
# Motion background + fixed colors for alert boxes
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(-45deg, #ff9a9e, #fad0c4, #a1c4fd, #c2e9fb);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #000000;
    }

    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .block-container {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }

    textarea {
        background-color: #fff;
        color: #000;
        border-radius: 10px;
        padding: 10px;
    }

    /* Button styling */
    button {
        background-color: #000 !important;
        color: #fff !important;
        border-radius: 10px;
        padding: 10px 16px;
    }

    /* Fix success message box (REAL) */
    .stAlert-success {
        background-color: #d4edda !important;
        color: #155724 !important;
        border-left: 5px solid #28a745 !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }

    /* Fix error message box (FAKE) */
    .stAlert-error {
        background-color: #f8d7da !important;
        color: #721c24 !important;
        border-left: 5px solid #dc3545 !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


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
