"""
Heart Disease Prediction - Student Demo App
--------------------------------------------
A small Streamlit web app that lets students interact with the trained
model from the "2 - Heart Disease Prediction" notebook, without needing
to install Python or run Jupyter.

It loads the same heart_disease_svc_model.pkl produced by that notebook
(a StandardScaler + Support Vector Classifier pipeline), and recreates the
exact same LabelEncoders (fit on the same fixed category lists) used
during training, so predictions match the notebook exactly.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# ---------------------------------------------------------------------------
# Load the trained model (same file the notebook produced)
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "heart_disease_svc_model.pkl"

SEX_OPTIONS = ["Male", "Female"]
CP_OPTIONS = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def build_label_encoders():
    """
    Recreates the exact same LabelEncoders used in the notebook (Step 5.1).
    LabelEncoder assigns codes based on the sorted order of the category
    list, so fitting on the same fixed lists reproduces the same codes
    deterministically - no need to pickle/ship the encoders separately.
    The StandardScaler is already inside the saved pipeline.
    """
    encoders = {}
    encoders["sex"] = LabelEncoder().fit(SEX_OPTIONS)
    encoders["cp"] = LabelEncoder().fit(CP_OPTIONS)
    return encoders

model = load_model()
encoders = build_label_encoders()

# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.title("❤️ Heart Disease Predictor")
st.markdown(
    """
    This is a teaching demo built from the **Heart Disease Prediction** notebook
    (UCI Cleveland dataset, 303 patients). Enter a patient's details below and the
    trained Support Vector Classifier (SVC) will predict whether they are likely
    to have heart disease.

    *Model accuracy on held-out test data: **74%** (61 test patients).*
    """
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=55, step=1,
                          help="Patients in the training data were 29 to 77 years old.")
    sex = st.selectbox("Sex", SEX_OPTIONS)
with col2:
    cp = st.selectbox("Chest Pain Type", CP_OPTIONS, index=3,
                      help="Asymptomatic means the patient does not have the usual chest pain.")
    thalach = st.number_input("Maximum Heart Rate (beats per minute)", min_value=40, max_value=250,
                              value=150, step=1,
                              help="Highest heart rate reached during an exercise test "
                                   "(71 to 202 in the training data).")

if st.button("Predict Heart Disease", type="primary", use_container_width=True):
    # Same column names and order as the training data in the notebook
    encoded = pd.DataFrame({
        "age": [float(age)],
        "sex": encoders["sex"].transform([sex]),
        "cp": encoders["cp"].transform([cp]),
        "thalach": [float(thalach)],
    })

    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE LIKELY")
    else:
        st.success("### ✅ NO HEART DISEASE LIKELY")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption(
    "Built for teaching purposes from the UCI Cleveland Heart Disease dataset using only "
    "4 attributes. This is a learning exercise, not a medical tool - it must not be used "
    "to diagnose real patients."
)
