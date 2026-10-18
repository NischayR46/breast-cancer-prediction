"""Streamlit app: predict whether a breast tumour is benign or malignant.

Run with:  streamlit run app.py
Uses the Wisconsin Diagnostic Breast Cancer dataset bundled with scikit-learn.
For learning/portfolio use only - not a medical device.
"""
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# The 10 "mean" measurements are used so the form stays short.
FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
]


@st.cache_resource
def train_model():
    data = load_breast_cancer(as_frame=True)
    X = data.frame[FEATURES]
    y = data.target  # 0 = malignant, 1 = benign
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, X


st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺")
st.title("Breast Cancer Prediction")
st.caption("Educational demo using the Wisconsin Diagnostic Breast Cancer dataset. Not medical advice.")

model, acc, X = train_model()
st.write(f"Model: Random Forest | Test accuracy: **{acc:.1%}**")

st.subheader("Enter tumour measurements")
values = {}
cols = st.columns(2)
for i, feat in enumerate(FEATURES):
    lo, hi, mean = float(X[feat].min()), float(X[feat].max()), float(X[feat].mean())
    values[feat] = cols[i % 2].slider(feat, lo, hi, mean, step=(hi - lo) / 200)

if st.button("Predict"):
    row = pd.DataFrame([values])
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    if pred == 1:
        st.success(f"Prediction: Benign (confidence {proba[1]:.1%})")
    else:
        st.error(f"Prediction: Malignant (confidence {proba[0]:.1%})")
