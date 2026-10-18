Breast Cancer Prediction

A machine learning project that classifies breast tumours as benign or malignant from cell-nucleus measurements, with a small Streamlit web app for live predictions.

Dataset

Wisconsin Diagnostic Breast Cancer dataset (569 samples, 30 numeric features), loaded directly from scikit-learn.

Approach
Exploratory data analysis (class balance, feature distributions, correlations)
Train/test split (80/20, stratified) and feature scaling
Compared Logistic Regression, SVM, Random Forest and K-Nearest Neighbours
Evaluated with accuracy, precision, recall, F1 and confusion matrix
Results

Add your final numbers here after running the notebook, for example: Best model: <model name>, test accuracy: <xx>%, recall on malignant class: <xx>%

Project structure
breast_cancer_prediction.ipynb - analysis and model comparison
app.py - Streamlit app
requirements.txt - dependencies
How to run
bash
pip install -r requirements.txt
jupyter notebook breast_cancer_prediction.ipynb
streamlit run app.py
Disclaimer

Educational project only. Not for medical use.
