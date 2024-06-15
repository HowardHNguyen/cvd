import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Load models
rf_model = joblib.load('/content/drive/MyDrive/models/rf_model.pkl')
gbm_model = joblib.load('/content/drive/MyDrive/models/gbm_model.pkl')

# Load data from Google Colab
data = pd.read_csv('/content/drive/MyDrive/data/frmgham2.csv')

# Set up the Streamlit app layout
st.title("Cardiovascular Disease Prediction by Howard Nguyen")
st.write("Select normal values for * marked fields if you don't know the exact values")

# Input features in the left column
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    age = st.slider('Enter your age:', min_value=32, max_value=81, value=54)
    tot_chol = st.slider('Total Cholesterol:', min_value=107, max_value=696, value=200)
    sysbp = st.slider('Systolic Blood Pressure:', min_value=83, max_value=295, value=151)
    diabp = st.slider('Diastolic Blood Pressure:', min_value=30, max_value=150, value=89)
    bmi = st.slider('BMI:', min_value=14.43, max_value=56.80, value=26.77)
    cur_smoker = st.selectbox('Current Smoker:', [0, 1])
    glucose = st.slider('Glucose:', min_value=39, max_value=478, value=117)
    diabetes = st.selectbox('Diabetes:', [0, 1])
    heartrate = st.slider('Heart Rate:', min_value=37, max_value=220, value=91)
    cigs_per_day = st.slider('Cigarettes Per Day:', min_value=0, max_value=90, value=20)
    bp_meds = st.selectbox('On BP Meds:', [0, 1])
    prev_strk = st.selectbox('Stroke:', [0, 1])
    hypertension = st.selectbox('Hypertension:', [0, 1])

# Collect input features
features = [age, tot_chol, sysbp, diabp, bmi, cur_smoker, glucose, diabetes, heartrate, cigs_per_day, bp_meds, prev_strk, hypertension]
input_data = np.array(features).reshape(1, -1)

# Predict and display results in the right column
with col2:
    if st.button('Predict'):
        # Make predictions
        rf_proba = rf_model.predict_proba(input_data)[0][1]  # Probability of CVD for Random Forest
        gbm_proba = gbm_model.predict_proba(input_data)[0][1]  # Probability of CVD for Gradient Boosting Machine

        # Display predictions
        st.write(f"Random Forest Prediction: CVD with probability {rf_proba:.2f}")
        st.write(f"Gradient Boosting Machine Prediction: CVD with probability {gbm_proba:.2f}")

        # Plot Prediction Probability Distribution
        fig, ax = plt.subplots()
        ax.bar(['Random Forest', 'Gradient Boosting Machine'], [rf_proba, gbm_proba], color=['blue', 'orange'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probability Distribution')
        st.pyplot(fig)

        # Feature importances for Random Forest
        st.subheader("Feature Importances (Random Forest)")
        rf_importances = rf_model.feature_importances_
        feature_names = ['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 'CURSMOKE', 'GLUCOSE', 'DIABETES', 'HEARTRTE', 'CIGPDAY', 'BPMEDS', 'STROKE', 'HYPERTEN']
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': rf_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots()
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='blue')
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importances')
        st.pyplot(fig)

        # ROC Curve and AUC
        st.subheader("Model Performance")
        
        # Dummy labels and predictions for ROC curve demonstration
        y_val = np.random.randint(0, 2, 100)  # Example true labels
        rf_probas = np.random.rand(100)  # Example predicted probabilities for Random Forest
        gbm_probas = np.random.rand(100)  # Example predicted probabilities for Gradient Boosting Machine

        fpr_rf, tpr_rf, _ = roc_curve(y_val, rf_probas)
        fpr_gbm, tpr_gbm, _ = roc_curve(y_val, gbm_probas)

        fig, ax = plt.subplots()
        ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_val, rf_probas):.2f})')
        ax.plot(fpr_gbm, tpr_gbm, label=f'Gradient Boosting Machine (AUC = {roc_auc_score(y_val, gbm_probas):.2f})')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        st.pyplot(fig)
