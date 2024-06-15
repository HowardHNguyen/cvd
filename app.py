import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score

# Load the calibrated models from GitHub
try:
    rf_model_calibrated = joblib.load('https://github.com/HowardHNguyen/cvd/raw/master/rf_model_calibrated.pkl')
    gbm_model_calibrated = joblib.load('https://github.com/HowardHNguyen/cvd/raw/master/gbm_model_calibrated.pkl')
except Exception as e:
    st.error(f"Error loading models: {e}")

# Load the dataset
try:
    data = pd.read_csv('https://github.com/HowardHNguyen/cvd/raw/master/frmgham2.csv')
except Exception as e:
    st.error(f"Error loading data: {e}")

# Handle missing values by replacing them with the mean of the respective columns
if 'data' in locals():
    data.fillna(data.mean(), inplace=True)

# Define the feature columns
feature_columns = ['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 'CURSMOKE', 
                   'GLUCOSE', 'DIABETES', 'HEARTRTE', 'CIGPDAY', 'BPMEDS', 
                   'STROKE', 'HYPERTEN']

# Sidebar for input parameters
st.sidebar.header('Enter your parameters')

def user_input_features():
    age = st.sidebar.slider('Enter your age:', 32, 81, 54)
    totchol = st.sidebar.slider('Total Cholesterol:', 107, 696, 200)
    sysbp = st.sidebar.slider('Systolic Blood Pressure:', 83, 295, 151)
    diabp = st.sidebar.slider('Diastolic Blood Pressure:', 30, 150, 89)
    bmi = st.sidebar.slider('BMI:', 14.43, 56.80, 26.77)
    cursmoke = st.sidebar.selectbox('Current Smoker:', (0, 1))
    glucose = st.sidebar.slider('Glucose:', 39, 478, 117)
    diabetes = st.sidebar.selectbox('Diabetes:', (0, 1))
    heartrate = st.sidebar.slider('Heart Rate:', 37, 220, 91)
    cigpday = st.sidebar.slider('Cigarettes Per Day:', 0, 90, 20)
    bpmeds = st.sidebar.selectbox('On BP Meds:', (0, 1))
    stroke = st.sidebar.selectbox('Stroke:', (0, 1))
    hyperten = st.sidebar.selectbox('Hypertension:', (0, 1))
    
    data = {
        'AGE': age,
        'TOTCHOL': totchol,
        'SYSBP': sysbp,
        'DIABP': diabp,
        'BMI': bmi,
        'CURSMOKE': cursmoke,
        'GLUCOSE': glucose,
        'DIABETES': diabetes,
        'HEARTRTE': heartrate,
        'CIGPDAY': cigpday,
        'BPMEDS': bpmeds,
        'STROKE': stroke,
        'HYPERTEN': hyperten
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Apply the model to make predictions
try:
    rf_proba_calibrated = rf_model_calibrated.predict_proba(input_df)[:, 1]
    gbm_proba_calibrated = gbm_model_calibrated.predict_proba(input_df)[:, 1]
except Exception as e:
    st.error(f"Error making predictions: {e}")

st.write("""
# Cardiovascular Disease Prediction App
This app predicts the probability of cardiovascular disease (CVD) using user inputs.
""")

st.subheader('Predictions')
try:
    st.write(f"Random Forest Prediction: CVD with probability {rf_proba_calibrated[0]:.2f}")
    st.write(f"Gradient Boosting Machine Prediction: CVD with probability {gbm_proba_calibrated[0]:.2f}")
except:
    pass

# Plot the prediction probability distribution
st.subheader('Prediction Probability Distribution')
try:
    fig, ax = plt.subplots()
    bars = ax.bar(['Random Forest', 'Gradient Boosting Machine'], [rf_proba_calibrated[0], gbm_proba_calibrated[0]], color=['blue', 'orange'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom')  # va: vertical alignment
    st.pyplot(fig)
except:
    pass

# Plot feature importances for Random Forest
st.subheader('Feature Importances (Random Forest)')
try:
    fig, ax = plt.subplots()
    importances = rf_model_calibrated.feature_importances_
    indices = np.argsort(importances)
    ax.barh(range(len(indices)), importances[indices], color='blue', align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_columns[i] for i in indices])
    ax.set_xlabel('Importance')
    st.pyplot(fig)
except:
    pass

# Plot ROC curve for both models
st.subheader('Model Performance')
try:
    fig, ax = plt.subplots()
    fpr_rf, tpr_rf, _ = roc_curve(data['CVD'], rf_model_calibrated.predict_proba(data[feature_columns])[:, 1])
    fpr_gbm, tpr_gbm, _ = roc_curve(data['CVD'], gbm_model_calibrated.predict_proba(data[feature_columns])[:, 1])
    ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(data["CVD"], rf_model_calibrated.predict_proba(data[feature_columns])[:, 1]):.2f})')
    ax.plot(fpr_gbm, tpr_gbm, label=f'Gradient Boosting Machine (A
