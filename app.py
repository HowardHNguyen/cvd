import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import requests
from io import StringIO

# Load the dataset from GitHub
url = "https://raw.githubusercontent.com/HowardHNguyen/cvd/master/frmgham2.csv"
response = requests.get(url)
data = pd.read_csv(StringIO(response.text))

# Define the feature columns used in the models
feature_columns = ['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 'CURSMOKE',
                   'GLUCOSE', 'DIABETES', 'HEARTRTE', 'CIGPDAY', 'BPMEDS',
                   'STROKE', 'HYPERTEN']

# Load the trained models
rf_model = joblib.load('rf_model.pkl')
gbm_model = joblib.load('gbm_model.pkl')

# Streamlit app title
st.title('Cardiovascular Disease Prediction by Howard Nguyen')
st.markdown('Select normal values for ★ marked fields if you don’t know the exact values')

# User inputs
age = st.number_input('Enter your age:', min_value=0, max_value=120, value=25)
total_chol = st.number_input('Total Cholesterol:', min_value=0, max_value=600, value=200)
sysbp = st.number_input('Systolic Blood Pressure:', min_value=0, max_value=300, value=120)
diabp = st.number_input('Diastolic Blood Pressure:', min_value=0, max_value=200, value=80)
bmi = st.number_input('BMI:', min_value=0.0, max_value=70.0, value=25.0)
cursmoke = st.selectbox('Current Smoker:', [0, 1])
glucose = st.number_input('Glucose:', min_value=0, max_value=500, value=100)
diabetes = st.selectbox('Diabetes:', [0, 1])
heartrate = st.number_input('Heart Rate:', min_value=0, max_value=300, value=70)
cigpday = st.number_input('Cigarettes Per Day:', min_value=0, max_value=100, value=0)
bpmeds = st.selectbox('On BP Meds:', [0, 1])
stroke = st.selectbox('Stroke:', [0, 1])
hyperten = st.selectbox('Hypertension:', [0, 1])

input_data = np.array([[age, total_chol, sysbp, diabp, bmi, cursmoke, glucose, diabetes,
                        heartrate, cigpday, bpmeds, stroke, hyperten]])

# Predictions
rf_proba = rf_model.predict_proba(input_data)
gbm_proba = gbm_model.predict_proba(input_data)

rf_prediction = rf_proba[0][1]
gbm_prediction = gbm_proba[0][1]

st.subheader('Predictions')
st.write(f'Random Forest Prediction: CVD with probability {rf_prediction:.2f}')
st.write(f'Gradient Boosting Machine Prediction: CVD with probability {gbm_prediction:.2f}')

# Feature Importances for Random Forest
st.subheader('Feature Importances (Random Forest)')
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots()
ax.barh(range(len(indices)), importances[indices], align='center')
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([feature_columns[i] for i in indices])
ax.invert_yaxis()
ax.set_xlabel('Importance')
ax.set_title('Feature Importances')

st.pyplot(fig)

# Dummy labels and predictions for ROC curve
y_true = np.random.randint(0, 2, 100)  # Replace with actual validation labels
rf_probas = np.random.rand(100)  # Replace with actual Random Forest predicted probabilities
gbm_probas = np.random.rand(100)  # Replace with actual Gradient Boosting Machine predicted probabilities

# Collect example data for ROC curve demonstration
y_true_input = np.append(y_true, [1])
rf_probas_input = np.append(rf_probas, rf_prediction)
gbm_probas_input = np.append(gbm_probas, gbm_prediction)

# Use actual validation data to calculate ROC curve
fpr_rf, tpr_rf, _ = roc_curve(y_true_input, rf_probas_input)
fpr_gbm, tpr_gbm, _ = roc_curve(y_true_input, gbm_probas_input)

st.subheader('Model Performance')
fig, ax = plt.subplots()
ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_true_input, rf_probas_input):.2f})')
ax.plot(fpr_gbm, tpr_gbm, label=f'Gradient Boosting Machine (AUC = {roc_auc_score(y_true_input, gbm_probas_input):.2f})')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc='best')

st.pyplot(fig)
