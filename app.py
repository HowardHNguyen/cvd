# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab. By Howard Nguyen

Original file is located at
    https://colab.research.google.com/drive/1ebdsi6xiiE4m8jY6FYiCgywDKogpdryi
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import joblib
import requests
from io import StringIO

# Load your dataset from GitHub
url = 'https://github.com/HowardHNguyen/cvd/blob/master/frmgham2.csv'
response = requests.get(url)
data = pd.read_csv(StringIO(response.text))

# Features and labels
X = data[['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 'CURSMOKE', 'GLUCOSE', 'DIABETES',
          'HEARTRTE', 'CIGPDAY', 'BPMEDS', 'STROKE', 'HYPERTEN']]
y = data['CVD']

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the models (Assuming models are pre-trained)
rf_model = joblib.load('rf_model.pkl')
gbm_model = joblib.load('gbm_model.pkl')

# Streamlit app
st.title('Cardiovascular Disease Prediction by Howard Nguyen')
st.write('Select normal values for ★ marked fields if you don\'t know the exact values')

# Input fields
age = st.number_input('Enter your age:', min_value=0, max_value=120, value=25)
totchol = st.number_input('Total Cholesterol:', min_value=0, max_value=700, value=200)
sysbp = st.number_input('Systolic Blood Pressure:', min_value=0, max_value=300, value=120)
diabp = st.number_input('Diastolic Blood Pressure:', min_value=0, max_value=200, value=80)
bmi = st.number_input('BMI:', min_value=0.0, max_value=100.0, value=25.0)
cursmoke = st.selectbox('Current Smoker:', options=[0, 1])
glucose = st.number_input('Glucose:', min_value=0, max_value=500, value=100)
diabetes = st.selectbox('Diabetes:', options=[0, 1])
heartrate = st.number_input('Heart Rate:', min_value=0, max_value=300, value=70)
cigpday = st.number_input('Cigarettes Per Day:', min_value=0, max_value=100, value=0)
bpmeds = st.selectbox('On BP Meds:', options=[0, 1])
stroke = st.selectbox('Stroke:', options=[0, 1])
hypertension = st.selectbox('Hypertension:', options=[0, 1])

# Prepare input data for prediction
input_data = np.array([[
    age, totchol, sysbp, diabp, bmi, cursmoke, glucose, diabetes,
    heartrate, cigpday, bpmeds, stroke, hypertension
]])
input_df = pd.DataFrame(input_data, columns=[
    'AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 'CURSMOKE', 'GLUCOSE', 'DIABETES',
    'HEARTRTE', 'CIGPDAY', 'BPMEDS', 'STROKE', 'HYPERTEN'
])

# Function to predict using Random Forest
def predict_rf(data):
    prediction = rf_model.predict(data)
    prediction_proba = rf_model.predict_proba(data)
    return prediction, prediction_proba

# Function to predict using Gradient Boosting Machine
def predict_gbm(data):
    prediction = gbm_model.predict(data)
    prediction_proba = gbm_model.predict_proba(data)
    return prediction, prediction_proba

# Make predictions
if st.button('Predict'):
    rf_pred, rf_proba = predict_rf(input_df)
    gbm_pred, gbm_proba = predict_gbm(input_df)
    
    st.subheader('Predictions')
    st.write(f'Random Forest Prediction: {"CVD" if rf_pred[0] else "No CVD"} with probability {rf_proba[0][1]:.2f}')
    st.write(f'Gradient Boosting Machine Prediction: {"CVD" if gbm_pred[0] else "No CVD"} with probability {gbm_proba[0][1]:.2f}')
    
    # Use actual validation data to calculate ROC curve
    y_proba_rf = rf_model.predict_proba(X_val)[:, 1]
    y_proba_gbm = gbm_model.predict_proba(X_val)[:, 1]
    
    # Plot ROC curve
    st.subheader('Model Performance')
    fpr_rf, tpr_rf, _ = roc_curve(y_val, y_proba_rf)
    fpr_gbm, tpr_gbm, _ = roc_curve(y_val, y_proba_gbm)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_val, y_proba_rf):.2f})')
    plt.plot(fpr_gbm, tpr_gbm, label=f'Gradient Boosting Machine (AUC = {roc_auc_score(y_val, y_proba_gbm):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    st.pyplot(plt)
    
    # Feature importance plot for Random Forest
    st.subheader('Feature Importances (Random Forest)')
    feature_importances = rf_model.feature_importances_
    features = input_df.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    plt.gca().invert_yaxis()
    st.pyplot(plt)