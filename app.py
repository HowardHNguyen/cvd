import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# Load models
rf_model = joblib.load('rf_model.pkl')
gbm_model = joblib.load('gbm_model.pkl')

# Load the dataset
data_url = 'https://raw.githubusercontent.com/HowardHNguyen/cvd/master/frmgham2.csv'
data = pd.read_csv(data_url)

# Split data into features and labels
X = data[['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 'CURSMOKE',
          'GLUCOSE', 'DIABETES', 'HEARTRTE', 'CIGPDAY', 'BPMEDS', 'STROKE', 'HYPERTEN']]
y = data['CVD']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure there are no NaN or infinite values in the validation data
X_val = X_val.replace([np.inf, -np.inf], np.nan).dropna()
y_val = y_val.loc[X_val.index]

# Define the app layout
st.title('Cardiovascular Disease Prediction by Howard Nguyen')
st.write('Select normal values for * marked fields if you don\'t know the exact values')

col1, col_space, col2 = st.columns([1, 0.1, 1])

with col1:
    # User inputs with sliders	age = st.slider('Enter your age:', min_value=32, max_value=81, value=32)	totchol = st.slider('Total Cholesterol:', min_value=107, max_value=696, value=200)	sysbp = st.slider('Systolic Blood Pressure:', min_value=83, max_value=295, value=120)	diabp = st.slider('Diastolic Blood Pressure:', min_value=30, max_value=150, value=80)	bmi = st.slider('BMI:', min_value=14.43, max_value=56.8, value=25.0)	cursmoke = st.selectbox('Current Smoker:', [0, 1])	glucose = st.slider('Glucose:', min_value=39, max_value=478, value=100)	diabetes = st.selectbox('Diabetes:', [0, 1])	heartrate = st.slider('Heart Rate:', min_value=37, max_value=220, value=70)	cigpday = st.slider('Cigarettes Per Day:', min_value=0, max_value=90, value=0)	bpmeds = st.selectbox('On BP Meds:', [0, 1])	stroke = st.selectbox('Stroke:', [0, 1])	hyperten = st.selectbox('Hypertension:', [0, 1])

predict_button = col2.button('Predict Your CVD Probability')

if predict_button:
    input_data = pd.DataFrame({
        'AGE': [age], 'TOTCHOL': [totchol], 'SYSBP': [sysbp], 'DIABP': [diabp],
        'BMI': [bmi], 'CURSMOKE': [cursmoke], 'GLUCOSE': [glucose], 'DIABETES': [diabetes],
        'HEARTRTE': [heartrate], 'CIGPDAY': [cigpday], 'BPMEDS': [bpmeds],
        'STROKE': [stroke], 'HYPERTEN': [hyperten]
    })

    # Make predictions
    rf_proba = rf_model.predict_proba(input_data)
    gbm_proba = gbm_model.predict_proba(input_data)
    rf_pred = rf_model.predict(input_data)[0]
    gbm_pred = gbm_model.predict(input_data)[0]

    with col2:
        # Show predictions
        st.write(f"Random Forest Prediction: {'CVD' if rf_pred else 'No CVD'} with probability {rf_proba[0][1]:.2f}")
        st.write(f"Gradient Boosting Machine Prediction: {'CVD' if gbm_pred else 'No CVD'} with probability {gbm_proba[0][1]:.2f}")

        st.subheader("Prediction Probability Distribution")
        plt.figure(figsize=(10, 5))
        plt.hist(rf_proba[:, 1], bins=10, alpha=0.5, label='Random Forest')
        plt.hist(gbm_proba[:, 1], bins=10, alpha=0.5, label='Gradient Boosting Machine')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        st.pyplot(plt)

        st.subheader("Feature Importances (Random Forest)")
        feature_importances = rf_model.feature_importances_
        features = X.columns
        plt.figure(figsize=(10, 5))
        plt.barh(features, feature_importances, color='royalblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        st.pyplot(plt)

        # Static ROC Curve Calculation using validation data
        fpr_rf, tpr_rf, _ = roc_curve(y_val, rf_model.predict_proba(X_val)[:, 1])
        fpr_gbm, tpr_gbm, _ = roc_curve(y_val, gbm_model.predict_proba(X_val)[:, 1])

        st.subheader("Model Performance")
        plt.figure(figsize=(10, 5))
        plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_val, rf_model.predict_proba(X_val)[:, 1]):.2f})')
        plt.plot(fpr_gbm, tpr_gbm, label=f'Gradient Boosting Machine (AUC = {roc_auc_score(y_val, gbm_model.predict_proba(X_val)[:, 1]):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        st.pyplot(plt)
