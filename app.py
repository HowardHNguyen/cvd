import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score
import os
import urllib.request

# Function to download the file
def download_file(url, dest):
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        st.error(f"Error downloading {url}: {e}")
        return False

# URLs for the model files
rf_model_url = 'https://raw.githubusercontent.com/HowardHNguyen/cvd/master/rf_model_calibrated.pkl'
gbm_model_url = 'https://raw.githubusercontent.com/HowardHNguyen/cvd/master/gbm_model_calibrated.pkl'

# Local paths for the model files
rf_model_path = 'rf_model_calibrated.pkl'
gbm_model_path = 'gbm_model_calibrated.pkl'

# Download the models if not already present
if not os.path.exists(rf_model_path):
    st.info(f"Downloading {rf_model_path}...")
    download_file(rf_model_url, rf_model_path)

if not os.path.exists(gbm_model_path):
    st.info(f"Downloading {gbm_model_path}...")
    download_file(gbm_model_url, gbm_model_path)

# Load the calibrated models
try:
    rf_model_calibrated = joblib.load(rf_model_path)
    gbm_model_calibrated = joblib.load(gbm_model_path)
except Exception as e:
    st.error(f"Error loading models: {e}")

# Load the dataset
data_url = 'https://raw.githubusercontent.com/HowardHNguyen/cvd/master/frmgham2.csv'
try:
    data = pd.read_csv(data_url)
except Exception as e:
    st.error(f"Error loading data: {e}")

# Handle missing values by replacing them with the mean of the respective columns
if 'data' in locals():
    data.fillna(data.mean(), inplace=True)

# Define the feature columns
feature_columns = ['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 'CURSMOKE', 
                   'GLUCOSE', 'DIABETES', 'HEARTRTE', 'CIGPDAY', 'BPMEDS', 
                   'STROKE', 'HYPERTEN', 'LDLC','HDLC']

# Sidebar for input parameters
st.sidebar.header('Enter your parameters')

def user_input_features():
    age = st.sidebar.slider('Enter your age:', 32, 81, 54)
    totchol = st.sidebar.slider('Total Cholesterol:', 107, 696, 175)
    sysbp = st.sidebar.slider('Systolic Blood Pressure:', 83, 295, 130)
    diabp = st.sidebar.slider('Diastolic Blood Pressure:', 30, 150, 80)
    bmi = st.sidebar.slider('BMI:', 14.43, 56.80, 28.27)
    heartrate = st.sidebar.slider('Heart Rate:', 37, 220, 60)
    glucose = st.sidebar.slider('Glucose:', 39, 478, 117)
    cigpday = st.sidebar.slider('Cigarettes Per Day:', 0, 90, 0)
    ldlc = st.sidebar.slider('LDLC:', 20, 565, 180) 
    hdlc = st.sidebar.slider('HDLC:', 10, 189, 80) 
    stroke = st.sidebar.selectbox('Stroke:', (0, 1))
    cursmoke = st.sidebar.selectbox('Current Smoker:', (0, 1))   
    diabetes = st.sidebar.selectbox('Diabetes:', (0, 1))
    bpmeds = st.sidebar.selectbox('On BP Meds:', (0, 1))
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
        'HYPERTEN': hyperten,
        'LDLC': ldlc,
        'HDLC': hdlc
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Ensure input_df columns match the trained model feature columns
input_df = input_df[feature_columns]

# Clean feature names to ensure no hidden characters or spaces
input_df.columns = input_df.columns.str.strip()

# Explicitly set data types to match model expectations
input_df = input_df.astype({
    'AGE': 'int64',
    'TOTCHOL': 'int64',
    'SYSBP': 'int64',
    'DIABP': 'int64',
    'BMI': 'float64',
    'CURSMOKE': 'int64',
    'GLUCOSE': 'int64',
    'DIABETES': 'int64',
    'HEARTRTE': 'int64',
    'CIGPDAY': 'int64',
    'BPMEDS': 'int64',
    'STROKE': 'int64',
    'HYPERTEN': 'int64',
    'LDLC': 'int64',
    'HDLC': 'int64'
})

# Apply the model to make predictions
if st.sidebar.button('PREDICT NOW'):
    try:
        rf_proba_calibrated = rf_model_calibrated.predict_proba(input_df)[:, 1]
        gbm_proba_calibrated = gbm_model_calibrated.predict_proba(input_df)[:, 1]
    except Exception as e:
        st.error(f"Error making predictions: {e}")

    st.write("""
    ## Your CVD Probability Prediction Results
    This app predicts the probability of cardiovascular disease (CVD) using user inputs.
    """)

    st.subheader('Predictions')
    try:
        st.write(f"- Random Forest model: Your CVD with probability prediction is {rf_proba_calibrated[0]:.2f}")
        st.write(f"- Gradient Boosting Machine model: Your CVD with probability prediction is {gbm_proba_calibrated[0]:.2f}")
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

    # Plot ROC curve for both models
    st.subheader('Model Performance')
    try:
        fig, ax = plt.subplots()
        fpr_rf, tpr_rf, _ = roc_curve(data['CVD'], rf_model_calibrated.predict_proba(data[feature_columns])[:, 1])
        fpr_gbm, tpr_gbm, _ = roc_curve(data['CVD'], gbm_model_calibrated.predict_proba(data[feature_columns])[:, 1])
        ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(data["CVD"], rf_model_calibrated.predict_proba(data[feature_columns])[:, 1]):.2f})')
        ax.plot(fpr_gbm, tpr_gbm, label=f'Gradient Boosting Machine (AUC = {roc_auc_score(data["CVD"], gbm_model_calibrated.predict_proba(data[feature_columns])[:, 1]):.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='best')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting ROC curve: {e}")

    # Plot feature importances for Random Forest
    st.subheader('Risk Factors / Feature Importances (RF)')
    try:
        rf_base_model = rf_model_calibrated.estimator  # Access the base estimator
        fig, ax = plt.subplots()
        importances = rf_base_model.feature_importances_
        indices = np.argsort(importances)
        ax.barh(range(len(indices)), importances[indices], color='blue', align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_columns[i] for i in indices])
        ax.set_xlabel('Importance')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting feature importances: {e}")

    # Add explanations for the features
    st.markdown("""
    - **Stroke:** The history of stroke is the most significant factor.
    - **BMI (Body Mass Index):** Higher BMI indicates higher risk.
    - **SYSBP (Systolic Blood Pressure):** Elevated systolic blood pressure is a critical indicator.
    - **TOTCHOL (Total Cholesterol):** Higher cholesterol levels contribute to the risk.
    - **GLUCOSE:** Higher glucose levels are also important in the prediction.
    - **AGE:** Older age increases the risk of CVD.
    - **DIABP (Diastolic Blood Pressure):** Elevated diastolic blood pressure plays a role.
    - **HEARTRTE (Heart Rate):** Higher heart rate is a contributing factor.
    - **CIGPDAY (Cigarettes Per Day):** The number of cigarettes smoked per day impacts the risk.
    - **BPMEDS (Blood Pressure Medication):** Use of BP medication is taken into account.
    - **HYPERTEN (Hypertension):** Having hypertension is a minor but notable factor.
    - **DIABETES:** The presence of diabetes is a minor factor in this prediction.
    - **CURSMOKE (Current Smoker):** Whether the individual is currently smoking has the least impact compared to other factors.
    """)

else:
    st.write("## CVD Prediction App by Howard Nguyen")
    st.write("#### Enter your parameters and click Predict to get the results.")
