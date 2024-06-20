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

# Load the models
try:
    rf_model_calibrated = joblib.load(rf_model_path)
    gbm_model_calibrated = joblib.load(gbm_model_path)
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Load the dataset
data_url = 'https://raw.githubusercontent.com/HowardHNguyen/cvd/master/frmgham2.csv'
try:
    data = pd.read_csv(data_url)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

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

# Debug: Print the input dataframe to verify
st.write("### Input DataFrame")
st.write(input_df)

# Check the distribution of the target variable
st.write("### Distribution of Target Variable (CVD)")
st.write(data['CVD'].value_counts(normalize=True))

# Apply the model to make predictions
if st.sidebar.button('PREDICT NOW'):
    try:
        rf_proba_calibrated = rf_model_calibrated.predict_proba(input_df)[:, 1]
        gbm_proba_calibrated = gbm_model_calibrated.predict_proba(input_df)[:, 1]
        st.write("### Debug: Model Predictions")
        st.write(f"Random Forest Probability: {rf_proba_calibrated}")
        st.write(f"Gradient Boosting Machine Probability: {gbm_proba_calibrated}")
    except Exception as e:
        st.error(f"Error making predictions: {e}")

    st.write("""
    ## Your CVD Probability Prediction Results
    This app predicts the probability of cardiovascular disease (CVD) using user inputs.
    """)

    st.subheader('Predictions')
    try:
        st.write(f"- Random Forest model: Your CVD probability prediction is {rf_proba_calibrated[0]:.2f}")
        st.write(f"- Gradient Boosting Machine model: Your CVD probability prediction is {gbm_proba_calibrated[0]:.2f}")
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
        st.write("### Debug: ROC Data")
        st.write(f"Random Forest: FPR={fpr_rf}, TPR={tpr_rf}")
        st.write(f"Gradient Boosting Machine: FPR={fpr_gbm}, TPR={tpr_gbm}")
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

    # Plot calibration curve for both models
    st.subheader('Calibration Curve')
    try:
        fig, ax = plt.subplots()
        prob_true_rf, prob_pred_rf = calibration_curve(data['CVD'], rf_model_calibrated.predict_proba(data[feature_columns])[:, 1], n_bins=10)
        prob_true_gbm, prob_pred_gbm = calibration_curve(data['CVD'], gbm_model_calibrated.predict_proba(data[feature_columns])[:, 1], n_bins=10)
        st.write("### Debug: Calibration Curve Data")
        st.write(f"Random Forest: True={prob_true_rf}, Predicted={prob_pred_rf}")
        st.write(f"Gradient Boosting Machine: True={prob_true_gbm}, Predicted={prob_pred_gbm}")
        ax.plot(prob_pred_rf, prob_true_rf, marker='o', label='Random Forest')
        ax.plot(prob_pred_gbm, prob_true_gbm, marker='s', label='Gradient Boosting Machine')
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('True Probability')
        ax.set_title('Calibration Curve')
        ax.legend(loc='best')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting calibration curve: {e}")

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

    # Print feature importances for GBM
    st.write("### Feature Importances for Gradient Boosting Machine")
    try:
        gbm_base_model = gbm_model_calibrated.estimator  # Access the base estimator
        gbm_importances = gbm_base_model.feature_importances_
        gbm_feature_importances = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': gbm_importances
        }).sort_values(by='Importance', ascending=False)
        st.write(gbm_feature_importances)
    except Exception as e:
        st.error(f"Error printing GBM feature importances: {e}")

else:
    st.write("## CVD Prediction App by Howard Nguyen")
    st.write("#### Enter your parameters and click Predict to get the results.")
