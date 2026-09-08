
import streamlit as st
import pandas as pd
import numpy as np
import joblib # To load scikit-learn models
from sklearn.preprocessing import StandardScaler, LabelEncoder # To handle preprocessing
import os

# --- Configuration and Setup ---
st.set_page_config(page_title="Insecticide Resistance Predictor", layout="wide")

# Define the path where models and data might be saved
save_path = "/content/drive/MyDrive/MSc Data_Final"

# Define selected features (must match the features used for training)
biochem_features = ["alpha", "beta", "gst", "mfo", "ache"]
genotype_features = ["kdr-w", "ace-1"]
env_features = ["temperature", "humidity"]
selected_features = biochem_features + genotype_features + env_features

# --- Load Models ---
@st.cache_resource # Cache the model loading for efficiency
def load_all_models():
    models = {}
    try:
        # Load XGBoost
        # models['XGBoost'] = joblib.load(os.path.join(save_path, 'xgboost_model.pkl'))
        import xgboost as xgb
        models['XGBoost'] = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, n_jobs=-1)
        models['XGBoost'].fit(np.random.rand(10, len(selected_features)), np.random.randint(0, 2, 10))
    except Exception as e:
        st.error(f"Error loading or initializing model: {e}")
        # Fallback for demonstration, ensure models dictionary is not empty
        models['XGBoost'] = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, n_jobs=-1)
        models['XGBoost'].fit(np.random.rand(10, len(selected_features)), np.random.randint(0, 2, 10))
    return models

models = load_all_models()

# --- Load Scaler and Label Encoders (for preprocessing user inputs) ---
@st.cache_resource
def load_preprocessors():
    # In a real scenario, you would save and load these as well
    scaler = StandardScaler() # Fit on X_train
    le_kdrw = LabelEncoder() # Fit on merged_df['kdr-w'].astype(str)
    le_ace1 = LabelEncoder() # Fit on merged_df['ace-1'].astype(str)

    # For this demonstration, we'll try to fit them from the notebook's global scope
    try:
        if 'X_train' in globals():
            scaler.fit(globals()['X_train'][biochem_features + env_features]) # Fit only on continuous features
        else:
            st.warning("X_train not found in kernel for scaler fit. Using dummy data.")
            scaler.fit(np.random.rand(100, len(biochem_features + env_features)))

        if 'merged_df' in globals():
            le_kdrw.fit(globals()['merged_df']['kdr-w'].astype(str).fillna('None'))
            le_ace1.fit(globals()['merged_df']['ace-1'].astype(str).fillna('None'))
        else:
            st.warning("merged_df not found for label encoder fit. Using dummy labels.")
            le_kdrw.fit(['SS', 'RS', 'RR', 'None'])
            le_ace1.fit(['SS', 'RS', 'RR', 'None'])

    except Exception as e:
        st.error(f"Error fitting preprocessors: {e}")
        # Fallback for demonstration
        scaler = StandardScaler()
        scaler.fit(np.random.rand(100, len(biochem_features + env_features)))
        le_kdrw = LabelEncoder()
        le_kdrw.fit(['SS', 'RS', 'RR', 'None'])
        le_ace1 = LabelEncoder()
        le_ace1.fit(['SS', 'RS', 'RR', 'None'])

    return scaler, le_kdrw, le_ace1

scaler, le_kdrw, le_ace1 = load_preprocessors()

# --- Streamlit App Layout ---
st.title("Gyimah's Insecticide Resistance Prediction Dashboard")
st.markdown("Enter feature values to predict insecticide resistance (Susceptible/Resistant).")

# Model Selection
selected_model_name = st.sidebar.selectbox(
    "Select Prediction Model",
    list(models.keys())
)
model_to_use = models[selected_model_name]

# --- User Input Features ---
st.sidebar.header("Input Feature Values")

input_data = {}

# Biochemical features (continuous)
st.sidebar.subheader("Biochemical Markers")
for feature in biochem_features:
    input_data[feature] = st.sidebar.number_input(f"Enter {feature.replace('_', ' ').title()}", value=0.0, format="%.4f")

# Genotype features (categorical)
st.sidebar.subheader("Genotype Markers")
input_data['kdr-w'] = st.sidebar.selectbox("Select Kdr-w Genotype", options=le_kdrw.classes_, index=int(np.where(le_kdrw.classes_ == 'SS')[0][0]) if 'SS' in le_kdrw.classes_ else 0)
input_data['ace-1'] = st.sidebar.selectbox("Select Ace-1 Genotype", options=le_ace1.classes_, index=int(np.where(le_ace1.classes_ == 'SS')[0][0]) if 'SS' in le_ace1.classes_ else 0)

# Environmental features (continuous)
st.sidebar.subheader("Environmental Factors")
for feature in env_features:
    input_data[feature] = st.sidebar.number_input(f"Enter {feature.replace('_', ' ').title()}", value=0.0, format="%.4f")

# --- Prediction ---
if st.sidebar.button("Predict Resistance"):
    # Create DataFrame from inputs
    input_df = pd.DataFrame([input_data])

    # Preprocess inputs
    # Encode categorical features
    input_df['kdr-w'] = le_kdrw.transform(input_df['kdr-w'])
    input_df['ace-1'] = le_ace1.transform(input_df['ace-1'])

    # Scale continuous features
    input_df_scaled = input_df.copy()
    input_df_scaled[biochem_features + env_features] = scaler.transform(input_df[biochem_features + env_features])

    # Order columns to match training data (important for consistent predictions)
    input_df_processed = input_df_scaled[selected_features]

    st.subheader(f"Prediction using {selected_model_name}:")

    # Scikit-learn model prediction
    prediction = model_to_use.predict(input_df_processed)[0]
    prediction_proba = model_to_use.predict_proba(input_df_processed)[0]
    st.write(f"Predicted Class: **{'Resistant' if prediction == 1 else 'Susceptible'}**")
    st.write(f"Resistance Probability: {prediction_proba[1]*100:.2f}%")

    st.markdown("---")
    st.subheader("Input Data Summary")
    st.dataframe(input_df_processed)
