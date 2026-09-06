
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
        if not os.path.exists(os.path.join(save_path, 'xgboost_model.pkl')):
            st.error("XGBoost model file not found at " + os.path.join(save_path, 'xgboost_model.pkl') + ". Please ensure it's saved.")
            # Fallback for demonstration if model file is truly missing
            import xgboost as xgb
            dummy_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, n_jobs=-1)
            dummy_model.fit(np.random.rand(10, len(selected_features)), np.random.randint(0, 2, 10))
            models['XGBoost'] = dummy_model
        else:
            models['XGBoost'] = joblib.load(os.path.join(save_path, 'xgboost_model.pkl'))

    except Exception as e:
        st.error(f"Error loading or initializing model: {e}")
        # Fallback for demonstration, ensure models dictionary is not empty
        import xgboost as xgb # Ensure xgb is imported for fallback
        models['XGBoost'] = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, n_jobs=-1)
        models['XGBoost'].fit(np.random.rand(10, len(selected_features)), np.random.randint(0, 2, 10))
    return models

models = load_all_models()

# --- Load Scaler and Label Encoders (for preprocessing user inputs) ---
@st.cache_resource
def load_preprocessors():
    scaler = StandardScaler()
    le_kdrw = LabelEncoder()
    le_ace1 = LabelEncoder()

    try:
        if os.path.exists(os.path.join(save_path, 'scaler.pkl')):
            scaler = joblib.load(os.path.join(save_path, 'scaler.pkl'))
        else:
            st.error("Scaler file not found at " + os.path.join(save_path, 'scaler.pkl') + ". Please ensure it's saved.")
            # Fallback
            scaler.fit(np.random.rand(100, len(biochem_features + env_features)))

        if os.path.exists(os.path.join(save_path, 'le_kdrw.pkl')) and os.path.exists(os.path.join(save_path, 'le_ace1.pkl')):
            le_kdrw = joblib.load(os.path.join(save_path, 'le_kdrw.pkl'))
            le_ace1 = joblib.load(os.path.join(save_path, 'le_ace1.pkl'))
        else:
            st.error("LabelEncoder files not found. Please ensure they are saved.")
            # Fallback
            le_kdrw.fit(['SS', 'RS', 'RR', 'None'])
            le_ace1.fit(['SS', 'RS', 'RR', 'None'])

    except Exception as e:
        st.error(f"Error loading or fitting preprocessors: {e}")
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
st.markdown("Enter feature values or upload a file to predict insecticide resistance (Susceptible/Resistant).")

# Model Selection
selected_model_name = st.sidebar.selectbox(
    "Select Prediction Model",
    list(models.keys())
)
model_to_use = models[selected_model_name]

# --- User Input Features Section ---
st.sidebar.header("Input Feature Values (Single Entry)")

input_data = {}

# Biochemical features (continuous)
st.sidebar.subheader("Biochemical Markers")
for feature in biochem_features:
    input_data[feature] = st.sidebar.number_input(f"Enter {feature.replace('_', ' ').title()}", value=0.0, format="%.4f", key=f"single_{feature}")

# Genotype features (categorical)
st.sidebar.subheader("Genotype Markers")
input_data['kdr-w'] = st.sidebar.selectbox("Select Kdr-w Genotype", options=le_kdrw.classes_, index=int(np.where(le_kdrw.classes_ == 'SS')[0][0]) if 'SS' in le_kdrw.classes_ else 0, key="single_kdr-w")
input_data['ace-1'] = st.sidebar.selectbox("Select Ace-1 Genotype", options=le_ace1.classes_, index=int(np.where(le_ace1.classes_ == 'SS')[0][0]) if 'SS' in le_ace1.classes_ else 0, key="single_ace-1")

# Environmental features (continuous)
st.sidebar.subheader("Environmental Factors")
for feature in env_features:
    input_data[feature] = st.sidebar.number_input(f"Enter {feature.replace('_', ' ').title()}", value=0.0, format="%.4f", key=f"single_{feature}")

# --- Prediction for Single Entry ---
if st.sidebar.button("Predict Resistance (Single Entry)"):
    # Create DataFrame from inputs
    input_df = pd.DataFrame([input_data])

    # Preprocess inputs
    # Encode categorical features
    input_df['kdr-w'] = le_kdrw.transform(input_df['kdr-w'].astype(str))
    input_df['ace-1'] = le_ace1.transform(input_df['ace-1'].astype(str))

    # Scale continuous features
    input_df_scaled = input_df.copy()
    input_df_scaled[biochem_features + env_features] = scaler.transform(input_df[biochem_features + env_features])

    # Order columns to match training data (important for consistent predictions)
    input_df_processed = input_df_scaled[selected_features]

    st.subheader(f"Prediction using {selected_model_name} (Single Entry):")

    # Scikit-learn model prediction
    prediction = model_to_use.predict(input_df_processed)[0]
    prediction_proba = model_to_use.predict_proba(input_df_processed)[0]
    st.write(f"Predicted Class: **{'Resistant' if prediction == 1 else 'Susceptible'}**")
    st.write(f"Resistance Probability: {prediction_proba[1]*100:.2f}%")

    st.markdown("---")
    st.subheader("Input Data Summary (Single Entry)")
    st.dataframe(input_df_processed)


# --- File Upload Section ---
st.header("Upload File for Batch Prediction")
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            batch_df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            batch_df = pd.read_excel(uploaded_file)

        st.subheader("Uploaded Data Preview:")
        st.dataframe(batch_df.head())

        if st.button("Run Batch Prediction"):
            # Ensure all selected_features are in the uploaded dataframe
            missing_cols = [col for col col in selected_features if col not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing columns in uploaded file: {', '.join(missing_cols)}. Please upload a file with all required features.")
            else:
                # Select only the relevant features for prediction
                batch_df_selected = batch_df[selected_features].copy()

                # Preprocess batch data
                # Encode categorical features (kdr-w, ace-1)
                for col in genotype_features:
                    if col in batch_df_selected.columns:
                        le = le_kdrw if col == 'kdr-w' else le_ace1
                        batch_df_selected[col] = batch_df_selected[col].astype(str).fillna('None')
                        
                        # Check for unknown categories and handle gracefully
                        unknown_categories = set(batch_df_selected[col].unique()) - set(le.classes_)
                        if unknown_categories:
                            st.warning(f"Unknown categories found in '{col}' in uploaded data: {', '.join(unknown_categories)}. These will be mapped to 'None' if 'None' is in encoder classes, otherwise treated as an error.")
                            # Map unknown categories to 'None' if 'None' was seen during training
                            # Otherwise, the transform will raise an error, which is desired.
                            if 'None' in le.classes_:
                                batch_df_selected[col] = batch_df_selected[col].apply(lambda x: 'None' if x in unknown_categories else x)

                        batch_df_selected[col] = le.transform(batch_df_selected[col])

                # Scale continuous features
                batch_df_scaled = batch_df_selected.copy()
                batch_df_scaled[biochem_features + env_features] = scaler.transform(batch_df_scaled[biochem_features + env_features])

                # Make predictions
                batch_predictions = model_to_use.predict(batch_df_scaled)
                batch_probabilities = model_to_use.predict_proba(batch_df_scaled)[:, 1] # Probability of resistant (class 1)

                # Add predictions to the original dataframe for display
                batch_df['Predicted_Resistance'] = np.where(batch_predictions == 1, 'Resistant', 'Susceptible')
                batch_df['Resistance_Probability'] = batch_probabilities * 100

                st.subheader(f"Batch Predictions using {selected_model_name}:")
                st.dataframe(batch_df[['Predicted_Resistance', 'Resistance_Probability']])
                st.download_button(
                    label="Download Predictions",
                    data=batch_df.to_csv(index=False).encode('utf-8'),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")

