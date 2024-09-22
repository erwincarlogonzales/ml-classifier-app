import pandas as pd
import numpy as np
import streamlit as st
import joblib
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import config

from scripts.preprocessing import create_preprocessing_pipeline
from scripts.categorical_encoder import encode_categorical

# Define st_shap to embed SHAP plots in Streamlit
def st_shap(plot):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html)

@st.cache_resource
# Function to load the models
def load_models():
    lgbm_model = joblib.load(config.MODEL_LGBM_PATH)
    dec_tree = joblib.load(config.MODEL_DEC_TREE_PATH)
    return lgbm_model, dec_tree

@st.cache_data
# Function to load and fit the preprocessor
def load_and_fit_preprocessor(file_path):
    flight_df = pd.read_csv(file_path)
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(flight_df)
    return preprocessor, flight_df

# Function to collect user inputs
def user_input_features():
    inputs = {
        'Online boarding': st.sidebar.slider('Online Boarding 🎫', 1, 5, 3),
        'Inflight wifi service': st.sidebar.slider('Inflight WIFI service 🛜', 1, 5, 3),
        'Inflight entertainment': st.sidebar.slider('Inflight Entertainment 🎮', 1, 5, 3),
        'Checkin service': st.sidebar.slider('Checkin Service ✅', 1, 5, 3),
        'Seat comfort': st.sidebar.slider('Seat Comfort 💺', 1, 5, 3),
        'Age': st.sidebar.number_input('Age 📆', 0, 100, 18),
        'Flight Distance': st.sidebar.number_input('Flight Distance 🛫', 0, 10000, 100),
        'Business Travel': st.sidebar.selectbox('Business Travel 💼', ['Yes', 'No']),
        'Loyal Customer': st.sidebar.selectbox('Loyal Customer 🤗', ['Yes', 'No']),
        'Class': st.sidebar.selectbox('Class 💼', ['Eco', 'Eco Plus', 'Business'])
    }
    return pd.DataFrame(inputs, index=[0])

# Function to make predictions and explain the results
def make_prediction(model, preprocessor, input_df, explainer, lime_explainer):
    try:
        input_df_transformed = preprocessor.transform(encode_categorical(input_df))
        prediction = model.predict(input_df_transformed)

        # Get SHAP values for the prediction
        shap_values = explainer.shap_values(input_df_transformed)

        # If shap_values is a list (for classification), use the appropriate class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Assuming binary classification, use the second class

        # Generate LIME explanation
        lime_explanation = lime_explainer.explain_instance(
            data_row=input_df_transformed[0],
            predict_fn=model.predict_proba,
            num_features=config.LIME_NUM_FEATURES
        )

        return 'Satisfied 😊' if prediction[0] else 'Not Satisfied 😞', shap_values, explainer.expected_value, input_df_transformed, lime_explanation

    except Exception as e:
        st.error(f"Error occurred during prediction: {str(e)}. Please contact support.")
        return None, None, None, None, None

# Display logo and hero image
st.sidebar.image('images/logo.png', width=100)
st.image('images/hero.jpg', use_column_width=True)

st.title('Cebu Pacific Customer Flight Satisfaction ✈️')
st.header('Please Tell Us About Your Experience! 🤩')

# Load models and preprocessor
lgbm_model, dec_tree = load_models()
preprocessor, flight_df = load_and_fit_preprocessor(config.DATA_PATH)

# Model selection
model_choice = st.sidebar.selectbox('Choose Model 🛠️', ['LightGBM', 'Decision Tree'])
selected_model = lgbm_model if model_choice == 'LightGBM' else dec_tree

# Initialize SHAP and LIME Explainers based on model type
explainer = shap.TreeExplainer(selected_model)
lime_explainer = LimeTabularExplainer(
    training_data=preprocessor.transform(encode_categorical(flight_df)),
    feature_names=flight_df.columns,
    class_names=['Not Satisfied', 'Satisfied'],
    mode='classification'
)

# Add app description
st.write("""
    This app predicts flight satisfaction based on user inputs.
    You can select between two machine learning models: LightGBM and Decision Tree ✨
""")

# Collect user inputs
st.sidebar.header('User Input')
input_df = user_input_features()

# Display user inputs
st.write('### User Inputs')
for key, value in input_df.iloc[0].items():
    st.write(f'**{key}**: {value}')
    
# Predict button
if st.button('Make Prediction ✨'):
    result, shap_values, expected_value, input_df_transformed, lime_explanation = make_prediction(
        selected_model, preprocessor, input_df, explainer, lime_explainer
    )
    if result:
        st.write(f'### Prediction: {result}')
        
        # Display SHAP force plot
        # st.write('### SHAP Force Plot:')
        # st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0], input_df_transformed[0]))
        
        # Display LIME explanation
        st.write('### LIME Explanation:')
        lime_html = lime_explanation.as_html()
        st.components.v1.html(lime_html)
        
# Example usage > in the terminal type: streamlit run app-prod.py