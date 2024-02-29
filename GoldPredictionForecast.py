import streamlit as st
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pickle

# Load data
data = pd.read_csv('gold_price_data.csv')
# Load the machine learning model for Project 1
with open('GoldPricePredictormodel.pkl', 'rb') as file:
    model_project1 = pickle.load(file)

# Load the time series forecasting model for Project 2
model_project2 = load_model('gold_price_forecasting_model.h5')

# Streamlit app title and header
st.markdown("<h1 style = 'color: #070F2B; text-align: center; font-family: helvetica '>Gold Price Forecasting: Leveraging Machine Learning and Time Series Analysis</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #820300; text-align: center; font-family: Trebuchet MS (sans-serif)': cursive '>Built By SapphireDataChic</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)
# Image
st.image('pngwing.com-17.png', width=500, use_column_width=True)

# Overview section
st.markdown('''
## Overview:

Welcome to the Gold Price Forecasting app, Built By SapphireDataChic! This application leverages both machine learning algorithms and time series analysis techniques to predict future gold prices. Gold prices are influenced by various factors such as geopolitical events, economic indicators, market demand, and currency fluctuations. By analyzing historical gold price data and relevant features, we aim to build accurate predictive models that forecast future gold prices.

---
''')

# Sidebar
st.sidebar.image('pngwing.com-19.png', caption='Welcome User')

# Features section
st.markdown('''
## Features:

### 1. Gold Price Prediction Using Machine Learning:

- **Algorithm Selection:** XGBoost algorithms for gold price prediction.
- **Feature Selection:** Select relevant features such as stock market indices (e.g., S&P 500), commodity prices (e.g., oil and silver), and currency exchange rates (e.g., EUR/USD).
- **Optimization:** Perform hyperparameter optimization to find the best model configuration.

### 2. Gold Price Forecasting Using Time Series Analysis:

- **LSTM Model:** Utilize Long Short-Term Memory (LSTM) neural networks for time series forecasting.
- **Temporal Features:** Analyze historical gold price data along with temporal features such as year, month, and day.
- **Visualization:** Visualize historical gold prices and forecasted future prices using interactive charts.

---
''')

# How to Use section
st.markdown('''
## How to Use:

1. Select the desired analysis mode: Machine Learning or Time Series Analysis.
2. Configure the parameters and features for analysis.
3. View the results: Predicted gold prices and forecasted future prices.
4. Analyze the accuracy and performance metrics of the models.

---
''')

# About section
st.markdown('''
## About:

This Gold Price Forecasting app is developed by SapphireDataChic as part of a data science project. The goal is to provide valuable insights into gold price movements and help investors, financial analysts, and policymakers make informed decisions in the gold market.
''')

prediction = None

# Streamlit App for Project 1
if st.sidebar.checkbox("Gold Price Prediction Using Machine Learning"):
    # User input features
    # GLD = st.sidebar.slider("Gold Price", min_value=0.0, max_value=100.0)
    # feature2 = st.sidebar.selectbox("Feature 2", ['Option 1', 'Option 2', 'Option 3'])

    # User input features for the first model
    feature1 = st.sidebar.slider("SPX", min_value=min(data['SPX']), max_value=max(data['SPX']), value=min(data['SPX']))
    feature2 = st.sidebar.slider("USO", min_value=min(data['USO']), max_value=max(data['USO']), value=min(data['USO']))
    feature3 = st.sidebar.slider("SLV", min_value=min(data['SLV']), max_value=max(data['SLV']), value=min(data['SLV']))
    feature4 = st.sidebar.slider("GLD", min_value=min(data['GLD']), max_value=max(data['GLD']), value=min(data['GLD']))

    input_var = pd.DataFrame({
                'SPX': [feature1],
                'USO': [feature2],
                'SLV': [feature3],
                'GLD': [feature4]
    })


    # Make prediction using loaded model
    prediction = model_project1.predict(input_var)


predict_button = st.button('Push To Predict')
if predict_button:
    # Display prediction
    st.success(f"Predicted Value Of Gold Is {round(prediction[0], 1)}")




# Streamlit App for Project 2
# if st.sidebar.checkbox("Gold-Price-Forecasting Using Time Series Forecasting"):
#Load historical gold price data
historical_data = pd.read_csv('gold_price_data.csv')
    # feature_year = st.sidebar.slider("Year", min_value=min(data['year']), max_value=max(data['year']), value=min(data['year']))
    # feature_month = st.sidebar.slider("Month", min_value=min(data['month']), max_value=max(data['month']), value=min(data['month']))
    # feature_day = st.sidebar.slider("Day", min_value=min(data['day']), max_value=max(data['day']), value=min(data['day']))

st.dataframe(historical_data)
selected_data['Date'] = selected_data['Date'].astype(str)
selected_data[['year', 'month', 'day']] = selected_data['Date'].str.split('-', expand=True)
selected_data
#     # Make prediction using loaded model
# prediction = model_project1.predict([[feature1, feature2, feature3, feature4, feature5]])

# # # Display prediction
# st.write("Predicted Gold Price (Model 1):", prediction)

# # # User input features for the second model

# # # Make forecast using loaded model
# # # You need to complete this part by providing the necessary input data for forecasting
# # # For example, you may need to preprocess the data and reshape it as required by the model
# forecast_input = [[feature_gld, feature_year, feature_month, feature_day]]
# forecast = model_project2.predict(forecast_input)

# # # Display forecasted prices
# st.write("Forecasted Gold Prices (Model 2):")
# st.write(forecast)

