import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle

# Load data
data = pd.read_csv('gold_price_data.csv')
# Load the machine learning model for Project 1
with open('GoldPricePredictormodel.pkl', 'rb') as file:
    model_project1 = pickle.load(file)

# Streamlit app title and header
st.markdown("<h1 style='color: #070F2B; text-align: center; font-family: helvetica'>Gold Price Forecasting: Leveraging Machine Learning and Time Series Analysis</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='margin: -30px; color: #820300; text-align: center; font-family: Trebuchet MS, sans-serif;'>Built By SapphireDataChic</h4>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
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
prediction = None

# Predict button
predict_button = st.button('Push To Predict')

if predict_button:
    # Perform prediction
    prediction = predict_price(spx, gld, uso, slv)
    prediction_rounded = "{:.4f}".format(prediction)
    st.write(f"Predicted Gold Price is $: {prediction_rounded}")

#Model2

# Load data
data2 = pd.read_csv('gold_price_data.csv')

# Define 'selected_data' and split 'Date' column into 'year', 'month', and 'day' columns
selected_data = data2[['Date', 'GLD']]
selected_data['Date'] = selected_data['Date'].astype(str)
date_split = selected_data['Date'].str.split('-', expand=True)
st.dataframe(date_split)
# Print out the split columns
print("Split columns:", date_split.columns)

# # Check if the split resulted in three columns
# if len(date_split.columns) == 3:
#     selected_data[['year', 'month', 'day']] = date_split
#     # Drop the original 'Date' column
#     selected_data.drop(columns=['Date'], inplace=True)
# else:
#     print("Number of columns after splitting:", len(date_split.columns))
#     raise ValueError("Unexpected number of columns after splitting the 'Date' column.")

# # Display the DataFrame without the 'Date' column
# st.write(new_data)

# # Plot gold prices over time
# plt.figure(figsize=(12, 6))
# plt.plot(data2['Date'], new_data['GLD'])
# plt.title('Gold Prices Over Time With Day, Month, Year')
# plt.xlabel('Year')
# plt.ylabel('Gold Price')
# st.pyplot()

# # Create the plot using Seaborn
# sns.set_style("whitegrid")
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=new_data, x='year', y='GLD')
# plt.title('Gold Prices Over Time')
# plt.xlabel('Year')
# plt.ylabel('Gold Price')
# st.pyplot()

# # Define the features and target
# X = new_data[['day', 'month', 'year']]  # Features (input variables)
# y_pred = new_data['GLD']  # Target variable (output variable)

# # Reshape the target variable for compatibility with neural network input
# dataset = y_pred.values.reshape(-1, 1)

# # Scale the data
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(dataset)

# # Define train and test sizes
# train_size = int(len(dataset) * 0.75)
# test_size = len(dataset) - train_size

# # Create training and testing data
# train_data = scaled_data[0:train_size, :]
# test_data = scaled_data[train_size - 60:, :]

# # Define time steps
# time_steps = 60

# # Pad sequences
# x_test_padded = pad_sequences(test_data, maxlen=100, dtype='float32', padding='pre')

# # Load the saved model
# model = load_model("gold_price_forecasting_model.h5")

# # Predictions
# predictions = model.predict(x_test_padded)
# predictions = scaler.inverse_transform(predictions)

# # Plot predictions
# plt.figure(figsize=(16, 6))
# plt.plot(predictions, label='Predictions')
# plt.plot(y_test, label='Actuals')
# plt.legend()
# st.pyplot()
