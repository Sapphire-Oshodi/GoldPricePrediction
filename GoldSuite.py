import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import streamlit as st
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import timedelta, date, datetime


# Streamlit app title and header
st.markdown("<h1 style='color: #070F2B; text-align: center; font-family: helvetica'>Gold Intelligence Suite: Empowering Investors with Prediction and Forecasting</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='margin: -30px; color: #820300; text-align: center; font-family: Trebuchet MS, sans-serif;'>Built By SapphireDataChic</h4>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.image('pngwing.com-17.png', width=500, use_column_width=True)

# Sidebar
st.sidebar.image('pngwing.com-19.png', caption='Welcome User')

# Function to display About section
def display_about():
    st.header("About Gold Intelligence Suite")
    st.markdown("""
    <p style='font-size: 16px; font-family: Arial, sans-serif; color: #070F2B;'>Welcome to Gold Intelligence Suite, the two-in-one app for gold price prediction and forecasting, built by SapphireDataChic! Our application combines advanced machine learning algorithms with time series analysis techniques to predict future gold prices accurately.</p>

    <h2 style='font-size: 18px; font-family: Arial, sans-serif; color: #070F2B;'>Our Approach</h2>
    <p style='font-size: 16px; font-family: Arial, sans-serif; color: #070F2B;'>Developing this app, we take a meticulous approach to forecasting gold prices. We harness the power of historical gold price data and relevant features to develop robust predictive models. By carefully analyzing past trends and incorporating key factors that influence gold prices, our models are designed to provide accurate forecasts.</p>

    <h2 style='font-size: 18px; font-family: Arial, sans-serif; color: #070F2B;'>Features</h2>
    <ul style='font-size: 16px;'>
    <li>Machine Learning Prediction: Predict gold prices using machine learning models.</li>
    <li>Time Series Forecasting: Forecast gold prices using time series analysis.</li>
    <li>Customizable Inputs: Adjust features such as SPX, USO, SLV, and GLD to tailor predictions to your needs.</li>
    </ul>
    """, unsafe_allow_html=True)

# Function to display How to Use section
def display_how_to_use():
    st.header("How to Use")
    st.markdown("""
    1. Select "Machine Learning Prediction" to predict gold prices using machine learning models.
    2. Adjust the sliders for features such as SPX, USO, SLV, and GLD.
    3. Click the "Predict" button to see the predicted gold price.
    4. Select "Time Series Forecasting" to forecast gold prices using time series analysis.
    
    Thank you for choosing the Gold Intelligence Suite for your predictive analysis needs. Let's unlock the potential of gold price prediction together!
    """)

# Function to display the Predict section
def display_predict():
    st.header("Gold Price Prediction")
    # Read the gold price data
    data = pd.read_csv('gold_price_data.csv')

    # Display the historical data with a header
    st.write(data)

# Create a main menu in the sidebar using selectbox
welcome_menu = st.sidebar.selectbox(
    'Select Analysis Type',
    ('About', 'How to Use', 'Predict | Forecast')
)
# Display selected main menu section
if welcome_menu == 'About':
    display_about()
elif welcome_menu == 'How to Use':
    display_how_to_use()
elif welcome_menu == 'Gold Price Prediction':  
    display_predict()  # Call the function to display the Predict section


# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('gold_price_data.csv', parse_dates=["Date"])
    return data

data = load_data()

# Add checkbox for Gold Price Prediction
if st.sidebar.checkbox("Gold Price Prediction"):

    # Sidebar inputs for Gold Price Prediction
    spx = st.sidebar.slider('SPX', min_value=min(data['SPX']), max_value=max(data['SPX']), value=min(data['SPX']))
    gld = st.sidebar.slider('GLD', min_value=min(data['GLD']), max_value=max(data['GLD']), value=min(data['GLD']))
    uso = st.sidebar.slider('USO', min_value=min(data['USO']), max_value=max(data['USO']), value=min(data['USO']))
    slv = st.sidebar.slider('SLV', min_value=min(data['SLV']), max_value=max(data['SLV']), value=min(data['SLV']))

    # Button to trigger prediction
    if st.button('Predict'):
        # Train and Test Split
        x = data.drop(['Date', 'EUR/USD'], axis=1)
        y = data['EUR/USD']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=10)

        # Feature Scaling
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # XGBoost model
        model_xgb = XGBRegressor()
        model_xgb.fit(x_train_scaled, y_train)

        # Prediction function
        def predict_price(spx, gld, uso, slv):
            input_data = pd.DataFrame({
                'SPX': [spx],
                'GLD': [gld],
                'USO': [uso],
                'SLV': [slv]
            })
            input_data_scaled = scaler.transform(input_data)
            prediction = model_xgb.predict(input_data_scaled)
            return prediction[0]

        # Perform prediction for gold price
        prediction = predict_price(spx, gld, uso, slv)
        # Display the predicted gold price with 4 decimal places
        st.write(f"Predicted Gold Price is: ${prediction:.4f}")

        # Display an image related to the prediction
        st.image("pngwing.com-12.png", caption="Visualization of Gold Price Prediction", use_column_width=True)


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('gold_price_data.csv', parse_dates=["Date"])
    return data

def perform_forecasting(data, model, selected_date):
    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Split data into train and test sets
    train_size = int(len(data) * 0.75)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['GLD']])
    
    # Prepare test data
    time_steps = 60
    test_data = scaled_data[train_size - time_steps:, :]
    x_test = []
    n_cols = 1

    # Create input sequences for the model
    for i in range(time_steps, len(test_data)):
        x_test.append(test_data[i - time_steps:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], n_cols))
    
    # Pad sequences
    x_test_padded = pad_sequences(x_test, maxlen=100, dtype='float32', padding='pre')
    
    # Make predictions
    predictions = model.predict(x_test_padded)
    
    # Invert scaling to get actual prices
    predictions = scaler.inverse_transform(predictions)

    # Create DataFrame for forecasted prices with dates
    forecast_df = pd.DataFrame({
        'Date': data['Date'].iloc[-len(predictions):],  # Dates
        'Actual Price': data['GLD'].iloc[-len(predictions):],  # Actual prices
        'Predicted Price': predictions.flatten()  # Predicted prices
    })

    # Convert selected_date to a Pandas datetime object
    selected_date = pd.Timestamp(selected_date)

    # Filter the forecasted DataFrame based on the selected date
    forecast_df = forecast_df[forecast_df['Date'] >= selected_date]

    return forecast_df


# Main function
def main():
    # Load data
    new_data = load_data()
    
    # Get the minimum and maximum dates from the dataset
    min_date = new_data['Date'].min().date()
    max_date = new_data['Date'].max().date()

    # Calculate the default value as the middle of the range
    default_date = min_date + (max_date - min_date) // 2

    # Create a sidebar checkbox for Gold Price Forecasting
    forecasting_enabled = st.sidebar.checkbox("Gold Price Forecasting")

    # Load the trained model for Gold Price Forecasting
    model = load_model("1GoldForecastingmodel.h5")

    # Display selected sections based on the checkbox state
    if forecasting_enabled:
        st.markdown("<h3 style='color: #070F2B; text-align: left; font-family: Arial'>Gold Price Forecasting</h3>", unsafe_allow_html=True)
        
        # Date input widget with restricted date range using slider
        selected_date = st.sidebar.slider("Select a Date", min_value=min_date, max_value=max_date, value=default_date)

        # Button to trigger forecasting
        if st.button('Forecast'):
            # Perform forecasting
            forecast_df = perform_forecasting(new_data, model, selected_date)

            # Display the forecasted DataFrame
            st.write("Forecasted Gold Prices:")
            st.write(forecast_df)

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(new_data['Date'], new_data['GLD'], label='Historical Data')
            ax.plot(forecast_df['Date'], forecast_df['Predicted Price'], label='Predictions')
            ax.set_xlabel('Date')
            ax.set_ylabel('Gold Price')
            ax.set_title('Gold Price Forecast')
            ax.legend()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
