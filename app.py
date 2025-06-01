import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from tensorflow.keras.models import load_model # type: ignore
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

start = '2010-01-01'
end = dt.datetime.now()

st.title("📈 Stock Price Prediction")

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# ⚠️ Fix here: remove quotes from user_input
df = yf.download(user_input, start=start, end=end)

st.subheader('Data From 2010 to 2024')
st.write(df.describe())

# ✅ Fix: Use capital 'Close'
st.subheader('Closing Price VS Time Chart with 100MA & 200MA')
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price')
plt.plot(ma100, 'r', label='100 MA')
plt.plot(ma200, 'g', label='200 MA')
plt.legend()
st.pyplot(fig)

# Splitting data
data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70):])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_train)

# Load model
model = load_model('keras_model.h5')

# Preparing test data
past_100_days = data_train.tail(100)
final_df = pd.concat([past_100_days, data_test], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

y_predicted = model.predict(x_test)

# Rescale back
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
