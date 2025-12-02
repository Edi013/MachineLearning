import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping

# 1. Download stock data
symbol = "TSLA"
data = yf.download(symbol, start="2018-01-01", end="2025-12-01")  # adjust dates
data = data[['Close']]  # Use only closing price

# 2. Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. Prepare sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # GRU expects 3D input
    return X, y

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

# Split into train/test
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 4. Build GRU model
model = Sequential()
model.add(GRU(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(GRU(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

# 5. Training
early_stop = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop]
)

# 6. Evaluate & predict
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# 7. Results
plt.figure(figsize=(12,6))
plt.plot(actual_prices, color='blue', label='Actual TSLA Price')
plt.plot(predicted_prices, color='red', label='Predicted TSLA Price')
plt.title(f'{symbol} Stock Price Prediction using GRU')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
