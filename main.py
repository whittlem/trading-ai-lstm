import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import pickle
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from eodhd import APIClient

# Configurable hyperparameters
seq_length = 20
batch_size = 64
lstm_units = 50
epochs = 100

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API key
API_TOKEN = os.getenv("API_TOKEN")

if API_TOKEN is not None:
    print(f"API key loaded: {API_TOKEN[:4]}********")
else:
    raise LookupError("Failed to load API key.")

def get_ohlc_data(use_cache: bool = False) -> pd.DataFrame:
    ohlcv_file = "data/ohlcv.csv"

    if use_cache:
        if os.path.exists(ohlcv_file):
            return pd.read_csv(ohlcv_file, index_col=None)
        else:
            api = APIClient(API_TOKEN)
            df = api.get_historical_data(
                symbol="HSPX.LSE",
                interval="d",
                iso8601_start="2010-05-17",
                iso8601_end="2023-10-04",
            )
            df.to_csv(ohlcv_file, index=False)
            return df
    else:
        api = APIClient(API_TOKEN)
        return api.get_historical_data(
            symbol="HSPX.LSE",
            interval="d",
            iso8601_start="2010-05-17",
            iso8601_end="2023-10-04",
        )

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i : i + seq_length])
        y.append(data[i + seq_length, 3])  # The prediction target "close" is the 4th column (index 3)
    return np.array(x), np.array(y)

def get_features(df: pd.DataFrame = None, feature_columns: list = ["open", "high", "low", "close", "volume"]) -> list:
    return df[feature_columns].values

def get_target(df: pd.DataFrame = None, target_column: str = "close") -> list:
    return df[target_column].values

def get_scaler(use_cache: bool = True) -> MinMaxScaler:
    scaler_file = "data/scaler.pkl"

    if use_cache:
        if os.path.exists(scaler_file):
            # Load the scaler
            with open(scaler_file, "rb") as f:
                return pickle.load(f)
        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
            with open(scaler_file, "wb") as f:
                pickle.dump(scaler, f)
            return scaler
    else:
        return MinMaxScaler(feature_range=(0, 1))

def scale_features(scaler: MinMaxScaler = None, features: list = []):
    return scaler.fit_transform(features)

def get_lstm_model(use_cache: bool = False) -> Sequential:
    model_file = "data/lstm_model.h5"

    if use_cache:
        if os.path.exists(model_file):
            # Load the model
            return load_model(model_file)
        else:
            # Train the LSTM model and save it
            model = Sequential()
            model.add(LSTM(units=lstm_units, activation='tanh', input_shape=(seq_length, 5)))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))

            model.compile(optimizer="adam", loss="mean_squared_error")
            model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

            # Save the entire model to a HDF5 file
            model.save(model_file)

            return model

    else:
        # Train the LSTM model
        model = Sequential()
        model.add(LSTM(units=lstm_units, activation='tanh', input_shape=(seq_length, 5)))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

        return model

def get_predicted_x_test_prices(x_test: np.ndarray = None):
    predicted = model.predict(x_test)

    # Create a zero-filled matrix to aid in inverse transformation
    zero_filled_matrix = np.zeros((predicted.shape[0], 5))

    # Replace the 'close' column of zero_filled_matrix with the predicted values
    zero_filled_matrix[:, 3] = np.squeeze(predicted)

    # Perform inverse transformation
    return scaler.inverse_transform(zero_filled_matrix)[:, 3]

def plot_x_test_actual_vs_predicted(actual_close_prices: list = [], predicted_x_test_close_prices = []) -> None:
    # Plotting the actual and predicted close prices
    plt.figure(figsize=(14, 7))
    plt.plot(actual_close_prices, label="Actual Close Prices", color="blue")
    plt.plot(predicted_x_test_close_prices, label="Predicted Close Prices", color="red")
    plt.title("Actual vs Predicted Close Prices")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

def predict_next_close(df: pd.DataFrame = None, scaler: MinMaxScaler = None) -> float:
    # Take the last 20 days of data and scale it
    last_x_days = df.iloc[-seq_length:][["open", "high", "low", "close", "volume"]].values
    last_x_days_scaled = scaler.transform(last_x_days)

    # Reshape this data to be a single sequence and make the prediction
    last_x_days_scaled = np.reshape(last_x_days_scaled, (1, seq_length, 5))

    # Predict the future close price
    future_close_price = model.predict(last_x_days_scaled)

    # Create a zero-filled matrix for the inverse transformation
    zero_filled_matrix = np.zeros((1, 5))

    # Put the predicted value in the 'close' column (index 3)
    zero_filled_matrix[0, 3] = np.squeeze(future_close_price)

    # Perform the inverse transformation to get the future price on the original scale
    return scaler.inverse_transform(zero_filled_matrix)[0, 3]

if __name__ == "__main__":
    # Retrieve 3369 days of S&P 500 data
    df = get_ohlc_data(use_cache=True)
    print(df)

    features = get_features(df)
    target = get_target(df)

    scaler = get_scaler(use_cache=True)
    scaled_features = scale_features(scaler, features)

    x, y = create_sequences(scaled_features, seq_length)

    train_size = int(0.8 * len(x))  # Create a train/test split of 80/20%
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Re-shape input to fit lstm layer
    x_train = np.reshape(x_train, (x_train.shape[0], seq_length, 5))  # 5 features
    x_test = np.reshape(x_test, (x_test.shape[0], seq_length, 5))  # 5 features

    model = get_lstm_model(use_cache=True)

    predicted_x_test_close_prices = get_predicted_x_test_prices(x_test)
    print("Predicted close prices:", predicted_x_test_close_prices)
    # print(len(predicted_x_test_close_prices))

    # Plot the actual and predicted close prices for the test data
    # plot_x_test_actual_vs_predicted(df["close"].tail(len(predicted_x_test_close_prices)).values, predicted_x_test_close_prices)

    # Predict the next close price
    predicted_next_close =  predict_next_close(df, scaler)
    print("Predicted next close price:", predicted_next_close)