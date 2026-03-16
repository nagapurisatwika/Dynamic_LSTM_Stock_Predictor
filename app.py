import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

from data_processing import prepare_data
from model import build_model
from prediction import predict_future

st.title("Dynamic Stock Price Prediction using LSTM")

# USER INPUT (NO DEFAULT VALUES)
ticker = st.text_input("Enter Stock Symbol")

start = st.date_input("Select Start Date")

end = st.date_input("Select End Date")

epochs = st.number_input("Enter Training Epochs", min_value=1)

days = st.number_input("Enter Future Prediction Days", min_value=1)

if st.button("Run Prediction"):

    # CHECK INPUTS
    if ticker == "":
        st.error("Please enter a stock symbol")
        st.stop()

    if start >= end:
        st.error("Start date must be earlier than end date")
        st.stop()

    # DOWNLOAD DATA
    data = yf.download(ticker, start=start, end=end)

    if data.empty:
        st.error("No data found for this stock")
        st.stop()

    st.subheader("Downloaded Stock Data")
    st.write(data.tail())

    prices = data[['Close']].values

    if len(prices) < 70:
        st.error("Dataset too small. Choose earlier start date")
        st.stop()

    # PREPROCESS DATA
    X, y, scaler = prepare_data(prices)

    if len(X) == 0:
        st.error("Sequence creation failed")
        st.stop()

    # TRAIN TEST SPLIT
    split = int(len(X) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    # BUILD MODEL
    model = build_model((X.shape[1], 1))

    st.write("Training Model...")

    model.fit(
        X_train,
        y_train,
        epochs=int(epochs),
        batch_size=32,
        verbose=0
    )

    # TEST PREDICTIONS
    test_predictions = model.predict(X_test)

    test_predictions = scaler.inverse_transform(test_predictions)
    y_test_actual = scaler.inverse_transform(y_test)

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))

    st.subheader("Model Accuracy")
    st.write("RMSE:", rmse)

    # FUTURE PREDICTION
    last_seq = X[-1]

    predictions = predict_future(
        model,
        last_seq,
        scaler,
        int(days)
    )

    st.subheader("Future Predictions")
    st.write(predictions)

    # HISTORY GRAPH
    st.subheader("Stock Price History")

    fig = plt.figure()

    plt.plot(data['Close'])
    plt.title("Historical Stock Prices")

    st.pyplot(fig)

    # ACTUAL VS PREDICTED GRAPH
    st.subheader("Actual vs Predicted Prices")

    fig2 = plt.figure()

    plt.plot(y_test_actual, label="Actual Price")
    plt.plot(test_predictions, label="Predicted Price")

    plt.legend()

    st.pyplot(fig2)