import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_data(data, seq_len=60):

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(data)

    X = []
    y = []

    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i])
        y.append(scaled_data[i])

    return np.array(X), np.array(y), scaler