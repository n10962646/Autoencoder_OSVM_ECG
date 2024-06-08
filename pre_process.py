import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data():
    x_train = np.load('x_train.npy')
    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')
    y_train = np.load('y_train.npy')
    
    y_test = np.where(y_test == 'normal', 0, 1)
    y_train = np.where(y_train == 'normal', 0, 1)
    
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train.reshape(-1, 1)).reshape(x_train.shape)
    x_test = scaler.transform(x_test.reshape(-1, 1)).reshape(x_test.shape)

    return x_train, x_test, y_train, y_test

def split_into_windows(signal, window_size, step_size):
    windows = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        window = signal[start:start + window_size]
        windows.append(window)
    return np.array(windows)

def preprocess_data(x_data, window_size=100, step_size=50):
    global_ecg = x_data
    
    local_ecg = np.array([split_into_windows(signal, window_size, step_size) for signal in x_data])

    return global_ecg, local_ecg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ECG data.")
    parser.add_argument("--window_size", type=int, default=100, help="Size of the window for splitting signals.")
    parser.add_argument("--step_size", type=int, default=50, help="Step size for splitting signals.")

    args = parser.parse_args()
