import os
import numpy as np
from tqdm import tqdm
import argparse

def load_npz(directory, signal_length):
    X = []
    y = []
    
    file_count = 0
    
    pbar = tqdm(total=len(os.listdir(directory)), desc="Loading data")
    
    for filename in os.listdir(directory):
        if filename.endswith(".npz"):
            filepath = os.path.join(directory, filename)
            try:
                data = np.load(filepath)
                ecg_signal = data['signal']
                label = data['label']
                
                if len(ecg_signal) < signal_length:
                    ecg_signal = np.pad(ecg_signal, (0, signal_length - len(ecg_signal)), 'constant')
                elif len(ecg_signal) > signal_length:
                    ecg_signal = ecg_signal[:signal_length]
                
                X.append(ecg_signal)
                y.append(label)
                
                if file_count < 5:
                    print(f"File: {filename}, Label: {label}")
                
                file_count += 1
                
                pbar.update(1)
            except Exception as e:
                print(f"Error loading file {filepath}: {e}")
    
    pbar.close()
    
    return np.array(X), np.array(y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load individual .npz files and process ECG signals.")
    parser.add_argument("directory", type=str, help="Directory containing .npz files.")
    parser.add_argument("signal_length", type=int, help="Desired length of the ECG signals.")

    args = parser.parse_args()

    X, y = load_npz(args.directory, args.signal_length)

