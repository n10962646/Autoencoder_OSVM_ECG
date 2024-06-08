import os
import numpy as np
import scipy.io
from tqdm import tqdm
import argparse

def mat_to_npy(directory, output_directory, signal_length=9000):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    total_files = sum(1 for filename in os.listdir(directory) if filename.endswith(".mat"))
    
    pbar = tqdm(total=total_files, desc="Converting .mat files")
    
    file_count = 0

    for filename in os.listdir(directory):
        if filename.endswith(".mat"):
            filepath = os.path.join(directory, filename)
            try:
                mat_data = scipy.io.loadmat(filepath)
                ecg_signal = mat_data['val'][0]
                
                if len(ecg_signal) >= signal_length:
                    ecg_signal = ecg_signal[:signal_length]
                elif len(ecg_signal) == signal_length:
                    pass  
                else:
                    continue  
                    
                output_filepath = os.path.join(output_directory, filename.replace('.mat', '.npy'))
                np.save(output_filepath, ecg_signal)

                if file_count < 5:
                    print(f"File: {filename}")

                file_count += 1
                
                pbar.update(1) 

            except Exception as e:
                print(f"Error converting file {filepath}: {e}")
    
    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .mat files to .npy files.")
    parser.add_argument("input_directory", type=str, help="Directory containing .mat files.")
    parser.add_argument("output_directory", type=str, help="Directory to save .npy files.")
    parser.add_argument("--signal_length", type=int, default=9000, help="Length of the ECG signal (default: 9000).")

    args = parser.parse_args()

    mat_to_npy(args.input_directory, args.output_directory, args.signal_length)
