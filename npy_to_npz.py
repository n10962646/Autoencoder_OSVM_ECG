import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

def npy_to_npz(input_directory, output_directory, label_df_path):

    label_df = pd.read_csv(label_df_path)

    os.makedirs(output_directory, exist_ok=True)
    
    pbar = tqdm(total=len(os.listdir(input_directory)), desc="Saving signals")
    
    file_count = 0
    
    for filename in os.listdir(input_directory):
        if filename.endswith(".npy"):
            filepath = os.path.join(input_directory, filename)
            try:
                ecg_signal = np.load(filepath)
                label = label_df.loc[label_df['filename'] == filename[:-4], 'label'].values[0]
                
                output_filepath = os.path.join(output_directory, filename.replace('.npy', '.npz'))
                np.savez(output_filepath, signal=ecg_signal, label=label)
                
                if file_count < 5:
                    print(f"File: {filename}, Label: {label}")
                
                file_count += 1
                
                pbar.update(1)
            except Exception as e:
                print(f"Error loading file {filepath}: {e}")
    
    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save ECG signals and labels into individual .npz files.")
    parser.add_argument("input_directory", type=str, help="Directory containing .npy files.")
    parser.add_argument("output_directory", type=str, help="Directory to save .npz files.")
    parser.add_argument("label_df_path", type=str, help="Path to the CSV file containing labels.")

    args = parser.parse_args()

    npy_to_npz(args.input_directory, args.output_directory, args.label_df_path)
