import numpy as np
import matplotlib.pyplot as plt
import os

# Function to generate smooth trend from the input signal
def generate_smooth_trend(signal):
    smooth_trend = np.diff(signal)  # Compute the difference between adjacent time-series signal points
    return smooth_trend

# Function to define the autoencoder architecture
class Autoencoder:
    def __init__(self, input_dim, latent_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        # Define encoder layers (for simplicity, a linear encoder is used)
        encoder = lambda x: x
        return encoder

    def build_decoder(self):
        # Define decoder layers (for simplicity, a linear decoder is used)
        decoder = lambda x: x
        return decoder

# Concatenate encoded trend information with global features and reconstruct global ECG signal
def restore_global_signal(encoded_trend, global_features, autoencoder):
    # Reshape encoded trend to match the dimensions for concatenation
    encoded_trend_reshaped = encoded_trend.reshape(-1, 1)
    # Concatenate encoded trend with global features
    concatenated_input = np.concatenate((encoded_trend_reshaped, global_features), axis=1)
    # Restore the global signal using the autoencoder decoder
    restored_signal = autoencoder.decoder(concatenated_input)
    return restored_signal

# Load the .mat file (make sure to have scipy installed)
from scipy.io import loadmat

# Load the data
mat_data = loadmat('data/training/A00001.mat')
data_array = mat_data['val'][0]

# Determine the length of the signal
signal_length = len(data_array)

# Calculate the indices for splitting
split_length = int(signal_length * 0.1)

# Split the data array into local and global signals
local_signal = data_array[:split_length]
global_signal = data_array[split_length:]

# Generate smooth trend from the global signal
smooth_trend = generate_smooth_trend(global_signal)

# Define autoencoder parameters
input_dim = len(smooth_trend)
latent_dim = 10  # Adjust as needed

# Create and initialize the autoencoder
autoencoder = Autoencoder(input_dim, latent_dim)

# Dummy global features (replace with actual features if available)
global_features = np.random.randn(len(smooth_trend), latent_dim)

# Restore the global ECG signal using TGM
restored_signal = restore_global_signal(smooth_trend, global_features, autoencoder)

# Plotting
plt.figure(figsize=(15, 6))
plt.plot(global_signal, label='True Global ECG Signal', color='blue')
plt.plot(restored_signal, label='Restored Global ECG Signal', linestyle='--', color='red')
plt.title('Global ECG Signal Restoration using Trend Generation Module')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Check if the 'image' directory exists, if not create it
if not os.path.exists('image'):
    os.makedirs('image')

# Save the plot as an image in the 'image' directory
plt.savefig('image/treng_mod.png')

# Show the plot
plt.show()
