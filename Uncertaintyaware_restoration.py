import numpy as np
import matplotlib.pyplot as plt
import os

# Function to calculate the uncertainty-aware restoration loss
def uncertainty_aware_restoration_loss(true_signal, restored_signal, uncertainty):
    loss = np.mean((np.square(true_signal - restored_signal) / uncertainty) + np.log(uncertainty))
    return loss

# Define the restoration function
def restore_signal(signal, uncertainty):
    # For illustration purposes, simply copying the signal as restored signal
    restored_signal = np.copy(signal)
    return restored_signal

# Load the .mat file (make sure to have scipy installed)
from scipy.io import loadmat

# Load the data
mat_data = loadmat('data/training/A00001.mat')
data_array = mat_data['val'][0]

# Split the data array into global and local signals
signal_length = len(data_array)
split_length = int(signal_length * 0.1)
local_signal = data_array[:split_length]
global_signal = data_array[split_length:]

# Generate some random uncertainties for demonstration
local_uncertainty = np.random.rand(split_length)
global_uncertainty = np.random.rand(signal_length - split_length)

# Restore the global and local signals
restored_global_signal = restore_signal(global_signal, global_uncertainty)
restored_local_signal = restore_signal(local_signal, local_uncertainty)

# Calculate the differences between true and restored signals
global_difference = global_signal - restored_global_signal
local_difference = local_signal - restored_local_signal

# Plotting
plt.figure(figsize=(17, 8))

# Global Signal Restoration Plot
plt.subplot(2, 2, 1)
plt.plot(global_signal, label='True Global Signal', color='blue')
plt.plot(restored_global_signal, label='Restored Global Signal', linestyle='--', color='red')
plt.title('Global Signal Restoration')
plt.legend()

# Local Signal Restoration Plot
plt.subplot(2, 2, 2)
plt.plot(local_signal, label='True Local Signal', color='green')
plt.plot(restored_local_signal, label='Restored Local Signal', linestyle='--', color='orange')
plt.title('Local Signal Restoration')
plt.legend()

# Difference Plot for Global Signal
plt.subplot(2, 2, 3)
plt.plot(global_difference, label='Difference', color='blue')
plt.title('Difference between True and Restored Global Signals')
plt.legend()

# Difference Plot for Local Signal
plt.subplot(2, 2, 4)
plt.plot(local_difference, label='Difference', color='green')
plt.title('Difference between True and Restored Local Signals')
plt.legend()

plt.tight_layout()

# Check if the 'image' directory exists, if not create it
if not os.path.exists('image'):
    os.makedirs('image')

# Save the plot as an image in the 'image' directory
plt.savefig('image/uaware_rest.png')

# Show the plot
plt.show()

