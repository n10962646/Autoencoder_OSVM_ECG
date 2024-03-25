import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

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

# Dummy values for uncertainties (replace with actual values if available)
global_uncertainty = np.random.rand(len(global_signal))
local_uncertainty = np.random.rand(len(local_signal))

# Dummy values for restored signals (replace with actual restored signals)
restored_global_signal = np.random.rand(len(global_signal))
restored_local_signal = np.random.rand(len(local_signal))
restored_trend_signal = np.random.rand(len(global_signal))

# Define the loss function
def loss_function(true_signal, restored_signal, uncertainty):
    loss = np.mean(np.square(true_signal - restored_signal) / uncertainty + np.log(uncertainty))
    return loss

# Calculate losses
alpha = 1.0  # Trade-off parameter for local restoration loss
beta = 1.0   # Trade-off parameter for trend restoration loss

global_loss = loss_function(global_signal, restored_global_signal, global_uncertainty)
local_loss = loss_function(local_signal, restored_local_signal, local_uncertainty)
trend_loss = loss_function(global_signal, restored_trend_signal, np.ones_like(global_signal))  # Assuming no uncertainty for trend

# Create meshgrid for plotting
alpha_range = np.linspace(0, 2, 10)
beta_range = np.linspace(0, 2, 10)
alpha_mesh, beta_mesh = np.meshgrid(alpha_range, beta_range)

# Calculate total loss for different combinations of alpha and beta
total_loss_mesh = global_loss + alpha_mesh * local_loss + beta_mesh * trend_loss

# Plotting 3D surface
fig = plt.figure(figsize=(15, 9))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(alpha_mesh, beta_mesh, total_loss_mesh, cmap='viridis', edgecolor='none')
ax.set_title('Total Loss with Varying Alpha and Beta')
ax.set_xlabel('Alpha')
ax.set_ylabel('Beta')
ax.set_zlabel('Total Loss')

fig.colorbar(surf, shrink=0.5, aspect=5)

# Check if the 'image' directory exists, if not create it
if not os.path.exists('image'):
    os.makedirs('image')

# Save the plot as an image in the 'image' directory
plt.savefig('image/loss_func.png')

plt.show()

