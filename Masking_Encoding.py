import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os


# Load data from MATLAB .mat file
mat_data = loadmat('data/training/A00001.mat')
data_array = mat_data['val'][0]  # Extract the NumPy array

# Split the data into global ECG signal (x_g) and local heartbeat signal (x_l)
x_g = data_array[:9000]  # Assuming global ECG signal has 5000 elements
x_l = data_array[1000:]  # Assuming local heartbeat signal has 4000 elements

# Generate random masks
D = len(x_g)
d = len(x_l)
M_g = np.random.choice([0, 1], size=D)  # Global mask
M_l = np.random.choice([0, 1], size=d)  # Local mask

# Apply masks to signals
x_masked_g = x_g * M_g
x_masked_l = x_l * M_l

# Plot original and masked signals
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(x_g, label='Global ECG Signal ($x_g$)')
plt.title('Global ECG Signal')
plt.legend(loc='lower right')

plt.subplot(4, 1, 2)
plt.plot(x_masked_g, label='Masked Global ECG Signal ($x_{masked\_global}$)', linestyle='--')
plt.title('Masked Global ECG Signal')
plt.legend(loc='lower right')

plt.subplot(4, 1, 3)
plt.plot(x_l, label='Local Heartbeat Signal ($x_l$)')
plt.title('Local Heartbeat Signal')
plt.legend(loc='lower right')

plt.subplot(4, 1, 4)
plt.plot(x_masked_l, label='Masked Local Heartbeat Signal ($x_{masked\_local}$)', linestyle='--')
plt.title('Masked Local Heartbeat Signal')
plt.legend(loc='lower right')

plt.tight_layout()

# Check if the 'image' directory exists, if not create it
if not os.path.exists('image'):
    os.makedirs('image')

# Save the plot as an image in the 'image' directory
plt.savefig('image/mask_encod.png')

# Show the plot
plt.show()


