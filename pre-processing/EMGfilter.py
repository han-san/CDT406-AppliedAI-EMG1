from cProfile import label

import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import freqz

# Load the CSV file without headers
data = pd.read_csv(r'C:\Users\old22001\Downloads\WyoFlex_Dataset\WyoFlex_Dataset\VOLTAGE DATA\P1C1S1M1F1O1', header=None)
# Convert the data to a NumPy array
data_array = data.to_numpy()

# Convert the data to a 1D NumPy array
data_array = data_array.flatten()


# Define filter parameters
order = 4  # Filter order
cutoff = 100  # Cutoff frequency in Hz
fs = 1000  # Sampling frequency in Hz

# Design a low-pass Butterworth filter
b, a = sp.signal.butter(order, cutoff / (fs / 2), btype='low', analog=False)

print(b)
print(a)

# Create a window from the first 200 samples
window = data_array[8000:11000]  # Adjust the indices as needed

#calculate the mean and standard deviation
mean = np.mean(window)
std = np.std(window)

#mean subtraction
window = window - mean

#plot after mean subtraction
plt.figure()
plt.plot(window, label='Windowed Data (First 200 Samples)')
plt.ylabel('Amplitude')
plt.xlabel('Sample Number')
plt.title('Windowed Data (Mean Subtracted)')
plt.legend()
plt.show()



#absolute value
window = np.abs(window)

#plot after absolute value
plt.figure()
plt.plot(window, label='Windowed Data (First 200 Samples)')
plt.ylabel('Amplitude')
plt.xlabel('Sample Number')
plt.title('Windowed Data (Absolute Value)')
plt.legend()
plt.show()

#normalize the data
window_normalized = (window - mean) / std

# Plot the normalized windowed data
plt.figure()
plt.plot(window_normalized, label='Windowed Data (First 200 Samples)')
plt.ylabel('Amplitude')
plt.xlabel('Sample Number')
plt.title('Windowed Data (Normalized)')
plt.legend()
plt.show()

# Apply the filter to the windowed data
filtered_window = sp.signal.filtfilt(b, a, window)

# plot the filtered windowed data
plt.figure()
plt.plot(filtered_window, label='Windowed Data (First 200 Samples)')
plt.ylabel('Amplitude')
plt.xlabel('Sample Number')
plt.title('Windowed Data (Filtered)')
plt.legend()
plt.show()


