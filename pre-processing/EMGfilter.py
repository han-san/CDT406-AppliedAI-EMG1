from cProfile import label

import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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



filtered_signal = sp.signal.lfilter(b, a, data_array)

plt.plot(data_array, label='Original Signal')
plt.plot(filtered_signal, label='Filtered Signal')
plt.ylabel('Amplitude')
plt.xlabel('Sample Number')
plt.show()

# Print the loaded data
# print(data_array)