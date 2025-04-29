import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import freqz

# Load the CSV file without headers
# data = pd.read_csv(r'C:\Users\old22001\Downloads\WyoFlex_Dataset\WyoFlex_Dataset\VOLTAGE DATA\P1C1S1M1F1O1', header=None)
# # Convert the data to a NumPy array
# data_array = data.to_numpy()
#
# # Convert the data to a 1D NumPy array
# data_array = data_array.flatten()
#
# # Create a window from the first 200 samples
# window = data_array[:200]  # Adjust the indices as needed

def filter_function(data_array, filter):
    """
    This function applies mean subtraction, absolute value, normalization
    and a low-pass Butterworth filter to the input data array.
    Parameters: data_array (numpy.ndarray): The input data array to be processed.
                filter (int): If 1, apply the filter; if 0, do not apply the filter.
    """
    # Numerators and denominators for a low-pass Butterworth filter
    #8th order low-pass Butterworth filter fs = 1000 Hz fc = 200 Hz
    a1 = [1, -1.5906, 2.0838, -1.5326, 0.8694, -0.3192, 0.0821, -0.0122, 0.0009]
    b1 = [0.0023, 0.0182, 0.0636, 0.1272, 0.1590, 0.1272, 0.0636, 0.0182, 0.0023]


    #calculate the mean and standard deviation
    mean = np.mean(data_array)
    std = np.std(data_array)

    #mean subtraction
    data_array = data_array - mean

    #plot after mean subtraction
    # plt.figure()
    # plt.plot(window, label='Windowed Data (200 Samples)')
    # plt.ylabel('Amplitude')
    # plt.xlabel('Sample Number')
    # plt.title('Windowed Data (Mean Subtracted)')
    # plt.legend()
    # plt.show()

    #absolute value
    data_array = np.abs(data_array)

    #plot after absolute value
    # plt.figure()
    # plt.plot(window, label='Windowed Data (200 Samples)')
    # plt.ylabel('Amplitude')
    # plt.xlabel('Sample Number')
    # plt.title('Windowed Data (Absolute Value)')
    # plt.legend()
    # plt.show()

    #normalize the data
    data_array_normalized = (data_array - mean) / std

    # Plot the normalized windowed data
    # plt.figure()
    # plt.plot(window_normalized, label='Windowed Data (200 Samples)')
    # plt.ylabel('Amplitude')
    # plt.xlabel('Sample Number')
    # plt.title('Windowed Data (Normalized)')
    # plt.legend()
    # plt.show()

    # Apply the filter to the windowed data
    # filtfilt means 16th order filter but zero phase distortion
    if filter == 1:
        # Apply the filter to the windowed data
        filtered_data_array = sp.signal.filtfilt(b1, a1, data_array_normalized)

        # plot the filtered windowed data
        # plt.figure()
        # plt.plot(filtered_window, label='Windowed Data (200 Samples)')
        # plt.ylabel('Amplitude')
        # plt.xlabel('Sample Number')
        # plt.title('Windowed Data (Filtered)')
        # plt.legend()
        # plt.show()

        return filtered_data_array
    else:
        # return normalized data
        return data_array_normalized


# # Call the filter_function with the data_array
# filtered_data = filter_function(window)
#
# # Optionally, plot the filtered data
# plt.figure()
# plt.plot(filtered_data, label='Filtered Data')
# plt.ylabel('Amplitude')
# plt.xlabel('Sample Number')
# plt.title('Filtered Data')
# plt.legend()
# plt.show()