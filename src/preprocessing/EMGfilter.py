import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from enum import Enum
from scipy.signal import freqz
from scipy.signal import filtfilt
from scipy.signal import butter
from scipy.signal import lfilter


# C:\Users\old22001\Downloads\testP.csv
#C:\Users\old22001\Downloads\WyoFlex_Dataset\WyoFlex_Dataset\VOLTAGE DATA\P1C1S1M1F1O1
# Load the CSV file without headers
data = pd.read_csv(r'C:\Users\old22001\Downloads\testP.csv', header=4)
# data = pd.read_csv(r'C:\Users\old22001\Downloads\WyoFlex_Dataset\WyoFlex_Dataset\VOLTAGE DATA\P1C1S1M1F1O1.csv', header=none)

# # Convert the data to a NumPy array
# data_array = data.to_numpy()
data_array = data["C2 (V)"].to_numpy()
data_array_time = data["Time (s)"].to_numpy()
#
# # Convert the data to a 1D NumPy array
data_array = data_array.flatten()
#
#plotly plot
# fig = go.Figure([go.Scatter(x=data_array_time, y=data_array)])
# fig.show()

# plt.figure()
# plt.plot(data_array, label='Data')
# plt.ylabel('Amplitude')
# plt.xlabel('Sample Number')
# plt.title('Windowed Data')
# plt.legend()
# plt.show()


class filter (Enum):
    """
    Enum class for filter options.
    """
    NO_FILTER = 0
    FILTER = 1


def filter_function(data_array, filter):
    """
    This function applies mean subtraction, absolute value, normalization
    and a low-pass Butterworth filter to the input data array.
    Parameters: data_array (numpy.ndarray): The input data array to be processed.
                filter (int): If 1, apply the filter; if 0, do not apply the filter.
    """
    #0.0042	0	-0.0168	0	0.0252	0	-0.0168	0	0.0042
    # Numerators and denominators for a low-pass Butterworth filter b = numerator a = denominator
    #4th order band-pass Butterworth filter fs = 5000 Hz fc1 = 20 Hz fc2 = 600 Hz
    b1 = np.array([0.0042,	0.,	-0.0168,	0.,	0.0252,	0.,	-0.0168,	0.,	0.0042])
    a1 = np.array([1.0000,	-6.3816,	17.8992,	-28.8773,	29.3561,	-19.2729,	7.9812,	-1.9056,	0.2009])

    # Testing butterworth filter
   # b, a = butter(4, [20, 500], 'bandpass', analog=False, output='ba', fs=5000)
    #print(b)
    #print(a)
    print(b1)
    print(a1)

    # calculate the mean
    mean = float(np.mean(data_array))

    # mean subtraction
    data_array_centered = data_array - mean

    # plot after mean subtraction
    plt.figure()
    plt.plot(data_array_centered, label='Windowed Data (200 Samples)')
    plt.ylabel('Amplitude')
    plt.xlabel('Sample Number')
    plt.title('Windowed Data (Mean Subtracted)')
    plt.legend()
    plt.show()

    # Apply the filter to the windowed data
    # filtfilt means 8th order filter but zero phase distortion
    if filter == 1:
        # Apply the filter to the windowed data
        filtered_data_array = lfilter(b1, a1, data_array_centered)

        # plot after filtering
        plt.figure()
        plt.plot(data_array_centered, label='Windowed Data (filtered)')
        plt.ylabel('Amplitude')
        plt.xlabel('Sample Number')
        plt.title('Windowed Data (filtered)')
        plt.legend()
        plt.show()

        # absolute value
        filtered_data_array_abs = np.abs(filtered_data_array)

        # calculate the std and mean from filtered data
        #mean = float(np.mean(filtered_data_array_abs))
        #std = np.std(filtered_data_array_abs)
        # normalize the data
        data_array_filtered_output = (filtered_data_array_abs - np.min(filtered_data_array_abs)) / (
                np.max(filtered_data_array_abs) - np.min(filtered_data_array_abs))

        #return filtered and normalized data
        return data_array_filtered_output
    else:
        # absolute value
        data_array_centered_abs = np.abs(data_array_centered)

        #plot after absolute value
        plt.figure()
        plt.plot(data_array_centered_abs, label='Windowed Data (absolute value)')
        plt.ylabel('Amplitude')
        plt.xlabel('Sample Number')
        plt.title('Windowed Data (absolute value)')
        plt.legend()
        plt.show()

        # calculate the std and mean
        # mean_abs = float(np.mean(data_array_centered_abs))
        # std_abs = np.std(data_array_centered_abs)
        # normalize the data
        data_array_output = (data_array_centered_abs - np.min(data_array_centered_abs)) / (np.max(data_array_centered_abs) - np.min(data_array_centered_abs))


        # return normalized data
        return data_array_output


# # Create a window from the first 200 samples
window = data_array[:100]  # Adjust the indices as needed

# # Call the filter_function with the data_array
filtered_data = filter_function(window, 1)
#
# # Optionally, plot the filtered data
plt.figure()
plt.plot(filtered_data, label='Filtered Data')
plt.ylabel('Amplitude')
plt.xlabel('Sample Number')
plt.title('Filtered Data')
plt.legend()
plt.show()

