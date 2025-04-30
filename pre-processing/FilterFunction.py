import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import freqz
# from pathlib import Path


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


    #calculate the mean
    mean = np.mean(data_array)

    #mean subtraction
    data_array = data_array - mean

    #absolute value
    data_array = np.abs(data_array)

    #calculate the std and mean
    std = np.std(data_array)
    mean = np.mean(data_array)

    #normalize the data
    data_array_normalized = (data_array - mean) / std

    # Apply the filter to the windowed data
    # filtfilt means 16th order filter but zero phase distortion
    if filter == 1:
        # Apply the filter to the windowed data
        filtered_data_array = sp.signal.filtfilt(b1, a1, data_array_normalized)

        return filtered_data_array
    else:
        # return normalized data
        return data_array_normalized


# Call the filter_function with the data_array
# filtered_data = filter_function(window, 1)
