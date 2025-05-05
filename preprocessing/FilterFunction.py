import numpy as np
from enum import Enum
from scipy.signal import lfilter
# from pathlib import Path


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
    # Numerators and denominators for a band-pass Butterworth filter
    # 4th order band-pass Butterworth filter fs = 5000 Hz fc1 = 20 Hz fc2 = 600 Hz designed in Matlab
    b1 = np.array([0.0042, 0., -0.0168, 0., 0.0252, 0., -0.0168, 0., 0.0042])
    a1 = np.array([1.0000, -6.3816, 17.8992, -28.8773, 29.3561, -19.2729, 7.9812, -1.9056, 0.2009])

    # calculate the mean
    mean = float(np.mean(data_array))

    # mean subtraction
    data_array_centered = data_array - mean

    # Apply the filter to the windowed data
    # filtfilt means 8th order filter but zero phase distortion
    if filter == 1:
        # Apply the filter to the windowed data
        filtered_data_array = lfilter(b1, a1, data_array_centered)
        # absolute value
        filtered_data_array_abs = np.abs(filtered_data_array)

        # normalize the data
        data_array_filtered_output = (filtered_data_array_abs - np.min(filtered_data_array_abs)) / (
                    np.max(filtered_data_array_abs) - np.min(filtered_data_array_abs))

        # return filtered and normalized data
        return data_array_filtered_output
    else:
        # absolute value
        data_array_centered_abs = np.abs(data_array_centered)

        # normalize the data
        data_array_output = (data_array_centered_abs - np.min(data_array_centered_abs)) / (
                    np.max(data_array_centered_abs) - np.min(data_array_centered_abs))

        # return normalized data
        return data_array_output


# Call the filter_function with the data_array
# filtered_data = filter_function(window, 1)
