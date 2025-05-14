import numpy as np
from enum import Enum
from scipy.signal import sosfiltfilt

# from pathlib import Path
from preprocessing.Moving_average_filter import calculate_moving_average


class filter(Enum):
    """
    Enum class for filter options.
    """

    NO_FILTER = 0
    FILTER = 1


class filter_type(Enum):
    """
    Enum class for filter types.
    """

    Range20TO125 = "cheby1_20to125"
    Range125TO250 = "cheby1_125to250"
    Range20TO250 = "cheby1_20to250"
    Range20TO500 = "butter_20to500"


class normalization_type(Enum):
    """
    Enum class for normalization types.
    """

    min_max = "min-max"
    z_score = "z-score"


def filter_function(
    data_array,
    filter=0,
    filter_type=None,
    normalization_type=None,
    use_moving_average=0,
):
    """
    This function applies mean subtraction, absolute value, normalization
    and a low-pass Butterworth filter to the input data array.
    Parameters: data_array (numpy.ndarray): The input data array to be processed.
                filter (int): If 1, apply the filter; if 0, do not apply the filter.
                filter_type (str): The type of filter to use ('cheby1_20to125', 'cheby1_125to250', 'cheby1_20to250', 'butter_20to500', ).
                normalization_type (str): The type of normalization to use ('min-max', 'z-score').
    """
    # cheby1 8th order pass ripple 0.1dB 20-250Hz
    cheby1sos20to250 = np.array(
        [
            [
                7.74510118e-09,
                1.54902024e-08,
                7.74510118e-09,
                1.00000000e00,
                -1.83072505e00,
                8.74096486e-01,
            ],
            [
                1.00000000e00,
                2.00000000e00,
                1.00000000e00,
                1.00000000e00,
                -1.99680513e00,
                9.97401158e-01,
            ],
            [
                1.00000000e00,
                2.00000000e00,
                1.00000000e00,
                1.00000000e00,
                -1.98989275e00,
                9.90660250e-01,
            ],
            [
                1.00000000e00,
                2.00000000e00,
                1.00000000e00,
                1.00000000e00,
                -1.97582182e00,
                9.77159581e-01,
            ],
            [
                1.00000000e00,
                -2.00000000e00,
                1.00000000e00,
                1.00000000e00,
                -1.86501440e00,
                9.66745068e-01,
            ],
            [
                1.00000000e00,
                -2.00000000e00,
                1.00000000e00,
                1.00000000e00,
                -1.93806784e00,
                9.41707601e-01,
            ],
            [
                1.00000000e00,
                -2.00000000e00,
                1.00000000e00,
                1.00000000e00,
                -1.83248144e00,
                9.09366516e-01,
            ],
            [
                1.00000000e00,
                -2.00000000e00,
                1.00000000e00,
                1.00000000e00,
                -1.86478941e00,
                8.80606308e-01,
            ],
        ]
    )
    # cheby1 8th order pass ripple 0.1dB 20-125Hz
    cheby1sos20to125 = np.array(
        [
            [
                1.64903657e-11,
                3.29807315e-11,
                1.64903657e-11,
                1.00000000e00,
                -1.93361479e00,
                9.46290611e-01,
            ],
            [
                1.00000000e00,
                2.00000000e00,
                1.00000000e00,
                1.00000000e00,
                -1.94330916e00,
                9.49367982e-01,
            ],
            [
                1.00000000e00,
                2.00000000e00,
                1.00000000e00,
                1.00000000e00,
                -1.94019212e00,
                9.60501057e-01,
            ],
            [
                1.00000000e00,
                2.00000000e00,
                1.00000000e00,
                1.00000000e00,
                -1.96502534e00,
                9.67496863e-01,
            ],
            [
                1.00000000e00,
                -2.00000000e00,
                1.00000000e00,
                1.00000000e00,
                -1.98219393e00,
                9.83381300e-01,
            ],
            [
                1.00000000e00,
                -2.00000000e00,
                1.00000000e00,
                1.00000000e00,
                -1.95987802e00,
                9.85511447e-01,
            ],
            [
                1.00000000e00,
                -2.00000000e00,
                1.00000000e00,
                1.00000000e00,
                -1.99158800e00,
                9.92336515e-01,
            ],
            [
                1.00000000e00,
                -2.00000000e00,
                1.00000000e00,
                1.00000000e00,
                -1.99716523e00,
                9.97766712e-01,
            ],
        ]
    )
    # cheby1 8th order pass ripple 0.1dB 125-250Hz
    cheby1sos125to250 = np.array(
        [
            [
                6.52154309e-11,
                1.30430862e-10,
                6.52154309e-11,
                1.00000000e00,
                -1.89173522e00,
                9.47185948e-01,
            ],
            [
                1.00000000e00,
                2.00000000e00,
                1.00000000e00,
                1.00000000e00,
                -1.96937367e00,
                9.93390749e-01,
            ],
            [
                1.00000000e00,
                2.00000000e00,
                1.00000000e00,
                1.00000000e00,
                -1.88738953e00,
                9.86710779e-01,
            ],
            [
                1.00000000e00,
                2.00000000e00,
                1.00000000e00,
                1.00000000e00,
                -1.95348905e00,
                9.79972356e-01,
            ],
            [
                1.00000000e00,
                -2.00000000e00,
                1.00000000e00,
                1.00000000e00,
                -1.93424875e00,
                9.66317181e-01,
            ],
            [
                1.00000000e00,
                -2.00000000e00,
                1.00000000e00,
                1.00000000e00,
                -1.87521450e00,
                9.63811101e-01,
            ],
            [
                1.00000000e00,
                -2.00000000e00,
                1.00000000e00,
                1.00000000e00,
                -1.91250084e00,
                9.54094194e-01,
            ],
            [
                1.00000000e00,
                -2.00000000e00,
                1.00000000e00,
                1.00000000e00,
                -1.87755312e00,
                9.49888670e-01,
            ],
        ]
    )
    # butterworth 8th order pass ripple 0.1dB 20-500Hz
    buttersos20to500 = np.array(
        [
            [0.00419699, 0.00839399, 0.00419699, 1.0, -1.10421484, 0.32796109],
            [1.0, 2.0, 1.0, 1.0, -1.34463672, 0.65505184],
            [1.0, -2.0, 1.0, 1.0, -1.95141778, 0.95212526],
            [1.0, -2.0, 1.0, 1.0, -1.98137739, 0.98201597],
        ]
    )

    if filter_type == "cheby1_20to125":
        sos = cheby1sos20to125
    elif filter_type == "cheby1_20to250":
        sos = cheby1sos20to250
    elif filter_type == "butter_20to500":
        sos = buttersos20to500
    elif filter_type == "cheby1_125to250":
        sos = cheby1sos125to250
    else:
        raise ValueError(
            "Invalid filter_type. Choose from 'cheby1_20to125', 'cheby1_125to250', 'cheby1_20to250' ,'butter_20to500'."
        )

    # calculate the mean
    mean = float(np.mean(data_array))

    # mean subtraction
    data_array_centered = data_array - mean

    # Apply the filter to the windowed data

    if filter == 1:
        # Apply the filter to the windowed data
        filtered_data_array = sosfiltfilt(sos, data_array_centered)

        filtered_data_array_abs = np.abs(filtered_data_array)

        if use_moving_average == 1:
            filtered_data_array_abs_mv = calculate_moving_average(
                filtered_data_array_abs
            )
        else:
            filtered_data_array_abs_mv = filtered_data_array_abs
        if normalization_type == "z-score":
            # z-score normalization
            data_array_filtered_output = (
                filtered_data_array_abs_mv - np.mean(filtered_data_array_abs_mv)
            ) / np.std(filtered_data_array_abs_mv)
        else:
            # min-max normalization
            data_array_filtered_output = (
                filtered_data_array_abs_mv - np.min(filtered_data_array_abs_mv)
            ) / (
                np.max(filtered_data_array_abs_mv) - np.min(filtered_data_array_abs_mv)
            )

        return data_array_filtered_output
    else:
        # absolute value
        data_array_centered_abs = np.abs(data_array_centered)

        if use_moving_average == 1:
            print("Using moving average")
            data_array_centered_abs_mv = calculate_moving_average(
                data_array_centered_abs
            )
        else:
            print("Not using moving average")
            data_array_centered_abs_mv = data_array_centered_abs

        if normalization_type == "z-score":
            # z-score normalization
            data_array_output = (
                data_array_centered_abs_mv - np.mean(data_array_centered_abs_mv)
            ) / np.std(data_array_centered_abs_mv)
        elif normalization_type == "min-max":
            # min-max normalization
            data_array_output = (
                data_array_centered_abs_mv - np.min(data_array_centered_abs_mv)
            ) / (
                np.max(data_array_centered_abs_mv) - np.min(data_array_centered_abs_mv)
            )
        else:
            raise ValueError(
                "Invalid normalization_type. Choose from 'z-score' or 'min-max'."
            )
        # return normalized data
        return data_array_output


# Call the filter_function with the data_array
# filtered_data = filter_function(window, filter=filter.NO_FILTER, filter_type=filter_type.R20TO125, normalization_type=normalization_type.min_max)
