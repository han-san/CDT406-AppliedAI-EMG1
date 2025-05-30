from __future__ import annotations

from enum import Enum

import numpy as np
import numpy.typing as npt
from scipy.signal import sosfiltfilt  # type: ignore[import-untyped]

from preprocessing.Moving_average_filter import (
    MovingAverageType,
    exponential_moving_average,
    simple_moving_average,
)


class FilterType(Enum):
    """Enum class for filter types."""

    RANGE_20_TO_125_CHEBY1 = 0
    RANGE_125_TO_250_CHEBY1 = 1
    RANGE_20_TO_250_CHEBY1 = 2
    RANGE_20_TO_500_BUTTER = 3


class NormalizationType(Enum):
    """Enum class for normalization types."""

    MIN_MAX = 0
    Z_SCORE = 1


def filter_function(
    data_array: npt.NDArray[np.float32],
    *,
    filter_type: FilterType | None = None,
    normalization_type: NormalizationType | None = None,
    moving_average_type: MovingAverageType | None = None,
) -> npt.NDArray[np.float32]:
    """Filter the data passed in in different ways.

    This function applies mean subtraction, absolute value, and then optionally a
    filter, normalization, and moving average to the input array.
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

    if filter_type == FilterType.RANGE_20_TO_125_CHEBY1:
        sos = cheby1sos20to125
    elif filter_type == FilterType.RANGE_20_TO_250_CHEBY1:
        sos = cheby1sos20to250
    elif filter_type == FilterType.RANGE_20_TO_500_BUTTER:
        sos = buttersos20to500
    elif filter_type == FilterType.RANGE_125_TO_250_CHEBY1:
        sos = cheby1sos125to250
    elif filter_type is None:
        pass
    else:
        raise ValueError(
            "Invalid filter_type. Choose from 'cheby1_20to125', 'cheby1_125to250', 'cheby1_20to250' ,'butter_20to500'."
        )

    # calculate the mean
    mean = float(np.mean(data_array))

    # mean subtraction
    data_array = data_array - mean

    # Apply the filter to the windowed data

    if filter_type is not None:
        # Apply the filter to the windowed data
        filtered_data_array = sosfiltfilt(sos, data_array)

        data_array = np.abs(filtered_data_array)

    if moving_average_type == MovingAverageType.EMA:
        data_array = exponential_moving_average(data_array)
    elif moving_average_type == MovingAverageType.SMA:
        data_array = simple_moving_average(data_array)
    elif moving_average_type is None:
        pass
    else:
        err = f"Invalid moving average type [{moving_average_type}]."
        raise ValueError(err)

    if normalization_type == NormalizationType.Z_SCORE:
        # z-score normalization
        data_array = (data_array - np.mean(data_array)) / np.std(data_array)
    elif normalization_type == NormalizationType.MIN_MAX:
        # min-max normalization
        data_array = (data_array - np.min(data_array)) / (
            np.max(data_array) - np.min(data_array)
        )
    elif normalization_type is None:
        pass
    else:
        err = f"Invalid normalization type [{normalization_type}]."
        raise ValueError(err)

    return data_array
