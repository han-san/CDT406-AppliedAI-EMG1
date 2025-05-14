from enum import Enum

import numpy as np
import numpy.typing as npt


class MovingAverageType(Enum):
    """A type of moving average."""

    SMA = 0
    EMA = 1


def exponential_moving_average(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Calculate the exponential moving average of the provided sequence."""
    alpha = 0.1
    ema = np.zeros_like(x)
    x = np.array(x)
    # Initialize with the first value
    ema[0] = x[0]
    times_alpha = x * alpha
    for i in range(1, len(x)):
        ema[i] = times_alpha[i] + (1 - alpha) * ema[i - 1]
    return ema


def simple_moving_average(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Calculate the simple moving average of the provided sequence."""
    N = 50  # Window size
    # Pad the input to avoid losing samples
    padded_x = np.pad(x, (N, 0), mode="edge")
    cumsum = np.cumsum(padded_x)
    moving_avg = (cumsum[N:] - cumsum[:-N]) / float(N)
    return moving_avg
