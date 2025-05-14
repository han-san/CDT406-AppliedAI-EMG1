import numpy as np


def calculate_moving_average(x):
    alpha = 0.1
    ema = np.zeros_like(x)
    x = np.array(x)
    ema[0] = x[0]
    #Initialize with the first value
    for i in range(1, len(x)):
        ema[i] = alpha * x[i] + (1 - alpha) * ema[i - 1]
    return ema

def simple_moving_average(x):
    N = 50  # Window size
    # Pad the input to avoid losing samples
    padded_x = np.pad(x, (N, 0), mode='edge')
    cumsum = np.cumsum(padded_x)
    moving_avg = (cumsum[N:] - cumsum[:-N]) / float(N)
    return moving_avg