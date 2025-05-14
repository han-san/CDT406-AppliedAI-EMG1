
import numpy
import numpy as np
import pandas as pd
from enum import Enum


def calculate_moving_average(x):
    #N=50
    #cumsum = numpy.cumsum(numpy.insert(x, 0, 0))
    #return (cumsum[N:] - cumsum[:-N]) / float(N)
    alpha = 0.1
    ema = np.zeros_like(x)
    x = np.array(x)
    ema[0] = x[0]
    # Initialize with the first value
    for i in range(1, len(x)):
        ema[i] = alpha * x[i] + (1 - alpha) * ema[i - 1]
    return ema
    # Call the function and print the result

