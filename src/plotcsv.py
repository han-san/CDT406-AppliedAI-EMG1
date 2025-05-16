import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing.FilterFunction import (
    FilterType,
    NormalizationType,
    filter_function,
)
from preprocessing.Moving_average_filter import MovingAverageType


def plot_file(filepath: Path) -> None:
    voltages = pd.read_csv(filepath, header=None, skiprows=1)
    # voltages[1] = filter_function(voltages[1], 1)

    window = voltages[1].to_numpy()

    mean = np.mean(window)
    std = np.std(window)
    # mean subtraction
    window = window - mean
    window = np.abs(window)
    voltages[3] = (window - mean) / std
    voltages[4] = filter_function(
        voltages[1].to_numpy(),
        filter_type=None,
        normalization_type=NormalizationType.Z_SCORE,
        moving_average_type=None,
    )
    voltages[5] = filter_function(
        voltages[1].to_numpy(),
        filter_type=FilterType.RANGE_20_TO_500_BUTTER,
        normalization_type=NormalizationType.Z_SCORE,
        moving_average_type=MovingAverageType.SMA,
    )
    voltages[6] = filter_function(
        voltages[1].to_numpy(),
        filter_type=FilterType.RANGE_20_TO_500_BUTTER,
        normalization_type=NormalizationType.Z_SCORE,
        moving_average_type=MovingAverageType.SMA,
    )
    voltages[7] = filter_function(
        voltages[1].to_numpy(),
        filter_type=FilterType.RANGE_20_TO_500_BUTTER,
        normalization_type=NormalizationType.MIN_MAX,
        moving_average_type=MovingAverageType.SMA,
    )
    voltages[8] = filter_function(
        voltages[1].to_numpy(),
        filter_type=FilterType.RANGE_20_TO_250_CHEBY1,
        normalization_type=NormalizationType.Z_SCORE,
        moving_average_type=MovingAverageType.SMA,
    )
    voltages.columns = pd.Index(
        [
            "seconds",
            "original",
            "labels",
            "first normalization (Classifies both rest and hold)",
            "no filter, but normalization",
            "butter 20-500 z-score no MA",
            "butter 20-500 z-score with MA",
            "butter 20-500 min_max with MA",
            "cheby1 20-250 z-score with MA",
        ],
    )
    voltages.plot(x=0, title=filepath.name)
    plt.show()


def plot_myoflex(filepath: Path) -> None:
    if "M6" not in filepath.name:
        return
    if "S4" not in filepath.name:
        return
    if "O2" not in filepath.name:
        return
    with filepath.open("r") as f:
        measurements = np.array(
            [float(s) for s in f.read().split(",")],
            dtype=np.float32,
        )
        plt.plot(measurements)
        plt.title(filepath.name)
        plt.show()


path = Path(sys.argv[1])
if path.is_file():
    plot_file(path)
elif path.is_dir():
    if path.name == "originaldataset":
        for f in path.iterdir():
            plot_myoflex(f)
    else:
        for f in path.glob("**/*.csv"):
            plot_file(f)
else:
    err = "Arg provided is not a directory or a file."
    raise ValueError(err)
