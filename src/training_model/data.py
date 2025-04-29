from enum import Enum
from pathlib import Path

import numpy as np
import numpy.typing as npt

# import scipy as sp  # type: ignore[import-untyped]


class State(Enum):
    """The output representing a state."""

    GRIP = (1.0, 0.0, 0.0)
    RELEASE = (0.0, 1.0, 0.0)
    REST = (0.0, 0.0, 1.0)


# FIXME: Maybe provide the ability to filter as a module in pre-processing?
# Define filter parameters
order = 4  # Filter order
cutoff = 100  # Cutoff frequency in Hz
fs = 1000  # Sampling frequency in Hz

# Design a low-pass Butterworth filter
# b, a = sp.signal.butter(order, cutoff / (fs / 2), btype="low", analog=False)


# def filter_window(data: list[float]) -> npt.NDArray[np.float32]:
#     """Filter the data."""
#     return sp.signal.lfilter(b, a, np.array(data, dtype=np.float32))


class Data:
    """Represents a list of data measurements."""

    class Window:
        """Represents a window of data measurements."""

        def __init__(
            self,
            window: npt.NDArray[np.float32],
            labels: list[State],
        ) -> None:
            """Construct a window of data."""
            self.window = np.array(window)

            mean = np.mean(window)
            std = np.std(window)
            # mean subtraction
            window = window - mean
            window = np.abs(window)
            window_normalized = (window - mean) / std
            # filtered_window = sp.signal.filtfilt(b, a, window_normalized)

            self.window = window_normalized
            # self.window = filtered_window
            self.labels = labels

        def __len__(self) -> int:
            """Return the length of the window."""
            return len(self.window)

        window: npt.NDArray[np.float32]
        labels: list[State]

    def __init__(self, data: list[float], labels: list[State]) -> None:
        """Construct a data object representing a list of float measurements."""
        self._data = np.array(data, dtype=np.float32)
        # self._data = np.array(data, dtype=np.float32)
        # plt.plot(data, color="green", label="Original Signal")
        # plt.plot(self._data, color="red", label="Filtered Signal")
        # plt.ylabel("Amplitude")
        # plt.xlabel("Sample Number")
        # plt.show()
        # exit()

        self._labels = labels

    def window(self, begin: int, end: int) -> Window:
        """Return a window of the data, from begin to end."""
        return self.Window(self._data[begin:end], self._labels[begin:end])

    def __len__(self) -> int:
        """Return the number of measurements in the data."""
        return len(self._data)

    _data: npt.NDArray[np.float32]
    _labels: list[State]


def load_data_file(filepath: Path) -> Data:
    """Load the measurements from a data file."""
    with filepath.open("r") as f:
        filename = filepath.name
        label = None
        if "M1" in filename:
            label = State.GRIP
        elif "M2" in filename:
            label = State.RELEASE
        elif "M6" in filename:
            label = State.REST
        else:
            err = ValueError("The filename doesn't include a valid gesture classifier.")
            raise err

        # FIXME: REMOVE TRIM WHEN HANDLING OWN MEASUREMENTS
        #        We remove 3 seconds from the start of measurements
        #        since they are in some kind of rest state
        measurements = [float(s) for s in f.read().split(",")[3000:]]
        return Data(measurements, [label] * len(measurements))


def load_data_files(filepaths: list[Path]) -> list[Data]:
    """Load the measurements from multiple data files."""
    return [load_data_file(path) for path in filepaths]


def create_windows(data: Data, window_size: int) -> list[Data.Window]:
    """Split measurements into multiple windows."""
    # TODO(johan): Decide what to do with the last window if it is < windowSize.
    #   - ignore?
    #   - use average of all measurements (a group from previous years did this)?
    window_count = len(data) // window_size
    return [
        data.window(i * window_size, i * window_size + window_size)
        for i in range(window_count)
    ]
