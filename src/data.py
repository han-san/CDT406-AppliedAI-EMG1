from enum import Enum

import numpy as np
import numpy.typing as npt

from preprocessing.FilterFunction import (
    FilterType,
    NormalizationType,
    filter_function,
)
from preprocessing.Moving_average_filter import MovingAverageType

# The amount of measurements included in each reading.
channel_count = 1


class DataType(Enum):
    """The dataset being used."""

    MYOFLEX = 0
    OURS = 1


class State(Enum):
    """The output representing a state."""

    REST = (1.0, 0.0, 0.0, 0.0)
    GRIP = (0.0, 1.0, 0.0, 0.0)
    HOLD = (0.0, 0.0, 1.0, 0.0)
    RELEASE = (0.0, 0.0, 0.0, 1.0)


def to_state(i: int) -> State:
    if i == 0:
        return State.REST
    if i == 1:
        return State.GRIP
    if i == 2:
        return State.HOLD
    if i == 3:
        return State.RELEASE
    err = ValueError("Invalid State int representation passed as argument.")
    raise err


class Window:
    """Represents a window of data measurements."""

    def __init__(
        self,
        window: npt.NDArray[np.float32],
    ) -> None:
        """Construct a window of data."""
        self.window = np.array(
            filter_function(
                window,
                filter_type=FilterType.RANGE_20_TO_500_BUTTER,
                normalization_type=NormalizationType.MIN_MAX,
                moving_average_type=MovingAverageType.EMA,
            ),
            dtype=np.float32,
        )

    def __len__(self) -> int:
        """Return the length of the window."""
        return len(self.window)

    window: npt.NDArray[np.float32]


class LabeledWindow:
    """Represents a window of data measurements paired with their labelings."""

    def __init__(self, window: Window, labels: list[State]) -> None:
        """Construct a window containing labels for each measurement."""
        self.window = window
        self.labels = labels

    def __len__(self) -> int:
        """Return the length of the window."""
        return len(self.window)

    window: Window
    labels: list[State]


class Data:
    """Represents a list of data measurements."""

    def __init__(self, data: list[float], labels: list[State]) -> None:
        """Construct a data object representing a list of float measurements."""
        self._data = np.array(data, dtype=np.float32)
        self._labels = labels

    def window(self, begin: int, end: int) -> LabeledWindow:
        """Return a window of the data, from begin to end."""
        return LabeledWindow(
            Window(self._data[begin:end]),
            self._labels[begin:end],
        )

    def __len__(self) -> int:
        """Return the number of measurements in the data."""
        return len(self._data)

    _data: npt.NDArray[np.float32]
    _labels: list[State]


def create_windows(
    data: Data,
    *,
    window_size: int,
    overlap: int,
) -> list[LabeledWindow]:
    """Split measurements into multiple windows."""
    # TODO(johan): Decide what to do with the last window if it is < windowSize.
    #   - ignore?
    #   - use average of all measurements (a group from previous years did this)?
    data_windows = np.lib.stride_tricks.sliding_window_view(data._data, window_size)[
        :: window_size - overlap,
        :,
    ]
    label_value_windows = np.lib.stride_tricks.sliding_window_view(
        np.array(data._labels),
        window_size,
    )[:: window_size - overlap, :]
    label_windows = [
        [State(value) for value in window] for window in label_value_windows
    ]

    return [
        LabeledWindow(Window(np.array(window)), labels)
        for window, labels in zip(data_windows, label_windows)
    ]


class Input:
    """Input that can be fed to the AI model."""

    def __init__(self, window: Window) -> None:
        """Construct input from a window."""
        # The LSTM layer expects 3D input, where the dimensions are
        # (samples, time steps, features).
        # https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
        # One input represents the (time steps, features) part, and then an array of inputs
        # creates the (samples, _, _) dimension.
        self.input = window.window.reshape((len(window.window), channel_count))

    input: npt.NDArray[np.float32]


class Output:
    """Output from the AI model."""

    def __init__(self, output_state: State) -> None:
        """Construct output corresponding to the provided state."""
        self.output = np.array(output_state.value, dtype=np.float32)

    output: npt.NDArray[np.float32]
