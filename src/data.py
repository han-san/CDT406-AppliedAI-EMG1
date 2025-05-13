from enum import Enum
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]

from preprocessing.FilterFunction import filter_function

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
            self.window = filter_function(window, 1)
            self.labels = labels

        def __len__(self) -> int:
            """Return the length of the window."""
            return len(self.window)

        window: npt.NDArray[np.float32]
        labels: list[State]

    def __init__(self, data: list[float], labels: list[State]) -> None:
        """Construct a data object representing a list of float measurements."""
        self._data = np.array(data, dtype=np.float32)
        self._labels = labels

    def window(self, begin: int, end: int) -> Window:
        """Return a window of the data, from begin to end."""
        return self.Window(self._data[begin:end], self._labels[begin:end])

    def __len__(self) -> int:
        """Return the number of measurements in the data."""
        return len(self._data)

    _data: npt.NDArray[np.float32]
    _labels: list[State]


def load_myoflex_data_file(filepath: Path) -> Data:
    """Read the data from a MyoFlex data file."""
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

        measurements = [float(s) for s in f.read().split(",")[3000:]]
        return Data(measurements, [label] * len(measurements))


def load_our_data_file(filepath: Path) -> Data:
    """Read the data from our data file."""
    readings = pd.read_csv(filepath, index_col=0, names=["time", "voltage", "label"])
    voltages = readings["voltage"]

    labels = [to_state(label) for label in readings["label"]]

    return Data(voltages.to_numpy(), labels)


def load_data_file(filepath: Path, data_type: DataType) -> Data:
    """Load the measurements from a data file."""
    if data_type == DataType.MYOFLEX:
        return load_myoflex_data_file(filepath)
    if data_type == DataType.OURS:
        return load_our_data_file(filepath)

    err = ValueError("DataType variable contains invalid enum type.")
    raise err


def load_data_files(filepaths: list[Path], data_type: DataType) -> list[Data]:
    """Load the measurements from multiple data files."""
    return [load_data_file(path, data_type) for path in filepaths]


def create_windows(data: Data, *, window_size: int, overlap: int) -> list[Data.Window]:
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
        Data.Window(np.array(window), labels)
        for window, labels in zip(data_windows, label_windows)
    ]


class Input:
    """Input that can be fed to the AI model."""

    def __init__(self, windows: list[Data.Window]) -> None:
        """Construct input from windows."""
        np_windows = np.array(
            [window.window for window in windows],
            dtype=np.float32,
        )
        # The LSTM layer expects 3D input, where the dimensions are
        # (samples, time steps, features).
        # https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
        self.input = np_windows.reshape(
            (len(np_windows), len(np_windows[0]), channel_count),
        )

    # 2D array
    input: npt.NDArray[np.float32]


class Output:
    """Output from the AI model."""

    def __init__(self, output_states: list[State]) -> None:
        """Construct from the 2D output from the AI model."""
        self.output = np.array(
            [
                np.array(output_state.value, dtype=np.float32)
                for output_state in output_states
            ],
            dtype=np.float32,
        )

    # 2D array
    output: npt.NDArray[np.float32]


def get_io_from_myoflex(data_dir: Path) -> tuple[Input, Output]:
    """Load the input and output data from files in the provided directory."""
    # https://www.nature.com/articles/s41597-023-02223-x
    # Format for filenames in wyoflex dataset:
    # P[1-28]C[1-3]S[1-4]M[1-10]F[1-2]O[1-2]
    # P = participant
    # C = cycle
    # S = sensor (channel)
    # M = movement
    #     1 = flexion
    #     2 = extension
    #     3 = ulnar deviation
    #     4 = radial deviation
    #     5 = hook grip
    #     6 = power grip
    #     7 = spherical grip
    #     8 = precision grip
    #     9 = lateral grip
    #     10 = pinch grip
    # F = forearm
    #     1 = right, 2 = left
    # O = offset (sets the baseline at 0)
    #     1 = no offset, 2 = offset
    #     (The article says the opposite, but looking at the data says otherwise)
    #
    # currently testing movements 1, 2, and 6 with sensor 1 and offset

    data_paths = list(data_dir.glob("**/*S1M[126]F*O2"))
    print(f"Loading {len(data_paths)} measurement files.")

    data_measurements = load_data_files(data_paths, DataType.MYOFLEX)

    segmented_measurements = [
        create_windows(data, window_size=200, overlap=50) for data in data_measurements
    ]

    # Flatten windows so we can train on them in one go.
    model_windows = [item for row in segmented_measurements for item in row]
    model_input = Input(model_windows)

    def classify_window(window: Data.Window) -> State:
        """Classify the window based on its composition of labels.

        The classification is based on how many consecutive labels are found at the end
        of the window. If the last 20% of the window contains only labels for a
        transient state, the window gets labeled as that transient state. Otherwise, if
        the last 50% of the window gets labeled as a static state, the window gets
        labeled as that static state.

        This gives a higher priority to transient states when we are moving towards
        them. In our labeling, we are quite conservative when moving from a transient
        state to a static state, so we instead
        """
        labels = window.labels
        last_twenty = labels[-len(labels) // 2 :]
        first_twenty = labels[: len(labels) // 2]
        # We assume that labeling is consecutive; moves in order rest, grip, hold,
        # release, rest; and windows will never contain more than 2 types of labels.

        # If the last 20% are all grips/releases, classify the window as grip/release.
        if last_twenty[-1] in (State.GRIP, State.RELEASE):
            return last_twenty[0]
        # If the first 20% are all grips/releases, classify the window as grip/release.
        if first_twenty[0] in (State.GRIP, State.RELEASE):
            return first_twenty[-1]
        # If there are no transient states at front or back of window, it should contain
        # only one static state, so we just the latest label.
        return labels[-1]

    window_labels = [classify_window(window) for window in model_windows]
    model_desired_output = Output(window_labels)
    return model_input, model_desired_output


def get_io_from_our_data(data_dir: Path) -> tuple[Input, Output]:
    data_paths = list(data_dir.rglob("*.csv"))
    print(f"Loading {len(data_paths)} measurement files.")

    data_measurements = load_data_files(data_paths, DataType.OURS)

    sample_rate = 5000
    ms = 1000
    sample_ratio = sample_rate // ms

    segmented_measurements = [
        create_windows(data, window_size=200 * sample_ratio, overlap=50 * sample_ratio)
        for data in data_measurements
    ]

    # Flatten windows so we can train on them in one go.
    model_windows = [item for row in segmented_measurements for item in row]
    model_input = Input(model_windows)

    # FIXME: Since we haven't yet decided exactly how to handle the labeling, we use the
    # label of the last measurement in the window as the desired label.
    window_labels = [window.labels[-1] for window in model_windows]
    model_desired_output = Output(window_labels)
    return model_input, model_desired_output


def get_input_and_output_from_data_files(
    data_dir: Path,
    data_type: DataType,
) -> tuple[Input, Output]:
    if data_type == DataType.MYOFLEX:
        return get_io_from_myoflex(data_dir)

    if data_type == DataType.OURS:
        return get_io_from_our_data(data_dir)

    err = ValueError("DataType variable contains invalid enum type.")
    raise err
