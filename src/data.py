from enum import Enum
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]

# The amount of measurements included in each reading.
channel_count = 1


class DataType(Enum):
    """The dataset being used."""

    MYOFLEX = 0
    OURS = 1


class State(Enum):
    """The output representing a state."""

    GRIP = (1.0, 0.0, 0.0, 0.0)
    RELEASE = (0.0, 1.0, 0.0, 0.0)
    REST = (0.0, 0.0, 1.0, 0.0)
    HOLD = (0.0, 0.0, 0.0, 1.0)


# FIXME: Maybe provide the ability to filter as a module in preprocessing?
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

        # FIXME: REMOVE TRIM WHEN HANDLING OWN MEASUREMENTS
        #        We remove 3 seconds from the start of measurements
        #        since they are in some kind of rest state
        measurements = [float(s) for s in f.read().split(",")[3000:]]
        return Data(measurements, [label] * len(measurements))


def load_our_data_file(filepath: Path) -> Data:
    """Read the data from our data file."""
    readings = pd.read_csv(filepath, header=4)
    # FIXME: We make the measurements have a sample rate of 1000hz.
    #        We also reverse the measurements, since they start at 20s and decrease.
    stepped = readings.iloc[::-5]
    voltages = stepped["C2 (V)"]
    # FIXME: Here we do labeling on specific files, later on the files should already be labeled.
    rest_stop = 0.0
    grip_stop = 0.0
    hold_stop = 0.0
    release_stop = 0.0
    if filepath.name == "testP5.csv":
        rest_stop = 9
        grip_stop = 12
        hold_stop = 14.3
        release_stop = 16.3
    elif filepath.name == "testJ4.csv":
        rest_stop = 7
        grip_stop = 8.8
        hold_stop = 11.8
        release_stop = 14
    elif filepath.name == "testJ.csv":
        rest_stop = 7.3
        grip_stop = 9.7
        hold_stop = 13
        release_stop = 14.3
    elif filepath.name == "testP4.csv":
        rest_stop = 9.3
        grip_stop = 11
        hold_stop = 14.5
        release_stop = 16.25
    elif filepath.name == "testJ3.csv":
        rest_stop = 6.4
        grip_stop = 8.6
        hold_stop = 12.5
        release_stop = 14
    elif filepath.name == "testJ2.csv":
        rest_stop = 7
        grip_stop = 8.7
        hold_stop = 12.4
        release_stop = 14
    elif filepath.name == "testP2.csv":
        rest_stop = 7.2
        grip_stop = 9
        hold_stop = 12.3
        release_stop = 14.7
    elif filepath.name == "testP3.csv":
        rest_stop = 9.2
        grip_stop = 11
        hold_stop = 14.4
        release_stop = 16
    else:
        err = ValueError(f"Unknown file [{filepath.name}] passed as measurement.")
        raise err
    # Turn them into ms.
    rest_stop *= 1000
    grip_stop *= 1000
    hold_stop *= 1000
    release_stop *= 1000

    rests = [State.REST] * int(rest_stop)
    grips = [State.GRIP] * int(grip_stop - rest_stop)
    holds = [State.HOLD] * int(hold_stop - grip_stop)
    releases = [State.RELEASE] * int(release_stop - hold_stop)
    other_rests = [State.REST] * int(len(voltages) - release_stop)
    labels = rests + grips + holds + releases + other_rests
    print(len(labels))

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

    # Since we haven't yet decided exactly how to handle the labeling, we use the label of
    # the first measurement in the window as the desired label.
    window_labels = [window.labels[0] for window in model_windows]
    model_desired_output = Output(window_labels)
    return model_input, model_desired_output


def get_io_from_our_data(data_dir: Path) -> tuple[Input, Output]:
    data_paths = list(data_dir.iterdir())
    print(f"Loading {len(data_paths)} measurement files.")

    data_measurements = load_data_files(data_paths, DataType.OURS)

    segmented_measurements = [
        create_windows(data, window_size=200, overlap=50) for data in data_measurements
    ]

    # Flatten windows so we can train on them in one go.
    model_windows = [item for row in segmented_measurements for item in row]
    model_input = Input(model_windows)

    # Since we haven't yet decided exactly how to handle the labeling, we use the label of
    # the first measurement in the window as the desired label.
    window_labels = [window.labels[0] for window in model_windows]
    model_desired_output = Output(window_labels)
    return model_input, model_desired_output


def get_input_and_output_from_data_files(
    data_dir: Path,
    data_type: DataType,
) -> tuple[Input, Output]:
    if data_type == DataType.MYOFLEX:
        return get_io_from_myoflex(data_dir)
    elif data_type == DataType.OURS:
        return get_io_from_our_data(data_dir)
    else:
        err = ValueError("DataType variable contains invalid enum type.")
        raise err
