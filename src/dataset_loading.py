from enum import Enum
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from data import (
    Data,
    DataType,
    Input,
    LabeledWindow,
    Output,
    State,
    create_windows,
    to_state,
)


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
    readings = pd.read_csv(filepath, index_col=0, header=0)
    voltages = readings["measurement"]

    labels = [to_state(label) for label in readings["label"]]

    return Data(voltages.to_list(), labels)


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


class ClassificationMethod(Enum):
    """The method used for classifying a window of training data."""

    TWENTY_PERCENT_EDGE_TRANSIENT = 0
    PROPORTIONAL = 1


def classify_window(
    window: LabeledWindow, classification_method: ClassificationMethod
) -> State:
    """Classify the window based on its composition of labels.

    There are multiple aspects to the classification. We assume that a window cannot
    contain both transient states at the same time (the static states should be held for
    longer than 200ms). We also assume that a window can't contain both static states at
    the same time if there is no transient state.

    1. If there are no transient states in the window, it should be entirely one static
    state and should be labeled as such.

    2. If an entire sequence of a transient state is in the window (meaning the first
    and last measurements aren't of the transient state, but there is a transient
    state), it should be labeled as that transient state.

    3. If the last 20% of the labels are of a transient state, the window should be
    labeled as that state. Otherwise, if the first 20% of the labels are of a transient
    state, it should be labeled as that state. Otherwise, the latest non-transient state
    should be the label.

    This gives a higher priority to the transient states when we are moving towards
    them.
    """
    if classification_method == ClassificationMethod.TWENTY_PERCENT_EDGE_TRANSIENT:
        labels = window.labels

        grip_in_window = State.GRIP in window.labels
        release_in_window = State.RELEASE in window.labels
        transient_state_in_window = grip_in_window or release_in_window
        # We shouldn't have both transient states in one window. Rests and holds should
        # continue for a while.
        assert not (grip_in_window and release_in_window)
        if not transient_state_in_window:
            # If there isn't a transient state in the window, it should be entirely one
            # static state.
            assert all(x == labels[0] for x in labels)
            return labels[-1]

        transient_state = State.GRIP if grip_in_window else State.RELEASE
        edge_labels_are_not_transient = (
            labels[0] != transient_state and labels[-1] != transient_state
        )
        if edge_labels_are_not_transient:
            # The entire transient state sequence must be in the window, so we treat
            # it as a transient state.
            return transient_state

        last_twenty = labels[-len(labels) // 5 :]
        first_twenty = labels[: len(labels) // 5]
        # If the last 20% are all grips/releases, classify the window as grip/release.
        if all(x == transient_state for x in last_twenty):
            return last_twenty[0]

        # If the first 20% are all grips/releases, classify the window as grip/release.
        if all(x == transient_state for x in first_twenty):
            return last_twenty[0]

        # The first of the last twenty labels is guaranteed to be the latest static
        # state at this point, so that becomes the label for the window.
        assert last_twenty[0] != transient_state
        return last_twenty[0]

    err = f"Invalid classification method [{classification_method}]"
    raise ValueError(err)


def get_io_from_myoflex(data_dir: Path) -> tuple[list[Input], list[Output]]:
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
    model_input = [Input(window.window) for window in model_windows]

    window_labels = [
        classify_window(window, ClassificationMethod.TWENTY_PERCENT_EDGE_TRANSIENT)
        for window in model_windows
    ]
    model_desired_output = [Output(label) for label in window_labels]
    return model_input, model_desired_output


def get_io_from_our_data(
    data_dir: Path,
) -> tuple[list[Input], list[Output]]:
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
    model_input = [Input(window.window) for window in model_windows]

    window_labels = [
        classify_window(window, ClassificationMethod.TWENTY_PERCENT_EDGE_TRANSIENT)
        for window in model_windows
    ]
    model_desired_output = [Output(label) for label in window_labels]
    return model_input, model_desired_output


def get_input_and_output_from_data_files(
    data_dir: Path,
    data_type: DataType,
) -> tuple[list[Input], list[Output]]:
    if data_type == DataType.MYOFLEX:
        return get_io_from_myoflex(data_dir)

    if data_type == DataType.OURS:
        return get_io_from_our_data(data_dir)

    err = ValueError("DataType variable contains invalid enum type.")
    raise err
