"""AI model functionality.

Provides the functionality needed to create, train, and execute our AI model.
"""

from enum import Enum
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tensorflow as tf  # type: ignore[import-untyped]
import matplotlib.pyplot as plt

# The amount of measurement readings we use as input.
timestep_window_size = 200
# The amount of measurements included in each reading.
channel_count = 1


class State(Enum):
    """The output representing a state."""

    GRIP = (1.0, 0.0, 0.0)
    RELEASE = (0.0, 1.0, 0.0)
    REST = (0.0, 0.0, 1.0)


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


class Model:
    """Our AI Model."""

    class Type(Enum):
        """The type of model."""

        LSTM = 0

    def __init__(self, model_type: Type, timesteps: int, samples: int) -> None:
        """Construct the AI model."""
        if model_type == Model.Type.LSTM:
            # Create the same type of model as in https://doi.org/10.3390/app12199700.
            # TODO(johan): Check if we need to do more work for LSTM to be stateful.
            #     - https://keras.io/getting_started/faq/#how-can-i-use-stateful-rnns
            #     - https://www.tensorflow.org/tutorials/structured_data/time_series
            # The 'unroll=True' has to be set in the LSTM layer to be able to run the model
            # using the tflite-runtime. Otherwise the model contains the OP
            # "FlexTensorListReserve" which is only available in the regular tensorflow
            # package.
            self.model = tf.keras.models.Sequential(
                [
                    tf.keras.Input(shape=(timesteps, samples)),
                    tf.keras.layers.Dense(
                        units=32, activation=tf.keras.activations.tanh
                    ),
                    tf.keras.layers.LSTM(units=16, unroll=True),
                    tf.keras.layers.Dense(
                        units=32, activation=tf.keras.activations.tanh
                    ),
                    tf.keras.layers.Dense(
                        units=3, activation=tf.keras.activations.softmax
                    ),
                ],
            )

            # "Configures the model for training"
            # TODO(johan): Might want to change the arguments.
            self.model.compile(
                loss=tf.keras.losses.MeanSquaredError(),
            )
        else:
            msg = f"Trying to construct model with invalid enum value {model_type}"
            raise ValueError(msg)

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

    def train(
        self,
        model_input: Input,
        desired_output: Output,
        *,
        batch_size: int,
        epochs: int,
    ) -> None:
        """Train the model using the provided input for some number of epochs."""
        # TODO(johan): Actually generate target labels properly.
        # TODO(johan): Figure out what we should set the batch size to.
        print("Starting training!")
        history = self.model.fit(
            model_input.input,
            desired_output.output,
            epochs=epochs,
            verbose=2,
        )
        print("Finished training!")
        print(history.history)
        plt.plot(history.history["loss"])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.show()

    def execute(self, model_input: Input) -> Output:
        """Return the output of the model using the provided input."""
        return self.model.predict(model_input.input)

    model: tf.keras.Model


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

        measurements = [float(s) for s in f.read().split(",")]
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
