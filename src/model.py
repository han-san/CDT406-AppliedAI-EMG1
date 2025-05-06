"""AI model functionality.

Provides the functionality needed to create, train, and execute our AI model.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import tensorflow as tf  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from data import Input, Output

# FIXME: Unify the sample rate variables
sample_rate = 5000
sample_rate_to_ms_ratio = sample_rate // 1000
window_size_in_ms = 200

# The amount of measurement readings we use as input.
timestep_window_size = window_size_in_ms * sample_rate_to_ms_ratio


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
                        units=32,
                        activation=tf.keras.activations.tanh,
                    ),
                    tf.keras.layers.LSTM(units=16, unroll=True),
                    tf.keras.layers.Dense(
                        units=32,
                        activation=tf.keras.activations.tanh,
                    ),
                    tf.keras.layers.Dense(
                        units=4,
                        activation=tf.keras.activations.softmax,
                    ),
                ],
            )

            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[
                    "accuracy",
                    tf.keras.metrics.F1Score(average="macro"),
                    tf.keras.metrics.TruePositives(),
                    tf.keras.metrics.TrueNegatives(),
                    tf.keras.metrics.FalsePositives(),
                    tf.keras.metrics.FalseNegatives(),
                ],
            )
        else:
            msg = f"Trying to construct model with invalid enum value {model_type}"
            raise ValueError(msg)

    def train(
        self,
        model_input: Input,
        desired_output: Output,
        *,
        batch_size: int | None,
        epochs: int,
    ) -> None:
        """Train the model using the provided input for some number of epochs."""
        # TODO(johan): Figure out what we should set the batch size to.
        # TODO(johan): We want to split input into validation/testing sets.
        print("Starting training!")
        history = self.model.fit(
            model_input.input,
            desired_output.output,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
        )
        print("Finished training!")
        print(history.history)
        plt.plot(history.history["accuracy"], label="Accuracy")
        plt.plot(history.history["loss"], label="Loss")
        plt.plot(history.history["f1_score"], label="f1")
        plt.legend()
        plt.title("Model Loss and Accuracy")
        plt.ylabel("Loss/Accuracy")
        plt.xlabel("Epoch")
        plt.show()

    def execute(self, model_input: Input) -> Output:
        """Return the output of the model using the provided input."""
        return self.model.predict(model_input.input)

    model: tf.keras.Model
