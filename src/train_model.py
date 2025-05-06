import sys
from pathlib import Path

import numpy as np
import tensorflow as tf  # type: ignore[import-untyped]

from data import (
    DataType,
    Input,
    Output,
    State,
    channel_count,
    get_input_and_output_from_data_files,
)
from model import (
    Model,
    timestep_window_size,
)
from run_model import load_and_run_tflite

model_path = Path("../model/model.tflite")


data_dir = Path(sys.argv[1])
assert data_dir.is_dir()

model_input, model_desired_output = get_input_and_output_from_data_files(
    data_dir,
    DataType.OURS,
)

if model_path.exists():
    # FIXME(Johan): Use validation set to test.
    load_and_run_tflite(
        tf.lite.Interpreter,
        model_path,
        model_input,
        model_desired_output,
    )
    exit()


def limit_training_data(train_in: Input, train_out: Output) -> None:
    """Equalizes the training data of the different classes.

    Limits the training data so that each class at maximum has the same number of
    windows as the one of the two transient states with the most training data.
    """
    rng = np.random.default_rng()
    comb = list(zip(train_in.input, train_out.output))
    rng.shuffle(comb)

    train_in.input = np.array([x for x, _ in comb])
    train_out.output = np.array([y for _, y in comb])

    rest_condition = [np.all(state == State.REST.value) for state in train_out.output]
    grip_condition = [np.all(state == State.GRIP.value) for state in train_out.output]
    hold_condition = [np.all(state == State.HOLD.value) for state in train_out.output]
    release_condition = [
        np.all(state == State.RELEASE.value) for state in train_out.output
    ]

    rest_outputs = train_out.output[rest_condition]
    grip_outputs = train_out.output[grip_condition]
    hold_outputs = train_out.output[hold_condition]
    release_outputs = train_out.output[release_condition]

    rest_inputs = train_in.input[rest_condition]
    grip_inputs = train_in.input[grip_condition]
    hold_inputs = train_in.input[hold_condition]
    release_inputs = train_in.input[release_condition]

    longest_transient = max(len(grip_inputs), len(release_inputs))

    new_input = np.concat(
        (
            rest_inputs[:longest_transient],
            grip_inputs[:longest_transient],
            hold_inputs[:longest_transient],
            release_inputs[:longest_transient],
        ),
    )
    new_output = np.concat(
        (
            rest_outputs[:longest_transient],
            grip_outputs[:longest_transient],
            hold_outputs[:longest_transient],
            release_outputs[:longest_transient],
        ),
    )

    comb = list(zip(new_input, new_output))
    rng.shuffle(comb)

    train_in.input = np.array([x for x, _ in comb])
    train_out.output = np.array([y for _, y in comb])


limit_training_data(model_input, model_desired_output)

model = Model(Model.Type.LSTM, timestep_window_size, channel_count)
# FIXME: Figure out what batch_size we should have.
model.train(model_input, model_desired_output, batch_size=64, epochs=1000)

# output = model.execute(model_inputs[0])

# desired_output_2 = model_desired_outputs[0].output
# print(f"output: {output}, desired output: {desired_output_2}")

print(model.model.summary())
print(f"input shape: {model.model.input_shape}")

# Convert the model to TFLite/LiteRT format.
converter = tf.lite.TFLiteConverter.from_keras_model(model.model)
tflite_model = converter.convert()

# Save the model.
with model_path.open("wb") as f:
    f.write(tflite_model)
