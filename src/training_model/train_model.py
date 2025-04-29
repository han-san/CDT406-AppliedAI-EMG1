import sys
from pathlib import Path

import numpy as np
import tensorflow as tf  # type: ignore[import-untyped]

from data import create_windows, load_data_files
from model import (
    Model,
    channel_count,
    timestep_window_size,
)

model_path = Path("../../model/model.tflite")


def _load_and_run_tflite(
    model_input: Model.Input,
    expected_output: Model.Output,
) -> None:
    """Test the accuracy of the model."""
    # Load the model using TFLite/LiteRT.
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]

    correct = 0
    total = 0

    for in_window, out in zip(model_input.input, expected_output.output):
        input_data = in_window
        input_data = input_data.reshape(input_shape)
        interpreter.set_tensor(input_details[0]["index"], input_data)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]["index"])

        if np.argmax(output_data[0]) == np.argmax(out):
            correct += 1

        total += 1

    print(f"Accuracy: {correct / total * 100}%")


data_dir = Path(sys.argv[1])
assert data_dir.is_dir()
# data_paths = list(data_dir.iterdir())

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


data_measurements = load_data_files(data_paths)

segmented_measurements = [
    create_windows(data, timestep_window_size) for data in data_measurements
]

# Flatten windows so we can train on them in one go.
model_windows = [item for row in segmented_measurements for item in row]
model_input = Model.Input(model_windows)

# Since we haven't yet decided exactly how to handle the labeling, we use the label of
# the first measurement in the window as the desired label.
window_labels = [window.labels[0] for window in model_windows]
model_desired_output = Model.Output(window_labels)

if model_path.exists():
    # FIXME(Johan): Use validation set to test.
    _load_and_run_tflite(model_input, model_desired_output)
    exit()


model = Model(Model.Type.LSTM, timestep_window_size, channel_count)
# FIXME: Figure out what batch_size we should have.
model.train(model_input, model_desired_output, batch_size=1, epochs=5)

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
