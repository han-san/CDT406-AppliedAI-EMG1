import sys
from pathlib import Path

import numpy as np
import tensorflow as tf  # type: ignore[import-untyped]

from model import (
    Model,
    channel_count,
    create_windows,
    load_data_files,
    timestep_window_size,
)

data_dir = Path(sys.argv[1])
assert data_dir.is_dir()
# data_paths = list(data_dir.iterdir())
data_paths = list(data_dir.glob("**/*M[2-4]*"))
print(f"Loading {len(data_paths)} measurement files.")

model = Model(Model.Type.LSTM, timestep_window_size, channel_count)

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


# FIXME: Figure out what batch_size we should have.
model.train(model_input, model_desired_output, batch_size=1, epochs=5)

# output = model.execute(model_inputs[0])

# desired_output_2 = model_desired_outputs[0].output
# print(f"output: {output}, desired output: {desired_output_2}")

exit()

print(model.model.summary())
print(f"input shape: {model.model.input_shape}")

# Convert the model to TFLite/LiteRT format.
converter = tf.lite.TFLiteConverter.from_keras_model(model.model)
tflite_model = converter.convert()

# Save the model.
model_path = Path("../../model/model.tflite")
with model_path.open("wb") as f:
    f.write(tflite_model)

# ============== Everything below here is just for testing ==============

# Load the model using TFLite/LiteRT.
interpreter = tf.lite.Interpreter(model_path=str(model_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]
print(input_shape)
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
print(input_data)
interpreter.set_tensor(input_details[0]["index"], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]["index"])

# Actual results
print(output_data)
