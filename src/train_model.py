import sys
from pathlib import Path

import tensorflow as tf  # type: ignore[import-untyped]

from data import (
    DataType,
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


model = Model(Model.Type.LSTM, timestep_window_size, channel_count)
# FIXME: Figure out what batch_size we should have.
model.train(model_input, model_desired_output, batch_size=None, epochs=5)

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
