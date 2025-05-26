import sys
from pathlib import Path

import tensorflow as tf  # type: ignore[import-untyped]

if len(sys.argv) != 2:
    print("Usage: python convert.py modelname.keras")

model_path = Path(sys.argv[1])
if not model_path.is_file():
    err = f"Provided model [{model_path}] is not a file."
    raise ValueError(err)

if model_path.suffix != ".keras":
    err = f"Provided model [{model_path}] does not have a '.keras' suffix. Is it a keras model file?"
    raise ValueError(err)

model = tf.keras.models.load_model(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

out_path = model_path.with_suffix(".tflite")

with out_path.open("wb") as f:
    f.write(tflite_model)
