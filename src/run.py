import sys
from pathlib import Path

import tflite_runtime.interpreter as tflite

from data import get_input_and_output_from_data_files
from run_model import load_and_run_tflite

model_path = Path("../model/model.tflite")

data_dir = Path(sys.argv[1])
assert data_dir.is_dir()

model_input, model_desired_output = get_input_and_output_from_data_files(data_dir)
load_and_run_tflite(tflite.Interpreter, model_path, model_input, model_desired_output)
