import sys
from pathlib import Path

import tflite_runtime.interpreter as tflite

from data import DataType
from dataset_loading import get_input_and_output_from_data_files
from run_model import run_metrics_on_tflite_model
from tflite_model import TFLiteModel

model_path = Path("../model/model.tflite")

data_dir = Path(sys.argv[1])
assert data_dir.is_dir()

model_input, model_desired_output = get_input_and_output_from_data_files(
    data_dir,
    DataType.OURS,
)
model = TFLiteModel(tflite.Interpreter, model_path)
run_metrics_on_tflite_model(model, model_input, model_desired_output)
