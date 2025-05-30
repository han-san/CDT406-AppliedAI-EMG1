import sys
from pathlib import Path

import tensorflow as tf  # type: ignore[import-untyped]

from data import DataType
from dataset_loading import get_input_and_output_from_data_files
from preprocessing.FilterFunction import FilterType, NormalizationType
from preprocessing.Moving_average_filter import MovingAverageType
from run_model import run_metrics_on_tflite_model
from tflite_model import TFLiteModel

if len(sys.argv) != 3:
    print("Usage: python run.py model_path data_dir")

model_path = Path(sys.argv[1])

data_dir = Path(sys.argv[2])
assert data_dir.is_dir()

model_name = model_path.name

# We figure out what pre-processing to do based on the model's filename.
filter_type: FilterType
if "20to250" in model_name:
    filter_type = FilterType.RANGE_20_TO_250_CHEBY1
elif "20to125" in model_name:
    filter_type = FilterType.RANGE_20_TO_125_CHEBY1
elif "125to250" in model_name:
    filter_type = FilterType.RANGE_125_TO_250_CHEBY1
elif "20to500" in model_name:
    filter_type = FilterType.RANGE_20_TO_500_BUTTER
else:
    err = "No filter type found in model's filename."
    raise ValueError(err)

normalization: NormalizationType
if "zscore" in model_name:
    normalization = NormalizationType.Z_SCORE
elif "minmax" in model_name:
    normalization = NormalizationType.MIN_MAX
else:
    err = "No normalization type found in model's filename."
    raise ValueError(err)

ma: MovingAverageType
if "ema" in model_name:
    ma = MovingAverageType.EMA
elif "sma" in model_name:
    ma = MovingAverageType.SMA
else:
    err = "No moving average type found in model's filename."
    raise ValueError(err)

print("Using preprocessing:")
print(filter_type)
print(normalization)
print(ma)
print(model_path.name)

model_input, model_desired_output = get_input_and_output_from_data_files(
    data_dir,
    DataType.OURS,
    filter_type,
    normalization,
    ma,
)
model = TFLiteModel(tf.lite.Interpreter, model_path)
run_metrics_on_tflite_model(model, model_input, model_desired_output)
