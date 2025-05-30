import argparse
import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf  # type: ignore[import-untyped]

from data import (
    DataType,
    Input,
    Output,
    State,
    channel_count,
)
from dataset_loading import get_input_and_output_from_data_files
from model import (
    Model,
    timestep_window_size,
)
from preprocessing.FilterFunction import FilterType, NormalizationType
from preprocessing.Moving_average_filter import MovingAverageType

filter_names = ["20to125", "125to250", "20to250", "20to500"]
normalization_names = ["minmax", "zscore"]
moving_average_names = ["sma", "ema"]
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filter", required=True, choices=filter_names)
parser.add_argument(
    "-n",
    "--normalization",
    required=True,
    choices=normalization_names,
)

parser.add_argument(
    "-ma",
    "--movingaverage",
    required=True,
    choices=moving_average_names,
)

parser.add_argument("-eq", "--equalize", action="store_true")
parser.add_argument("-d", "--datadir", required=True, type=Path)
parser.add_argument("-m", "--modelsavedir", required=True, type=Path)
parser.add_argument("-p", "--prefix")

args = parser.parse_args()


class Config:
    filter: FilterType
    normalization: NormalizationType
    moving_average: MovingAverageType
    equalize_training_data: bool
    data_dir: Path
    model_dir: Path


config = Config()
config.filter = FilterType(filter_names.index(args.filter))
config.normalization = NormalizationType(normalization_names.index(args.normalization))
config.moving_average = MovingAverageType(
    moving_average_names.index(args.movingaverage),
)
config.equalize_training_data = args.equalize
config.data_dir = args.datadir
config.model_dir = args.modelsavedir

assert config.data_dir.is_dir()

model_input, model_desired_output = get_input_and_output_from_data_files(
    config.data_dir,
    DataType.OURS,
    config.filter,
    config.normalization,
    config.moving_average,
)

eq_name = "eq" if config.equalize_training_data else "noeq"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_name = f"{current_time}-model-{args.prefix}-{eq_name}-{args.filter}-{args.normalization}-{args.movingaverage}"
model_path = config.model_dir / f"{model_name}.tflite"


def limit_training_data(
    train_in: list[Input],
    train_out: list[Output],
) -> None:
    """Equalizes the training data of the different classes.

    Limits the training data so that each class at maximum has the same number of
    windows as the one of the two transient states with the most training data.
    """
    rng = np.random.default_rng()
    in_data = [inp.input for inp in train_in]
    out_data = [out.output for out in train_out]
    comb = list(zip(in_data, out_data))
    rng.shuffle(comb)

    shuffled_in_data = np.array([x for x, _ in comb])
    shuffled_out_data = np.array([y for _, y in comb])

    rest_condition = [np.all(state == State.REST.value) for state in shuffled_out_data]
    grip_condition = [np.all(state == State.GRIP.value) for state in shuffled_out_data]
    hold_condition = [np.all(state == State.HOLD.value) for state in shuffled_out_data]
    release_condition = [
        np.all(state == State.RELEASE.value) for state in shuffled_out_data
    ]

    rest_outputs = shuffled_out_data[rest_condition]
    grip_outputs = shuffled_out_data[grip_condition]
    hold_outputs = shuffled_out_data[hold_condition]
    release_outputs = shuffled_out_data[release_condition]

    rest_inputs = shuffled_in_data[rest_condition]
    grip_inputs = shuffled_in_data[grip_condition]
    hold_inputs = shuffled_in_data[hold_condition]
    release_inputs = shuffled_in_data[release_condition]

    longest_transient = max(len(grip_inputs), len(release_inputs))

    new_input = np.concatenate(
        (
            rest_inputs[:longest_transient],
            grip_inputs[:longest_transient],
            hold_inputs[:longest_transient],
            release_inputs[:longest_transient],
        ),
    )
    new_output = np.concatenate(
        (
            rest_outputs[:longest_transient],
            grip_outputs[:longest_transient],
            hold_outputs[:longest_transient],
            release_outputs[:longest_transient],
        ),
    )

    comb = list(zip(new_input, new_output))
    rng.shuffle(comb)

    for dest_in, src_in in zip(train_in, shuffled_in_data):
        dest_in.input = src_in

    for dest_out, src_out in zip(train_out, shuffled_out_data):
        dest_out.output = src_out


if config.equalize_training_data:
    limit_training_data(model_input, model_desired_output)

rest_condition = [
    np.all(output.output == State.REST.value) for output in model_desired_output
]
grip_condition = [
    np.all(output.output == State.GRIP.value) for output in model_desired_output
]
hold_condition = [
    np.all(output.output == State.HOLD.value) for output in model_desired_output
]
release_condition = [
    np.all(output.output == State.RELEASE.value) for output in model_desired_output
]
rest_count = np.count_nonzero(rest_condition)
grip_count = np.count_nonzero(grip_condition)
hold_count = np.count_nonzero(hold_condition)
release_count = np.count_nonzero(release_condition)
total_count = rest_count + grip_count + hold_count + release_count
assert total_count == len(model_desired_output)

# FIXME: This is probably an incorrect way of calculating the bias.
# Taken from https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#optional_set_the_correct_initial_bias,
# but the example is a binary classification whereas we have multiple classes.
bias = np.log(
    [
        rest_count / (grip_count + hold_count + release_count),
        grip_count / (rest_count + hold_count + release_count),
        hold_count / (rest_count + grip_count + release_count),
        release_count / (rest_count + grip_count + hold_count),
    ],
)

# Give higher weight to less frequent classes.
# From: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#calculate_class_weights
weights = {
    0: (1 / rest_count) * (total_count / 2.0),
    1: (1 / grip_count) * (total_count / 2.0),
    2: (1 / hold_count) * (total_count / 2.0),
    3: (1 / release_count) * (total_count / 2.0),
}

model = Model(
    Model.Type.LSTM,
    timestep_window_size,
    channel_count,
    training_data_bias=None,
)
print(model.model.summary())
# FIXME: Figure out what batch_size we should have.
model.train(
    model_input,
    model_desired_output,
    model_name,
    batch_size=64,
    epochs=2000,
    class_weight=None,
)

print(model.model.summary())
print(f"input shape: {model.model.input_shape}")

# Convert the model to TFLite/LiteRT format.
converter = tf.lite.TFLiteConverter.from_keras_model(model.model)
tflite_model = converter.convert()

# Save the model.
with model_path.open("wb") as f:
    f.write(tflite_model)
