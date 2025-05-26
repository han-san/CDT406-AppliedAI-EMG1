import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import (  # type: ignore[import-untyped]
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from data import LabeledWindow, Window, to_state
from dataset_loading import ClassificationMethod, classify_window

# The online measurements are done with a sample rate of 7500 Hz.
online_timestep_window_size = 1500


def online_evaluate(
    measured_input: pd.DataFrame,
    expected_output: pd.DataFrame,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Evaluate the online testing measurements.

    Input and expected_output should have columns 'voltage' and 'state'.
    Expected_output has a sample rate of 5kHz while input has a sample rate of 7.5kHz.
    """
    # Make sure the input can be cleanly split into windows.
    assert (len(expected_output) % online_timestep_window_size) == 0

    window_count = len(measured_input) // online_timestep_window_size

    input_state_arr = measured_input["label"].to_numpy()
    expected_output_arr = expected_output["measurement"].to_numpy()
    expected_output_state_arr = expected_output["label"].to_numpy()

    input_window_labels = np.array_split(input_state_arr, window_count)

    expected_output_windows = [
        Window(window, None, None, None)
        for window in np.array_split(expected_output_arr, window_count)
    ]
    expected_output_window_labels = [
        [to_state(state) for state in window]
        for window in np.array_split(
            expected_output_state_arr,
            window_count,
        )
    ]
    assert len(expected_output_windows) == len(expected_output_window_labels)

    labeled_expected_output = [
        LabeledWindow(window, labels)
        for window, labels in zip(
            expected_output_windows,
            expected_output_window_labels,
        )
    ]
    # Predicted states is classified using majority voting.
    predicted_states = [
        to_state(int(np.max(window_labels))) for window_labels in input_window_labels
    ]
    expected_states = [
        classify_window(out, ClassificationMethod.TWENTY_PERCENT_EDGE_TRANSIENT)
        for out in labeled_expected_output
    ]
    assert len(predicted_states) == len(expected_states)

    a = np.array([np.argmax(v.value) for v in expected_states])
    b = np.array([np.argmax(v.value) for v in predicted_states])

    return a, b


if len(sys.argv) != 3:
    print("Usage: python online_evaluation.py predicted_data.csv true_data.csv")
    sys.exit(1)

input_path = Path(sys.argv[1])
expected_output_path = Path(sys.argv[2])
input_glob_name = "*BB.csv"
output_glob_name = "*BB_manual_labeled.csv"
input_files = list(input_path.rglob(input_glob_name))
expected_output_files = list(expected_output_path.rglob(output_glob_name))
labeled_tag = "_manual_labeled"
sorted_input_files = sorted(input_files, key=lambda path: path.name)
sorted_expected_output_files = sorted(
    expected_output_files,
    key=lambda path: path.name.replace(labeled_tag, ""),
)
assert len(sorted_input_files) == len(sorted_expected_output_files)
inputs_outputs = list(zip(sorted_input_files, sorted_expected_output_files))
for i, o in inputs_outputs:
    assert i.name == o.name.replace(labeled_tag, "")

input_dfs: list[pd.DataFrame] = []
output_dfs: list[pd.DataFrame] = []
for in_filepath, out_filepath in inputs_outputs:
    output_df = pd.read_csv(out_filepath, header=0)
    assert output_df["time"][0] == 0

    # The amount of measurements might not be evenly divisible by the window size,
    # so we get rid of some of the trailing measurements to make it evenly divisible.
    extraneous_meaurements_count = len(output_df) % online_timestep_window_size
    output_df = output_df[: len(output_df) - extraneous_meaurements_count]
    assert (len(output_df) % online_timestep_window_size) == 0
    assert len(output_df) > 0

    output_dfs.append(output_df)

    # FIXME: This needs to be tested with real files.
    input_df = pd.read_csv(in_filepath, names=["time", "measurement", "label"])
    input_df = input_df[: len(output_df)]
    assert (len(input_df) % online_timestep_window_size) == 0

    assert input_df["time"][0] == 0
    assert len(input_df) > 0

    first_label = input_df["label"][0]
    first_differing_label = input_df[input_df["label"] != first_label]
    assert (first_differing_label.index[0] % online_timestep_window_size) == 0

    input_dfs.append(input_df)

inputs = pd.concat(input_dfs)
outputs = pd.concat(output_dfs)

target_outputs, predicted_outputs = online_evaluate(inputs, outputs)
macro_f1 = f1_score(target_outputs, predicted_outputs, average="macro")
class_f1s = f1_score(target_outputs, predicted_outputs, average=None)
total_accuracy = accuracy_score(target_outputs, predicted_outputs, normalize=True)
# FIXME: Filter so we only include indexes that are true states.
rest_predictions = [
    (i, predict)
    for i, (target, predict) in enumerate(zip(target_outputs, predicted_outputs))
    if target in (0, 4)
]
rest_targets = [(i, a) for i, a in enumerate(target_outputs) if a in (0, 4)]
for (i1, _), (i2, _) in zip(rest_predictions, rest_targets):
    assert i1 == i2

grip_predictions = [
    (i, predict)
    for i, (target, predict) in enumerate(zip(target_outputs, predicted_outputs))
    if target == 1
]
grip_targets = [(i, a) for i, a in enumerate(target_outputs) if a == 1]
for (i1, _), (i2, _) in zip(grip_predictions, grip_targets):
    assert i1 == i2

hold_predictions = [
    (i, predict)
    for i, (target, predict) in enumerate(zip(target_outputs, predicted_outputs))
    if target == 2
]
hold_targets = [(i, a) for i, a in enumerate(target_outputs) if a == 2]
for (i1, _), (i2, _) in zip(hold_predictions, hold_targets):
    assert i1 == i2

release_predictions = [
    (i, predict)
    for i, (target, predict) in enumerate(zip(target_outputs, predicted_outputs))
    if target == 3
]
release_targets = [(i, a) for i, a in enumerate(target_outputs) if a == 3]
for (i1, _), (i2, _) in zip(release_predictions, release_targets):
    assert i1 == i2

rest_accuracy = accuracy_score(
    [t for _, t in rest_targets],
    [p for _, p in rest_predictions],
    normalize=True,
)
grip_accuracy = accuracy_score(
    [t for _, t in grip_targets],
    [p for _, p in grip_predictions],
    normalize=True,
)
hold_accuracy = accuracy_score(
    [t for _, t in hold_targets],
    [p for _, p in hold_predictions],
    normalize=True,
)
release_accuracy = accuracy_score(
    [t for _, t in release_targets],
    [p for _, p in release_predictions],
    normalize=True,
)

print(f"mf1: {macro_f1:.4f}")
print(f"per class f1: {class_f1s}")
print(f"total accuracy: {total_accuracy:.4f}")
print(f"rest accuracy: {rest_accuracy:.4f}")
print(f"grip accuracy: {grip_accuracy:.4f}")
print(f"hold accuracy: {hold_accuracy:.4f}")
print(f"release accuracy: {release_accuracy:.4f}")

cm = confusion_matrix(target_outputs, predicted_outputs)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["rest", "grip", "hold", "release"],
)
total_rests = sum(cm[0])
total_grips = sum(cm[1])
total_holds = sum(cm[2])
total_releases = sum(cm[3])
total_classifications = total_rests + total_grips + total_holds + total_releases
print(f"total rests: {total_rests}, {total_rests / total_classifications * 100:.2f}%")
print(f"total grips: {total_grips}, {total_grips / total_classifications * 100:.2f}%")
print(f"total holds: {total_holds}, {total_holds / total_classifications * 100:.2f}%")
print(
    f"total releases: {total_releases}, {total_releases / total_classifications * 100:.2f}%"
)

disp.plot()
plt.title("Confusion matrix for online testing")
plt.show()
