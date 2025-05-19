import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from data import Input, Output
from tflite_model import TFLiteModel


def run_metrics_on_tflite_model(
    model: TFLiteModel,
    model_input: list[Input],
    expected_output: list[Output],
) -> None:
    """Test the accuracy of the model."""
    correct_classifications = [0, 0, 0, 0]
    incorrect_classifications = [0, 0, 0, 0]
    total_classifications = [0, 0, 0, 0]

    predictions = []
    for i, (in_window, out) in enumerate(
        zip(model_input, expected_output),
    ):
        start = time.perf_counter()

        output_data = model.infer(in_window)
        predictions.append(output_data)

        dt = time.perf_counter() - start

        end_color = "\033[0m"
        wrong_col = "\033[1;31m"
        right_col = "\033[1;32m"

        target_col = "\033[1;34m"

        output_index = np.argmax(output_data)
        target_index = np.argmax(out.output)
        output_colors = ["", "", "", ""]
        target_colors = ["", "", "", ""]

        if output_index == target_index:
            correct_classifications[output_index] += 1
            output_colors[output_index] = right_col
        else:
            incorrect_classifications[output_index] += 1
            output_colors[output_index] = wrong_col
            target_colors[target_index] = target_col

        total_classifications[target_index] += 1

        correct_count = sum(correct_classifications)
        print(
            f"{i} - {correct_count / (i + 1) * 100:.2f}% {dt * 1000:.3f}ms out:[",
            end="",
        )
        for out_col, target_col, data in zip(output_colors, target_colors, output_data):
            print(f"{out_col}{target_col}{data * 100:.2f}%{end_color}, ", end="")
        print("\b\b]")

    print()
    print(
        f"rest: {correct_classifications[0]}({incorrect_classifications[0]})/{total_classifications[0]}, grip: {correct_classifications[1]}({incorrect_classifications[1]})/{total_classifications[1]}, hold: {correct_classifications[2]}({incorrect_classifications[2]})/{total_classifications[2]}, release: {correct_classifications[3]}({incorrect_classifications[3]})/{total_classifications[3]}",
    )

    a = [np.argmax(v.output) for v in expected_output]
    b = [np.argmax(v) for v in predictions]

    cm = confusion_matrix(a, b)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["rest", "grip", "hold", "release"],
    )
    disp.plot()
    plt.title("Confusion matrix")
    plt.show()
