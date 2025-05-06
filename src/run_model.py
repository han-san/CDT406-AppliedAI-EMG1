import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from data import Input, Output, State


def load_and_run_tflite(
    interpreter: Any,  # Either tf.lite.Interpreter or tflite.interpreter.
    model_path: Path,
    model_input: Input,
    expected_output: Output,
) -> None:
    """Test the accuracy of the model."""
    # Load the model using TFLite/LiteRT.
    interpreter = interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]

    correct_classifications = [0, 0, 0, 0]
    incorrect_classifications = [0, 0, 0, 0]
    total_classifications = [0, 0, 0, 0]

    predictions = []
    for i, (in_window, out) in enumerate(
        zip(model_input.input, expected_output.output),
    ):
        start = time.perf_counter()
        input_data = in_window
        input_data = input_data.reshape(input_shape)
        interpreter.set_tensor(input_details[0]["index"], input_data)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]["index"])
        predictions.append(output_data[0])

        dt = time.perf_counter() - start

        end_color = "\033[0m"
        wrong_col = "\033[1;31m"
        right_col = "\033[1;32m"

        target_col = "\033[1;34m"

        output_index = np.argmax(output_data[0])
        target_index = np.argmax(out)
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
        for out_col, target_col, data in zip(
            output_colors, target_colors, output_data[0]
        ):
            print(f"{out_col}{target_col}{data * 100:.2f}%{end_color}, ", end="")
        print("\b\b]")

    print()
    print(
        f"rest: {correct_classifications[0]}({incorrect_classifications[0]})/{total_classifications[0]}, grip: {correct_classifications[1]}({incorrect_classifications[1]})/{total_classifications[1]}, hold: {correct_classifications[2]}({incorrect_classifications[2]})/{total_classifications[2]}, release: {correct_classifications[3]}({incorrect_classifications[3]})/{total_classifications[3]}"
    )

    a = [np.argmax(v) for v in expected_output.output]
    b = [np.argmax(v) for v in predictions]

    cm = confusion_matrix(a, b)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["rest", "grip", "hold", "release"],
    )
    disp.plot()
    plt.title("Confusion matrix")
    plt.show()
