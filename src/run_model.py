import time
from pathlib import Path
from typing import Any

import numpy as np

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

    correct_grip = 0
    correct_hold = 0
    correct_rest = 0
    correct_release = 0
    incorrect_grip = 0
    incorrect_hold = 0
    incorrect_rest = 0
    incorrect_release = 0

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

        dt = time.perf_counter() - start

        if np.argmax(output_data[0]) == np.argmax(out):
            if np.argmax(out) == np.argmax(State.GRIP.value):
                correct_grip += 1
            if np.argmax(out) == np.argmax(State.RELEASE.value):
                correct_release += 1
            if np.argmax(out) == np.argmax(State.REST.value):
                correct_rest += 1
            if np.argmax(out) == np.argmax(State.HOLD.value):
                correct_hold += 1
        else:
            if np.argmax(out) == np.argmax(State.GRIP.value):
                incorrect_grip += 1
            if np.argmax(out) == np.argmax(State.RELEASE.value):
                incorrect_release += 1
            if np.argmax(out) == np.argmax(State.REST.value):
                incorrect_rest += 1
            if np.argmax(out) == np.argmax(State.HOLD.value):
                incorrect_hold += 1

        total = i + 1

        correct = correct_grip + correct_hold + correct_release + correct_rest
        total_rest = correct_rest + incorrect_rest
        total_release = correct_release + incorrect_release
        total_grip = correct_grip + incorrect_grip
        total_hold = correct_hold + incorrect_hold
        print(
            f"\rdt: {dt}s, iteration: {total}, accuracy: Total: {correct / total * 100}%, Rest: {correct_rest / (total_rest if total_rest else 1) * 100}, Grip: {correct_grip / (total_grip if total_grip else 1) * 100}, Hold: {correct_hold / (total_hold if total_hold else 1) * 100}, Release: {correct_release / (total_release if total_release else 1) * 100}     ",
            end="",
        )

    print()
    print(
        f"grip: {correct_grip}/{total_grip}, rest: {correct_rest}/{total_rest}, release: {correct_release}/{total_release}, hold: {correct_hold}/{total_hold}"
    )
