import time
from pathlib import Path
from typing import Any

import numpy as np

from data import Input, Output


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

    correct = 0

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
            correct += 1

        total = i + 1

        print(
            f"\rdt: {dt}s, iteration: {total}, accuracy: {correct / total * 100}%     ",
            end="",
        )
