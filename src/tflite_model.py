from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from data import Input


class TFLiteModel:
    """A TFLite model that can be used to classify input."""

    def __init__(
        self,
        interpreter: Any,  # Either tf.lite.Interpreter or tflite.Interpreter.
        model_path: Path,
    ) -> None:
        """Load a TFLite model from disk using the provided interpreter."""
        self.model = interpreter(model_path=str(model_path))
        self.model.allocate_tensors()
        input_details = self.model.get_input_details()[0]
        output_details = self.model.get_output_details()[0]
        self.input_shape = input_details["shape"]
        self.input_tensor_index = input_details["index"]
        self.output_tensor_index = output_details["index"]

    def infer(self, inference_input: Input) -> npt.NDArray[np.float32]:
        """Get the classification output using window as input."""
        input_data = inference_input.input.reshape(self.input_shape)
        self.model.set_tensor(self.input_tensor_index, input_data)

        self.model.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        return self.model.get_tensor(self.output_tensor_index)[0]

    model: Any
    input_shape: Any
    input_tensor_index: Any
    output_tensor_index: Any
