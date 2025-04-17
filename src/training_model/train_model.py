import numpy as np
import numpy.typing as npt
import tensorflow as tf

# NOTE: We assume we are running this program from the same directory as this python file.
modelPath = "../../model/model.tflite"

# The amount of measurement readings we use as input.
timestepWindowSize = 200
# The amount of measurements included in each reading.
channelCount = 1

# TODO: Get path from user.
dataFilePath = ""


def createModel(timesteps: int, samples: int):
    # Create the same type of model as in https://doi.org/10.3390/app12199700.
    # The 'unroll=True' has to be set in the LSTM layer to be able to run the model using the tflite-runtime.
    # Otherwise the model contains the OP "FlexTensorListReserve" which is only available in
    # the regular tensorflow package.
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=(timesteps, samples)),
            tf.keras.layers.Dense(units=32, activation=tf.keras.activations.tanh),
            tf.keras.layers.LSTM(units=16, unroll=True),
            tf.keras.layers.Dense(units=32, activation=tf.keras.activations.tanh),
            tf.keras.layers.Dense(units=3, activation=tf.keras.activations.softmax),
        ],
    )

    return model


# TODO: Actually load a data file.
def loadDataFile(filePath: str) -> npt.NDArray[np.float32]:
    return np.array([1.0 for n in range(timestepWindowSize)])


def loadDataFiles(filePaths: list[str]):
    result = []
    for path in filePaths:
        result.append(loadDataFile(path))
    return np.array(result)


def createWindows(data: npt.NDArray[np.float32], windowSize: int):
    # TODO: Decide what to do with the last window if it is < windowSize.
    #   - ignore?
    #   - use average of all measurements (think one of the groups from previous years did this)?
    # TODO: Implement
    return np.array([data])


def createInputFromWindows(windows):
    # The LSTM layer expects 3D input, where the dimensions are (samples, time steps, features).
    # https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
    return windows.reshape((len(windows), timestepWindowSize, channelCount))


model = createModel(timestepWindowSize, channelCount)

data = loadDataFile(dataFilePath)
dataWindows = createWindows(data, timestepWindowSize)

input = createInputFromWindows(dataWindows)

# Here we need to train!

# "Configures the model for training"
# TODO: Might want to change the arguments.
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
)

# TODO: Actually generate target labels properly.
# TODO: Figure out what we should set the batch size to.
print("Starting training!")
history = model.fit(input, np.array([np.array([1.0, 0.0, 0.0])]), epochs=3)
print("Finished training!")
print(history.history)

# Get a result from the model.
predictions = model.predict(input)
print(predictions)

print(model.summary())
print(f"input shape: {model.input_shape}")

# Convert the model to TFLite/LiteRT format.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open(modelPath, "wb") as f:
    f.write(tflite_model)

# ============== Everything below here is just for testing ==============

# Load the model using TFLite/LiteRT.
interpreter = tf.lite.Interpreter(model_path=modelPath)
interpreter.allocate_tensors()

inputDetails = interpreter.get_input_details()
outputDetails = interpreter.get_output_details()

inputShape = inputDetails[0]["shape"]
print(inputShape)
inputData = np.array(np.random.random_sample(inputShape), dtype=np.float32)
print(inputData)
interpreter.set_tensor(inputDetails[0]["index"], inputData)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
outputData = interpreter.get_tensor(outputDetails[0]["index"])

# Actual results
print(outputData)
