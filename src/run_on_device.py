import sys
from pathlib import Path
import struct
import array
import time

from data import Window, Input
import numpy as np

import tflite_runtime.interpreter as tflite

from tflite_model import TFLiteModel

v_ref = 1.8
power = (1 << 12) - 1

def value_to_voltage(val):
    return val * v_ref / power

def exit_gracefully():
    print(sum(times)/len(times))
    adc_in.close()
    inference_out.close()
    exit(0)

model_path = Path(sys.argv[1])

model = TFLiteModel(tflite.Interpreter, model_path)
print('TFLite model is loaded')

buffer_size = 1500
number_of_states = 4

struct_fmt = f'<{buffer_size * 2}B'
in_pipe = '/home/EMG1/setup/adc_pipe'
out_pipe = '/home/EMG1/setup/inference_pipe'

adc_in = open(in_pipe, "rb")
inference_out = open(out_pipe, "wb")
print('Starting')
times = []

try:
    while True:
        now = time.perf_counter()
        received = adc_in.read(struct.calcsize(struct_fmt))
        if len(received) > 0:
            data = array.array('H')
            data.frombytes(received)
            del data[2::3]
            voltages = list(map(value_to_voltage, data))
            window = Window(np.array(voltages, dtype=np.float32))
            res = model.infer(Input(window))
            now = time.perf_counter() - now
            packed = struct.pack(f'<{number_of_states}f', res[0], res[1], res[2], res[3])
            try:
                inference_out.write(packed)
                inference_out.flush()
            except IOError:
                print(sum(times)/len(times))
                adc_in.close()
                exit(0)

            times.append(now)
        else:
            break
except KeyboardInterrupt:
    exit_gracefully()

exit_gracefully()
