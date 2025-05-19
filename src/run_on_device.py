import sys
from pathlib import Path
import struct
import array
import time

from data import Window, Input
from preprocessing.FilterFunction import FilterType, NormalizationType
from preprocessing.Moving_average_filter import MovingAverageType
import numpy as np

import tflite_runtime.interpreter as tflite

from tflite_model import TFLiteModel

v_ref = 1.8
power = (1 << 12) - 1

def value_to_voltage(val):
    return val * v_ref / power

times = []
inference_times = []

def exit_gracefully():
    log = open('/home/EMG1/setup/model.log', 'w')
    log.write('Waiting\n')
    log.write(str(times))
    log.write('\nInference\n')
    log.write(str(inference_times))
    log.close()
    print(f"Average: {sum(times)/len(times)}, {sum(inference_times)/len(inference_times)}")
    adc_in.close()
    inference_out.close()
    exit(0)

model_path = Path(sys.argv[1])

filter_t = FilterType.RANGE_20_TO_500_BUTTER
normalization_t = NormalizationType.Z_SCORE
moving_t = MovingAverageType.EMA

model = TFLiteModel(tflite.Interpreter, model_path)
print('TFLite model is loaded')

buffer_size = 1500
number_of_states = 4

in_struct_fmt = f'<{buffer_size * 2}B'
out_struct_fmt = f'<{buffer_size * 2}B{number_of_states}f'
in_pipe = '/home/EMG1/setup/adc_pipe'
out_pipe = '/home/EMG1/setup/inference_pipe'

adc_in = open(in_pipe, "rb")
inference_out = open(out_pipe, "wb")
print('Starting')

try:
    while True:
        now = time.perf_counter()
        received = adc_in.read(struct.calcsize(in_struct_fmt))
        if len(received) > 0:
            start = time.perf_counter()
            data = array.array('H')
            data.frombytes(received)
            del data[2::3]
            voltages = list(map(value_to_voltage, data))
            window = Window(np.array(voltages, dtype=np.float32), filter_t, normalization_t, moving_t)
            res = model.infer(Input(window))
            end = time.perf_counter()
            now = end - now
            start = end - start
            packed = struct.pack(out_struct_fmt, *received, *res)
            try:
                inference_out.write(packed)
                inference_out.flush()
            except IOError:
                log = open('/home/EMG1/setup/model.log', 'w')
                log.write('Waiting\n')
                log.write(str(times))
                log.write('\nInference\n')
                log.write(str(inference_times))
                log.close()
                print(f"Average: {sum(times)/len(times)}, {sum(inference_times)/len(inference_times)}")
                adc_in.close()
                exit(0)

            times.append(now)
            inference_times.append(start)
        else:
            break
except KeyboardInterrupt:
    exit_gracefully()

exit_gracefully()
