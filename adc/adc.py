import iio
import time
import array

dev_name = 'TI-am335x-adc.0.auto'
channel = 0
buffer_size = 1500
num_samples = 3_000 * 5 * 60
readings = bytearray()

contexts = iio.scan_contexts()

uri, _ = contexts.popitem()
ctx = iio.Context(_context = uri)

dev = ctx.find_device('TI-am335x-adc.0.auto')
dev.channels[channel].enabled = True
buffer = iio.Buffer(dev, buffer_size)

times = []

while True:
	buffer.refill()
	samples = buffer.read()

	readings.extend(samples)

	num_samples -= min(num_samples, len(samples))
	if num_samples == 0:
		break

dev.channels[channel].enabled = False

readings_s = array.array('H')
readings_s.frombytes(readings)

# f = open("adc_record.csv", "w")
# start = 0
# for i in readings_s:
# 	f.write(f"{start}, {i}\n")
# 	start += 0.00013
# f.close()