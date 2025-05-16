import iio
import signal

dev_name = 'TI-am335x-adc.0.auto'
channel = 0
buffer_size = 1500
pipe = '/home/EMG1/setup/adc_pipe'

contexts = iio.scan_contexts()

uri, _ = contexts.popitem()
ctx = iio.Context(_context = uri)

dev = ctx.find_device(dev_name)
dev.channels[channel].enabled = True
buffer = iio.Buffer(dev, buffer_size)

f = open(pipe, "wb")

def exit_gracefully():
	f.close()
	dev.channels[channel].enabled = False
	exit(0)

def sig_handler(signal, stack):
	print("Handling SIGTERM")
	exit_gracefully()

def read_adc(f, buffer):
	while True:
		num_samples = buffer_size * 2
		readings = bytearray()
		while num_samples > 0:
			buffer.refill()
			samples = buffer.read()
			readings.extend(samples)
			num_samples -= min(num_samples, len(samples))
		try:
			f.write(readings)
		except IOError:
			dev.channels[channel].enabled = False
			exit(0)

signal.signal(signal.SIGTERM, sig_handler)

read_adc(f, buffer)


# f = open("adc_record.csv", "w")
# start = 0
# for i in readings_s:
# 	f.write(f"{start}, {i}\n")
# 	start += 0.00013
# f.close()