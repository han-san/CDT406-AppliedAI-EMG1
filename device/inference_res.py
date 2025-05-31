import struct
from enum import Enum
import array

class State(Enum):
	REST = 0
	GRIP = 1
	HOLD = 2
	RELEASE = 3

	def __str__(self):
		return self.name

v_ref = 3.3
power = (1 << 12) - 1

def value_to_voltage(val):
    return val * v_ref / power


in_pipe = '/home/EMG1/setup/inference_pipe'

number_of_states = 4
buffer_size = 1500
start = 0
step = 1 / (15_000 / 2)

struct_fmt = f'<{buffer_size * 2}B{number_of_states}f'

def max4(rest, grip, hold, release):
	m = max(rest[1], grip[1], hold[1], release[1])
	if m == rest[1]:
		return rest[0]
	if m == grip[1]:
		return grip[0]
	if m == hold[1]:
		return hold[0]
	if m == release[1]:
		return release[0]

f = open(in_pipe, "rb")
log = open("states.log", "w")
try:
	while True:
		received = f.read(struct.calcsize(struct_fmt))
		if len(received) > 0:
			data = struct.unpack(struct_fmt, received)
			samples = bytearray(data[0:-number_of_states])
			rest, grip, hold, release = data[-number_of_states:]
			res = max4((State.REST, rest), (State.GRIP, grip), (State.HOLD, hold), (State.RELEASE, release))
			print("\033[2J\r"+str(res))
			ar = array.array('H')
			ar.frombytes(samples)
			for i in ar:
				log.write(f'{start}, {value_to_voltage(i)}, {res.value}\n')
				start += step
		else:
			break
except KeyboardInterrupt:
	f.close()
	log.close()
f.close()
log.close()