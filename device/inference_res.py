import struct

in_pipe = '/home/EMG1/setup/inference_pipe'

number_of_states = 4

struct_fmt = f'<{number_of_states}f'

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
try:
	while True:
		received = f.read(struct.calcsize(struct_fmt))
		if len(received) > 0:
			rest, grip, hold, release = struct.unpack(struct_fmt, received)
			print("\033[2J\r"+max4(("REST", rest), ("GRIP", grip), ("HOLD", hold), ("RELEASE", release)))
		else:
			break
except KeyboardInterrupt:
	f.close()
f.close()