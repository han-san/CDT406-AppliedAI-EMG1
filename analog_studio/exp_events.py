from exp_LED import turnon_LED, turnoff_LED
from exp_audio import turnon_audio, turnoff_audio

class ExperimentEvent:
	def execute(self):
		raise NotImplementedError("Empty event has no handler")

class ExperimentEventLEDOn(ExperimentEvent):
	def __init__(self, pin: int, hdwf):
		self._pin = pin
		self._hdwf = hdwf

	def execute(self):
		turnon_LED(self._hdwf, self._pin)
		# print("%s ON" % self._pin)

class ExperimentEventLEDOff(ExperimentEvent):
	def __init__(self, pin: int, hdwf):
		self._pin = pin
		self._hdwf = hdwf

	def execute(self):
		turnoff_LED(self._hdwf, self._pin)
		# print("%s OFF" % self._pin)

class ExperimentEventAudioOn(ExperimentEvent):
	def __init__(self, channel, hdwf):
		self._channel = channel
		self._hdwf = hdwf

	def execute(self):
		turnon_audio(self._hdwf, self._channel)

class ExperimentEventAudioOff(ExperimentEvent):
	def __init__(self, channel, hdwf):
		self._channel = channel
		self._hdwf = hdwf

	def execute(self):
		turnoff_audio(self._hdwf, self._channel)