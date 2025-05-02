from exp_LED import turnon_LED, turnoff_LED

class ExperimentEvent:
	def execute(self):
		pass

class ExperimentEventLEDOn(ExperimentEvent):
	def __init__(self, pin: int, hdwf):
		self._pin = pin
		self._hdwf = hdwf

	def execute(self):
		turnon_LED(self._hdwf, self._pin)
		print("%s ON" % self._pin)

class ExperimentEventLEDOff(ExperimentEvent):
	def __init__(self, pin: int, hdwf):
		self._pin = pin
		self._hdwf = hdwf

	def execute(self):
		turnoff_LED(self._hdwf, self._pin)
		print("%s OFF" % self._pin)