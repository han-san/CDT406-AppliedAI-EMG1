import exp_timeline as timeline
import exp_events as events
import exp_states as states

class ExperimentInstructionsContext:
	def __init__(self, frequency: int, event_timeline: timeline.ExperimentEventTimeline(), state_timeline: timeline.ExperimentStateTimeline(), hdwf):
		self.time = int(0)
		self.event_timeline = event_timeline
		self.state_timeline = state_timeline
		self.frequency = frequency
		self.hdwf = hdwf

class ExperimentInstructions:
	def __init__(self, arg):
		self._arg = arg

	def execute(self, context: ExperimentInstructionsContext):
		raise NotImplementedError("No action for abstract instruction")

class ExperimentInstructionsWait(ExperimentInstructions):
	def execute(self, context: ExperimentInstructionsContext):
		context.time += int(self._arg * context.frequency)

class ExperimentInstructionsLEDOn(ExperimentInstructions):
	def execute(self, context: ExperimentInstructionsContext):
		e_tl = context.event_timeline
		e_tl.add_to_timeline(context.time, events.ExperimentEventLEDOn(int(self._arg), context.hdwf))

class ExperimentInstructionsLEDOff(ExperimentInstructions):
	def execute(self, context: ExperimentInstructionsContext):
		e_tl = context.event_timeline
		e_tl.add_to_timeline(context.time, events.ExperimentEventLEDOff(int(self._arg), context.hdwf))

class ExperimentInstructionsChangeState(ExperimentInstructions):
	def execute(self, context: ExperimentInstructionsContext):
		s_tl = context.state_timeline
		s_tl.add_to_timeline(context.time, states.ExperimentStates(int(self._arg)))