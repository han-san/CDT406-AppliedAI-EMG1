from collections import deque
import experiment.exp_events as events
import experiment.exp_states as states

class ExperimentTimeline:
	def __init__(self):
		self._timeline = deque()

	def add_to_timeline(self, timestamp: int, elem):
		self._timeline.append((timestamp, elem))

	def is_time(self, timestamp: int):
		if bool(self._timeline) and self._timeline[0][0] <= timestamp:
			return True
		else:
			return False

class ExperimentEventTimeline(ExperimentTimeline):
	def add_to_timeline(self, timestamp: int, elem: events.ExperimentEvent):
		super().add_to_timeline(timestamp, elem)

	def do_event(self, timestamp: int):
		while self.is_time(timestamp):
			(ts, event) = self._timeline.popleft()
			# print("%s:" % timestamp)
			event.execute()

class ExperimentStateTimeline(ExperimentTimeline):
	def add_to_timeline(self, timestamp: int, elem: states.ExperimentStates):
		super().add_to_timeline(timestamp, elem)

	def get_current_state(self, timestamp: int, old_state: states.ExperimentStates):
		if self.is_time(timestamp):
			(ts, state) = self._timeline.popleft()
			return state
		else:
			return old_state

	def peek_current_state(self, timestamp: int):
		if self.is_time(timestamp):
			(ts, state) = self._timeline[0]
			return state
		return -1
