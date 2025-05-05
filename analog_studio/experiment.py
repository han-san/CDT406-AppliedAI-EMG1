from exp_timeline import ExperimentEventTimeline, ExperimentStateTimeline
from exp_instructions import ExperimentInstructionsContext
from exp_states import ExperimentStates


class Experiment:
    def __init__(
        self, frequency: int, red_pin: int, yellow_pin: int, green_pin: int, hdwf
    ):
        self._frequency = frequency
        self._red_pin = red_pin
        self._yellow_pin = yellow_pin
        self._green_pin = green_pin
        self._event_timeline = ExperimentEventTimeline()
        self._expected_state_timeline = ExperimentStateTimeline()
        self._actual_state_timeline = ExperimentStateTimeline()
        self._experiment_time = 0
        self._hdwf = hdwf
        self._init_state = -1

    def read_instructions(self, instructions):
        ctx = ExperimentInstructionsContext(
            self._frequency,
            self._event_timeline,
            self._expected_state_timeline,
            self._hdwf,
            self._red_pin,
            self._yellow_pin,
            self._green_pin,
        )
        for inst in instructions:
            inst.execute(ctx)
        self._experiment_time = ctx.time
        self._init_state = self._expected_state_timeline.peek_current_state(0)

    def state(self, old_state: ExperimentStates, sample_num: int):
        return self._expected_state_timeline.get_current_state(sample_num, old_state)

    def event(self, sample_num: int):
        self._event_timeline.do_event(sample_num)

    def experiment_duration(self):
        return self._experiment_time

    def actual_experiment_state(self, state: ExperimentStates, sample_num: int):
        self._actual_state_timeline.add_to_timeline(sample_num, state)

    def act_state(self, old_state: ExperimentStates, sample_num: int):
        return self._actual_state_timeline.get_current_state(sample_num, old_state)

    def init_state(self):
        return self._init_state
