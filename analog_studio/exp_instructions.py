from enum import Enum

import exp_timeline as timeline
import exp_events as events
import exp_states as states


class ExperimentLEDColor(Enum):
    RED = 0
    YELLOW = 1
    GREEN = 2


class ExperimentInstructionsContext:
    def __init__(
        self,
        frequency: int,
        event_timeline: timeline.ExperimentEventTimeline(),
        state_timeline: timeline.ExperimentStateTimeline(),
        hdwf,
        red: int,
        yellow: int,
        green: int,
    ):
        self.time = int(0)
        self.event_timeline = event_timeline
        self.state_timeline = state_timeline
        self.frequency = frequency
        self.hdwf = hdwf
        self.red = red
        self.yellow = yellow
        self.green = green


def pin_by_color(color: ExperimentLEDColor, ctx: ExperimentInstructionsContext):
    match color:
        case ExperimentLEDColor.RED:
            return ctx.red
        case ExperimentLEDColor.YELLOW:
            return ctx.yellow
        case ExperimentLEDColor.GREEN:
            return ctx.green


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
        pin = pin_by_color(self._arg, context)
        e_tl.add_to_timeline(
            context.time, events.ExperimentEventLEDOn(pin, context.hdwf)
        )


class ExperimentInstructionsLEDOff(ExperimentInstructions):
    def execute(self, context: ExperimentInstructionsContext):
        e_tl = context.event_timeline
        pin = pin_by_color(self._arg, context)
        e_tl.add_to_timeline(
            context.time, events.ExperimentEventLEDOff(pin, context.hdwf)
        )


class ExperimentInstructionsChangeState(ExperimentInstructions):
    def execute(self, context: ExperimentInstructionsContext):
        s_tl = context.state_timeline
        s_tl.add_to_timeline(context.time, states.ExperimentStates(int(self._arg)))


class ExperimentInstructionsAudioOn(ExperimentInstructions):
    def execute(self, context: ExperimentInstructionsContext):
        e_tl = context.event_timeline
        e_tl.add_to_timeline(
            context.time,
            events.ExperimentEventAudioOn(int(self._arg), context.hdwf),
        )


class ExperimentInstructionsAudioOff(ExperimentInstructions):
    def execute(self, context: ExperimentInstructionsContext):
        e_tl = context.event_timeline
        e_tl.add_to_timeline(
            context.time,
            events.ExperimentEventAudioOff(int(self._arg), context.hdwf),
        )
