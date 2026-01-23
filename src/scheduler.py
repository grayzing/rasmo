from enum import Enum
from collections import deque
from time import sleep

from nr import NrUe, UeRrcState, NrGnb, AdvancedSleepMode
from embb_traffic_generator import EmbbTrafficGenerator

time_difference_tolerance: float = 1e-4

class ActionType(Enum):
    NrHandover = "NrHandover"
    AdvancedSleepModeAdjust = "AdvancedSleepModeAdjust"

class EventHookType(Enum):
    RrcStateChange = 0
    AsmChange = 1

class Event:
    def __init__(self, execution_time: float, action: ActionType, parent: 'Scheduler', args: dict | None = None) -> None:
        self.execution_time = execution_time
        self.args: dict | None = args
        self.action: ActionType = action
        self.parent = parent

    def do(self) -> None:
        """
        Perform whatever action is specified, using args as arguments.
        
        """
        if self.action == ActionType.NrHandover:
            assert self.args

            assert self.args["destination_gnb"]
            assert self.args["source_gnb"]
            assert self.args["ue"]
            destination_gnb: NrGnb = self.args["destination_gnb"]
            source_gnb: NrGnb = self.args["source_gnb"]
            ue: NrUe = self.args["ue"]

            source_gnb.request_handover(ue,destination_gnb)



        elif self.action == ActionType.AdvancedSleepModeAdjust:
            assert self.args

            assert self.args["advanced_sleep_mode"]
            assert self.args["target_gnb"]

            advanced_sleep_mode: AdvancedSleepMode = self.args["advanced_sleep_mode"]
            target_gnb: NrGnb = self.args["target_gnb"]

            target_gnb.radio_unit.set_advanced_sleep_mode(advanced_sleep_mode)

            for event in self.parent.event_hooks[EventHookType.AsmChange]:
                event.do

        print("Executed event with ActionType ", self.action, " at time ", self.execution_time)


class Scheduler:
    def __init__(self, delta: float) -> None:
        self.queue: deque[Event] = deque()
        self.delta: float = delta
        self.object_space: ObjectSpace = ObjectSpace()

        self.event_hooks: dict[EventHookType, list[Event]] = {event_hook_type : [] for event_hook_type in EventHookType}
        self.traffic_generator: EmbbTrafficGenerator | None = None

        self.time = 0

    def push_event(self, time_to_execute: float, action: ActionType, args: dict | None = None) -> None:
        self.queue.appendleft(Event(self.time + time_to_execute, action, self, args))

    def add_traffic_generator(self, traffic_generator: EmbbTrafficGenerator):
        self.traffic_generator = traffic_generator

    def run(self, stop_time: float, verbose: bool = False) -> None:
        assert stop_time > 0
        time: float = self.time
        while time < stop_time:
            if verbose:
                print("Time step: ", time)
            candidate_event: Event
            if self.queue:
                candidate_event = self.queue[-1]
                if abs(candidate_event.execution_time-time) <= time_difference_tolerance:
                    candidate_event = self.queue.pop()
                    candidate_event.do()

            time += self.delta
            self.time = time
            sleep(0.5)

class ObjectSpace:
    def __init__(self) -> None:
        self.children: set = set()

    def add_child(self, child):
        if not child in self.children:
            self.children.add(child)

    

