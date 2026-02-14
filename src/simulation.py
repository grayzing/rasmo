from enum import Enum
from collections import deque
from time import sleep
from math import fmod
import tkinter, turtle
import numpy as np
import torch

from nr import NrUe, NrGnb, AdvancedSleepMode, UeMobilityModel, AsmTransitionState
from utils import Vector, euclidean_distance, rsrp

from graphics import UE_IMAGE, RU_IMAGE, RU_OFF_IMAGE, GRAPHICAL_SCALING_FACTOR

from warnings import warn

time_difference_tolerance: float = 1

class ActionType(Enum):
    NrHandover = "NrHandover"
    NrHandoverRequestReceive = "NrHandoverRequestReceive"
    NrHandoverRequestResponseReceive = "NrHandoverRequestResponseReceive"
    NrUeKpiReportReceive = "NrUeKpiReportReceive"
    FtpPacketReceive = "FtpPacketReceive"
    AdvancedSleepModeAdjust = "AdvancedSleepModeAdjust"
    AdvancedSleepModeTransitionEnd = "AdvancedSleepModeTransitionEnd"

class EventHookType(Enum):
    RrcStateChange = 0
    AsmChange = 1

class GnbSleepModeStrategy(Enum):
    Control = 0
    Naive = 1
    RASM = 2

class Event:
    def __init__(self, execution_time: float, action: ActionType, parent: 'Simulation', args: dict | None = None) -> None:
        self.execution_time = execution_time
        self.args: dict | None = args
        self.action: ActionType = action
        self.parent = parent

    def do(self) -> None:
        """
        Perform whatever action is specified, using args as arguments.
        
        """
        if self.action == ActionType.FtpPacketReceive:
            assert self.args

            assert self.args["destination_ue"]
            assert self.args["source_gnb"]
            assert self.args["packet"]    

            #packet: Packet = self.args["packet"]
            source_gnb: NrGnb = self.args["source_gnb"]
            destination_ue: NrUe = self.args["destination_ue"]

            print("Packet received from gnb ", source_gnb.cell_id, " by UE ", destination_ue.id)

        elif self.action == ActionType.AdvancedSleepModeAdjust:
            assert self.args

            assert self.args["advanced_sleep_mode"]
            assert self.args["target_gnb"]

            advanced_sleep_mode: AdvancedSleepMode = self.args["advanced_sleep_mode"]
            target_gnb: NrGnb = self.args["target_gnb"]

            if advanced_sleep_mode != AdvancedSleepMode.ACTIVE:
                target_gnb.turtle.shape(RU_OFF_IMAGE)
            else:
                target_gnb.turtle.shape(RU_IMAGE)

            target_gnb.radio_unit.set_advanced_sleep_mode(advanced_sleep_mode)
            target_gnb.radio_unit.asm_transition_state = AsmTransitionState.DEBO
            self.parent.schedule_event(advanced_sleep_mode.value[1], ActionType.AdvancedSleepModeTransitionEnd, args={
                "target_gnb": target_gnb
            })
        elif self.action == ActionType.AdvancedSleepModeTransitionEnd:
            assert self.args

            assert self.args["target_gnb"]
            target_gnb: NrGnb = self.args["target_gnb"]

            target_gnb.radio_unit.asm_transition_state = AsmTransitionState.NONE
            
            target_gnb.allocate()
        
        print("Executed event with ActionType ", self.action, " at time ", self.execution_time)


class Simulation:
    def __init__(self, delta: float = 0.125, energy_saving_strategy: GnbSleepModeStrategy = GnbSleepModeStrategy.Control) -> None:
        self.queue: deque[Event] = deque()
        self.delta: float = delta
        self.gnbs: dict[int, NrGnb] = {}
        self.ues: dict[int, NrUe] = {}
        self.items = []

        self.event_hooks: dict[EventHookType, list[Event]] = {event_hook_type : [] for event_hook_type in EventHookType}
        #self.traffic_generator: EmbbTrafficGenerator = EmbbTrafficGenerator(50_000)

        self.time = 0

        self.energy_saving_strategy: GnbSleepModeStrategy = energy_saving_strategy

        self.screen: turtle._Screen = turtle.Screen()
        self.screen.setup(width=1500,height=1500)
        self.screen.title("graphdqn testbed")
        self.screen.tracer(0)

        self.screen.register_shape(UE_IMAGE)
        self.screen.register_shape(RU_IMAGE)
        self.screen.register_shape(RU_OFF_IMAGE)

        self.ue_connection_turtle = turtle.Turtle()
        self.ue_connection_turtle.speed(0)
        self.ue_connection_turtle.penup()
        self.ue_connection_turtle.pencolor("blue")
        self.ue_connection_turtle.hideturtle()

        self.simulation_statistics_turtle = turtle.Turtle()
        self.simulation_statistics_turtle.speed(0)
        self.simulation_statistics_turtle.penup()
        self.simulation_statistics_turtle.hideturtle()

    def update_component_connection_display(self):
        for ue in self.ues.values() :
            if ue.serving_gnb:
                self.ue_connection_turtle.goto(ue.get_position().x/GRAPHICAL_SCALING_FACTOR, ue.get_position().y/GRAPHICAL_SCALING_FACTOR)
                self.ue_connection_turtle.pendown()
                self.ue_connection_turtle.goto(ue.serving_gnb.get_position().x/GRAPHICAL_SCALING_FACTOR, ue.serving_gnb.get_position().y/GRAPHICAL_SCALING_FACTOR)
                self.ue_connection_turtle.penup()

    def get_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate state for GraphDQN model.

        Takes geolocations of every node in simulation, RSRP between every connected node (just UEs and gNBs), current ASM
        
        :return: Edge tensor, edge weights, vertex embeddings
        :rtype: tuple[Tensor, Tensor, Tensor]
        """
        n = len(self.ues)
        m = len(self.gnbs)
        source: list [int] = []
        destination: list [int] = []
        w: list[float] = []
        v: list[list[float]] = [[0,0,0,0,0,0] for _ in range(m + n)]

        for gnb in self.gnbs.values():
            prb_utilization = sum(prb for prb in gnb.allocate().values()) / gnb.ue_scheduler.total_prbs
            average_throughput = 1
            if len(gnb.connected_ues) > 0:
                average_throughput = sum(ue.average_throughput for ue in gnb.connected_ues) / len(gnb.connected_ues)
            v[gnb.cell_id] = [gnb.position.x, gnb.position.y, gnb.position.z, gnb.radio_unit.advanced_sleep_mode.value[2], prb_utilization, average_throughput]
            for ue in gnb.connected_ues:
                source.append(gnb.cell_id)
                destination.append(ue.id)
                w.append(rsrp(gnb.position, ue.position, gnb.tx_freq, gnb.tx_power)/-160)

        for ue in self.ues.values():
            v[m + ue.id] = [ue.position.x, ue.position.y, ue.position.z, 1, 1, 1]

        feature_vector: torch.Tensor = torch.tensor(v, dtype=torch.float32)
        edge_vector: torch.Tensor = torch.tensor([source, destination], dtype=torch.int)
        weight_vector: torch.Tensor = torch.tensor(w, dtype=torch.float32)

        return feature_vector, edge_vector.to(torch.int64), weight_vector

    def schedule_event(self, time_to_execute: float, action: ActionType, args: dict | None = None) -> None:
        self.queue.append(Event(self.time + time_to_execute, action, self, args))

    def set_advanced_sleep_mode(self, target_gnb: NrGnb, advanced_sleep_mode: AdvancedSleepMode):
        if target_gnb.radio_unit.asm_transition_state != AsmTransitionState.NONE:
            warn("Tried to change ASM of gNB while it was in ASM transition")
            return
        tte = advanced_sleep_mode.value[0]
        target_gnb.radio_unit.asm_transition_state = AsmTransitionState.TRAN
        
        self.schedule_event(tte, ActionType.AdvancedSleepModeAdjust, args={
            "advanced_sleep_mode": advanced_sleep_mode,
            "target_gnb": target_gnb
        })

    def initialize_network(self, n: int, m: int):
        """
        Generate n gnbs, m ues
        
        :param n: Number of gNBs
        :type n: int
        :param m: Number of UEs
        :type m: int
        """

        for i in range(n):
            self.gnbs[i] = NrGnb(0,0,25,25,i,self)
            self.gnbs[i].parent_scheduler = self

        for i in range(m):
            self.ues[i] = NrUe(0,0,1.5,0, 1.5, i, self)
            self.ues[i].parent_scheduler = self

            if np.random.randint(0,2) == 0:
                self.ues[i].velocity.x *= -1
            
            if np.random.randint(0,2) == 0:
                self.ues[i].velocity.y *= -1

            self.ues[i].position.x = np.random.randint(-350, 350)
            self.ues[i].position.y = np.random.randint(-350, 350)

        self.gnbs[0].position = Vector(0,0,25)

        for i in range(1,7):
            self.gnbs[i].position = Vector(np.cos(np.deg2rad(60 * i)) * 250, np.sin(np.deg2rad(60 * i)) * 250, 25)

        for i in range(7, 19):
            self.gnbs[i].position = Vector(np.cos(np.deg2rad(30 * i)) * 500, np.sin(np.deg2rad(30 * i)) * 500, 25)

        for i in range(len(self.gnbs)):
            self.gnbs[i].turtle.setposition(self.gnbs[i].position.x/GRAPHICAL_SCALING_FACTOR,self.gnbs[i].position.y/GRAPHICAL_SCALING_FACTOR)


    def get_best_gnb(self, ue: NrUe) -> NrGnb | None:
        """
        Get best gNB for the UE to handover to
        
        :param ue: UE to handover
        :type ue: NrUe
        :return: gNB for the UE to be handed off to
        :rtype: NrGnb | None
        """
        if not self.ues:
            return
        
        best_gnb: NrGnb | None = self.gnbs[0]
        for gnb in self.gnbs.values():
            tentative_distance: float = euclidean_distance(gnb.position, ue.position)
            if self.energy_saving_strategy == GnbSleepModeStrategy.Naive:
                if tentative_distance <= euclidean_distance(best_gnb.position, ue.position):
                    best_gnb = gnb
            else:
                if tentative_distance <= euclidean_distance(best_gnb.position, ue.position) and best_gnb.radio_unit.advanced_sleep_mode == AdvancedSleepMode.ACTIVE and best_gnb.radio_unit.asm_transition_state == AsmTransitionState.NONE:
                    best_gnb = gnb

        if self.energy_saving_strategy != GnbSleepModeStrategy.Naive:
            if euclidean_distance(best_gnb.position, ue.position) > 250 or best_gnb.radio_unit.advanced_sleep_mode != AdvancedSleepMode.ACTIVE:
                return None
        else:
            if euclidean_distance(best_gnb.position, ue.position) > 250:
                return None
        
        return best_gnb
    
    def reward(self) -> float:
        """
        Calculate reward given the state of the simulation
        
        :return: The sum of the average RSRP, average throughput, and sum of all sleep modes in the system
        :rtype: float
        """
        average_rsrp: float = sum(ue.rsrp() for ue in self.ues.values()) / len(self.ues)
        average_average_throughput: float = sum(ue.average_throughput for ue in self.ues.values()) / len(self.ues)
        sleep_mode_sum = sum(gnb.radio_unit.advanced_sleep_mode.value[2] for gnb in self.gnbs.values())
        return average_rsrp + average_average_throughput + sleep_mode_sum
    
    def total_energy_usage(self) -> float:
        return sum(gnb.radio_unit.get_power_consumption() for gnb in self.gnbs.values())

    def step(self):
        candidate_event: Event
        while self.queue:
            if abs(self.queue[-1].execution_time - self.time) <= 2:
                candidate_event = self.queue.pop()
                candidate_event.do()
            else:
                break

        # Handover to gNB with best distance, update instantaneous rate
        for ue in self.ues.values():
            ue.update_instantaneous_rate()
            best_gnb: NrGnb | None = self.get_best_gnb(ue)

            if best_gnb != ue.serving_gnb:
                if ue.serving_gnb:
                    ue.serving_gnb.remove_ue(ue)
                ue.serving_gnb = best_gnb
                if best_gnb:
                    best_gnb.connected_ues.append(ue)
                    if self.energy_saving_strategy == GnbSleepModeStrategy.Naive and best_gnb.radio_unit.advanced_sleep_mode != AdvancedSleepMode.ACTIVE:
                        self.set_advanced_sleep_mode(best_gnb, AdvancedSleepMode.ACTIVE)
                    print("Attach UE with ID ", ue.id, "to gNB with ID ", best_gnb.cell_id)

        # Allocate PRBs for attached UEs
        for gnb in self.gnbs.values():
            gnb.allocate()
            # print(alloc)
                        

        # Change UE position based on mobility model
        for ue in self.ues.values():
            if ue.mobility_model == UeMobilityModel.RandomWalk:
                new_position: Vector = Vector(0,0,ue.position.z)
                new_position_dx = ue.velocity.x * self.delta
                new_position_dy = ue.velocity.y * self.delta
                
                new_position.x = ue.position.x + new_position_dx
                new_position.y = ue.position.y + new_position_dy

                ue.set_position(new_position)

        if self.energy_saving_strategy == GnbSleepModeStrategy.Control:
            pass
        elif self.energy_saving_strategy == GnbSleepModeStrategy.Naive:
            for gnb in self.gnbs.values():
                if len(gnb.connected_ues) == 0 and gnb.radio_unit.asm_transition_state == AsmTransitionState.NONE and gnb.radio_unit.advanced_sleep_mode == AdvancedSleepMode.ACTIVE:
                    print("Set GNB with id ", gnb.cell_id, " to sleep")
                    self.set_advanced_sleep_mode(gnb, AdvancedSleepMode.SM1)

        self.ue_connection_turtle.clear()
        self.simulation_statistics_turtle.clear()
        self.update_component_connection_display()
        self.screen.update()

        self.time += self.delta
        

    def run(self, stop_time: float, verbose: bool = False) -> None:
        assert stop_time > 0
        time: float = self.time
        while time < stop_time:
            if verbose:
                print("Time step: ", time, "ms")
            
            self.step()

            time += self.delta
            self.time = time
            #sleep(0.25)


    

