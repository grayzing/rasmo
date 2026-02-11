from enum import Enum
from collections import deque
from time import sleep
from math import fmod
import tkinter, turtle
import numpy as np
import torch

from nr import NrUe, UeRrcState, NrGnb, AdvancedSleepMode, KeyPerformanceIndicator, UeMobilityModel
from embb_traffic_generator import EmbbTrafficGenerator
from utils import Vector, euclidean_distance
from embb_traffic_generator import EmbbTrafficGenerator, Packet

from graphics import UE_IMAGE, RU_IMAGE, RU_OFF_IMAGE, GRAPHICAL_SCALING_FACTOR

time_difference_tolerance: float = 1e-3

class ActionType(Enum):
    NrHandover = "NrHandover"
    NrHandoverRequestReceive = "NrHandoverRequestReceive"
    NrHandoverRequestResponseReceive = "NrHandoverRequestResponseReceive"
    NrUeKpiReportReceive = "NrUeKpiReportReceive"
    FtpPacketReceive = "FtpPacketReceive"
    AdvancedSleepModeAdjust = "AdvancedSleepModeAdjust"
    NrDownlinkDataTransmit = "NrDownlinkDataTransmit"

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
        if self.action == ActionType.NrUeKpiReportReceive:
            assert self.args

            assert self.args["destination_gnb"]
            assert self.args["ue"]
            assert self.args["kpi_report"]

            print("KPI report received by gNB at time ", self.parent.time)

            destination_gnb: NrGnb = self.args["destination_gnb"]
            ue: NrUe = self.args["ue"]
            kpi_report: dict = self.args["kpi_report"]

        elif self.action == ActionType.FtpPacketReceive:
            assert self.args

            assert self.args["destination_ue"]
            assert self.args["source_gnb"]
            assert self.args["packet"]    

            packet: Packet = self.args["packet"]
            source_gnb: NrGnb = self.args["source_gnb"]
            destination_ue: NrUe = self.args["destination_ue"]

            print("Packet received from gnb ", source_gnb.cell_id, " by UE ", destination_ue.id)

        elif self.action == ActionType.AdvancedSleepModeAdjust:
            assert self.args

            assert self.args["advanced_sleep_mode"]
            assert self.args["target_gnb"]

            advanced_sleep_mode: AdvancedSleepMode = self.args["advanced_sleep_mode"]
            target_gnb: NrGnb = self.args["target_gnb"]

            target_gnb.radio_unit.set_advanced_sleep_mode(advanced_sleep_mode)

        print("Executed event with ActionType ", self.action, " at time ", self.execution_time)


class Scheduler:
    def __init__(self, delta: float = 0.125) -> None:
        self.queue: deque[Event] = deque()
        self.delta: float = delta
        self.gnbs: dict[int, NrGnb] = {}
        self.ues: dict[int, NrUe] = {}
        self.items = []

        self.event_hooks: dict[EventHookType, list[Event]] = {event_hook_type : [] for event_hook_type in EventHookType}
        self.traffic_generator: EmbbTrafficGenerator = EmbbTrafficGenerator(50_000)

        self.time = 0

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

    def get_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate state for GraphDQN model.

        Takes geolocations of every node in simulation, RSRP between every connected node (just UEs and gNBs), current ASM
        
        :return: Edge tensor, edge weights, vertex embeddings
        :rtype: tuple[Tensor, Tensor, Tensor]
        """
        s = [[0 for j in range(len(self.ues))] for i in range(len(self.gnbs))]
        v = [1 for j in range(len(self.gnbs)+len(self.ues))]

        for gnb in self.gnbs:
            for ue in self.ues:
                pass

        return (torch.tensor(s),torch.tensor(v))

    def schedule_event(self, time_to_execute: float, action: ActionType, args: dict | None = None) -> None:
        self.queue.append(Event(self.time + time_to_execute, action, self, args))

    def add_traffic_generator(self, traffic_generator: EmbbTrafficGenerator):
        self.traffic_generator = traffic_generator

    def set_advanced_sleep_mode(self, target_gnb: NrGnb, advanced_sleep_mode: AdvancedSleepMode):
        tte = self.time
        
        #self.schedule_event(self.time, ActionType.AdvancedSleepModeAdjust, args={})

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

            self.items.append(self.gnbs[i])

        for i in range(m):
            self.ues[i] = NrUe(0,0,1.5,0, 1.5, i, self)
            self.ues[i].parent_scheduler = self

            self.ues[i].position.x = np.random.randint(-350, 350)
            self.ues[i].position.y = np.random.randint(-350, 350)

            self.items.append(self.ues[i])

        for i in range(n):
            self.gnbs[i].position.x = (i % 4 - 1.5) * 500
            self.gnbs[i].position.y = (i // 4 - 1.5) * 500
            self.gnbs[i].turtle.setposition(self.gnbs[i].position.x/GRAPHICAL_SCALING_FACTOR,self.gnbs[i].position.y/GRAPHICAL_SCALING_FACTOR)

        

    # 0 0 0 0
    # 0 0 0 0
    # 0 0 0 0
    # 0 0 0 0
    def add_gnb(self, gnb: NrGnb):
        self.gnbs[len(self.gnbs)] = gnb
        gnb.parent_scheduler = self
        gnb.cell_id = len(self.gnbs) - 1

        self.items.append(gnb)

    def add_ue(self, ue: NrUe):
        self.ues[len(self.ues)] = ue
        ue.parent_scheduler = self
        ue.id = len(self.ues) - 1

        self.items.append(ue)

    def get_closest_gnb(self, ue: NrUe) -> NrGnb | None:
        if not self.ues:
            return
        
        closest_gnb: NrGnb | None = None
        for gnb_pair in self.gnbs:
            gnb = self.gnbs[gnb_pair]
            tentative_distance: float = euclidean_distance(gnb.position, ue.position)

            if tentative_distance > 250:
                continue

            if closest_gnb is None:
                closest_gnb = gnb
                continue

            if tentative_distance <= euclidean_distance(closest_gnb.position, ue.position):
                closest_gnb = gnb

        return closest_gnb

    def step(self):
        candidate_event: Event
        while self.queue:
            #print(self.queue[-1].execution_time)
            if abs(self.queue[-1].execution_time - self.time) <= time_difference_tolerance:
                candidate_event = self.queue.pop()
                candidate_event.do()
            else:
                break

        # Handover to closest Gnb
        for gnb_pair in self.gnbs:
            #print(fmod(time,gnb.kpi_report_period))
            gnb = self.gnbs[gnb_pair]
            if fmod(self.time,gnb.kpi_report_period) <= time_difference_tolerance:
                for ue in gnb.connected_ues:
                    closest_gnb: NrGnb | None = self.get_closest_gnb(ue)

                    if ue.serving_gnb:
                        ue.serving_gnb.connected_ues.remove(ue)

                    ue.serving_gnb = closest_gnb
                    
                    if closest_gnb:
                        closest_gnb.connected_ues.append(ue)
                        closest_gnb.allocate()

        # Change UE position based on mobility model
        for ue in self.ues.values():
            if ue.mobility_model == UeMobilityModel.RandomWalk:
                new_position: Vector = Vector(0,0,ue.position.z)
                new_position_dx = ue.velocity.x * self.delta
                new_position_dy = ue.velocity.y * self.delta
                
                if np.random.randint(0,2) > -1:
                    new_position.x = ue.position.x + new_position_dx
                else:
                    new_position.x = ue.position.x - new_position_dx

                if np.random.randint(0,2) > -1:
                    new_position.y = ue.position.y + new_position_dy
                else:
                    new_position.y = ue.position.y - new_position_dy

                ue.set_position(new_position)

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


    

