from utils import Vector
from enum import Enum
from warnings import warn
import numpy as np
import turtle
from graphics import UE_IMAGE, RU_IMAGE, RU_OFF_IMAGE, GRAPHICAL_SCALING_FACTOR
from utils import path_loss_los, path_loss_nlos, sinr, se_table

class AdvancedSleepMode(Enum):
    ACTIVE = (0, 0, 0)
    SM1 = (0.00355, 0.0071, 1)
    SM2 = (0.5, 1, 2)
    SM3 = (5, 10, 3)
    SM4 = (500, 1000, 4)
    
class AsmTransitionState(Enum):
    NONE = 0
    TRAN = 1
    DEBO = 2

class UeMobilityModel(Enum):
    ConstantPosition = "ConstantPosition",
    RandomWalk = "RandomWalk"
    
class ProportionalFairScheduler:
    def __init__(self, num_users: int, total_prbs: int = 66, alpha: float = 0.001):
        self.num_users = num_users
        self.alpha = alpha

        self.total_prbs = total_prbs

    def calculate_metric(self, ue: 'NrUe') -> float:
        """
        Calculate PF metric for the given UE. 
        
        :param ue: The UE to calculate the PF metric for.
        :type ue: 'NrUe'
        :return: The PF metric
        :rtype: float
        """
        return ue.instantaneous_rate / ue.average_throughput
    
    def schedule(self, ues: list['NrUe']) -> dict['NrUe', int]:
        active_users = [
            ue for ue in ues
            if ue.instantaneous_rate > 0
        ]

        #print(active_users)

        if len(active_users) < 1:
            return {}
        
        user_metrics = []
        for uid, user in enumerate(active_users):
            metric = self.calculate_metric(user)
            user_metrics.append((user, uid, metric, user.instantaneous_rate))

        user_metrics.sort(key=lambda x: x[1], reverse=True)

        allocations = {}

        remaining_prbs = self.total_prbs

        for ue, uid, metric, inst_rate in user_metrics:
            if remaining_prbs <= 0:
                break
                
            # Allocate based on demand and metric
            # Simple approach: equal share for top users
            alloc_prbs = max(1, self.total_prbs // len(user_metrics))
            alloc_prbs = min(alloc_prbs, remaining_prbs)
            
            allocations[ue] = alloc_prbs
            remaining_prbs -= alloc_prbs
            
            # Calculate actual rate achieved (simplified)
            # In reality: depends on MCS, PRBs, TTI duration
            actual_rate = inst_rate * (alloc_prbs / self.total_prbs)
            
            # Update user's average throughput
            ue.update_average_throughput(actual_rate, self.alpha, True)
            ue.assigned_prbs_history.append(alloc_prbs)
            
        # Update unscheduled users (aging)
        for ue in ues:
            if ue not in allocations:
                ue.update_average_throughput(0, self.alpha, False)

        return allocations

class RadioUnit:
    def __init__(self, parent: 'NrGnb') -> None:
        """
        Initialize the RU

        :param parent: The NrGnb that owns this RU
        :type parent: NrGnb
        """

        self.parent = parent

        self.tx_power: float = 35.0 #dB

        self.advanced_sleep_mode: AdvancedSleepMode = AdvancedSleepMode.ACTIVE
        self.asm_transition_state: AsmTransitionState = AsmTransitionState.NONE

    def get_power_consumption(self) -> float:
        """
        Return power consumption of 4T4R RU at different ASMs according to table on
        Power Modeling of the O-RAN O-RU & Application of Advanced Sleep Modes for
        Enhanced Energy Efficiency by Usman et. al
    
        :return: Power consumption of the RU in W.
        :rtype: float
        """
        assert self.advanced_sleep_mode in AdvancedSleepMode

        if self.advanced_sleep_mode == AdvancedSleepMode.ACTIVE:
            return 397.0
        elif self.advanced_sleep_mode == AdvancedSleepMode.SM1:
            return 88.0
        elif self.advanced_sleep_mode == AdvancedSleepMode.SM2:
            return 40.0
        elif self.advanced_sleep_mode == AdvancedSleepMode.SM3:
            return 28.0
        elif self.advanced_sleep_mode == AdvancedSleepMode.SM4:
            return 15.0
        
        return 0.0
        

    def set_advanced_sleep_mode(self, asm: AdvancedSleepMode)->None:
        """
        Change the advanced sleep mode of the RU.
        SM1 will turn off the PA and AF, takes one frame to activate/deactivate.
        
        :param self: Description
        :param asm: Description
        :type asm: ADVANCED_SLEEP_MODE
        """
        self.advanced_sleep_mode = asm


class NrGnb:
    """
    A class for the NrGnb. Since this is a system-level simulator, there is only
    some rough simulation of resource-block allocation

    """
    def __init__(self, x: float, y: float, z: float, height: float, cell_id: int = 0, parent = None) -> None:
        """
        Initialize NrGnb in accordance to 3GPP specifications for UMA scenario.
        
        :param x: The x value of the gNB's position (in meters)
        :type x: float
        :param y: The y value of the gNB's position (in meters)
        :type y: float
        :param z: The z value of the gNB's position (in meters)
        :type z: float
        :param height: The height of the gNB's antennae (in meters)
        :type height: float
        :param cell_id: The ID of the gNB. Only handled by NrHelper usually.
        :type cell_id: int
        :param parent: The NrHelper that owns the gNB. Optional.
        :type parent: NrHelper | None
        """
        self.cell_id: int = cell_id
        self.position: Vector = Vector(x,y,z)
        self.radio_unit: RadioUnit = RadioUnit(self)

        self.connected_ues: list[NrUe] = []
        self.parent_scheduler = parent

        self.tx_power: float = 35 #dB
        self.tx_freq: float = 30 #Ghz
        self.bandwidth: float = 100 #Mhz

        self.num_prbs: int = 66 # For numerology 3

        self.ue_scheduler: ProportionalFairScheduler = ProportionalFairScheduler(len(self.connected_ues), self.num_prbs)

        self.initialize_turtle()

    def initialize_turtle(self)->None:
        self.turtle = turtle.Turtle()
        self.turtle.penup()
        self.turtle.speed(0)
        self.turtle.setposition(self.position.x/GRAPHICAL_SCALING_FACTOR,self.position.y/GRAPHICAL_SCALING_FACTOR)
        self.turtle.shape(RU_IMAGE)
        self.turtle.setheading(90)

    def get_parent(self):
        return self.parent_scheduler
    
    def set_parent(self, parent):
        self.parent_scheduler = parent

    def get_position(self) -> Vector:
        return self.position

    def allocate(self):
        alloc = self.ue_scheduler.schedule(self.connected_ues)
        for ue in alloc:
            ue.assigned_prbs = alloc[ue]
        return alloc

    def remove_ue(self, ue: 'NrUe'):
        if not ue in self.connected_ues:
            return
        self.ue_scheduler.num_users -= 1
        self.connected_ues.remove(ue)
        self.allocate()
        ue.assigned_prbs = 0
    

class NrUe:
    def __init__(self, x: float, y: float, z: float, rnti: int, antenna_height: float, id: int = 0, parent_scheduler = None, servingGnb: NrGnb | None = None) -> None:
        """
        Docstring for __init__
        
        :param x: The x value of the UE's position (in meters)
        :type x: float
        :param y: The y value of the UE's position (in meters)
        :type y: float
        :param z: The z value of the UE's position (in meters)
        :type z: float
        :param servingGnb: The gNB serving this UE.
        :param rnti: The RNTI of the UE.
        :type rnti: float
        :param antenna_height: The height of the UE antenna.
        :type antenna_height: float
        :param parent_scheduler: The sceduler that can modify this UE.
        :type parent_scheduler: Scheduler 
        :type servingGnb: NrGnb | None
        """
        self.serving_gnb: NrGnb | None = servingGnb

        self.position: Vector = Vector(x,y,z)

        self.antenna_height: float = antenna_height # meters
        
        self.parent_scheduler = parent_scheduler

        self.mobility_model = UeMobilityModel.RandomWalk
        self.velocity: Vector = Vector(0.00083, 0, 0.00083) # meters/ms

        self.id: int = id

        self.average_throughput: float = 1.0
        self.instantaneous_rate: float = 10e-6 # small number to prevent division by zero error

        self.assigned_prbs_history: list[float] = []
        self.throughput_history: list[float] = []

        self.min_guaranteed_rate: float = 0.0
        self.priority_class: float = 1

        self.assigned_prbs: int = 0

        self.initialize_turtle()

    def initialize_turtle(self):
        self.turtle = turtle.Turtle()
        self.turtle.penup()
        self.turtle.speed(0)
        self.turtle.setposition(self.position.x/GRAPHICAL_SCALING_FACTOR,self.position.y/GRAPHICAL_SCALING_FACTOR)
        self.turtle.shape(UE_IMAGE)
        self.turtle.setheading(90)

    def update_instantaneous_rate(self) -> None:
        if not self.serving_gnb:
            # print("No serving gnb")
            self.instantaneous_rate = -1
            return
        
        # Update channel conditions for UE
        sinr_db = sinr(self.serving_gnb.position, self.position, self.serving_gnb.tx_freq)
        
        # Map SINR to CQI and rate (simplified)
        cqi = max(1, min(15, int((sinr_db + 10) / 2)))
        
        # Spectral efficiency from CQI (approximate)
        se = se_table[cqi - 1] if cqi <= 15 else 5.55
        
        # Achievable rate (bps) for 1 PRB in 1ms TTI
        # 12 subcarriers * 14 symbols * bits per symbol
        bits_per_re = se  # bits per resource element
        re_per_prb = 12 * 14
        rate_per_prb = re_per_prb * bits_per_re * 1000  # per second
        # print(rate_per_prb)
        self.instantaneous_rate = rate_per_prb

    def update_average_throughput(self, actual_rate: float, alpha: float, 
                                  was_scheduled: bool):
        """
        Update average throughput using exponential moving average
        If not scheduled, average decreases (aging)
        """
        if was_scheduled:
            self.avg_throughput = (1 - alpha) * self.average_throughput + alpha * actual_rate
        else:
            self.avg_throughput = (1 - alpha) * self.average_throughput
            
        self.average_throughput = max(self.average_throughput, 1e-6)
        
        self.throughput_history.append(actual_rate)
        
    def rsrp(self) -> float:
        if not self.serving_gnb:
            return -np.inf
        return self.serving_gnb.tx_power - path_loss_los(self.serving_gnb.position, self.position, self.serving_gnb.tx_freq)
        
    def get_position(self) -> Vector:
        return self.position
    
    def set_position(self, position: Vector) -> None:
        self.position = position
        self.turtle.setposition(self.position.x/GRAPHICAL_SCALING_FACTOR,self.position.y/GRAPHICAL_SCALING_FACTOR)

def shannon(u: NrGnb, v: NrUe):
    return 1.44 * np.log2(1 + sinr(u.position, v.position, u.tx_freq))



