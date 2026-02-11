from utils import Vector
from enum import Enum
from warnings import warn
import numpy as np
import turtle
from graphics import UE_IMAGE, RU_IMAGE, RU_OFF_IMAGE, GRAPHICAL_SCALING_FACTOR


class AdvancedSleepMode(Enum):
    SM1 = (0.00355, 0.0071,0 )
    SM2 = (0.5, 1, 1)
    SM3 = (5, 10, 2)
    SM4 = (500, 1000, 3)
    ACTIVE = (505.6, 1011, 4)

class KeyPerformanceIndicator(Enum):
    Rsrp = "Rsrp"
    Geolocation = "Geolocation"

class UeMobilityModel(Enum):
    ConstantPosition = "ConstantPosition",
    RandomWalk = "RandomWalk"

class RoundRobinScheduler:
    def __init__(self, num_users: int, num_prbs: int, nr_gnb: 'NrGnb') -> None:
        self.num_users = num_users
        self.user_prb_mapping: dict[int, int] = {}
        self.num_prbs = num_prbs
        self.parent_gnb: NrGnb = nr_gnb
        self.prb_pointer = 0  # Which PRB to start with

    def schedule(self, achievable_rates_per_prb: np.ndarray) -> np.ndarray:
        allocation = np.zeros(self.num_prbs, dtype=int)
        
        for prb in range(self.num_prbs):
            # Assign this PRB to next user in cycle
            user = (self.prb_pointer + prb) % self.num_users
            allocation[prb] = user
            
        # Advance pointer for next slot
        self.prb_pointer = (self.prb_pointer + self.num_prbs) % self.num_users
        
        return allocation
    
    def calculate_user_rates(self, allocation: np.ndarray, 
                           rates_per_prb: np.ndarray) -> dict:
        self.user_prb_mapping = {ue.id : 0 for ue in self.parent_gnb.connected_ues}
        
        for prb, user in enumerate(allocation):
            self.user_prb_mapping[user] += rates_per_prb[prb]
            
        return self.user_prb_mapping

class RadioUnit:
    def __init__(self, parent: 'NrGnb') -> None:
        """
        Initialize the RU

        :param parent: The NrGnb that owns this RU
        :type parent: NrGnb
        """

        self.ru_analog_frontend_status = True
        self.ru_power_amplifier_status = True
        self.ru_fft_status = True
        self.ru_ifft_status = True

        self.parent = parent

        self.tx_power: float = 10.0 #dB
        self.rx_power: float = 5.0 #dB

        self.node_size: float = 65 # nm

        self.advanced_sleep_mode: AdvancedSleepMode = AdvancedSleepMode.ACTIVE

        

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
        if asm == AdvancedSleepMode.SM1:
            self.advanced_sleep_mode = AdvancedSleepMode.SM1

            self.ru_power_amplifier_status = False
            self.ru_analog_frontend_status = False

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
        self.cell_id = cell_id
        self.position: Vector = Vector(x,y,z)
        self.radio_unit: RadioUnit = RadioUnit(self)

        self.kpi_report_period: float = 24 # ms

        self.connected_ues: list[NrUe] = []
        self.parent_scheduler = parent

        self.tx_power: float = 35 #dB
        self.tx_freq: float = 30 #Ghz
        self.bandwidth: float = 100 #Mhz

        self.num_prbs: int = 66 # For numerology 3

        self.ue_scheduler: RoundRobinScheduler = RoundRobinScheduler(len(self.connected_ues), self.num_prbs, self)

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
        achievable_rates = np.zeros(len(self.connected_ues))

        for i in range(len(achievable_rates)):
            achievable_rates[i] = 1.2

        self.ue_scheduler.schedule(achievable_rates)

    def register_ue(self, ue):
        """
        Register UE with gNB.
        
        :param ue: The desired UE
        :type ue: NrUe
        """
        rng = np.random.default_rng()

        candidate_rnti: int = rng.integers(0,10**4)
        while candidate_rnti in self.connected_ues:
            candidate_rnti = rng.integers(0,10**4)

        self.connected_ues.append(ue)
        ue.rnti = candidate_rnti

        # TODO: Perform resource allocation for the UE, THEN have the UE decide whether to register with this gNB.
        ue.attach_to_gnb(self)

    def request_handover(self, ue: 'NrUe', target_gnb: 'NrGnb'):
        assert self.parent_scheduler

        if ue.rrc_state == UeRrcState.RRC_CONNECTED:
            pass

        self.parent_scheduler.schedule_event(
            5, 
            "KpiReportReceive", 
            args={
                "source_gnb": self,
                "destination_gnb": target_gnb,
                "ue": ue
            }
        )
        

class UeRrcState(Enum):
    """
    States for UE RRC.
    When UE is actively transmitting information to its serving gNB, its state is
    RRC_CONNECTED. If it has not transmitted information to its serving gNB for
    some period of time, its state is RRC_INACTIVE. If the UE is not being served
    by any gNB, its state is RCC_IDLE
    """
    RRC_CONNECTED = 1
    RRC_INACTIVE  = 2
    RRC_IDLE      = 3

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
        self.rrc_state = UeRrcState.RRC_IDLE
        self.serving_gnb: NrGnb | None = servingGnb
        self.rnti: int = rnti

        self.position: Vector = Vector(x,y,z)

        self.antenna_height: float = antenna_height # meters
        
        self.parent_scheduler = parent_scheduler

        self.mobility_model = UeMobilityModel.RandomWalk
        self.velocity: Vector = Vector(0.00083, 0, 0.00083) # meters/s

        self.id: int = id

        self.average_throughput: float = 1.0
        self.instantaneous_rate: float = 0

        self.throughput_history: list[float] = []

        self.min_guaranteed_rate: float = 0.0
        self.priority_class: float = 1

        self.turtle = turtle.Turtle()
        self.turtle.penup()
        self.turtle.speed(0)
        self.turtle.setposition(self.position.x/GRAPHICAL_SCALING_FACTOR,self.position.y/GRAPHICAL_SCALING_FACTOR)
        self.turtle.shape(UE_IMAGE)
        self.turtle.setheading(90)

        

    def update_average_throughput(self, actual_rate: float, alpha: float, was_scheduled: bool) -> None:
        pass

    def attach_to_gnb(self, target_gnb: NrGnb):
        assert self.parent_scheduler
        assert self.servingGnb
        
        if self.rrc_state == UeRrcState.RRC_IDLE or self.rrc_state == UeRrcState.RRC_INACTIVE:
            self.servingGnb = target_gnb
            return
        
    def rsrp(self) -> float | None:
        if not self.servingGnb:
            warn("Attempted to get RSRP without a serving gNB")
            return None
        
        return 1.0
        
    def get_position(self) -> Vector:
        return self.position
    
    def set_position(self, position: Vector) -> None:
        self.position = position
        self.turtle.setposition(self.position.x/GRAPHICAL_SCALING_FACTOR,self.position.y/GRAPHICAL_SCALING_FACTOR)
        
    def create_kpi_report(self) -> dict[KeyPerformanceIndicator, object]:
        assert self.parent_scheduler

        if not self.servingGnb:
            warn("Attempted to send KPI report without a serving gNB")
            return {}

        kpi_report: dict[KeyPerformanceIndicator, object] = {}

        kpi_report[KeyPerformanceIndicator.Rsrp] = self.rsrp()
        kpi_report[KeyPerformanceIndicator.Geolocation] = self.get_position()

        return kpi_report
        
    def set_rrc_state(self, rrc_state: UeRrcState):
        self.rrc_state = rrc_state

        



