from utils import Vector
from enum import Enum
import numpy as np

from medium_access_control import TimeDivisionDuplexingTable

class AdvancedSleepMode(Enum):
    ACTIVE = 0
    SM1 = 1
    SM2 = 2
    SM3 = 3
    SM4 = 4

class RuComponent:
    def __init__(self) -> None:
        self.status = True
    
    def set_status(self, status: bool)->None:
        """
        Change the status of the RuComponent.
        
        :param status: Status that you want to set the RuC to.
        :type status: bool
        """
        if self.status != status:
            self.status = status

class RuAnalogFrontend(RuComponent):
    def __init__(self) -> None:
        super().__init__()

class RuPowerAmplifier(RuComponent):
    def __init__(self) -> None:
        super().__init__()

class RuFFT(RuComponent):
    def __init__(self, fft_size: int = 1024) -> None:
        super().__init__()

        self.fft_size: int = fft_size

class RuIFFT(RuComponent):
    def __init__(self) -> None:
        super().__init__()
    
class RadioUnit:
    def __init__(self, parent: 'NrGnb') -> None:
        """
        Initialize the RU

        :param parent: The NrGnb that owns this RU
        :type parent: NrGnb
        """

        self.ru_analog_frontend = RuAnalogFrontend()
        self.ru_power_amplifier = RuPowerAmplifier()
        self.ru_fft = RuFFT()
        self.ru_ifft = RuIFFT()

        self.parent = parent

        self.tx_power: float = 10.0
        self.rx_power: float = 5.0

        self.node_size: float = 65 # nm

        self.resource_allocation_table: TimeDivisionDuplexingTable = TimeDivisionDuplexingTable(15, 7)

    def get_power_consumption(self)->float:
        """
        Calculate power consumption of the RU according to the model given in
        Power Modeling of the O-RAN O-RU & Application of Advanced Sleep Modes for
        Enhanced Energy Efficiency by Usman et. al
    
        :return: Power consumption of the RU in watts.
        :rtype: float
        """

        p_amp: float = self.ru_power_amplifier.status * self.tx_power / (0.8 * self.tx_power)
        return p_amp

    def set_advanced_sleep_mode(self, asm: AdvancedSleepMode)->None:
        """
        Change the advanced sleep mode of the RU.
        SM1 will turn off the PA and AF, takes one frame to activate/deactivate.
        
        :param self: Description
        :param asm: Description
        :type asm: ADVANCED_SLEEP_MODE
        """
        if asm == AdvancedSleepMode.SM1:
            self.ru_power_amplifier.set_status(False)


class NrGnb:
    """
    A class for the NrGnb. Since this is a system-level simulator, there is only
    some rough simulation of resource-block allocation

    """
    def __init__(self, x: float, y: float, z: float, height: float, cell_id: int = 0, parent: object | None = None) -> None:
        """
        Docstring for __init__
        
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

        self.connectedUes: list[NrUe] = []
        self.parent: object | None = parent

        

    def get_parent(self) -> object | None:
        return self.parent
    
    def set_parent(self, parent: object):
        self.parent = parent

    def register_ue(self, ue):
        """
        Register UE with gNB.
        
        :param ue: The desired UE
        :type ue: NrUe
        """
        rng = np.random.default_rng()

        candidate_rnti: int = rng.integers(0,10**4)
        while candidate_rnti in self.connectedUes:
            candidate_rnti = rng.integers(0,10**4)

        self.connectedUes.append(ue)
        ue.rnti = candidate_rnti

        # TODO: Perform resource allocation for the UE, THEN have the UE decide whether to register with this gNB.
        ue.attach_to_gnb(self)

    def request_handover(self, ue: 'NrUe', target_gnb: 'NrGnb'):
        if ue.rrc_state == UeRrcState.RRC_CONNECTED:
            pass
        
        self.connectedUes.remove(ue)
        target_gnb.register_ue(ue)
        

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
    def __init__(self, x: float, y: float, z: float, rnti: int, antenna_height: float, parent_scheduler: object | None = None, servingGnb: NrGnb | None = None) -> None:
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
        self.servingGnb: NrGnb | None = servingGnb
        self.rnti: int = rnti

        self.antenna_height: float = antenna_height # meters
        
        self.parent_scheduler = parent_scheduler

    def attach_to_gnb(self, target_gnb: NrGnb):
        assert self.parent_scheduler
        assert self.servingGnb
        
        if self.rrc_state == UeRrcState.RRC_IDLE or self.rrc_state == UeRrcState.RRC_INACTIVE:
            self.servingGnb = target_gnb
            return
        
    def set_rrc_state(self, rrc_state: UeRrcState):
        self.rrc_state = rrc_state

        



