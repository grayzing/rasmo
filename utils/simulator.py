import time
import torch

from collections import deque

# Constants
c = 3e8  # Speed of light in m/s
packet_propagation_delay_tolerance = 1e-3 # Since we are simulating at a very high level, we can ignore small delays

# Utility functions
def euclidean_distance(p1: tuple, p2: tuple) -> float:
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

class Packet:
    def __init__(self, data: bytes, source: 'BaseStation | UserEquipment', destination: 'BaseStation | UserEquipment', timestamp: float = time.time()):
        self.data = data
        self.source = source
        self.destination = destination
        self.timestamp = timestamp

class BaseStation:
    def __init__(self, id, num_transmit_antenna=3, num_receive_antenna=3):
        # Base stations are modelled as 3-sector gnBs, uMa scenario
        self.id = id
        self.sleep_mode = 1
        self.num_transmit_antenna = num_transmit_antenna
        self.num_receive_antenna = num_receive_antenna
        self.fdd_band: list[int] = []
        
        self.queue: deque[Packet] = deque()
        
        self.p = (0,0)  # Position (x, y)
        self.height = 25  # Antenna height in meters
        
    def connect_to_user_equipment(self, ue: 'UserEquipment'):
        pass
    
    def set_sleep_mode(self, mode: int):
        assert mode in [1, 2, 3, 4], "Invalid sleep mode"
        self.sleep_mode = mode
        
    
    
class UserEquipment:
    def __init__(self, id=0):
        self.p = (0, 0)  # Position (x, y)
        self.height = 1.5  # Height in meters
        self.id = id

class Simulator:
    def __init__(self):
        self.running = False
        
        self.user_equipments: list[UserEquipment] = []
        self.base_stations: list[BaseStation] = []
        
        self.packets_in_transmission: deque[Packet] = deque()
        
    def create_user_equipment(self) -> None:
        ue = UserEquipment(id=len(self.user_equipments))
        self.user_equipments.append(ue)
        
    def remove_user_equipment(self, ue: UserEquipment) -> None:
        assert ue in self.user_equipments, "User equipment not found"
        self.user_equipments.remove(ue)
        
    def create_base_station(self) -> None:
        bs = BaseStation(id=len(self.base_stations))
        self.base_stations.append(bs)
        
    def remove_base_station(self, bs: BaseStation) -> None:
        assert bs in self.base_stations, "Base station not found"
        self.base_stations.remove(bs)
        
    def start(self):
        self.running = True
        
        while self.running:
            current_time = time.time()
            if self.packets_in_transmission:
                packet = self.packets_in_transmission[0]
                
                # Process packets in transmission
                distance = euclidean_distance(packet.source.p, packet.destination.p)
                propagation_delay = distance / c
                if abs((current_time - packet.timestamp) - propagation_delay) <= packet_propagation_delay_tolerance:
                    if isinstance(packet.destination, BaseStation):
                        packet.destination.queue.append(packet)
                    self.packets_in_transmission.popleft()
                    
                print(f"{time.time()} Packet from {packet.source.id} to {packet.destination.id} in transmission, distance: {distance:.2f} m, propagation delay: {propagation_delay*1e3:.2f} ms")
            time.sleep(0.000001)  # Sleep to prevent busy waiting

    def stop(self):
        self.running = False

    def send_packet(self, source: BaseStation | UserEquipment, destination: BaseStation | UserEquipment, packet: bytes):
        timestamp = time.time()
        print(f"{timestamp} Sending packet from {source.id} to {destination.id}")
        self.packets_in_transmission.append(Packet(data=packet, source=source, destination=destination, timestamp=timestamp))

    def generate_edge_matrix(self):
        pass
    
