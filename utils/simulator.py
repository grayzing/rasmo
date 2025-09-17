import time
import numpy as np
import threading
import torch

from collections import deque

# Constants
c = 3e8  # Speed of light in m/s
packet_propagation_delay_tolerance = 1e-3 # Since we are simulating at a very high level, we can ignore small delays
advanced_sleep_mode_delay_tolerance = 1e-3
lorem_ipsum_text = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."

# Mappings
sleep_mode_time_mapping: dict[int, tuple[float,float]] = { # For this mapping, first entry of tuple is how long it takes to activate and deactivate, second entry is minimum time to stay in that mode
    1: (35.5e-6, 71e-6),
    2: (0.5e-3, 1e-3),
    3: (5e-3, 10e-3),
    4: (0.5, 1),
}

# Utility functions
def euclidean_distance(p1: tuple[float,float], p2: tuple[float,float]) -> float:
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
        self.p: tuple[float,float] = (0, 0)  # Position (x, y)
        self.height = 1.5  # Height in meters
        self.id = id
        
        self.base_station: BaseStation | None = None

class Simulator:
    def __init__(self):
        self.running = False
        
        self.user_equipments: list[UserEquipment] = []
        self.base_stations: list[BaseStation] = []
        
        self.packets_in_transmission: deque[Packet] = deque()
        self.advanced_sleep_mode_buffer: deque[tuple[BaseStation, int, float]] = deque()
        
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
        
    def generate_random_packet(self) -> None:
        random_base_station: BaseStation = self.base_stations[0]
        random_user_equipment: UserEquipment = self.user_equipments[0]
        random_packet: Packet = Packet(data=b"Hello, World!", source=random_base_station, destination=random_user_equipment)
        self.send_packet(random_packet.source, random_packet.destination, random_packet.data)
        
    def step(self) -> None:
        ue: UserEquipment
        for ue in self.user_equipments:
            if ue.base_station:
                bs = ue.base_station
                packet = Packet(data=lorem_ipsum_text, source=bs, destination=ue)
                self.send_packet(packet.source, packet.destination, packet.data)
            ue.p = (ue.p[0] + np.random.uniform(-1, 1), ue.p[1] + np.random.uniform(-1, 1)) # Random walk

    def start(self):
        self.running = True
        self.generate_random_packet()
        while self.running:
            current_time = time.time()
            if self.packets_in_transmission:
                packet = self.packets_in_transmission[0]
                print(f"{time.time()} Packet from {type(packet.source).__name__} {packet.source.id} to {type(packet.destination).__name__} {packet.destination.id} in transmission")
                
                # Process packets in transmission
                distance = euclidean_distance(packet.source.p, packet.destination.p)
                propagation_delay = distance / c
                if abs((current_time - packet.timestamp) - propagation_delay) <= packet_propagation_delay_tolerance:
                    if isinstance(packet.destination, BaseStation):
                        packet.destination.queue.append(packet)
                    self.packets_in_transmission.popleft()
                    print(f"{time.time()} Packet from {type(packet.source).__name__} {packet.source.id} to {type(packet.destination).__name__} {packet.destination.id} delivered")
            if self.advanced_sleep_mode_buffer:
                bs: BaseStation
                mode: int
                sleep_mode_command_timestamp: float
                bs, mode, sleep_mode_command_timestamp = self.advanced_sleep_mode_buffer[0]
                if abs((current_time - sleep_mode_command_timestamp) - sleep_mode_time_mapping[mode][0]) <= advanced_sleep_mode_delay_tolerance:
                    bs.set_sleep_mode(mode)
                    self.advanced_sleep_mode_buffer.popleft()
                    print(f"{time.time()} BaseStation {bs.id} changed to sleep mode {mode}")
            time.sleep(0.000001)  # Sleep to prevent busy waiting

    def stop(self):
        self.running = False

    def send_packet(self, source: BaseStation | UserEquipment, destination: BaseStation | UserEquipment, packet: bytes):
        timestamp = time.time()
        print(f"{timestamp} Sending packet from {source.id} to {destination.id}")
        self.packets_in_transmission.append(Packet(data=packet, source=source, destination=destination, timestamp=timestamp))

    def get_state_information(self)-> tuple[np.ndarray, list[tuple[int,int]], list[float]]:
        # Returns node feature vector X, edges between UEs and their associated BSs, and edge weights
        attached_ues: list[UserEquipment] = [ue for ue in self.user_equipments if ue.base_station]
        node_feature_vector = np.array([bs.sleep_mode for bs in self.base_stations] + [0] + [0 for _ in attached_ues]).reshape(-1, 1)  # Node feature vector
        # +1 for dummy node, which represents no action taken
        edges = torch.zeros((2, 2*len(attached_ues)), dtype=torch.long).tolist()  # Edges between UEs and their associated BSs
        edge_weights = []
        for i, ue in enumerate(attached_ues):
            if ue.base_station:
                # Undirected graph
                edges[0][i] = ue.id
                edges[1][i] = ue.base_station.id
                
                edges[0][i+1] = ue.base_station.id
                edges[1][i+1] = ue.id
                edge_weights.append(1.0)  # Default weight
        return node_feature_vector, edges, edge_weights
    
    def get_reward(self) -> float:
        return 1.0  # Placeholder reward function
