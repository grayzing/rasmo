from utils import propagation_delay, Vector

class Packet:
    def __init__(self, data_size: int, delay: float) -> None:
        self.data_size = data_size
        self.delay = delay 

class EmbbTrafficGenerator:
    def __init__(self, traffic_size: int) -> None:
        self.traffic_size: int = traffic_size #in bits
        self.arrival_rate: float = 1.0
        
    def generate_packet(self, origin: Vector, destination: Vector) -> Packet:
        """
        Generate a packet with traffic_size bits which will be transmitted from the origin vector to the destination vector
        
        :param origin: Position of origin of transmission (in this simulation a Radiounit)
        :type origin: Vector
        :param destination: Position of destination of transmission (in this simulation an NrUe)
        :type destination: Vector
        :return: New packet with size traffic_size
        :rtype: Packet
        """
        return Packet(self.traffic_size, propagation_delay(origin,destination))