import numpy as np

class Vector:
    """
    Class for vector in R^3
    """
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

c = 3e8 / 10e3 # Speed of light
se_table = [0.15, 0.23, 0.38, 0.6, 0.88, 1.18, 1.48, 1.91, 
            2.41, 2.73, 3.32, 3.9, 4.52, 5.12, 5.55]

def euclidean_distance(u: Vector, v: Vector) -> float:
    """
    Return euclidean distance between two vectors in R^3
    
    :param u: First vector
    :type u: Vector
    :param v: Second vector
    :type v: Vector
    :return: Euclidean distance between u and v
    :rtype: float
    """
    return np.sqrt((u.x-v.x)**2 + (u.y-v.y)**2 + (u.z-v.z)**2)

def euclidean_distance_2d(u: Vector, v: Vector) -> float:
    """
    Return euclidean distance between two vectors in R^2 (x and z only)
    
    :param u: First vector
    :type u: Vector
    :param v: Second vector
    :type v: Vector
    :return: Two-dimensional euclidean distance between u and v
    :rtype: float
    """
    return np.sqrt((u.x-v.x)**2 + (u.y-v.y)**2)

def propagation_delay(u: Vector, v: Vector) -> float:
    """
    Return propagation delay in ms between u and v
    
    :param u: First vector
    :type u: Vector
    :param v: Second vector
    :type v: Vector
    :return: Propagation delay in ms
    :rtype: float
    """

    return euclidean_distance(u,v) / c

def path_loss_los(u: Vector, v: Vector, frequency: float) -> float:
    """
    Calculate UMA Line-Of-Sight (LOS) pathloss between gNB at position u and UE at position v
    
    :param u: Position of gNB
    :type u: Vector
    :param v: Position of UE
    :type v: Vector
    :param frequency: Wave frequency
    :type frequency: float
    :return: UMA LOS Pathloss
    :rtype: float
    """
    distance_2d: float = euclidean_distance_2d(u,v)
    distance_3d: float = euclidean_distance(u,v)

    base_station_height: float = u.z
    user_equipment_height: float = v.z

    distance_bp_prime: float = 35 # meters

    if distance_2d >= 10 and distance_2d <= distance_bp_prime:
        return 28.0 + 22 * np.log10(distance_3d) + 20 * np.log10(frequency)
    
    elif distance_2d > distance_bp_prime and distance_2d <= 5_000:
        return 28.0 + 40 * np.log10(distance_3d) + 20*np.log10(frequency) - 9 * np.log10((distance_bp_prime)**2 + (base_station_height - user_equipment_height)**2)

    else:
        return 0
    
def path_loss_nlos(u: Vector, v: Vector, frequency: float) -> float:
    distance_2d: float = euclidean_distance_2d(u,v)
    distance_3d: float = euclidean_distance(u,v)

    if distance_2d < 10:
        return 0

    base_station_height: float = u.z
    user_equipment_height: float = v.z

    path_loss_prime_nlos = 13.54 + 39.08 * np.log10(distance_3d) + 20*np.log10(frequency) - 0.6 * (user_equipment_height - 1.5)

    return max(path_loss_prime_nlos, path_loss_los(u,v,frequency))

def los_probability(u: Vector, v: Vector) -> float:
    distance_2d: float = euclidean_distance_2d(u,v)
    return min(18/distance_2d, 1.0) * (1 - np.exp(-distance_2d/36)) + np.exp(-distance_2d/36)

def sinr(u: Vector, v: Vector, frequency: float) -> float:
    return 46 - path_loss_los(u,v,frequency) - (-174 + 10*np.log10(20e6))

def rsrp(u: Vector, v: Vector, frequency: float, tx_power: float) -> float:
    return tx_power - path_loss_nlos(u,v,frequency)