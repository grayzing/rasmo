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
isd = 200 # meters
spatial_proximity_weighing_factor = 1 # For calculating total interference at a UE
rsrp_bound = -90
se_table = [0.15, 0.23, 0.38, 0.6, 0.88, 1.18, 1.48, 1.91, 
            2.41, 2.73, 3.32, 3.9, 4.52, 5.12, 5.55]

def euclidean_distance(u: Vector, v: Vector) -> float:
    """
    Return euclidean distance between two vectors in R^3 in km.
    
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
    Return euclidean distance between two vectors in R^2 in km (x and y only)
    
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

def transmission_delay() -> float:
    return 0.0

def total_delay(u: Vector, v: Vector) -> float:
    return propagation_delay(u,v) + transmission_delay() + np.random.uniform(0,1)

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
        return 32.4 + 21 * np.log10(distance_3d) + 20 * np.log10(frequency)
    
    elif distance_2d > distance_bp_prime and distance_2d <= 5_000:
        return 28.0 + 40 * np.log10(distance_3d) + 20*np.log10(frequency) - 9.5 * np.log10((distance_bp_prime)**2 + (base_station_height - user_equipment_height)**2)

    else:
        return 0
    
def path_loss_nlos(u: Vector, v: Vector, frequency: float) -> float:
    distance_2d: float = euclidean_distance_2d(u,v)
    distance_3d: float = euclidean_distance(u,v)

    if distance_2d < 10:
        return 0

    base_station_height: float = u.z
    user_equipment_height: float = v.z

    path_loss_prime_nlos = 22.4 + 35.3 * np.log10(distance_3d) + 21.3*np.log10(frequency) - 0.3 * (user_equipment_height - 1.5)

    return max(path_loss_prime_nlos, path_loss_los(u,v,frequency))

def los_probability(u: Vector, v: Vector) -> float:
    distance_2d: float = euclidean_distance_2d(u,v)
    return 18/distance_2d + np.exp(-(distance_2d/36) * (1-(18/distance_2d))) # TODO: Implement the probability in calculating NLOS/LOS for pathloss.

def path_loss(u: Vector, v: Vector, frequency: float) -> float:
    los_prob = los_probability(u,v)
    rand = np.random.uniform(0,1)
    if rand <= los_prob:
        return path_loss_los(u,v,frequency)
    else:
        return path_loss_nlos(u,v,frequency)

def sinr(u: Vector, v: Vector, frequency: float) -> float:
    return 46 - path_loss(u,v,frequency) - (-174 + 10*np.log10(20e6)) # TODO: Do a more rigorous calculation of interference from neighboring cells?

def rsrp(u: Vector, v: Vector, frequency: float, tx_power: float) -> float:
    return np.round(tx_power - path_loss_los(u,v,frequency), 10)