import numpy as np

class Vector:
    """
    Class for vector in R^3
    """
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

c = 3e8 # Speed of light

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
    return np.sqrt((u.x-v.x)**2 + (u.z-v.z)**2)
