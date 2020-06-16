import numpy as np

from sklearn.neighbors import NearestNeighbors


def distance(x, y, p_norm=2):
    return np.sum(np.abs(x - y) ** p_norm) ** (1 / p_norm)


def sample_inside_sphere(dimensionality, radius, p_norm=2):
    direction_unit_vector = (2 * np.random.rand(dimensionality) - 1)
    direction_unit_vector = direction_unit_vector / distance(direction_unit_vector, np.zeros(dimensionality), p_norm)

    return direction_unit_vector * np.random.rand() * radius


def rbf(d, gamma):
    if gamma == 0.0:
        return 0.0
    else:
        return np.exp(-(d / gamma) ** 2)


def rbf_score(point, majority_points, gamma, p_norm=2):
    result = 0.0

    for majority_point in majority_points:
        result += rbf(distance(point, majority_point, p_norm), gamma)

    return result
