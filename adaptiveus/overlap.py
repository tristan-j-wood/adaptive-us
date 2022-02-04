import adaptiveus.adaptive as adp
import numpy as np
from adaptiveus.log import logger
from scipy.integrate import quad


def minimum_gaussian(x, gaussian_1, gaussian_2) -> float:
    """Returns the minimum value of two Gaussians at point x"""
    return min(adp.gaussian_value(x, *gaussian_1.params),
               adp.gaussian_value(x, *gaussian_2.params))


def calculate_overlaps(gaussian_1: 'adaptiveus.adaptive.Gaussian',
                       gaussian_2: 'adaptiveus.adaptive.Gaussian') -> list:
    """Calculates the fractional overlap between two Gaussians"""

    integral = quad(minimum_gaussian,
                    -np.inf,
                    np.inf,
                    args=(gaussian_1, gaussian_2))

    area_1 = adp.area(gaussian_1)
    area_2 = adp.area(gaussian_2)

    norm_overlap_1 = integral[0] / area_1
    norm_overlap_2 = integral[0] / area_2

    assert norm_overlap_1 <= 1 and norm_overlap_2 <= 1

    return [norm_overlap_1, norm_overlap_2]
