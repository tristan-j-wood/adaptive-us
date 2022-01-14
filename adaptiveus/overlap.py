import adaptiveus.adaptive as adp
import numpy as np
from adaptiveus.log import logger
from scipy.integrate import quad


def _choose_integrand_parameters(gaussian_1, gaussian_2, roots) -> tuple:
    """
    Select the appopriate Gaussians used for the integration. First it tries
    to identify which Gaussian lies under the another between the two roots.
    If unsucessful, it takes the Gaussian with the higher mean as the first
    set of parameters
    """

    midpoint = (min(roots) + max(roots)) / 2

    y_1 = adp.value(midpoint, *gaussian_1.params)
    y_2 = adp.value(midpoint, *gaussian_2.params)

    if y_1 > y_2:
        return gaussian_1, gaussian_2

    elif y_1 < y_2:
        return gaussian_2, gaussian_1

    else:
        if gaussian_2.params[1] > gaussian_1.params[1]:
            return gaussian_2, gaussian_1

        else:
            return gaussian_1, gaussian_2


def _calculate_integral(gaussian_1, gaussian_2, roots) -> float:
    """Calculates the sum of integrals using the intercepts as limits"""

    if len(roots) == 2:
        logger.info(f'Found two real roots: {roots}')

        lower_gaussian, upper_gaussian = _choose_integrand_parameters(
            gaussian_1, gaussian_2, roots)

        a_1, b_1, c_1 = lower_gaussian.params
        a_2, b_2, c_2 = upper_gaussian.params

        lower_int = quad(adp.value, -np.inf, min(roots),
                         args=(a_1, b_1, c_1))
        middle_int = quad(adp.value, min(roots), max(roots),
                          args=(a_2, b_2, c_2))
        upper_int = quad(adp.value, max(roots), np.inf,
                         args=(a_1, b_1, c_1))

        # Zeorth element is the value of the integral
        return lower_int[0] + middle_int[0] + upper_int[0]

    else:
        # There will always be two real roots for sampled data
        raise ValueError("Must be 2 real roots")


def _calculate_intercepts(gaussian_1, gaussian_2) -> np.ndarray:
    """
    Calculates the point(s) of intersection between two Gaussians. Finds the
    roots of the quadratic polynomial equation relating to two intersecting
    Gaussians
    """

    a_1, b_1, c_1 = gaussian_1.params
    a_2, b_2, c_2 = gaussian_2.params

    a = (c_1**2 / c_2**2) - 1
    b = - 2 * ((c_1**2 / c_2**2) * b_2 - b_1)
    c = 2 * c_1**2 * np.log(a_1 / a_2) + (c_1**2 / c_2**2
                                          ) * b_2**2 - b_1**2

    return np.roots([a, b, c])


def calculate_overlaps(gaussian_1: 'adaptiveus.adaptive.Gaussian',
                       gaussian_2: 'adaptiveus.adaptive.Gaussian') -> list:
    """Calculates the fractional overlap between two Gaussians"""

    roots = _calculate_intercepts(gaussian_1, gaussian_2)
    logger.info(f'Intersections of Gaussians at {roots}')

    area_1 = adp.area(gaussian_1)
    area_2 = adp.area(gaussian_2)

    if all(np.isreal(roots)):
        integral = _calculate_integral(gaussian_1, gaussian_2, roots)

        norm_overlap_1 = integral / area_1
        norm_overlap_2 = integral / area_2

        return [norm_overlap_1, norm_overlap_2]

    else:
        # There will always be two real roots for sampled data
        raise ValueError(f"Found no real roots: {roots}")
