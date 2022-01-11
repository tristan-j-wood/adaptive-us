from adaptiveus.log import logger
import numpy as np
from scipy.integrate import quad


def _gaussian(x, a, b, c) -> float:
    """Value of the Gaussian at point x"""
    return a * np.exp(-(x - b)**2 / (2. * c**2))


def _get_area(a, c) -> float:
    """
    Returns integral of Gaussian between -∞ and ∞:

    I = ∫ dx a * exp(-(x-b)^2 / (2*c^2)) = ac √(2π)
    """
    return a * c * (2 * np.pi)**0.5


def _choose_integrand_parameters(params_1, params_2, roots) -> tuple:
    """
    Select the appopriate parameters used for the integration. First it tries
    to identify which Gaussian lies under the another between the two roots.
    If unsucessful, it takes the Gaussian with the higher mean as the first
    set of parameters
    """

    midpoint = (min(roots) + max(roots)) / 2

    y_1 = _gaussian(midpoint, *params_1)
    y_2 = _gaussian(midpoint, *params_2)

    if y_1 > y_2:
        return params_1, params_2

    elif y_1 < y_2:
        return params_2, params_1

    else:
        if params_2[1] > params_1[1]:
            return params_2, params_1

        else:
            return params_1, params_2


def _calculate_integral(params_1, params_2, roots) -> float:
    """Calculates the sum of integrals using the intercepts as limits"""

    if len(roots) == 2:
        logger.info(f'Found two real roots: {roots}')

        lower, upper = _choose_integrand_parameters(params_1, params_2, roots)
        a_1, b_1, c_1 = lower
        a_2, b_2, c_2 = upper

        lower_int = quad(_gaussian, -np.inf, min(roots),
                         args=(a_1, b_1, c_1))
        middle_int = quad(_gaussian, min(roots), max(roots),
                          args=(a_2, b_2, c_2))
        upper_int = quad(_gaussian, max(roots), np.inf,
                         args=(a_1, b_1, c_1))

        return lower_int[0] + middle_int[0] + upper_int[0]

    else:
        # There will always be two real roots for sampled data
        raise ValueError("Must be 2 real roots")


def _calculate_intercepts(params_1, params_2) -> np.ndarray:
    """
    Calculates the point(s) of intersection between two Gaussians. Finds the
    roots of the quadratic polynomial equation relating to two intersecting
    Gaussians
    """

    a_1, b_1, c_1 = params_1
    a_2, b_2, c_2 = params_2

    a = (c_1**2 / c_2**2) - 1
    b = - 2 * ((c_1**2 / c_2**2) * b_2 - b_1)
    c = 2 * c_1**2 * np.log(a_1 / a_2) + (c_1**2 / c_2**2
                                          ) * b_2**2 - b_1**2

    return np.roots([a, b, c])


def calculate_overlap(params_1, params_2) -> list:
    """Calculates the fractional overlap between two Gaussians"""

    roots = _calculate_intercepts(params_1, params_2)
    logger.info(f'Intersections of Gaussians at {roots}')

    area_1 = _get_area(params_1[0], params_1[2])
    area_2 = _get_area(params_2[0], params_2[2])

    if all(np.isreal(roots)):
        integral = _calculate_integral(params_1, params_2, roots)

        norm_overlap_1 = integral / area_1
        norm_overlap_2 = integral / area_2

        return [norm_overlap_1, norm_overlap_2]

    else:
        # There will always be two real roots for sampled data
        raise ValueError(f"Found no real roots: {roots}")
