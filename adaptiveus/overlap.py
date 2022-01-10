from adaptiveus.log import logger
import numpy as np
from scipy.integrate import quad


def _gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2. * c**2))


def _get_area(a, c):
    return a * c * (2 * np.pi)**0.5


def _choose_integrand_parameters(params_1, params_2, roots):
    """Select the appopriate parameters used for the integration"""

    midpoint = (min(roots) + max(roots)) / 2

    y_1 = _gaussian(midpoint, *params_1)
    y_2 = _gaussian(midpoint, *params_2)

    if y_1 > y_2:
        return params_1, params_2

    elif y_1 < y_2:
        return params_2, params_1

    else:
        return ValueError("Could not determine which parameters to use")


def _calculate_integral(params_1, params_2, roots):
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

    elif len(roots) == 1:
        logger.info(f'Found one real root: {roots}')
        return NotImplementedError

    else:
        return ValueError("Must be either 1 or 2 real roots")


def _calculate_intercepts(params_1, params_2):
    """Calculates the point(s) of intersection between two Gaussians"""

    a_1, b_1, c_1 = params_1
    a_2, b_2, c_2 = params_2

    a = (c_1**2 / c_2**2) - 1
    b = - 2 * ((c_1**2 / c_2**2) * b_2 - b_1)
    c = 2 * c_1**2 * np.log(a_1 / a_2) + (c_1**2 / c_2**2
                                          ) * b_2**2 - b_1**2

    return np.roots([a, b, c])


# def calculate_same_gaussian_overlap(params_1, params_2) -> float:
#     """Calculates the overlap between two Gaussians with identical a and c"""
#
#     a, b_1, c = params_1[0], params_1[1], params_1[2]
#     b_2 = params_2[1]
#
#     intercept = (b_2**2 - b_1**2) / (2 * b_2 - 2 * b_1)
#
#     lower_int = quad(_gaussian, -np.inf, intercept, args=(a, b_2, c))
#     upper_int = quad(_gaussian, intercept, np.inf, args=(a, b_1, c))
#
#     normalisation_area = _get_area(a, c)
#
#     return (lower_int[0] + upper_int[0]) / normalisation_area


def calculate_overlap(params_1, params_2):
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
        logger.info(f'Found no real roots: {roots}')
        # insert stuff for when there are no roots
        return NotImplementedError
