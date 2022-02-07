import adaptiveus.adaptive as adp
import numpy as np
from scipy.integrate import quad


def _minimum_gaussian(x, gaussian_1, gaussian_2) -> float:
    """Returns the minimum value of two Gaussians at point x"""
    return min(adp.gaussian_value(x, *gaussian_1.params),
               adp.gaussian_value(x, *gaussian_2.params))


def _integrate_gaussians(gaussian_1, gaussian_2) -> float:
    """
    Integrates the area under the two Gaussians, taking the lowest Gaussian
    as the height of the integrand at each point. Splits the integral
    calculation between the midpoints so avoid scipy quad missing the non-zero
    regions of the integrand
    """
    midpoint = (gaussian_1.params[1] + gaussian_2.params[1]) / 2

    integral_up_to_midpoint = quad(_minimum_gaussian,
                                   -np.inf,
                                   midpoint,
                                   args=(gaussian_1, gaussian_2))

    integral_from_midpoint = quad(_minimum_gaussian,
                                  midpoint,
                                  np.inf,
                                  args=(gaussian_1, gaussian_2))

    # First element is integral, second is error esimate
    total_integral = integral_up_to_midpoint[0] + integral_from_midpoint[0]

    return total_integral


def _fix_overlaps_below_unity(overlap) -> float:
    """
    If the error in the integration produces an overlap just over one, this
    function will reset the overlap to unity
    """
    if overlap > 1:
        assert np.isclose(overlap, 1, atol=1e-3)
        return 1.0

    else:
        return overlap


def calculate_overlaps(gaussian_1: 'adaptiveus.adaptive.Gaussian',
                       gaussian_2: 'adaptiveus.adaptive.Gaussian') -> list:
    """Calculates the fractional overlap between two Gaussians"""
    integral = _integrate_gaussians(gaussian_1, gaussian_2)

    area_1 = adp.area(gaussian_1)
    area_2 = adp.area(gaussian_2)

    norm_overlap_1 = integral / area_1
    norm_overlap_2 = integral / area_2

    norm_overlap_1 = _fix_overlaps_below_unity(norm_overlap_1)
    norm_overlap_2 = _fix_overlaps_below_unity(norm_overlap_2)

    return [norm_overlap_1, norm_overlap_2]
