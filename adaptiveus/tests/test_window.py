import numpy as np
import pytest
import adaptiveus as adp
from adaptiveus.adaptive import _Gaussian
import os


def test_window():

    adaptive = adp.adaptive.Window()

    # All attributes should be None when no data is loaded
    assert adaptive.window_num is None
    assert adaptive.ref_zeta is None
    assert adaptive.kappa is None
    assert adaptive.obs_zetas is None

    assert isinstance(adaptive.gaussian, _Gaussian)

    # Plotting method should raise exception when no data is loaded
    with pytest.raises(ValueError):
        adaptive.plot_data()

    adaptive.load(filename='data.txt')

    # All attributes should not be None when data is loaded
    assert type(adaptive.window_num) is int
    assert type(adaptive.kappa) is float
    assert type(adaptive.ref_zeta) is float
    assert type(adaptive.obs_zetas) is list

    # Plotting method should work when data is loaded
    adaptive.plot_data()

    assert all(adaptive.gaussian.params)
    assert os.path.exists('fitted_data.pdf')

    with pytest.raises(ValueError):
        adaptive.load(filename='bad_data.txt')


def test_gaussian():

    adaptive = adp.adaptive._Gaussian()
    assert not all(adaptive.params)

    adaptive.params = 1, 1, 1

    # Area of this Gaussian should be √(2π)
    assert np.isclose(np.sqrt(2*np.pi), adaptive.area)

    # Value of the Gaussian at x = 3 should be the following
    assert np.isclose(adaptive(3), 0.1353352832)
#


def test_convergence():

    adaptive = adp.adaptive.Window()
    adaptive.load(filename='data.txt')

    adaptive.convergence_of_gaussian()

    assert os.path.exists('param_conv.pdf')
