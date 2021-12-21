import numpy as np
import pytest
import adaptiveus as adp
from adaptiveus.adaptive import _Gaussian
import os


def test_empty_window():

    adaptive = adp.adaptive.Windows()

    # Window class should contain no windows
    assert not len(adaptive)

    with pytest.raises(ValueError):
        adaptive.plot_histogram()

    # Add empty window to Windows
    tmp_window = adp.adaptive._Window()
    adaptive = adaptive + tmp_window

    assert len(adaptive)

    with pytest.raises(ValueError):
        adaptive.plot_histogram()

    # All attributes should be None when no data is loaded
    for window in adaptive:
        assert window.window_n is None
        assert window.zeta_ref is None
        assert window.kappa is None
        assert window.obs_zetas is None

        assert isinstance(window.gaussian, _Gaussian)


def test_loaded_window():

    adaptive = adp.adaptive.Windows()
    [adaptive.load(f'data_{i}.txt') for i in range(4)]

    # Gaussians are not fitted until requested so all parameters are None
    for window in adaptive:
        assert not all(window.gaussian.params)

    # All windows should be plotted
    adaptive.plot_histogram()
    assert os.path.exists('window_histogram.pdf')

    # Only window 6 should be plotted (which is the first loaded window)
    adaptive.plot_histogram(indexes=[0])
    assert os.path.exists('window_histogram_6.pdf')

    # IndexError raised if indices are specified which are not in Windows
    with pytest.raises(IndexError):
        adaptive.plot_histogram(indexes=[1, 6])

    # Incorrectly formatted files should raise a ValueError
    adaptive = adp.adaptive.Windows()
    with pytest.raises(ValueError):
        adaptive.load(filename='bad_data.txt')


def test_gaussian():

    adaptive = adp.adaptive.Windows()
    adaptive.load(f'data_0.txt')

    window = adaptive[0]
    assert not all(window.gaussian.params)

    # Fit a Gaussian to the real data
    window.fit_gaussian()
    assert all(window.gaussian.params)

    # Set up a toy Gaussian
    window.gaussian.params = 1, 1, 1

    # Area of this Gaussian should be √(2π)
    assert np.isclose(np.sqrt(2*np.pi), window.gaussian.area)

    # Value of the Gaussian at x = 3 should be the following
    assert np.isclose(window.gaussian(3), 0.1353352832)


def test_convergence():

    adaptive = adp.adaptive.Windows()
    adaptive.load(f'data_0.txt')
    window = adaptive[0]

    window.gaussian_converged()
    assert os.path.exists(f'param_conv_6.pdf')
    assert os.path.exists(f'gaussian_conv_6.pdf')

    # Parameters should not converge with an impossible threshold
    converged = window.gaussian_converged(b_threshold=0, c_threshold=0)
    assert not converged

    # Parameters should converged with an excess threshold
    converged = window.gaussian_converged(b_threshold=1000, c_threshold=1000)
    assert converged
