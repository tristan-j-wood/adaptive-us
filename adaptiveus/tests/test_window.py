import numpy as np
import pytest
import adaptiveus as adp
from adaptiveus.adaptive import _Gaussian
import os
from adaptiveus.utils import work_in_zipped_dir
here = os.path.abspath(os.path.dirname(__file__))


def test_empty_window():

    adaptive = adp.adaptive.Windows()

    # Window class should contain no windows
    assert not len(adaptive)

    with pytest.raises(ValueError):
        adaptive.plot_histogram()

    # Add empty window to Windows
    tmp_window = adp.adaptive.Window()
    adaptive.append(tmp_window)

    assert len(adaptive)

    with pytest.raises(ValueError):
        adaptive.plot_histogram()

    # All attributes should be None when no data is loaded
    for window in adaptive:
        assert window.window_n is None
        assert window.zeta_ref is None
        assert window.kappa is None
        assert not len(window.obs_zetas)

        assert isinstance(window.gaussian, _Gaussian)


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
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
    assert os.path.exists('window_histogram_3.pdf')

    # IndexError raised if indices are specified which are not in Windows
    with pytest.raises(IndexError):
        adaptive.plot_histogram(indexes=[1, 6])

    # Incorrectly formatted files should raise a ValueError
    adaptive = adp.adaptive.Windows()
    with pytest.raises(ValueError):
        adaptive.load(filename='bad_data.txt')


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
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


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_convergence():

    adaptive = adp.adaptive.Windows()
    adaptive.load(f'data_0.txt')
    window = adaptive[0]

    window.gaussian_converged()
    assert os.path.exists(f'param_conv_3.pdf')
    assert os.path.exists(f'gaussian_conv_3.pdf')

    # Parameters should not converge with an impossible threshold
    converged = window.gaussian_converged(b_threshold=0, c_threshold=0)
    assert not all(converged)

    # Parameters should converged with an excess threshold
    converged = window.gaussian_converged(b_threshold=1000, c_threshold=1000)
    assert all(converged)


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_discrepancy():

    adaptive = adp.adaptive.Windows()
    adaptive.load(filename='data_0.txt')
    adaptive.load(filename='data_1.txt')
    adaptive.load(filename='data_2.txt')
    adaptive.load(filename='data_3.txt')

    # Plotting should fail if discprepancy hasn't been calculated
    with pytest.raises(AssertionError):
        adaptive.plot_discrepancy()

    adaptive.calculate_discrepancy(idx=3)
    adaptive.calculate_discrepancy(idx=4)
    adaptive.calculate_discrepancy(idx=5)
    adaptive.calculate_discrepancy(idx=6)

    adaptive.plot_discrepancy()
