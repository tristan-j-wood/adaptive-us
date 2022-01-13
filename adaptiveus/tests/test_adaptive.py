import numpy as np
import pytest
import adaptiveus as adp
from adaptiveus.adaptive import Gaussian
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

        assert isinstance(window.gaussian, Gaussian)


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

    with pytest.raises(ValueError):
        adaptive.load('data_0.txt')

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
    assert np.isclose(np.sqrt(2*np.pi), adp.adaptive.area(window.gaussian))

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
    assert not converged

    # Parameters should converged with an excess threshold
    converged = window.gaussian_converged(b_threshold=1000, c_threshold=1000)
    assert converged


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_discrepancy():

    adaptive = adp.adaptive.Windows()

    # Discrepancy should fail if data hasn't been loaded
    for window in adaptive:
        with pytest.raises(ValueError):
            _ = window.discrepancy

    # Plotting should fail if data hasn't been loaded
    with pytest.raises(ValueError):
        adaptive.plot_discrepancy()

    adaptive.load(filename='data_0.txt')
    adaptive.load(filename='data_1.txt')
    adaptive.load(filename='data_2.txt')
    adaptive.load(filename='data_3.txt')

    adaptive.plot_discrepancy()
