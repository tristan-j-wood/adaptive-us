import adaptiveus as adp
import pytest
import os
import numpy as np
from adaptiveus.utils import work_in_zipped_dir
here = os.path.abspath(os.path.dirname(__file__))


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_get_params():

    windows = adp.adaptive.Windows()
    [windows.load(filename=f'data_{i}.txt') for i in range(4)]
    [windows.calculate_overlap(idx0=i, idx1=i+1) for i in [3, 4, 5]]

    windows.plot_overlaps()

    # Test should fail when specified windows don't exist
    with pytest.raises(StopIteration):
        windows.calculate_overlap(idx0=4, idx1=7)

    gaussian_1, gaussian_2 = adp.adaptive.Gaussian(), adp.adaptive.Gaussian()

    gaussian_1.params = 1, 1, 1
    gaussian_2.params = 1, 2, 0.2

    s_1, s_2 = adp.overlap.calculate_overlaps(gaussian_1, gaussian_2)

    assert np.isclose(s_1, 0.158, atol=1e-3)
    assert np.isclose(s_2, 0.791, atol=1e-3)



