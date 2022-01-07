import adaptiveus as adp
import pytest
import os
from adaptiveus.utils import work_in_zipped_dir
here = os.path.abspath(os.path.dirname(__file__))


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_get_params():

    windows = adp.adaptive.Windows()
    # Load in multiple at a time?
    windows.load(filename='data_0.txt')
    windows.load(filename='data_1.txt')
    windows.load(filename='data_2.txt')
    windows.load(filename='data_3.txt')

    # How to choose between window number and index?
    windows.calculate_overlap(idx0=3, idx1=4)
    windows.calculate_overlap(idx0=4, idx1=5)
    windows.calculate_overlap(idx0=5, idx1=6)
    windows.plot_overlaps()

    # Test should fail when specified windows don't exist
    with pytest.raises(StopIteration):
        windows.calculate_overlap(idx0=4, idx1=7)

    params_1 = 1, 1, 1
    params_2 = 1, 2, 0.2

    adp.adaptive.calculate_overlap(params_1, params_2)
