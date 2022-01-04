import adaptiveus as adp
from adaptiveus.utils import work_in_zipped_dir
import os
here = os.path.abspath(os.path.dirname(__file__))


# @work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_get_params():

    windows = adp.adaptive.Windows()
    # Load in multiple at a time?
    windows.load(filename='data_0.txt')
    windows.load(filename='data_1.txt')
    windows.load(filename='data_2.txt')
    windows.load(filename='data_3.txt')

    # How to choose between window number and index?
    windows.calculate_overlap(indexes=[3, 4])
    windows.calculate_overlap(indexes=[4, 5])
    windows.calculate_overlap(indexes=[5, 6])
    windows.plot_overlaps()

    params_1 = 1, 1, 1
    params_2 = 1, 2, 0.2

    overlap = adp.adaptive.Overlap(params_1, params_2)
    overlap.calculate_overlap()

    # Maybe write a test that iterates over a huge range of possible Gaussians
