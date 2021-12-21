import adaptiveus as adp


def test_get_params():

    windows = adp.adaptive.Windows()
    # Load in multiple at a time?
    windows.load(filename='data_0.txt')
    windows.load(filename='data_1.txt')
    windows.load(filename='data_2.txt')
    windows.load(filename='data_3.txt')

    # How to choose between window number and index?
    windows.calculate_overlap(indexes=[0, 1])
    windows.calculate_overlap(indexes=[1, 2])
    windows.calculate_overlap(indexes=[2, 3])
    windows.plot_overlaps()

    params_1 = 1, 1, 1
    params_2 = 1, 2, 0.2

    overlap = adp.adaptive.Overlap(params_1, params_2)
    overlap.calculate_overlap()

    # Maybe write a test that iterates over a huge range of possible Gaussians
