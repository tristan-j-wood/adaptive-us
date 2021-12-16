import adaptiveus as adp


def test_load_data():

    adaptive = adp.adaptive.Window()

    adaptive.load(filename='input.txt')

    assert type(adaptive.window_num) is int
    assert type(adaptive.kappa) is float
    assert type(adaptive.ref_zeta) is float
    assert type(adaptive.obs_zeta) is list

    print(f'{adaptive.window_num}',
          f'{adaptive.kappa}',
          f'{adaptive.ref_zeta}',
          f'{adaptive.obs_zeta}', sep='\n')
