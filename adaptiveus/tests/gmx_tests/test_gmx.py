import os
import adaptiveus as adp
from adaptiveus.umbrella import UmbrellaSampling
from adaptiveus.utils import work_in_zipped_dir
here = os.path.abspath(os.path.dirname(__file__))
adp.Config.n_cores = 4


# @work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_gmx_non_adaptive():

    adaptive = UmbrellaSampling(driver='gmx',
                                kappa=1000,
                                temp=300,
                                interval=5,
                                dt=2,
                                traj='pullx.xvg',
                                init_ref=0.515,
                                final_ref=0.7)

    adaptive.run_non_adaptive_sampling(n_windows=4, ps=5)
    #
    # adaptive.run_adaptive_sampling(ps=10)
