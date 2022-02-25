import os
import adaptiveus as adp
from adaptiveus.utils import work_in_zipped_dir
here = os.path.abspath(os.path.dirname(__file__))
adp.Config.n_cores = 4


# @work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_gmx_non_adaptive():

    gromacs = adp.gmx.GMXAdaptive(zeta_func=None,
                                  kappa=100,
                                  temp=None,
                                  interval=None,
                                  dt=None)

    gromacs.run_md_window(traj=None,
                          driver='gmx',
                          ref=0.548899,
                          idx=2)
