import os
import adaptiveus as adp
from adaptiveus.utils import work_in_zipped_dir
here = os.path.abspath(os.path.dirname(__file__))
adp.Config.n_cores = 1


# @work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_gmx_non_adaptive():

    gromacs = adp.gmx.GMXAdaptive(kappa=50,
                                  temp=300,
                                  interval=5,
                                  dt=1)

    gromacs.run_md_window(driver='gmx',
                          ref=0.6,
                          idx=1)
