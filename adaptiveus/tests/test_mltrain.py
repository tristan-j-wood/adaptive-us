import os
import mltrain as mlt
import adaptiveus as adp
import numpy as np
from adaptiveus.umbrella import UmbrellaSampling
from mltrain.sampling.reaction_coord import AverageDistance
from adaptiveus.utils import work_in_zipped_dir
here = os.path.abspath(os.path.dirname(__file__))
adp.Config.n_cores = 4


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def load_files():

    # Load in mltrain files
    traj = mlt.ConfigurationSet()
    traj.load_xyz(filename='rxn_coord.xyz', charge=-1, mult=1)
    traj[0].save_xyz(filename='init.xyz')
    system = mlt.System(mlt.Molecule('init.xyz', charge=-1, mult=1),
                        box=[10, 10, 10])
    gap = mlt.potentials.GAP('potential', system=system)

    return traj, gap


traj, gap = load_files()


# @work_in_zipped_dir(os.path.join(here, 'data.zip'))
# def test_mltrain_non_adaptive():
#
#     # Initialise UmbrellaSampling class
#     adaptive = UmbrellaSampling(traj=traj,
#                                 driver=gap,
#                                 zeta_func=AverageDistance([0, 1]),
#                                 kappa=100,
#                                 temp=300,
#                                 interval=5,
#                                 dt=0.5)
#
#     assert adaptive.zeta_func is not None
#     assert adaptive.driver is not None
#
#     # Run umbrella sampling without any adaptive and convergence testing
#     adaptive.run_non_adaptive_sampling(n_windows=10,
#                                        fs=3000)
#
#     assert os.path.exists('param_conv_0.pdf')
#     assert os.path.exists('gaussian_conv_0.pdf')
#
#     assert adaptive.windows[1].lhs_overlap is not None
#     assert adaptive.windows[1].rhs_overlap is not None
#
#     assert len(adaptive.windows) == 10
#     # assert np.isclose(adaptive.windows[0].zeta_ref, 2)
#     # assert np.isclose(adaptive.windows[-1].zeta_ref, 2.2)
#
#     assert os.path.exists('overlap.pdf')
#     assert os.path.exists('window_histogram.pdf')
#     assert os.path.exists('discrepancy.pdf')
#
#     adaptive.calculate_free_energy()


# @work_in_zipped_dir(os.path.join(here, 'data.zip'))
# def test_mltrain_adaptive():
#
#     # Initialise UmbrellaSampling class
#     adaptive = UmbrellaSampling(traj=traj,
#                                 driver=gap,
#                                 zeta_func=AverageDistance([0, 1]),
#                                 kappa=100,
#                                 temp=300,
#                                 interval=5,
#                                 dt=0.5,
#                                 init_ref=2,
#                                 final_ref=2.2)
#
#     # Run umbrella sampling without any adaptive and convergence testing
#     adaptive.run_adaptive_sampling(fs=2000)
#
#     assert adaptive.windows[1].zeta_ref is not None
#     assert len(adaptive.windows) > 0
#
#     assert os.path.exists('overlap.pdf')
#     assert os.path.exists('window_histogram.pdf')
#     assert os.path.exists('discrepancy.pdf')
#
#
# def test_overlap_error_func():
#
#     # Initialise UmbrellaSampling class
#     adaptive = UmbrellaSampling(traj=traj,
#                                 driver=gap,
#                                 zeta_func=AverageDistance([0, 1]),
#                                 kappa=100,
#                                 temp=300,
#                                 interval=5,
#                                 dt=0.5,
#                                 init_ref=2,
#                                 final_ref=2.2)
#
#     # Output of this function with these parameters should be zero
#     output = adaptive._overlap_error_func(x=1.135, s=0.5, b=1, c=0.1)
#     assert np.isclose(output, 0, atol=1e-3)
#
#
# @work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_adjust_kappa():

    # Initialise UmbrellaSampling class
    adaptive = UmbrellaSampling(traj=traj,
                                driver=gap,
                                zeta_func=AverageDistance([0, 1]),
                                kappa=50,
                                temp=300,
                                interval=5,
                                dt=0.5)

    adaptive.run_adaptive_sampling(fs=1000)
