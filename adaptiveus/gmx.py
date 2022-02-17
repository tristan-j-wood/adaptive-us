import adaptiveus as adp
import gromacs as gmx
import os
from typing import Callable
from adaptiveus._base import MDDriver
adp.Config.n_cores = 4


class GMXAdaptive(MDDriver):

    def __init__(self,
                 zeta_func,
                 kappa: float,
                 temp: float,
                 interval: int,
                 dt: float,
                 gro_filename: str = 'structure.gro',
                 top_filename: str = 'topol.top',
                 mdp_filename: str = 'umbrella.mdp',
                 xtc_filename: str = 'trajectory.xtc'):
        """
        Perform adaptive umbrella sampling using gromacs to drive the dynamics

        -----------------------------------------------------------------------
        Arguments:
            zeta_func: Reaction coordinate, as the function of atomic positions

            kappa: Value of the spring constant, κ, used in umbrella sampling

            temp: Temperature in K to initialise velocities and to run NVT MD.
                  Must be positive

            interval: Interval between saving the geometry

            dt: Time-step in fs

            gro_filename: GROMACS structure file

            top_filename: GROMACS topology file

            mdp_filename: GROMACS parameter file

            xtc_filename: GROMACS trajectory file

            """
        self.zeta_func:         Callable = zeta_func
        self.kappa:             float = kappa

        self.temp:              float = temp
        self.interval:          int = interval
        self.dt:                float = dt

        self.gro_filename = gro_filename
        self.top_filename = top_filename
        self.mdp_filename = mdp_filename
        self.mdp_filename = mdp_filename
        self.xtc_filename = xtc_filename

    @staticmethod
    def _get_obs_zetas_from_xvg(filename) -> tuple:
        """"""
        file_lines = open(filename, 'r').readlines()

        x_values, obs_zetas = [], []
        for line in file_lines:
            if not line[0] == '#' and not line[0] == '@':
                x_values.append(float(line.strip('\n').split('\t')[0]))
                obs_zetas.append(float(line.strip('\n').split('\t')[1]))

        return x_values, obs_zetas

    def _write_window_file(self, win_n, ref, kappa) -> None:
        """"""
        obs_zetas = self._get_obs_zetas_from_xvg(f'umbrella_{win_n}_pullx.xvg')

        with open(f'window_{win_n}', 'w') as outfile:
            print(f'{win_n} {ref} {kappa}', file=outfile)

            for obs_zeta in obs_zetas:
                print(f'{obs_zeta}', file=outfile)

        return None

    def _select_frame_for_us(self, ref, pulling_filename):
        """"""

        times, obs_zetas = self._get_obs_zetas_from_xvg(pulling_filename)
            
        # Get the zetas from the pulling file

        # Find the time associated with a specific reference (find closest)

        # Return the fraction along the trajectory at which this ref lies
        # Could also check this reference is close enough

        # Get the frame associated with this fraction

        return NotImplementedError

    def _edit_mdp_file(self, ref, idx):
        """"""
        mdp = gmx.fileformats.mdp.MDP(filename=self.mdp_filename)

        mdp['pull_coord1_k'] = self.kappa
        mdp['pull-coord1-start'] = 'no'
        mdp['pull-coord1-init'] = ref

        mdp.write(f'{self.mdp_filename}_{idx}')

        return NotImplementedError

    def run_md_window(self,
                      traj,
                      driver='gmx',
                      ref=None,
                      idx=None,
                      **kwargs):
        """
        Run a single umbrella window using mltrain and save the sampled
        reaction coordinates

        -----------------------------------------------------------------------
        Arguments:
            traj: Trajectory from which to initialise the umbrella over, e.g.
                  a 'pulling' trajectory that has sufficient sampling of a
                  range f reaction coordinates

            driver: GROMACS

            ref: Reference value for the harmonic bias

            idx: Index for the umbrella sampling window

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units
        """

        self._select_frame_for_us(ref=ref)

        self._edit_mdp_file(ref, idx)

        gmx.grompp(f=self.mdp_filename,
                   p=self.top_filename,
                   r=self.gro_filename,
                   o=f'umbrella_{idx}.tpr')

        # What is v?
        gmx.mdrun(v=True, deffnm=f'umbrella_{idx}')

        self._write_window_file(win_n=idx, ref=ref, kappa=self.kappa)

        return NotImplementedError

    def calculate_free_energy(self,
                              windows=None,
                              zetas=None):
        """"""

        return NotImplementedError
