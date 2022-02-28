import adaptiveus as adp
import gromacs as gmx
import MDAnalysis as mda
import numpy as np
import os
from adaptiveus._base import MDDriver
adp.Config.n_cores = 4


class GMXAdaptive(MDDriver):

    def __init__(self,
                 kappa: float,
                 temp: float,
                 interval: int,
                 dt: float,
                 gro_filename:  str = 'structure.gro',
                 top_filename:  str = 'topol.top',
                 mdp_filename:  str = 'md_umbrella.mdp',
                 xtc_filename:  str = 'pull.xtc',
                 pull_filename: str = 'pullx.xvg'):
        """
        Perform adaptive umbrella sampling using gromacs to drive the dynamics

        -----------------------------------------------------------------------
        Arguments:
            kappa: Value of the spring constant, κ, used in umbrella sampling

            temp: Temperature in K to initialise velocities and to run NVT MD.
                  Must be positive

            interval: Interval between saving the geometry

            dt: Time-step in fs

            gro_filename: GROMACS structure file

            top_filename: GROMACS topology file

            mdp_filename: GROMACS parameter file

            xtc_filename: GROMACS trajectory file

            pull_filename: GROMACS pulling (x-coordinate) file

            """
        self.kappa:             float = kappa

        self.temp:              float = temp
        self.interval:          int = interval
        self.dt:                float = dt

        self.gro_filename = gro_filename
        self.top_filename = top_filename
        self.mdp_filename = mdp_filename
        self.mdp_filename = mdp_filename
        self.xtc_filename = xtc_filename
        self.pull_filename = pull_filename

    @staticmethod
    def _get_obs_zetas_from_xvg(filename) -> list:
        """Get the observed reaction coordinates from an xvg formatted file"""
        file_lines = open(filename, 'r').readlines()

        obs_zetas = []
        for line in file_lines:
            if not line[0] == '#' and not line[0] == '@':
                obs_zetas.append(float(line.strip('\n').split('\t')[1]))

        return obs_zetas

    def _write_window_file(self, idx: int, ref: float, kappa: float) -> None:
        """Write the observed zetas, window num, ref and kappa to a file"""
        obs_zetas = self._get_obs_zetas_from_xvg(f'umbrella_{idx}_pullx.xvg')

        with open(f'window_{idx}', 'w') as outfile:
            print(f'{idx} {ref} {kappa}', file=outfile)

            for obs_zeta in obs_zetas:
                print(f'{obs_zeta}', file=outfile)

        return None

    def _select_frame_for_us(self, ref: float, idx: int) -> None:
        """Select a frame from the GROMACS .xtc pulling trajectory with a
        reaction coordinate closest to the given reference. Write this frame
        to a gro file"""
        assert os.path.exists(self.pull_filename)

        obs_zetas = self._get_obs_zetas_from_xvg(self.pull_filename)
        system = mda.Universe(self.gro_filename, self.xtc_filename)

        closest_frame_idx = np.argmin([abs(ref - zeta) for zeta in obs_zetas])
        fraction_along_traj = closest_frame_idx / len(obs_zetas)

        # Get the index of the first frame from the pulling traj
        index_of_traj = int(len(system.trajectory) * fraction_along_traj)

        atoms = system.select_atoms('all')
        atoms.write(f'win_frame_{idx}.gro', frames=[index_of_traj])

        return None

    def _edit_mdp_file(self, ref: float) -> None:
        """Edit the MD parameter file with the updated kappa and ref values"""
        mdp = gmx.fileformats.mdp.MDP(filename=self.mdp_filename)

        # Currently does not modify dt, interval or total time automatically

        mdp['pull_coord1_k'] = self.kappa
        mdp['pull_coord1_start'] = 'no'
        mdp['pull_coord1_init'] = ref

        mdp.write(f'{self.mdp_filename}')

        return None

    def run_md_window(self,
                      ref=None,
                      idx=None,
                      **kwargs) -> None:
        """
        Run a single umbrella window using mltrain and save the sampled
        reaction coordinates

        -----------------------------------------------------------------------
        Arguments:

            ref: Reference value for the harmonic bias

            idx: Index for the umbrella sampling window

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units
        """

        assert ref is not None and idx is not None

        self._select_frame_for_us(ref=ref, idx=idx)

        self._edit_mdp_file(ref)

        gmx.grompp(f=self.mdp_filename,
                   p=self.top_filename,
                   r=f'win_frame_{idx}.gro',
                   c=f'win_frame_{idx}.gro',
                   n='index.ndx',
                   maxwarn='99',
                   o=f'umbrella_{idx}.tpr')

        # What is v?
        gmx.mdrun(v=True, deffnm=f'umbrella_{idx}')

        self._write_window_file(idx=idx, ref=ref, kappa=self.kappa)

        return None

    def calculate_free_energy(self, windows=None, zetas=None):
        """"""
        return NotImplementedError
