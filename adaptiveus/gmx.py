import matplotlib.pyplot as plt
import adaptiveus as adp
import gromacs as gmx
import MDAnalysis as mda
import numpy as np
import os
from adaptiveus.log import logger
from adaptiveus._base import MDDriver
from typing import Tuple
adp.Config.n_cores = 4


def _get_data_from_xvg(filename) -> dict:
    """Extract the x and y data from an xvg file"""
    file_lines = open(filename, 'r').readlines()

    data = {}
    for line in file_lines:
        if not line[0] == '#' and not line[0] == '@':

            x_vals = float(line.strip('\n').split()[0])
            y_vals = float(line.strip('\n').split()[1])

            data[x_vals] = y_vals

    return data


def _reduce_data(dataset_a, dataset_b) -> Tuple[list, list]:
    """Reduce dataset a and dataset b where they share common keys"""

    reshaped_a, reshaped_b = [], []
    for key in dataset_a.keys() & dataset_b.keys():

        reshaped_a.append(dataset_a[key])
        reshaped_b.append(dataset_b[key])

    return reshaped_a, reshaped_b


def _calculate_total_energy():
    """"""

    os.system("gmx energy -f pull.edr <<< 'Potential'")

    zetas = _get_data_from_xvg('pullx.xvg')
    pot_energies = _get_data_from_xvg(filename='energy.xvg')

    reshaped_zetas, reshaped_pot_energies = _combine_data(zetas, pot_energies)

    kappa, ref = 10000, 3

    bias = _calculate_harmonic_bias(kappa, ref, reshaped_zetas)
    total_energy = [sum(e) for e in zip(bias, reshaped_pot_energies)]

    plt.plot(reshaped_zetas, reshaped_pot_energies)
    plt.plot(reshaped_zetas, bias)
    plt.plot(reshaped_zetas, total_energy)
    plt.savefig('tmp2.pdf')


def _n_simulation_steps(dt: float,
                        kwargs: dict) -> int:
    """Calculate the number of simulation steps from a set of keyword
    arguments e.g. kwargs = {'fs': 100}

    ---------------------------------------------------------------------------
    Arguments:
        dt: Timestep in fs

        kwargs:

    Returns:
        (int): Number of simulation steps to perform
    """
    if dt < 0.09E-3 or dt > 5E-3:
        logger.warning('Unexpectedly small or large timestep - is it in fs?')

    if 'ps' in kwargs:
        time_ps = kwargs['ps']

    elif 'fs' in kwargs:
        time_ps = 1E-3 * kwargs['fs']

    elif 'ns' in kwargs:
        time_ps = 1E3 * kwargs['ns']

    else:
        raise ValueError('Simulation time not found')

    n_steps = max(int(time_ps / dt), 1)                 # Run at least one step

    return n_steps


class GMXAdaptive(MDDriver):

    def __init__(self,
                 kappa: float,
                 temp: float,
                 interval: int,
                 dt: float,
                 pull_filename: str,
                 gro_filename:  str = 'structure.gro',
                 top_filename:  str = 'topol.top',
                 mdp_filename:  str = 'md_umbrella.mdp',
                 xtc_filename:  str = 'pull.xtc'):
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

        self.dt:                float = dt / 1000  # Timestep in ps for gmx
        self.n_steps = None

        self.gro_filename = gro_filename
        self.top_filename = top_filename
        self.mdp_filename = mdp_filename
        self.mdp_filename = mdp_filename
        self.xtc_filename = xtc_filename
        self.pull_filename = pull_filename

    @property
    def zetas(self) -> list:
        """Get the zetas from the gmx pulling xvg file"""
        zetas = _get_data_from_xvg(self.pull_filename)
        energies = _get_data_from_xvg('energy.xvg')

        reduced_zetas, _ = _reduce_data(zetas, energies)

        return reduced_zetas

    def calculate_pot_energy(self) -> list:
        """Gets the potential energy from the gmx energy file"""
        os.system("gmx energy -f pull.edr <<< 'Potential'")

        zetas = _get_data_from_xvg(self.pull_filename)
        energies = _get_data_from_xvg('energy.xvg')

        _, reduced_energies = _reduce_data(zetas, energies)

        return reduced_energies

    def calculate_bias_energy(self, kappas, ref):
        """Calculates the bias energy for configurations in a trajectory"""
        zetas = _get_data_from_xvg(self.pull_filename)
        energies = _get_data_from_xvg('energy.xvg')

        reduced_zetas, _ = _reduce_data(zetas, energies)
        print(kappas)
        print(reduced_zetas)

        return [(kappas / 2) * (zeta - ref)**2 for zeta in reduced_zetas]

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

        with open(f'window_{idx}.txt', 'w') as outfile:
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

    def _edit_mdp_file(self, ref: float, idx: int) -> None:
        """Edit the MD parameter file with the updated kappa and ref values"""
        mdp = gmx.fileformats.mdp.MDP(filename=self.mdp_filename)

        # Think about the units for kappa
        mdp['pull_coord1_k'] = self.kappa

        mdp['pull_coord1_start'] = 'no'
        mdp['pull_coord1_init'] = ref

        mdp['pull_nstxout'] = 1
        mdp['pull_pbc_ref_prev_step_com'] = 'yes'

        mdp['dt'] = self.dt
        mdp['nsteps'] = self.n_steps

        mdp.write(f'{self.mdp_filename.replace(".mdp", f"_{idx}.mdp")}')

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

            traj: Gromacs pulling xvg filename

            ref: Reference value for the harmonic bias

            idx: Index for the umbrella sampling window

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units
        """
        assert ref is not None and idx is not None

        self.n_steps = _n_simulation_steps(self.dt, kwargs)

        self._select_frame_for_us(ref=ref, idx=idx)

        self._edit_mdp_file(ref, idx=idx)

        gmx.grompp(f=f'{self.mdp_filename.replace(".mdp", f"_{idx}.mdp")}',
                   p=self.top_filename,
                   r=f'win_frame_{idx}.gro',
                   c=f'win_frame_{idx}.gro',
                   n='index.ndx',
                   maxwarn='99',
                   o=f'umbrella_{idx}.tpr')

        # What is v?
        gmx.mdrun(v=True, deffnm=f'umbrella_{idx}', s=f'umbrella_{idx}')

        self._write_window_file(idx=idx, ref=ref, kappa=self.kappa)

        return None

    def calculate_free_energy(self, windows=None, zetas=None
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """"""
        idxs = [window.window_n for window in windows]

        with open('tpr-files.dat', 'w') as outfile:
            for idx in idxs:
                print(f'umbrella_{idx}.tpr', file=outfile)

        with open('pullx-files.dat', 'w') as outfile:
            for idx in idxs:
                print(f'umbrella_{idx}_pullx.xvg', file=outfile)

        os.system(f"gmx wham -it tpr-files.dat -ix pullx-files.dat "
                  f"-o -hist -unit kCal -b 0 -temp {self.temp} -dt {self.dt}")

        file_lines = open('profile.xvg', 'r').readlines()

        xs, ys = [], []
        for line in file_lines:
            if not line[0] == '#' and not line[0] == '@':
                xs.append(float(line.strip('\n').split('\t')[0]))
                ys.append(float(line.strip('\n').split('\t')[1]))

        plt.plot(xs, ys)
        plt.savefig('tmp.pdf')

        return np.asarray(xs), np.asarray(ys)
