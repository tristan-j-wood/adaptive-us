import numpy as np
from mltrain.sampling.umbrella import UmbrellaSampling as MltrainUS
from adaptiveus._base import MDDriver
from adaptiveus.bias import Bias
from typing import Optional, Callable, Tuple


class MltrainAdaptive(MDDriver):

    def __init__(self,
                 zeta_func: 'mltrain.sampling.reaction_coord.ReactionCoordinate',
                 traj: 'mltrain.sampling.md.Trajectory',
                 mlp: 'mltrain.potentials._base.MLPotential',
                 kappa: float,
                 temp: float,
                 interval: int,
                 dt: float):
        """
        Perform adaptive umbrella sampling using mltrain to drive the dynamics

        -----------------------------------------------------------------------
        Arguments:
            zeta_func: Reaction coordinate, as the function of atomic positions

            traj: Trajectory from which to initialise the umbrella over, e.g.
                  a 'pulling' trajectory that has sufficient sampling of a
                  range f reaction coordinates


            mlp: Machine learnt potential

            kappa: Value of the spring constant, κ, used in umbrella sampling

            temp: Temperature in K to initialise velocities and to run NVT MD.
                  Must be positive

            interval: (int) Interval between saving the geometry

            dt: (float) Time-step in fs
        """
        self.zeta_func:         Optional[Callable] = zeta_func
        self.kappa:             float = kappa

        self.temp:              float = temp
        self.interval:          int = interval
        self.dt:                float = dt

        self.mlp = mlp
        self.traj = traj

    @property
    def zetas(self):
        """Get the zetas from the mltrain trajectory and zeta function"""
        return self.zeta_func(self.traj)

    def calculate_pot_energy(self):
        """"""
        self.mlp.predict(self.traj)

        energies = [config.energy.predicted for config in self.traj]

        # Check this
        return [energy - min(energies) for energy in energies]

    def calculate_bias_energy(self, kappas, ref):
        """Calculates the bias energy for configurations in a trajectory"""
        bias = Bias(self.zeta_func, kappa=kappas, reference=ref)
        return bias(self.traj)

    def run_md_window(self,
                      ref: float,
                      idx: int,
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
        umbrella = MltrainUS(zeta_func=self.zeta_func, kappa=self.kappa)

        umbrella.run_umbrella_sampling(traj=self.traj,
                                       mlp=self.mlp,
                                       temp=self.temp,
                                       interval=self.interval,
                                       dt=self.dt,
                                       init_ref=ref,
                                       final_ref=ref,
                                       n_windows=1,
                                       **kwargs)

        umbrella.windows[0].save(f'window_{idx}.txt')

        self._add_window_idx_to_file(idx)

        return None

    @staticmethod
    def _add_window_idx_to_file(idx) -> None:
        """Add the window index to the start of the reaction coordinate file"""

        file_lines = open(f'window_{idx}.txt', 'r').readlines()
        file_lines[0] = f'{idx} {file_lines[0]}'

        with open(f'window_{idx}.txt', 'w') as outfile:
            for line in file_lines:
                print(line.strip('\n'), file=outfile)

        return None

    def calculate_free_energy(self,
                              windows: 'adaptiveus.adaptive.Windows',
                              zetas: np.ndarray,
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the free energy using WHAM in mltrain"""
        umbrella = MltrainUS(zeta_func=self.zeta_func,
                             kappa=self.kappa,
                             temp=self.temp)

        [window.bin(zetas=zetas) for window in windows]
        umbrella.windows = windows

        free_energies = umbrella.wham()

        return free_energies
