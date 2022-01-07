import numpy as np
from mltrain.sampling.umbrella import UmbrellaSampling
from adaptiveus.adaptive import Windows
from typing import Optional, Callable


class Adaptive:

    def __init__(self,
                 method: str,
                 zeta_func: 'mltrain.sampling.reaction_coord.ReactionCoordinate',
                 kappa: float = 20):
        """
        Perform adaptive umbrella sampling using (currently only) mltrain to
        drive the dynamics

        -----------------------------------------------------------------------
        Arguments:
            method: Name of umbrella sampling driver: 'mltrain', 'gmx'

            zeta_func: Reaction coordinate, as the function of atomic positions

            kappa: Value of the spring constant, κ, used in umbrella sampling
        """

        self.method = method
        # The zeta function is currently set up for mltrain only
        self.zeta_func:         Callable = zeta_func
        self.kappa = kappa

    def _run_single_window(self, ref, config):

        if self.method == 'mlt':
            adaptive = MltrainAdaptive()
            adaptive.run_mlt_adaptive()

            # Get data and stuff from running these and then process?

        return NotImplementedError

    def run_adaptive(self,
                     traj: Optional,  # Coordinate along the reaction coord
                     mlp: Optional['mltrain.potentials._base.MLPotential'],
                     temp: float,
                     interval: int,
                     dt: float,
                     init_ref: Optional[float] = None,
                     final_ref: Optional[float] = None,
                     n_windows: Optional[int] = 10,
                     **kwargs):
        """
        Run adaptive umbrella sampling for ml-train

        -----------------------------------------------------------------------
        Arguments:
            traj: Trajectory from which to initialise the umbrella over, e.g.
                  a 'pulling' trajectory that has sufficient sampling of a
                  range f reaction coordinates

            mlp: Machine learnt potential

            temp: Temperature in K to initialise velocities and to run NVT MD.
                  Must be positive

            interval: Interval between saving the geometry

            dt: Time-step in fs

            init_ref: Value of reaction coordinate in Å for
                      first window

            final_ref: Value of reaction coordinate in Å for
                       first window

            n_windows: Number of windows to run in the umbrella sampling

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units
        """

        if init_ref is None:
            init_ref = self.zeta_func(traj[0])

        self._run_single_window(init_ref, traj)

        # Run 1 window and check stuff. Choose k and then run next window


class MltrainAdaptive:

    def __init__(self,
                 zeta_func: 'mltrain.sampling.reaction_coord.ReactionCoordinate',
                 kappa: float = 20):
        """
        Perform adaptive umbrella sampling using mltrain to drive the dynamics

        -----------------------------------------------------------------------
        Arguments:
            zeta_func: Reaction coordinate, as the function of atomic positions

            kappa: Value of the spring constant, κ, used in umbrella sampling
        """

        self.kappa = kappa
        self.zeta_func:         Callable = zeta_func

    def _run_umbrella_window(self, traj, mlp, temp, interval, dt, ref, idx,
                             **kwargs) -> None:
        """
        Run a single umbrella window using mltrain and save the sampled
        reaction coordinates
        """

        umbrella = UmbrellaSampling(zeta_func=self.zeta_func,
                                    kappa=self.kappa)

        umbrella.run_umbrella_sampling(traj=traj,
                                       mlp=mlp,
                                       temp=temp,
                                       interval=interval,
                                       dt=dt,
                                       init_ref=ref,
                                       final_ref=ref,
                                       n_windows=1,
                                       **kwargs)

        umbrella.windows[0].save(f'window_{idx}.txt')

        return None

    @staticmethod
    def add_window_idx_to_file(idx) -> None:
        """Add the window index to the start of the reaction coordinate file"""

        file_lines = open(f'window_{idx}.txt', 'r').readlines()
        file_lines[0] = f'{idx} {file_lines[0]}'

        with open(f'window_{idx}.txt', 'w') as outfile:
            for line in file_lines:
                print(line.strip('\n'), file=outfile)

        return None

    def _reference_values(self, traj, num, init_ref, final_ref) -> np.ndarray:
        """Return the set of reference values across the reaction coordinate"""

        if init_ref is None:
            init_ref = self.zeta_func(traj[0])

        if final_ref is None:
            final_ref = self.zeta_func(traj[-1])

        return np.linspace(init_ref, final_ref, num)

    def run_mlt_adaptive(self,
                         traj: 'mltrain.ConfigurationSet',
                         mlp: 'mltrain.potentials._base.MLPotential',
                         temp: float,
                         interval: int,
                         dt: float,
                         init_ref: Optional[float] = None,
                         final_ref: Optional[float] = None,
                         n_windows: int = 10,
                         **kwargs) -> None:
        """
        Run adaptive umbrella sampling for ml-train

        -----------------------------------------------------------------------
        Arguments:
            traj: Trajectory from which to initialise the umbrella over, e.g.
                  a 'pulling' trajectory that has sufficient sampling of a
                  range f reaction coordinates

            mlp: Machine learnt potential

            temp: Temperature in K to initialise velocities and to run NVT MD.
                  Must be positive

            interval: (int) Interval between saving the geometry

            dt: (float) Time-step in fs

            init_ref: (float | None) Value of reaction coordinate in Å for
                       first window

            final_ref: (float | None) Value of reaction coordinate in Å for
                       first window

            n_windows: (int) Number of windows to run in the umbrella sampling

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units
        """

        ref_values = self._reference_values(traj, n_windows,
                                            init_ref, final_ref)

        window_idxs = []
        for idx, ref in enumerate(ref_values):
            self._run_umbrella_window(traj=traj,
                                      mlp=mlp,
                                      temp=temp,
                                      interval=interval,
                                      dt=dt,
                                      ref=ref,
                                      idx=idx,
                                      **kwargs)
            window_idxs.append(idx)

        adaptive = Windows()

        for idx in window_idxs:
            self.add_window_idx_to_file(idx)
            # Save to a zip file
            adaptive.load(f'window_{idx}.txt')

        for idx in window_idxs[:-1]:
            adaptive.calculate_overlap([idx, idx+1])

        adaptive.plot_histogram()
        adaptive.plot_overlaps()

        return None
