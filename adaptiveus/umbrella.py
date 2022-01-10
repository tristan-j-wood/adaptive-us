from mltrain.sampling.umbrella import UmbrellaSampling
from adaptiveus.adaptive import Windows
from typing import Optional, Callable


class Adaptive:

    def __init__(self,
                 method: str,
                 mlp: Optional['mltrain.potentials._base.MLPotential'],
                 zeta_func: 'mltrain.sampling.reaction_coord.ReactionCoordinate',
                 kappa: float,
                 temp: float,
                 interval: int,
                 dt: float):
        """
        Perform adaptive umbrella sampling using (currently only) mltrain to
        drive the dynamics

        -----------------------------------------------------------------------
        Arguments:
            method: Name of umbrella sampling driver: 'mlt', 'gmx'

            mlp: Machine learnt potential

            zeta_func: Reaction coordinate, as the function of atomic positions

            kappa: Value of the spring constant, κ, used in umbrella sampling

            temp: Temperature in K to initialise velocities and to run NVT MD.
                  Must be positive

            interval: (int) Interval between saving the geometry

            dt: (float) Time-step in fs
        """

        self.method = method
        self.mlp = mlp
        # The zeta function is currently set up for mltrain only
        self.zeta_func:         Callable = zeta_func
        self.kappa = kappa

        self.temp = temp
        self.interval = interval
        self.dt = dt

        self.windows = Windows()

    def _run_single_window(self, ref, traj, idx, **kwargs) -> None:
        """Run a single umbrella sampling window using a specified method"""

        if self.method == 'mlt':
            adaptive = MltrainAdaptive(zeta_func=self.zeta_func,
                                       kappa=self.kappa,
                                       temp=self.temp,
                                       interval=self.interval,
                                       dt=self.dt)

            adaptive.run_mlt_window(traj=traj, mlp=self.mlp, ref=ref, idx=idx,
                                    **kwargs)

            self.windows.load(f'window_{idx}.txt')

        else:
            raise NotImplementedError

        return None

    def run_adaptive(self,
                     traj: Optional,
                     init_ref: Optional[float] = None,
                     final_ref: Optional[float] = None,
                     **kwargs):
        """
        Run adaptive umbrella sampling for ml-train

        -----------------------------------------------------------------------
        Arguments:
            traj: Trajectory from which to initialise the umbrella over, e.g.
                  a 'pulling' trajectory that has sufficient sampling of a
                  range f reaction coordinates

            init_ref: Value of reaction coordinate in Å for
                      first window

            final_ref: Value of reaction coordinate in Å for
                       first window

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units
        """

        if init_ref is None:
            init_ref = self.zeta_func(traj[0])

        self._run_single_window(init_ref, traj, idx=0, **kwargs)

        # disc = self.windows[0].discrepancy

        # adaptive.calculate_overlap([idx, idx+1])
        #
        # adaptive.plot_histogram()
        # adaptive.plot_overlaps()

        # Run 1 window and check stuff. Choose k and then run next window

        # while ref < final_ref:
        #
        #     self._run_single_window(new_ref, traj)
        #
        #     calculate_overlap(idx_0, idx_1)
        #
        #     calculate_d()
        #
        #     calculate_new_k_and_ref()


class MltrainAdaptive:

    def __init__(self,
                 zeta_func: 'mltrain.sampling.reaction_coord.ReactionCoordinate',
                 kappa: float,
                 temp: float,
                 interval: int,
                 dt: float):
        """
        Perform adaptive umbrella sampling using mltrain to drive the dynamics

        -----------------------------------------------------------------------
        Arguments:
            zeta_func: Reaction coordinate, as the function of atomic positions

            kappa: Value of the spring constant, κ, used in umbrella sampling

            temp: Temperature in K to initialise velocities and to run NVT MD.
                  Must be positive

            interval: (int) Interval between saving the geometry

            dt: (float) Time-step in fs
        """

        self.kappa = kappa
        self.zeta_func:         Callable = zeta_func

        self.temp = temp
        self.interval = interval
        self.dt = dt

    def run_mlt_window(self, traj, mlp, ref, idx, **kwargs) -> None:
        """
        Run a single umbrella window using mltrain and save the sampled
        reaction coordinates

        -----------------------------------------------------------------------
        Arguments:
            traj: Trajectory from which to initialise the umbrella over, e.g.
                  a 'pulling' trajectory that has sufficient sampling of a
                  range f reaction coordinates

            mlp: Machine learnt potential

            ref: Reference value for the harmonic bias

            idx: Index for the umbrella sampling window

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units
        """

        umbrella = UmbrellaSampling(zeta_func=self.zeta_func,
                                    kappa=self.kappa)

        umbrella.run_umbrella_sampling(traj=traj,
                                       mlp=mlp,
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
