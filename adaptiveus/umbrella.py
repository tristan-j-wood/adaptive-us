import numpy as np
from scipy import special, optimize
from adaptiveus.log import logger
from mltrain.sampling.umbrella import UmbrellaSampling
from adaptiveus.adaptive import Windows
from typing import Optional, Callable


class Adaptive:

    def __init__(self,
                 method: str,
                 mlp: Optional['mltrain.potentials._base.MLPotential'],
                 zeta_func: Optional['mltrain.sampling.reaction_coord.ReactionCoordinate'],
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

        self.method:            str = method
        self.mlp:               Optional = mlp
        self.zeta_func:         Optional[Callable] = zeta_func

        self.kappa:             float = kappa
        self.temp:              float = temp
        self.interval:          int = interval
        self.dt:                float = dt

        self.windows:           [Windows] = Windows()

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

    def _run_non_adaptive(self, traj, init_ref, final_ref, n_windows,
                          **kwargs) -> None:
        """Run non-adaptive umbrella sampling"""

        refs = self._reference_values(traj, n_windows, init_ref, final_ref)

        for idx, ref in enumerate(refs):
            self._run_single_window(ref=ref, traj=traj, idx=idx, **kwargs)

        for idx in range(n_windows - 1):
            self.windows.calculate_overlap(idx0=idx, idx1=idx+1)

        self.windows.plot_overlaps()

        return None

    @staticmethod
    def overlap_error_func(x, s, b, c) -> float:
        """
        Returns the value of expression for which the difference between the
        overlap and the overlap integral fraction for Gaussians with identical
        a and c parameters is zero
        """
        # Write a test
        int_func = (x**2 - b**2) / (2 * x - 2 * b)

        erf_1 = special.erf((b - int_func) / (c * np.sqrt(2)))
        erf_2 = special.erf((x - int_func) / (c * np.sqrt(2)))

        return 2 * s - 2 - erf_1 + erf_2

    def _find_error_function_roots(self, s_target, params) -> float:
        """
        Finds b value of the Gaussian with fixed a and c parameters which
        gives an overlap equal to the target overlap
        """

        b, c = params[1], params[2]
        inital_guess = b * 1.01  # Avoids dividing by zero

        root = optimize.fsolve(self.overlap_error_func, x0=inital_guess,
                               args=(s_target, b, c), maxfev=10000)

        assert len(root) == 1

        return float(root[0])

    def _calculate_next_ref(self, idx, s_target) -> float:
        """Calculate the next reference point based on a target overlap"""

        window = self.windows[idx]
        window.fit_gaussian()

        next_ref = self._find_error_function_roots(s_target,
                                                   window.gaussian.params)

        return next_ref

    def _test_convergence(self, traj, ref, **kwargs) -> bool:
        """Test the convergence of a fitted Gaussian over a simulation"""
        self._run_single_window(ref=ref, traj=traj, idx=0, **kwargs)

        converged = self.windows[0].gaussian_converged()
        self.windows.pop(0)

        return converged

    def _calculate_free_energy(self):
        return NotImplementedError

    def _adjust_kappa(self):
        return NotImplementedError

    def run_adaptive(self,
                     traj: Optional,
                     init_ref: Optional[float] = None,
                     final_ref: Optional[float] = None,
                     n_windows: Optional[int] = 10,
                     adaptive: bool = True,
                     s_target: Optional[float] = 0.2,
                     test_convergence: bool = True,
                     **kwargs) -> None:
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

            n_windows: Number of windows to run umbrella sampling for if
                       adaptive is False

            adaptive: Run adaptive sampling if True. Otherwise run normal
                      umbrella sampling but include convergence, overlap and
                      discrepancy calculations and plotting

            s_target: Target fractional overlap in adaptive sampling

            test_convergence: Test the convergence of the data in a window

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units
        """

        init_ref = self.zeta_func(
            traj[0]) if init_ref is None else init_ref

        final_ref = self.zeta_func(
            traj[-1]) if final_ref is None else final_ref

        if test_convergence:
            converged = self._test_convergence(traj=traj, ref=init_ref,
                                               **kwargs)
            logger.info(f'Gaussian parameters converged: {converged}')

        if not adaptive:
            self._run_non_adaptive(traj=traj, init_ref=init_ref,
                                   final_ref=final_ref, n_windows=n_windows,
                                   **kwargs)

        else:

            idx, ref = 0, init_ref
            while ref <= final_ref:
                self._run_single_window(ref=ref,
                                        traj=traj, idx=idx, **kwargs)

                ref = self._calculate_next_ref(idx=idx, s_target=s_target)

                if idx != 0:
                    self.windows.calculate_overlap(idx0=idx-1, idx1=idx)
                idx += 1

                self._adjust_kappa()

        self.windows.plot_discrepancy()
        self.windows.plot_histogram()

        self._calculate_free_energy()

        return None

    def _reference_values(self, traj, num, init_ref, final_ref) -> np.ndarray:
        """Set the values of the reference for each window, if the
        initial and final reference values of the reaction coordinate are None
        then use the values in the start or end of the trajectory"""

        if init_ref is None:
            init_ref = self.zeta_func(traj[0])

        if final_ref is None:
            final_ref = self.zeta_func(traj[-1])

        return np.linspace(init_ref, final_ref, num)


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

        self.zeta_func:         Optional[Callable] = zeta_func
        self.kappa:             float = kappa

        self.temp:              float = temp
        self.interval:          int = interval
        self.dt:                float = dt

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
