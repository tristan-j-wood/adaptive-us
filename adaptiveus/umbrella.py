import mltrain.sampling.md
import numpy as np
from multiprocessing import Pool
from scipy import special, optimize
from adaptiveus.log import logger
from adaptiveus.adaptive import Windows
from adaptiveus.config import Config
from mltrain.sampling.umbrella import UmbrellaSampling as MltrainUS
from typing import Optional, Callable
from copy import deepcopy


class UmbrellaSampling:

    def __init__(self,
                 traj: 'mltrain.configurations.trajectory',
                 driver: 'mltrain.potentials._base.MLPotential',
                 zeta_func: Optional['mltrain.sampling.reaction_coord.ReactionCoordinate'],
                 kappa: float,
                 temp: float,
                 interval: int,
                 dt: float,
                 init_ref: Optional[float] = None,
                 final_ref: Optional[float] = None,
                 ):
        """
        Perform adaptive umbrella sampling using (currently only) mltrain to
        drive the dynamics

        -----------------------------------------------------------------------
        Arguments:

            traj: .xyz trajectory from which to initialise the umbrella over,
                  e.g. a 'pulling' trajectory that has sufficient sampling of a
                  range of reaction coordinates

            driver: Driver for the dynamics. E.g., machine learnt potential

            zeta_func: Reaction coordinate, as the function of atomic positions

            kappa: Value of the spring constant, κ, used in umbrella sampling

            temp: Temperature in K to initialise velocities and to run NVT MD.
                  Must be positive

            interval: Interval between saving the geometry

            dt: Time-step in fs

            init_ref: Value of reaction coordinate in Å for
                      first window

            final_ref: Value of reaction coordinate in Å for
                       first window
        """

        self.driver = driver    # Only mltrain implemented
        self.zeta_func:         Optional[Callable] = zeta_func

        self.traj = traj        # Only mltrain trajectories implemented

        self.kappa:             float = kappa
        self.temp:              float = temp
        self.interval:          int = interval
        self.dt:                float = dt

        self.windows:           Windows = Windows()

        self.init_ref = self._set_reference_point(init_ref, idx=0)
        self.final_ref = self._set_reference_point(final_ref, idx=-1)

    def _set_reference_point(self, ref, idx) -> float:
        """
        Sets the reference based either on the specified reference or a
        value from the trajectory
        """
        return self.zeta_func(self.traj[idx]) if ref is None else ref

    def _run_single_window(self, ref, idx, **kwargs
                           ) -> 'adaptiveus.adaptive.window':
        """Run a single umbrella sampling window using a specified method"""

        if isinstance(self.driver, mltrain.potentials._base.MLPotential):
            adaptive = MltrainAdaptive(zeta_func=self.zeta_func,
                                       kappa=self.kappa,
                                       temp=self.temp,
                                       interval=self.interval,
                                       dt=self.dt)

            adaptive.run_mlt_window(traj=self.traj, mlp=self.driver, ref=ref,
                                    idx=idx, **kwargs)
            window = Windows()

            #######################################
            # Will this know where to look?
            window.load(f'window_{idx}.txt')

            print(window[0])
            #######################################

        else:
            raise NotImplementedError

        return window[0]

    @staticmethod
    def _overlap_error_func(x, s, b, c) -> float:
        """
        Returns the value of expression for which the difference between the
        overlap, S, and the overlap integral for Gaussians, with identical
        a and c parameters, is zero:

        S = 1 + 0.5 * [erf((b - A) / c √2) + erf((x - A) / c √2)]

        2S - 2 - erf((b - A) / c √2) + erf((x - A) / c √2) = 0
        """

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

        root = optimize.fsolve(self._overlap_error_func, x0=inital_guess,
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

    def _test_convergence(self, ref, **kwargs) -> bool:
        """Test the convergence of a fitted Gaussian over a simulation"""
        window = self._run_single_window(ref=ref, idx=0, **kwargs)
        self.windows.append(window)

        converged = window.gaussian_converged()

        return converged

    def _calculate_free_energy(self):
        return NotImplementedError

    def _adjust_kappa(self):
        return NotImplementedError

    def run_non_adaptive_sampling(self,
                                  n_windows: Optional[int] = 10,
                                  **kwargs) -> None:
        """
        Run adaptive umbrella sampling for ml-train

        -----------------------------------------------------------------------
        Arguments:
            n_windows: Number of windows to run umbrella sampling for

            test_convergence: Test the convergence of the data in a window

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units
        """

        converged = self._test_convergence(ref=self.init_ref, **kwargs)
        logger.info(f'Gaussian parameters converged: {converged}')

        refs = self._reference_values(n_windows)

        n_processes = min(len(refs)-1, Config.n_cores)

        # Start from index 1 as test_convergence runs the first window
        with Pool(processes=n_processes) as pool:
            results = [pool.apply_async(self._run_single_window,
                                        args=(ref.copy(),
                                              idx+1),
                                        kwds=deepcopy(kwargs))
                       for idx, ref in enumerate(refs[1:])]

        for result in results:

            try:
                self.windows.append(result.get(timeout=2))

            except Exception as err:
                logger.error(f'Raised an exception in simulation: \n{err}')
                continue

        for idx in range(n_windows - 1):
            self.windows.calculate_overlap(idx0=idx, idx1=idx+1)

        self.windows.plot_overlaps()
        self.windows.plot_discrepancy()
        self.windows.plot_histogram()

        return None

    def run_adaptive_sampling(self,
                              s_target: Optional[float] = 0.2,
                              **kwargs) -> None:
        """
        Run adaptive umbrella sampling for ml-train

        -----------------------------------------------------------------------
        Arguments:
            s_target: Target fractional overlap in adaptive sampling

            test_convergence: Test the convergence of the data in a window

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units
        """

        converged = self._test_convergence(ref=self.init_ref, **kwargs)
        logger.info(f'Gaussian parameters converged: {converged}')

        ref = self._calculate_next_ref(idx=0, s_target=s_target)

        idx, ref = 1, ref
        while ref <= self.final_ref:

            window = self._run_single_window(ref=ref, idx=idx, **kwargs)

            self.windows.append(window)
            self._adjust_kappa()

            self.windows.calculate_overlap(idx0=idx-1, idx1=idx)

            ref = self._calculate_next_ref(idx=idx, s_target=s_target)
            idx += 1

        self.windows.plot_discrepancy()
        self.windows.plot_histogram()

        self._calculate_free_energy()

        return None

    def _reference_values(self, num) -> np.ndarray:
        """Set the values of the reference for each window"""
        return np.linspace(self.init_ref, self.final_ref, num)


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

    def run_mlt_window(self,
                       traj: 'mltrain.sampling.md.Trajectory',
                       mlp: 'mltrain.potentials._base.MLPotential',
                       ref: float,
                       idx: int,
                       **kwargs) -> None:
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

        umbrella = MltrainUS(zeta_func=self.zeta_func, kappa=self.kappa)

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
