import mltrain.sampling.md
import numpy as np
from multiprocessing import Pool
from scipy import special, optimize, interpolate
from adaptiveus.log import logger
from adaptiveus.adaptive import Windows
from adaptiveus.config import Config
from adaptiveus.plotting import _plot_1d_total_energy, _plot_2d_total_energy
from adaptiveus.plotting import _get_minima_idxs
from typing import Optional, Callable, Tuple
from copy import deepcopy


def _interpolate(x_vals, y_vals, kind) -> np.ndarray:
    """Interpolates a function using scipy interp1d"""
    interpolator = interpolate.interp1d(x_vals, y_vals, kind=kind)

    return interpolator(np.linspace(min(x_vals), max(x_vals), 100))


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


class UmbrellaSampling:

    def __init__(self,
                 driver: ['mltrain.potentials._base.MLPotential', str],
                 kappa: float,
                 temp: float,
                 interval: int,
                 dt: float,
                 zeta_func: Optional['mltrain.sampling.reaction_coord.ReactionCoordinate'] = None,
                 traj: ['mltrain.configurations.trajectory', str] = None,
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

            driver: Driver for the dynamics. E.g., ML potential, gmx

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

        self.zeta_func:         Optional[Callable] = zeta_func  # mltrain only
        self.traj:              Optional = traj

        self.kappa:             float = kappa
        self.default_kappa:     float = kappa

        self.temp:              float = temp
        self.interval:          int = interval
        self.dt:                float = dt

        self.windows:           Windows = Windows()

        # Currently only works for mltrain
        self.init_ref = self._set_reference_point(init_ref, idx=0)
        self.final_ref = self._set_reference_point(final_ref, idx=-1)

        if isinstance(driver, mltrain.potentials._base.MLPotential):

            from adaptiveus.mltrain import MltrainAdaptive

            self.driver = MltrainAdaptive(zeta_func=self.zeta_func,
                                          mlp=driver,
                                          traj=self.traj,
                                          kappa=self.kappa,
                                          temp=self.temp,
                                          interval=self.interval,
                                          dt=self.dt)

        elif driver == 'gmx':

            from adaptiveus.gmx import GMXAdaptive

            self.driver = GMXAdaptive(kappa=self.kappa,
                                      temp=self.temp,
                                      pull_filename=self.traj,
                                      interval=self.interval,
                                      dt=self.dt)

    def _set_reference_point(self, ref, idx) -> float:
        """
        Sets the reference based either on the specified reference or a
        value from the trajectory
        """
        return self.zeta_func(self.traj[idx]) if ref is None else ref

    def _run_single_window(self, ref, idx, **kwargs
                           ) -> 'adaptiveus.adaptive.window':
        """Run a single umbrella sampling window using a specified method"""

        self.driver.run_md_window(ref=ref, idx=idx, **kwargs)

        windows = Windows()
        windows.load(f'window_{idx}.txt')

        return windows[0]

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

        window.bins_converged()

        return converged

    def calculate_free_energy(self, n_bins: int = 100
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the free energy using the default method specified by the
        driver
        """
        zetas = np.linspace(self.init_ref, self.final_ref, num=n_bins)

        free_energies = self.driver.calculate_free_energy(self.windows,
                                                          zetas=zetas)

        return free_energies

    def _calculate_and_plot_total_energy(self, xi, kappas, ref) -> np.ndarray:
        """
        Calculate the total energy from the sum of bias and potential energy.
        Plot these energies in 1D and 2D
        """

        reduced_xi = np.linspace(min(xi), max(xi), 100)
        xs, ys = np.meshgrid(xi, kappas)

        pot_energies = self.driver.calculate_pot_energy()
        pot_energies = _interpolate(xi[::3], pot_energies[::3], kind='cubic')

        bias_energy = self.driver.calculate_bias_energy(kappas=ys, ref=ref)
        bias_energy = _interpolate(xi, bias_energy, kind='quadratic')

        tot_energy = pot_energies + bias_energy
        tot_energy = np.asarray([_interpolate(reduced_xi[::3],
                                              tot_energy[i][::3],
                                              kind='cubic'
                                              ) for i in range(len(tot_energy))])

        # Plot 1D total energy for the middle kappa
        _plot_1d_total_energy(reduced_xi,
                              bias_energy,
                              pot_energies,
                              tot_energy,
                              ref=ref,
                              k_idx=int(len(kappas)/2))

        _plot_2d_total_energy(reduced_xi,
                              kappas,
                              tot_energy,
                              ref=ref)

        return tot_energy

    def _adjust_kappa(self, ref, kappa_threshold=0.01) -> None:
        """
        Adjusts the value of kappa based the prediction of the sum of the
        bias energy and potential energy. If one minimum is found near the
        reference value, kappa will be set corresponding to this bias
        """

        # be consistent with xi and zeta
        xi = self.driver.zetas
        kappas = np.linspace(0, 2 * self.default_kappa, num=30)

        tot_energy = self._calculate_and_plot_total_energy(xi, kappas, ref)

        xi_interp = np.linspace(min(xi), max(xi), 100)
        for i, kappa in enumerate(kappas):

            minima_idxs = _get_minima_idxs(tot_energy[i])

            if len(minima_idxs) == 1:
                if abs(xi_interp[minima_idxs[0]] - ref) < kappa_threshold:
                    logger.info(f"Setting kappa to {kappa}")
                    self.kappa = kappa

                    return None

        logger.info(f"Setting kappa to {self.default_kappa}")
        self.kappa = self.default_kappa

        return None

    def run_non_adaptive_sampling(self,
                                  n_windows: Optional[int] = 10,
                                  test_convergence: bool = True,
                                  **kwargs) -> None:
        """
        Run non-adaptive umbrella sampling for ml-train

        -----------------------------------------------------------------------
        Arguments:
            n_windows: Number of windows to run umbrella sampling for

            test_convergence: Run convergence testing

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units
        """
        refs = self._reference_values(n_windows)

        self._test_convergence(ref=self.init_ref, **kwargs)

        for idx, ref in enumerate(refs[1:]):
            window = self._run_single_window(ref=ref, idx=idx+1, **kwargs)
            self.windows.append(window)

        # n_processes = min(len(refs)-1, Config.n_cores)
        #
        # # Parallelisation currently doesn't work with gmx
        # with Pool(processes=n_processes) as pool:
        #
        #     if test_convergence:
        #         converged = self._test_convergence(ref=self.init_ref, **kwargs)
        #         logger.info(f'Gaussian parameters converged: {converged}')
        #
        #     # Start from index 1 as test_convergence runs the first window
        #     results = [pool.apply(self._run_single_window,
        #                           args=(ref.copy(),
        #                                 idx+1),
        #                           kwds=deepcopy(kwargs))
        #                for idx, ref in enumerate(refs[1:])]
        #
        # for result in results:
        #     self.windows.append(result)

        for idx in range(n_windows - 1):
            self.windows.calculate_overlap(idx0=idx, idx1=idx+1)

        self.windows.plot_overlaps()
        self.windows.plot_discrepancy()
        self.windows.plot_histogram()

        self.calculate_free_energy()

        return None

    def run_adaptive_sampling(self,
                              s_target: Optional[float] = 0.1,
                              **kwargs) -> None:
        """
        Run adaptive umbrella sampling for ml-train

        -----------------------------------------------------------------------
        Arguments:
            s_target: Target fractional overlap in adaptive sampling

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units
        """
        converged = self._test_convergence(ref=self.init_ref, **kwargs)
        logger.info(f'Gaussian parameters converged: {converged}')

        ref = self._calculate_next_ref(idx=0, s_target=s_target)

        with open('kappa_ref.txt', 'w') as outfile:
            idx, ref = 1, ref
            while ref <= self.final_ref:

                window = self._run_single_window(ref=ref, idx=idx, **kwargs)
                self.windows.append(window)

                self._adjust_kappa(ref)
                self.windows.calculate_overlap(idx0=idx-1, idx1=idx)

                ref = self._calculate_next_ref(idx=idx, s_target=s_target)

                print(f'{self.kappa} {ref}', file=outfile)

                idx += 1

        self.windows.plot_overlaps()
        self.windows.plot_discrepancy()
        self.windows.plot_histogram()

        self.calculate_free_energy()

        return None

    def _reference_values(self, num) -> np.ndarray:
        """Set the values of the reference for each window"""
        return np.linspace(self.init_ref, self.final_ref, num)
