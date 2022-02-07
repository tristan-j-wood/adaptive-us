import mltrain.sampling.md
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import art3d
from multiprocessing import Pool
from scipy import special, optimize, interpolate
from adaptiveus.log import logger
from adaptiveus.adaptive import Windows
from adaptiveus.config import Config
from mltrain.sampling.umbrella import UmbrellaSampling as MltrainUS
from mltrain.sampling.bias import Bias
from typing import Optional, Callable, Tuple
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
        self.default_kappa:     float = kappa

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

    def _run_single_window(self,
                           ref,
                           idx,
                           **kwargs) -> 'adaptiveus.adaptive.window':
        """Run a single umbrella sampling window using a specified method"""
        if isinstance(self.driver, mltrain.potentials._base.MLPotential):

            adaptive = MltrainAdaptive(zeta_func=self.zeta_func,
                                       kappa=self.kappa,
                                       temp=self.temp,
                                       interval=self.interval,
                                       dt=self.dt)

            adaptive.run_mlt_window(traj=self.traj, mlp=self.driver, ref=ref,
                                    idx=idx, **kwargs)
            windows = Windows()
            windows.load(f'window_{idx}.txt')

        else:
            raise NotImplementedError

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

        return converged

    def calculate_free_energy(self, n_bins: int = 100
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the free energy using the default method specified by the
        driver
        """
        zetas = np.linspace(self.init_ref, self.final_ref, num=n_bins)

        if isinstance(self.driver, mltrain.potentials._base.MLPotential):

            adaptive = MltrainAdaptive(zeta_func=self.zeta_func,
                                       kappa=self.kappa,
                                       temp=self.temp,
                                       interval=self.interval,
                                       dt=self.dt)

            free_energies = adaptive.calculate_free_energy(self.windows, zetas)

        else:
            raise NotImplementedError

        return free_energies

    def _calculate_pot_energy(self) -> list:
        """
        Calculates the potential energy for configurations in a trajectory
        using the specified driver
        """
        if isinstance(self.driver, mltrain.potentials._base.MLPotential):

            self.driver.predict(self.traj)
            energies = [config.energy.predicted for config in self.traj]

        else:
            raise NotImplementedError

        return [energies[i] - min(energies) for i in range(len(energies))]

    def _calculate_bias_along_coord(self, kappa, ref) -> float:
        """Calculates the bias energy for configurations in a trajectory"""
        bias = Bias(self.zeta_func, kappa=kappa, reference=ref)
        bias = bias(self.traj)

        return bias

    @staticmethod
    def _add_point(ax, x, y, z, fc='red', ec='red', radius=0.01) -> None:
        """Add a point onto a 3D surface for plotting"""
        xy_len, z_len = ax.get_figure().get_size_inches()

        axis_length = [x[1] - x[0] for x in
                       [ax.get_xbound(), ax.get_ybound(), ax.get_zbound()]]

        axis_rotation = {
            'y': ((x, z, y),
                  axis_length[2] / axis_length[0] * xy_len / z_len)}

        for a, ((x0, y0, z0), ratio) in axis_rotation.items():
            p = Circle((x0, y0), radius, fc=fc, ec=ec, linewidth=0.1)
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=z0, zdir=a)

        return None

    @staticmethod
    def _idxs_at_edge(z, idx) -> bool:
        """Checks if idx is at the edge of z"""
        if idx == 0 or idx == len(z) - 1:
            return False
        else:
            return True

    def _plot_2d_total_energy(self, x, y, z, ref) -> None:
        """
        Plots a 2D surface of the bias energy + potential energy for a given
        reference value. Annotes the minima on the surface
        """
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        cmap = plt.cm.get_cmap('plasma')

        x_grid, y_grid = np.meshgrid(x, y)
        ax.plot_surface(x_grid, y_grid, z, cmap=cmap, linewidth=0,
                        antialiased=False)

        for i in range(len(y_grid)):
            grad = np.gradient(z[i])
            # Get the indexes where the sign change
            idxs = np.where(np.diff(np.sign(grad)) != 0)[0] + 1

            minima_idxs = []
            for idx in idxs:
                if not self._idxs_at_edge(z[i], idx):
                    if (z[i][idx - 1] and z[i][idx + 1]) > z[i][idx]:
                        minima_idxs.append(idx)

            for idx in minima_idxs:
                self._add_point(ax, x[idx], y_grid[i][0], z[i][idx])

            reference_idx = (np.abs(x - ref)).argmin()
            self._add_point(ax, ref, y_grid[i][0], z[i][reference_idx], fc='w',
                            ec='w')

        ax.set_xlabel('Reaction coordinate / Å')
        ax.set_ylabel(r'$\kappa~/~eV Å^{-2}$')
        ax.set_zlabel('Total energy / eV')

        plt.savefig('2d_total_energy.pdf')
        plt.close()

        return None

    @staticmethod
    def _plot_1d_total_energy(x, bias_e, pot_e, total_e, ref, k_idx) -> None:
        """
        Plots the bias energy, potential energy and total energy for a given
        kappa and reference value. Annotes the minima on the curve
        """
        plt.close()
        cmap = plt.cm.get_cmap('plasma')

        grad = np.gradient(total_e[k_idx])
        # Get the indexes where the sign changes
        idxs = np.where(np.diff(np.sign(grad)) != 0)[0] + 1

        minima_idxs = []
        for idx in idxs:
            if (total_e[k_idx][idx-1] and total_e[k_idx][idx+1]) > total_e[k_idx][idx]:
                minima_idxs.append(idx)

        plt.plot(x, pot_e, label='Potential energy', color=cmap(0.1))
        plt.plot(x, bias_e[k_idx], label='Bias energy', color=cmap(0.3))
        plt.plot(x, total_e[k_idx], label='Total energy', color=cmap(0.7))

        for idx in minima_idxs:
            plt.plot(x[idx], total_e[k_idx][idx], label='Minima', c='None',
                     marker='o', markerfacecolor='None', markeredgecolor='red',
                     markeredgewidth=1)

        reference_idx = (np.abs(x - ref)).argmin()

        plt.plot(ref, total_e[k_idx][reference_idx], label='Reference',
                 c='None', marker='o', markerfacecolor='None',
                 markeredgecolor='k', markeredgewidth=1)

        plt.xlabel('Reaction coordinate / Å')
        plt.ylabel('Energy / eV')
        plt.legend()

        plt.savefig('1d_total_energy.pdf')
        plt.close()

        return None

    @staticmethod
    def _interpolate(x_vals, y_vals, kind):
        """Interpolates a function using scipy interp1d"""
        interpolate_class = interpolate.interp1d(x_vals,
                                                 y_vals,
                                                 kind=kind)

        return interpolate_class(np.linspace(min(x_vals), max(x_vals), 100))

    def _calculate_and_plot_total_energy(self, xi, kappas, ref) -> np.ndarray:
        """
        Calculate the total energy from the sum of bias and potential energy.
        Plot these energies in 1D and 2D
        """

        xi_interp = np.linspace(min(xi), max(xi), 100)
        xs, ys = np.meshgrid(xi, kappas)

        pot_energy = self._calculate_pot_energy()
        pot_energy = self._interpolate(xi[::3], pot_energy[::3], kind='cubic')

        bias_energy = self._calculate_bias_along_coord(kappa=ys, ref=ref)
        bias_energy = self._interpolate(xi, bias_energy, kind='quadratic')

        tot_energy = bias_energy + pot_energy
        tot_energy = [self._interpolate(xi_interp[::3],
                                        tot_energy[i][::3],
                                        kind='cubic'
                                        ) for i in range(len(tot_energy))]

        tot_energy = np.asarray(tot_energy)

        # Plot 1D total energy for the middle kappa
        self._plot_1d_total_energy(xi_interp,
                                   bias_energy,
                                   pot_energy,
                                   tot_energy,
                                   ref,
                                   k_idx=int(len(kappas)/2))

        self._plot_2d_total_energy(xi_interp, kappas, tot_energy, ref)

        return tot_energy

    @staticmethod
    def _get_minima_idxs(i, y_val):
        """Find the indexes of the minima from a set of data"""
        grad = np.gradient(y_val[i])
        idxs = np.where(np.diff(np.sign(grad)) != 0)[0] + 1

        minima_idxs = []
        for idx in idxs:
            if (y_val[i][idx - 1] and y_val[i][idx + 1]) > y_val[i][idx]:
                minima_idxs.append(idx)

        return minima_idxs

    def _adjust_kappa(self, ref, kappa_threshold=0.025) -> None:
        """
        Adjusts the value of kappa based the prediction of the sum of the
        bias energy and potential energy
        """
        xi = self.zeta_func(self.traj)
        kappas = np.linspace(0, self.default_kappa, num=30)

        tot_energy = self._calculate_and_plot_total_energy(xi, kappas, ref)

        xi_interp = np.linspace(min(xi), max(xi), 100)
        for i, kappa in enumerate(kappas):

            minima_idxs = self._get_minima_idxs(i, tot_energy)

            # Sets kappa to default value if no suitable kappa is found
            if i == len(kappas) - 1:
                self.kappa = self.default_kappa

            elif len(minima_idxs) == 1:
                if abs(xi_interp[minima_idxs[0]] - ref) < kappa_threshold:
                    self.kappa = kappa

                    break

        return None

    def run_non_adaptive_sampling(self,
                                  n_windows: Optional[int] = 10,
                                  **kwargs) -> None:
        """
        Run non-adaptive umbrella sampling for ml-train

        -----------------------------------------------------------------------
        Arguments:
            n_windows: Number of windows to run umbrella sampling for

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units
        """
        refs = self._reference_values(n_windows)
        n_processes = min(len(refs)-1, Config.n_cores)

        with Pool(processes=n_processes) as pool:

            converged = self._test_convergence(ref=self.init_ref, **kwargs)
            logger.info(f'Gaussian parameters converged: {converged}')

            # Start from index 1 as test_convergence runs the first window
            results = [pool.apply(self._run_single_window,
                                  args=(ref.copy(),
                                        idx+1),
                                  kwds=deepcopy(kwargs))
                       for idx, ref in enumerate(refs[1:])]

        for result in results:
            self.windows.append(result)

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

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units
        """
        converged = self._test_convergence(ref=self.init_ref, **kwargs)
        logger.info(f'Gaussian parameters converged: {converged}')

        ref = self._calculate_next_ref(idx=0, s_target=s_target)

        with open('parameters.txt', 'w') as outfile:
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

    def calculate_free_energy(self, windows, zetas
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the free energy using WHAM in mltrain"""
        umbrella = MltrainUS(zeta_func=self.zeta_func,
                             kappa=self.kappa,
                             temp=self.temp)

        [window.bin(zetas=zetas) for window in windows]
        umbrella.windows = windows

        free_energies = umbrella.wham()

        return free_energies
