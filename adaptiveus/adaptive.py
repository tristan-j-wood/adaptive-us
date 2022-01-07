import numpy as np
import matplotlib.pyplot as plt
from adaptiveus.log import logger
from adaptiveus.overlap import calculate_overlap
from scipy.optimize import curve_fit
from typing import Optional, Sequence
if 'paper' in plt.style.available:
    plt.style.use('paper')


class Windows(list):
    """Collective US windows"""

    def __init__(self):
        """Collection of all the windows from an US simulation"""
        super().__init__()

    def load(self, filename: str) -> None:
        """
        Load sampled reaction coordinates from a .txt file, with the first
        line containing the window number, reference and kappa, respectively.

        E.g., 5 2.500 10.00
              2.4615
              2.4673
              ...

        -----------------------------------------------------------------------
        Arguments:
            filename:

        Raises:
            (ValueError): If an unsupported file extension is present
        """

        if not filename.endswith('.txt'):
            raise ValueError(f"Cannot load {filename}. Must be an .txt file")

        file_lines = open(filename, 'r').readlines()
        header_line = file_lines.pop(0)
        header_items = header_line.split()

        if len(header_items) != 3:
            raise ValueError("First line must contain window number, "
                             "reference and kappa")

        if int(header_items[0]) in [window.window_n for window in self]:
            raise ValueError("Identically numbered window trying to be loaded")

        window = Window(window_n=int(header_items[0]),
                        zeta_ref=float(header_items[1]),
                        kappa=float(header_items[2]))

        for idx, line in enumerate(file_lines):
            try:
                window.obs_zetas.append(float(line))
            except ValueError:
                raise ValueError(f'Could not convert line {idx}:\n {line}')

        self.append(window)

        return None

    @property
    def zeta_refs(self) -> Optional[np.ndarray]:
        """
        Array of ζ_ref for each window

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray(float) | None):
        """
        if len(self) == 0:
            return None

        return np.array([window.zeta_ref for window in self])

    def _get_fitted_window_from_index(self, idx):
        """
        Returns the window associated with the specified index and fits a
        Gaussian to this window data
        """

        try:
            window = next(window for window in self if window.window_n == idx)
        except StopIteration:
            raise StopIteration(f"Window {idx} not loaded")

        window.fit_gaussian()

        return window

    def calculate_overlap(self, idx0: int, idx1: int) -> None:
        """
        Calculates the overlap between two specified windows. Sets the LHS and
        RHS normalised overlap for the two windows
        """

        window_a = self._get_fitted_window_from_index(idx0)
        window_b = self._get_fitted_window_from_index(idx1)

        overlaps = calculate_overlap(window_a.gaussian.params,
                                     window_b.gaussian.params)

        window_a.rhs_overlap = overlaps[0]
        window_b.lhs_overlap = overlaps[1]

        return None

    def plot_overlaps(self) -> None:
        """Plots the overlap as a function of window number"""

        if not any([window.lhs_overlap for window in self]) or not any(
                [window.rhs_overlap for window in self]):
            raise AssertionError('Cannot plot overlaps. '
                                 'Please set window.overlaps')

        lhs_overlaps = [window.lhs_overlap for window in self]
        rhs_overlaps = [window.rhs_overlap for window in self]

        x_vals = [window.window_n for window in self]

        plt.plot(x_vals, rhs_overlaps, marker='o', color='r', linestyle='--',
                 markersize=7, mfc='white', label='RHS Overlap')
        plt.plot(x_vals, lhs_overlaps, marker='o', color='b', linestyle='--',
                 markersize=7, mfc='white', label='LHS Overlap')

        plt.axhline(0.3, linestyle='dotted', label='Threshold', color='k',
                    alpha=0.8)

        plt.xlabel('Window index')
        plt.ylabel('Normalised overlap')
        plt.ylim(0, 1)
        plt.legend()

        plt.savefig('overlap.pdf')
        plt.close()

        return None

    def plot_discrepancy(self) -> None:
        """Plots the discrepancy (D = |mean - ref|) for all windows"""
        if len(self) == 0:
            raise ValueError("No windows are loaded")

        discrepancies = [window.discrepancy for window in self]
        x_vals = [window.window_n for window in self]

        plt.plot(x_vals, discrepancies, marker='o', color='k', linestyle='--',
                 markersize=7, mfc='white')

        plt.xlabel('Window index')
        plt.ylabel('Discrepancy / Å')
        plt.ylim(0, 0.1)

        plt.savefig('discrepancy.pdf')
        plt.close()

        return None

    def plot_histogram(self, indexes: Optional[Sequence[int]] = None) -> None:
        """
        Plots observed reaction coordinates as a histogram for a set of
        windows. If indexes is specified, only a subset of the windows will
        be plotted. E.g., indexes=[0, 1] will plot the first and second windows
        """
        plt.close()

        if indexes is not None:
            selected_windows = [self[i] for i in indexes]
            file_ext = '_'.join(str(self[i].window_n) for i in indexes)
            filename = f'window_histogram_{file_ext}.pdf'
        else:
            selected_windows = list(self)
            filename = f'window_histogram.pdf'

        if len(selected_windows) == 0:
            raise ValueError("No windows to plot")

        for window in selected_windows:
            if len(window.obs_zetas) == 0:
                raise ValueError("Not observed zetas. Is the data loaded?")

            window.fit_gaussian()

            hist, bin_edges = np.histogram(window.obs_zetas, bins=500)
            bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2

            plt.plot(window.window_range, window.gaussian(window.window_range))

            # Color of the fitted Gaussian matches the histogram color
            plt.hist(bin_centres, len(bin_centres), weights=hist, alpha=0.4,
                     color=plt.gca().lines[-1].get_color())

        plt.xlabel('Reaction coordinate / Å')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        return None


class Window:
    """Data associated with a window from US"""

    def __init__(self,
                 window_n: Optional[int] = None,
                 zeta_ref: Optional[float] = None,
                 kappa: Optional[float] = None):
        """
        Window from an umbrella sampling simulation. The data in this
        window can be used to fit a Gaussian. The convergence of this data
        can be tested.

        -----------------------------------------------------------------------
        Arguments:

            window_n: Window index number

            zeta_ref: Value of the reference value, ζ, used for this window

            kappa: Value of the spring constant, κ, used in umbrella sampling
        """

        self.window_n = window_n
        self.zeta_ref = zeta_ref
        self.kappa = kappa

        self.obs_zetas:  Optional[list] = []
        self.gaussian = _Gaussian()

        self.lhs_overlap, self.rhs_overlap = None, None

    def __str__(self):
        return f'window_{self.window_n}'

    @property
    def discrepancy(self):
        """
        Calculates the discrepancy for current window.
        D = |mean - ref|
        """
        if len(self.obs_zetas) == 0:
            raise ValueError("Not observed zetas. Is the data loaded?")

        if not all(self.gaussian.params):
            self.fit_gaussian()

        return abs(self.gaussian.mean - self.zeta_ref)

    @property
    def number_of_samples(self) -> int:
        """Number of sampled points in the window trajectory"""
        return len(self.obs_zetas)

    @property
    def window_range(self) -> np.ndarray:
        """An extended array of the window range for plotting"""
        x_range = abs(max(self.obs_zetas) - min(self.obs_zetas))
        min_x = min(self.obs_zetas) - 0.1 * x_range
        max_x = max(self.obs_zetas) + 0.1 * x_range

        return np.linspace(min_x, max_x, 500)

    def fit_gaussian(self) -> None:
        """Fits Gaussian parameters to the window data"""

        if len(self.obs_zetas) == 0:
            raise ValueError("Not observed zetas. Is the data loaded?")

        self.gaussian.fit(self.obs_zetas)

        return None

    def _get_fractional_data(self) -> list:
        """Returns a list of the window data in 10% cumulative amounts"""

        if len(self.obs_zetas) == 0:
            raise AssertionError('Cannot get a fraction of non existing data. '
                                 'Please set window.obs_zetas')

        # Set to log scale?
        data_intervals = np.linspace(self.number_of_samples / 10,
                                     self.number_of_samples, 10,
                                     dtype=int)

        return [self.obs_zetas[:frac] for frac in data_intervals]

    def _plot_gaussian_convergence(self, gaussians):
        """
        Plots the histogram data and Gaussians fitted to 10% incremements
        of the window trajectory
        """

        for gaussian in gaussians:
            plt.plot(self.window_range, gaussian(self.window_range))

        hist, bin_edges = np.histogram(self.obs_zetas, density=False, bins=500)
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2

        plt.hist(bin_centres, len(bin_centres), weights=hist, alpha=0.4,
                 color='#1f77b4')

        plt.xlabel('Reaction coordinate / Å')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(f'gaussian_conv_{self.window_n}.pdf')
        plt.close()

        return None

    def _plot_param_convergence(self, b_data, c_data):
        """Plots the mean and standard deviation as a fraction of the traj"""

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        b_param_plt = ax1.plot(np.linspace(0.1, 1, 10), b_data,
                               linestyle='--', markersize=7, marker='o',
                               mfc='white', color='k', label='Mean')

        c_param_plt = ax2.plot(np.linspace(0.1, 1, 10), c_data,
                               linestyle='--', markersize=7, marker='o',
                               mfc='white', color='b',
                               label='Standard deviation')

        lines = b_param_plt + c_param_plt
        labels = [label.get_label() for label in lines]
        ax1.legend(lines, labels, loc='best')

        ax1.set_xlabel('Fraction of window trajectory')
        ax1.set_ylabel('Mean / Å')
        ax1.set_xlim(0)
        # ax1.set_ylim(min(b_data) - 0.1 * min(b_data),
        #              max(b_data) + 0.1 * max(b_data))

        ax2.set_ylabel('Standard deviation / Å')
        # ax2.set_ylim(min(c_data) - 0.1 * min(c_data),
        #              max(c_data) + 0.1 * max(c_data))

        plt.tight_layout()
        plt.savefig(f'param_conv_{self.window_n}.pdf')
        plt.close()

    def _fractional_gaussians(self, data) -> list:
        """Fits and plots Gaussians at fractions of the data"""

        gaussians = []
        for subdata in data:
            gaussian = _Gaussian()
            gaussian.fit(subdata)
            gaussians.append(gaussian)

        self._plot_gaussian_convergence(gaussians)

        return gaussians

    @staticmethod
    def _parameter_converged(params, threshold) -> bool:
        """Tests whether a parameter has converged within a given threshold"""

        p_prev = params[0]
        for idx, value in enumerate(params[1:]):

            if abs(value - p_prev) < threshold:
                return True
            else:
                p_prev = value

        return False

    def gaussian_converged(self, b_threshold=0.05, c_threshold=0.01) -> bool:
        """
        Returns True if both b and c parameters of the Gaussian have
        converged. Plots b and c parameters along the trajectory
        """
        fractional_data = self._get_fractional_data()
        fractional_gaussians = self._fractional_gaussians(fractional_data)

        # Set threshold based on expected SD and fraction along RC for mean
        b_params = [gaussian.mean for gaussian in fractional_gaussians]
        b_converged = self._parameter_converged(b_params,
                                                threshold=b_threshold)

        c_params = [gaussian.std for gaussian in fractional_gaussians]
        c_converged = self._parameter_converged(c_params,
                                                threshold=c_threshold)

        self._plot_param_convergence(b_params, c_params)

        return b_converged * c_converged

    def bins_converged(self):
        """Tests the convergence of the bin heights relative to previous
        using an autocorrelation function"""

        return NotImplementedError


class _Gaussian:
    """Gaussian function fit to a set of data"""

    def __init__(self,
                 a: Optional[float] = None,
                 b: Optional[float] = None,
                 c: Optional[float] = None):
        """
        Gaussian parameterised by a, b and c constants:

        y = a * exp(-(x-b)^2 / (2*c^2))
        """

        self.params = a, b, c

    @property
    def mean(self) -> float:
        return self.params[1]

    @property
    def std(self) -> float:
        return self.params[2]

    @property
    def area(self) -> float:
        """
        Returns integral of Gaussian between -∞ and ∞:

        I = ∫ dx a * exp(-(x-b)^2 / (2*c^2)) = ac √(2π)
        """
        assert all(self.params)
        return self.params[0] * self.params[2] * (2 * np.pi)**0.5

    def __call__(self, x):
        """Returns y-value of Gaussian given input parameters and x-value"""

        assert all(self.params)
        return self.value(x, *self.params)

    @staticmethod
    def value(x, a, b, c):
        return a * np.exp(-(x - b)**2 / (2. * c**2))

    def fit(self, data) -> None:
        """Fit a Gaussian to a set of data"""

        hist, bin_edges = np.histogram(data, density=False, bins=500)
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2

        initial_guess = [1.0, 1.0, 1.0]
        try:
            self.params, _ = curve_fit(self.value, bin_centres, hist,
                                       p0=initial_guess,
                                       maxfev=10000)

            # c parameter is always non-negative
            self.params[2] = abs(self.params[2])

        except RuntimeError:
            logger.error('Failed to fit a gaussian to this data')

        return None
