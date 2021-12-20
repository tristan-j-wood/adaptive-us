from adaptiveus.log import logger
from adaptiveus.overlap import Overlap
import numpy as np
from scipy.optimize import curve_fit
from typing import Optional
import matplotlib.pyplot as plt
plt.style.use('paper')


class Windows(list):
    """Collective US windows"""

    def __init__(self):
        """Collection of all the windows from an US simulation"""
        super().__init__()

    def __add__(self,
                other: 'adaptiveus.adaptive._Window'):
        """Add a window to the collective windows """

        self.append(other)

        return self

    def load(self, filename: str) -> None:
        """
        Load sampled reaction coordinates from a .txt file, with the first
        line containing the window number, reference and kappa, respectively

        -----------------------------------------------------------------------
        Arguments:
            filename:

        Raises:
            (ValueError): If an unsupported file extension is present
        """

        if not filename.endswith('.txt'):
            raise ValueError(f"Cannot load {filename}. Must be an .txt file")

        else:

            file_lines = open(filename, 'r').readlines()
            header_line = file_lines.pop(0)

            if len(header_line.split()) != 3:
                raise ValueError("First line must contain window number, "
                                 "reference and kappa")

            window = _Window(window_n=int(header_line.split()[0]),
                             zeta_ref=float(header_line.split()[1]),
                             kappa=float(header_line.split()[2]))

            try:
                window.obs_zetas = [float(line) for line in file_lines]

            except ValueError:
                logger.error(
                    "Could not convert file zetas into floats. Is the "
                    "format in the file correct?")

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

        return np.array([w_k.zeta_ref for w_k in self])

    def calculate_overlap(self, idx_1, idx_2) -> float:
        """
        Calculates the overlap between two specified windows.
        Returns the minimum overlap of both normalised overlaps.
        Not implemented yet
        """

        overlap = Overlap()
        overlaps = overlap.calc_overlap(self[idx_1], self[idx_2])

        return min(overlaps)

    def plot_histogram(self, indexes=None) -> None:
        """Plots the histogram data and fitted Gaussian for a n windows"""

        if indexes is not None:
            selected_windows = [self[i] for i in indexes]
        else:
            selected_windows = self

        for window in selected_windows:
            if window.obs_zetas is None:
                raise ValueError("Observed zetas are None. Is the data loaded?")

            window.gaussian.fit_gaussian(window.obs_zetas)

            hist, bin_edges = np.histogram(window.obs_zetas, bins=500)
            bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2

            plt.plot(window.window_range, window.gaussian(window.window_range))
            plt.hist(bin_centres, len(bin_centres), weights=hist, alpha=0.4,
                     color=plt.gca().lines[-1].get_color())

        plt.xlabel('Reaction coordinate / Å')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(f'all_windows.pdf')
        plt.close()

        return None


class _Window:
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

        self.obs_zetas:  Optional[list] = None
        self.gaussian = _Gaussian()

    def __str__(self):
        return f'window_{self.window_n}'

    def __len__(self) -> int:
        """Number of sampled points in the window trajectory"""
        return len(self.obs_zetas)

    @property
    def window_range(self) -> np.ndarray:
        """An extended array of the window range for plotting"""
        min_x = min(self.obs_zetas) * 0.9
        max_x = max(self.obs_zetas) * 1.1

        return np.linspace(min_x, max_x, 500)

    def _get_fractional_data(self):
        """Returns a list of the window data in 10% cumulative amounts"""

        assert self.obs_zetas is not None
        data_intervals = np.linspace(len(self) / 10, len(self), 10, dtype=int)

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

        fig, ax = plt.subplots()
        twin = ax.twinx()

        b_param_plt = ax.plot(np.linspace(0.1, 1, 10), b_data,
                              linestyle='--', markersize=7, marker='o',
                              mfc='white', color='k', label='Mean')

        c_param_plt = twin.plot(np.linspace(0.1, 1, 10), c_data,
                                linestyle='--', markersize=7, marker='o',
                                mfc='white', color='b',
                                label='Standard deviation')

        lines = b_param_plt + c_param_plt
        labels = [label.get_label() for label in lines]
        ax.legend(lines, labels, loc='best')

        ax.set_xlabel('Fraction of window trajectory')
        ax.set_ylabel('Mean / Å')
        ax.set_xlim(0)
        ax.set_ylim(min(b_data) - 0.1 * min(b_data),
                    max(b_data) + 0.1 * max(b_data))

        twin.set_ylabel('Standard deviation / Å')
        twin.set_ylim(min(c_data) - 0.1 * min(c_data),
                      max(c_data) + 0.1 * max(c_data))

        plt.tight_layout()
        plt.savefig(f'param_conv_{self.window_n}.pdf')
        plt.close()

    @staticmethod
    def _parameter_converged(param, threshold) -> float:
        """Returns the percentage at which the parameters converged"""

        p_prev = param[0]
        for idx, value in enumerate(param[1:]):

            if abs(value - p_prev) < threshold:
                return (idx + 2) * 10.0

            else:
                p_prev = value

        return 100.0

    def gaussian_converged(self, b_threshold=0.05, c_threshold=0.01) -> None:
        """
        Have the a,b,c parameters of the Gaussian and the bin heights to the
        Gaussian values converged? Converged if difference between current
        value and previous is below a threshold
        """
        data = self._get_fractional_data()

        gaussians, b_params, c_params = [], [], []
        for i, subdata in enumerate(data):

            gaussian = _Gaussian()
            gaussian.fit_gaussian(subdata)
            gaussians.append(gaussian)

            b_params.append(gaussian.params[1])
            c_params.append(abs(gaussian.params[2]))

        self._plot_gaussian_convergence(gaussians)
        self._plot_param_convergence(b_params, c_params)

        b_conv = self._parameter_converged(b_params, b_threshold)
        c_conv = self._parameter_converged(c_params, c_threshold)

        logger.info(f'Mean converged {b_conv}% into the window')
        logger.info(f'Standard deviation converged {c_conv}% into the window')

        return None

    def bins_converged(self):
        """Tests the convergence of the bin heights relative to previous
        using an autocorrelation function"""

        return NotImplementedError


class _Gaussian:
    """Gaussian function fit to a set of data"""

    def __init__(self,
                 a: float = None,
                 b: float = None,
                 c: float = None):
        """
        Gaussian parameterised by a, b and c constants:

        y = a * exp(-(x-b)^2 / (2*c^2))
        """
        self.params = a, b, c

    @property
    def area(self):
        """
        Returns integral of Gaussian between -∞ and ∞:

        I = ∫ dx a * exp(-(x-b)^2 / (2*c^2)) = ac √(2π)
        """
        assert all(self.params)
        return self.params[0] * self.params[1] * (2 * np.pi)**0.5

    def __call__(self, x):
        """Returns y-value of Gaussian given input parameters and x-value"""

        assert all(self.params)
        return self.value(x, *self.params)

    @staticmethod
    def value(x, a, b, c):
        return a * np.exp(-(x - b)**2 / (2. * c**2))

    def fit_gaussian(self, data):
        """Fit a Gaussian to a set of data"""

        hist, bin_edges = np.histogram(data, density=False, bins=500)
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2

        initial_guess = [1.0, 1.0, 1.0]
        try:
            self.params, _ = curve_fit(self.value, bin_centres, hist,
                                       p0=initial_guess,
                                       maxfev=10000)
        except RuntimeError:
            logger.error('Failed to fit a gaussian to this data')

        return None
