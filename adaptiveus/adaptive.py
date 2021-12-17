from adaptiveus.log import logger
import numpy as np
from scipy.optimize import curve_fit
from typing import Optional, List
import matplotlib.pyplot as plt
plt.style.use('paper')


class Window:
    """Data associated with a window from US"""

    def __init__(self):
        """
        Window from an umbrella sampling simulation. The data in this
        window can be used to fit a Gaussian. The convergence of this data
        can be tested.
        """

        self.window_num:    Optional[int] = None
        self.ref_zeta:      Optional[float] = None
        self.kappa:         Optional[float] = None
        self.obs_zetas:      Optional[list] = None

        self.gaussian = _Gaussian()

    @property
    def traj_len(self):
        return len(self.obs_zetas)

    @property
    def window_range(self):
        min_x = min(self.obs_zetas) * 0.9
        max_x = max(self.obs_zetas) * 1.1

        return np.linspace(min_x, max_x, 500)

    def load(self, filename: str) -> None:
        """
        Load sampled window data assocaited with kappa and reference, and
        Fit a gaussian to this data.
        """

        # Do I want this load function here or in different class? I may want
        # to be working with multile windows at once, which may or may not
        # have data loaded in from a file

        file_lines = open(filename, 'r').readlines()
        header_line = file_lines.pop(0)

        if len(header_line.split()) != 3:
            raise ValueError("First line must contain window number, "
                             "reference and kappa")

        self.window_num = int(header_line.split()[0])
        self.ref_zeta = float(header_line.split()[1])
        self.kappa = float(header_line.split()[2])

        try:
            self.obs_zetas = [float(line) for line in file_lines]

        except ValueError:
            logger.error("Could not convert file zetas into floats. Is the "
                         "format in the file correct?")

        return None

    def plot_data(self, filename='fitted_data.pdf'):
        """Histograms the data and optionally plots the fitted Gaussian"""

        if self.obs_zetas is None:
            raise ValueError("Observed zetas are None. Is the data loaded?")

        self.gaussian.fit_gaussian(self.obs_zetas)

        hist, bin_edges = np.histogram(self.obs_zetas, density=False, bins=500)
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2

        plt.plot(self.window_range, self.gaussian(self.window_range))
        plt.hist(bin_centres, len(bin_centres), weights=hist, alpha=0.4,
                 color=plt.gca().lines[-1].get_color())

        plt.xlabel('Reaction coordinate / Å')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(filename)

        return None

    def _get_fractional_data(self):
        """Returns a list of the window data in 10% cumulative amounts"""

        data_intervals = np.linspace(self.traj_len / 10,
                                     self.traj_len, 10,
                                     dtype=int)

        return [self.obs_zetas[:frac] for frac in data_intervals]

    def convergence_of_gaussian(self):
        """
        Tests the convergence of the (a,b,c) parameters of the Gaussian and
        the convergence of the bin heights to the Gaussian values
        """
        gaussian = _Gaussian()
        data = self._get_fractional_data()

        plt.close()

        b_params, c_params = [], []
        for subdata in data:

            gaussian.fit_gaussian(subdata)
            b_params.append(gaussian.params[1])
            c_params.append(gaussian.params[2])

            plt.plot(self.window_range, gaussian(self.window_range))

        self.plot_data(filename='param_conv.pdf')

        plt.close()
        plt.plot(np.linspace(0, 1, 10), b_params)
        plt.plot(np.linspace(0, 1, 10), c_params)
        plt.savefig('tmp.pdf')

        # take the difference between a point and the previous point and
        # see if the difference is small enough (threshold) and thus converged

        return NotImplementedError

    def convergence_of_bins(self):
        """Tests the convergence of the bin heights relative to previous"""

        # autocorrelation function?
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
