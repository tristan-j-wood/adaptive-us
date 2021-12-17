from adaptiveus.log import logger
import numpy as np
from scipy.optimize import curve_fit
from typing import Optional
import matplotlib.pyplot as plt
plt.style.use('paper')


class Window:
    """Data associated with a window from US"""

    def __init__(self):
        """"""

        self.window_num = None
        self.ref_zeta = None
        self.kappa = None
        self.obs_zeta = None

        self.gaussian = _Gaussian()

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
            self.obs_zeta = [float(line) for line in file_lines]

        except ValueError:
            logger.error("Could not convert file zetas into floats. Is the "
                         "format in the file correct?")

        assert self.obs_zeta is not None
        self.gaussian.fit_gaussian(self.obs_zeta)

        return None

    def plot_data(self):
        """Histograms the data and optionally plots the fitted Gaussian"""

        if self.obs_zeta is None:
            raise ValueError("Observed zetas are None. Is the data loaded?")

        min_x = min(self.obs_zeta) * 0.9
        max_x = max(self.obs_zeta) * 1.1

        x_range = np.linspace(min_x, max_x, 500)

        hist, bin_edges = np.histogram(self.obs_zeta, density=False, bins=500)
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2

        plt.plot(x_range, self.gaussian(x_range))
        plt.hist(bin_centres, len(bin_centres), weights=hist, alpha=0.4,
                 color=plt.gca().lines[-1].get_color())

        plt.xlabel('Reaction coordinate / Å')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('fitted_data.pdf')

        return None


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
        """Function for the Gaussian"""
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
