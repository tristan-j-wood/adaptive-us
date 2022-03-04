from abc import ABC, abstractmethod
import numpy as np


class MDDriver(ABC):
    """Abstract base class for MD driver"""

    @property
    @abstractmethod
    def zetas(self) -> np.ndarray:
        """Get the zetas from the input pulling trajectory"""

    @abstractmethod
    def run_md_window(self,
                      traj,
                      ref: float,
                      idx: int,
                      **kwargs):
        """
        Run a single umbrella window and save the sampled reaction
        coordinates. The sampled reaction coordinate file contains the values
        of the reaction coordinate with the first line containing the window
        number, reference and kappa, respectively.

        E.g., 5 2.500 10.00
              2.4615
              2.4673
              ...
        """

    @abstractmethod
    def calculate_pot_energy(self) -> list:
        """"""

    @abstractmethod
    def calculate_bias_energy(self, kappas, ref):
        """"""

    @abstractmethod
    def calculate_free_energy(self,
                              windows: 'adaptiveus.adaptive.Windows',
                              zetas: np.ndarray):
        """Calculate the free energy from a set of observed reaction
        coordinates"""
