from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class MDDriver(ABC):
    """Abstract base class for MD driver"""

    @abstractmethod
    def run_md_window(self,
                      traj,
                      driver: Optional,
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
    def calculate_free_energy(self,
                              windows: 'adaptiveus.adaptive.Windows',
                              zetas: np.ndarray):
        """Calculate the free energy from a set of observed reaction
        coordinates"""
