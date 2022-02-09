from abc import ABC, abstractmethod
from typing import Optional


class MDDriver(ABC):
    """Abstract base class for MD driver"""

    @abstractmethod
    def run_md_window(self,
                      traj,
                      mlp: Optional,
                      ref: float,
                      idx: int,
                      **kwargs):
        """Run a single umbrella window and save the sampled reaction
        coordinates"""

    @abstractmethod
    def calculate_free_energy(self, windows, zetas):
        """Calculate the free energy from a set of observed reaction
        coordinates"""
