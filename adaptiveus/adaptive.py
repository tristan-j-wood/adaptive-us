from adaptiveus.log import logger
import numpy as np
from typing import Optional


class Window:

    def __init__(self):
        """"""

        self.window_num = None

        self.ref_zeta = None
        self.kappa = None

        self.obs_zeta = None

    def load(self, filename: str) -> None:

        file_lines = open(filename, 'r').readlines()
        header_line = file_lines.pop(0)

        self.window_num = int(header_line.split()[0])
        self.ref_zeta = float(header_line.split()[1])
        self.kappa = float(header_line.split()[2])

        self.obs_zeta = [float(line) for line in file_lines]

        return None
