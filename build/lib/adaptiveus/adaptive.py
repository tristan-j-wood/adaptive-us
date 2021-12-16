# import adaptiveus
import numpy as np
from typing import Optional


class Window:

    # def __init__(self):
    #     """"""

    def load(self, filename: str) -> None:

        file_lines = open(filename, 'r').readlines()
        header_line = file_lines.pop(0)

        ref_zeta = float(header_line.split()[0])
        kappa = float(header_line.split()[1])

        obs_zeta = [float(line) for line in file_lines]

        logger.info(f'Zetas: {obs_zeta}')

        return None
