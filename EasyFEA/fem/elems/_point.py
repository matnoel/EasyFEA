# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Point element module."""

import numpy as np

from .._group_elems import _GroupElem

class POINT(_GroupElem):
    
    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def origin(self) -> list[int]:
        return super().origin

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def faces(self) -> list[int]:
        return [0]

    def _N(self) -> np.ndarray:
        pass

    def _dN(self) -> np.ndarray:
        pass

    def _ddN(self) -> np.ndarray:
        pass
    
    def _dddN(self) -> np.ndarray:
        pass

    def _ddddN(self) -> np.ndarray:
        pass