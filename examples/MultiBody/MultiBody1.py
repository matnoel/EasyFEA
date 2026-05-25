# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA import Display, PyVista
from EasyFEA.Geoms import Domain

from EasyFEA.Simulations._multibody import Solid

if __name__ == "__main__":

    Display.Clear()

    L = 1  # m
    h = 13e-2

    sec = Domain((0, 0), (h, h))
    bar = sec.Mesh_Extrude([], (0, 0, L))
    bar.Rotate(90, direction=(0, 1))

    # TODO 40: Create a frame
    # Create a multi-body system from a list of solids
    # Create links between solids

    PyVista.Plot_Mesh(bar).show()

    solid1 = Solid(bar)

    pass
