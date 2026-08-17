# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Mesh9
=====

Meshing of a perforated plate with a structured mesh.

Radial lines cut the plate into eight blocks, each bounded by two radials, an arc and a
plate edge. The arcs and the radials are sized to give N divisions; the plate edges then
take N as well, since a block forces equal counts on opposite sides.
"""

# sphinx_gallery_thumbnail_number = 2

import numpy as np

from EasyFEA import Terminal, ElemType, PyVista
from EasyFEA.Geoms import Circle, Domain, Line

if __name__ == "__main__":
    dim = 2

    if dim == 2:
        elemType = ElemType.QUAD4
    else:
        elemType = ElemType.HEXA8

    Terminal.Clear()

    # ----------------------------------------------
    # Geom
    # ----------------------------------------------
    H = 90
    L = 45
    D = 10
    e = 20

    N = 5  # divisions on every curve
    mS = (np.pi / 4 * D / 2) / N  # one arc, divided N times

    circle = Circle((L / 2, H / 2), D, mS)
    domain = Domain((0, 0), (L, H), mS)

    # corners and edge midpoints, counterclockwise from the middle of the right edge, so
    # that they follow the same order as the arc ends and midpoints they are joined to
    corners = [
        (L, H / 2),
        (L, H),
        (L / 2, H),
        (0, H),
        (0, H / 2),
        (0, 0),
        (L / 2, 0),
        (L, 0),
    ]
    arcs = circle.Get_Contour().geoms
    radials = [
        Line(pt, corner)
        for pt, corner in zip([p for arc in arcs for p in (arc.pt1, arc.pt3)], corners)
    ]
    for radial in radials:
        radial.meshSize = radial.length / N  # N divisions, whatever the length

    PyVista.Plot_Geoms([domain, circle, *radials]).show()

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    if dim == 2:
        mesh = domain.Mesh_2D(
            [circle], elemType, isOrganised=True, additionalLines=radials
        )
    else:
        mesh = domain.Mesh_Extrude(
            [circle],
            extrude=(0, 0, e),
            layers=[3],
            elemType=elemType,
            isOrganised=True,
            additionalLines=radials,
        )

    if len(mesh.orphanNodes) > 0:
        plotter = PyVista.Plot_Nodes(mesh, mesh.orphanNodes)
        plotter.add_title("Orphan nodes detected")
        plotter.show()

    PyVista.Plot_Mesh(mesh).show()
