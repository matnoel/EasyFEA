# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Hyperelas3
==========

A L shape part undergoing bending deformation.
"""
# sphinx_gallery_thumbnail_number = -1

from EasyFEA import Display, Models, ElemType, Simulations, PyVista
from EasyFEA.Geoms import Point, Points

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    dim = 3

    # geom
    L = 250
    thickness = 50
    w = 50

    # load
    sigMax = 8 * 1e6 / (w * thickness)

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    meshSize = L / 10

    p1 = Point(0, 0)
    p2 = Point(L, 0)
    p3 = Point(L, L, r=50)
    p4 = Point(2 * L - w, L)
    p5 = Point(2 * L, L)
    p6 = Point(2 * L, 2 * L)
    p7 = Point(2 * L - w, 2 * L)
    p8 = Point(0, 2 * L)

    contour = Points([p1, p2, p3, p4, p5, p6, p7, p8], meshSize)

    if dim == 2:
        mesh = contour.Mesh_2D([], ElemType.TRI3)
    else:
        mesh = contour.Mesh_Extrude([], [0, 0, -thickness], [3], ElemType.PRISM6)

    nodes_y0 = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
    nodes_Load = mesh.Nodes_Conditions(lambda x, y, z: x == 2 * L)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    elas = Models.Elastic.Isotropic(dim, E=210000, v=0.25, planeStress=True)
    material = Models.HyperElastic.SaintVenantKirchhoff(
        dim, elas.get_lambda(), elas.get_mu(), thickness=thickness
    )

    simu = Simulations.HyperElastic(mesh, material)

    simu.add_dirichlet(nodes_y0, [0] * dim, simu.Get_unknowns())
    simu.add_surfLoad(nodes_Load, [sigMax], ["y"])

    simu.Solve()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    PyVista.Plot_Mesh(simu).show()
    PyVista.Plot_BoundaryConditions(simu).show()
    PyVista.Plot(simu, "ux", deformFactor=1).show()
    PyVista.Plot(simu, "uy", deformFactor=1).show()

    print(simu)
