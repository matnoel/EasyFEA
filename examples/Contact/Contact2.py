# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Contact2
========

Frictionless contact between a deformable block and a rigid indenter.
"""

from EasyFEA import Terminal, Folder, Models, ElemType, PyVista
from EasyFEA.Geoms import Point, Domain, Points

from _utils import RigidContact

if __name__ == "__main__":
    Terminal.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    dim = 3  # 2 or 3
    result = "Svm"

    R = 10  # block size
    thickness = R / 3  # out-of-plane extent (3D)
    meshSize = R / 20 if dim == 2 else R / 8

    N = 30  # load steps
    delta = 0.1
    penalty = 1e7  # contact stiffness (larger -> less penetration)

    folder = Folder.Results_Dir()

    # ----------------------------------------------
    # Body
    # ----------------------------------------------
    body = Domain(
        (-R / 2, 0),
        (R / 2, R),
        meshSize,
    )
    if dim == 2:
        mesh = body.Mesh_2D(
            [],
            ElemType.QUAD4,
            isOrganised=True,
        )
    else:
        nz = max(1, round(thickness / meshSize))
        mesh = body.Mesh_Extrude(
            [],
            [0, 0, thickness],
            [nz],
            ElemType.HEXA8,
            isOrganised=True,
        )
    nodes_y0 = mesh.Nodes_Conditions(lambda x, y, z: y == 0)

    # ----------------------------------------------
    # Indenter
    # ----------------------------------------------
    r = R / 3
    contour = Points(
        [
            Point(-R / 2, R, r=r),
            Point(R / 2, R, r=r),
            Point(R / 2, 2 * R),
            Point(-R / 2, 2 * R),
        ]
    )
    if dim == 2:
        indenter = contour.Mesh_2D([], ElemType.TRI3)
    else:
        indenter = contour.Mesh_Extrude([], [0, 0, thickness], [nz], ElemType.TETRA4)
    # lower (contact) surface of the punch: below the end of the fillets
    nodes_contact = indenter.Nodes_Conditions(lambda x, y, z: y <= R + r + 1e-6)

    indenter.Set_Tag(nodes_contact, "contact")

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    material = Models.Elastic.Isotropic(
        dim, E=210000, v=0.3, planeStress=True, thickness=thickness
    )
    simu = RigidContact(mesh, material, penalty)

    list_indenter = [indenter]
    print(f"Penalty contact solve in {dim}D (Newton per step):")
    for i in range(N):
        # update indenter
        indenter = list_indenter[0].copy()
        indenter.Translate(dy=-(i + 1) / N * delta)  # lower the rigid indenter
        list_indenter.append(indenter)
        simu._contactMesh = indenter
        # solve contact
        simu.Bc_Init()
        simu.add_dirichlet(nodes_y0, [0] * dim, simu.Get_unknowns())
        simu.Solve()
        simu.Save_Iter()

    print(simu)

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    def Plot_Iter(plotter, n):
        simu.Set_Iter(n)
        PyVista.Plot(
            simu, result, 1, color="k", nColors=21, show_grid=True, plotter=plotter
        )
        PyVista.Plot(list_indenter[n], color="gray", alpha=0.4, plotter=plotter)
        PyVista.Plot_Elements(
            list_indenter[n], color="k", dimElem=1, linewidth=2, plotter=plotter
        )
        PyVista._setCameraPosition(plotter, mesh.inDim)

    PyVista.Movie_func(Plot_Iter, N, folder=folder, filename=f"{result}.gif")

    plotter = PyVista._Plotter()
    result = "uy"
    Plot_Iter(plotter, -1)
    plotter.show()
