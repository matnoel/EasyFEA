# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Elas8
=====

A cantilever cylinder undergoing torsion, loaded either by a surface traction.
"""

import numpy as np

from EasyFEA import Terminal, ElemType, Models, Simulations, PyVista
from EasyFEA.Geoms import Circle

if __name__ == "__main__":
    Terminal.Clear()

    # --------------------------------------------
    # Mesh
    # --------------------------------------------

    L = 100  # mm
    D = 30

    circle = Circle((0, 0), D, n=(1, 0, 0))
    circle.meshSize = circle.length / 20

    mesh = circle.Mesh_Extrude([], [L, 0, 0], [L // circle.meshSize], ElemType.PRISM6)

    PyVista.Plot_Mesh(mesh).show()

    # --------------------------------------------
    # Simu
    # --------------------------------------------

    model = Models.Elastic.Isotropic(3, E=210000, v=0.3)
    simu = Simulations.Elastic(mesh, model)

    nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    simu.add_dirichlet(nodesX0, [0] * 3, ["x", "y", "z"])

    nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L)
    yC, zC = 0, 0
    T = 10  # N.mm
    J = np.pi * D**4 / 32  # polar second moment of area
    p = T / J  # so that the traction t = p * r integrates to a moment M_x = c
    simu.add_surfLoad(
        nodesXL,
        [
            lambda x, y, z: -p * (z - zC),
            lambda x, y, z: p * (y - yC),
        ],
        ["y", "z"],
    )

    u = simu.Solve()

    # --------------------------------------------
    # Results
    # --------------------------------------------
    # Saint-Venant torsion of a circular shaft: phi = T * L / (G * J)
    phi_ana = T * L / (model.get_mu() * J)  # rad

    # numerical twist from the tip-face nodes: a rigid rotation theta about x
    # gives uy = -theta*dz, uz = theta*dy, so theta = (dy*uz - dz*uy) / r^2
    uy = simu.Result("uy")[nodesXL]
    uz = simu.Result("uz")[nodesXL]
    dy = mesh.coord[nodesXL, 1] - yC
    dz = mesh.coord[nodesXL, 2] - zC
    r2 = dy**2 + dz**2
    phi_num = np.mean((dy * uz - dz * uy) / r2, where=r2 > 2)

    err = abs(phi_num - phi_ana) / abs(phi_ana) * 100
    Terminal.MyPrint(f"\nphi analytical = {phi_ana:.6e} rad")
    Terminal.MyPrint(f"\nphi numerical  = {phi_num:.6e} rad")
    Terminal.MyPrint(f"\nrelative error = {err:.3f} %")

    PyVista.Plot_BoundaryConditions(simu).show()

    pltr = PyVista._Plotter(shape=(3, 1))

    PyVista.Plot(simu, "uy", plotMesh=True, plotter=pltr, verticalColobar=False)
    pltr.subplot(1, 0)
    PyVista.Plot(simu, "uz", plotMesh=True, plotter=pltr, verticalColobar=False)
    pltr.subplot(2, 0)
    PyVista.Plot(
        simu, "displacement_norm", plotMesh=True, plotter=pltr, verticalColobar=False
    )
    pltr.show()
