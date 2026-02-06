# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Poisson2
========

Poisson equation with unit load.

Reference: https://scikit-fem.readthedocs.io/en/latest/gettingstarted.html
"""

import matplotlib.pyplot as plt
import numpy as np

from EasyFEA import Display, ElemType, Models, Simulations
from EasyFEA.FEM import Field, BiLinearForm, LinearForm
from EasyFEA.Geoms import Domain

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    contour = Domain((0, 0), (1, 1), 1 / 8)

    mesh = contour.Mesh_2D([], ElemType.TRI15, isOrganised=True)

    nodes = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])

    # ----------------------------------------------
    # Formulations
    # ----------------------------------------------

    field = Field(mesh.groupElem, 1)

    @BiLinearForm
    def bilinear_form(u: Field, v: Field):
        return u.grad.dot(v.grad)

    @LinearForm
    def linear_form(v: Field):
        x, y, _ = v.Get_coords()
        f = np.sin(np.pi * x) * np.sin(np.pi * y)
        return f * v

    weakForms = Models.WeakForms(field, computeK=bilinear_form, computeF=linear_form)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    simu = Simulations.WeakForms(mesh, weakForms)

    simu.add_dirichlet(nodes, [0], ["u"])

    simu.Solve()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    Display.Plot_Result(simu, "u", plotMesh=True)

    # compute error
    x, y, z = mesh.coord.T
    u_an = 1 / 2 / np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    error = np.linalg.norm(u_an - simu.u) / np.linalg.norm(u_an)

    plt.show()
