# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Poisson
=======

Poisson equation with unit load.

Reference: https://scikit-fem.readthedocs.io/en/latest/listofexamples.html#example-1-poisson-equation-with-unit-load
"""

from EasyFEA import Display, ElemType, Models, Simulations, np
from EasyFEA.fem import Field, BiLinearForm, LinearForm
from EasyFEA.Geoms import Domain

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    contour = Domain((0, 0), (1, 1), 1 / 2**2)

    mesh = contour.Mesh_2D([], ElemType.TRI15, isOrganised=True)

    # ----------------------------------------------
    # Formulations
    # ----------------------------------------------

    field = Field(mesh.groupElem, 1)

    @BiLinearForm
    def bilinear_form(u: Field, v: Field):
        return u.grad.dot(v.grad)

    @LinearForm
    def linear_form(v: Field):
        x, y, z = v.Get_coords()
        return np.sin(np.pi * x) * np.sin(np.pi * y) * v
        return 1.0 * v

    weakFormManager = Models.WeakFormManager(
        field, computeK=bilinear_form, computeF=linear_form
    )

    simu = Simulations.WeakFormSimu(mesh, weakFormManager)

    nodes = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])
    simu.add_dirichlet(nodes, [0], ["0"])

    simu.Solve()

    # ----------------------------------------------
    # Formulations
    # ----------------------------------------------

    Display.Plot_Result(simu, "u", plotMesh=True)

    x, y, z = mesh.coord.T
    excepted = 1 / 2 / np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    diff = np.linalg.norm(excepted - simu.u) / np.linalg.norm(excepted)

    Display.plt.show()
