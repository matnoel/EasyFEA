# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Poisson1
========

Poisson equation with unit load.

Reference: https://scikit-fem.readthedocs.io/en/latest/listofexamples.html#example-1-poisson-equation-with-unit-load
"""

from EasyFEA import Display, ElemType, Models, Simulations
from EasyFEA.FEM import Field, BiLinearForm, LinearForm
from EasyFEA.Geoms import Domain

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    contour = Domain((0, 0), (1, 1), 1 / 2**6)

    mesh = contour.Mesh_2D([], ElemType.TRI3, isOrganised=True)

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
        return 1.0 * v

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

    Display.Plot_Result(simu, "u")

    Display.plt.show()
