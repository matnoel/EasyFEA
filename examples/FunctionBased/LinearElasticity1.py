# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
LinearElasticity
================

A cantilever beam undergoing bending traction.

Reference: https://scikit-fem.readthedocs.io/en/latest/listofexamples.html#example-11-three-dimensional-linear-elasticity
"""

from EasyFEA import Display, ElemType, Models, Simulations, np
from EasyFEA.fem import Field, BiLinearForm, FeArray, Trace
from EasyFEA.Geoms import Domain

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    dim = 2

    elastic = Models.ElasIsot(2, 1e3, 0.3, planeStress=True)
    lmbda = elastic.get_lambda()
    mu = elastic.get_mu()

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    contour = Domain((0, 0), (1, 1), 1 / 8)

    if dim == 2:
        mesh = contour.Mesh_2D([], ElemType.QUAD4, isOrganised=True)
    else:
        mesh = contour.Mesh_Extrude(
            [], [0, 0, 1], [8], ElemType.HEXA8, isOrganised=True
        )

    # ----------------------------------------------
    # Formulations
    # ----------------------------------------------

    field = Field(mesh.groupElem, dim)

    def E(u: Field) -> FeArray:
        return 0.5 * (u.grad.T + u.grad)

    def S(u: Field) -> FeArray:
        Eps = E(u)
        return 2 * mu * Eps + lmbda * Trace(Eps) * np.eye(dim)

    @BiLinearForm
    def bilinear_form(u: Field, v: Field):

        Sig = S(u)
        Eps = E(v)

        return Sig.ddot(Eps)

    simu = Simulations.ElasticSimu(mesh, elastic)

    K = bilinear_form._assemble(field)

    diff = K - simu.Get_K_C_M_F()[0]

    weakFormManager = Models.WeakFormManager(field, bilinear_form)

    # simu = Simulations.WeakFormSimu(mesh, weakFormManager)

    nodes_x0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodes_x1 = mesh.Nodes_Conditions(lambda x, y, z: x == 1)
    simu.add_dirichlet(nodes_x0, [0], ["x"])
    simu.add_dirichlet(nodes_x1, [1], ["y"])

    simu.Solve()

    # ----------------------------------------------
    # Formulations
    # ----------------------------------------------

    # u = simu.u.reshape(-1, dim)[:, 0]
    u = simu.Result("ux")

    Display.Plot_Result(simu, u, plotMesh=True)

    x, y, z = mesh.coord.T
    u_an = 1 / 2 / np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    error = np.linalg.norm(u_an - simu.u) / np.linalg.norm(u_an)

    Display.plt.show()
