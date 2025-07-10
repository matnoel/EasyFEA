# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
LinearElasticity2
=================

A cantilever beam undergoing bending deformation in dynamic.

Note that this simulation is also performed in `examples/Elastic/Elas1.py`.
"""

from EasyFEA import Folder, Display, ElemType, Models, Simulations, PyVista, np
from EasyFEA.fem import Field, BiLinearForm, FeArray, Sym_Grad, Trace
from EasyFEA.Geoms import Domain

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    dim = 3

    # outputs
    makeMovie = True
    folder = Folder.Join(Folder.RESULTS_DIR, "FunctionBased", "LinearElasticity2")

    # geom
    L = 120  # mm
    h = 13
    F = -800  # N

    # model
    elastic = Models.ElasIsot(dim, 210000, 0.3, planeStress=True, thickness=h)
    lmbda = elastic.get_lambda()
    mu = elastic.get_mu()
    rho = 8100 * 1e-9

    # load
    Tmax = 0.5
    N = 50
    dt = Tmax / N
    time = -dt

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    contour = Domain((0, 0), (L, h), h / 3)

    if dim == 2:
        mesh = contour.Mesh_2D([], ElemType.QUAD4, isOrganised=True)
    else:
        mesh = contour.Mesh_Extrude(
            [], [0, 0, h], [3], ElemType.HEXA8, isOrganised=True
        )

    nodes_x0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodes_xL = mesh.Nodes_Conditions(lambda x, y, z: x == L)

    # ----------------------------------------------
    # Formulations
    # ----------------------------------------------

    field = Field(mesh.groupElem, dim)

    def S(u: Field) -> FeArray:
        Eps = Sym_Grad(u)
        return 2 * mu * Eps + lmbda * Trace(Eps) * np.eye(dim)

    @BiLinearForm
    def computeK(u: Field, v: Field):
        Sig = S(u)
        Eps = Sym_Grad(v)
        return Sig.ddot(Eps)

    @BiLinearForm
    def ComputeM(u: Field, v: Field):
        return rho * u.dot(v)

    @BiLinearForm
    def ComputeC(u: Field, v: Field):
        M = rho * u.dot(v)
        K = S(u).ddot(Sym_Grad(v))
        C = K * 1e-3 + M * 1e-3
        return C

    weakForms = Models.WeakFormManager(field, computeK, ComputeC, ComputeM)

    # ----------------------------------------------
    # Simulations
    # ----------------------------------------------

    simu = Simulations.WeakFormSimu(mesh, weakForms)

    # static simulation
    simu.add_dirichlet(nodes_x0, [0] * dim, simu.Get_unknowns())
    simu.add_dirichlet(nodes_xL, [-10], ["y"])

    simu.Solve()

    # dynamic simulation
    simu.Solver_Set_Hyperbolic_Algorithm(dt)
    simu.Save_Iter()

    simu.Bc_Init()
    simu.add_dirichlet(nodes_x0, [0] * dim, simu.Get_unknowns())

    while time <= Tmax:

        time += dt

        simu.Solve()
        simu.Save_Iter()

        print(f"{time:.3f} s", end="\r")

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    Display.Plot_Result(simu, "uy", 1, plotMesh=True)

    if makeMovie:
        PyVista.Movie_simu(
            simu,
            "uy",
            folder,
            "uy.gif",
            deformFactor=1,
            show_edges=True,
            N=400,
            nodeValues=True,
        )

    Display.plt.show()
