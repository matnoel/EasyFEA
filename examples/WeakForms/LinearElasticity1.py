# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
LinearElasticity1
=================

A cantilever beam undergoing bending deformation.

Note that this simulation is also performed in `examples/Elastic/Elas1.py`.
"""

from EasyFEA import Display, ElemType, Models, Simulations, np
from EasyFEA.FEM import Field, BiLinearForm, FeArray, Sym_Grad, Trace
from EasyFEA.Geoms import Domain

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    dim = 2

    # geom
    L = 120  # mm
    h = 13

    # load
    F = -800  # N

    # model
    E = 210000  # MPa
    v = 0.3

    elastic = Models.Elastic.Isotropic(dim, E, v, planeStress=True, thickness=h)
    lmbda = elastic.get_lambda()
    mu = elastic.get_mu()

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    contour = Domain((0, 0), (L, h), h / 3)

    if dim == 2:
        mesh = contour.Mesh_2D([], ElemType.QUAD9, isOrganised=True)
    else:
        mesh = contour.Mesh_Extrude(
            [], [0, 0, h], [3], ElemType.HEXA27, isOrganised=True
        )

    nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L)

    # ----------------------------------------------
    # Formulations
    # ----------------------------------------------

    field = Field(mesh.groupElem, dim)

    def S(u: Field) -> FeArray:
        Eps = Sym_Grad(u)
        return 2 * mu * Eps + lmbda * Trace(Eps) * np.eye(dim)

    @BiLinearForm
    def ComputeK(u: Field, v: Field):
        Sig = S(u)
        Eps = Sym_Grad(v)
        return Sig.ddot(Eps)

    weakForms = Models.WeakForms(field, ComputeK)

    # ----------------------------------------------
    # Simulations
    # ----------------------------------------------

    simu = Simulations.WeakForms(mesh, weakForms)

    simu.add_dirichlet(nodesX0, [0] * dim, simu.Get_unknowns())
    simu.add_surfLoad(nodesXL, [F / h**2], ["y"])

    simu.Solve()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    Display.Plot_Result(simu, "uy", 10, plotMesh=True)

    def von_mises_stress(u):
        Stress = S(u)
        if dim == 2:
            xx = Stress[..., 0, 0]
            yy = Stress[..., 1, 1]
            xy = Stress[..., 0, 1]
            return np.sqrt(xx**2 + yy**2 - xx * yy + 3 * xy**2)
        else:
            Stress_d = Stress - 1 / 3 * Trace(Stress) * np.eye(3)
            return np.sqrt(3 / 2 * Stress_d.ddot(Stress_d))

    Svm_e = field.Evaluate_e(von_mises_stress, simu.u)

    Display.Plot_Result(simu, Svm_e, plotMesh=True, ncolors=11)

    Display.plt.show()
