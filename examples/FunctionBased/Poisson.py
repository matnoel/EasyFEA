# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Poisson
=======

Poisson equation with unit load.

Reference: https://scikit-fem.readthedocs.io/en/latest/listofexamples.html#example-1-poisson-equation-with-unit-load
"""

from EasyFEA import Display, ElemType, np
from EasyFEA.fem import Field, BiLinearForm, LinearForm
from EasyFEA.Geoms import Domain

import scipy.sparse.linalg as sla

# ----------------------------------------------
# Class
# ----------------------------------------------


if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    contour = Domain((0, 0), (1, 1), 1 / 2**6)

    mesh = contour.Mesh_2D([], ElemType.TRI3, isOrganised=True)

    from EasyFEA import Materials, Simulations

    mat = Materials.Thermal(2, 1, 0)
    simu = Simulations.ThermalSimu(mesh, mat, useIterativeSolvers=False)

    nodes = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])
    simu.add_dirichlet(nodes, [0], ["t"])
    simu.add_neumann(mesh.nodes, [np.ones(mesh.Nn) / mesh.Nn], ["t"])

    simu.Solve()

    # Display.Plot_Result(simu, "thermal")

    # Display.Plot_Mesh(mesh)

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

    A = bilinear_form.Assemble(field)

    b = linear_form.Assemble(field)

    dofsKnown = nodes

    x = np.zeros(mesh.Nn)
    x[dofsKnown] = 0

    dofsUnknown = [node for node in mesh.nodes if node not in dofsKnown]

    Ai = A[dofsUnknown, :].tocsc()
    Aii = Ai[:, dofsUnknown].tocsr()
    Aic = Ai[:, dofsKnown].tocsr()
    bi = b.toarray().ravel()[dofsUnknown]
    xc = x[dofsKnown]

    bDirichlet = Aic @ xc

    xi = sla.cg(Aii, bi - bDirichlet)[0]

    # apply result to global vector
    x[dofsUnknown] = xi

    Display.Plot_Result(mesh, x)

    pass
