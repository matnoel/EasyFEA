# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Contact4
========

Frictionless contact between two DEFORMABLE elastic bodies (a block and a smaller indenter) solved with **Lagrange multipliers**.

The two bodies are meshed independently and merged into one mesh with distinct nodes on the shared ``y=R`` interface (``mergePoints=False``). Each indenter contact node is tied (node-to-segment, along the contact normal) to the block surface via a ``LagrangeCondition``, enforcing non-penetration **exactly** through the bordered saddle-point system — no penalty, no penetration, and the two bodies' dofs are coupled exactly. The problem stays linear elastic (one solve per load step).

The ties are bilateral (no release on separation), which suits this near-full, compressive flat-punch contact; an active set would be the general extension. ``MutliBodyContact`` lives in ``_utils.py``.
"""

import numpy as np

from EasyFEA import Terminal, Folder, Models, ElemType, PyVista
from EasyFEA.Geoms import Domain

from _utils import MutliBodyContact

if __name__ == "__main__":
    Terminal.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    R = 10  # block size
    a = 3  # indenter size
    meshSize = R / 10
    useSymmetry = True
    elemType = ElemType.QUAD4

    N = 30  # load steps
    load = 5000  # peak downward traction on the indenter top

    folder = Folder.Results_Dir()

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    # body
    # ----
    x0 = 0.0 if useSymmetry else -R / 2
    body = Domain((x0, 0), (R / 2, R), meshSize)
    mesh1 = body.Mesh_2D([], elemType, isOrganised=True)

    # indenter
    # --------
    x0 = 0.0 if useSymmetry else -a / 2
    indenter = Domain((x0, R), (a / 2, R + a), meshSize)
    mesh2 = indenter.Mesh_2D([], elemType, isOrganised=True)

    list_mesh = [mesh1, mesh2]
    list_label = ["body", "indenter"]
    mesh, mapping = mesh1.Merge(
        [mesh1, mesh2],
        constructUniqueElements=False,
        mergePoints=False,
        return_mapping=True,
    )
    assert mesh1.Nn + mesh2.Nn == mesh.Nn

    # import surfaces
    for i, map in enumerate(mapping):
        nodes = map[list_mesh[i].Nodes_Tags("S0")]
        mesh.Set_Tag(nodes, list_label[i])

    nodes_bottom = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
    nodes_up = mesh.Nodes_Conditions(lambda x, y, z: y == R + a)
    nodes_contact = mesh.Nodes_Conditions(lambda x, y, z: y == R)
    # symmetry plane (skip the clamped corner: a duplicate Dirichlet would make the
    # Lagrange-bordered system singular)
    nodes_sym = mesh.Nodes_Conditions(lambda x, y, z: (x == 0) & (y > 0))
    mesh.Set_Tag(nodes_contact, "contact")

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    # heterogeneous Young's modules (steel + aluminum)
    E = np.ones(mesh.Ne) * 210000
    E[mesh.Elements_Tags("body")] = 70000

    material = Models.Elastic.Isotropic(2, E=E, v=0.3, planeStress=False)
    simu = MutliBodyContact(mesh, material)
    simu.solver = (
        "scipy"  # Lagrange saddle-point: needs a solver that pivots zero diagonals
    )
    simu.Set_contact("indenter", "body")  # tie indenter contact nodes to the block

    print("Two-body Lagrange contact (one linear solve per step):")
    for i in range(N):
        sigY = -(i + 1) / N * load
        simu.Bc_Init()
        simu.add_dirichlet(nodes_bottom, [0, 0], simu.Get_unknowns())
        simu.add_dirichlet(nodes_sym, [0], ["x"])
        simu.add_surfLoad(nodes_up, [sigY], ["y"])
        simu.Add_contact_conditions()
        simu.Solve()
        simu.Save_Iter()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    PyVista.Movie_simu(simu, "Svm", folder=folder, filename="Svm.gif", N=N)

    PyVista.Plot(simu, "uy", plotMesh=True, deformFactor=1).show()
