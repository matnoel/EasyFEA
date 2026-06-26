# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Contact2
========

Frictionless contact between a thin elastic arch strip and a rigid block, solved with the penalty method and Newton-Raphson.

The rigid block is treated as an obstacle: at every contact-surface Gauss point the normal gap ``gₙ`` to the obstacle surface is measured and, where it is negative (penetration), a penalty traction ``εₙ⟨-gₙ⟩ n`` resists it.These contributions are added to the elastic residual/tangent through ``Operators.NonLinear.PenaltyContact`` and the non-linear system ``A(u) Δu = -R(u)`` is solved with Newton at each load step.

Runs in 2D (arch strip) and 3D (extruded arch strip); set ``dim`` below. The non-linear simulation (``ElasticContact``) lives in ``_utils.py``.
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

    e = 10
    L = 3 * e
    t = 1  # strip thickness
    h = 10  # strip height
    r = 3  # fillet radius
    thickness = e / 2  # out-of-plane extent (3D)
    mS = t / 5 if dim == 2 else t

    N = 20  # load steps
    delta = 2 * t
    penalty = 1e6  # contact stiffness

    folder = Folder.Results_Dir()

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    # body
    # ----
    p1 = Point(-L / 2 - e)
    p2 = Point(-L / 2, r=r)
    p3 = Point(-e / 2, h - t, r=r)
    p4 = Point(e / 2, h - t, r=r)
    p5 = Point(L / 2, r=r)
    p6 = Point(L / 2 + e)

    lower = Points([p1, p2, p3, p4, p5, p6])
    upper = lower.copy()
    upper.Translate(dy=t)
    contour = Points(list(lower.points) + list(upper.points[::-1]), mS)

    if dim == 2:
        mesh = contour.Mesh_2D([], ElemType.TRI3)
    else:
        nz = max(1, round(thickness / mS))
        mesh = contour.Mesh_Extrude([], [0, 0, thickness], [nz], ElemType.TETRA4)

    # block
    # -----
    domain = Domain((-L / 2 - 2 * e, -5 * t), (L / 2 + 2 * e, 0))

    if dim == 2:
        block = domain.Mesh_2D([], ElemType.QUAD4, isOrganised=True)
    else:
        block = domain.Mesh_Extrude(
            [], [0, 0, thickness * 2], [1], ElemType.HEXA8, isOrganised=True
        )
        block.Translate(dz=-block.center[2] / 2)

    nodes_top = mesh.Nodes_Conditions(lambda x, y, z: y == h)
    nodes_contact = block.Nodes_Conditions(lambda x, y, z: y == 0)
    block.Set_Tag(nodes_contact, "contact")

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    material = Models.Elastic.Isotropic(
        dim, E=210000, v=0.3, planeStress=True, thickness=thickness
    )
    simu = RigidContact(mesh, material, penalty)
    simu._contactMesh = block

    print(f"Penalty contact solve in {dim}D (Newton per step):")
    for i in range(N):
        # udpate load
        dep = [0.0] * simu.dim
        dep[1] = -(i + 1) / N * delta
        # solve contact
        simu.Bc_Init()
        simu.add_dirichlet(nodes_top, dep, simu.Get_unknowns())
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
        PyVista.Plot(block, color="gray", alpha=0.4, plotter=plotter)
        PyVista.Plot_Elements(block, color="k", dimElem=1, linewidth=2, plotter=plotter)

    PyVista.Movie_func(Plot_Iter, N, folder=folder, filename="contact.gif")

    plotter = PyVista._Plotter()
    result = "uy"
    Plot_Iter(plotter, -1)
    plotter.show()
