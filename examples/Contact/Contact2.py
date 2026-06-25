# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Contact2
========

Frictionless contact between a thin elastic arch strip and a rigid block,
solved with the penalty method and Newton-Raphson.

The strip is clamped at its top and the rigid block is raised into it from below;
its filleted underside progressively flattens onto the block. As in ``Contact1``,
the contact load is carried entirely through the penalty term
(``Operators.NonLinear.PenaltyContact``) — driving the approach through the rigid
obstacle's motion rather than a prescribed displacement of the strip avoids the
spurious zero-strain translation mode that would otherwise let the strip pass
straight through the block.

The whole strip boundary is handed to the contact operator; only the Gauss points
that actually penetrate the block (negative gap) contribute a contact force.

Runs in 2D (arch strip) and 3D (extruded arch strip); set ``dim`` below. The
non-linear simulation (``ElasticContact``) lives in ``_utils.py``.

WARNING
-------
The assumption of small displacements is highly questionable for this simulation.
"""

import matplotlib.pyplot as plt

from EasyFEA import Terminal, Folder, Models, ElemType, PyVista
from EasyFEA.Geoms import Point, Domain, Points

from _utils import RigidContact

if __name__ == "__main__":
    Terminal.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    dim = 3  # 2 or 3
    result = "displacement_norm"

    e = 10
    L = 3 * e
    t = 1  # strip thickness
    h = 10  # strip height
    r = 3  # fillet radius
    depth = e / 2  # out-of-plane extent (3D)
    mS = t / 5 if dim == 2 else t

    N = 20  # load steps
    inc = 2 * t / N  # block rise per step
    penalty = 1e6  # contact stiffness

    folder = Folder.Results_Dir()

    # ----------------------------------------------
    # Meshes
    # ----------------------------------------------
    # deformable body: thin arch strip built from a lower polyline and its upward offset
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

    # rigid obstacle: block below the strip (its top, y=0, is the contact surface)
    domain = Domain((-L / 2 - 2 * e, -5 * t), (L / 2 + 2 * e, 0))

    if dim == 2:
        mesh = contour.Mesh_2D([], ElemType.TRI3)
        obstacle = domain.Mesh_2D([], ElemType.QUAD4, isOrganised=True)
    else:
        nz = max(1, round(depth / mS))
        mesh = contour.Mesh_Extrude([], [0, 0, depth], [nz], ElemType.TETRA4)
        obstacle = domain.Mesh_Extrude(
            [], [0, 0, depth], [1], ElemType.HEXA8, isOrganised=True
        )

    nodes_top = mesh.Nodes_Conditions(lambda x, y, z: y == h)
    nodes_obstacle = obstacle.Nodes_Conditions(lambda x, y, z: y == 0)
    obstacle.Set_Tag(nodes_obstacle, "contact")

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    material = Models.Elastic.Isotropic(
        dim, E=210000, v=0.3, planeStress=True, thickness=depth
    )
    simu = RigidContact(mesh, material, penalty)
    simu._contactMesh = obstacle

    print(f"Penalty contact solve in {dim}D (Newton per step):")
    for i in range(N):
        simu.Bc_Init()

        dep = [0.0] * simu.dim
        dep[1] = -i * inc
        simu.add_dirichlet(nodes_top, dep, simu.Get_unknowns())
        simu.Solve()
        simu.Save_Iter()
        print(
            f"  step {i + 1:2d}/{N}  Eps max = {simu.Result('Strain').max() * 100:5.2f} %"
        )

    print(simu)

    # ----------------------------------------------
    # Plot
    # ----------------------------------------------
    def DoAnim(plotter, n):
        simu.Set_Iter(n)
        PyVista.Plot(
            simu, result, 1, color="k", plotter=plotter, nColors=10, show_grid=True
        )
        PyVista.Plot(obstacle, plotter=plotter, plotMesh=True, alpha=0.2)

    PyVista.Movie_func(DoAnim, N, folder=folder, filename="contact.gif")

    plotter = PyVista.Plot(simu, "uy", plotMesh=True, deformFactor=1)
    PyVista.Plot_Mesh(obstacle, alpha=0.4, plotter=plotter)
    plotter.show()
    plt.show()
