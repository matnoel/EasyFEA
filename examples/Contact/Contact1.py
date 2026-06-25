# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Contact1
========

Hertz-type frictionless contact between a deformable block and a rigid indenter, solved as a genuine non-linear problem with Newton-Raphson and a **penalty** regularisation of the unilateral (Signorini) condition.

Rather than detecting penetrating nodes and imposing their projection as Dirichlet conditions (an active-set trick that produces no contact force and makes the contact nodes stick), the rigid indenter is treated as an obstacle: at
every contact-surface Gauss point the normal gap ``gₙ`` to the obstacle surface is measured and, where it is negative (penetration), a penalty traction ``εₙ⟨-gₙ⟩ n`` resists it. These contributions are added to the elastic
residual/tangent through ``Operators.NonLinear.PenaltyContact`` and the non-linear system ``A(u) Δu = -R(u)`` is solved with Newton at each load step.

Runs in 2D (block + half-disc punch) and 3D (box + half-cylinder punch); set ``dim`` below. The non-linear simulation (``ElasticContact``) lives in ``_utils.py``.

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

    R = 10  # block size
    thickness = R / 3  # out-of-plane extent (3D)
    meshSize = R / 20 if dim == 2 else R / 8

    N = 30  # load steps
    inc = 1 / N  # indenter descent per step
    penalty = 1e7  # contact stiffness (larger -> less penetration)

    folder = Folder.Results_Dir()

    # ----------------------------------------------
    # Deformable body: block (top face is the contact surface)
    # ----------------------------------------------
    body = Domain(
        (-R / 2, 0),
        (R / 2, R),
        meshSize,
    )
    if dim == 2:
        mesh = body.Mesh_2D(
            [],
            ElemType.QUAD4,
            isOrganised=True,
        )
    else:
        nz = max(1, round(thickness / meshSize))
        mesh = body.Mesh_Extrude(
            [],
            [0, 0, thickness],
            [nz],
            ElemType.HEXA8,
            isOrganised=True,
        )
    nodes_y0 = mesh.Nodes_Conditions(lambda x, y, z: y == 0)

    # ----------------------------------------------
    # Rigid obstacle: indenter (filleted lower surface = a half-disc / half-cylinder)
    # ----------------------------------------------
    r = R / 2
    contour = Points(
        [
            Point(-R / 2, R, r=r),
            Point(R / 2, R, r=r),
            Point(R / 2, 2 * R),
            Point(-R / 2, 2 * R),
        ]
    )
    if dim == 2:
        obstacle = contour.Mesh_2D([], ElemType.TRI3)
    else:
        obstacle = contour.Mesh_Extrude([], [0, 0, thickness], [nz], ElemType.TETRA4)
    # lower (contact) surface of the punch: below the end of the fillets
    nodes_obstacle = obstacle.Nodes_Conditions(lambda x, y, z: y <= R + r + 1e-6)

    obstacle.Set_Tag(nodes_obstacle, "contact")

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    material = Models.Elastic.Isotropic(
        dim, E=210000, v=0.3, planeStress=True, thickness=thickness
    )
    simu = RigidContact(mesh, material, penalty)

    obstacles = [obstacle]
    print(f"Penalty contact solve in {dim}D (Newton per step):")
    for i in range(N):
        obstacle = obstacle.copy()
        obstacle.Translate(dy=-inc)  # lower the rigid indenter
        obstacles.append(obstacle)

        simu.Bc_Init()
        simu.add_dirichlet(nodes_y0, [0] * dim, simu.Get_unknowns())
        simu._contactMesh = obstacle
        simu.Solve()
        simu.Save_Iter()
        print(
            f"  step {i + 1:2d}/{N}  Eps max = {simu.Result('Strain').max() * 100:5.2f} %"
        )

    print(simu)

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    def DoAnim(plotter, n):
        simu.Set_Iter(n)
        PyVista.Plot(
            simu, "Svm", 1, color="k", plotter=plotter, nColors=10, show_grid=True
        )
        PyVista.Plot(obstacles[n], plotter=plotter, plotMesh=True, alpha=0.2)

    PyVista.Movie_func(DoAnim, N, folder=folder, filename="contact.gif")

    plotter = PyVista.Plot(simu, "uy", plotMesh=True, deformFactor=1)
    PyVista.Plot_Mesh(obstacle, alpha=0.4, plotter=plotter)
    plotter.show()
    plt.show()
