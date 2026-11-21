# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
MeshOptim1
==========

Mesh optimization using the ZZ1 criterion for a bending bracket.
"""

# sphinx_gallery_thumbnail_number = -2

import matplotlib.pyplot as plt
import numpy as np

from EasyFEA import (
    Terminal,
    Matplotlib,
    Folder,
    Models,
    ElemType,
    Mesh,
    Simulations,
    PyVista,
    IO,
    MatrixType,
)
from EasyFEA.Geoms import Point, Points
from EasyFEA.FEM import Calc_projector, Kinematics, Mesh_Optim


def ZZ1(simu: Simulations.Elastic) -> tuple[float, np.ndarray]:
    """ZZ1 error, globally and per element: the strain energy of the raw stress against the node-smoothed one [F. Pled, Vers une stratégie robuste ... ingénierie mécanique, p. 20-21]."""
    mesh, material = simu.mesh, simu.material
    thickness = material.thickness if simu.dim == 2 else 1.0
    list_kinematics = [
        Kinematics(groupElem, simu.displacement, MatrixType.rigi)
        for groupElem in mesh.Get_list_groupElem(simu.dim)
    ]
    list_Sigma = [material.Compute_Sigma(kinematics) for kinematics in list_kinematics]
    # element stresses averaged at the nodes
    Sigma_n = mesh.Get_Node_Values(
        np.concatenate([np.mean(Sigma, 1) for Sigma in list_Sigma])
    )
    list_W_e, list_Ws_e = [], []
    for kinematics, Sigma in zip(list_kinematics, list_Sigma):
        groupElem = kinematics.groupElem
        wJ = thickness * groupElem.Get_weightedJacobian_e_pg(MatrixType.rigi)
        Epsilon = kinematics.Compute_Epsilon()
        Sigma_s = groupElem.Interpolate_e_pg(Sigma_n)
        list_W_e.append((wJ * (0.5 * Sigma @ Epsilon)).integrate())
        list_Ws_e.append((wJ * (0.5 * Sigma_s @ Epsilon)).integrate())
    W_e, Ws_e = np.concatenate(list_W_e), np.concatenate(list_Ws_e)
    W = W_e.sum()
    return float(abs(W - Ws_e.sum()) / W), np.abs(Ws_e - W_e) / W


if __name__ == "__main__":
    Terminal.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    dim = 2

    # outputs
    folder = Folder.Results_Dir()
    plotProj = False
    makeMovie = True
    makeParaview = False

    # geom
    L = 120  # mm
    h = L * 0.3
    b = h

    # load
    P = 800  # N
    lineLoad = P / h  # N/mm
    surfLoad = P / h / b  # N/mm2

    # criteria
    threshold = (
        1 / 100 if dim == 2 else 0.04
    )  # Target error for the optimization process
    iterMax = 20  # Maximum number of iterations
    coef = 1 / 10  # Scaling coefficient for the optimization process

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    if dim == 2:
        elemType = ElemType.TRI3
    else:
        elemType = ElemType.TETRA4
    N = 2

    pt1 = Point(isOpen=True, r=-10)
    pt2 = Point(x=L)
    pt3 = Point(x=L, y=h)
    pt4 = Point(x=h, y=h, r=10)
    pt5 = Point(x=h, y=L)
    pt6 = Point(y=L)
    pt7 = Point(x=h, y=h)

    points = Points([pt1, pt2, pt3, pt4, pt5, pt6], h / N)

    PyVista.Plot_Geoms(points.Get_Contour()).show()

    def DoMesh(refineGeom=None) -> Mesh:
        """Function used to generate the mesh."""
        if dim == 2:
            return points.Mesh_2D([], elemType, [], [refineGeom])
        else:
            return points.Mesh_Extrude([], [0, 0, b], [], elemType, [], [refineGeom])

    # Construct the initial mesh
    mesh = DoMesh()
    PyVista.Plot_Mesh(mesh).show()

    # ----------------------------------------------
    # Material and Simulation
    # ----------------------------------------------
    material = Models.Elastic.Isotropic(dim, E=210000, v=0.3, thickness=b)
    simu = Simulations.Elastic(mesh, material)
    simu.rho = 8100 * 1e-9

    def Solve_on(mesh: Mesh) -> tuple[bool, np.ndarray]:
        """Solves on `mesh`; converged once the ZZ1 error is below `threshold`."""
        simu.mesh = mesh

        # get the nodes
        nodes_Fixed = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
        nodes_Load = mesh.Nodes_Conditions(lambda x, y, z: x == L)

        # do the simulation
        simu.Bc_Init()
        simu.add_dirichlet(
            nodes_Fixed, [0] * dim, simu.Get_unknowns(), description="Fixed"
        )
        simu.add_surfLoad(nodes_Load, [-surfLoad], ["y"])

        simu.Solve()

        simu.Save_Iter()

        error, error_e = ZZ1(simu)
        print(f"error = {error * 100:.3f} %")
        return error < threshold, error_e

    mesh = Mesh_Optim(DoMesh, Solve_on, iterMax, coef)
    PyVista.Plot_BoundaryConditions(simu).show()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    PyVista.Plot_Mesh(mesh).show()
    PyVista.Plot(simu, ZZ1(simu)[1], nodeValues=False, nColors=11).show()

    if plotProj:
        simu.Set_Iter(0)
        mesh0 = simu.mesh
        u0 = np.reshape(simu.displacement, (mesh0.Nn, -1))

        simu.Set_Iter(1)
        mesh1 = simu.mesh

        proj = Calc_projector(mesh0, mesh1)
        uProj = np.zeros((mesh1.Nn, dim), dtype=float)
        for d in range(dim):
            uProj[:, d] = proj @ u0[:, d]

        ax = Matplotlib.Plot(
            mesh0, np.linalg.norm(u0, axis=1), plotMesh=True, title="u0"
        )
        ax.plot(*mesh1.coord[:, :dim].T, ls="", marker="+", c="k", label="new nodes")
        ax.legend()
        Matplotlib.Plot(
            mesh1, np.linalg.norm(uProj, axis=1), plotMesh=True, title="uProj"
        )

    if makeParaview:
        IO.Paraview.Save_simu(simu, folder)

    if makeMovie:

        def func(plotter, n):
            simu.Set_Iter(n)

            PyVista.Plot_Mesh(simu, plotter=plotter)

            zz1 = ZZ1(simu)[0]

            plotter.add_title(f"ZZ1 = {zz1 * 100:.2f} %")

        PyVista.Movie_func(func, simu.Niter, folder, "bracket.gif")

    Matplotlib.Plot_Tic_History()
    plt.show()
