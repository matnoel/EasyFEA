# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Modal1
======

Modal analysis of a wall structure.
"""

from EasyFEA import (
    Display,
    np,
    ElemType,
    Materials,
    Simulations,
    PyVista,
)
from EasyFEA.Geoms import Domain

from scipy.sparse import linalg, eye

if __name__ == "__main__":

    Display.Clear()

    dim = 3
    isFixed = True
    Nmode = 3

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    contour = Domain((0, 0), (1, 1), 1 / 10)
    thickness = 1 / 10

    if dim == 2:
        mesh = contour.Mesh_2D([], ElemType.QUAD9, isOrganised=True)
    else:
        mesh = contour.Mesh_Extrude(
            [], [0, 0, -thickness], [2], ElemType.HEXA27, isOrganised=True
        )
    nodesY0 = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
    nodesSupY0 = mesh.Nodes_Conditions(lambda x, y, z: y > 0)

    Display.Plot_Mesh(mesh)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    material = Materials.ElasIsot(dim, planeStress=True, thickness=thickness)

    simu = Simulations.ElasticSimu(mesh, material)

    simu.Solver_Set_Hyperbolic_Algorithm(0.1)

    K, C, M, F = simu.Get_K_C_M_F()

    if isFixed:
        simu.add_dirichlet(nodesY0, [0] * dim, simu.Get_unknowns())
        known, unknown = simu.Bc_dofs_known_unknown(simu.problemType)
        K_t = K[unknown, :].tocsc()[:, unknown].tocsr()
        M_t = M[unknown, :].tocsc()[:, unknown].tocsr()

    else:
        K_t = K + K.min() * eye(K.shape[0]) * 1e-12
        M_t = M

    # eigenValues, eigenVectors = linalg.eigs(K_t, Nmode, M_t, which="SR")
    eigenValues, eigenVectors = linalg.eigs(K_t, Nmode, M_t, sigma=0, which="LR")

    eigenValues = eigenValues.real
    eigenVectors = eigenVectors.real
    freq_t = np.sqrt(eigenValues) / 2 / np.pi

    # ----------------------------------------------
    # Plot modes
    # ----------------------------------------------
    for n in range(eigenValues.size):

        if isFixed:
            mode = np.zeros((mesh.Nn, dim))
            mode[nodesSupY0, :] = np.reshape(eigenVectors[:, n], (-1, dim))
        else:
            mode = np.reshape(eigenVectors[:, n], (-1, dim))

        simu._Set_u_n(simu.problemType, mode.ravel())
        simu.Save_Iter()

        sol = np.linalg.norm(mode, axis=1)
        deformFactor = 1 / 5 / np.abs(sol).max()

        plotter = PyVista.Plot(simu, opacity=0.5)
        PyVista.Plot(simu, None, deformFactor, opacity=0.8, color="r", plotter=plotter)
        plotter.add_title(f"mode {n+1}")
        plotter.show()

    axModes = Display.Init_Axes()
    axModes.plot(np.arange(eigenValues.size), freq_t, ls="", marker=".")
    axModes.set_xlabel("modes")
    axModes.set_ylabel("freq [Hz]")
    axModes.grid()

    Display.plt.show()
