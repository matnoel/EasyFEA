# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.


"""
TopologyOptimisation1
=====================

An educational implementation of topology optimization inspired by `Week 10- Topology Optimisation — A Step-by-Step Tutorial <https://github.com/MCM-QMUL/TopOpt_teach/blob/main/Week10_topology_optimisation_tutorial_step%20by%20step.ipynb>`_ created by (Dr Wei Tan, Queen Mary University of London), which in turn builds upon the seminal 88-line topology optimization MATLAB code by Ole Sigmund (2001), published in *Structural and Multidisciplinary Optimization*, 21(2), pp. 120–127.
"""
# sphinx_gallery_thumbnail_number = 3

import matplotlib.pyplot as plt
import numpy as np

from EasyFEA import Display, Folder, PyVista, ElemType, Models, Simulations
from EasyFEA.FEM import FeArray, Field, BiLinearForm, Sym_Grad, Trace
from EasyFEA.Geoms import Domain

if __name__ == "__main__":

    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    dim = 2

    L, H = 60, 30
    # L, H = 120, 60

    # optim topo
    iterMax = 60
    volFrac = 0.4
    penal = 3
    rMin = 3

    # outputs
    generateMovie = True
    folder = Folder.Results_Dir()

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    meshSize = 1 if dim == 2 else H / 10
    contour = Domain((0, 0), (L, H), meshSize)
    assert H / meshSize % 2 == 0

    if dim == 2:
        mesh = contour.Mesh_2D([], ElemType.QUAD4, isOrganised=True)
    else:
        mesh = contour.Mesh_Extrude(
            [], [0, 0, H], [H / meshSize], ElemType.HEXA8, isOrganised=True
        )

    nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)

    zMean = 0 if dim == 2 else H / 2
    nodesLoad = mesh.Nodes_Point((L, H / 2, zMean))
    # nodesLoad = mesh.Nodes_Conditions(lambda x, y, z: y == H)

    # ----------------------------------------------
    # Mesh-Independence Sensitivity Filter (Sigmund, 1998)
    # ----------------------------------------------

    # get the coordinates of each elements
    coord_e = mesh.coord[mesh.connect].mean(1)

    # compute Hij
    elements = mesh.groupElem.elements
    Hij = np.array(
        [
            np.maximum(0, rMin - np.linalg.norm(coord_e[i] - coord_e, axis=-1) + 1e-12)
            for i in range(mesh.Ne)
        ]
    )

    # # plot neighbor elements
    # elem = 100
    # ax = Display.Plot_Mesh(mesh, alpha=0)
    # Display.Plot_Elements(
    #     mesh, mesh.connect[Hij[elem] != 0].ravel(), 2, color="blue", ax=ax
    # )
    # Display.Plot_Elements(mesh, mesh.connect[elem].ravel(), 2, ax=ax)

    # ----------------------------------------------
    # Formulations
    # ----------------------------------------------

    elastic = Models.Elastic.Isotropic(dim, E=1, v=0.3, planeStress=True)
    mu = elastic.get_mu()
    lmbda = elastic.get_lambda()

    def S(u: Field) -> FeArray:
        Eps = Sym_Grad(u)
        return 2 * mu * Eps + lmbda * Trace(Eps) * np.eye(dim)

    p_e = np.ones(mesh.Ne, dtype=float) * volFrac

    @BiLinearForm
    def ComputeK(u: Field, v: Field):
        Sig = S(u)
        Eps = Sym_Grad(v)
        return Sig.ddot(Eps)

    @BiLinearForm
    def ComputePenalizedK(u: Field, v: Field):
        simpScaling = FeArray.asfearray(np.reshape(p_e**penal, (-1, 1)))
        return simpScaling * ComputeK(u, v)

    field = Field(mesh.groupElem, dim)
    model = Models.WeakForms(field, ComputePenalizedK, thickness=H)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    simu = Simulations.WeakForms(mesh, model)
    simu._Solver_Set_PETSc4Py_Options("none", "lu")

    simu.add_dirichlet(nodesX0, [0] * dim, simu.Get_unknowns())
    simu.add_neumann(nodesLoad, [-1], ["y"])

    # ----------------------------------------------
    # Optim topo
    # ----------------------------------------------

    err = 1.0
    list_compliance: list[float] = []
    list_p_e: list[np.ndarray] = []
    iter = 0

    while err > 0.005 and iter < iterMax:
        iter += 1
        pOld_e = p_e.copy()

        # solve u
        simu.Need_Update()
        u = simu.Solve()
        simu.Save_Iter()

        # compute compliance for elements
        u_e = field.groupElem.Locates_sol_e(u, dim)
        K_e = ComputeK.Integrate_e(field)
        uKu_e = np.einsum("ei,eij,ej->e", u_e, K_e, u_e, optimize="optimal")
        c = (p_e**penal * uKu_e).sum()

        # compute sensitivity for elements
        dCdP_e = -(penal * (p_e ** (penal - 1.0)) * uKu_e)

        # use sensitivity filter
        dCdP_e = np.einsum("ij,j,j", Hij, p_e, dCdP_e) / (
            np.einsum("i,ij", p_e, Hij) + 1e-12
        )

        # OC update (enforce volume)
        lmin, lmax = 0.0, 1e5
        pmin, pmax = 0.001, 1.0
        move = 0.2
        pNew_e = p_e.copy()

        while (lmax - lmin) > 1e-4 * (lmax + lmin + 1e-16):
            lmid = 0.5 * (lmax + lmin)
            candidate = p_e * np.sqrt(np.maximum(-dCdP_e / lmid, 1e-9))
            # Apply move limits and physical bounds [pmin, pmax]
            pNew_e = np.maximum(
                pmin,
                np.maximum(
                    p_e - move,
                    np.minimum(pmax, np.minimum(p_e + move, candidate)),
                ),
            )
            # update lambda to fit volume fraction
            if pNew_e.sum() - volFrac * mesh.Ne > 0:
                lmin = lmid
            else:
                lmax = lmid

        # get updated density and compliance
        p_e = pNew_e
        list_p_e.append(p_e)
        list_compliance.append(c)

        # compute relative error : || p_e - pOld_e || / || pOld_e ||
        err = np.linalg.norm(p_e - pOld_e) / np.linalg.norm(pOld_e)

        Display.MyPrint(
            f"Iteration {str(iter).zfill(len(str(iterMax)))}, compliance = {c:.3e}, volume fraction = {p_e.mean():.3f}, err = {err:.3e}",
            end="\r",
        )

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    axC = Display.Init_Axes()
    axC.plot(range(len(list_compliance)), list_compliance, ls="-", marker=".")
    axC.set_xlabel("Iteration")
    axC.set_ylabel("Compliance")
    plt.show()

    PyVista.Plot_BoundaryConditions(simu).show()

    def get_thresh(p_e: np.ndarray, min=0.5, max=1.0):
        grid = PyVista._pvMesh(mesh, p_e, nodeValues=False)
        for result in simu.Results_Available():
            grid[result] = simu.Result(result).reshape(mesh.Nn, -1)
        thresh = grid.threshold((min, max))
        return thresh

    thresh = get_thresh(p_e)
    PyVista.Plot(thresh, color="k").show()
    PyVista.Plot(thresh, "uy").show()

    if generateMovie:

        def Func(plotter: PyVista.pv.Plotter, iter):
            simu.Set_Iter(iter)
            thresh = get_thresh(list_p_e[iter])
            plotter.add_title(
                f"{str(iter+1).zfill(len(str(iterMax)))}/{len(list_compliance)}"
            )
            PyVista.Plot(thresh, color="k", plotter=plotter)

        PyVista.Movie_func(Func, len(list_compliance), folder, "optim.gif")
