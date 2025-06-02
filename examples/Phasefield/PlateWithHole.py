# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Performs damage simulation for a plate with a hole subjected to compression."""

from EasyFEA import (
    Display,
    Folder,
    plt,
    np,
    Tic,
    Mesher,
    ElemType,
    Mesh,
    Materials,
    Simulations,
    Paraview,
)
from EasyFEA.Geoms import Point, Domain, Circle

import multiprocessing

# Display.Clear()

useParallel = False
nProcs = 4  # number of processes in parallel

# ----------------------------------------------
# Configurations
# ----------------------------------------------

doSimu = True
meshTest = True
optimMesh = True

# Post processing
plotMesh = False
plotIter = False
plotResult = True
plotEnergy = False
showFig = True

saveParaview = False
makeMovie = False

# splits = ["Bourdin","Amor","Miehe","Stress"] # Splits Isotropes
# splits = ["He","AnisotStrain","AnisotStress","Zhang"] # Splits Anisotropes
# splits = ["Bourdin","Amor","Miehe","Stress","He","AnisotStrain","AnisotStress","Zhang"]
# splits = ["Zhang"]
# splits = ["AnisotStrain","AnisotStress","Zhang"]
splits = ["Miehe"]

regus = ["AT1"]  # ["AT1", "AT2"]
# regus = ["AT1", "AT2"]

solver = (
    Materials.PhaseField.SolverType.History
)  # ["History", "HistoryDamage", "BoundConstrain"]
maxIter = 1000
tolConv = 1e-0

# ----------------------------------------------
# Mesh
# ----------------------------------------------
l0 = 0.12e-3


def DoMesh(
    L: float, h: float, diam: float, thickness: float, l0: float, split: str
) -> Mesh:

    clC = l0 if meshTest else l0 / 2
    if optimMesh:
        clD = l0 * 4
        refineZone = diam * 1.5 / 2
        if split in ["Bourdin", "Amor"]:
            refineGeom = Domain(
                Point(0, h / 2 - refineZone), Point(L, h / 2 + refineZone), clC
            )
        else:
            refineGeom = Domain(
                Point(L / 2 - refineZone, 0), Point(L / 2 + refineZone, h), clC
            )
    else:
        clD = l0 if meshTest else l0 / 2
        refineGeom = None

    point = Point()
    domain = Domain(point, Point(L, h), clD)
    circle = Circle(Point(L / 2, h / 2), diam, clD, isHollow=True)

    folder = Folder.RESULTS_DIR
    ax = Display.Init_Axes()
    domain.Plot(ax, color="k", plotPoints=False)
    circle.Plot(ax, color="k", plotPoints=False)
    # if refineGeom != None:
    #     refineGeom.Plot(ax, color='k', plotPoints=False)
    # ax.scatter(((L+diam)/2, L/2), (h/2, (h+diam)/2), c='k')
    ax.axis("off")
    Display.Save_fig(folder, "sample", True)

    mesh = Mesher().Mesh_2D(domain, [circle], ElemType.TRI3, refineGeoms=[refineGeom])

    ax = Display.Plot_Mesh(mesh, lw=0.3, facecolors="white")
    ax.axis("off")
    ax.set_title("")
    Display.Save_fig(folder, "mesh", transparent=True)

    return mesh


# ----------------------------------------------
# Do Simu
# ----------------------------------------------
def DoSimu(split: str, regu: str):

    folder_save = Folder.PhaseField_Folder(
        "PlateWithHole_Benchmark",
        "Elas_Isot",
        split,
        regu,
        "DP",
        tolConv,
        solver,
        meshTest,
        optimMesh,
    )

    Display.MyPrint(folder_save, "green")

    # ----------------------------------------------
    # Geom
    # ----------------------------------------------

    L = 15e-3
    h = 30e-3
    thickness = 1
    diam = 6e-3

    unitU = "μm"
    unitF = "kN/mm"
    unit = 1e6

    if doSimu:

        mesh = DoMesh(L, h, diam, thickness, l0, split)

        # Get Nodes
        nodes_lower = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
        nodes_upper = mesh.Nodes_Conditions(lambda x, y, z: y == h)
        nodes_x0y0 = mesh.Nodes_Conditions(lambda x, y, z: (x == 0) & (y == 0))
        nodes_y0z0 = mesh.Nodes_Conditions(lambda x, y, z: (y == 0) & (z == 0))
        nodes_edges = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])
        nodes_upper = mesh.Nodes_Conditions(lambda x, y, z: y == h)

        # ----------------------------------------------
        # Boundary conditions
        # ----------------------------------------------

        threshold = 0.6
        # u_max = 25e-6
        u_max = 35e-6

        uinc0 = 8e-8
        uinc1 = 2e-8

        config = f"""
        while ud <= u_max:
        
        ud += uinc0 if simu.damage.max() < threshold else uinc1

        u_max = {u_max}
        uinc0 = {uinc0:.1e} (simu.damage.max() < {threshold})
        uinc1 = {uinc1:.1e}

        simu.add_dirichlet(nodes_lower, [0], ["y"])
        simu.add_dirichlet(nodes_x0y0, [0], ["x"])
        simu.add_dirichlet(nodes_upper, [-ud], ["y"])
        if dim == 3:
            simu.add_dirichlet(nodes_y0z0, [0], ["z"])
        """

        # ----------------------------------------------
        # Material
        # ----------------------------------------------
        E = 12e9
        v = 0.3
        planeStress = False
        material = Materials.Elas_Isot(2, E, v, planeStress, thickness)

        gc = 1.4
        pfm = Materials.PhaseField(material, split, regu, gc, l0, solver=solver)

        # ----------------------------------------------
        # Simulation
        # ----------------------------------------------
        simu = Simulations.PhaseFieldSimu(mesh, pfm, verbosity=False)

        simu.Results_Set_Bc_Summary(config)

        dofsY_upper = simu.Bc_dofs_nodes(nodes_upper, ["y"])

        def Apply_BC(ud: float):
            simu.Bc_Init()
            simu.add_dirichlet(nodes_lower, [0], ["y"])
            simu.add_dirichlet(nodes_x0y0, [0], ["x"])
            simu.add_dirichlet(nodes_upper, [-ud], ["y"])

        # INIT
        displacement = []
        force = []
        ud = -uinc0
        iter = 0
        nDetect = 0

        if plotIter:
            axIter = Display.Plot_Result(simu, "damage", nodeValues=True)

            force = np.asarray(force)
            displacement = np.asarray(displacement)
            _, axLoad = Display.Plot_Force_Displacement(
                force / unit, displacement * unit, f"ud [{unitU}]", f"f [{unitF}]"
            )

        while ud <= u_max:

            iter += 1
            ud += uinc0 if simu.damage.max() < threshold else uinc1

            Apply_BC(ud)

            u, d, Kglob, convergence = simu.Solve(tolConv, maxIter)
            simu.Save_Iter()

            # stop if the simulation does not converge
            if not convergence:
                break

            f = np.sum(Kglob[dofsY_upper, :] @ u)

            simu.Results_Set_Iteration_Summary(iter, ud * unit, unitU, ud / u_max, True)

            # Detect damaged edges
            if np.any(d[nodes_edges] >= 1):
                nDetect += 1
                if nDetect == 10:
                    break

            displacement = np.concatenate((displacement, [ud]))
            force = np.concatenate((force, [f]))

            if plotIter:
                Display.Plot_Result(simu, "damage", nodeValues=True, ax=axIter)
                plt.figure(axIter.figure)
                plt.pause(1e-12)

                force = np.asarray(force)
                displacement = np.asarray(displacement)
                Display.Plot_Force_Displacement(
                    force / unit,
                    displacement * unit,
                    f"ud [{unitU}]",
                    f"f [{unitF}]",
                    ax=axLoad,
                )[1]
                plt.figure(axLoad.figure)
                plt.pause(1e-12)

        # ----------------------------------------------
        # Saving
        # ----------------------------------------------
        force = np.asarray(force)
        displacement = np.asarray(displacement)
        print()
        Simulations.Save_pickle(
            (force, displacement), folder_save, "force-displacement"
        )
        simu.Save(folder_save)

    else:
        # ----------------------------------------------
        # Load
        # ----------------------------------------------
        simu: Simulations.PhaseFieldSimu = Simulations.Load_Simu(folder_save)
        force, displacement = Simulations.Load_pickle(folder_save, "force-displacement")

    # ----------------------------------------------
    # Post processing
    # ---------------------------------------------
    if plotEnergy:
        Display.Plot_Energy(simu, force, displacement, N=400, folder=folder_save)

    if plotResult:
        Display.Plot_BoundaryConditions(simu)
        Display.Plot_Iter_Summary(simu, folder_save, None, None)
        Display.Plot_Force_Displacement(
            force / unit,
            displacement * unit,
            f"ud [{unitU}]",
            f"f [{unitF}]",
            folder_save,
        )
        Display.Plot_Result(
            simu,
            "damage",
            nodeValues=True,
            colorbarIsClose=True,
            folder=folder_save,
            filename="damage",
        )

    if plotMesh:
        Display.Plot_Mesh(mesh)

    if saveParaview:
        Paraview.Make_Paraview(simu, folder_save)

    if makeMovie:
        Display.Movie_Simu(
            simu,
            "damage",
            folder_save,
            "damage.mp4",
            N=200,
            plotMesh=False,
            deformFactor=1.5,
        )

    if doSimu:
        Tic.Plot_History(folder_save, details=False)

    if showFig:
        plt.show()

    Tic.Clear()
    plt.close("all")


if __name__ == "__main__":

    # generates configs
    Splits = []
    Regus = []
    for split in splits.copy():
        for regu in regus.copy():
            Splits.append(split)
            Regus.append(regu)

    if useParallel:
        items = [(split, regu) for split, regu in zip(Splits, Regus)]
        with multiprocessing.Pool(nProcs) as pool:
            for result in pool.starmap(DoSimu, items):
                pass
    else:
        [DoSimu(split, regu) for split, regu in zip(Splits, Regus)]
