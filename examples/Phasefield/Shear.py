# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Performs a damage simulation for a plate subjected to shear."""

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
    PyVista,
)
from EasyFEA.Geoms import Point, Points, Domain, Line, Contour

import multiprocessing

# Display.Clear()

useParallel = False
nProcs = 4  # number of processes in parallel

# ----------------------------------------------
# Configuration
# ----------------------------------------------
dim = 2

doSimu = True
# Mesh
meshTest = True
openCrack = True
optimMesh = True

# phasefield
maxIter = 1000

tolConv = 1e-0  # 1e-1, 1e-2, 1e-3

pfmSolver = Materials.PhaseField.SolverType.History

# splits = ["Bourdin","Amor","Miehe","Stress"] # Splits Isotropes
# splits = ["He","AnisotStrain","AnisotStress","Zhang"] # Splits Anisotropes sans bourdin
# splits = ["Bourdin","Amor","Miehe","Stress","He","AnisotStrain","AnisotStress","Zhang"]
splits = ["Amor"]

# regus = ["AT1", "AT2"]
regus = ["AT1"]  # "AT1", "AT2"

# PostProcessing
plotResult = True
showResult = True
plotMesh = False
plotEnergy = False
saveParaview = False
Nparaview = 400
makeMovie = False

# ----------------------------------------------
# Mesh
# ----------------------------------------------
L = 1e-3
# m
l0 = 1e-5
thickness = 1 if dim == 2 else 0.1 / 1000


def DoMesh(split: str) -> Mesh:
    # meshSize
    clC = l0 if meshTest else l0 / 2
    if optimMesh:
        # a coarser mesh can be used outside the refined zone
        clD = clC * 3
        # refines the mesh in the area where the crack will propagate
        gap = L * 0.05
        h = L if split == "Bourdin" else L / 2 + gap
        refineDomain = Domain(Point(L / 2 - gap, 0), Point(L, h, thickness), clC)
    else:
        clD = clC
        refineDomain = None

    # geom
    pt1 = Point()
    pt2 = Point(L)
    pt3 = Point(L, L)
    pt4 = Point(0, L)
    contour = Points([pt1, pt2, pt3, pt4], clD)

    if dim == 2:
        ptC1 = Point(0, L / 2, isOpen=openCrack)
        ptC2 = Point(L / 2, L / 2)
        cracks = [Line(ptC1, ptC2, clC, isOpen=openCrack)]
    if dim == 3:
        meshSize = clD if optimMesh else clC
        ptC1 = Point(0, L / 2, 0, isOpen=openCrack)
        ptC2 = Point(L / 2, L / 2, 0)
        ptC3 = Point(L / 2, L / 2, thickness)
        ptC4 = Point(0, L / 2, thickness, isOpen=openCrack)
        l1 = Line(ptC1, ptC2, meshSize, openCrack)
        l2 = Line(ptC2, ptC3, meshSize, False)
        l3 = Line(ptC3, ptC4, meshSize, openCrack)
        l4 = Line(ptC4, ptC1, meshSize, openCrack)
        cracks = [Contour([l1, l2, l3, l4], isOpen=openCrack)]

    # folder = Folder.Join("",results=True)
    # ax = Display.Init_Axes()
    # contour.Get_Contour().Plot(ax, color='k', plotPoints=False)
    # cracks[0].Plot(ax, color='k', plotPoints=False)
    # # if refineDomain != None:
    # #     refineDomain.Plot(ax, color='k', plotPoints=False)
    # ax.axis('off')
    # Display.Save_fig(folder,"sample",True)

    if dim == 2:
        mesh = Mesher().Mesh_2D(contour, [], ElemType.TRI3, cracks, [refineDomain])
    elif dim == 3:
        mesh = Mesher().Mesh_Extrude(
            contour,
            [],
            [0, 0, thickness],
            [4],
            ElemType.PRISM6,
            cracks,
            [refineDomain],
            additionalLines=[l1],
        )

    return mesh


# ----------------------------------------------
# Simu
# ----------------------------------------------
def DoSimu(split: str, regu: str):

    # Builds the path to the folder based on the problem data
    folderName = "Shear_Benchmark"
    if dim == 3:
        folderName += "_3D"
    folder_save = Folder.PhaseField_Folder(
        folderName,
        "Elas_Isot",
        split,
        regu,
        "DP",
        tolConv,
        pfmSolver,
        meshTest,
        optimMesh,
        not openCrack,
    )

    Display.MyPrint(folder_save, "green")

    if doSimu:

        mesh = DoMesh(split)

        # Nodes recovery
        nodes_crack = mesh.Nodes_Conditions(lambda x, y, z: (y == L / 2) & (x <= L / 2))
        nodes_upper = mesh.Nodes_Conditions(lambda x, y, z: y == L)
        nodes_lower = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
        nodes_left = mesh.Nodes_Conditions(lambda x, y, z: (x == 0) & (y > 0) & (y < L))
        nodes_right = mesh.Nodes_Conditions(
            lambda x, y, z: (x == L) & (y > 0) & (y < L)
        )

        # Builds edge nodes
        nodes_edges = []
        for nodes in [nodes_lower, nodes_right, nodes_upper]:
            nodes_edges.extend(nodes)

        # ----------------------------------------------
        # Material
        # ----------------------------------------------
        material = Materials.Elas_Isot(
            dim, E=210e9, v=0.3, planeStress=False, thickness=thickness
        )
        Gc = 2.7e3  # J/m2
        pfm = Materials.PhaseField(material, split, regu, Gc, l0, pfmSolver)

        # ----------------------------------------------
        # Boundary conditions
        # ----------------------------------------------
        u_inc = 5e-8 if meshTest else 1e-8
        N = 400 if meshTest else 2000

        loadings = np.linspace(u_inc, u_inc * N, N, endpoint=True)

        config = f"""
        u_inc = {u_inc:.1e}
        N = {N}

        for iter, dep in enumerate(loadings):

        loadings = np.linspace(u_inc, u_inc*N, N, endpoint=True)

        if not openCrack:
            simu.add_dirichlet(nodes_crack, [1], ["d"], problemType="damage")
        simu.add_dirichlet(nodes_left, [0], ["y"])
        simu.add_dirichlet(nodes_right, [0], ["y"])
        simu.add_dirichlet(nodes_upper, [dep,0], ["x","y"])
        simu.add_dirichlet(nodes_lower, [0]*dim, simu.Get_dofs())
        """

        def Loading(dep):
            """Boundary conditions"""
            simu.Bc_Init()
            if not openCrack:
                simu.add_dirichlet(nodes_crack, [1], ["d"], problemType="damage")
            simu.add_dirichlet(nodes_left, [0], ["y"])
            simu.add_dirichlet(nodes_right, [0], ["y"])
            simu.add_dirichlet(nodes_upper, [dep, 0], ["x", "y"])
            simu.add_dirichlet(nodes_lower, [0] * dim, simu.Get_unknowns())

        # ----------------------------------------------
        # Simulation
        # ----------------------------------------------
        simu = Simulations.PhaseFieldSimu(mesh, pfm, verbosity=False)
        simu.Results_Set_Bc_Summary(config)

        dofsX_upper = simu.Bc_dofs_nodes(nodes_upper, ["x"])

        # INIT
        N = len(loadings)
        nDetect = 0
        displacement = []
        force = []
        for iter, dep in enumerate(loadings):

            # apply new boundary conditions
            Loading(dep)

            # solve and save iter
            u, _, Kglob, converg = simu.Solve(tolConv, maxIter, convOption=2)
            simu.Save_Iter()

            # print iter solution
            simu.Results_Set_Iteration_Summary(iter, dep * 1e6, "µm", iter / N, True)

            # If the solver has not converged, stop the simulation.
            if not converg:
                break

            # resulting force on upper edge
            f = np.sum(Kglob[dofsX_upper, :] @ u)

            displacement.append(dep)
            force.append(f)

            # Detect damaged edges
            if np.any(simu.damage[nodes_edges] >= 1):
                nDetect += 1
                if nDetect == 10:
                    break

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
        simu: Simulations.PhaseFieldSimu = Simulations.Load_Simu(folder_save)
        force, displacement = Simulations.Load_pickle(folder_save, "force-displacement")

    # ----------------------------------------------
    # PostProcessing
    # ---------------------------------------------
    if plotResult:
        Display.Plot_Iter_Summary(simu, folder_save, None, None)
        Display.Plot_BoundaryConditions(simu)
        Display.Plot_Force_Displacement(
            force * 1e-6, displacement * 1e6, "ud [µm]", "f [kN/mm]", folder_save
        )
        ax = Display.Plot_Result(
            simu,
            "damage",
            nodeValues=True,
            plotMesh=False,
            folder=folder_save,
            filename="damage",
            ncolors=25,
        )

        # ax = Display.Plot_Result(simu, "damage", 1.5, ncolors=21, clim=(0,0.9))
        # ax.axis('off'); ax.set_title("")
        # Display.Save_fig(folder, "deform damage")

    if plotMesh:
        # PyVista.Plot_Mesh(simu.mesh).show()
        # PyVista.Plot(simu, "ux", show_edges=True).show()
        ax = Display.Plot_Mesh(simu.mesh, lw=0.3, facecolors="white")
        ax.axis("off")
        ax.set_title("")
        Display.Save_fig(folder_save, "mesh", transparent=True)

    if saveParaview:
        Paraview.Make_Paraview(simu, folder_save, Nparaview)

    if makeMovie:
        # PyVista.Plot_Mesh(simu.mesh).show()
        simu.Set_Iter(-1)
        nodes_upper = simu.mesh.Nodes_Conditions(lambda x, y, z: y == L)
        depMax = simu.Result("displacement_norm")[nodes_upper].max()
        deformFactor = L * 0.1 / depMax

        def Func(plotter, n):

            simu.Set_Iter(n)

            grid = PyVista._pvGrid(simu, "damage", deformFactor)

            tresh = grid.threshold((0, 0.8))

            PyVista.Plot(
                tresh,
                "damage",
                deformFactor,
                show_edges=True,
                plotter=plotter,
                clim=(0, 1),
            )

        PyVista.Movie_func(Func, len(simu.results), folder_save, "damage.mp4")

    if plotEnergy:
        Display.Plot_Energy(simu, N=400, folder=folder_save)

    Tic.Resume()

    if doSimu:
        Tic.Plot_History(folder_save, False)

    if showResult:
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
