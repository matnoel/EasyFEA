# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
MeshOptim1
==========

Mesh optimization using the ZZ1 criterion for a bending bracket.
"""
# sphinx_gallery_thumbnail_number = -2

from EasyFEA import (
    Display,
    Folder,
    Models,
    Tic,
    plt,
    np,
    Mesher,
    ElemType,
    Mesh,
    Simulations,
    Paraview,
    PyVista,
)
from EasyFEA.Geoms import Point, Points
from EasyFEA.fem import Calc_projector

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    dim = 2

    # outputs
    folder = Folder.Join(Folder.RESULTS_DIR, "Meshes", "MeshOptim1")
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
    treshold = (
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
            return Mesher().Mesh_2D(points, [], elemType, [], [refineGeom])
        else:
            return Mesher().Mesh_Extrude(
                points, [], [0, 0, b], [], elemType, [], [refineGeom]
            )

    # Construct the initial mesh
    mesh = DoMesh()
    PyVista.Plot_Mesh(mesh).show()

    # ----------------------------------------------
    # Material and Simulation
    # ----------------------------------------------
    material = Models.ElasIsot(dim, E=210000, v=0.3, thickness=b)
    simu = Simulations.ElasticSimu(mesh, material)
    simu.rho = 8100 * 1e-9

    def DoSimu(refineGeom: str):
        simu.mesh = DoMesh(refineGeom)

        # get the nodes
        nodes_Fixed = simu.mesh.Nodes_Conditions(lambda x, y, z: x == 0)
        nodes_Load = simu.mesh.Nodes_Conditions(lambda x, y, z: x == L)

        # do the simulation
        simu.Bc_Init()
        simu.add_dirichlet(
            nodes_Fixed, [0] * dim, simu.Get_unknowns(), description="Fixed"
        )
        simu.add_surfLoad(nodes_Load, [-surfLoad], ["y"])

        simu.Solve()

        simu.Save_Iter()

        return simu

    simu = Simulations.Mesh_Optim_ZZ1(DoSimu, folder, treshold, iterMax, 1 / 10)

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    PyVista.Plot_Mesh(simu.mesh).show()
    PyVista.Plot(simu, "ZZ1_e", nodeValues=False, nColors=11).show()

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

        ax = Display.Plot_Result(
            mesh0, np.linalg.norm(u0, axis=1), plotMesh=True, title="u0"
        )
        ax.plot(*mesh1.coord[:, :dim].T, ls="", marker="+", c="k", label="new nodes")
        ax.legend()
        Display.Plot_Result(
            mesh1, np.linalg.norm(uProj, axis=1), plotMesh=True, title="uProj"
        )

    if makeParaview:
        Paraview.Save_simu(simu, folder, nodeFields=["ZZ1_e"])

    if makeMovie:

        def func(plotter, n):
            simu.Set_Iter(n)

            PyVista.Plot_Mesh(simu, plotter=plotter)
            # PyVista.Plot_BoundaryConditions(simu, plotter=plotter)

            zz1 = simu._Calc_ZZ1()[0]

            plotter.add_title(f"ZZ1 = {zz1 * 100:.2f} %")

        PyVista.Movie_func(func, len(simu.results), folder, "bracket.gif")

    Tic.Plot_History()
    plt.show()
