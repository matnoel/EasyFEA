# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Elas7
=====

Control lever for a molding machine used to blow plastic bottles.

References
----------

This example comes from:

* A French research article: `MMC et RDM comparées sur un cas de dimensionnement de levier <https://www.researchgate.net/publication/274440971_MMC_et_RDM_comparees_sur_un_cas_de_dimensionnement_de_levier>`_
* The book: `Mécanique des systèmes et des milieux déformables <https://www.editions-ellipses.fr/accueil/15577-29372-mecanique-des-systemes-et-des-milieux-deformables-cours-exercices-et-problemes-corriges-3e-edition-9782340096851.html>`_ (3rd edition)
"""

from EasyFEA import (
    Display,
    Folder,
    Models,
    np,
    ElemType,
    Simulations,
    PyVista,
    Paraview,
)
from EasyFEA.Geoms import Points, Point, Circle

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    # outputs
    folder = Folder.Join(Folder.RESULTS_DIR, "Elastic", "Elas7", mkdir=True)
    makeMovie = False
    makeParaview = True

    # geom
    dim = 3
    thickness = 25

    # sinusoidal load
    F = 13000  # N
    area = np.pi * 25 / 2 * thickness
    coefs = np.linspace(0, 1, 5)
    loads = F * np.sin(np.pi / 2 * coefs)

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    meshSize = 42 / 10

    # get the contour
    pt1 = Point(0, 80)
    pt2 = Point(-245, 21)
    pt3 = Point(-245, -21)
    pt4 = Point(0, -80)
    pt5 = Point(114, -80)
    pt6 = Point(114 + 13, -80 + 13)
    pt7 = Point(114 + 13, -67 / 2)
    pt8 = Point(114 + 13 + 25, -67 / 2)
    pt9 = Point(114 + 13 + 25, -34 / 2)
    pt10 = Point(114 + 13, -34 / 2, r=10)
    pt11 = Point(114 + 13, 34 / 2, r=10)
    pt12 = Point(114 + 13 + 25, 34 / 2)
    pt13 = Point(114 + 13 + 25, 67 / 2)
    pt14 = Point(114 + 13, 67 / 2)
    pt15 = Point(114 + 13, 80 - 13)
    pt16 = Point(114, 80)

    # fmt: off
    contour = Points([
            pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9, pt10,
            pt11, pt12, pt13, pt14, pt15, pt16,
        ],
        meshSize,
    )
    # fmt: on

    # get circles
    circle1 = Circle(Point(-220), 25, meshSize, isHollow=True)
    circle2 = Circle(Point(), 25, meshSize, isHollow=True)
    circle3 = Circle(Point(114 + 13 - 25, 80 - 25), 25, meshSize, isHollow=True)
    circle4 = Circle(Point(114 + 13 - 25, -80 + 25), 25, meshSize, isHollow=True)
    circle5 = Circle(Point(67), 84, meshSize, isHollow=True)

    inclusions = [circle1, circle2, circle3, circle4, circle5]

    # get the mesh
    if dim == 2:
        mesh = contour.Mesh_2D(inclusions, ElemType.TRI3)
    else:
        mesh = contour.Mesh_Extrude(inclusions, [0, 0, thickness], [5], ElemType.PRISM6)

    # get loading nodes
    nodesLoad = mesh.Nodes_Cylinder(circle1)
    nodesLoad = nodesLoad[mesh.coord[nodesLoad, 1] <= 0]

    # get fixed nodes
    nodesCircle2 = mesh.Nodes_Cylinder(circle2)
    nodesCircle3 = mesh.Nodes_Cylinder(circle3)
    nodesCircle4 = mesh.Nodes_Cylinder(circle4)
    fixedNodes = np.concat((nodesCircle2, nodesCircle3, nodesCircle4))

    # PyVista.Plot_Mesh(mesh).show()

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    material = Models.ElasIsot(
        dim, E=210000, v=0.25, planeStress=True, thickness=thickness
    )

    simu = Simulations.ElasticSimu(mesh, material)

    fixed = [0] * dim
    unknowns = simu.Get_unknowns()

    # loop over the load
    for load in loads:
        # Apply boundary conditions
        simu.Bc_Init()
        simu.add_dirichlet(fixedNodes, fixed, unknowns)
        simu.add_surfLoad(nodesLoad, [-load / area], ["y"])

        # Solve the simulation
        simu.Solve()

        # Save the iteration results
        simu.Save_Iter()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    print(simu)

    PyVista.Plot_BoundaryConditions(simu).show()

    PyVista.Plot(simu, "Svm", deformFactor=200, plotMesh=True, nodeValues=False).show()

    if makeMovie:
        PyVista.Movie_simu(
            simu,
            "Svm",
            folder,
            "Svm.gif",
            deformFactor=200,
            nodeValues=False,
            show_edges=True,
        )

    if makeParaview:
        Paraview.Save_simu(
            simu, folder, elementFields=["Svm", "Stress", "Strain", "Wdef_e"]
        )
