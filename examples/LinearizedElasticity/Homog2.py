# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Homog2
======

Perform homogenization on several RVE.
"""
# sphinx_gallery_thumbnail_number = -1

from EasyFEA import Display, Models, plt, np, ElemType, Simulations
from EasyFEA.Geoms import Point, Points, Line
from EasyFEA.fem import FeArray

from Homog1 import Compute_ukl

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    # use Periodic boundary conditions ?
    usePBC = True

    geom = "D666"  # hexagon
    # geom = "D2"  # rectangle
    # geom = "D6"

    hollowInclusion = True

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    N = 5
    elemType = ElemType.TRI6

    if geom == "D666":
        a = 1
        R = 2 * a / np.sqrt(3)
        r = R / np.sqrt(2) / 2
        phi = np.pi / 6

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # Create the contour geometrie
        p0 = Point(0, R)
        p1 = Point(-cos_phi * R, sin_phi * R)
        p2 = Point(-cos_phi * R, -sin_phi * R)
        p3 = Point(0, -R)
        p4 = Point(cos_phi * R, -sin_phi * R)
        p5 = Point(cos_phi * R, sin_phi * R)
        # edge length and area
        s = Line(p0, p1).length
        area = 3 * np.sqrt(3) / 2 * s**2

        contour = Points([p0, p1, p2, p3, p4, p5], s / N)
        corners = contour.points

        # Create the inclusion
        p6 = Point(0, (R - r))
        p7 = Point(-cos_phi * (R - r), sin_phi * (R - r))
        p8 = Point(-cos_phi * (R - r), -sin_phi * (R - r))
        p9 = Point(0, -(R - r))
        p10 = Point(cos_phi * (R - r), -sin_phi * (R - r))
        p11 = Point(cos_phi * (R - r), sin_phi * (R - r))
        inclusions = [Points([p6, p7, p8, p9, p10, p11], s / N, hollowInclusion)]

    elif geom == "D2":
        a = 1  # width
        b = 1.4  # height
        e = 1 / 10  # thickness
        area = a * b
        meshSize = e / N * 2

        # Create the contour geometry
        p0 = Point(-a / 2, b / 2)
        p1 = Point(-a / 2, -b / 2)
        p2 = Point(a / 2, -b / 2)
        p3 = Point(a / 2, b / 2)
        contour = Points([p0, p1, p2, p3], meshSize)
        corners = contour.points

        # Create the inclusion geometry
        p4 = p0 + [e, -e]
        p5 = p1 + [e, e]
        p6 = p2 + [-e, e]
        p7 = p3 + [-e, -e]
        inclusions = [Points([p4, p5, p6, p7], meshSize, hollowInclusion)]

    elif geom == "D6":
        a = 1  # height
        b = 2  # width
        c = np.sqrt(a**2 + b**2)

        e = b / 10  # thickness
        l1 = b / 2

        area = a * b

        theta = np.arctan(a / b)
        alpha = (np.pi / 2 - theta) / 2
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        phi = np.pi / 3
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        l2 = (b - l1 * sin_alpha) / 2
        hx = e / cos_phi / 4
        hy = e / sin_phi / 4

        # symmetry functions
        def Sym_x(point: Point) -> Point:
            return Point(-point.x, point.y)

        def Sym_y(point: Point) -> Point:
            return Point(point.x, -point.y)

        # points in the non-rotated base
        p0 = Point(
            l1 / 2 + l2 * cos_phi + e / 2 * cos_alpha, l2 * sin_phi - e / 2 * sin_alpha
        )
        p1 = p0 + [-e * cos_alpha, e * sin_alpha]
        p2 = Point(l1 / 2 - hy, hx)
        p3 = Sym_x(p2)
        p4 = Sym_x(p1)
        p5 = Sym_x(p0)
        p6 = Point(-l1 / 2 - np.sqrt(hx**2 + hy**2))
        p7 = Sym_y(p5)
        p8 = Sym_y(p4)
        p9 = Sym_y(p3)
        p10 = Sym_y(p2)
        p11 = Sym_y(p1)
        p12 = Sym_y(p0)
        p13 = Sym_x(p6)

        # do some tests to check if the geometry has been created correctly
        t1 = Line(p2, p10).length
        t2 = Line(p2, p13).length
        t3 = Line(p10, p13).length
        assert (
            np.abs(e - (t1 + t2 + t3) / 3) / e <= 1e-12
        )  # check that t1 = t2 = t3 = e
        t4 = Line(p0, p1).length
        assert np.abs(t4 - e) / e <= 1e-12  # check that t4 = e

        alpha = -alpha
        rot = np.array(
            [
                [np.cos(alpha), -np.sin(alpha), 0],
                [np.sin(alpha), np.cos(alpha), 0],
                [0, 0, 1],
            ]
        )

        rotate_points = []
        ax = Display.Init_Axes()
        for p, point in enumerate(
            [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13]
        ):
            assert isinstance(point, Point)

            newCoord = rot @ point.coord

            ax.scatter(*newCoord[:2], c="black")
            ax.text(*newCoord[:2], f"p{p}", c="black")

            rotate_points.append(Point(*newCoord))

        corners = [rotate_points[p] for p in [0, 1, 4, 5, 7, 8, 11, 12]]

        hollowInclusion = True  # dont change

        contour = Points(rotate_points, e / N * 2)

        inclusions = []

    else:
        raise Exception("Unknown geom")

    mesh = contour.Mesh_2D(inclusions, elemType)

    Display.Plot_Mesh(mesh, title="RVE")
    # Display.Plot_Tags(mesh)

    nodes_matrix = mesh.Nodes_Tags(["S0"])
    elements_matrix = mesh.Elements_Nodes(nodes_matrix)

    if not hollowInclusion:
        nodes_inclusion = mesh.Nodes_Tags(["S1"])
        elements_inclusion = mesh.Elements_Nodes(nodes_inclusion)

    nCorners = len(corners)
    nEdges = nCorners // 2

    if usePBC:
        nodes_kubc = mesh.Nodes_Points(corners)
        paired_nodes = mesh.Get_Paired_Nodes(nodes_kubc, True)
    else:
        nodes_kubc = mesh.Nodes_Tags([f"L{i}" for i in range(6)])
        paired_nodes = None

    # ----------------------------------------------
    # Material and simulation
    # ----------------------------------------------
    E = np.ones(mesh.Ne) * 70 * 1e9
    v = np.ones(mesh.Ne) * 0.45

    if not hollowInclusion:
        E[elements_inclusion] = 200 * 1e9
        v[elements_inclusion] = 0.3

    Display.Plot_Result(mesh, E * 1e-9, nodeValues=False, title="E [GPa]")
    Display.Plot_Result(mesh, v, nodeValues=False, title="v")

    material = Models.ElasIsot(2, E, v, planeStress=False)

    simu = Simulations.ElasticSimu(mesh, material)

    # ----------------------------------------------
    # Homogenization
    # ----------------------------------------------
    r2 = np.sqrt(2)
    E11 = np.array([[1, 0], [0, 0]])
    E22 = np.array([[0, 0], [0, 1]])
    E12 = np.array([[0, 1 / r2], [1 / r2, 0]])

    u11 = Compute_ukl(simu, nodes_kubc, E11, paired_nodes)
    u22 = Compute_ukl(simu, nodes_kubc, E22, paired_nodes)
    u12 = Compute_ukl(simu, nodes_kubc, E12, paired_nodes, True)

    u11_e = mesh.Locates_sol_e(u11, asFeArray=True)
    u22_e = mesh.Locates_sol_e(u22, asFeArray=True)
    u12_e = mesh.Locates_sol_e(u12, asFeArray=True)

    # ----------------------------------------------
    # Effective elasticity tensor (C_hom)
    # ----------------------------------------------
    U_e = FeArray.zeros(*u11_e.shape, 3)

    U_e[..., 0] = u11_e
    U_e[..., 1] = u22_e
    U_e[..., 2] = u12_e

    matrixType = "mass"
    wJ_e_pg = mesh.Get_weightedJacobian_e_pg(matrixType)
    B_e_pg = mesh.Get_B_e_pg(matrixType)

    C_Mat = Models.Reshape_variable(material.C, *B_e_pg.shape[:2])

    C_hom = (wJ_e_pg * C_Mat @ B_e_pg @ U_e).sum((0, 1)) / mesh.area

    print(f"c1111 = {C_hom[0, 0]}")
    print(f"c1122 = {C_hom[0, 1]}")
    print(f"c1212 = {C_hom[2, 2] / 2}")

    plt.show()
