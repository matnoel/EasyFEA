# Copyright (C) 2021-2024 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Mesh a heterogeneous RVE with cracks."""

from EasyFEA import (Display, Folder, np, plt,
                     Mesher, ElemType,
                     Materials, Simulations)
from EasyFEA.Geoms import Point, Line, Domain, Circle

if __name__ == "__main__":

    Display.Clear()

    # ----------------------------------------------
    # Config
    # ----------------------------------------------

    L = 1
    elemType = ElemType.TRI3
    meshSize = L/40

    # ----------------------------------------------
    # Functions
    # ----------------------------------------------

    def Create_Crack(pt1: tuple[float, float], pt2: tuple[float, float], isOpen=True) -> Line:
        """Creates a crack.

        Parameters
        ----------
        pt1 : tuple[float, float]
            Point 1 (x, y)
        pt2 : tuple[float, float]
            Point 2 (x, y)
        isOpen : bool, optional
            The crack is open, by default True

        Returns
        -------
        Line
            The crack
        """

        assert len(pt1) == 2, "Must give 2 coordinates"
        assert len(pt2) == 2, "Must give 2 coordinates"

        pt1 = Point(*pt1)
        pt2 = Point(*pt2)

        return Line(pt1, pt2, meshSize, isOpen)

    def Create_Circle(pt: tuple[float, float], diam: float, isHollow=False) -> Circle:
        """Creates a circle.

        Parameters
        ----------
        pt : tuple[float, float]
            center point (x, y)
        diam : float
            diameter
        isHollow : bool, optional
            circle is hollow/empty, by default True

        Returns
        -------
        Circle
            The circle
        """
        
        assert len(pt) == 2, "Must give 2 coordinates"

        center = Point(*pt)

        return Circle(center, diam, meshSize/2, isHollow)

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    # contour
    contour = Domain(Point(), Point(L, L), meshSize)

    # circles
    circle1 = Create_Circle((L/4, L/4), L/10)
    circle2 = Create_Circle((3*L/4, L/4), L/8)
    circle3 = Create_Circle((3*L/4, 3*L/4), L/6)
    circle4 = Create_Circle((L/4, 3*L/4), L/5)
    circle5 = Create_Circle((L/2, L/2), L/3, isHollow=True)
    circle6 = Create_Circle((L, L), L/6, isHollow=True)

    inclusions = [circle1, circle2, circle3, circle4, circle5]

    # cracks
    crack1 = Create_Crack((3*L/5, L/10), (4*L/5, L/10))
    crack2 = Create_Crack((L/10, L/2), (2*L/10, L/2-L/10))
    crack3 = Create_Crack((8*L/10, 5*L/10), (8*L/10+L/20, 5*L/10+L/20))

    cracks = [crack1, crack2, crack3]

    # mesh
    mesh = Mesher().Mesh_2D(contour, inclusions, elemType, cracks, additionalSurfaces=[circle6])

    # ----------------------------------------------
    # Simu
    # ----------------------------------------------

    E = np.ones(mesh.Ne) * 10 # Mpa
    v = np.ones(mesh.Ne) * 0.45

    elements = list(set(range(mesh.Ne)) - set(mesh.Elements_Tags("S0")))
    E[elements] = 50
    v[elements] = 0.3

    mat = Materials.Elas_Isot(2, E, v, planeStress=False)

    simu = Simulations.ElasticSimu(mesh, mat)

    nodes_y0 = mesh.Nodes_Conditions(lambda x,y,z: y==0)
    nodes_yL = mesh.Nodes_Conditions(lambda x,y,z: y==L)

    simu.add_dirichlet(nodes_y0, [0, 0], ["x", "y"])
    simu.add_dirichlet(nodes_yL, [-.1, .2], ["x", "y"])

    simu.Solve()

    deformFactor = L/10 / simu.Result("displacement_norm").max()

    # ----------------------------------------------
    # Display
    # ----------------------------------------------

    # geoms = [contour]; geoms.extend(inclusions)
    # contour.Plot_Geoms(geoms)

    ax = Display.Plot_Result(simu, E, plotMesh=True, nodeValues=False, title="E")

    # Display.Plot_Mesh(mesh)
    Display.Plot_Tags(mesh, True)

    Display.Plot_Result(simu, "uy", deformFactor)
    Display.Plot_Result(simu, "Svm", deformFactor, plotMesh=True)

    Display.plt.show()