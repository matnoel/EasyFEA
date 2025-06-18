# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Mesh7
=====

Meshing a 3D part in revolution.
"""
# sphinx_gallery_thumbnail_number = 2

from EasyFEA import Display, ElemType, np
from EasyFEA.Geoms import Point, Points, Circle, Line

if __name__ == "__main__":

    Display.Clear()

    width = 1
    height = 2
    radius = 1

    meshSize = width / 5

    pt1 = Point(radius, 0, r=width / 3)
    pt2 = Point(radius + width, 0, r=width / 3)
    pt3 = Point(radius + width, height, r=-width / 3)
    pt4 = Point(radius, height, r=-width / 3)

    contour = Points([pt1, pt2, pt3, pt4], meshSize)

    circle1 = Circle(Point(width / 2 + radius, height * 1 / 4), width / 3, width / 10)
    circle2 = Circle(Point(width / 2 + radius, height * 3 / 4), width / 3, width / 10)
    circle3 = Circle(
        Point(width / 2 + radius, height / 2), width / 3, width / 10, False
    )
    inclusions = [circle1, circle2, circle3]

    axis = Line(Point(), Point(radius / 3, height))
    axis.name = "rot axis"

    geoms = [contour.Get_Contour()]
    geoms.extend([circle1, circle2, circle3])
    geoms.append(axis)
    contour.Plot_Geoms(geoms)

    angle = 360 * 4 / 6

    perimeter = angle * np.pi / 180 * radius

    layers = [perimeter // meshSize]

    mesh = contour.Mesh_Revolve(inclusions, axis, angle, layers, ElemType.PRISM6)
    Display.Plot_Mesh(mesh)

    Display.plt.show()
