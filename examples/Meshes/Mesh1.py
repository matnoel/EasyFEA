# Copyright (C) 2021-2024 Université Gustave Eiffel. All rights reserved.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.txt for more information.

"""Meshing a domain."""

from EasyFEA import Display, Mesher, ElemType
from EasyFEA.Geoms import Point, Domain

if __name__ == '__main__':

    Display.Clear()

    contour = Domain(Point(), Point(1,1,1))

    def DoMesh(dim, elemType):
        if dim == 2:
            mesh = Mesher().Mesh_2D(contour, [], elemType, isOrganised=True)
        elif dim == 3:
            mesh = Mesher().Mesh_Extrude(contour, [], [0, 0, 1], [10], elemType=elemType, isOrganised=True)

        Display.Plot_Mesh(mesh)

    [DoMesh(2, elemType) for elemType in ElemType.Get_2D()]

    [DoMesh(3, elemType) for elemType in ElemType.Get_3D()]

    contour.Plot()

    Display.plt.show()