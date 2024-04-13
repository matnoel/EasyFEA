"""Refined mesh in zones."""

from EasyFEA import Display, Mesher, ElemType
from EasyFEA.Geoms import Point, Circle, Domain

if __name__ == '__main__':

    Display.Clear()

    L = 1
    meshSize = L/4

    contour = Domain(Point(), Point(L, L), meshSize)
    circle = Circle(Point(L/2,L/2), L/3, meshSize)
    inclusions = [circle]

    refine1 = Domain(Point(0, L), Point(L, L*0.8), meshSize/8)
    refine2 = Circle(circle.center, L/2, meshSize/8)
    refine3 = Circle(Point(), L/2, meshSize/8)
    refineGeoms = [refine1, refine2, refine3]

    def DoMesh(dim, elemType):
        if dim == 2:
            mesh = Mesher().Mesh_2D(contour, inclusions, elemType, refineGeoms=refineGeoms)
        elif dim == 3:
            mesh = Mesher().Mesh_Extrude(contour, inclusions, [0, 0, -L], [3], elemType, refineGeoms=refineGeoms)

        Display.Plot_Mesh(mesh)

    [DoMesh(2, elemType) for elemType in ElemType.Get_2D()]

    [DoMesh(3, elemType) for elemType in ElemType.Get_3D()]

    geoms = [contour, circle, refine1, refine2, refine3]
    contour.Plot_Geoms(geoms)

    Display.plt.show()