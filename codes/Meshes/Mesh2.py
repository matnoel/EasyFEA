import Display
from Interface_Gmsh import Interface_Gmsh, ElemType
from Geoms import Point, Line, Circle, Points, Domain

if __name__ == '__main__':

    Display.Clear()

    h = 180
    N = 5

    pt1 = Point()
    pt2 = Point(x=h)
    pt3 = Point(y=h)

    contour = Points([pt1, pt2, pt3], h / N)

    def DoMesh(dim, elemType):
        if dim == 2:
            mesh = Interface_Gmsh().Mesh_2D(contour, [], elemType)
        elif dim == 3:
            mesh = Interface_Gmsh().Mesh_3D(contour, [], [0, 0, 2*h], [10], elemType=elemType)

        Display.Plot_Mesh(mesh)

    [DoMesh(2, elemType) for elemType in ElemType.get_2D()]

    [DoMesh(3, elemType) for elemType in ElemType.get_3D()]

    contour.Get_Contour().Plot()

    Display.plt.show()