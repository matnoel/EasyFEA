import Display
from Interface_Gmsh import Interface_Gmsh, ElemType
from Geom import Point, Line, Circle, PointsList, Domain

if __name__ == '__main__':

    Display.Clear()

    contour = Domain(Point(), Point(1,1,1))

    def DoMesh(dim, elemType):
        if dim == 2:
            mesh = Interface_Gmsh().Mesh_2D(contour, [], elemType, isOrganised=True)
        elif dim == 3:
            mesh = Interface_Gmsh().Mesh_3D(contour, [], [0, 0, 1], [10], elemType=elemType, isOrganised=True)

        Display.Plot_Mesh(mesh)

    [DoMesh(2, elemType) for elemType in ElemType.get_2D()]

    [DoMesh(3, elemType) for elemType in ElemType.get_3D()]

    contour.Plot()

    Display.plt.show()