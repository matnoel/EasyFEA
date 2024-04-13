"""Meshing of a specimen for a spatially oriented tensile test."""

from EasyFEA import Display, Mesher, np
from EasyFEA.Geoms import Point, Line, CircleArc, Contour, Domain

if __name__  == '__main__':

    Display.Clear()

    dim = 3

    L = 1
    H = 2
    e = L * 0.5

    p1 = Point(-L/2)
    p2 = Point(L/2)
    p3 = p2 + [0, H]
    p4 = p1 + [0, H]

    p5 = Point(e/2, H/2)
    p6 = Point(-e/2, H/2)

    l1 = Line(p1, p2)
    l2 = CircleArc(p2, p3, P=p5)
    l3 = Line(p3,p4)
    l4 = CircleArc(p4,p1,P=p6)

    contour = Contour([l1,l2,l3,l4])
    contour2 = Domain(p1-[0,H/2], p2)
    contour3 = contour2.Copy(); contour3.Translate(dy=H+H/2)

    surfaces = [(contour2, []), (contour3, [])]

    ax = contour.Plot()
    contour2.Plot(ax)
    contour3.Plot(ax)

    if dim == 2:
        mesh = Mesher().Mesh_2D(contour, isOrganised=True, elemType='QUAD4', surfaces=surfaces)
    else:        
        mesh = Mesher().Mesh_Extrude(contour, [], [0,0,e], [3], isOrganised=True, elemType='HEXA8', surfaces=surfaces)

    oldArea = mesh.area
    mesh.Rotate(-45, mesh.center)
    assert np.abs(mesh.area - oldArea)/oldArea <= 1e-12
    mesh.Rotate(45, mesh.center, (1,0))
    assert np.abs(mesh.area - oldArea)/oldArea <= 1e-12

    Display.Plot_Mesh(mesh)

    Display.plt.show()