import numpy as np

import Display
from Interface_Gmsh import Interface_Gmsh, GroupElem
from Geom import Point, PointsList, Circle, Line, CircleArc, Contour

Display.Clear()

radius = 20
w = 10
a = 4
height = 100

meshSize = radius/5

pt1 = Point(0, 0)
pt2 = Point(radius, 0)
pt3 = Point(radius, w)
pt4 = Point(radius, height-w)
pt5 = Point(radius, height)
pt6 = Point(0, height)
pt7 = Point(0, height/2+a)
pt8 = Point(0, height/2-a)

centerArc1 = Point(height, height/2)
centerArc2 = Point(0, height/2)

line1 = Line(pt1, pt2, meshSize)
line2 = Line(pt2, pt3, meshSize)
line3 = CircleArc(pt3, centerArc1, pt4, meshSize)
line4 = Line(pt4, pt5, meshSize)
line5 = Line(pt5, pt6, meshSize)
line6 = Line(pt6, pt7, meshSize)
line7 = CircleArc(pt7, centerArc2, pt8, meshSize)
line8 = Line(pt8, pt1, meshSize)

contour = Contour([line1, line2, line3, line4, line5, line6, line7, line8])
inclusions = []

axis = Line(Point(-1), Point(-1,height))

ax = Display.plt.subplots()[1]
ax.plot(axis.coordo[:,:2][:,0], axis.coordo[:,:2][:,1], c='black', ls='-.', label='axis')
# ax.scatter(*line7.center.coordo[:2])
mesh2D = Interface_Gmsh().Mesh_2D(contour, inclusions)
ax.legend()
Display.Plot_Model(mesh2D, ax=ax)

angle = np.pi*2 *2/3
nLayers = np.abs(angle * radius) // meshSize

def DoMesh(dim, elemType):
    mesh = Interface_Gmsh().Mesh_Revolve(contour, inclusions, axis, angle, nLayers, elemType)
    Display.Plot_Mesh(mesh)

[DoMesh(3, elemType) for elemType in GroupElem.get_Types3D()]

Display.plt.show()