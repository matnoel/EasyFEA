import numpy as np

import Display
from Geom import Point, PointsList, Circle, Line, CircleArc, Contour
from Interface_Gmsh import Interface_Gmsh

Display.Clear()

radius = 10
w = 10
height = 50

meshSize = radius/5

pt1 = Point()
pt2 = Point(radius, 0)
pt3 = Point(radius, w)
pt4 = Point(radius, height-w)
pt5 = Point(radius, height)
pt6 = Point(0, height)

centerArc = Point(height, height/2)

line1 = Line(pt1, pt2, meshSize)
line2 = Line(pt2, pt3, meshSize)
line3 = CircleArc(pt3, centerArc, pt4, meshSize)
line4 = Line(pt4, pt5, meshSize)
line5 = Line(pt5, pt6, meshSize)
line6 = Line(pt6, pt1, meshSize)

contour = Contour([line1, line2, line3, line4, line5, line6])
inclusions = []

# axis = Line(Point(), Point(0,height))
axis = Line(Point(-1), Point(-1,height))

ax = Display.plt.subplots()[1]
ax.plot(axis.coordo[:,:2][:,0], axis.coordo[:,:2][:,1], c='black', ls='-.', label='revolve axis')
# ax.scatter(*line3.center.coordo[:2])
mesh2D = Interface_Gmsh().Mesh_2D(contour, inclusions)
ax.legend()
Display.Plot_Model(mesh2D, ax=ax)

angle = np.pi*2 *2/3
nLayers = np.abs(angle * radius) // meshSize

mesh = Interface_Gmsh().Mesh_Revolve(contour, inclusions, axis, angle, 1, elemType="TETRA4")
Display.Plot_Mesh(mesh)

mesh = Interface_Gmsh().Mesh_Revolve(contour, inclusions, axis, angle, nLayers, elemType="PRISM6")
Display.Plot_Mesh(mesh)

mesh = Interface_Gmsh().Mesh_Revolve(contour, inclusions, axis, angle, nLayers, elemType="HEXA8")
Display.Plot_Mesh(mesh)

Display.plt.show()