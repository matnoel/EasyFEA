# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Performs damage simulation on a CT specimen."""

from EasyFEA import (Display, Folder, plt, np,
                     Mesher, ElemType,
                     Materials, Simulations)
from EasyFEA.Geoms import Point, Points, Circle, Line, Contour, Domain
from EasyFEA import PyVista_Interface as pvi
from EasyFEA import Paraview_Interface

Display.Clear()

folder = Folder.Dir(__file__)

# ----------------------------------------------
# Config
# ----------------------------------------------

dim = 2

doSimu = True
meshTest = True
optimMesh = True
plotGeom = False
plotIter = False

makeParaview = False
makeMovie = True

L = 60 # mm
e = 4
t = 30
r = 2
t2 = 15
diam = 8
thickness= 8

Gc = 100

nL = 100
l0 = L/nL

clC = l0 if meshTest else l0/2 # meshSize on the crack
clD = l0*2 if optimMesh else clC

mS = l0/2 

split = "Miehe"
regu = "AT1"

folder_save = Folder.PhaseField_Folder(f"CT_{dim}D", "Isot", split, regu, "", 1, "", meshTest, optimMesh, nL=nL)

Display.MyPrint(folder_save, 'green')

# ----------------------------------------------
# Geom
# ----------------------------------------------

pt1 = Point(0,-L/2)
pt2 = Point(L,-L/2)
pt3 = Point(L,L/2)
pt4 = Point(0,L/2)
pt5 = Point(0, e/2)
pt6 = Point(t+r, e/2, r=r)
pt7 = Point(t+r, -e/2, r=r)
pt8 = Point(0, -e/2)
points = Points([pt1,pt2,pt3,pt4,pt5,pt6,pt7,pt8], clD)

contour = points.Get_Contour()

circle1 = Circle(Point(t2, -L/2+t2), diam, clD)
circle2 = Circle(Point(t2, L/2-t2), diam, clD)

if plotGeom:
    ax = contour.Plot()
    contour.Plot_Geoms([circle1, circle2], ax=ax)

# ----------------------------------------------
# Mesh
# ----------------------------------------------

refineGeom = Domain(Point(t, -e*1.5), Point(L, e*1.5, thickness), clC) if optimMesh else None

if plotGeom:
    refineGeom.Plot(ax=ax)

if dim == 2:
    crack = Line(Point(t+r, 0, isOpen=True), Point(t+r+6, 0), clC, isOpen=True)

    mesh = Mesher().Mesh_2D(contour, [circle1, circle2], ElemType.TRI3, cracks=[crack], refineGeoms=[refineGeom])
else:

    elemType = ElemType.PRISM6

    pc1 = Point(t+r, 0, 0, True)
    pc2 = Point(t+r+6, 0, 0)
    pc3 = pc2 + [0,0,thickness]
    pc4 = Point(t+r, 0, thickness, True)
    # 
    line1 = Line(pc1, pc2, clC, True)
    line2 = Line(pc2, pc3, clC, False)
    line3 = Line(pc3, pc4, clC, True)
    line4 = Line(pc4, pc1, clC, True)
    # 
    crack = Contour([line1,line2,line3,line4], isOpen=True)
    cracks = [crack]

    if plotGeom:
        crack.Plot(ax=ax)

    mesh = Mesher().Mesh_Extrude(contour, [circle1, circle2], [0,0,thickness], [4], elemType, cracks=cracks, additionalLines=[line1], refineGeoms=[refineGeom])

# pvi.Plot_Mesh(mesh).show()
# pvi.Plot_Nodes(mesh, mesh.orphanNodes).show()

nodes_1 = mesh.Nodes_Cylinder(circle1)
nodes_2 = mesh.Nodes_Cylinder(circle2)

nodes_xL = mesh.Nodes_Conditions(lambda x,y,z: x==L)

# ----------------------------------------------
# Simu
# ----------------------------------------------

mat = Materials.Elas_Isot(dim, thickness=thickness, planeStress=True)

pfm = Materials.PhaseField(mat, split, regu, Gc, l0)

if doSimu:
    
    displacements = np.linspace(0, L/40, 50)

    config = f"""
    displacements = np.linspace(0, L/40, 50)

    for i, dep in enumerate(displacements):

    if dim == 2:
        simu.add_dirichlet(nodes_1, [0,-dep], ["x","y"])
        simu.add_dirichlet(nodes_2, [0,dep], ["x","y"])
    else:
        simu.add_dirichlet(nodes_1, [0,-dep, 0], ["x", "y", "z"])
        simu.add_dirichlet(nodes_2, [0,dep, 0], ["x", "y", "z"])
    """

    simu = Simulations.PhaseFieldSimu(mesh, pfm)
    simu.Results_Set_Bc_Summary(config)

    if plotIter:
        ax = Display.Plot_Result(simu, "damage", 1, plotMesh=True)

    for i, dep in enumerate(displacements):

        simu.Bc_Init()    

        if dim == 2:
            simu.add_dirichlet(nodes_1, [0,-dep], ["x","y"])
            simu.add_dirichlet(nodes_2, [0,dep], ["x","y"])
        else:
            simu.add_dirichlet(nodes_1, [0,-dep, 0], ["x", "y", "z"])
            simu.add_dirichlet(nodes_2, [0,dep, 0], ["x", "y", "z"])

        # pvi.Plot_BoundaryConditions(simu).show()

        u, d, K, converg = simu.Solve()
        
        simu.Results_Set_Iteration_Summary(i, dep, "mm", i/displacements.size, remove=True)

        assert converg

        simu.Save_Iter()

        if np.any(d[nodes_xL] >= 1):
            break

        if plotIter:        
            Display.Plot_Result(simu, "damage", 1, plotMesh=True, ax=ax)
            plt.pause(1e-12)

    simu.Save(folder_save)
    
else:
    simu: Simulations.PhaseFieldSimu = Simulations.Load_Simu(folder_save)

# ----------------------------------------------
# Post-processing
# ----------------------------------------------

if makeParaview:
    Paraview_Interface.Make_Paraview(simu, folder)

if makeMovie:
    pvi.Movie_simu(simu, "damage", folder_save, f"damage.mp4", show_edges=True, clim=(0,1), deformFactor=1, n_colors=11)

Display.Plot_Mesh(mesh)
Display.Plot_Tags(mesh, folder=folder_save)

Display.Plot_Result(simu, "damage", folder=folder_save)
Display.Plot_Result(simu, "uy", deformFactor=1)

pvi.Plot(simu, "damage", 1, show_edges=True).show()

plt.show()