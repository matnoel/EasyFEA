from EasyFEA import (Display, Mesher, ElemType,
                     Materials, Simulations)
from EasyFEA.Geoms import Point, Domain

# ----------------------------------------------
# Mesh
# ----------------------------------------------
L = 120 # mm
h = 13

domain = Domain(Point(), Point(L,h), h/5*2)
mesh = Mesher().Mesh_2D(domain, [], ElemType.QUAD4, isOrganised=True)

# ----------------------------------------------
# Simulation
# ----------------------------------------------
E = 210000 # MPa
v = .3
F = -800 # N

mat = Materials.Elas_Isot(2, E, v, planeStress=True, thickness=h)

simu = Simulations.ElasticSimu(mesh, mat)

nodesX0 = mesh.Nodes_Conditions(lambda x,y,z: x==0)
nodesXL = mesh.Nodes_Conditions(lambda x,y,z: x==L)

simu.add_dirichlet(nodesX0, [0]*2, ["x","y"])
simu.add_surfLoad(nodesXL, [F/h/h], ["y"])

simu.Solve()

# ----------------------------------------------
# Results
# ----------------------------------------------
Display.Plot_Mesh(mesh)
Display.Plot_BoundaryConditions(simu)
Display.Plot_Result(simu, 'uy', plotMesh=True)
Display.Plot_Result(simu, 'Svm', plotMesh=True, ncolors=11)

Display.plt.show()