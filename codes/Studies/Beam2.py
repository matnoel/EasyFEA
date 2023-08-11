import Display
import Materials
import Simulations
from Geom import Line, Point, Section, Domain
from Interface_Gmsh import Interface_Gmsh

import numpy as np

Display.Clear()

l = 100 # mm

pA = Point(2*l, 0)
pB = Point(l, 0)
pC = Point(l, l)
pD = Point(0, 0)
pE = Point(0, l)

line1 = Line(pA, pC)
line2 = Line(pA, pB)
line3 = Line(pB, pC)
line4 = Line(pC, pE)
line5 = Line(pB, pD)
line6 = Line(pB, pE)
listLine = [line1, line2, line3, line4, line5, line6]

meshSection = Interface_Gmsh().Mesh_2D(Domain(Point(-4/2, -8/2), Point(4/2, 8/2)))
section = Section(meshSection)
Display.Plot_Mesh(meshSection)

E = 276 # MPa
v = 0.3

beamDim = 2

listBeam = [Materials.Beam_Elas_Isot(beamDim, line, section, E, v) for line in listLine]
structure = Materials.Beam_Structure(listBeam)

mesh = Interface_Gmsh().Mesh_Beams(listBeam, "SEG2")
Display.Plot_Mesh(mesh)
Display.Plot_Model(mesh)

simu = Simulations.Simu_Beam(mesh, structure)

nodesToLink = []
[nodesToLink.extend(mesh.Nodes_Point(point)) for point in [pA, pB, pC, pE]]
nodesToLink = np.unique(nodesToLink)

nodesRigi = mesh.Nodes_Point(pE)
nodesRigi = np.append(nodesRigi, mesh.Nodes_Point(pD))
nodesA = mesh.Nodes_Point(pA)

# Pour chaque point on vient connecter les poutres
for point in [pA, pB, pC]:

    nodes = mesh.Nodes_Point(point)

    firstNodes = nodes[0]
    others = nodes[1:]

    [simu.add_connection_hinged(np.array([firstNodes, n])) for n in others]

    pass

simu.add_dirichlet(nodesRigi, [0,0], ['x','y'])
simu.add_neumann(nodesA, [-40*9.81], ['y'])

simu.Solve()

matrixDep = simu.Results_displacement_matrix()
depMax = np.max(np.linalg.norm(matrixDep, axis=1))

Display.Plot_BoundaryConditions(simu)
Display.Plot_Result(simu, "ux", deformation=True, factorDef=20/depMax)
Display.Plot_Result(simu, "uy", deformation=True, factorDef=20/depMax)
Display.Plot_Result(simu, "rz", deformation=True, factorDef=20/depMax)
Display.Plot_Result(simu, "fx", deformation=True, factorDef=20/depMax)
Display.Plot_Result(simu, "fy", deformation=True, factorDef=20/depMax)

Epsilon_e_pg = simu._Calc_Epsilon_e_pg(simu.displacement)
Internal_e = simu._Calc_InternalForces_e_pg(Epsilon_e_pg).mean(1)
Sigma_e = simu._Calc_Sigma_e_pg(Epsilon_e_pg).mean(1)
Display.Plot_Result(simu, Sigma_e[:,0], title='Sxx')
Display.Plot_Result(simu, Internal_e[:,0], title='N')

print(simu)

Display.plt.show()