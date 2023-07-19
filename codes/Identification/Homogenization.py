import Display
from Interface_Gmsh import Interface_Gmsh
from Geom import *
import Materials
import Simulations
from BoundaryCondition import BoundaryCondition, LagrangeCondition

plt = Display.plt

# Example from : Computational Homogenization of Heterogeneous Materials with Finite Elements
# http://link.springer.com/10.1007/978-3-030-18383-7
# SECTION 4.7

# Affichage.Clear()

# use PER boundary conditions ?
usePER = True 

# --------------------------------------
# Mesh
# --------------------------------------
p0 = Point(-1/2, -1/2)
p1 = Point(1/2, -1/2)
p2 = Point(1/2, 1/2)
p3 = Point(-1/2, 1/2)
pts = [p0, p1, p2, p3]

meshSize = 1/30

points = PointsList(pts, meshSize, isCreux=False)

f = 0.4

r = 1 * np.sqrt(f/np.pi)

inclusion = Circle(Point(), 2*r, meshSize, isCreux=False)

# e = 1/6
# l = 1-(2*e)
# inclusion = Domain(Point(-e, -l/2), Point(e, l/2), meshSize)

gmshInterface = Interface_Gmsh(False, False)

mesh = gmshInterface.Mesh_2D(points, [inclusion], "TRI6")

coordo = mesh.coordoGlob

Display.Plot_Mesh(mesh)
Display.Plot_Model(mesh)

nodesLeft = mesh.Nodes_Conditions(lambda x,y,z: x==-1/2)
nodesLeft = nodesLeft[np.argsort(coordo[nodesLeft,1])][1:-1]

nodesRight = mesh.Nodes_Conditions(lambda x,y,z: x==1/2)
nodesRight = nodesRight[np.argsort(coordo[nodesRight,1])][1:-1]

nodesUpper = mesh.Nodes_Conditions(lambda x,y,z: y==1/2)
nodesUpper = nodesUpper[np.argsort(coordo[nodesUpper,0])][1:-1]

nodesLower = mesh.Nodes_Conditions(lambda x,y,z: y==-1/2)
nodesLower = nodesLower[np.argsort(coordo[nodesLower,0])][1:-1]

nodesB0 = np.concatenate((nodesLower, nodesLeft))
nodesB1 = np.concatenate((nodesUpper, nodesRight))

if usePER:
    nodesBord = mesh.Nodes_Tags(["P0", "P1", "P2", "P3"])
else:
    nodesBord = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])

# --------------------------------------
# Model and simu
# --------------------------------------

elementsInclusion = mesh.Elements_Tags(["S1"])
elementsMatrice = mesh.Elements_Tags(["S0"])

E = np.zeros_like(mesh.groupElem.elements, dtype=float)
v = np.zeros_like(mesh.groupElem.elements, dtype=float)

E[elementsInclusion] = 50
E[elementsMatrice] = 1

v[elementsInclusion] = 0.3
v[elementsMatrice] = 0.45

# E[:] = 50
# v[:] = 0.3

comp = Materials.Elas_Isot(2, E, v, contraintesPlanes=False)

simu = Simulations.Simu_Displacement(mesh, comp, useNumba=True)

Display.Plot_Result(simu, E, nodeValues=False, title="E")
Display.Plot_Result(simu, v, nodeValues=False, title="v")

# --------------------------------------
# Homogenization
# --------------------------------------

r2 = np.sqrt(2)
E11 = np.array([[1, 0],[0, 0]])
E22 = np.array([[0, 0],[0, 1]])
E12 = np.array([[0, 1/r2],[1/r2, 0]])

def CalcDisplacement(Ekl: np.ndarray, pltSol=False):

    simu.Bc_Init()

    simu.add_dirichlet(nodesBord, [lambda x, y, z: Ekl.dot([x, y])[0], lambda x, y, z: Ekl.dot([x, y])[1]], ["x","y"])    

    if usePER:        
        
        # impose que le champ u soit à moyenne nulle
        useMean0 = False

        for n0, n1 in zip(nodesB0, nodesB1):
                
            nodes = np.array([n0, n1])

            # plt.gca().scatter(coordo[nodes, 0],coordo[nodes, 1], marker='+', c='red')

            for direction in ["x", "y"]:
                ddls = BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodes, [direction])
                
                values = Ekl @ [coordo[n0,0]-coordo[n1,0], coordo[n0,1]-coordo[n1,1]]
                value = values[0] if direction == "x" else values[1]

                # value = 0

                condition = LagrangeCondition("displacement", nodes, ddls, [direction], [value], [1, -1])
                simu._Bc_Add_Lagrange(condition)

        if useMean0:            

            nodes = mesh.nodes
            vect = np.ones(mesh.Nn) * 1/mesh.Nn 

            # sum u_i / Nn = 0
            ddls = BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodes, ["x"])        
            condition = LagrangeCondition("displacement", nodes, ddls, ["x"], [0], [vect])
            simu._Bc_Add_Lagrange(condition)

            # sum v_i / Nn = 0
            ddls = BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodes, ["y"])        
            condition = LagrangeCondition("displacement", nodes, ddls, ["y"], [0], [vect])
            simu._Bc_Add_Lagrange(condition)

    # Affichage.Plot_BoundaryConditions(simu)

    pass

    ukl = simu.Solve()

    # print(np.mean(simu.Get_Resultat("ux")))
    # print(np.mean(simu.Get_Resultat("uy")))

    simu.Save_Iteration()

    if pltSol:
        # Affichage.Plot_Result(simu, "ux", deformation=False)
        # Affichage.Plot_Result(simu, "uy", deformation=False)

        Display.Plot_Result(simu, "Sxx", facteurDef=0.3, deformation=True, nodeValues=True)
        Display.Plot_Result(simu, "Syy", facteurDef=0.3, deformation=True, nodeValues=True)
        Display.Plot_Result(simu, "Sxy", facteurDef=0.3, deformation=True, nodeValues=True)

    return ukl

u11 = CalcDisplacement(E11, False)
u22 = CalcDisplacement(E22, False)
u12 = CalcDisplacement(E12, True)

u11_e = mesh.Localises_sol_e(u11)
u22_e = mesh.Localises_sol_e(u22)
u12_e = mesh.Localises_sol_e(u12)

# --------------------------------------
# Effective elasticity tensor
# --------------------------------------

U_e = np.zeros((u11_e.shape[0],u11_e.shape[1], 3))

U_e[:,:,0] = u11_e; U_e[:,:,1] = u22_e; U_e[:,:,2] = u12_e

matriceType = "masse"
jacobien_e_pg = mesh.Get_jacobien_e_pg(matriceType)
poids_pg = mesh.Get_poid_pg(matriceType)
B_e_pg = mesh.Get_B_dep_e_pg(matriceType)

C_Mat = Materials.Resize_variable(comp.C, mesh.Ne, poids_pg.size)

C_hom = np.einsum('ep,p,epij,epjk,ekl->il', jacobien_e_pg, poids_pg, C_Mat, B_e_pg, U_e, optimize='optimal') * 1/mesh.aire

# TODO attention ici il faut utiliser toute l'ai meme si il y'a des trous remarques ZAKARIA


print(f"f = {f}")
print(f"c1111 = {C_hom[0,0]}")
print(f"c1122 = {C_hom[0,1]}")
print(f"c1212 = {C_hom[2,2]/2}")

plt.show()