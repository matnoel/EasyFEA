from TicTac import Tic
import Materials
from Geom import *
import Display as Display
import Interface_Gmsh as Interface_Gmsh
import Simulations
import Folder
import PostTraitement as PostTraitement

import matplotlib.pyplot as plt

Display.Clear()

# L'objectif de regarder les champs de contraintes en fonctions des chargements

# Options
dim = 2
comp = "Elas_Isot"
split = "Zhang" # ["Bourdin","Amor","Miehe","Stress"]
regu = "AT2"
contraintesPlanes = True

nom="_".join([comp, split, regu])

nomDossier = "PlateWithHole_Chargement"

folder = Folder.New_File(nomDossier, results=True)

# Data
coef = 1e-3

L=15*coef
H=30*coef
h=H/2
ep=1*coef
diam=6*coef
r=diam/2

E=12e9
v=0.2
SIG = 5 #Pa

gc = 1.4
l_0 = 0.12 *coef*3

# Création du maillage
clD = l_0*2
clC = l_0/2

point = Point()
domain = Domain(point, Point(x=L, y=H), clD)
circle = Circle(Point(x=L/2, y=H-h), diam, clC, isCreux=True)
val = diam*2
# refineGeom = Domain(Point(x=L/2-val/2, y=(H-h)-val/2), Point(x=L/2+val/2, y=(H-h)+val/2), meshSize=clC/2)
refineGeom = None

interfaceGmsh = Interface_Gmsh.Interface_Gmsh(affichageGmsh=False, verbosity=False)
if dim == 2:
    mesh = interfaceGmsh.Mesh_2D(domain, [circle], "QUAD8", refineGeom=refineGeom)
else:
    mesh = interfaceGmsh.Mesh_3D(domain, [circle], [0,0,10*coef], 4, "HEXA8", refineGeom=refineGeom)

# Récupérations des noeuds de chargement
nodesY0 = mesh.Nodes_Conditions(lambda x,y,z: y==0)
nodesH = mesh.Nodes_Conditions(lambda x,y,z: y==H)
nodeX0Y0 = mesh.Nodes_Conditions(lambda x,y,z: (x==0) & (y==0))
if dim == 2:
    noeuds_cercle = mesh.Nodes_Circle(circle)
else:
    noeuds_cercle = mesh.Nodes_Cylindre(circle,[0,0,1])

# prends les noeuds du bas
noeuds_cercle = noeuds_cercle[np.where(mesh.coordo[noeuds_cercle,1]<=circle.center.y)]

# loi de comportement
comportement = Materials.Elas_Isot(dim, E=E, v=v, contraintesPlanes=True, epaisseur=ep)
phaseFieldModel = Materials.PhaseField_Model(comportement, split, regu, gc, l_0)

simu = Simulations.Simu_PhaseField(mesh, phaseFieldModel, verbosity=False)

simu.add_dirichlet(nodesY0, [0], ["y"])
simu.add_dirichlet(nodeX0Y0, [0], ["x"])

pc = circle.center.coordo

def FuncEval(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Evaluation de la fonction sig cos(theta)^2 vect_n"""
    
    # Calcul de l'angle
    theta = np.arctan((x-pc[0])/(y-pc[1]))

    # Coordonnées des points de gauss sous forme de matrice
    coord = np.zeros((x.shape[0],x.shape[1],3))
    coord[:,:,0] = x
    coord[:,:,1] = y
    coord[:,:,2] = z

    # Construction du vecteur normal
    vect = coord - pc
    vectN = np.einsum('npi,np->npi', vect, 1/np.linalg.norm(vect, axis=2))
    
    # Chargement
    loads = SIG * np.einsum('np,npi->npi',np.cos(theta)**2, vectN)

    return loads

funcEvalX = lambda x,y,z: FuncEval(x,y,z)[:,:,0]
funcEvalY = lambda x,y,z: FuncEval(x,y,z)[:,:,1]

# simu.add_surfLoad(noeuds_cercle, [funcEvalX], ["x"])
# simu.add_surfLoad(noeuds_cercle, [funcEvalY], ["y"])
simu.add_surfLoad(noeuds_cercle, [funcEvalX, funcEvalY], ["x","y"], description=r"$\mathbf{q}(\theta) = \sigma \ cos^2(\theta) \ \mathbf{n}(\theta)$")

# simu.add_surfLoad(nodesH, [-SIG], ['y'])

Display.Plot_BoundaryConditions(simu)

simu.Solve()

simu.Save_Iteration()

Display.Section("Résultats")

# Affichage.Plot_Model(mesh)
# Affichage.Plot_Mesh(mesh)

Display.Plot_Result(simu, "Sxx", nodeValues=True, coef=1/SIG, title=r"$\sigma_{xx}/\sigma$", folder=folder, filename='Sxx', cmap='seismic')
Display.Plot_Result(simu, "Syy", nodeValues=True, coef=1/SIG, title=r"$\sigma_{yy}/\sigma$", folder=folder, filename='Syy')
Display.Plot_Result(simu, "Sxy", nodeValues=True, coef=1/SIG, title=r"$\sigma_{xy}/\sigma$", folder=folder, filename='Sxy')
Display.Plot_Result(simu, "Svm", coef=1/SIG, title=r"$\sigma_{vm}/\sigma$", folder=folder, filename='Svm')

Display.Plot_Result(simu, "psiP")

# Affichage.Plot_Result(simu, "ux")
# Affichage.Plot_Result(simu, "uy")


# vectF = simu.Get_K_C_M_F()[0] @ simu.displacement
# Affichage.Plot_Result(simu, vectF.reshape(-1,dim)[:,1])

Tic.Resume()

plt.show()





