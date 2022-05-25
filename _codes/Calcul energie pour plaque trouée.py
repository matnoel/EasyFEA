
from TicTac import TicTac
import Materiau
from Geom import *
import Affichage
import Interface_Gmsh
import Simu
import Dossier

import matplotlib.pyplot as plt

Affichage.Clear()

# Options

test=True

comp = "Elas_Isot"
split = "Amor" # ["Bourdin","Amor","Miehe"]
regu = "AT1" # "AT1", "AT2"
contraintesPlanes = True

nom="_".join([comp, split, regu])

nomDossier = "Benchmarck_Compression"

if test:
    file = Dossier.Append([nomDossier, "Test", nom])
else:
    file = Dossier.Append([nomDossier, nom])

file = Dossier.NewFile(file, results=True)

# Data

L=15e-3
h=30e-3
ep=1e-3
diam=6e-3

E=12e9
v=0.2
Sig = 10 #Pa

gc = 1.4
l_0 = 0.12e-3

psiP_A = (v*(1-2*v)+1)/(2*(1+v))
psiP_B = 9*v**2/(1+v)

fig, axp = plt.subplots()
vv = np.arange(0, 0.5, 0.005)

# test = (vv*(1-2*vv)+1)/(2*(1+vv))

axp.plot(vv, (vv*(1-2*vv)+1)/(2*(1+vv)), label="psiP_A*E/Sig^2")
axp.plot(vv, 9*vv**2/(1+vv), label="psiP_B*E/Sig^2")
axp.grid()
axp.legend()
axp.set_xlabel("v")


print(f"Pour v={v} : psiP_A*E/Sig^2 = {np.round(psiP_A,3)} et psiP_B*E/Sig^2 = {np.round(psiP_B,3)}")

# Création de la simulations

if test:
    clD = l_0*2
    clC = l_0
else:
    clD = l_0/2
    clC = l_0/2

point = Point()
domain = Domain(point, Point(x=L, y=h), clD)
circle = Circle(Point(x=L/2, y=h/2), diam, clC)

interfaceGmsh = Interface_Gmsh.Interface_Gmsh(affichageGmsh=False)
mesh = interfaceGmsh.PlaqueTrouée(domain, circle, "QUAD4")

# Affichage.Plot_Maillage(mesh)
# plt.show()

comportement = Materiau.Elas_Isot(2, E=E, v=v, contraintesPlanes=contraintesPlanes, epaisseur=ep)
phaseFieldModel = Materiau.PhaseFieldModel(comportement, split, regu, gc, l_0)
materiau = Materiau.Materiau(phaseFieldModel=phaseFieldModel)

simu = Simu.Simu(mesh, materiau, verbosity=True)

# Récupérations des noeuds

B_lower = Line(point,Point(x=L))
B_upper = Line(Point(y=h),Point(x=L, y=h))
c = diam/10
domainA = Domain(Point(x=(L-c)/2, y=h/2+0.8*diam/2), Point(x=(L+c)/2, y=h/2+0.8*diam/2+c))
domainB = Domain(Point(x=L/2+0.8*diam/2, y=(h-c)/2), Point(x=L/2+0.8*diam/2+c, y=(h+c)/2))

nodes0 = mesh.Get_Nodes_Line(B_lower)
nodesh = mesh.Get_Nodes_Line(B_upper)
node00 = mesh.Get_Nodes_Conditions(lambda x: x==0, lambda y: y==0)
nodesA = mesh.Get_Nodes_Domain(domainA)
nodesB = mesh.Get_Nodes_Domain(domainB)

ax = Affichage.Plot_Maillage(mesh)

for ns in [nodes0, nodesh, node00, nodesA, nodesB]:
    Affichage.Plot_NoeudsMaillage(mesh, ax=ax, noeuds=ns)



simu.add_dirichlet("displacement", nodes0, ["y"], [0])
simu.add_dirichlet("displacement", node00, ["x"], [0])

simu.add_surfLoad("displacement", nodesh, ["y"], [-Sig])

simu.Assemblage_u()

simu.Solve_u(useCholesky=True)

SxxA = np.mean(simu.Get_Resultat("Sxx", True)[nodesA])
SyyA = np.mean(simu.Get_Resultat("Syy", True)[nodesA])
SxyA = np.mean(simu.Get_Resultat("Sxy", True)[nodesA])

print(f"\nEn A : Sxx/Sig = {np.round(SxxA/Sig,2)}, Syy/Sig = {np.round(SyyA/Sig,2)}, Sxy/Sig = {np.round(SxyA/Sig,2)}")

SxxB = np.mean(simu.Get_Resultat("Sxx", True)[nodesB])
SyyB = np.mean(simu.Get_Resultat("Syy", True)[nodesB])
SxyB = np.mean(simu.Get_Resultat("Sxy", True)[nodesB])

print(f"En B : Sxx/Sig = {np.round(SxxB/Sig,2)}, Syy/Sig = {np.round(SyyB/Sig,2)}, Sxy/Sig = {np.round(SxyB/Sig,2)}")

Affichage.Plot_Result(simu, "Sxx", valeursAuxNoeuds=True, coef=1/Sig, unite="/Sig")
Affichage.Plot_Result(simu, "Syy", valeursAuxNoeuds=True, coef=1/Sig, unite="/Sig")
Affichage.Plot_Result(simu, "Sxy", valeursAuxNoeuds=True, coef=1/Sig, unite="/Sig")

Affichage.Plot_Result(simu, "psiP", valeursAuxNoeuds=True, coef=E/Sig**2, unite=f"*E/Sig^2 pour v={v}")

psipa = np.mean(simu.Get_Resultat("psiP", True)[nodesA])*E/Sig**2
psipb = np.mean(simu.Get_Resultat("psiP", True)[nodesB])*E/Sig**2

print(f"\nPour v={v} : psiP_A*E/Sig^2 = {np.round(psipa,3)} et psiP_B*E/Sig^2 = {np.round(psipb,3)}")

TicTac.getResume()

plt.show()



pass


