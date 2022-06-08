
from unittest import result
from TicTac import TicTac
import Materiau
from Geom import *
import Affichage
import Interface_Gmsh
import Simu
import Dossier
import pandas as pd
import PostTraitement

import matplotlib.pyplot as plt

Affichage.Clear()

# Options

test=True
plotAllResult = True

comp = "Elas_Isot"
split = "Miehe" # ["Bourdin","Amor","Miehe","Stress"]
regu = "AT1" # "AT1", "AT2"
contraintesPlanes = True

nom="_".join([comp, split, regu])

nomDossier = "Calcul Energie plaque trouée"

folder = Dossier.NewFile(nomDossier, results=True)



# Data

L=15e-3
h=30e-3
ep=1e-3
diam=6e-3

E=12e9
v=0.2
SIG = 10 #Pa

gc = 1.4
l_0 = 0.12e-3*1.5


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

# Récupérations des noeuds

B_lower = Line(point,Point(x=L))
B_upper = Line(Point(y=h),Point(x=L, y=h))
c = diam/10
domainA = Domain(Point(x=(L-c)/2, y=h/2+0.8*diam/2), Point(x=(L+c)/2, y=h/2+0.8*diam/2+c))
domainB = Domain(Point(x=L/2+0.8*diam/2, y=(h-c)/2), Point(x=L/2+0.8*diam/2+c, y=(h+c)/2))

nodes0 = mesh.Get_Nodes_Line(B_lower)
nodesh = mesh.Get_Nodes_Line(B_upper)
node00 = mesh.Get_Nodes_Point(Point())
nodesA = mesh.Get_Nodes_Domain(domainA)
nodesB = mesh.Get_Nodes_Domain(domainB)

if plotAllResult:
    ax = Affichage.Plot_Maillage(mesh)
    for ns in [nodes0, nodesh, node00, nodesA, nodesB]:
        Affichage.Plot_NoeudsMaillage(mesh, ax=ax, noeuds=ns)

columns = ['v','A (DP)','B (DP)','A (CP)','B (CP)','A (Analytique CP)','B (Analytique CP)']

df = pd.DataFrame(columns=columns)

for v in [0.2,0.3,0.4]:
    result = {
        'v': v
    }
    for isCP in [False,True]:
        comportement = Materiau.Elas_Isot(2, E=E, v=v, contraintesPlanes=isCP, epaisseur=ep)
        phaseFieldModel = Materiau.PhaseFieldModel(comportement, split, regu, gc, l_0)
        materiau = Materiau.Materiau(phaseFieldModel=phaseFieldModel, verbosity=False)

        simu = Simu.Simu(mesh, materiau, verbosity=False)

        simu.add_dirichlet("displacement", nodes0, [0], ["y"])
        simu.add_dirichlet("displacement", node00, [0], ["x"])

        simu.add_surfLoad("displacement", nodesh, [-SIG], ["y"])

        # Affichage.Plot_BoundaryConditions(simu)

        simu.Assemblage_u()

        simu.Solve_u(useCholesky=True)

        psipa = np.mean(simu.Get_Resultat("psiP", True)[nodesA])*E/SIG**2
        psipb = np.mean(simu.Get_Resultat("psiP", True)[nodesB])*E/SIG**2

        if isCP:
            result['A (CP)'] = psipa
            result['B (CP)'] = psipb

            # Affichage.Plot_Result(simu, "psiP", valeursAuxNoeuds=True, coef=E/SIG**2, unite=f"*E/Sig^2 {nom} pour v={v}",folder=folder)
            Affichage.Plot_Result(simu, "psiP", valeursAuxNoeuds=False, coef=E/SIG**2, unite=f"*E/Sig^2 {nom} pour v={v}",folder=folder)
        else:
            result['A (DP)'] = psipa
            result['B (DP)'] = psipb

    psiP_A = 1**2*(v*(1-2*v)+1)/(2*(1+v))
    psiP_B = 3**2*v**2/(1+v)

    result['A (Analytique CP)'] = psiP_A
    result['B (Analytique CP)'] = psiP_B
    
    new = pd.DataFrame(result, index=[0])

    df = pd.concat([df, new], ignore_index=True)

Affichage.NouvelleSection("Résultats")

print(nom+'\n')
print(df)

SxxA = np.max(simu.Get_Resultat("Sxx", True)[nodesA])
SyyA = np.max(simu.Get_Resultat("Syy", True)[nodesA])
SxyA = np.max(simu.Get_Resultat("Sxy", True)[nodesA])

SigA=np.array([[SxxA, SxyA, 0],[SxyA, SyyA, 0],[0,0,0]])
print(f"\nEn A : Sig/SIG = \n{SigA/SIG}\n")

SxxB = np.max(simu.Get_Resultat("Sxx", True)[nodesB])
SyyB = np.max(simu.Get_Resultat("Syy", True)[nodesB])
SxyB = np.max(simu.Get_Resultat("Sxy", True)[nodesB])

SigB=np.array([[SxxB, SxyB, 0],[SxyB, SyyB, 0],[0,0,0]])
print(f"\nEn B : Sig/SIG = \n{SigB/SIG}\n")

if plotAllResult:
    Affichage.Plot_Result(simu, "Sxx", valeursAuxNoeuds=True, coef=1/SIG, unite="/Sig")
    Affichage.Plot_Result(simu, "Syy", valeursAuxNoeuds=True, coef=1/SIG, unite="/Sig")
    Affichage.Plot_Result(simu, "Sxy", valeursAuxNoeuds=True, coef=1/SIG, unite="/Sig")

fig, axp = plt.subplots()
vv = np.arange(0, 0.5, 0.005)

# test = (vv*(1-2*vv)+1)/(2*(1+vv))

axp.plot(vv, (vv*(1-2*vv)+1)/(2*(1+vv)), label="psiP_A*E/Sig^2")
axp.plot(vv, 9*vv**2/(1+vv), label="psiP_B*E/Sig^2")
axp.grid()
axp.legend()
axp.set_xlabel("v")

# PostTraitement.Save_fig(folder, "iso densité non corrigé")

Affichage.NouvelleSection("Correction")

def calcSplit(Sxx, Sxy, Syy):
    Sig = np.array([[Sxx, Sxy,0],[Sxy, Syy, 0],[0,0,0]])
    Eps = (1+v)/E*Sig - v/E*np.trace(Sig)*np.eye(3)
    eigvals = np.linalg.eigvals(Eps)
    eigvals[2]=0
    Epsi = np.diag(eigvals)
    EpsiP = (Epsi+np.abs(Epsi))/2
    EpsiM = (Epsi-np.abs(Epsi))/2
    
    trEps = np.trace(Eps)
    trEpsP = (trEps+np.abs(trEps))/2
    trEpsM = (trEps-np.abs(trEps))/2

    return trEpsP, trEpsM, EpsiP, EpsiM

trEps_AP, trEps_AM, EpsiAP, EpsiAM = calcSplit(SxxA, SxyA, SyyA)
trEps_BP, trEps_BM, EpsiBP, EpsiBM = calcSplit(SxxB, SxyB, SyyB)

l = comportement.get_lambda()
mu = comportement.get_mu()

psiP_A2 = ((l/2+mu)*(SxxA-v*SyyA)**2)

psiPA = (l/2*trEps_AP**2 + mu * np.einsum('ij,ij', EpsiAP,EpsiAP))*E/SIG**2
psiPB = (l/2*trEps_BP**2 + mu * np.einsum('ij,ij', EpsiBP,EpsiBP))*E/SIG**2

print("Apres correction")
print(f"ecart A = {np.abs(psiPA-psipa)/psipa*100} %")
print(f"ecart B = {np.abs(psiPB-psipb)/psipb*100} %")

trSA = SxxA + SyyA
ExxA = (SxxA - v*SyyA)/E
EyyA = (SyyA - v*SxxA)/E
ExyA = (1+v)/E*SxyA
# EzzA = -v/E*trSA-

trEpsA = trSA*(1-2*v)/E - (trEps_AP+trEps_AM)

# detEA = trEA - 














TicTac.getResume()

plt.show()



pass


