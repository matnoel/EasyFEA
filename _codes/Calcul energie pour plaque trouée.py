
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

plotAllResult = False

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


# Création du maillage
clD = l_0*2
clC = l_0

point = Point()
domain = Domain(point, Point(x=L, y=h), clD)
circle = Circle(Point(x=L/2, y=h/2), diam, clC)

interfaceGmsh = Interface_Gmsh.Interface_Gmsh(affichageGmsh=True)
mesh = interfaceGmsh.PlaqueTrouée(domain, circle, "TRI3")

Affichage.Plot_Maillage(mesh)
plt.show()

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

            Affichage.Plot_Result(simu, "psiP", valeursAuxNoeuds=True, coef=E/SIG**2, unite=f"*E/Sig^2 {nom} pour v={v}",folder=folder)
            # Affichage.Plot_Result(simu, "psiP", valeursAuxNoeuds=False, coef=E/SIG**2, unite=f"*E/Sig^2 {nom} pour v={v}",folder=folder)
        else:
            result['A (DP)'] = psipa
            result['B (DP)'] = psipb

    Miehe_psiP_A = 1**2*(v*(1-2*v)+1)/(2*(1+v))
    Stress_psiP_B = 3**2*v**2/(1+v)

    result['A (Analytique CP)'] = Miehe_psiP_A
    result['B (Analytique CP)'] = Stress_psiP_B
    
    new = pd.DataFrame(result, index=[0])

    df = pd.concat([df, new], ignore_index=True)

# df.to_excel(Dossier.Append([folder, f"{nom}.xlsx"]), index=False)

Affichage.NouvelleSection("Résultats")

print(nom+'\n')
print(df)

SxxA = np.mean(simu.Get_Resultat("Sxx", True)[nodesA])
SyyA = np.mean(simu.Get_Resultat("Syy", True)[nodesA])
SxyA = np.mean(simu.Get_Resultat("Sxy", True)[nodesA])

Sig_A=np.array([[SxxA, SxyA, 0],[SxyA, SyyA, 0],[0,0,0]])
print(f"\nEn A : Sig/SIG = \n{Sig_A/SIG}\n")

SxxB = np.mean(simu.Get_Resultat("Sxx", True)[nodesB])
SyyB = np.mean(simu.Get_Resultat("Syy", True)[nodesB])
SxyB = np.mean(simu.Get_Resultat("Sxy", True)[nodesB])

Sig_B=np.array([[SxxB, SxyB, 0],[SxyB, SyyB, 0],[0,0,0]])
print(f"\nEn B : Sig/SIG = \n{Sig_B/SIG}\n")

if plotAllResult:
    Affichage.Plot_Result(simu, "Sxx", valeursAuxNoeuds=True, coef=1/SIG, unite="/Sig")
    Affichage.Plot_Result(simu, "Syy", valeursAuxNoeuds=True, coef=1/SIG, unite="/Sig")
    Affichage.Plot_Result(simu, "Sxy", valeursAuxNoeuds=True, coef=1/SIG, unite="/Sig")



Affichage.NouvelleSection("Calcul analytique")


fig, axp = plt.subplots()

list_v = np.arange(0, 0.5,0.0005)

# test = (vv*(1-2*vv)+1)/(2*(1+vv))

axp.plot(list_v, (list_v*(1-2*list_v)+1)/(2*(1+list_v)), label="psiP_A*E/Sig^2")
axp.plot(list_v, 9*list_v**2/(1+list_v), label="psiP_B*E/Sig^2")
axp.grid()
axp.legend()
axp.set_xlabel("v")

PostTraitement.Save_fig(folder, "Analytique")

list_Miehe_psiP_A=[]
list_Miehe_psiP_B=[]

list_Stress_psiP_A=[]
list_Stress_psiP_B=[]

for v in list_v:

    # Split Miehe
    Eps_A = (1+v)/E*Sig_A - v/E*np.trace(Sig_A)*np.eye(3); trEps_A = np.trace(Eps_A); trEpsP_A = (trEps_A+np.abs(trEps_A))/2
    Eps_B = (1+v)/E*Sig_B - v/E*np.trace(Sig_B)*np.eye(3); trEps_B = np.trace(Eps_B); trEpsP_B = (trEps_B+np.abs(trEps_B))/2

    Epsi_A = np.diag(np.linalg.eigvals(Eps_A)); Epsip_A = (Epsi_A+np.abs(Epsi_A))/2
    Epsi_B = np.diag(np.linalg.eigvals(Eps_B)); Epsip_B = (Epsi_B+np.abs(Epsi_B))/2

    # l = v*E/((1+v)*(1-2*v))
    l = v*E/(1-v**2)
    mu=E/(2*(1+v))

    Miehe_psiP_A = l/2*trEpsP_B**2 + mu*np.einsum('ij,ij',Epsip_A,Epsip_A)
    Stress_psiP_B = l/2*trEpsP_B**2 + mu*np.einsum('ij,ij',Epsip_B,Epsip_B)

    list_Miehe_psiP_A.append(Miehe_psiP_A)
    list_Miehe_psiP_B.append(Stress_psiP_B)

    # Split Stress
    Sigi_A = np.diag(np.linalg.eigvals(Sig_A)); Sigip_A = (Sigi_A+np.abs(Sigi_A))/2
    Sigi_B = np.diag(np.linalg.eigvals(Sig_B)); Sigip_B = (Sigi_B+np.abs(Sigi_B))/2
    
    trSig_A = np.trace(Sig_A); trSigP_A = (trSig_A+np.abs(trSig_A))/2
    trSig_B = np.trace(Sig_B); trSigP_B = (trSig_B+np.abs(trSig_B))/2

    Stress_psiP_A = ((1+v)/E*np.einsum('ij,ij',Sigip_A,Sigip_A) - v/E * trSigP_A**2)/2
    Stress_psiP_B = ((1+v)/E*np.einsum('ij,ij',Sigip_B,Sigip_B) - v/E * trSigP_B**2)/2

    list_Stress_psiP_A.append(Stress_psiP_A)
    list_Stress_psiP_B.append(Stress_psiP_B)


fig, ax = plt.subplots()

ax.plot(list_v, np.array(list_Miehe_psiP_A)*E/SIG**2, label="psiP_A*E/Sig^2")
ax.plot(list_v, np.array(list_Miehe_psiP_B)*E/SIG**2, label="psiP_B*E/Sig^2")
ax.grid()
ax.legend()
ax.set_xlabel("v")
ax.set_title("Miehe")

PostTraitement.Save_fig(folder, "Miehe")

fig, ax = plt.subplots()

ax.plot(list_v, np.array(list_Stress_psiP_A)*E/SIG**2, label="psiP_A*E/Sig^2")
ax.plot(list_v, np.array(list_Stress_psiP_B)*E/SIG**2, label="psiP_B*E/Sig^2")
ax.grid()
ax.legend()
ax.set_xlabel("v")
ax.set_title("Stress")

PostTraitement.Save_fig(folder, "Stress")









TicTac.getResume()

plt.show()



pass


