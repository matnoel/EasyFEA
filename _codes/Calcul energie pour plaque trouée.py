
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

plotAllResult = True

comp = "Elas_Isot"
split = "Stress" # ["Bourdin","Amor","Miehe","Stress"]
regu = "AT1" # "AT1", "AT2"
contraintesPlanes = True

nom="_".join([comp, split, regu])

loadInHole = False

nomDossier = "Calcul Energie plaque trouée"

if loadInHole:
    nomDossier += "_loadInHole"

folder = Dossier.NewFile(nomDossier, results=True)

# Data

L=15e-3
h=30e-3
ep=1e-3
diam=6e-3
r=diam/2

E=12e9
v=0.2
SIG = 10 #Pa

gc = 1.4
l_0 = 0.12e-3*1.2

# Création du maillage
clD = l_0*2
clC = l_0

point = Point()
domain = Domain(point, Point(x=L, y=h), clD)
circle = Circle(Point(x=L/2, y=h/2), diam, clC)

interfaceGmsh = Interface_Gmsh.Interface_Gmsh(affichageGmsh=False)
mesh = interfaceGmsh.PlaqueTrouée(domain, circle, "TRI3")

# Récupérations des noeuds de chargement
B_lower = Line(point,Point(x=L))
B_upper = Line(Point(y=h),Point(x=L, y=h))
nodes0 = mesh.Get_Nodes_Line(B_lower)
nodesh = mesh.Get_Nodes_Line(B_upper)
node00 = mesh.Get_Nodes_Point(Point())
nodesCircle = mesh.Get_Nodes_Circle(circle)
nodesCircle = nodesCircle[np.where(mesh.coordo[nodesCircle,1]<= circle.center.y)]


# Noeuds en A et en B
nodeA = mesh.Get_Nodes_Point(Point(x=L/2, y=h/2+r))
nodeB = mesh.Get_Nodes_Point(Point(x=L/2+r, y=h/2))

if plotAllResult:
    ax = Affichage.Plot_Maillage(mesh)
    for ns in [nodes0, nodesh, node00, nodeA, nodeB,nodesCircle]:
        Affichage.Plot_NoeudsMaillage(mesh, ax=ax, noeuds=ns)

columns = ['v','A (DP)','B (DP)','A (CP)','B (CP)','A (Analytique CP)','B (Analytique CP)']

df = pd.DataFrame(columns=columns)

list_V = [0.2,0.3,0.4]

for v in list_V:
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

        if loadInHole:
            simu.add_surfLoad("displacement", nodesCircle, [lambda x,y,z : SIG*(y-circle.center.y)/r], ["y"])            
        else:
            simu.add_surfLoad("displacement", nodesh, [-SIG], ["y"])

        # Affichage.Plot_BoundaryConditions(simu)

        simu.Assemblage_u()

        simu.Solve_u(useCholesky=True)

        psipa = np.mean(simu.Get_Resultat("psiP", True)[nodeA])*E/SIG**2
        psipb = np.mean(simu.Get_Resultat("psiP", True)[nodeB])*E/SIG**2

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

SxxA = simu.Get_Resultat("Sxx", True)[nodeA][0]
SyyA = simu.Get_Resultat("Syy", True)[nodeA][0]
SxyA = simu.Get_Resultat("Sxy", True)[nodeA][0]

Sig_A=np.array([[SxxA, SxyA, 0],[SxyA, SyyA, 0],[0,0,0]])
print(f"\nEn A : Sig/SIG = \n{Sig_A/SIG}\n")

SxxB = simu.Get_Resultat("Sxx", True)[nodeB][0]
SyyB = simu.Get_Resultat("Syy", True)[nodeB][0]
SxyB = simu.Get_Resultat("Sxy", True)[nodeB][0]

Sig_B=np.array([[SxxB, SxyB, 0],[SxyB, SyyB, 0],[0,0,0]])
print(f"\nEn B : Sig/SIG = \n{Sig_B/SIG}\n")

if plotAllResult:
    Affichage.Plot_Result(simu, "Sxx", valeursAuxNoeuds=True, coef=1/SIG, unite="/Sig", folder=folder)
    Affichage.Plot_Result(simu, "Syy", valeursAuxNoeuds=True, coef=1/SIG, unite="/Sig", folder=folder)
    Affichage.Plot_Result(simu, "Sxy", valeursAuxNoeuds=True, coef=1/SIG, unite="/Sig", folder=folder)



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



fig, ax1 = plt.subplots()

ax1.plot(list_v, np.array(list_Miehe_psiP_A)*E/SIG**2, label="psiP_A*E/Sig^2")
ax1.plot(list_v, np.array(list_Miehe_psiP_B)*E/SIG**2, label="psiP_B*E/Sig^2")
ax1.grid()
# if split == "Miehe":    
#     ax1.scatter(list_V, np.array(df['A (CP)'].tolist()),label='Miehe A')
#     ax1.scatter(list_V, np.array(df['B (CP)'].tolist()),label='Miehe B')
ax1.legend()
ax1.set_xlabel("v")
ax1.set_title("Miehe")

PostTraitement.Save_fig(folder, "Miehe")

fig, ax2 = plt.subplots()

ax2.plot(list_v, np.array(list_Stress_psiP_A)*E/SIG**2, label="psiP_A*E/Sig^2")
ax2.plot(list_v, np.array(list_Stress_psiP_B)*E/SIG**2, label="psiP_B*E/Sig^2")
ax2.grid()
# if split == "Stress":    
#     ax2.scatter(list_V, np.array(df['A (CP)'].tolist()),label='Stress A')
#     ax2.scatter(list_V, np.array(df['B (CP)'].tolist()),label='Stress B')
ax2.legend()
ax2.set_xlabel("v")
ax2.set_title("Stress")

PostTraitement.Save_fig(folder, "Stress")









TicTac.getResume()

plt.show()



pass


