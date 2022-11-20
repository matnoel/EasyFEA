
from TicTac import Tic
import Materials
from Geom import *
import Affichage as Affichage
import Interface_Gmsh as Interface_Gmsh
import Simulations
import Folder
import pandas as pd
import PostTraitement as PostTraitement

import matplotlib.pyplot as plt

Affichage.Clear()

# Options

plotAllResult = False

comp = "Elas_Isot" # "Elas_Isot" "Elas_IsotTrans"
split = "Miehe" # ["Bourdin","Amor","Miehe","Stress","AnisotMiehe","AnisotStress"]
regu = "AT1" # "AT1", "AT2"
contraintesPlanes = True

nom="_".join([comp, split, regu])

loadInHole = False

nomDossier = "Calcul Energie plaque trouée"

if loadInHole:
    nomDossier += "_loadInHole"

folder = Folder.New_File(nomDossier, results=True)

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
l_0 = 0.12e-3*5

# Création du maillage
clD = l_0*2
clC = l_0

point = Point()
domain = Domain(point, Point(x=L, y=h), clD)
circle = Circle(Point(x=L/2, y=h/2), diam, clC)

interfaceGmsh = Interface_Gmsh.Interface_Gmsh(affichageGmsh=False)
mesh = interfaceGmsh.Mesh_PlaqueAvecCercle2D(domain, circle, "TRI3")

# Récupérations des noeuds de chargement
B_lower = Line(point,Point(x=L))
B_upper = Line(Point(y=h),Point(x=L, y=h))
nodes0 = mesh.Nodes_Line(B_lower)
nodesh = mesh.Nodes_Line(B_upper)
node00 = mesh.Nodes_Point(Point())
nodesCircle = mesh.Nodes_Circle(circle)
nodesCircle = nodesCircle[np.where(mesh.coordo[nodesCircle,1]<= circle.center.y)]


# Noeuds en A et en B
nodeA = mesh.Nodes_Point(Point(x=L/2, y=h/2+r))
nodeB = mesh.Nodes_Point(Point(x=L/2+r, y=h/2))

if plotAllResult:
    ax = Affichage.Plot_Maillage(mesh)
    for ns in [nodes0, nodesh, node00, nodeA, nodeB]:
        Affichage.Plot_Noeuds(mesh, ax=ax, noeuds=ns,c='red')
    PostTraitement.Save_fig(folder, 'mesh')

columns = ['v','A (ana CP)','B (ana CP)',
            'A (CP)','errA (CP)','B (CP)','errB (CP)',
            'A (DP)','errA (DP)','B (DP)','errB (DP)']

df = pd.DataFrame(columns=columns)

list_V = [0.2,0.3,0.4]

Miehe_psiP_A = lambda v: 1**2*(v*(1-2*v)+1)/(2*(1+v))
Miehe_psiP_B = lambda v: 3**2*v**2/(1+v)

# l = lambda v: v*E/((1+v)*(1-2*v))
# mu = lambda v: E/(2*(1+v))
# Miehe_psiP_A = lambda v: (l(v)/2+mu(v))*(SIG/E*(1.358-v*0.042))**2*E/SIG**2
# Miehe_psiP_B = lambda v: 3**2*v**2/(1+v)

for v in list_V:
    result = {
        'v': v
    }
    for isCP in [False,True]:
        comportement = Materials.Elas_Isot(2, E=E, v=v, contraintesPlanes=isCP, epaisseur=ep)
        phaseFieldModel = Materials.PhaseField_Model(comportement, split, regu, gc, l_0)
        materiau = Materials.Create_Materiau(phaseFieldModel, verbosity=False)

        simu = Simulations.Create_Simu(mesh, materiau, verbosity=False)

        simu.add_dirichlet(nodes0, [0], ["y"])
        simu.add_dirichlet(node00, [0], ["x"])

        if loadInHole:

            simu.add_surfLoad(nodesCircle, [lambda x,y,z: SIG*(x-circle.center.x)/r * np.abs((y-circle.center.y)/r)], ["x"])
            simu.add_surfLoad(nodesCircle, [lambda x,y,z: SIG*(y-circle.center.y)/r * np.abs((y-circle.center.y)/r)], ["y"])

            # simu.add_surfLoad(nodesCircle, [lambda x,y,z : SIG*(y-circle.center.y)/r], ["y"])
        else:
            simu.add_surfLoad(nodesh, [-SIG], ["y"])

        # Affichage.Plot_BoundaryConditions(simu)

        simu.Solve()

        psipa = np.mean(simu.Get_Resultat("psiP", True)[nodeA])*E/SIG**2
        psipb = np.mean(simu.Get_Resultat("psiP", True)[nodeB])*E/SIG**2

        if isCP:
            result['A (CP)'] = psipa
            result['B (CP)'] = psipb

            result['errA (CP)'] = np.abs(psipa-Miehe_psiP_A(v))/Miehe_psiP_A(v)
            result['errB (CP)'] = np.abs(psipb-Miehe_psiP_B(v))/Miehe_psiP_B(v)

            Affichage.Plot_Result(simu, "psiP", valeursAuxNoeuds=True, coef=E/SIG**2, title=fr"$\psi_{0}^+\ E / \sigma^2 \ pour \ \nu={v}$", folder=folder,filename=f"psiP {nom} v={v}", colorbarIsClose=True)
        else:
            result['A (DP)'] = psipa
            result['B (DP)'] = psipb

            result['errA (DP)'] = np.abs(psipa-Miehe_psiP_A(v))/Miehe_psiP_A(v)
            result['errB (DP)'] = np.abs(psipb-Miehe_psiP_B(v))/Miehe_psiP_B(v)

    

    result['A (ana CP)'] = Miehe_psiP_A(v)
    result['B (ana CP)'] = Miehe_psiP_B(v)
    
    new = pd.DataFrame(result, index=[0])

    df = pd.concat([df, new], ignore_index=True)

# df.to_excel(Folder.Join([folder, f"{nom}.xlsx"]), index=False)

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
    Affichage.Plot_Result(simu, "Sxx", valeursAuxNoeuds=True, coef=1/SIG, title=r"$\sigma_{xx} / \sigma$",folder=folder, filename='Sxx', colorbarIsClose=True)
    Affichage.Plot_Result(simu, "Syy", valeursAuxNoeuds=True, coef=1/SIG, title=r"$\sigma_{yy} / \sigma$", folder=folder, filename='Syy', colorbarIsClose=True)
    Affichage.Plot_Result(simu, "Sxy", valeursAuxNoeuds=True, coef=1/SIG, title=r"$\sigma_{xy} / \sigma$", folder=folder, filename='Sxy', colorbarIsClose=True)



Affichage.NouvelleSection("Calcul analytique")


fig, axp = plt.subplots()

list_v = np.arange(0, 0.5,0.0005)

# test = (vv*(1-2*vv)+1)/(2*(1+vv))

# axp.plot(list_v, (list_v*(1-2*list_v)+1)/(2*(1+list_v)), label="psiP_A*E/Sig^2")
axp.plot(list_v, Miehe_psiP_A(list_v), label='A')
axp.plot(list_v, Miehe_psiP_B(list_v), label='B')
axp.grid()
axp.legend(fontsize=14)
axp.set_xlabel(r"$\nu$",fontsize=14)
axp.set_ylabel("$\psi_{0}^+\ E / \sigma^2$",fontsize=14)
axp.set_title(r'Split sur $\varepsilon$',fontsize=14)

PostTraitement.Save_fig(folder, "calc analytique")

list_Amor_psiP_A=[]
list_Amor_psiP_B=[]

list_Miehe_psiP_A=[]
list_Miehe_psiP_B=[]

list_Stress_psiP_A=[]
list_Stress_psiP_B=[]

for v in list_v:
    
    Eps_A = (1+v)/E*Sig_A - v/E*np.trace(Sig_A)*np.eye(3); trEps_A = np.trace(Eps_A); trEpsP_A = (trEps_A+np.abs(trEps_A))/2
    Eps_B = (1+v)/E*Sig_B - v/E*np.trace(Sig_B)*np.eye(3); trEps_B = np.trace(Eps_B); trEpsP_B = (trEps_B+np.abs(trEps_B))/2

    # Eps_A = (1+v)/E*Sig_A - v/E*np.trace(Sig_A)*np.eye(3); trEps_A = np.trace(Eps_A); trEpsP_A = (trEps_A+np.abs(trEps_A))/2
    # Eps_B = (1+v)/E*Sig_B - v/E*np.trace(Sig_B)*np.eye(3); trEps_B = np.trace(Eps_B); trEpsP_B = (trEps_B+np.abs(trEps_B))/2

    l = v*E/((1+v)*(1-2*v))
    # l = v*E/(1-v**2)
    mu=E/(2*(1+v))

    # Split Amor
    bulk = l + 2/3*mu

    spherA = 1/3*trEps_A*np.eye(3)
    spherB = 1/3*trEps_B*np.eye(3)

    EpsD_A = Eps_A-spherA
    EpsD_B = Eps_B-spherB

    Amor_psi_A = 1/2*bulk*trEpsP_A**2 + mu * np.einsum('ij,ij', EpsD_A, EpsD_A)
    Amor_psi_B = 1/2*bulk*trEpsP_B**2 + mu * np.einsum('ij,ij', EpsD_B, EpsD_B)

    list_Amor_psiP_A.append(Amor_psi_A)
    list_Amor_psiP_B.append(Amor_psi_B)

    # Split Miehe
    Epsi_A = np.diag(np.linalg.eigvals(Eps_A)); Epsip_A = (Epsi_A+np.abs(Epsi_A))/2
    Epsi_B = np.diag(np.linalg.eigvals(Eps_B)); Epsip_B = (Epsi_B+np.abs(Epsi_B))/2

    Miehe_psiP_A = l/2*trEpsP_A**2 + mu*np.einsum('ij,ij',Epsip_A,Epsip_A)
    Miehe_psiP_B = l/2*trEpsP_B**2 + mu*np.einsum('ij,ij',Epsip_B,Epsip_B)

    list_Miehe_psiP_A.append(Miehe_psiP_A)
    list_Miehe_psiP_B.append(Miehe_psiP_B)

    # Split Stress
    Sigi_A = np.diag(np.linalg.eigvals(Sig_A)); Sigip_A = (Sigi_A+np.abs(Sigi_A))/2
    Sigi_B = np.diag(np.linalg.eigvals(Sig_B)); Sigip_B = (Sigi_B+np.abs(Sigi_B))/2
    
    trSig_A = np.trace(Sig_A); trSigP_A = (trSig_A+np.abs(trSig_A))/2
    trSig_B = np.trace(Sig_B); trSigP_B = (trSig_B+np.abs(trSig_B))/2

    Stress_psiP_A = ((1+v)/E*np.einsum('ij,ij',Sigip_A,Sigip_A) - v/E * trSigP_A**2)/2
    Miehe_psiP_B = ((1+v)/E*np.einsum('ij,ij',Sigip_B,Sigip_B) - v/E * trSigP_B**2)/2

    list_Stress_psiP_A.append(Stress_psiP_A)
    list_Stress_psiP_B.append(Miehe_psiP_B)



fig, ax1 = plt.subplots()

ax1.plot(list_v, np.array(list_Miehe_psiP_A)*E/SIG**2, label='A')
ax1.plot(list_v, np.array(list_Miehe_psiP_B)*E/SIG**2, label='B')
ax1.grid()
if split == "Miehe":    
    ax1.scatter(list_V, np.array(df['A (CP)'].tolist()),label='num A')
    ax1.scatter(list_V, np.array(df['B (CP)'].tolist()),label='num B')
ax1.legend(fontsize=14)
ax1.set_xlabel(r"$\nu$",fontsize=14)
ax1.set_ylabel("$\psi_{0}^+\ E / \sigma^2$",fontsize=14)
ax1.set_title(r'Split sur $\varepsilon$',fontsize=14)

PostTraitement.Save_fig(folder, "Miehe psiP")

fig, ax2 = plt.subplots()

stressA = lambda v: 1/E*(SxxA**2+SyyA**2-2*SxxA*SyyA*v)

ax2.plot(list_v, np.array(list_Stress_psiP_A)*E/SIG**2, label='A')
ax2.plot(list_v, np.array(list_Stress_psiP_B)*E/SIG**2, label='B')
# ax2.plot(list_v, np.ones(list_v.shape), label='A')
# ax2.plot(list_v, stressA(list_v)*E/SIG**2, label='AA')
ax2.grid()
if split == "Stress":    
    ax2.scatter(list_V, np.array(df['A (CP)'].tolist()),label='num A')
    ax2.scatter(list_V, np.array(df['B (CP)'].tolist()),label='num B')
ax2.legend(fontsize=14)
ax2.set_xlabel(r"$\nu$",fontsize=14)
ax2.set_ylabel("$\psi_{0}^+\ E / \sigma^2$",fontsize=14)
ax2.set_title('Split sur $\sigma$',fontsize=14)

PostTraitement.Save_fig(folder, "Stress psiP")

fig, ax3 = plt.subplots()

ax3.plot(list_v, np.array(list_Amor_psiP_A)*E/SIG**2, label='A')
ax3.plot(list_v, np.array(list_Amor_psiP_B)*E/SIG**2, label='B')
ax3.grid()
if split == "Amor":    
    ax3.scatter(list_V, np.array(df['A (CP)'].tolist()),label='num A')
    ax3.scatter(list_V, np.array(df['B (CP)'].tolist()),label='num B')
ax3.legend(fontsize=14)
ax3.set_xlabel(r"$\nu$",fontsize=14)
ax3.set_ylabel("$\psi_{0}^+\ E / \sigma^2$",fontsize=14)
ax3.set_title('Split Amor',fontsize=14)

PostTraitement.Save_fig(folder, "Amor psiP")







Tic.getResume()

plt.show()



pass


