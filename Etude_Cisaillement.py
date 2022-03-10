import os

from classes.Affichage import Affichage
from classes.Materiau import Materiau
from classes.ModelGmsh import ModelGmsh
from classes.Simu import Simu
from classes.Mesh import Mesh

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from classes.TicTac import TicTac

os.system("cls")    #nettoie le terminal

# Data --------------------------------------------------------------------------------------------

plotResult = True

dim = 2

# Paramètres géométrie
L = 1;  #mm
l0 = 7.5e-3 # taille fissure test femobject

# Paramètres maillage
# taille = l0/2
taille = 1e-2 #taille maille test fem object

# Construction du modele et du maillage --------------------------------------------------------------------------------
modelGmsh = ModelGmsh(dim, organisationMaillage=False, typeElement=0, tailleElement=taille, affichageGmsh=False)

(coordos, connect) = modelGmsh.ConstructionRectangleAvecFissure(L, L)

mesh = Mesh(dim, coordos, connect)

# Récupère les noeuds qui m'interessent
noeuds_Milieu = mesh.Get_Nodes(conditionX=lambda x: x <= L/2, conditionY=lambda y: y == L/2)
noeuds_Haut = mesh.Get_Nodes(conditionY=lambda y: y == L)
noeuds_Bas = mesh.Get_Nodes(conditionY=lambda y: y == 0)
noeuds_Gauche = mesh.Get_Nodes(conditionX=lambda x: x == 0)
noeuds_Droite = mesh.Get_Nodes(conditionX=lambda x: x == L)



NoeudsBord=[]
NoeudsBord.extend(noeuds_Bas); NoeudsBord.extend(noeuds_Droite); NoeudsBord.extend(noeuds_Gauche); NoeudsBord.extend(noeuds_Haut)

# ------------------------------------------------------------------------------------------------------
Affichage.NouvelleSection("Simulations")

materiau = Materiau(dim,v=0.3, contraintesPlanes=False)

simu = Simu(dim, mesh, materiau, verbosity=False)

u_tn = np.zeros(mesh.Nn*dim)
d_tn = np.zeros(mesh.Nn)

d_tn[noeuds_Milieu] = 1

deteriorations = [d_tn]
coordos = [np.zeros((mesh.Nn,3))]

# Renseignement des conditions limites

# Endommagement initiale
simu.Condition_Dirichlet(noeuds_Milieu, valeur=1, option="d")

def RenseigneConditionsLimites():
        # # Conditions en déplacements à Gauche et droite
        # simu.Condition_Dirichlet(noeuds_Gauche, valeur=0.0, directions=["y"])
        # simu.Condition_Dirichlet(noeuds_Droite, valeur=0.0, directions=["y"])

        # Conditions en déplacements en Bas
        simu.Condition_Dirichlet(noeuds_Bas, valeur=0.0, directions=["x", "y"])

        # Conditions en déplacements en Haut
        simu.Condition_Dirichlet(noeuds_Haut, valeur=0.0, directions=["y"])

RenseigneConditionsLimites()

def AffichageCL():
        # Affichage noeuds du maillage
        fig1, ax = Affichage.Plot_Maillage(simu)
        Affichage.Plot_NoeudsMaillage(simu, ax, noeuds=noeuds_Haut, marker='*', c='blue')
        Affichage.Plot_NoeudsMaillage(simu, ax, noeuds=noeuds_Milieu, marker='o', c='red')
        Affichage.Plot_NoeudsMaillage(simu, ax, noeuds=noeuds_Bas, marker='.', c='blue')
        Affichage.Plot_NoeudsMaillage(simu, ax, noeuds=noeuds_Gauche, marker='.', c='black')
        Affichage.Plot_NoeudsMaillage(simu, ax, noeuds=noeuds_Droite, marker='.', c='black')

# AffichageCL()
# plt.show()


N = 30

# u_inc = 5e-8
u_inc = 5e-3
dep = 0

for iter in range(N):
        
        # Construit H
        simu.ConstruitH(u_tn)

        #-------------------------- PFM problem ------------------------------------
        
        simu.Assemblage_d(Gc=2.7, l=l0)    # Assemblage
        
        d_tn = simu.Solve_d()   # resolution
        
        deteriorations.append(d_tn)

        #-------------------------- Dep problem ------------------------------------
        simu.Assemblage_u(d=d_tn)

        # Déplacement en haut
        dep += u_inc

        simu.Condition_Dirichlet(noeuds_Haut, valeur=dep, directions=["x"])
        
        u_tn = simu.Solve_u()

        simu.Clear_Condition_Dirichlet()
        RenseigneConditionsLimites()

        ux = u_tn[np.arange(0, mesh.Nn*2, 2)]
        uy = u_tn[np.arange(1, mesh.Nn*2, 2)]
        
        coordos.append(np.array([ux, uy, np.zeros(mesh.Nn)]).T)

        min = np.min(d_tn)
        max = np.max(d_tn)
        print(iter+1," : min d = {:.5f}, max d = {:.5f} ".format(min, max))
       
        # if d_tn.max()>1:
        #         break       


# ------------------------------------------------------------------------------------------------------
Affichage.NouvelleSection("Post traitement")

AffichageCL()

connectPolygon = mesh.get_connect_Faces()

fig, ax = plt.subplots()

# np.save()

coord_xy = mesh.coordo[:,[0,1]]
# coord_xy = mesh.coordo[:,[0,1]] + coordos[iter][:,[0,1]]  *1
vertices = [[coord_xy[connectPolygon[ix][iy]] for iy in range(len(connectPolygon[0]))] for ix in range(len(connectPolygon))]

iter = 0
for d in deteriorations:
        
        # print("\ndmax = {}".format(np.max(d)))
        # print("dmin = {}".format(np.min(d)))
                
        
        # affichage maillage              
        
        # pc = matplotlib.collections.LineCollection(vertices, edgecolor='black', lw=0.5)
        # ax.add_collection(pc)

        # Valeur aux noeuds
        levels = np.linspace(d.min(), d.max(), 100)        
        pc = ax.tricontourf(coord_xy[:,0], coord_xy[:,1], mesh.get_connectTriangle(), d, levels, cmap='jet')

        # IdnoeudsNegatif = [mesh.noeuds[i].id for i in range(mesh.Nn) if d[i] < 0]
        # x = coord_xy[IdnoeudsNegatif, 0]
        # y = coord_xy[IdnoeudsNegatif, 1]        
        # ax.scatter(x, y)

        cb = fig.colorbar(pc, ax=ax)
        ax.axis('equal')
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')        
        ax.set_title(iter)
        
        plt.pause(0.0001)
        
        if iter+1 != len(deteriorations):
                ax.clear()
                cb.remove()
    
        iter+=1
plt.show()

