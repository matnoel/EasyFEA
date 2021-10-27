import os
from typing import cast

from numpy.core.arrayprint import dtype_is_implied
from numpy.core.function_base import linspace

from Affichage import Affichage
from class_Materiau import Materiau
from class_ModelGmsh import ModelGmsh
from class_Noeud import Noeud
from class_Simu import Simu
from class_Mesh import Mesh

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.collections

from class_TicTac import TicTac


os.system("cls")    #nettoie le terminal

# Data --------------------------------------------------------------------------------------------

plotResult = True

dim = 2

# Paramètres géométrie
L = 1;  #mm

# Paramètres maillage
type = ModelGmsh.get_typesMaillage2D()[0]
taille = L/40

# Materiau
materiau = Materiau(dim,v=0.2)

# Construction du modele et du maillage --------------------------------------------------------------------------------
modelGmsh = ModelGmsh(dim, organisationMaillage=True, typeElement=type, tailleElement=taille)

(coordos, connect) = modelGmsh.ConstructionRectangle(L, L)

mesh = Mesh(dim, coordos, connect)

# Récupère les noeuds qui m'interessent
noeuds_Haut = [mesh.noeuds[i] for i in range(mesh.Nn) if mesh.noeuds[i].coordo[1] == L]
noeuds_Milieu = [mesh.noeuds[i] for i in range(mesh.Nn) if np.isclose(mesh.noeuds[i].coordo[1], L/2,0.01) and (mesh.noeuds[i].coordo[0] <= L/2 or np.isclose(mesh.noeuds[i].coordo[0], L/2))]
noeuds_Bas = [mesh.noeuds[i] for i in range(mesh.Nn) if mesh.noeuds[i].coordo[1] == 0]
noeuds_Gauche = [mesh.noeuds[i] for i in range(mesh.Nn) if mesh.noeuds[i].coordo[0] == 0]
noeuds_Droite = [mesh.noeuds[i] for i in range(mesh.Nn) if mesh.noeuds[i].coordo[0] == L]

NoeudsBord=[]
NoeudsBord.extend(noeuds_Bas); NoeudsBord.extend(noeuds_Droite); NoeudsBord.extend(noeuds_Gauche); NoeudsBord.extend(noeuds_Haut)
# Affichage maillage pour verifier


fig, ax = Affichage.PlotMesh(mesh, None)
Affichage.AfficheNoeudsMaillage(dim, ax, noeuds=noeuds_Haut, marker='*', c='blue')
Affichage.AfficheNoeudsMaillage(dim, ax, noeuds=noeuds_Milieu, marker='o', c='red')
Affichage.AfficheNoeudsMaillage(dim, ax, noeuds=noeuds_Bas, marker='.', c='blue')
Affichage.AfficheNoeudsMaillage(dim, ax, noeuds=noeuds_Gauche, marker='.', c='black')
Affichage.AfficheNoeudsMaillage(dim, ax, noeuds=noeuds_Droite, marker='.', c='black')        
plt.show()

# ------------------------------------------------------------------------------------------------------
Affichage.NouvelleSection("Simulations")

simu = Simu(dim, mesh, materiau, verbosity=False)

u_tn = np.zeros(mesh.Nn*dim)
d_tn = np.zeros(mesh.Nn)

for n in noeuds_Milieu:
        n = cast(Noeud, n)
        d_tn[n.id] = 1

deteriorations = [d_tn]
coordos = [np.zeros((mesh.Nn,3))]

dep = 0
dep_iter = 0.005*10
for iter in range(100):
        
        print(iter+1)
        
        # Construit H
        simu.ConstruitH(u_tn, d_tn)

        # PFM problem ======================================================================
        
        simu.Assemblage_d(Gc=2700, l=0.015)    # Assemblage
        simu.Condition_Dirichlet(noeuds_Milieu, valeur=1, option="d")       # CL        
        d_tn = simu.Solve_d(resolution=2)   # resolution
        
        deteriorations.append(d_tn)

        # Linear displacement problem

        simu.Assemblage_u(epaisseur=1, d=d_tn)
        
        # Déplacement en haut
        dep += dep_iter
        simu.Condition_Dirichlet(noeuds_Haut, valeur=dep, directions=["x"])

        # Gauche et droite
        simu.Condition_Dirichlet(noeuds_Gauche, valeur=0, directions=["y"])
        simu.Condition_Dirichlet(noeuds_Droite, valeur=0, directions=["y"])

        # Encastrement
        simu.Condition_Dirichlet(noeuds_Bas, valeur=0, directions=["x", "y"])        
       
        u_tn = simu.Solve_u(resolution=1, save=False)

        ux = u_tn[np.arange(0, mesh.Nn*2, 2)]
        uy = u_tn[np.arange(1, mesh.Nn*2, 2)]
        
        coordos.append(np.array([ux, uy, np.zeros(mesh.Nn)]).T)
       
        

        

# Affichage
Affichage.NouvelleSection("Post traitement")

connectPolygon = mesh.get_connectPolygon()

fig, ax = plt.subplots()




iter = 0
for d in deteriorations:
        
        print("\ndmax = {}".format(np.max(d)))
        print("dmin = {}".format(np.min(d)))
        
        IdnoeudsNegatif = [mesh.noeuds[i].id for i in range(mesh.Nn) if d[i] < 0]

        coord_xy = mesh.coordo[:,[0,1]] + coordos[iter][:,[0,1]]  *0 

        # # affichage maillage              
        vertices = [[coord_xy[connectPolygon[ix][iy]] for iy in range(len(connectPolygon[0]))] for ix in range(len(connectPolygon))]
        pc = matplotlib.collections.LineCollection(vertices, edgecolor='black', lw=0.5)
        ax.add_collection(pc)

        # Valeur aux noeuds
        levels = linspace(0, 1, 100)        
        pc = ax.tricontourf(coord_xy[:,0], coord_xy[:,1], mesh.get_connectTriangle(), d, levels, cmap='jet')

        x = coord_xy[IdnoeudsNegatif, 0]
        y = coord_xy[IdnoeudsNegatif, 1]
        
        ax.scatter(x, y)

        cb = fig.colorbar(pc, ax=ax)
        ax.axis('equal')
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')        
        ax.set_title(iter+1)
        
        plt.pause(0.001)
        
        if iter != len(deteriorations):
                ax.clear()
                cb.remove()
    
        iter+=1
pass

