# %%

import Dossier
from Affichage import Affichage
from Materiau import PhaseFieldModel, Elas_Isot, Materiau
from ModelGmsh import ModelGmsh
from Simu import Simu
from Mesh import Mesh
from TicTac import TicTac

import numpy as np
import matplotlib.pyplot as plt

Affichage.Clear()

# Data --------------------------------------------------------------------------------------------

plotResult = True
save = False

dim = 2

# Paramètres géométrie
L = 1e-3;  #m
l0 = 7.5e-6 # taille fissure test femobject
Gc = 2.7e3

# Paramètres maillage
# taille = l0/2
taille = 1e-5 #taille maille test fem object

# Construction du modele et du maillage --------------------------------------------------------------------------------
modelGmsh = ModelGmsh(dim, organisationMaillage=False, typeElement=0, tailleElement=taille, affichageGmsh=False)

(coordos_tn, connect) = modelGmsh.ConstructionRectangleAvecFissure(L, L)

mesh = Mesh(dim, coordos_tn, connect)

# Récupère les noeuds qui m'interessent
noeuds_Milieu = mesh.Get_Nodes(conditionX=lambda x: x <= L/2, conditionY=lambda y: y == L/2)
noeuds_Haut = mesh.Get_Nodes(conditionY=lambda y: y == L)
noeuds_Bas = mesh.Get_Nodes(conditionY=lambda y: y == 0)
noeuds_Gauche = mesh.Get_Nodes(conditionX=lambda x: x == 0, conditionY=lambda y: y>0 and y <L)
noeuds_Droite = mesh.Get_Nodes(conditionX=lambda x: x == L, conditionY=lambda y: y>0 and y <L)

NoeudsBord=[]
NoeudsBord.extend(noeuds_Bas); NoeudsBord.extend(noeuds_Droite); NoeudsBord.extend(noeuds_Gauche); NoeudsBord.extend(noeuds_Haut)

# ------------------------------------------------------------------------------------------------------
Affichage.NouvelleSection("Simulations")

comportement = Elas_Isot(dim, E=210e9, v=0.3, contraintesPlanes=False)

phaseFieldModel = PhaseFieldModel(comportement, "Bourdin", "AT2", Gc=Gc, l_0=l0)

materiau = Materiau(comportement, ro=1, phaseFieldModel=phaseFieldModel)

simu = Simu(dim, mesh, materiau, verbosity=False)

u_0 = np.zeros(mesh.Nn*dim)

coordos_tn = [np.zeros((mesh.Nn,3))]

# Renseignement des conditions limites

# Endommagement initiale
simu.Condition_Dirichlet(noeuds_Milieu, valeur=1, option="d")

def RenseigneConditionsLimites():
        # # Conditions en déplacements à Gauche et droite
        simu.Condition_Dirichlet(noeuds_Gauche, valeur=0.0, directions=["y"])
        simu.Condition_Dirichlet(noeuds_Droite, valeur=0.0, directions=["y"])

        # Conditions en déplacements en Bas
        simu.Condition_Dirichlet(noeuds_Bas, valeur=0.0, directions=["x", "y"])

        # Conditions en déplacements en Haut
        simu.Condition_Dirichlet(noeuds_Haut, valeur=0.0, directions=["y"])

RenseigneConditionsLimites()

# N = 400
N = 10

u_tn = u_0

deteriorations=[]

u_inc = 5e-8
# u_inc = 5e-7
dep = 0

tic = TicTac()

for iter in range(N):
        
        # Construit H
        simu.CalcPsiPlus_e_pg(u_tn)

        #-------------------------- PFM problem ------------------------------------
        
        simu.Assemblage_d(Gc=Gc, l=l0)    # Assemblage
        
        d_tn = simu.Solve_d()   # resolution

        deteriorations.append(d_tn)

        #-------------------------- Dep problem ------------------------------------
        simu.Assemblage_u(d=d_tn)

        # Déplacement en haut
        dep += u_inc

        simu.Condition_Dirichlet(noeuds_Haut, valeur=dep, directions=["x"])
        
        u_tn = simu.Solve_u(useCholesky=True)

        simu.Clear_Condition_Dirichlet()
        RenseigneConditionsLimites()

        ux = u_tn[np.arange(0, mesh.Nn*2, 2)]
        uy = u_tn[np.arange(1, mesh.Nn*2, 2)]
        
        coordos_tn.append(np.array([ux, uy, np.zeros(mesh.Nn)]).T)

        min = np.min(d_tn)
        max = np.max(d_tn)
        norm = np.sum(d_tn)

        tResolution = tic.Tac("Resolution Phase Field", "Resolution Phase field", False)
        print(iter+1," : sum d = {:.5f}, time = {:.3f}".format(norm, tResolution))

        # simu.SaveParaview(nodesField=["damage"])
        # Affichage.Plot_Result(simu, "damage", valeursAuxNoeuds=True, affichageMaillage=True)

        # plt.show()
       
        # if d_tn.max()>1:
        #         break
        filename = Dossier.NewFile("EtudeCisaillement2D\\damage.vtu", results=True)
        simu.SaveParaview(filename, nodesField=["damage"]) 
        Stresses = simu.GetResultat("Sxx")




# ------------------------------------------------------------------------------------------------------
Affichage.NouvelleSection("Affichage")

def AffichageCL():
        # Affichage noeuds du maillage
        fig1, ax = Affichage.Plot_Maillage(simu)
        Affichage.Plot_NoeudsMaillage(simu, ax, noeuds=noeuds_Haut, marker='*', c='blue')
        Affichage.Plot_NoeudsMaillage(simu, ax, noeuds=noeuds_Milieu, marker='o', c='red')
        Affichage.Plot_NoeudsMaillage(simu, ax, noeuds=noeuds_Bas, marker='.', c='blue')
        Affichage.Plot_NoeudsMaillage(simu, ax, noeuds=noeuds_Gauche, marker='.', c='black')
        Affichage.Plot_NoeudsMaillage(simu, ax, noeuds=noeuds_Droite, marker='.', c='black')

# AffichageCL()
# if save:
#         plt.savefig(Dossier.GetPath(__file__)+"\\results\\ConditionsLimites.pdf")

if plotResult:

        import matplotlib
        import matplotlib.animation as animation

        # Importations des données du maillage
        connectFaces = mesh.get_connect_Faces()
        connectTriangle = mesh.get_connectTriangle()
        coord_xy = mesh.coordo[:,range(dim)]

        # Creation figure et axes avec options
        fig, ax = plt.subplots()
        # Paramètres colorbar
        levels = np.linspace(0, 1, 500)
        ticks = np.linspace(0,1,11)

        # Nom de la vidéo
        filename = Dossier.GetPath(__file__) + "\\results\\video.mp4"

        ffmpegpath = "D:\\SOFT\\ffmpeg\\bin\\ffmpeg.exe"
        matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath

        writer = animation.FFMpegWriter(fps=30)
        with writer.saving(fig, filename, 200):
                iter=0
                for d in deteriorations:
                        print(iter)
                        ax.clear()
                        image = ax.tricontourf(coord_xy[:,0], coord_xy[:,1], mesh.get_connectTriangle(), d, levels, cmap='jet')
                        
                        if iter == 0:
                                cb = plt.colorbar(image, ax=ax, ticks=ticks)

                        ax.axis('equal')                
                        ax.set_title(iter+1)

                        writer.grab_frame()
                        iter+=1

        # anim = animation.ArtistAnimation(fig, images, interval=100, repeat_delay=3000, blit=True)

        # histo = []

        # def AnimationEndommagement(frame):        

        #         print(frame)
                
        #         ax.clear()        
                
        #         d = deteriorations[frame]        
        #         tri = ax.tricontourf(coord_xy[:,0], coord_xy[:,1], connectTriangle, d, levels, cmap='jet')
        #         if len(histo)==0:
        #                 cb = plt.colorbar(tri, ax=ax, ticks=ticks)
        #                 histo.append(frame)
                
        #         ax.axis('equal')
        #         ax.set_xlabel('x [mm]')
        #         ax.set_ylabel('y [mm]')
        #         ax.set_title(frame)

        # writer = animation.FFMpegWriter(fps=10)
        # anim = animation.FuncAnimation(fig, AnimationEndommagement, frames=N, repeat=False)


        # if save:
        #         anim.save(filename, writer=writer)

TicTac.getResume()

plt.show()


# %%
