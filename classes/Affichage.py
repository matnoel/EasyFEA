
from typing import cast
import os
import numpy as np

import matplotlib.collections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def Plot_Result(simu, option: str , deformation=False, facteurDef=4, coef=1, unite="",
    affichageMaillage=False, valeursAuxNoeuds=False, oldfig=None, oldax=None, folder=""):

    """Affichage de la simulation

    Parameters
    ----------
    simu : Simu
        Simulation
    val : str
        Ce quil sera affiché
    deformation : bool, optional
        deformation du domaine, by default False
    facteurDef : int, optional
        facteur de deformation du domaine, by default 4
    affichageMaillage : bool, optional
        affcihe le mailllage, by default False
    """

    # Detecte si on donne bien ax et fig en meme temps
    assert (oldfig == None) == (oldax == None), "Doit fournir oldax et oldfix ensemble"

    if (not oldfig == None) and (not oldax == None):
        assert isinstance(oldfig, plt.Figure) and isinstance(oldax, plt.Axes)           

    # Va chercher les valeurs 0 a affciher

    from Simu import Simu
    from TicTac import TicTac

    tic = TicTac()
    simu = cast(Simu, simu) # ne pas ecrire simu: Simu ça créer un appel circulaire

    mesh = simu.mesh
    
    dim = mesh.dim

    if dim==3:
        valeursAuxNoeuds=False

    valeurs = simu.Get_Resultat(option, valeursAuxNoeuds)
    if not isinstance(valeurs, np.ndarray):
        return

    valeurs *= coef

    coordo, deformation = __GetCoordo(simu, deformation, facteurDef)

    # construit la matrice de connection pour les faces
    connect_Faces = mesh.get_connect_Faces()

    # Construit les faces non deformées
    coordo_redim = coordo[:,range(dim)]
    coord_par_face = coordo_redim[connect_Faces]
    
    if option == "damage":
        levels = np.linspace(0, 1, 200)
    else:
        levels = 200

    if dim == 2:
        # Construit les vertices
        if oldax == None:
            fig, ax = plt.subplots()
        else:
            fig = oldfig
            ax = oldax
            ax.clear()
            
        # Trace le maillage
        if affichageMaillage:
            pc = matplotlib.collections.LineCollection(coord_par_face, edgecolor='black', lw=0.5)
            ax.add_collection(pc)

        # Valeurs aux element
        if mesh.Ne == len(valeurs):
            pc = matplotlib.collections.PolyCollection(coord_par_face, lw=0.5, cmap='jet')
            pc.set_clim(valeurs.min(), valeurs.max())
            pc.set_array(valeurs)
            ax.add_collection(pc)
                            
            # dx_e = resultats["dx_e"]
            # dy_e = resultats["dy_e"]
            # # x,y=np.meshgrid(dx_e,dy_e)
            # pc = ax.tricontourf(dx_e, dy_e, valeurs, levels ,cmap='jet')            

        # Valeur aux noeuds
        elif mesh.Nn == len(valeurs):
            pc = ax.tricontourf(coordo[:,0], coordo[:,1], mesh.get_connectTriangle(), valeurs, levels ,
            cmap='jet')
            # tripcolor, tricontour, tricontourf

        if option == "damage":
            ticks = np.linspace(0,1,11)
            cb = plt.colorbar(pc, ax=ax, ticks=ticks)                    
        else:
            cb = plt.colorbar(pc, ax=ax)
            

        ax.axis('equal')
        # ax.set_xlabel('x [mm]')
        # ax.set_ylabel('y [mm]')

    
    elif mesh.dim == 3:

        # Construit les vertices
        coord_xyz = coordo            
        coord_par_face = coord_xyz[connect_Faces]
        
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        
        # Trace le maillage
        if affichageMaillage:                
            pc = Poly3DCollection(coord_par_face, edgecolor='black', linewidths=0.5, cmap='jet')
        else:
            pc = Poly3DCollection(coord_par_face, cmap='jet')
        ax.add_collection3d(pc)        
        
        valeursAuFaces = valeurs.reshape(mesh.Ne, 1).repeat(mesh.get_nbFaces(), axis=1).reshape(-1)
        
        # ax.scatter(coordo[:,0],coordo[:,1],coordo[:,2], linewidth=0, alpha=0)
        pc.set_clim(valeursAuFaces.min(), valeursAuFaces.max())
        pc.set_array(valeursAuFaces)

        cb = fig.colorbar(pc, ax=ax)       
        ax.add_collection(pc)            
        # ax.set_xlabel("x [mm]")
        # ax.set_ylabel("y [mm]")
        # ax.set_zlabel("z [mm]")            
        
        __ChangeEchelle(ax, coordo)

    
    if valeursAuxNoeuds:
        loc = "_n"
    else:
        loc = "_e"

    title = option+loc+unite

    ax.set_title(title)

    if folder != "":
        import PostTraitement
        PostTraitement.Save_fig(folder, title)

    tic.Tac("Post Traitement", "Affichage résultat", False)
    
    return fig, ax, cb
    
def Plot_Maillage(obj, ax=None, facteurDef=4, deformation=False, lw=0.5 ,alpha=1, folder=""):
    """Dessine le maillage de la simulation

    Parameters
    ----------
    obj : Simu or Mesh
        obj that contains mesh
    facteurDef : int, optional
        facteur de deformation, by default 4
    deformation : bool, optional
        affiche le maillage deformé, by default False
    lw : float, optional
        epaisseur des traits, by default 0.5
    alpha : int, optional
        transparence du maillage, by default 1

    Returns
    -------
    ax : plt.Axes
        Axes dans lequel on va creer la figure
    """

    from Simu import Simu
    from Mesh import Mesh

    match type(obj).__name__:
        case Simu.__name__:
            simu = cast(Simu, obj)
            mesh = simu.mesh
        case Mesh.__name__:
            mesh = cast(Mesh, obj)
            if deformation == True:
                print("Il faut donner la simulation pour afficher le maillage déformée")

    assert facteurDef > 1, "Le facteur de deformation doit être >= 1"

    coordo = mesh.coordoGlob

    dim = mesh.dim

    # construit la matrice de connection pour les faces
    connect_Faces = mesh.get_connect_Faces()

    # Construit les faces non deformées
    coord_NonDeforme_redim = coordo[:,range(dim)]
    coord_par_face = coord_NonDeforme_redim[connect_Faces]

    if deformation:
        coordoDeforme, deformation = __GetCoordo(simu, deformation, facteurDef)
        coordo_Deforme_redim = coordoDeforme[:,range(dim)]
        coordo_par_face_deforme = coordo_Deforme_redim[connect_Faces]

    # ETUDE 2D
    if dim == 2:
        
        if ax == None:
            fig, ax = plt.subplots()
        
        if deformation:
            # Superpose maillage non deformé et deformé
            # Maillage non deformés
            pc = matplotlib.collections.LineCollection(coord_par_face, edgecolor='black', lw=lw)
            ax.add_collection(pc)

            # Maillage deformé                
            pc = matplotlib.collections.LineCollection(coordo_par_face_deforme, edgecolor='red', lw=lw)
            ax.add_collection(pc)
        else:
            # Maillage non deformé
            if alpha == 0:
                pc = matplotlib.collections.LineCollection(coord_par_face, edgecolor='black', lw=lw)
            else:
                pc = matplotlib.collections.PolyCollection(coord_par_face, facecolors='c', edgecolor='black', lw=lw)
            ax.add_collection(pc)
        
        ax.autoscale()
        ax.axis('equal')
        # ax.set_xlabel("x [mm]")
        # ax.set_ylabel("y [mm]")

    # ETUDE 3D    
    if mesh.dim == 3:
        
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

        # fig = plt.figure()            
        # ax = fig.add_subplot(projection="3d")
        
        if deformation:
            # Supperpose les deux maillages
            # Maillage non deformé
            # ax.scatter(x,y,z, linewidth=0, alpha=0)
            ax.add_collection3d(Poly3DCollection(coord_par_face, edgecolor='black', linewidths=0.5, alpha=0))

            # Maillage deformé
            ax.add_collection3d(Poly3DCollection(coordo_par_face_deforme, edgecolor='red', linewidths=0.5, alpha=0))
        else:
            # ax.scatter(x,y,z, linewidth=0, alpha=0)
            ax.add_collection3d(Poly3DCollection(coord_par_face, facecolors='c', edgecolor='black', linewidths=1, alpha=1))


        # ax.autoscale()
        # ax.set_xlabel("x [mm]")
        # ax.set_ylabel("y [mm]")
        # ax.set_zlabel("z [mm]")

        __ChangeEchelle(ax, coordo)
    
    title = f"{mesh.elemType} : Ne = {mesh.Ne} et Nn = {mesh.Nn}"

    ax.set_title(title)

    if folder != "":
        import PostTraitement
        PostTraitement.Save_fig(folder, title)

    return ax

def Plot_NoeudsMaillage(mesh, ax=None, noeuds=[], showId=False, marker='.', c='blue', folder=""):
    """Affiche les noeuds du maillage"""        
    
    from Mesh import Mesh
    mesh = cast(Mesh, mesh)

    if ax == None:
        ax = Plot_Maillage(mesh, alpha=0)
    
    if len(noeuds) == 0:
        noeuds = mesh.nodes
    
    coordo = mesh.coordoGlob

    if mesh.dim == 2:
        ax.scatter(coordo[noeuds,0], coordo[noeuds,1], marker=marker, c=c)
        if showId:            
            for n, noeud in enumerate(noeuds): ax.text(coordo[n,0], coordo[n,1], str(noeud))
    elif  mesh.dim == 3:            
        ax.scatter(coordo[noeuds,0], coordo[noeuds,1], coordo[noeuds,2], marker=marker, c=c)
        if showId:
            for n, noeud in enumerate(noeuds): ax.text(coordo[n,0], coordo[n,1], coordo[n,2], str(noeud))
    
    if folder != "":
        import PostTraitement
        PostTraitement.Save_fig(folder, "noeuds")

    return ax

# TODO Affichage conditions limites
    
def __GetCoordo(simu, deformation: bool, facteurDef: float):
    
    from Simu import Simu

    simu = cast(Simu, simu)       

    coordo = simu.mesh.coordo

    if deformation:

        coordoDef = simu.GetCoordUglob()

        test = isinstance(coordoDef, np.ndarray)

        if test:
            coordo = coordo + coordoDef * facteurDef

        return coordo, test
    else:
        return coordo, deformation

def __ChangeEchelle(ax, coordo: np.ndarray):
    """Change la taille des axes pour l'affichage 3D

    Parameters
    ----------
    ax : plt.Axes
        Axes dans lequel on va creer la figure
    """
    # Change la taille des axes
    xmin = np.min(coordo[:,0]); xmax = np.max(coordo[:,0])
    ymin = np.min(coordo[:,1]); ymax = np.max(coordo[:,1])
    zmin = np.min(coordo[:,2]); zmax = np.max(coordo[:,2])
    
    max = np.max(np.abs([xmin, xmax, ymin, ymax, zmin, zmax]))
    
    factX = np.max(np.abs([xmin, xmax]))/max
    factY = np.max(np.abs([ymin, ymax]))/max
    factZ = np.max(np.abs([zmin, zmax]))/max
    
    ecartAuBord = 40

    ax.set_xlim([xmin-ecartAuBord, xmax+ecartAuBord])
    ax.set_ylim([ymin-ecartAuBord, ymax+ecartAuBord])
    ax.set_zlim([zmin-ecartAuBord, zmax+ecartAuBord])

    # ax.set_box_aspect((factX, factY, factZ))

def NouvelleSection(text: str):
    """Creer une nouvelle section dans la console"""
    # print("\n==========================================================")
    # print("{} :".format(text))
    bord = "======================="
    print("\n\n{} {} {}\n".format(bord,text,bord))

def Clear():
    """Nettoie la console"""
    os.system("cls")

    