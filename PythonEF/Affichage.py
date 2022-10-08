
import platform
from typing import List, cast
import os
import numpy as np
import pandas as pd

import matplotlib.collections
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

def Plot_Result(simu, option: str , deformation=False, facteurDef=4, coef=1, title="", affichageMaillage=False, valeursAuxNoeuds=False, folder="", filename="", colorbarIsClose=False, oldfig=None, oldax=None):
    """Affichage d'un résulat de la simulation

    Parameters
    ----------
    simu : _type_
        _description_
    option : str
        resultat que l'on souhaite utiliser. doit être compris dans Simu.ResultatsCalculables()
    deformation : bool, optional
        affiche la deformation, by default False
    facteurDef : int, optional
        facteur de deformation, by default 4
    coef : int, optional
        coef qui sera appliqué a la solution, by default 1
    title : str, optional
        titre de la figure, by default ""
    affichageMaillage : bool, optional
        affiche le maillage, by default False
    valeursAuxNoeuds : bool, optional
        affiche le resultat aux noeuds sinon l'affiche aux elements, by default False
    folder : str, optional
        dossier de sauvegarde, by default ""
    filename : str, optional
        nom du fichier de sauvegarde, by default ""
    colorbarIsClose : bool, optional
        la color bar est affiché proche de la figure, by default False
    oldfig : _type_, optional
        ancienne figure de matplotlib, by default None
    oldax : _type_, optional
        ancien axe de matplotlib, by default None

    Returns
    -------
    Figure, Axe, colorbar
        fig, ax, cb
    """

    # Detecte si on donne bien ax et fig en meme temps
    assert (oldfig == None) == (oldax == None), "Doit fournir oldax et oldfix ensemble"

    if (not oldfig == None) and (not oldax == None):
        assert isinstance(oldfig, plt.Figure) and isinstance(oldax, plt.Axes)           

    # Va chercher les valeurs 0 a affciher

    from Simu import Simu
    from TicTac import Tic

    simu = cast(Simu, simu) # ne pas ecrire simu: Simu ça créer un appel circulaire

    mesh = simu.mesh
    
    dim = mesh.dim

    if dim==3:
        valeursAuxNoeuds=True

    valeurs = simu.Get_Resultat(option, valeursAuxNoeuds)
    if not isinstance(valeurs, np.ndarray):
        return

    

    valeurs *= coef

    coordo, deformation = __GetCoordo(simu, deformation, facteurDef)

    # construit la matrice de connection pour les faces
    connect_Faces = mesh.connect_Faces

    # Construit les faces non deformées
    coordo_redim = coordo[:,range(dim)]

    if option == "damage":
        min = valeurs.min()
        max = valeurs.max()
        if max < 1:
            max=1
        levels = np.linspace(min, max, 200)
    else:
        levels = 200

    if dim == 2:

        coord_par_face = {}
        for elem in connect_Faces:
            faces = connect_Faces[elem]
            coord_par_face[elem] = coordo_redim[faces]

        connectTri = mesh.connectTriangle
        # Construit les vertices

        if oldax == None:
            fig, ax = plt.subplots()
        else:
            fig = oldfig
            ax = oldax
            ax.clear()
        

        for elem in coord_par_face:
            vertices = coord_par_face[elem]

            # Trace le maillage
            if affichageMaillage:
                pc = matplotlib.collections.LineCollection(vertices, edgecolor='black', lw=0.5)
                ax.add_collection(pc)

            # Valeurs aux element
            if mesh.Ne == len(valeurs):
                pc = matplotlib.collections.PolyCollection(vertices, lw=0.5, cmap='jet')
                pc.set_clim(valeurs.min(), valeurs.max())
                pc.set_array(valeurs)
                ax.add_collection(pc)
                                
                # dx_e = resultats["dx_e"]
                # dy_e = resultats["dy_e"]
                # # x,y=np.meshgrid(dx_e,dy_e)
                # pc = ax.tricontourf(dx_e, dy_e, valeurs, levels ,cmap='jet')            

            # Valeur aux noeuds
            elif mesh.Nn == len(valeurs):                
                pc = ax.tricontourf(coordo[:,0], coordo[:,1], connectTri[elem],
                valeurs, levels, cmap='jet')
                # tripcolor, tricontour, tricontourf
        
        ax.autoscale()
        ax.axis('equal')
        if simu.problemType in ["thermal"]:
            ax.axis('off')
        
        divider = make_axes_locatable(ax)
        if colorbarIsClose:
            cax = divider.append_axes('right', size='10%', pad=0.1)
            # # cax = divider.add_auto_adjustable_area(use_axes=ax, pad=0.1, adjust_dirs='right')
        else:
            cax=None
        
        if option == "damage":
            ticks = np.linspace(0,1,11)
            cb = plt.colorbar(pc, ax=ax, cax=cax, ticks=ticks)
        else:
            cb = plt.colorbar(pc, ax=ax, cax=cax)
        
        # ax.set_xlabel('x [mm]')
        # ax.set_ylabel('y [mm]')

    
    elif mesh.dim == 3:

        if oldax == None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
        else:
            fig = oldfig
            ax = oldax
            ax.clear()

        # Construit les vertices du maillage 3D en recupérant le maillage 2D
        maxVal = 0
        minVal = 0

        for groupElem2D in mesh.Get_list_groupElem(2):
            connect2D = groupElem2D.connect_e
            coordo2D = groupElem2D.coordoGlob
            vertices = np.asarray(coordo2D[connect2D]) # (Ne, nPe, 3)

            valeursAuxFaces = np.asarray(np.mean(valeurs[connect2D], axis=1))

            maxVal = np.max([maxVal, valeursAuxFaces.max()])
            minVal = np.min([minVal, valeursAuxFaces.min()])

            if affichageMaillage:
                pc = Poly3DCollection(vertices, edgecolor='black', linewidths=0.5, cmap='jet')
            else:
                pc = Poly3DCollection(vertices, cmap='jet')

            pc.set_array(valeursAuxFaces)

            ax.add_collection3d(pc, zs=2, zdir='x')
            # ax.add_collection3d(pc)
        
        pc.set_clim(minVal, maxVal)
        
        cb = fig.colorbar(pc, ax=ax)
        # ax.set_xlabel("x [mm]")
        # ax.set_ylabel("y [mm]")
        # ax.set_zlabel("z [mm]")            
            
        __ChangeEchelle(ax, coordo)

    if option == "damage":
        option = "\phi"
    elif option == "thermal":
        option = "T"
    elif "S" in option:
        optionFin = option.split('S')[-1]
        option = f"\sigma_{'{'+optionFin+'}'}"
    elif "E" in option:
        optionFin = option.split('E')[-1]
        option = f"\epsilon_{'{'+optionFin+'}'}"
    
    if valeursAuxNoeuds:
        loc = "^{n}"
    else:
        loc = "^{e}"

    if title == "":
        title = option+loc
    ax.set_title(fr"${title}$")

    if folder != "":
        import PostTraitement as PostTraitement
        if filename=="":
            filename=title
        PostTraitement.Save_fig(folder, filename, transparent=False)

    return fig, ax, cb
    
def Plot_Maillage(obj, ax=None, facteurDef=4, deformation=False, lw=0.5 ,alpha=1, folder="", title="") -> plt.Axes:
    """Dessine le maillage de la simulation

    Parameters
    ----------
    obj : Simu or Mesh
        objet qui contient le maillage
    ax : plt.Axes, optional
        Axes dans lequel on va creer la figure, by default None
    facteurDef : int, optional
        facteur de deformation, by default 4
    deformation : bool, optional
        affiche la deformation, by default False
    lw : float, optional
        epaisseur des traits, by default 0.5
    alpha : int, optional
        transparence des faces, by default 1
    folder : str, optional
        dossier de sauvegarde, by default ""
    filename : str, optional
        nom du fichier de sauvegarde, by default ""

    Returns
    -------
    plt.Axes
        Axes dans lequel on va creer la figure
    """

    from Simu import Simu
    from Mesh import Mesh

    typeobj = type(obj).__name__

    if typeobj == Simu.__name__:
        simu = cast(Simu, obj)
        mesh = simu.mesh
    elif typeobj == Mesh.__name__:
        mesh = cast(Mesh, obj)
        if deformation == True:
            print("Il faut donner la simulation pour afficher le maillage déformée")
    else:
        raise "Erreur"
    
    assert facteurDef > 1, "Le facteur de deformation doit être >= 1"

    coordo = mesh.coordoGlob

    dim = mesh.dim

    # construit la matrice de connection pour les faces
    connect_Faces = mesh.connect_Faces

    # Construit les faces non deformées
    coord_NonDeforme_redim = coordo[:,range(dim)]

    coord_par_face = {}

    if deformation:
        coordoDeforme, deformation = __GetCoordo(simu, deformation, facteurDef)
        coordo_Deforme_redim = coordoDeforme[:,range(dim)]
        coordo_par_face_deforme = {}

    for elemType in connect_Faces:
        faces = connect_Faces[elemType]
        coord_par_face[elemType] = coord_NonDeforme_redim[faces]

        if deformation:
            coordo_par_face_deforme[elemType] = coordo_Deforme_redim[faces]

    # ETUDE 2D
    if dim == 2:
        
        if ax == None:
            fig, ax = plt.subplots()

        for elemType in coord_par_face:
            coordFaces = coord_par_face[elemType]

            if deformation:
                coordDeforme = coordo_par_face_deforme[elemType]

                # Superpose maillage non deformé et deformé
                # Maillage non deformés            
                pc = matplotlib.collections.LineCollection(coordFaces, edgecolor='black', lw=lw, antialiaseds=True, zorder=1)
                ax.add_collection(pc)

                # Maillage deformé                
                pc = matplotlib.collections.LineCollection(coordDeforme, edgecolor='red', lw=lw, antialiaseds=True, zorder=1)
                ax.add_collection(pc)
            else:
                # Maillage non deformé
                if alpha == 0:
                    pc = matplotlib.collections.LineCollection(coordFaces, edgecolor='black', lw=lw)
                else:
                    pc = matplotlib.collections.PolyCollection(coordFaces, facecolors='c', edgecolor='black', lw=lw)
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
            # Affiche que les elements 2D

            for groupElem2D in mesh.Get_list_groupElem(2):
                faces = groupElem2D.get_connect_Faces()[groupElem2D.elemType]
                coordDeformeFaces = coordoDeforme[faces]
                coordFaces = groupElem2D.coordoGlob[faces]

                # Supperpose les deux maillages
                # Maillage non deformé
                # ax.scatter(x,y,z, linewidth=0, alpha=0)
                pcNonDef = Poly3DCollection(coordFaces, edgecolor='black', linewidths=0.5, alpha=0)
                ax.add_collection3d(pcNonDef)

                # Maillage deformé
                pcDef = Poly3DCollection(coordDeformeFaces, edgecolor='red', linewidths=0.5, alpha=0)
                ax.add_collection3d(pcDef)
                
        else:
            # Maillage non deformé

            # Si il n'y a pas de deformation on peut afficher que le maillage 2D

            for groupElem2D in mesh.Get_list_groupElem(2):

                connect2D = groupElem2D.connect_e
                coordo2D = groupElem2D.coordoGlob
                coordFaces = coordo2D[connect2D]

                pc = Poly3DCollection(coordFaces, facecolors='c', edgecolor='black', linewidths=0.5, alpha=alpha)

                ax.add_collection3d(pc, zs=0, zdir='z')

            
        __ChangeEchelle(ax, coordo)
        # ax.autoscale()
        # ax.set_xlabel("x [mm]")
        # ax.set_ylabel("y [mm]")
        # ax.set_zlabel("z [mm]")

        
    
    if title == "":
        title = f"{mesh.elemType} : Ne = {mesh.Ne} et Nn = {mesh.Nn}"

    ax.set_title(title)

    if folder != "":
        import PostTraitement as PostTraitement
        PostTraitement.Save_fig(folder, title)

    return ax

def Plot_NoeudsMaillage(mesh, ax=None, noeuds=[], showId=False, marker='.', c='red', folder=""):
    """Affiche les noeuds du maillage

    Parameters
    ----------
    mesh : Mesh
        maillage
    ax : plt.Axes, optional
        Axes dans lequel on va creer la figure, by default None
    noeuds : list[np.ndarray], optional
        noeuds à afficher, by default []
    showId : bool, optional
        affiche les numéros, by default False
    marker : str, optional
        type de marker (matplotlib.markers), by default '.'
    c : str, optional
        couleur du maillage, by default 'blue'
    folder : str, optional
        dossier de sauvegarde, by default ""    

    Returns
    -------
    plt.Axes
        Axes dans lequel on va creer la figure
    """
    
    from Mesh import Mesh
    mesh = cast(Mesh, mesh)

    if ax == None:
        ax = Plot_Maillage(mesh, alpha=0)
    
    if len(noeuds) == 0:
        noeuds = mesh.nodes    
    
    coordo = mesh.coordoGlob

    if mesh.dim == 2:
        ax.scatter(coordo[noeuds,0], coordo[noeuds,1], marker=marker, c=c, zorder=2.5)
        if showId:            
            for noeud in noeuds: ax.text(coordo[noeud,0], coordo[noeud,1], str(noeud))
    elif  mesh.dim == 3:            
        ax.scatter(coordo[noeuds,0], coordo[noeuds,1], coordo[noeuds,2], marker=marker, c=c)
        if showId:
            for noeud in noeuds: ax.text(coordo[noeud,0], coordo[noeud,1], coordo[noeud,2], str(noeud))
    
    if folder != "":
        import PostTraitement as PostTraitement
        PostTraitement.Save_fig(folder, "noeuds")

    return ax

def Plot_ElementsMaillage(mesh, ax=None, dimElem =None, noeuds=[], showId=False, c='red', folder=""):
    """Affiche les elements du maillage en fonction des numéros de noeuds

    Parameters
    ----------
    mesh : Mesh
        maillage
    ax : plt.Axes, optional
        Axes dans lequel on va creer la figure, by default None
    dimElem : int, optional
        dimension de l'element recherché, by default None
    noeuds : list, optional
        numeros des noeuds, by default []
    showId : bool, optional
        affiche les numéros, by default False    
    c : str, optional
        couleur utilisé pour afficher les elements, by default 'red'
    folder : str, optional
        dossier de sauvegarde, by default ""

    Returns
    -------
    plt.Axes
        Axes dans lequel on va creer la figure
    """

    from Mesh import Mesh
    mesh = cast(Mesh, mesh)

    if dimElem == None:
        dimElem = mesh.dim

    list_groupElem = mesh.Get_list_groupElem(dimElem)
    if len(list_groupElem) == 0: return

    if ax == None:
        ax = Plot_Maillage(mesh, alpha=1)

    for groupElemDim in list_groupElem:
        
        elemType = groupElemDim.elemType

        if len(noeuds) > 0:
            elements = groupElemDim.get_elementsIndex(noeuds)
        else:
            elements = np.arange(groupElemDim.Ne)

        if elements.size == 0: continue

        elementsID = groupElemDim.elementsID

        connect_e = groupElemDim.connect_e
        coordo_n = groupElemDim.coordoGlob
        coordoFaces_e = coordo_n[connect_e]
        coordoFaces = coordoFaces_e[elements]

        coordo_e = np.mean(coordoFaces_e, axis=1)
        
        if mesh.dim in [1,2]:
            if len(noeuds) > 0:
                if groupElemDim.dim == 1:
                    pc = matplotlib.collections.LineCollection(coordoFaces[:,:,range(mesh.dim)], edgecolor=c, lw=1, zorder=24)
                else:
                    pc = matplotlib.collections.PolyCollection(coordoFaces[:,:,range(mesh.dim)], facecolors=c, edgecolor='black', lw=0.5, alpha=1)
                ax.add_collection(pc)

            # ax.scatter(coordo[:,0], coordo[:,1], marker=marker, c=c, zorder=24)
            if showId:            
                for element in elements:
                    ax.text(coordo_e[element,0], coordo_e[element,1], str(elementsID[element]),
                    zorder=25, ha='center', va='center')
        elif  mesh.dim == 3:
            if len(noeuds) > 0:
                ax.add_collection3d(Poly3DCollection(coordoFaces, facecolors='red', edgecolor='black', linewidths=0.5, alpha=1))

            # ax.scatter(coordo[:,0], coordo[:,1], coordo[:,2], marker=marker, c=c, zorder=24)
            if showId:
                for element in elements:
                    ax.text(coordo_e[element,0], coordo_e[element,1], coordo_e[element,2], str(elementsID[element]),
                    zorder=25, ha='center', va='center')

    # ax.axis('off')
    
    if folder != "":
        import PostTraitement as PostTraitement 
        PostTraitement.Save_fig(folder, "noeuds")

    return ax


def Plot_BoundaryConditions(simu, folder=""):
    """Affichage des conditions limites

    Parameters
    ----------
    simu : Simu
        simulation
    folder : str, optional
        dossier de sauvegarde, by default ""

    Returns
    -------
    plt.Axes
        Axes dans lequel on va creer la figure
    """

    from Simu import Simu
    from BoundaryCondition import BoundaryCondition

    simu = cast(Simu, simu)

    dim = simu.dim

    dirchlets = simu.Get_Bc_Dirichlet()
    neumanns = simu.Get_Bc_Neuman()

    dirchlets.extend(neumanns)
    Conditions = dirchlets

    ax = Plot_Maillage(simu, alpha=0)

    assert isinstance(ax, plt.Axes)

    coordo = simu.mesh.coordoGlob

    Conditions = cast(List[BoundaryCondition], Conditions)

    for bc_Conditions in Conditions:

        problemType = bc_Conditions.problemType
        ddls = bc_Conditions.ddls
        valeurs_ddls = bc_Conditions.valeurs_ddls
        directions = bc_Conditions.directions
        
        # récupère les noeuds
        noeuds = bc_Conditions.noeuds

        if problemType == "damage":
            valeurs = [valeurs_ddls[0]]
        else:
            valeurs = np.round(list(np.sum(valeurs_ddls.copy().reshape(-1, len(directions)), axis=0)), 2)
        description = bc_Conditions.description

        titre = f"{description} {valeurs} {directions}"

        # if 'Neumann' in description and problemType == 'displacement':
        #     facteur = coordo.max()/50
        #     for n in noeuds:            
        #         dx=facteur
        #         dy=facteur
        #         ax.arrow(coordo[n,0], coordo[n,1],dx, dy, head_width = 0.05, head_length = 0.1, label=titre)
        #     continue
        
        # filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')

        if problemType == "damage":
            marker='o'
        elif problemType == "displacement":
            if len(directions) == 1:
                signe = np.sign(valeurs[0])
                if directions[0] == 'x':
                    if signe == -1:
                        marker='<'
                    else:
                        marker='>'
                elif directions[0] == 'y':
                    if signe == -1:
                        marker='v'
                    else:
                        marker='^'
                elif directions[0] == 'z':
                    marker='d'
            elif len(directions) == 2:
                marker='X'                
            elif len(directions) == 3:
                marker='s'

        if dim == 2:
            lw=0
            ax.scatter(coordo[noeuds,0], coordo[noeuds,1], marker=marker, linewidths=lw, label=titre, zorder=2.5)
        else:
            lw=3
            ax.scatter(coordo[noeuds,0], coordo[noeuds,1], coordo[noeuds,2], marker=marker, linewidths=lw, label=titre)
    
    plt.legend()

    if folder != "":
        import PostTraitement as PostTraitement 
        PostTraitement.Save_fig(folder, "Conditions limites")

    return ax

def Plot_ForceDep(deplacements: np.ndarray, forces: np.ndarray, xlabel='ud en m', ylabel='f en N', folder=""):
    fig, ax = plt.subplots()

    ax.plot(np.abs(deplacements), np.abs(forces), c='blue')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()

    if folder != "":
        import PostTraitement as PostTraitement 
        PostTraitement.Save_fig(folder, "forcedep")

def Plot_ResumeIter(simu, folder: str, iterMin=None, iterMax=None):
    from Simu import Simu

    assert isinstance(simu, Simu)

    # Recupère les résultats de simulation
    resultats = simu.Get_All_Results()
    df = pd.DataFrame(resultats)

    iterations = np.arange(df.shape[0])

    if iterMax == None:
        iterMax = iterations.max()

    if iterMin == None:
        iterMin = iterations.min()
    
    selectionIndex = list(filter(lambda iterations: iterations >= iterMin and iterations <= iterMax, iterations))

    iterations = iterations[selectionIndex]

    damageMaxIter = np.max(list(df["damage"].values), axis=1)[selectionIndex]
    try:
        tempsIter = df["tempsIter"].values[selectionIndex]
        nombreIter = df["nombreIter"].values[selectionIndex]
        getTempsAndNombreIter = True
    except:
        # tempsIter et nombreIter n'ont pas été sauvegardé
        getTempsAndNombreIter = False
    
    try:
        tolConvergence = df["dincMax"].values[selectionIndex]
        getTolConvergence = True
    except:
        getTolConvergence = False

    if getTempsAndNombreIter:

        if getTolConvergence:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True)
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)

        # On affiche le nombre d'itérations de convergence en fonction de l'endommagement
        ax1.grid()
        ax1.plot(iterations, damageMaxIter, color='blue')
        ax1.set_ylabel(r"$\phi$", rotation=0)
        
        ax2.grid()
        ax2.plot(iterations, nombreIter, color='blue')
        # ax2.set_yscale('log', base=10)
        ax2.set_ylabel("iteration")

        ax3.grid()
        ax3.plot(iterations, tempsIter, color='blue')
        ax3.set_ylabel("temps")

        if getTolConvergence:
            ax4.grid()
            ax4.plot(iterations, tolConvergence, color='blue')
            ax4.set_ylabel("tolerance")
        
    else:
        # On affiche l'endommagement max pour chaque itération
        fig, ax = plt.subplots()
        ax.plot(iterations, damageMaxIter, color='blue')
        ax.set_xlabel("iterations")
        ax.set_ylabel(r"$\phi$", rotation=0)
        ax.grid()

    if folder != "":
        import PostTraitement as PostTraitement 
        PostTraitement.Save_fig(folder, "resumeConvergence")

        
def __GetCoordo(simu, deformation: bool, facteurDef: float):
    """Recupération des coordonnée déformées si la simulation le permet

    Parameters
    ----------
    simu : Simu
        simulation
    deformation : bool
        calcul de la deformation
    facteurDef : float
        facteur de deformation

    Returns
    -------
    np.ndarray
        coordonnées du maillage globle déformé
    """
    
    from Simu import Simu

    simu = cast(Simu, simu)

    coordo = simu.mesh.coordoGlob

    if deformation:

        coordoDef = simu.GetCoordUglob()

        test = isinstance(coordoDef, np.ndarray)

        if test:
            coordo = coordo + coordoDef * facteurDef

        return coordo, test
    else:
        return coordo, deformation

def __ChangeEchelle(ax, coordo: np.ndarray):
    """Change la taille pour l'affichage en 3D\n
    Va centrer la pièce et faire en sorte que les axes soient de la bonne taille

    Parameters
    ----------
    ax : plt.Axes
        Axes dans lequel on va creer la figure
    coordo : np.ndarray
        coordonnées du maillage
    """

    # Change la taille des axes
    xmin = np.min(coordo[:,0]); xmax = np.max(coordo[:,0])
    ymin = np.min(coordo[:,1]); ymax = np.max(coordo[:,1])
    zmin = np.min(coordo[:,2]); zmax = np.max(coordo[:,2])
    
    max = np.max(np.abs([xmin, xmax, ymin, ymax, zmin, zmax]))
    min = np.min(np.abs([xmin, xmax, ymin, ymax, zmin, zmax]))
    maxRange = np.max(np.abs([xmin - xmax, ymin - ymax, zmin - zmax]))
    cc = 0.5 # -> zoom au mieu
    cc= 1 # dezoomé de 2
    maxRange = maxRange*0.5

    xmid = (xmax + xmin)/2
    ymid = (ymax + ymin)/2
    zmid = (zmax + zmin)/2

    ax.set_xlim([xmid-maxRange, xmid+maxRange])
    ax.set_ylim([ymid-maxRange, ymid+maxRange])
    ax.set_zlim([zmid-maxRange, zmid+maxRange])

    ax.set_box_aspect([1,1,1])
    

def NouvelleSection(text: str, verbosity=True):
    """Creation d'une nouvelle section

    Parameters
    ----------
    text : str
        titre de la section
    """
    # print("\n==========================================================")
    # print("{} :".format(text))
    bord = "======================="

    longeurTexte = len(text)

    longeurMax = 45

    bord = "="*int((longeurMax - longeurTexte)/2)

    section = f"\n\n{bord} {text} {bord}\n"

    if verbosity: print(section)

    return section

def Clear():
    """Nettoie le terminal de commande"""
    syst = platform.system()
    if syst in ["Linux","Darwin"]:
        os.system("clear")
    elif syst == "Windows":
        os.system("cls")