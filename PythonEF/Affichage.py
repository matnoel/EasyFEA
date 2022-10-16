
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

    from Simu import Simu, BeamModel
    from TicTac import Tic

    simu = cast(Simu, simu) # ne pas ecrire simu: Simu ça créer un appel circulaire

    mesh = simu.mesh # récupération du maillage
    
    dim = mesh.dim # dimension du maillage

    # Dimension dans lequel se trouve le maillage
    try:
        inDim = mesh.inDim 
    except:
        inDim = mesh.groupElem.inDim

    if dim == 3:
        # Quand on fait une simulation en 3D on ne peut afficher les résultats que sur les elements
        # En plus, il faut tracer la solution que sur les eléments 2D
        valeursAuxNoeuds = False

    if simu.problemType == "beam":
        # Actuellement je ne sais pas comment afficher les résultats nodaux donc j'affiche sur les elements
        valeursAuxNoeuds = False

    # Récupération du résultat
    valeurs = simu.Get_Resultat(option, valeursAuxNoeuds)
    if not isinstance(valeurs, np.ndarray):
        return

    valeurs *= coef

    coordoSansDef = simu.mesh.coordo

    # Recupération des coordonnée déformées si la simulation le permet
    coordo, deformation = __GetCoordo(simu, deformation, facteurDef)

    # construit la matrice de connection pour les faces
    connect_Faces = mesh.connect_Faces

    # Construit les faces non deformées
    coordo_redim = coordo[:,range(inDim)]

    # Construit les niveaux pour la colorbar
    if option == "damage":
        min = valeurs.min()-1e-12
        max = valeurs.max()+1e-12
        if max < 1:
            max=1
        levels = np.linspace(min, max, 200)
    else:
        levels = 200

    is3dBeamModel = False
    beamModel = simu.materiau.beamModel
    if isinstance(beamModel, BeamModel):
        if beamModel.dim == 3:
            is3dBeamModel = True

    if inDim in [1,2] and not is3dBeamModel:
        # Maillage contenu dans un plan 2D

        # dictionnaire pour stocker les coordonnées par faces
        coord_par_face = {}
        for elem in connect_Faces:
            # Récupérations des noeuds par faces
            faces = connect_Faces[elem]
            # recupere la coordonnées des noeuds de chaque elements
            coord_par_face[elem] = coordo_redim[faces]

        if oldax == None:
            # Création de la figure
            fig, ax = plt.subplots()
        else:
            # utilisation de l'ancienne figure
            fig = oldfig
            ax = oldax
            ax.clear()

        for elem in coord_par_face:
            # Pour chaque type d'element du maillage on va tracer la solution

            # coordonnées par element
            vertices = coord_par_face[elem]

            # Trace le maillage
            if affichageMaillage:
                if mesh.dim == 1:
                    # le maillage pour des elements 1D sont des points
                    coordFaces = vertices.reshape(-1,inDim)
                    ax.scatter(coordFaces[:,0], coordFaces[:,1], c='black', lw=0.1, marker='.')
                else:
                    # le maillage pour des elements 2D sont des lignes
                    pc = matplotlib.collections.LineCollection(vertices, edgecolor='black', lw=0.5)
                    ax.add_collection(pc)

            # Valeurs aux element
            if mesh.Ne == len(valeurs):
                # on va afficher le résultat sur les elements
                if mesh.dim == 1:
                    # on va afficher le résulat sur chaque ligne
                    pc = matplotlib.collections.LineCollection(vertices, lw=1.5, cmap='jet')
                else:
                    # on va afficher le résultat sur les faces
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
                # on va afficher le résultat sur les noeuds
                # récupération des triangles de chaque face pour utiliser la fonction trisurf
                connectTri = mesh.connectTriangle
                pc = ax.tricontourf(coordo[:,0], coordo[:,1], connectTri[elem], valeurs, levels, cmap='jet')
                # tripcolor, tricontour, tricontourf
        
        # Changement de la taille des axes
        if mesh.dim > 1:
            ax.axis('equal')
        # # TODO ICI prendre en compte que la solution à tourné
        # __ChangeEchelle(ax, coordoSansDef)
        
        ax.autoscale()
        if simu.problemType in ["thermal"]:
            ax.axis('off')
        
        # procédure pour essayer de rapporcher la colorbar de l'ax
        divider = make_axes_locatable(ax)
        if colorbarIsClose:
            cax = divider.append_axes('right', size='10%', pad=0.1)
            # # cax = divider.add_auto_adjustable_area(use_axes=ax, pad=0.1, adjust_dirs='right')
        else:
            cax=None
        
        # Construction de la colorbar
        if option == "damage":
            ticks = np.linspace(0,1,11)
            cb = plt.colorbar(pc, ax=ax, cax=cax, ticks=ticks)
        else:
            cb = plt.colorbar(pc, ax=ax, cax=cax)
        
        # Renome les axes
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

    
    elif inDim == 3 or is3dBeamModel:

        # Construction de la figure et de l'ax si nécessaire
        if oldax == None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
        else:
            fig = oldfig
            ax = oldax
            ax.clear()

        # initialisation des valeurs max et min de la colorbar 
        
        maxVal = 0
        minVal = 0

        # dimenson du maillage
        dim = mesh.dim
        if dim == 3:
            # Si le maillage est un maillage 3D alors on ne va affichier que les elements 2D du maillage
            # Un maillage 3D peut contenir plusieurs types d'element 2D
            # Par exemple quand PRISM6 -> TRI6 et QUAD8 en meme temps
            # En gros la couche extérieur
            dim = 2

        for groupElemDim in mesh.Get_list_groupElem(dim):
            # Récupération de la liste de noeuds de chaque element
            connectDim = groupElemDim.connect_e
            # Récupération de la coordonnée des noeuds
            coordoDim = groupElemDim.coordoGlob
            # coordonnées des noeuds pour chaque element
            vertices = np.asarray(coordoDim[connectDim]) # (Ne, nPe, 3)
            
            if valeursAuxNoeuds:
                # Si le résultat est stocké aux noeuds on va faire la moyenne des valeurs aux noeuds sur l'element
                valeursNoeudsSurElement = valeurs[connectDim]
                valeursAuxFaces = np.asarray(np.mean(valeursNoeudsSurElement, axis=1))
            else:
                valeursAuxFaces = valeurs

            # mise à jour 
            maxVal = np.max([maxVal, valeursAuxFaces.max()])
            minVal = np.min([minVal, valeursAuxFaces.min()])

            # On affiche le résultat avec ou sans l'affichage du maillage
            if affichageMaillage:
                if dim == 1:
                    pc = Line3DCollection(vertices, edgecolor='black', linewidths=0.5, cmap='jet')
                elif dim == 2:
                    pc = Poly3DCollection(vertices, edgecolor='black', linewidths=0.5, cmap='jet')
            else:
                if dim == 1:
                    pc = Line3DCollection(vertices, cmap='jet')
                if dim == 2:
                    pc = Poly3DCollection(vertices, cmap='jet')

            # On applique les couleurs aux faces
            pc.set_array(valeursAuxFaces)

            # ax.add_collection3d(pc, zs=2, zdir='x')
            ax.add_collection3d(pc)
            # ax.add_collection3d(pc)

        # On pose les limites de la colorbar et on l'affiche
        pc.set_clim(minVal, maxVal)
        cb = fig.colorbar(pc, ax=ax)

        # renome les axes
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")
        
        # Change l'echelle des axes
        __ChangeEchelle(ax, coordo)

    # On prépare le titre
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
    
    # On specifie si les valeurs sont sur les noeuds ou sur les elements
    if valeursAuxNoeuds:
        # loc = "^{n}"
        loc = ""
    else:
        loc = "^{e}"
    
    # si aucun titre n'a été renseigné on utilise le titre construit
    if title == "":
        title = option+loc
        ax.set_title(fr"${title}$")
    else:
        ax.set_title(f"{title}")

    # Si le dossier à été renseigné on sauvegarde la figure
    if folder != "":
        import PostTraitement as PostTraitement
        if filename=="":
            filename=title
        PostTraitement.Save_fig(folder, filename, transparent=False)

    # Renvoie la figure, l'axe et la colorbar
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

    from Simu import Simu, BeamModel
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

    inDim = mesh.groupElem.inDim

    # construit la matrice de connection pour les faces
    connect_Faces = mesh.connect_Faces

    # Construit les faces non deformées
    coord_NonDeforme_redim = coordo[:,range(inDim)]

    coord_par_face = {}

    if deformation:
        coordoDeforme, deformation = __GetCoordo(simu, deformation, facteurDef)
        coordo_Deforme_redim = coordoDeforme[:,range(inDim)]
        coordo_par_face_deforme = {}

    for elemType in connect_Faces:
        faces = connect_Faces[elemType]
        coord_par_face[elemType] = coord_NonDeforme_redim[faces]

        if deformation:
            coordo_par_face_deforme[elemType] = coordo_Deforme_redim[faces]
    
    is3dBeamModel = False
    if isinstance(obj, Simu):
        beamModel = simu.materiau.beamModel
        if isinstance(beamModel, BeamModel):
            if beamModel.dim == 3:
                is3dBeamModel = True
        
    if inDim in [1,2] and not is3dBeamModel:
        
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

            if mesh.dim == 1:
                coordFaces = coordFaces.reshape(-1,inDim)
                ax.scatter(coordFaces[:,0], coordFaces[:,1], c='black', lw=lw, marker='.')
                if deformation:
                    coordDeforme = coordo_par_face_deforme[elemType].reshape(-1, inDim)
                    ax.scatter(coordDeforme[:,0], coordDeforme[:,1], c='red', lw=lw, marker='.')
                        
            
        
        ax.autoscale()
        ax.axis('equal')
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

    # ETUDE 3D    
    elif inDim == 3 or is3dBeamModel:
        
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

        # fig = plt.figure()            
        # ax = fig.add_subplot(projection="3d")

        dim = mesh.dim
        if dim == 3:
            # Si le maillage est un maillage 3D alors on ne va affichier que les elements 2D du maillage
            # En gros la couche extérieur
            dim = 2

        if deformation:
            # Affiche que les elements 1D ou 2D en fonction du type de maillage

            for groupElemDim in mesh.Get_list_groupElem(dim):
                faces = groupElemDim.get_connect_Faces()[groupElemDim.elemType]
                coordDeformeFaces = coordoDeforme[faces]
                coordFaces = groupElemDim.coordoGlob[faces]

                if dim > 1:
                    # Supperpose les deux maillages
                    # Maillage non deformé
                    # ax.scatter(x,y,z, linewidth=0, alpha=0)
                    pcNonDef = Poly3DCollection(coordFaces, edgecolor='black', linewidths=0.5, alpha=0)
                    ax.add_collection3d(pcNonDef)

                    # Maillage deformé
                    pcDef = Poly3DCollection(coordDeformeFaces, edgecolor='red', linewidths=0.5, alpha=0)
                    ax.add_collection3d(pcDef)
                else:
                    # Superpose maillage non deformé et deformé
                    # Maillage non deformés            
                    pc = Line3DCollection(coordFaces, edgecolor='black', lw=lw, antialiaseds=True, zorder=1)
                    ax.add_collection3d(pc)

                    # Maillage deformé                
                    pc = Line3DCollection(coordDeformeFaces, edgecolor='red', lw=lw, antialiaseds=True, zorder=1)
                    ax.add_collection3d(pc)
                    
                    ax.scatter(coordo[:,0], coordo[:,1], coordo[:,2], c='black', lw=lw, marker='.')
                    ax.scatter(coordoDeforme[:,0], coordoDeforme[:,1], coordoDeforme[:,2], c='red', lw=lw, marker='.')

        else:
            # Maillage non deformé
            # Affiche que les elements 1D ou 2D en fonction du type de maillage

            for groupElemDim in mesh.Get_list_groupElem(dim):

                connectDim = groupElemDim.connect_e
                coordoDim = groupElemDim.coordoGlob
                coordFaces = coordoDim[connectDim]

                if dim > 1:
                    pc = Poly3DCollection(coordFaces, facecolors='c', edgecolor='black', linewidths=0.5, alpha=alpha)
                else:
                    pc = Line3DCollection(coordFaces, edgecolor='black', lw=lw, antialiaseds=True, zorder=1)
                    ax.scatter(coordoDim[:,0], coordoDim[:,1], coordoDim[:,2], c='black', lw=lw, marker='.')

                ax.add_collection3d(pc, zs=0, zdir='z')
            
        __ChangeEchelle(ax, coordo)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")
    
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

        if problemType in ["damage","thermal"]:
            marker='o'
        elif problemType in ["displacement","beam"]:
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
            elif len(directions) > 2:
                marker='s'

        if dim in [1,2]:
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

        uglob = simu.GetCoordUglob()
        
        test = isinstance(uglob, np.ndarray)

        if test:
            coordoDef = coordo + uglob * facteurDef

        return coordoDef, test
    else:
        return coordo, deformation

def __ChangeEchelle(ax, coordo: np.ndarray):
    """Change la taille des axes pour l'affichage en 3D\n
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

    if coordo[:,2].max() > 0:
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