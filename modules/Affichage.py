import platform
from typing import List, cast
from colorama import Fore
import os
import numpy as np
import pandas as pd

# Figures
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Pour tracer des collections
import matplotlib.collections
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import Folder

def Plot_Result(obj, option: str|np.ndarray, deformation=False, facteurDef=4, coef=1, plotMesh=False, nodeValues=True, folder="", filename="", title="", ax=None, colorbarIsClose=False, cmap="jet"):
    """Affichage d'un résulat de la simulation

    Parameters
    ----------
    obj : _Simu or Mesh
        objet qui contient le maillage
    option : str
        resultat que l'on souhaite utiliser. doit être compris dans Simu.ResultatsCalculables()
    deformation : bool, optional
        affiche la deformation, by default False
    facteurDef : int, optional
        facteur de deformation, by default 4
    coef : int, optional
        coef qui sera appliqué a la solution, by default 1
    plotMesh : bool, optional
        affiche le maillage, by default False
    nodeValues : bool, optional
        affiche le resultat aux noeuds sinon l'affiche aux elements, by default True
    folder : str, optional
        dossier de sauvegarde, by default ""
    filename : str, optional
        nom du fichier de sauvegarde, by default ""
    title : str, optional
        titre de la figure, by default ""
    ax : axe, optional
        ancien axe de matplotlib, by default None
    colorbarIsClose : bool, optional
        la color bar est affiché proche de la figure, by default False
    cmap : str, optional
        la color map utilisée proche de la figure, by default "jet" \n
        \t ["jet", "RdBu", "seismic", "binary"] -> https://matplotlib.org/stable/tutorials/colors/colormaps.html

    Returns
    -------
    Figure, Axe, colorbar
        fig, ax, cb
    """

    from Simulations import Simu, Mesh, MatriceType

    # ici on detecte la nature de l'objet
    if isinstance(obj, Simu):
        simu = obj
        mesh = simu.mesh
        use3DBeamModel = simu.use3DBeamModel

        if simu.problemType == MatriceType.beam:
            # Actuellement je ne sais pas comment afficher les résultats nodaux donc j'affiche sur les elements
            nodeValues = False

    elif isinstance(obj, Mesh):
        mesh = obj

        if deformation == True:
            deformation = False
            print("Il faut donner la simulation pour afficher le maillage déformée")
        use3DBeamModel = False

        if isinstance(option, str):
            raise Exception("Quand obj est un maillage il faut que option soit une array de dimension Nn ou Ne")
        
    else:
        raise Exception("Doit être une simulation ou un maillage")    
    
    if ax != None:
        assert isinstance(ax, plt.Axes)
        fig = ax.figure
    
    dim = mesh.dim # dimension du maillage    
    inDim = mesh.inDim # dimension dans lequel se trouve le maillage    

    # Construction de la figure et de l'axe si nécessaire
    if ax == None:
        if inDim in [1,2] and not use3DBeamModel:
            fig, ax = plt.subplots()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
    else:
        fig = fig
        ax = ax
        ax.clear()

    if dim == 3:
        nodeValues = True # Ne pas modifier, il faut passer par la solution aux noeuds pour localiser aux elements 2D !!!
        # Quand on fait une simulation en 3D on affiche les résultats que sur les elements 2D
        # Pour prendre moin de place
        # En plus, il faut tracer la solution que sur les eléments 2D

    if isinstance(option, str):
        valeurs = simu.Get_Resultat(option, nodeValues) # Récupération du résultat
        if not isinstance(valeurs, np.ndarray): return

    elif isinstance(option, np.ndarray):
        # Recupère la taille de l'array, la taille doit être aux noeuds ou aux elements
        # Si la taille n'est pas egale au nombre de noeuds ou d'elements renvoie une erreur
        sizeVecteur = option.size

        if sizeVecteur not in [mesh.Ne, mesh.Nn]:
            print("Le vecteur renseigné doit être de dimension Nn ou Ne")
            return

        valeurs = option
        
        if sizeVecteur == mesh.Ne and nodeValues:
            valeurs = Simu.Resultats_InterpolationAuxNoeuds(mesh, valeurs)
        elif sizeVecteur == mesh.Nn and not nodeValues:
            valeursLoc_e = mesh.Localises_sol_e(valeurs)
            valeurs = np.mean(valeursLoc_e, 1)
    else:
        raise Exception("Dois renseigner une chaine de caractère ou une array")
    
    valeurs *= coef # Application d'un coef sur les valeurs

    coordoNonDef = mesh.coordoGlob # coordonnées des noeuds sans déformations

    if deformation:
        # Recupération des coordonnée déformées si la simulation le permet
        coordoDef, deformation = __GetCoordo(simu, deformation, facteurDef)
    else:
        coordoDef = coordoNonDef.copy()
    
    coordoDef_InDim = coordoDef[:,range(inDim)]
    
    connect_Faces = mesh.dict_connect_Faces # construit la matrice de connection pour les faces    

    # Construit les bornes pour la colorbar
    if isinstance(option, str) and option == "damage":
        min = valeurs.min()-1e-12
        openCrack = False
        if openCrack:
            max = 0.98
        else:
            max = valeurs.max()+1e-12
            if max < 1:
                max = 1        
    else:
        max = np.max(valeurs)+1e-12
        min = np.min(valeurs)-1e-12

    levels = np.linspace(min, max, 200)

    if inDim in [1,2] and not use3DBeamModel:
        # Maillage contenu dans un plan 2D

        # dictionnaire pour stocker les coordonnées par faces
        dict_coordoFaceElem = {}
        for elem in connect_Faces:
            faces = connect_Faces[elem] # Récupérations des noeuds par faces
            # recupere la coordonnée des noeuds de chaque elements
            dict_coordoFaceElem[elem] = coordoDef_InDim[faces]

        for elem in dict_coordoFaceElem:
            # Pour chaque type d'element du maillage on va tracer la solution

            # coordonnées par element
            vertices = dict_coordoFaceElem[elem]

            # Trace le maillage
            if plotMesh:
                if mesh.dim == 1:
                    # le maillage pour des elements 1D sont des points
                    coordFaces = vertices.reshape(-1,inDim)
                    ax.scatter(coordFaces[:,0], coordFaces[:,1], c='black', lw=0.1, marker='.')
                else:
                    # le maillage pour des elements 2D sont des lignes
                    pc = matplotlib.collections.LineCollection(vertices, edgecolor='black', lw=0.5)
                    ax.add_collection(pc)

            # Valeurs aux elements
            if mesh.Ne == len(valeurs):
                # on va afficher le résultat sur les elements
                if mesh.dim == 1:
                    # on va afficher le résulat sur chaque ligne
                    pc = matplotlib.collections.LineCollection(vertices, lw=1.5, cmap=cmap)
                else:
                    # on va afficher le résultat sur les faces
                    pc = matplotlib.collections.PolyCollection(vertices, lw=0.5, cmap=cmap)                
                pc.set_clim(min, max)
                pc.set_array(valeurs)
                ax.add_collection(pc)

            # Valeur aux noeuds
            elif mesh.Nn == len(valeurs):
                # on va afficher le résultat sur les noeuds
                # récupération des triangles de chaque face pour utiliser la fonction trisurf
                connectTri = mesh.dict_connect_Triangle

                pc = ax.tricontourf(coordoDef[:,0], coordoDef[:,1], connectTri[elem], valeurs, levels, cmap=cmap, vmin=min, vmax=max)
                # tripcolor, tricontour, tricontourf

        ax.autoscale()
        epX = np.abs(coordoDef[:,0].max() - coordoDef[:,0].min())
        epY = np.abs(coordoDef[:,1].max() - coordoDef[:,1].min())
        if (epX > 0 and epY > 0):
            if np.abs(epX-epY)/epX > 0.2:
                ax.axis('equal')
        
        # procédure pour essayer de rapporcher la colorbar de l'ax
        divider = make_axes_locatable(ax)
        if colorbarIsClose:
            cax = divider.append_axes('right', size='10%', pad=0.1)
            # # cax = divider.add_auto_adjustable_area(use_axes=ax, pad=0.1, adjust_dirs='right')
        else:
            cax=None
        
        # Construction les ticks pour la colorbar
        if isinstance(option, str) and option == "damage":
            ticks = np.linspace(0,1,11)
        else:
            ticks = np.linspace(min,max,11)
        cb = plt.colorbar(pc, ax=ax, cax=cax, ticks=ticks)
        
        # Renome les axes
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

    
    elif inDim == 3 or use3DBeamModel:
        # initialisation des valeurs max et min de la colorbar         
        
        maxVal = max
        minVal = min

        # dimenson du maillage
        dim = mesh.dim
        if dim == 3:
            # Si le maillage est un maillage 3D alors on ne va affichier que les elements 2D du maillage
            # Un maillage 3D peut contenir plusieurs types d'element 2D
            # Par exemple quand PRISM6 -> TRI6 et QUAD8 en meme temps
            # En gros la couche extérieur
            dim = 2

        for groupElemDim in mesh.Get_list_groupElem(dim):
            indexeFaces = groupElemDim.indexesFaces
            # Récupération de la liste de noeuds de chaque element
            connectDim = groupElemDim.connect
            # Récupération de la coordonnée des noeuds
            coordoDim = groupElemDim.coordoGlob
            # coordonnées des noeuds pour chaque element
            vertices = np.asarray(coordoDim[connectDim[:,indexeFaces]]) # (Ne, nPe, 3)
            
            if nodeValues:
                # Si le résultat est stocké aux noeuds on va faire la moyenne des valeurs aux noeuds sur l'element
                valeursNoeudsSurElement = valeurs[connectDim]
                valeursAuxFaces = np.asarray(np.mean(valeursNoeudsSurElement, axis=1))
            else:
                valeursAuxFaces = valeurs

            # mise à jour 
            maxVal = np.max([maxVal, valeursAuxFaces.max()])
            minVal = np.min([minVal, valeursAuxFaces.min()])

            # On affiche le résultat avec ou sans l'affichage du maillage
            if plotMesh:
                if dim == 1:
                    pc = Line3DCollection(vertices, edgecolor='black', linewidths=0.5, cmap=cmap, zorder=0)
                elif dim == 2:
                    pc = Poly3DCollection(vertices, edgecolor='black', linewidths=0.5, cmap=cmap, zorder=0)
            else:
                if dim == 1:
                    pc = Line3DCollection(vertices, cmap=cmap, zorder=0)
                if dim == 2:
                    pc = Poly3DCollection(vertices, cmap=cmap, zorder=0)

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
        __ChangeEchelle(ax, coordoNonDef)

    # On prépare le titre
    if isinstance(option, str):
        if option == "damage":
            option = "\phi"
        elif option == "thermal":
            option = "T"
        elif "S" in option and not option in ["amplitudeSpeed"]:
            optionFin = option.split('S')[-1]
            option = f"\sigma_{'{'+optionFin+'}'}"
        elif "E" in option:
            optionFin = option.split('E')[-1]
            option = f"\epsilon_{'{'+optionFin+'}'}"
    
    # On specifie si les valeurs sont sur les noeuds ou sur les elements
    if nodeValues:
        # loc = "^{n}"
        loc = ""
    else:
        loc = "^{e}"
    
    # si aucun titre n'a été renseigné on utilise le titre construit
    if title == "" and isinstance(option, str):
        title = option+loc
        ax.set_title(fr"${title}$")
    else:
        ax.set_title(f"{title}")

    # Si le dossier à été renseigné on sauvegarde la figure
    if folder != "":
        if filename=="":
            filename=title
        Save_fig(folder, filename, transparent=False)

    # Renvoie la figure, l'axe et la colorbar
    return fig, ax, cb
    
def Plot_Mesh(obj, deformation=False, facteurDef=4, folder="", title="", ax=None, lw=0.5, alpha=1, facecolors='c', edgecolor='black') -> plt.Axes:
    """Dessine le maillage de la simulation

    Parameters
    ----------
    obj : _Simu or Mesh
        objet qui contient le maillage
    facteurDef : int, optional
        facteur de deformation, by default 4
    deformation : bool, optional
        affiche la deformation, by default False
    folder : str, optional
        dossier de sauvegarde, by default ""
    title : str, optional
        nom du fichier de sauvegarde, by default ""
    ax : plt.Axes, optional
        Axes dans lequel on va creer la figure, by default None
    lw : float, optional
        epaisseur des traits, by default 0.5
    alpha : int, optional
        transparence des faces, by default 1

    Returns
    -------
    plt.Axes
        Axes dans lequel on va creer la figure
    """

    from Simulations import Simu, Mesh

    if isinstance(obj, Simu):
        simu = obj
        mesh = simu.mesh
        use3DBeamModel = simu.use3DBeamModel
    elif isinstance(obj, Mesh):
        mesh = obj
        if deformation == True:
            print("Il faut donner la simulation pour afficher le maillage déformée")
        use3DBeamModel = False
    else:
        raise Exception("Doit être une simulation ou un maillage")
    
    assert facteurDef > 1, "Le facteur de deformation doit être >= 1"

    coordo = mesh.coordoGlob

    inDim = mesh.groupElem.inDim

    # construit la matrice de connection pour les faces
    connect_Faces = mesh.dict_connect_Faces

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
        
    if inDim in [1,2] and not use3DBeamModel:
        
        if ax == None:
            fig, ax = plt.subplots()

        for elemType in coord_par_face:
            coordFaces = coord_par_face[elemType]

            if deformation:
                coordDeforme = coordo_par_face_deforme[elemType]

                # Superpose maillage non deformé et deformé
                # Maillage non deformés            
                pc = matplotlib.collections.LineCollection(coordFaces, edgecolor=edgecolor, lw=lw, antialiaseds=True, zorder=1)
                ax.add_collection(pc)

                # Maillage deformé                
                pc = matplotlib.collections.LineCollection(coordDeforme, edgecolor='red', lw=lw, antialiaseds=True, zorder=1)
                ax.add_collection(pc)

            else:
                # Maillage non deformé
                if alpha == 0:
                    pc = matplotlib.collections.LineCollection(coordFaces, edgecolor=edgecolor, lw=lw, zorder=1)
                else:
                    pc = matplotlib.collections.PolyCollection(coordFaces, facecolors=facecolors, edgecolor=edgecolor, lw=lw, zorder=1)
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
    elif inDim == 3 or use3DBeamModel:
        
        if ax == None:
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
                faces = groupElemDim.Get_dict_connect_Faces()[groupElemDim.elemType]
                coordDeformeFaces = coordoDeforme[faces]
                coordFaces = groupElemDim.coordoGlob[faces]

                if dim > 1:
                    # Supperpose les deux maillages
                    # Maillage non deformé
                    # ax.scatter(x,y,z, linewidth=0, alpha=0)
                    pcNonDef = Poly3DCollection(coordFaces, edgecolor=edgecolor, linewidths=0.5, alpha=0, zorder=0)
                    ax.add_collection3d(pcNonDef)

                    # Maillage deformé
                    pcDef = Poly3DCollection(coordDeformeFaces, edgecolor='red', linewidths=0.5, alpha=0, zorder=0)
                    ax.add_collection3d(pcDef)
                else:
                    # Superpose maillage non deformé et deformé
                    # Maillage non deformés            
                    pc = Line3DCollection(coordFaces, edgecolor=edgecolor, lw=lw, antialiaseds=True, zorder=0)
                    ax.add_collection3d(pc)

                    # Maillage deformé                
                    pc = Line3DCollection(coordDeformeFaces, edgecolor='red', lw=lw, antialiaseds=True, zorder=0)
                    ax.add_collection3d(pc)
                    
                    ax.scatter(coordo[:,0], coordo[:,1], coordo[:,2], c='black', lw=lw, marker='.')
                    ax.scatter(coordoDeforme[:,0], coordoDeforme[:,1], coordoDeforme[:,2], c='red', lw=lw, marker='.')

        else:
            # Maillage non deformé
            # Affiche que les elements 1D ou 2D en fonction du type de maillage

            for groupElemDim in mesh.Get_list_groupElem(dim):
                
                indexesFaces = groupElemDim.indexesFaces
                connectDim = groupElemDim.connect[:, indexesFaces]
                coordoDim = groupElemDim.coordoGlob
                coordFaces = coordoDim[connectDim]

                if dim > 1:
                    pc = Poly3DCollection(coordFaces, facecolors=facecolors, edgecolor=edgecolor, linewidths=0.5, alpha=alpha, zorder=0)
                else:
                    pc = Line3DCollection(coordFaces, edgecolor=edgecolor, lw=lw, antialiaseds=True, zorder=0)
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
        Save_fig(folder, title)

    return ax

def Plot_Nodes(mesh, nodes=[], showId=False, marker='.', c='red', folder="", ax=None):
    """Affiche les noeuds du maillage

    Parameters
    ----------
    mesh : Mesh
        maillage
    ax : plt.Axes, optional
        Axes dans lequel on va creer la figure, by default None
    nodes : list[np.ndarray], optional
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
        ax = Plot_Mesh(mesh, alpha=0)
    
    if len(nodes) == 0:
        nodes = mesh.nodes    
    
    coordo = mesh.coordoGlob

    if mesh.inDim == 2:
        ax.scatter(coordo[nodes,0], coordo[nodes,1], marker=marker, c=c, zorder=2.5)
        if showId:            
            [ax.text(coordo[noeud,0], coordo[noeud,1], str(noeud), c=c) for noeud in nodes]
    elif mesh.inDim == 3:            
        ax.scatter(coordo[nodes,0], coordo[nodes,1], coordo[nodes,2], marker=marker, c=c, zorder=2.5)
        if showId:
            [ax.text(coordo[noeud,0], coordo[noeud,1], coordo[noeud,2], str(noeud), c=c) for noeud in nodes]
    
    if folder != "":
        Save_fig(folder, "noeuds")

    return ax

def Plot_Elements(mesh, nodes=[], dimElem=None, showId=False, c='red', folder="", ax=None):
    """Affiche les elements du maillage en fonction des numéros de noeuds

    Parameters
    ----------
    mesh : Mesh
        maillage
    nodes : list, optional
        numeros des noeuds, by default []
    dimElem : int, optional
        dimension de l'element recherché, by default None
    showId : bool, optional
        affiche les numéros, by default False    
    c : str, optional
        couleur utilisé pour afficher les elements, by default 'red'
    folder : str, optional
        dossier de sauvegarde, by default ""
    ax : plt.Axes, optional
        Axes dans lequel on va creer la figure, by default None

    Returns
    -------
    plt.Axes
        Axes dans lequel on va creer la figure
    """

    from Mesh import Mesh
    mesh = cast(Mesh, mesh)

    if dimElem == None:
        dimElem = mesh.dim-1 if mesh.inDim == 3 else mesh.dim

    list_groupElem = mesh.Get_list_groupElem(dimElem)
    if len(list_groupElem) == 0: return

    if ax == None:
        ax = Plot_Mesh(mesh, alpha=1)

    for groupElemDim in list_groupElem:

        if len(nodes) > 0:
            elements = groupElemDim.Get_Elements_Nodes(nodes)
        else:
            elements = np.arange(groupElemDim.Ne)

        if elements.size == 0: continue

        connect_e = groupElemDim.connect
        coordo_n = groupElemDim.coordoGlob
        indexeFaces = groupElemDim.indexesFaces
        coordoFaces_e = coordo_n[connect_e[:, indexeFaces]]
        coordoFaces = coordoFaces_e[elements]

        coordo_e = np.mean(coordoFaces_e, axis=1)
        
        if mesh.dim in [1,2]:
            if len(nodes) > 0:
                if groupElemDim.dim == 1:
                    pc = matplotlib.collections.LineCollection(coordoFaces[:,:,range(mesh.dim)], edgecolor=c, lw=1, zorder=3)
                else:
                    pc = matplotlib.collections.PolyCollection(coordoFaces[:,:,range(mesh.dim)], facecolors=c, edgecolor='black', lw=0.5, alpha=1, zorder=3)
                ax.add_collection(pc)

            # ax.scatter(coordo[:,0], coordo[:,1], marker=marker, c=c, zorder=3)
            if showId:
                [ax.text(coordo_e[element,0], coordo_e[element,1], element,
                zorder=25, ha='center', va='center') for element in elements]
        elif mesh.dim == 3:
            if len(nodes) > 0:
                ax.add_collection3d(Poly3DCollection(coordoFaces, facecolors=c, edgecolor='black', linewidths=0.5, alpha=1, zorder=3), zdir='z')

            # ax.scatter(coordo[:,0], coordo[:,1], coordo[:,2], marker=marker, c=c, zorder=3)
            if showId:
                [ax.text(coordo_e[element,0], coordo_e[element,1], coordo_e[element,2], element, zorder=25, ha='center', va='center') for element in elements]

    # ax.axis('off')
    
    if folder != "":
        Save_fig(folder, "noeuds")

    return ax

def Plot_BoundaryConditions(simu, folder=""):
    """Affichage des conditions limites

    Parameters
    ----------
    simu : _Simu
        simulation
    folder : str, optional
        dossier de sauvegarde, by default ""

    Returns
    -------
    plt.Axes
        Axes dans lequel on va creer la figure
    """

    from Simulations import Simu

    simu = cast(Simu, simu)

    dim = simu.dim

    # Récupérations des Conditions de chargement de déplacement ou de liaison
    dirchlets = simu.Bc_Dirichlet
    Conditions = dirchlets

    neumanns = simu.Bc_Neuman
    Conditions.extend(neumanns)

    lagranges = simu.Bc_LagrangeAffichage
    Conditions.extend(lagranges)

    ax = Plot_Mesh(simu, alpha=0)

    assert isinstance(ax, plt.Axes)

    coordo = simu.mesh.coordoGlob    

    for bc_Conditions in Conditions:

        problemType = bc_Conditions.problemType        
        valeurs_ddls = bc_Conditions.valeurs_ddls
        directions = bc_Conditions.directions

        # récupère les noeuds
        noeuds = bc_Conditions.noeuds

        if problemType == "damage":
            valeurs = [valeurs_ddls[0]]
        else:
            valeurs = np.round(list(np.sum(valeurs_ddls.copy().reshape(-1, len(directions)), axis=0)), 2)
        
        description = bc_Conditions.description

        titre = f"{description} {directions}"

        if problemType in ["damage","thermal"]:
            marker='o'
        elif problemType in ["displacement","beam"]:
            if len(directions) == 1:
                signe = np.sign(valeurs[0])
                if directions[0] == 'x':
                    if signe == -1:
                        marker = '<'
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
                if "Liaison" in description:
                    marker='o'
                else:
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
        Save_fig(folder, "Conditions limites")

    return ax

def Plot_Model(obj, showId=True,  ax=None, folder="", alpha=1) -> plt.Axes:

    from Simulations import Simu
    from Mesh import Mesh, GroupElem

    typeobj = type(obj).__name__

    if typeobj == Simu.__name__:
        simu = cast(Simu, obj)
        mesh = simu.mesh
    elif typeobj == Mesh.__name__:
        mesh = cast(Mesh, obj)
    else:
        raise Exception("Doit être une simulation ou un maillage")

    inDim = mesh.inDim

    # Création des axes si nécessaire
    if ax == None:
        # ax = Plot_Maillage(mesh, facecolors='c', edgecolor='black')
        # fig = ax.figure
        if mesh.inDim in [0,1,2]:
            fig, ax = plt.subplots()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

    # Ici pour chaque group d'element du maillage, on va tracer les elements appartenant au groupe d'element

    listGroupElem = cast(list[GroupElem], [])
    listDim = np.arange(mesh.dim+1, -1, -1, dtype=int)
    for dim in listDim:
        if dim < 3:
            # On ajoute pas les groupes d'elements 3D
            listGroupElem.extend(mesh.Get_list_groupElem(dim))

    # Liste de collections pendant la création
    collections = []

    for groupElem in listGroupElem:
        
        # Tags disponibles par le groupe d'element
        tags_e = groupElem.elementTags
        dim = groupElem.dim
        coordo = groupElem.coordoGlob[:, range(inDim)]
        faces = groupElem.Get_dict_connect_Faces()[groupElem.elemType]
        coordoFaces = coordo[faces]
        coordo_e = np.mean(coordoFaces, axis=1)

        nColor = 0
        for tag_e in tags_e:

            noeuds = groupElem.Get_Nodes_Tag(tag_e)
            elements = groupElem.Get_Elements_Tag(tag_e)
            if len(elements) == 0: continue

            coordo_faces = coordoFaces[elements]

            needPlot = True
            
            # Attribue la couleur
            if 'L' in tag_e:
                color = 'black'
            elif 'P' in tag_e:
                color = 'black'
            elif 'S' in tag_e:
                nColor = 1
                # nColor += 1
                if nColor > len(__colors):
                    nColor = 1
                color = __colors[nColor]
            elif 'C' in tag_e:
                needPlot = False
            else:
                color = (np.random.random(), np.random.random(), np.random.random())

            x_e = coordo_e[elements,0].mean()
            y_e = coordo_e[elements,1].mean()
            if inDim == 3:
                z_e = coordo_e[elements,2].mean()

            x_n = coordo[noeuds,0]
            y_n = coordo[noeuds,1]
            if inDim == 3:
                z_n = coordo[noeuds,2]

            if inDim in [1,2]:
                
                if len(noeuds) > 0 and needPlot:

                    if dim == 0:
                        collections.append(ax.scatter(x_n, y_n, c='black', marker='.', zorder=2, label=tag_e, lw=2))
                    elif dim == 1:
                        pc = matplotlib.collections.LineCollection(coordo_faces, lw=1.5, edgecolor='black', alpha=1, label=tag_e)
                        collections.append(ax.add_collection(pc))
                    else:
                        pc = matplotlib.collections.PolyCollection(coordo_faces, lw=1, alpha=alpha, facecolors=color, label=tag_e, edgecolor=color)
                        collections.append(ax.add_collection(pc))
                        # ax.legend()
                    
                    if showId and dim != 2:
                        ax.text(x_e, y_e, tag_e, zorder=25)
                else:
                    ax.scatter(x_n, y_n, c='black', marker='.', zorder=2)
                    if showId:
                        ax.text(x_e, y_e, tag_e, zorder=25)
                    
            else:
                if len(noeuds) > 0 and needPlot:
                    if dim == 0:
                        collections.append(ax.scatter(x_n, y_n, z_n, c='black', marker='.', zorder=2, label=tag_e, lw=2, zdir='z'))
                    elif dim == 1:
                        pc = Line3DCollection(coordo_faces, lw=1.5, edgecolor='black', alpha=1, label=tag_e)
                        # collections.append(ax.add_collection3d(pc, zs=z_e, zdir='z'))
                        collections.append(ax.add_collection3d(pc, zdir='z'))
                    elif dim == 2:
                        pc = Poly3DCollection(coordo_faces, lw=0, alpha=alpha, facecolors=color, label=tag_e)
                        pc._facecolors2d = color
                        pc._edgecolors2d = color
                        # collections.append(ax.add_collection3d(pc, zs=z_e, zdir='z'))
                        collections.append(ax.add_collection3d(pc, zdir='z'))

                    if showId:
                        ax.text(x_e, y_e, z_e, tag_e, zorder=25)
                else:
                    x_n = coordo[noeuds,0]
                    y_n = coordo[noeuds,1]
                    y_n = coordo[noeuds,1]
                    if showId:
                        ax.text(x_e, y_e, z_e, tag_e, zorder=25)
                    collections.append(ax.scatter(x_n, y_n, z_n, c='black', marker='.', zorder=2, label=tag_e))

    if inDim in [1, 2]:
        ax.autoscale()
        ax.axis('equal')
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
    else:
        __ChangeEchelle(ax, coordo)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")
    
    if folder != "":
        Save_fig(folder, "noeuds")

    __Annotation_Evenemenent(collections, fig, ax)

    return ax

def __Annotation_Evenemenent(collections: list, fig: plt.Figure, ax: plt.Axes):
    """Création d'un evenement qui va afficher dans le bas de figure le tag actuellement actif sous la souris"""
    
    def Set_Message(collection, event):
        if collection.contains(event)[0]:
            toolbar = ax.figure.canvas.toolbar
            coordo = ax.format_coord(event.xdata, event.ydata)
            toolbar.set_message(f"{collection.get_label()} : {coordo}")
            # TODO Caculer également la surface ou la longeur ?
            # Changer le titre à la place ?
    
    def hover(event):
        if event.inaxes == ax:
            # TODO existe til un moyen d'acceder direct a la collection qui contient levent ?
            [Set_Message(collection, event) for collection in collections]

    fig.canvas.mpl_connect("motion_notify_event", hover)

def Plot_ForceDep(deplacements: np.ndarray, forces: np.ndarray, xlabel='ud [m]', ylabel='f [N]', folder="", ax=None) -> tuple[plt.Figure, plt.Axes]:
    """Trace la courbe force en fonction du déplacement

    Parameters
    ----------
    deplacements : np.ndarray
        array de valeurs pour les déplacements
    forces : np.ndarray
        array de valeurs pour les forces
    xlabel : str, optional
        titre de l'axe x, by default 'ud [m]'
    ylabel : str, optional
        titre de l'axe y, by default 'f [N]'
    folder : str, optional
        path vers le dossier de sauvegarde, by default ""
    ax : plt.Axes, optional
        ax dans lequel on va tracer la figure, by default None

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        renvoie la figure et l'ax
    """

    if isinstance(ax, plt.Axes):
        fig = ax.figure
        ax.clear()
    else:        
        fig, ax = plt.subplots()

    ax.plot(np.abs(deplacements), np.abs(forces), c='blue')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()

    if folder != "":
        Save_fig(folder, "forcedep")

    return fig, ax
    
def Plot_Energie(simu, forces=np.array([]), deplacements=np.array([]), plotSolMax=True, Niter=200, NiterFin=100, folder=""):
    """Trace l'energie de chacune des itérations

    Parameters
    ----------
    simu : _Simu
        simulation
    forces : np.ndarray, optional
        array de valeurs, by default np.array([])
    deplacements : _type_, optional
        _description_, by default np.array([])
    plotSolMax : bool, optional
        affiche l'evolution de la solution maximul au cours des itération. (endommagement max pour une simulation d'endommagement), by default True
    Niter : int, optional
        nombre d'itération pour lesquels on va calculer l'energie, by default 200
    NiterFin : int, optional
        nombre d'itération avant la fin, by default 100
    folder : str, optional
        dossier de sauvagarde de la figure, by default ""
    """

    from Simulations import Simu
    from TicTac import Tic
    import PostTraitement as PostTraitement 

    assert isinstance(simu, Simu)

    # On verfie d'abord si la simulation peut calculer des energies
    if len(simu.Resultats_Get_dict_Energie())== 0:
        print("Cette simulation ne peut calculer les energies")
        return

    # Verifie si il est possible de tracer la courbe force déplacement
    testLoadAndDisplacement = len(forces) == len(deplacements) and len(forces) > 0    
        
    # Pour chaque incrément de déplacement on va caluler l'energie
    tic = Tic()
    
    # récupère les resultats de la simulation
    results =  simu._results
    N = len(results)

    if len(forces) > 0:
        ecart = np.abs(len(results) - len(forces))
        if ecart != 0:
            N -= ecart
    listIter = PostTraitement.Make_listIter(NiterMax=N-1, NiterFin=NiterFin, NiterCyble=Niter)
    
    Niter = len(listIter)

    list_dict_Energie = cast(list[dict[str, float]], [])
    listTemps = []
    if plotSolMax : listSolMax = []

    for i, iteration in enumerate(listIter):

        # Met a jour la simulation à l'iter i
        simu.Update_iter(iteration)

        if plotSolMax : listSolMax.append(simu.get_u_n(simu.problemType).max())

        list_dict_Energie.append(simu.Resultats_Get_dict_Energie())

        temps = tic.Tac("PostTraitement","Calc Energie", False)
        listTemps.append(temps)

        pourcentageEtTempsRestant = PostTraitement._GetPourcentageEtTemps(listIter, listTemps, i)

        print(f"Calc Energie {iteration+1}/{N} {pourcentageEtTempsRestant}    ", end='\r')
    print('\n')

    # Construction de la figure
    nrows = 1
    if plotSolMax:
        nrows += 1
    if testLoadAndDisplacement:
        nrows += 1
    fig, ax = plt.subplots(nrows, 1, sharex=True)

    iter_rows = iter(np.arange(nrows))

    # Récupère l'axe qui sera utilisé pour les abscisses
    if len(deplacements)>0:
        listX = deplacements[listIter] 
        nomX = "deplacement"
    else:
        listX = listIter 
        nomX = "iter"    

    # Transforme list_dict_Energie en dataframe    
    df = pd.DataFrame(list_dict_Energie)

    # Affiche les energies
    row = next(iter_rows)
    # Pour chaque energie on trace les valeurs
    for energie_str in df.columns:
        valeurs = df[energie_str].values
        ax[row].plot(listX, valeurs, label=energie_str)
    ax[row].set_ylabel(r"$Joules$")
    ax[row].legend()
    ax[row].grid()

    if plotSolMax:
        # Affiche l'endommagement max
        row = next(iter_rows)
        ax[row].plot(listX, listSolMax)
        ax[row].set_ylabel(r"$max(u_n)$")
        ax[row].grid()

    if testLoadAndDisplacement:
        # Affiche la force
        row = next(iter_rows)
        ax[row].plot(listX, np.abs(forces[listIter])*1e-3)
        ax[row].set_ylabel(r"$load \ [kN]$")
        ax[row].grid()        
    
    ax[-1].set_xlabel(nomX)

    if folder != "":        
        Save_fig(folder, "Energie")

    tic.Tac("PostTraitement","Cacul Energie phase field", False)

def Plot_ResumeIter(simu, folder="", iterMin=None, iterMax=None):
    """Affiche le resumé des itératons entre iterMin et iterMax

    Parameters
    ----------
    simu : _Simu
        Simulation
    folder : str, optional
        dossier de sauvegarde, by default ""
    iterMin : int, optional
        borne inférieur, by default None
    iterMax : int, optional
        borne supérieur, by default None
    """


    from Simulations import Simu

    assert isinstance(simu, Simu)

    # Recupère les résultats de simulation
    iterations, list_label_values = simu.Resultats_Get_ResumeIter_values()

    if iterMax == None:
        iterMax = iterations.max()

    if iterMin == None:
        iterMin = iterations.min()
    
    selectionIndex = list(filter(lambda iterations: iterations >= iterMin and iterations <= iterMax, iterations))

    nbGraph = len(list_label_values)

    iterations = iterations[selectionIndex]

    fig, axs = plt.subplots(nrows=nbGraph, sharex=True)
    
    for ax, label_values in zip(axs, list_label_values):
        ax.grid()
        ax.plot(iterations, label_values[1][iterations], color='blue')
        ax.set_ylabel(label_values[0])

    ax.set_xlabel("iterations")

    if folder != "":
        Save_fig(folder, "resumeConvergence")

__colors = {
    1 : 'tab:blue',
    2 : 'tab:orange',
    3 : 'tab:green',
    4 : 'tab:red',
    5 : 'tab:purple',
    6 : 'tab:brown',
    7 : 'tab:pink',
    8 : 'tab:gray',
    9 : 'tab:olive',
    10 : 'tab:cyan'
}
        
def __GetCoordo(simu, deformation: bool, facteurDef: float):
    """Recupération des coordonnée déformées si la simulation le permet avec la réponse de reussie ou non.

    Parameters
    ----------
    simu : _Simu
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
    
    from Simulations import Simu

    simu = cast(Simu, simu)

    coordo = simu.mesh.coordoGlob

    if deformation:

        uglob = simu.Resultats_matrice_displacement()
        
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
    
    maxRange = np.max(np.abs([xmin - xmax, ymin - ymax, zmin - zmax]))
    maxRange = maxRange*0.55

    xmid = (xmax + xmin)/2
    ymid = (ymax + ymin)/2
    zmid = (zmax + zmin)/2

    ax.set_xlim([xmid-maxRange, xmid+maxRange])
    ax.set_ylim([ymid-maxRange, ymid+maxRange])
    ax.set_zlim([zmid-maxRange, zmid+maxRange])
    ax.set_box_aspect([1,1,1])
        
def Save_fig(folder:str, filename: str,transparent=False, extension='pdf', dpi='figure'):

    if folder == "": return

    for char in ['NUL', '\ ', ',', '/',':','*', '?', '<','>','|']: filename = filename.replace(char, '')

    path = Folder.Join([folder, filename+'.'+extension])

    if not os.path.exists(folder):
        os.makedirs(folder)

    # dpi = 500    
    plt.savefig(path, dpi=dpi, transparent=transparent, bbox_inches='tight')   

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
    """Nettoie le terminal"""
    syst = platform.system()
    if syst in ["Linux","Darwin"]:
        os.system("clear")
    elif syst == "Windows":
        os.system("cls")