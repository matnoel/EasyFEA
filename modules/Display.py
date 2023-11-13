"""Display module for simulations and meshes."""

import platform
from typing import cast, Union
from colorama import Fore
import os
import numpy as np
import pandas as pd

# Figures
import matplotlib
import matplotlib.pyplot as plt
# Pour tracer des collections
import matplotlib.collections
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import Folder

def Plot_Result(obj, result: Union[str,np.ndarray], deformation=False, factorDef=4, coef=1.0, plotMesh=False, nodeValues=True, folder="", filename="", title="", ax=None, cmap="jet", colorbarIsClose=False, nColors=255, max=None, min=None):
    """Display a simulation result.

    Parameters
    ----------
    obj : Simu or Mesh
        object containing the mesh
    result : str
        result you want to display. Must be included in simu.Get_Results()
    deformation : bool, optional
        displays deformation, by default False
    factorDef : int, optional
        deformation factor, by default 4
    coef : float, optional
        coef to be applied to solution, by default 1.0
    plotMesh : bool, optional
        displays mesh, by default False
    nodeValues : bool, optional
        displays result to nodes otherwise displays it to elements, by default True
    folder : str, optional
        save folder, by default ""
    filename : str, optional
        filename, by default "" title: str, optional
    title: str, optional
        figure title, by default ""
    ax: axis, optional
        Axis to use, default None, by default None    
    cmap: str, optional
        the color map used near the figure, by default "jet" \n
        ["jet", "seismic", "binary"] -> https://matplotlib.org/stable/tutorials/colors/colormaps.html
    colorbarIsClose : bool, optional
        color bar is displayed close to figure, by default False
    nColors : int, optional
        number of colors for colorbar
    max: float, optional
        maximum value in the colorbar, by default None
    min: float, optional
        minimum value in the colorbar, by default None

    Returns
    -------
    Figure, Axis, colorbar
        fig, ax, cb
    """

    from Simulations import _Simu, Mesh, ModelType

    # here we detect the nature of the object
    if isinstance(obj, _Simu):
        simu = obj
        mesh = simu.mesh
        if simu.problemType == ModelType.beam:
            # Currently I don't know how to display nodal results, so I'm displaying on elements.
            nodeValues = False
            use3DBeamModel = simu.model.dim == 3
        else:
            use3DBeamModel = False

    elif isinstance(obj, Mesh):
        mesh = obj

        if deformation == True:
            deformation = False
            print("You have to give the simulation to display the deformed mesh.")
        use3DBeamModel = False

        if isinstance(result, str):
            raise Exception("When obj is a mesh, the option must be an array of dimension Nn or Ne.")
        
    else:
        raise Exception("Must be a simulation or mesh")
    
    if ax != None:
        assert isinstance(ax, plt.Axes)
        fig = ax.figure
    
    dim = mesh.dim # mesh dimension
    inDim = mesh.inDim # dimension in which the mesh is located

    # Construction of figure and axis if necessary
    if ax is None:
        if inDim in [1,2] and not use3DBeamModel:
            fig, ax = plt.subplots()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.view_init(elev=105, azim=-90)
    else:
        fig = fig
        ax = ax
        ax.clear()

    if dim == 3:
        nodeValues = True # Do not modify, you must use the node solution to locate the 2D elements!!!!
        # When mesh use 3D elements, results are displayed only on 2D elements.
        # To take up less space.

    # Retrieve values to be displayed
    if isinstance(result, str):
        valeurs = simu.Get_Result(result, nodeValues) # Retrieve result from option
        if not isinstance(valeurs, np.ndarray): return
    
    elif isinstance(result, np.ndarray):
        # Gets the size of the array, the size must be nodes or elements
        # If the size is not equal to the number of nodes or elements, returns an error
        sizeVecteur = result.size

        if sizeVecteur not in [mesh.Ne, mesh.Nn]:
            print("The vector must be of dimension Nn or Ne.")
            return

        valeurs = result*coef
        
        if sizeVecteur == mesh.Ne and nodeValues:
            valeurs = _Simu.Results_Nodes_Values(mesh, valeurs)
        elif sizeVecteur == mesh.Nn and not nodeValues:
            valeursLoc_e = mesh.Locates_sol_e(valeurs)
            valeurs = np.mean(valeursLoc_e, 1)
    else:
        raise Exception("Must fill a string or an array")
    
    valeurs *= coef # Apply coef to values
    coordoNonDef = mesh.coordoGlob # node coordinates without deformations

    # Recover deformed coordinates if simulation permits
    if deformation:        
        coordoDef, deformation = __GetCoordo(simu, deformation, factorDef)
    else:
        coordoDef = coordoNonDef.copy()
    
    coordoDef_InDim = coordoDef[:,range(inDim)]
    
    dict_connect_Faces = mesh.Get_dict_connect_Faces() # build faces for each group of elements

    # Builds boundary markers for the colorbar
    if isinstance(result, str) and result == "damage":
        min = valeurs.min()-1e-12
        openCrack = False
        if openCrack:
            max = 0.98
        else:
            max = valeurs.max()+1e-12
            if max < 1:
                max = 1        
    else:
        max = np.max(valeurs)+1e-12 if max is None else max
        min = np.min(valeurs)-1e-12 if min is None else min
    levels = np.linspace(min, max, nColors)

    if inDim in [1,2] and not use3DBeamModel:
        # Mesh contained in a 2D plane
        # Currently only designed for one element group!

        faces = dict_connect_Faces[mesh.groupElem.elemType]

        coordFaces = coordoDef_InDim[faces]        

        # Plot the mesh
        if plotMesh:
            if mesh.dim == 1:
                # mesh for 1D elements are points
                coordFaces = coordFaces.reshape(-1,inDim)
                ax.scatter(coordFaces[:,0], coordFaces[:,1], c='black', lw=0.1, marker='.')
            else:
                # mesh for 2D elements are lines
                pc = matplotlib.collections.LineCollection(coordFaces, edgecolor='black', lw=0.5)
                ax.add_collection(pc)

        # Element values
        if mesh.Ne == len(valeurs):            
            if mesh.dim == 1:
                # we will display the result on each line
                pc = matplotlib.collections.LineCollection(coordFaces, lw=1.5, cmap=cmap)
            else:
                # we'll display the result on the faces
                pc = matplotlib.collections.PolyCollection(coordFaces, lw=0.5, cmap=cmap)                
            pc.set_clim(min, max)
            pc.set_array(valeurs)
            ax.add_collection(pc)

        # Node values
        elif mesh.Nn == len(valeurs):
            # display the result on the nodes
            # retrieve triangles from each face to use the trisurf function
            connectTri = mesh.dict_connect_Triangle[mesh.groupElem.elemType]
            pc = ax.tricontourf(coordoDef[:,0], coordoDef[:,1], connectTri, valeurs, levels, cmap=cmap, vmin=min, vmax=max)
            # tripcolor, tricontour, tricontourf

        # scale the axis
        ax.autoscale()
        if mesh.dim != 1: ax.axis('equal')

        # procedure for trying to retrieve the colorbar from the axis
        divider = make_axes_locatable(ax)
        if colorbarIsClose:
            cax = divider.append_axes('right', size='10%', pad=0.1)
            # # cax = divider.add_auto_adjustable_area(use_axes=ax, pad=0.1, adjust_dirs='right')
        else:
            cax=None
        
        # Building the colorbar
        if isinstance(result, str) and result == "damage":
            ticks = np.linspace(0,1,11)
        else:
            ticks = np.linspace(min,max,11)
        cb = plt.colorbar(pc, ax=ax, cax=cax, ticks=ticks)
        
        # rename the axis
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

    
    elif inDim == 3 or use3DBeamModel:
        # initialization of max and min colorbar values
                
        maxVal = max
        minVal = min

        # mesh dimension
        dim = mesh.dim
        if dim == 3:
            # If the mesh is a 3D mesh, then only the 2D elements of the mesh will be displayed.
            # A 3D mesh can contain several types of 2D element.
            # For example, when PRISM6 -> TRI3 and QUAD4 at the same time
            # Basically, the outer layer
            dim = 2

        # constructs the face connection matrix
        connectFaces = []
        list_groupElemDim = mesh.Get_list_groupElem(dim)
        for groupElem in list_groupElemDim:
            connectFaces.extend(dict_connect_Faces[groupElem.elemType])
        connectFaces = np.array(connectFaces)

        coordFaces = coordoDef[connectFaces]

        if nodeValues:
            # If the result is stored at nodes, we'll average the node values over the element.

            valeursAuxFaces = []
            # for each group of elements, we'll calculate the value to be displayed on each element
            for groupElem in list_groupElemDim:
                valeursNoeudsSurElement = valeurs[groupElem.connect]
                values = np.asarray(np.mean(valeursNoeudsSurElement, axis=1))
                valeursAuxFaces.extend(values)
            valeursAuxFaces = np.asarray(valeursAuxFaces)
        else:
            valeursAuxFaces = valeurs

        # update max and min
        maxVal = valeursAuxFaces.max()
        minVal = valeursAuxFaces.min()        
        maxVal = np.max([maxVal, max])
        minVal = np.min([minVal, min])

        # Display result with or without mesh display
        if plotMesh:
            if dim == 1:
                pc = Line3DCollection(coordFaces, edgecolor='black', linewidths=0.5, cmap=cmap, zorder=0)
            elif dim == 2:
                pc = Poly3DCollection(coordFaces, edgecolor='black', linewidths=0.5, cmap=cmap, zorder=0)
        else:
            if dim == 1:
                pc = Line3DCollection(coordFaces, cmap=cmap, zorder=0)
            if dim == 2:
                pc = Poly3DCollection(coordFaces, cmap=cmap, zorder=0)

        # Colors are applied to the faces
        pc.set_array(valeursAuxFaces)        
        ax.add_collection3d(pc)        
        
        # We set the colorbar limits and display it
        pc.set_clim(minVal, maxVal)
        ticks = np.linspace(minVal,maxVal,11)
        cb = fig.colorbar(pc, ax=ax, ticks=ticks)

        # rename the axis
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")
        
        # Change axis scale
        _ScaleChange(ax, coordoNonDef)

    # Title
    optionTex = result
    if isinstance(result, str):
        if result == "damage":
            optionTex = "\phi"
        elif result == "thermal":
            optionTex = "T"
        elif "S" in result and not result in ["amplitudeSpeed"]:
            optionFin = result.split('S')[-1]
            optionTex = f"\sigma_{'{'+optionFin+'}'}"
        elif "E" in result:
            optionFin = result.split('E')[-1]
            optionTex = f"\epsilon_{'{'+optionFin+'}'}"
    
    # Specify whether values are on nodes or elements
    if nodeValues:
        # loc = "^{n}"
        loc = ""
    else:
        loc = "^{e}"
    
    # if no title has been entered, the constructed title is used
    if title == "" and isinstance(result, str):
        title = optionTex+loc
        ax.set_title(fr"${title}$")
    else:
        ax.set_title(f"{title}")

    # If the folder has been filled in, save the figure.
    if folder != "":
        if filename=="":
            filename=result
        Save_fig(folder, filename, transparent=False)

    # Returns figure, axis and colorbar
    return fig, ax, cb
    
def Plot_Mesh(obj, deformation=False, factorDef=4, folder="", title="", ax=None, lw=0.5, alpha=1.0, facecolors='c', edgecolor='black') -> plt.Axes:
    """Plot the mesh.

    Parameters
    ----------
    obj : Simu or Mesh
        object containing the mesh
    factorDef : int, optional
        deformation factor, default 4
    deformation : bool, optional
        displays deformation, default False
    folder : str, optional
        save folder, default "".
    title: str, optional
        backup file name, default "".
    ax: plt.Axes, optional
        Axis to use, default None
    lw: float, optional
        line thickness, default 0.5
    alpha : float, optional
        face transparency, default 1.0

    Returns
    -------
    plt.Axes
    """

    from Simulations import _Simu, Mesh

    if isinstance(obj, _Simu):
        simu = obj
        mesh = simu.mesh
        use3DBeamModel = simu.problemType == "beam" and simu.model == 3
    elif isinstance(obj, Mesh):
        mesh = obj
        if deformation == True:
            print("You have to give the simulation to display the distorted mesh.")
        use3DBeamModel = False
    else:
        raise Exception("Must be a simulation or mesh.")
    
    assert factorDef > 0, "The deformation factor must be > 0"

    coordo = mesh.coordoGlob

    inDim = mesh.groupElem.inDim

    # coordinates
    inDim = 3 if use3DBeamModel else mesh.groupElem.inDim
    coordo = coordo[:,range(inDim)]
    if deformation:
        coordoDeforme, deformation = __GetCoordo(simu, deformation, factorDef)
        coordoDeforme = coordoDeforme[:,range(inDim)]    

    # Dimensions of displayed elements
    dimElem = mesh.dim
    if dimElem == 3:
        # If the mesh is a 3D mesh, then only the 2D elements of the mesh will be displayed.
        # Basically the outer layer
        dimElem = 2
    
    # constructs the connection matrix for the faces
    dict_connect_Faces = mesh.Get_dict_connect_Faces()
    connectFaces = []
    for groupElem in mesh.Get_list_groupElem(dimElem):
        connectFaces.extend(dict_connect_Faces[groupElem.elemType])
    connectFaces = np.array(connectFaces)

    coordFaces = coordo[connectFaces]

    if deformation:
        coordFacesDeforme = coordoDeforme[connectFaces]
        
    if inDim in [1,2] and not use3DBeamModel:

        if ax is None:
            fig, ax = plt.subplots()

        if deformation:

            coordFacesDeforme = coordoDeforme[connectFaces]
            
            # Overlay undeformed and deformed mesh
            # Undeformed mesh
            pc = matplotlib.collections.LineCollection(coordFaces, edgecolor=edgecolor, lw=lw, antialiaseds=True, zorder=1)
            ax.add_collection(pc)

            # Deformed mesh
            pc = matplotlib.collections.LineCollection(coordFacesDeforme, edgecolor='red', lw=lw, antialiaseds=True, zorder=1)
            ax.add_collection(pc)

        else:
            # Undeformed mesh
            if alpha == 0:
                pc = matplotlib.collections.LineCollection(coordFaces, edgecolor=edgecolor, lw=lw, zorder=1)
            else:
                pc = matplotlib.collections.PolyCollection(coordFaces, facecolors=facecolors, edgecolor=edgecolor, lw=lw, zorder=1)
            ax.add_collection(pc)

        if mesh.dim == 1:
            coordFaces = coordFaces.reshape(-1,inDim)
            ax.scatter(coordFaces[:,0], coordFaces[:,1], c='black', lw=lw, marker='.')
            if deformation:
                coordDeforme = coordFacesDeforme.reshape(-1, inDim)
                ax.scatter(coordDeforme[:,0], coordDeforme[:,1], c='red', lw=lw, marker='.')            
        
        ax.autoscale()
        ax.axis('equal')
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

    # ETUDE 3D    
    elif inDim == 3 or use3DBeamModel:

        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
            ax.view_init(elev=105, azim=-90)

        if deformation:
            # Displays only 1D or 2D elements, depending on the mesh type

            if dimElem > 1:
                # Overlay the two meshes
                # Undeformed mesh
                # ax.scatter(x,y,z, linewidth=0, alpha=0)
                pcNonDef = Poly3DCollection(coordFaces, edgecolor=edgecolor, linewidths=0.5, alpha=0, zorder=0)
                ax.add_collection3d(pcNonDef)

                # Maillage deformé
                pcDef = Poly3DCollection(coordFacesDeforme, edgecolor='red', linewidths=0.5, alpha=0, zorder=0)
                ax.add_collection3d(pcDef)
            else:
                # Overlay undeformed and deformed mesh
                # Undeformed mesh
                pc = Line3DCollection(coordFaces, edgecolor=edgecolor, lw=lw, antialiaseds=True, zorder=0)
                ax.add_collection3d(pc)

                # Deformed mesh
                pc = Line3DCollection(coordFacesDeforme, edgecolor='red', lw=lw, antialiaseds=True, zorder=0)
                ax.add_collection3d(pc)
                
                ax.scatter(coordo[:,0], coordo[:,1], coordo[:,2], c='black', lw=lw, marker='.')
                ax.scatter(coordoDeforme[:,0], coordoDeforme[:,1], coordoDeforme[:,2], c='red', lw=lw, marker='.')                

        else:
            # Undeformed mesh
            # Displays only 1D or 2D elements, depending on the mesh type

            if dimElem > 1:
                pc = Poly3DCollection(coordFaces, facecolors=facecolors, edgecolor=edgecolor, linewidths=0.5, alpha=alpha, zorder=0)
            else:
                pc = Line3DCollection(coordFaces, edgecolor=edgecolor, lw=lw, antialiaseds=True, zorder=0)
                ax.scatter(coordo[:,0], coordo[:,1], coordo[:,2], c='black', lw=lw, marker='.')
            ax.add_collection3d(pc, zs=0, zdir='z')
            
        _ScaleChange(ax, coordo)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")
    
    if title == "":
        title = f"{mesh.elemType} : Ne = {mesh.Ne}, Nn = {mesh.Nn}"

    ax.set_title(title)

    if folder != "":
        Save_fig(folder, "mesh")

    return ax

def Plot_Nodes(mesh, nodes=[], showId=False, marker='.', c='red', folder="", ax=None) -> plt.Axes:
    """Plot mesh nodes.

    Parameters
    ----------
    mesh : Mesh
        mesh    
    nodes : list[np.ndarray], optional
        nodes to display, default []
    showId : bool, optional
        display numbers, default False
    marker : str, optional
        marker type (matplotlib.markers), default ''
    c : str, optional
        mesh color, default 'blue'
    folder : str, optional
        save folder, default "".    
    ax : plt.Axes, optional
        Axis to use, default None, default None

    Returns
    -------
    plt.Axes
    """
    
    from Mesh import Mesh
    mesh = cast(Mesh, mesh)

    if ax is None:
        ax = Plot_Mesh(mesh, alpha=0)
    
    if len(nodes) == 0:
        nodes = mesh.nodes    
    
    coordo = mesh.coordoGlob

    if mesh.inDim == 2:
        ax.plot(coordo[nodes,0], coordo[nodes,1], ls='', marker=marker, c=c, zorder=2.5)
        if showId:            
            [ax.text(coordo[noeud,0], coordo[noeud,1], str(noeud), c=c) for noeud in nodes]
    elif mesh.inDim == 3:            
        ax.plot(coordo[nodes,0], coordo[nodes,1], coordo[nodes,2], ls='', marker=marker, c=c, zorder=2.5)
        if showId:
            [ax.text(coordo[noeud,0], coordo[noeud,1], coordo[noeud,2], str(noeud), c=c) for noeud in nodes]
    
    if folder != "":
        Save_fig(folder, "nodes")

    return ax

def Plot_Elements(mesh, nodes=[], dimElem=None, showId=False, c='red', edgecolor='black', alpha=1.0, folder="", ax=None) -> plt.Axes:
    """Display mesh elements from given nodes.

    Parameters
    ----------
    mesh : Mesh
        mesh
    nodes : list, optional
        node numbers, by default []
    dimElem : int, optional
        dimension of elements, by default None
    showId : bool, optional
        display numbers, by default False    
    c : str, optional
        color used to display faces, by default 'red
    edgecolor : str, optional
        color used to display segments, by default 'black
    alpha : float, optional
        transparency of faces, by default 1.0
    folder : str, optional
        save folder, by default ""
    ax : plt.Axes, optional
        Axis to use, default None

    Returns
    -------
    plt.Axes
    """

    from Mesh import Mesh
    mesh = cast(Mesh, mesh)

    if dimElem is None:
        dimElem = mesh.dim-1 if mesh.inDim == 3 else mesh.dim

    list_groupElem = mesh.Get_list_groupElem(dimElem)[:1]
    if len(list_groupElem) == 0: return

    if ax is None:
        ax = Plot_Mesh(mesh, alpha=alpha)

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
            if groupElemDim.dim == 1:
                pc = matplotlib.collections.LineCollection(coordoFaces[:,:,range(mesh.inDim)], edgecolor=c, lw=1, zorder=2)
            else:
                pc = matplotlib.collections.PolyCollection(coordoFaces[:,:,range(mesh.inDim)], facecolors=c, edgecolor=edgecolor, lw=0.5, alpha=alpha, zorder=2)
            ax.add_collection(pc)

            # ax.scatter(coordo[:,0], coordo[:,1], marker=marker, c=c, zorder=2)
            if showId:
                [ax.text(coordo_e[element,0], coordo_e[element,1], element,
                zorder=25, ha='center', va='center') for element in elements]
        elif mesh.dim == 3:
            ax.add_collection3d(Poly3DCollection(coordoFaces, facecolors=c, edgecolor=edgecolor, linewidths=0.5, alpha=alpha, zorder=2), zdir='z')

            # ax.scatter(coordo[:,0], coordo[:,1], coordo[:,2], marker=marker, c=c, zorder=2)
            if showId:
                [ax.text(coordo_e[element,0], coordo_e[element,1], coordo_e[element,2], element, zorder=25, ha='center', va='center') for element in elements]

    # ax.axis('off')
    
    if folder != "":
        Save_fig(folder, "noeuds")

    return ax

def Plot_BoundaryConditions(simu, folder="", ax=None) -> plt.Axes:
    """Plot boundary conditions.

    Parameters
    ----------
    simu : Simu
        simulation
    folder : str, optional
        save folder, by default ""
    ax : plt.Axes, optional
        Axis to use, default None

    Returns
    -------
    plt.Axes
    """

    from Simulations import _Simu

    simu = cast(_Simu, simu)

    dim = simu.dim

    # Récupérations des Conditions de chargement de déplacement ou de liaison
    dirchlets = simu.Bc_Dirichlet
    Conditions = dirchlets

    neumanns = simu.Bc_Neuman
    Conditions.extend(neumanns)

    displays = simu.Bc_Display # boundary conditions for display
    Conditions.extend(displays)

    if ax is None:
        ax = Plot_Mesh(simu, alpha=0)

    assert isinstance(ax, plt.Axes)

    coordo = simu.mesh.coordoGlob    

    for bc_Conditions in Conditions:

        problemType = bc_Conditions.problemType        
        valeurs_ddls = bc_Conditions.dofsValues
        directions = bc_Conditions.directions

        # récupère les noeuds
        noeuds = bc_Conditions.nodes

        if problemType == "damage":
            valeurs = [valeurs_ddls[0]]
        else:
            valeurs = np.round(list(np.sum(valeurs_ddls.copy().reshape(-1, len(directions)), axis=0)), 2)
        
        description = bc_Conditions.description
        
        directions_str = str(directions).replace("'","")

        titre = f"{description} {directions_str}"

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
                if "Connection" in description:
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
        Save_fig(folder, "Boundary conditions")

    return ax

def Plot_Model(obj, showId=True, ax=None, folder="", alpha=1.0) -> plt.Axes:
    """Plot the model.

    Parameters
    ----------    
    obj : Simu or Mesh
        object containing the mesh
    showId : bool, optional
        show tags, by default True
    ax : plt.Axes, optional
        Axis to use, default None
    folder : str, optional
        save folder, by default ""
    alpha : float, optional
        transparency, by default 1.0

    Returns
    -------
    plt.Axes
    """

    from Simulations import _Simu
    from Mesh import Mesh, GroupElem

    typeobj = type(obj).__name__

    if typeobj == _Simu.__name__:
        simu = cast(_Simu, obj)
        mesh = simu.mesh
    elif typeobj == Mesh.__name__:
        mesh = cast(Mesh, obj)
    else:
        raise Exception("Must be a simulation or a mesh.")

    inDim = mesh.inDim

    # Create axes if necessary
    if ax is None:
        # ax = Plot_Mesh(mesh, facecolors='c', edgecolor='black')
        # fig = ax.figure
        if mesh.inDim in [0,1,2]:
            fig, ax = plt.subplots()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.view_init(elev=105, azim=-90)
    else:
        fig = ax.figure

    # Here, for each element group in the mesh, we plot the elements belonging to the element group.
    listGroupElem = cast(list[GroupElem], [])
    listDim = np.arange(mesh.dim, -1, -1, dtype=int)
    for dim in listDim:
        if dim < 3:
            # 3D elements are not added
            listGroupElem.extend(mesh.Get_list_groupElem(dim))

    # List of collections during creation
    collections = []
    for groupElem in listGroupElem:        
        # Tags available by element group
        tags_e = groupElem.elementTags
        dim = groupElem.dim
        coordo = groupElem.coordoGlob[:, range(inDim)]
        faces = mesh.Get_dict_connect_Faces()[groupElem.elemType]
        coordoFaces = coordo[faces]
        coordo_e = np.mean(coordoFaces, axis=1)

        nColor = 0
        for tag_e in tags_e:

            noeuds = groupElem.Get_Nodes_Tag(tag_e)
            elements = groupElem.Get_Elements_Tag(tag_e)
            if len(elements) == 0: continue

            coordo_faces = coordoFaces[elements]

            needPlot = True
            
            # Assigns color
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
        _ScaleChange(ax, coordo)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")
    
    if folder != "":
        Save_fig(folder, "geom")

    __Annotation_Event(collections, fig, ax)

    return ax

def __Annotation_Event(collections: list, fig: plt.Figure, ax: plt.Axes) -> None:
    """Create an event to display the element tag currently active under the mouse (at the bottom of the figure)."""
    
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

def Plot_Load_Displacement(displacement: np.ndarray, forces: np.ndarray, xlabel='u', ylabel='f', folder="", ax=None) -> tuple[plt.Figure, plt.Axes]:
    """Plot the forces displacement curve.

    Parameters
    ----------
    displacements : np.ndarray
        array of values for displacements
    forces : np.ndarray
        array of values for forces
    xlabel : str, optional
        x-axis title, by default 'u'.
    ylabel : str, optional
        y-axis title, by default 'f' folder : str, optional
    folder : str, optional
        save folder, by default ""
    ax : plt.Axes, optional
        ax in which to plot the figure, by default None

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        returns figure and ax
    """

    if isinstance(ax, plt.Axes):
        fig = ax.figure
        ax.clear()
    else:        
        fig, ax = plt.subplots()

    ax.plot(np.abs(displacement), np.abs(forces), c='blue')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()

    if folder != "":
        Save_fig(folder, "load_displacement")

    return fig, ax
    
def Plot_Energy(simu, load=np.array([]), displacement=np.array([]), plotSolMax=True, Niter=200, NiterFin=100, folder="") -> None:
    """Plot the energy for each iteration.

    Parameters
    ----------
    simu : Simu
        simulation
    load : np.ndarray, optional
        array of values, by default np.array([])
    displacement : np.ndarray, optional
        array of values, by default np.array([])
    plotSolMax : bool, optional
        displays the evolution of the maximul solution over iterations. (max damage for damage simulation), by default True
    Niter : int, optional
        number of iterations for which energy will be calculated, by default 200
    NiterFin : int, optional
        number of iterations before end, by default 100
    folder : str, optional        
        save folder, by default ""
    """

    from Simulations import _Simu
    from TicTac import Tic
    import PostProcessing as PostProcessing 

    assert isinstance(simu, _Simu)

    # First we check whether the simulation can calculate energies
    if len(simu.Results_dict_Energy()) == 0:
        print("This simulation don't calculate energies.")
        return

    # Check whether it is possible to plot the force-displacement curve
    pltLoad = len(load) == len(displacement) and len(load) > 0    
        
    # For each displacement increment we calculate the energy
    tic = Tic()
    
    # recovers simulation results
    results =  simu.results
    N = len(results)
    if len(load) > 0:
        ecart = np.abs(len(results) - len(load))
        if ecart != 0:
            N -= ecart
    listIter = PostProcessing.Make_listIter(NiterMax=N-1, NiterFin=NiterFin, NiterCyble=Niter)
    
    Niter = len(listIter)

    list_dict_Energy = cast(list[dict[str, float]], [])
    listTemps = []
    if plotSolMax : listSolMax = []

    for i, iteration in enumerate(listIter):

        # Update simulation at iteration i
        simu.Update_Iter(iteration)

        if plotSolMax : listSolMax.append(simu.get_u_n(simu.problemType).max())

        list_dict_Energy.append(simu.Results_dict_Energy())

        temps = tic.Tac("PostProcessing","Calc Energy", False)
        listTemps.append(temps)

        pourcentageEtTempsRestant = PostProcessing._RemainingTime(listIter, listTemps, i)

        print(f"Calc Energy {iteration+1}/{N} {pourcentageEtTempsRestant}    ", end='\r')
    print('\n')

    # Figure construction
    nrows = 1
    if plotSolMax:
        nrows += 1
    if pltLoad:
        nrows += 1
    fig, ax = plt.subplots(nrows, 1, sharex=True)

    iter_rows = iter(np.arange(nrows))

    # Retrieves the axis to be used for x-axes
    if len(displacement)>0:
        listX = displacement[listIter] 
        xlabel = "displacement"
    else:
        listX = listIter 
        xlabel = "iter"    

    # Transforms list_dict_Energie into a dataframe
    df = pd.DataFrame(list_dict_Energy)

    # Affiche les energies
    row = next(iter_rows)
    # For each energy, we plot the values
    for energie_str in df.columns:
        valeurs = df[energie_str].values
        ax[row].plot(listX, valeurs, label=energie_str)    
    ax[row].legend()
    ax[row].grid()

    if plotSolMax:
        # plot max solution
        row = next(iter_rows)
        ax[row].plot(listX, listSolMax)
        ax[row].set_ylabel(r"$max(u_n)$")
        ax[row].grid()

    if pltLoad:
        # plot the loading
        row = next(iter_rows)
        ax[row].plot(listX, np.abs(load[listIter])*1e-3)
        ax[row].set_ylabel("load")
        ax[row].grid()        
    
    ax[-1].set_xlabel(xlabel)

    if folder != "":        
        Save_fig(folder, "Energy")

    tic.Tac("PostProcessing","Calc Energy", False)

def Plot_Iter_Summary(simu, folder="", iterMin=None, iterMax=None) -> None:
    """Display summary of iterations between iterMin and iterMax.

    Parameters
    ----------
    simu : Simu
        Simulation
    folder : str, optional
        backup folder, by default ""
    iterMin : int, optional
        lower bound, by default None
    iterMax : int, optional
        upper bound, by default None
    """

    from Simulations import _Simu

    assert isinstance(simu, _Simu)

    # Recovers simulation results
    iterations, list_label_values = simu.Results_Iter_Summary()

    if iterMax is None:
        iterMax = iterations.max()

    if iterMin is None:
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
        
def __GetCoordo(simu, deformation: bool, facteurDef: float) -> np.ndarray:
    """Recover deformed coordinates if the simulation allows it with the response of passed or failed.

    Parameters
    ----------
    simu : Simu
        simulation
    deformation : bool
        deformation
    factorDef : float
        deformation factor

    Returns
    -------
    np.ndarray
    """
    
    from Simulations import _Simu

    simu = cast(_Simu, simu)

    coordo = simu.mesh.coordoGlob

    if deformation:

        uglob = simu.Results_displacement_matrix()
        
        test = isinstance(uglob, np.ndarray)

        if test:
            coordoDef = coordo + uglob * facteurDef

        return coordoDef, test
    else:
        return coordo, deformation

def _ScaleChange(ax, coordo: np.ndarray) -> None:
    """Change axis size for 3D display
    Will center the part and make the axes the right size
    Parameters
    ----------
    ax : plt.Axes
        Axes in which figure will be created
    coordo : np.ndarray
        mesh coordinates
    """

    # Change axis size
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
        
def Save_fig(folder:str, filename: str, transparent=False, extension='pdf', dpi='figure') -> None:
    """Save the current figure.

    Parameters
    ----------
    folder : str
        save folder
    filename : str
        filenemae
    transparent : bool, optional
        transparent, by default False
    extension : str, optional
        extension, by default 'pdf'
    dpi : str, optional
        dpi, by default 'figure'
    """

    if folder == "": return

    # the filename must not contain these characters
    for char in ['NUL', '\ ', ',', '/',':','*', '?', '<','>','|']: filename = filename.replace(char, '')

    path = Folder.Join([folder, filename+'.'+extension])

    if not os.path.exists(folder):
        os.makedirs(folder)

    # dpi = 500
    plt.savefig(path, dpi=dpi, transparent=transparent, bbox_inches='tight')   

def Section(text: str, verbosity=True) -> None:
    """New section."""    
    bord = "======================="

    longeurTexte = len(text)

    longeurMax = 45

    bord = "="*int((longeurMax - longeurTexte)/2)

    section = f"\n\n{bord} {text} {bord}\n"

    if verbosity: print(section)

    return section

def Clear() -> None:
    """Clear the terminal."""
    syst = platform.system()
    if syst in ["Linux","Darwin"]:
        os.system("clear")
    elif syst == "Windows":
        os.system("cls")