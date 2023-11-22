"""Display module for simulations and meshes."""

import Folder
from TicTac import Tic

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

def Plot_Result(obj, result: Union[str,np.ndarray], deformFactor=0.0, coef=1.0, nodeValues=True, 
                plotMesh=False, folder="", filename="", title="",
                cmap="jet", nColors=255, max=None, min=None, colorbarIsClose=False, ax: plt.Axes=None):
    """Display a simulation result.

    Parameters
    ----------
    obj : Simu or Mesh
        object containing the mesh
    result : str or np.ndarray
        result you want to display. Must be included in simu.Get_Results()
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    coef : float, optional
        coef to apply to the solution, by default 1.0
    nodeValues : bool, optional
        displays result to nodes otherwise displays it to elements, by default True
    plotMesh : bool, optional
        displays mesh, by default False    
    folder : str, optional
        save folder, by default "".
    filename : str, optional
        filename, by default ""
    title: str, optional
        figure title, by default ""
    cmap: str, optional
        the color map used near the figure, by default "jet" \n
        ["jet", "seismic", "binary"] -> https://matplotlib.org/stable/tutorials/colors/colormaps.html
    nColors : int, optional
        number of colors for colorbar
    max: float, optional
        maximum value in the colorbar, by default None
    min: float, optional
        minimum value in the colorbar, by default None
    colorbarIsClose : bool, optional
        color bar is displayed close to figure, by default False
    ax: axis, optional
        Axis to use, default None, by default None    

    Returns
    -------
    Figure, Axis, colorbar
        fig, ax, cb
    """
    
    tic = Tic()

    simu, mesh, coordo, inDim = __init_obj(obj, deformFactor)
    plotDim = mesh.dim # plot dimension

    deformFactor = 0 if simu is None else deformFactor

    # I can't yet display nodal values on lines
    nodeValues = False if plotDim is 1 else nodeValues

    # When mesh use 3D elements, results are displayed only on 2D elements.
    nodeValues = True if plotDim is 3 else nodeValues
    # Do not modify, you must use the node solution to locate the 2D elements !!!!
    # To display values on 2D elements, you first need to know the values at 3D nodes.

    # Retrieve values to be displayed
    if isinstance(result, str):
        if simu is None:
            raise Exception("obj is a mesh, so the result must be an array of dimension Nn or Ne")
        values = simu.Get_Result(result, nodeValues) # Retrieve result from option
        if not isinstance(values, np.ndarray): return
    
    elif isinstance(result, np.ndarray):
        values = result.copy()
        size = values.size        
        if size not in [mesh.Ne, mesh.Nn]:
            raise Exception("Must be an array of dimension Nn or Ne")
        if size == mesh.Ne and nodeValues:
            # calculate nodal values for element values
            values = simu.Results_Nodes_Values(mesh, values)
        elif size == mesh.Nn and not nodeValues:
            values_e = mesh.Locates_sol_e(values)
            values = np.mean(values_e, 1)        
    else:
        raise Exception("result must be a string or an array")
    
    values *= coef # Apply coef to values
    
    dict_Faces = mesh.Get_dict_connect_Faces() # build faces for each group of elements

    # Builds boundary markers for the colorbar
    if isinstance(result, str) and result == "damage":
        min = values.min()-1e-12
        max = np.max([values.max()+1e-12, 1])
    else:
        max = np.max(values)+1e-12 if max is None else max
        min = np.min(values)-1e-12 if min is None else min
    levels = np.linspace(min, max, nColors)

    if ax is not None:
        ax.clear()
        fig = ax.figure

    if inDim in [1,2]:
        # Mesh contained in a 2D plane
        # Currently only designed for one element group!

        if ax is None:
            fig, ax = plt.subplots()
        # rename the axis
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

        # construct coordinates for each faces
        faces = dict_Faces[mesh.groupElem.elemType]
        coordFaces = coordo[faces,:2]

        # Plot the mesh
        if plotMesh:
            if mesh.dim == 1:
                # mesh for 1D elements are points                
                ax.plot(*coordo.T, c='black', lw=0.1, marker='.', ls='')
            else:
                # mesh for 2D elements are lines
                pc = matplotlib.collections.LineCollection(coordFaces, edgecolor='black', lw=0.5)
                ax.add_collection(pc)

        # Plot element values
        if mesh.Ne == len(values):
            if mesh.dim == 1:
                pc = matplotlib.collections.LineCollection(coordFaces, lw=1.5, cmap=cmap)
            else:                
                pc = matplotlib.collections.PolyCollection(coordFaces, lw=0.5, cmap=cmap)                
            pc.set_clim(min, max)
            pc.set_array(values)
            ax.add_collection(pc)

        # Plot node values
        elif mesh.Nn == len(values):            
            # retrieve triangles from each face to use the trisurf function
            connectTri = mesh.dict_connect_Triangle[mesh.groupElem.elemType]
            # tripcolor, tricontour, tricontourf
            pc = ax.tricontourf(coordo[:,0], coordo[:,1], connectTri, values,
                                levels, cmap=cmap, vmin=min, vmax=max)

        # scale the axis
        ax.autoscale()
        if mesh.dim != 1:
            ax.axis('equal')
        
        # Building the colorbar
        if isinstance(result, str) and result == "damage":
            ticks = np.linspace(0,1,11)
        else:
            ticks = np.linspace(min,max,11)

        if colorbarIsClose:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='10%', pad=0.1)
            # # cax = divider.add_auto_adjustable_area(use_axes=ax, pad=0.1, adjust_dirs='right')
        else:
            cax=None
    
        cb = plt.colorbar(pc, ax=ax, cax=cax, ticks=ticks)
    
    elif inDim == 3:
        # If the mesh is a 3D mesh, only the 2D elements of the mesh will be displayed.
        # A 3D mesh can contain several types of 2D element.
        # For example, when PRISM6 -> TRI3 and QUAD4 at the same time

        plotDim = 2 if plotDim == 3 else plotDim

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.view_init(elev=105, azim=-90)
        # rename the axis
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")

        # constructs the face connection matrix
        connectFaces = []
        list_groupElemDim = mesh.Get_list_groupElem(plotDim)
        for groupElem in list_groupElemDim:
            connectFaces.extend(dict_Faces[groupElem.elemType])
        connectFaces = np.array(connectFaces)

        coordFaces: np.ndarray = coordo[connectFaces, :3]

        if nodeValues:
            # If the result is stored at nodes, we'll average the node values over the element.
            facesValues = []
            # for each group of elements, we'll calculate the value to be displayed on each element
            for groupElem in list_groupElemDim:                
                values_loc = values[groupElem.connect]
                values_e = np.mean(values_loc, axis=1)
                facesValues.extend(values_e)
            facesValues = np.array(facesValues)
        else:
            facesValues = values

        # update max and min
        max = np.max([facesValues.max(), max])
        min = np.min([facesValues.min(), min])

        # Display result with or without mesh display
        if plotMesh:
            if plotDim == 1:
                ax.plot(*mesh.coordoGlob.T, c='black', lw=0.1, marker='.', ls='')
                pc = Line3DCollection(coordFaces, cmap=cmap, zorder=0)
            elif plotDim == 2:
                pc = Poly3DCollection(coordFaces, edgecolor='black', linewidths=0.5, cmap=cmap, zorder=0)
        else:
            if plotDim == 1:
                pc = Line3DCollection(coordFaces, cmap=cmap, zorder=0)
            if plotDim == 2:
                pc = Poly3DCollection(coordFaces, cmap=cmap, zorder=0)

        # Colors are applied to the faces
        pc.set_array(facesValues)        
        ax.add_collection3d(pc)        
        
        # We set the colorbar limits and display it
        pc.set_clim(min, max)
        ticks = np.linspace(min,max,11)
        cb = fig.colorbar(pc, ax=ax, ticks=ticks)
        
        # Change axis scale
        _ScaleChange(ax, mesh.coordoGlob)

    # Title
    # if no title has been entered, the constructed title is used
    if title == "" and isinstance(result, str):
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
        title = optionTex+loc
        ax.set_title(fr"${title}$")
    else:
        ax.set_title(f"{title}")

    tic.Tac("Display","Plot_Result")

    # If the folder has been filled in, save the figure.
    if folder != "":
        if filename=="":
            filename = result
        Save_fig(folder, filename, transparent=False)

    

    # Returns figure, axis and colorbar
    return fig, ax, cb
    
def Plot_Mesh(obj, deformFactor=0.0, alpha=1.0, facecolors='c', edgecolor='black', lw=0.5,
              folder="", title="", ax: plt.Axes=None) -> plt.Axes:
    """Plot the mesh.

    Parameters
    ----------
    obj : Simu or Mesh
        object containing the mesh    
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    alpha : float, optional
        face transparency, default 1.0    
    facecolors: str, optional
        facecolors, default 'c' (cyan)
    edgecolor: str, optional
        edgecolor, default 'black'
    lw: float, optional
        line width, default 0.5    
    folder : str, optional
        save folder, default "".
    title: str, optional
        backup file name, default "".
    ax: plt.Axes, optional
        Axis to use, default None

    Returns
    -------
    plt.Axes
    """
    
    tic = Tic()

    simu, mesh, coordo, inDim = __init_obj(obj, deformFactor)

    deformFactor = 0 if simu is None else np.abs(deformFactor)

    # Dimensions of displayed elements
    plotDim = mesh.dim 
    # If the mesh is a 3D mesh, only the 2D elements of the mesh will be displayed.    
    if plotDim == 3: plotDim = 2
    
    # constructs the connection matrix for the faces
    dict_Faces = mesh.Get_dict_connect_Faces()
    connectFaces = []
    for groupElem in mesh.Get_list_groupElem(plotDim):
        connectFaces.extend(dict_Faces[groupElem.elemType])
    connectFaces = np.array(connectFaces)

    # faces coordinates
    coordFacesDef: np.ndarray = coordo[connectFaces, :inDim]
    coordFaces = mesh.coordoGlob[connectFaces, :inDim]
        
    if inDim in [1,2]:
        # in 2d space

        if ax is None:
            fig, ax = plt.subplots()
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

        if deformFactor > 0:            
            # Deformed mesh
            pc = matplotlib.collections.LineCollection(coordFacesDef, edgecolor='red', lw=lw, antialiaseds=True, zorder=1)
            ax.add_collection(pc)
            # Overlay undeformed and deformed mesh
            # Undeformed mesh
            pc = matplotlib.collections.LineCollection(coordFaces, edgecolor=edgecolor, lw=lw, antialiaseds=True, zorder=1) 
            ax.add_collection(pc)            
        else:
            # Undeformed mesh
            pc = matplotlib.collections.LineCollection(coordFaces, edgecolor=edgecolor, lw=lw, zorder=1)
            ax.add_collection(pc)
            if alpha > 0:
                pc = matplotlib.collections.PolyCollection(coordFaces, facecolors=facecolors, edgecolor=edgecolor, lw=lw, zorder=1, alpha=alpha)            
                ax.add_collection(pc)

        if mesh.dim == 1:
            # nodes
            ax.plot(*mesh.coordoGlob[:,:2].T, c='black', lw=lw, marker='.', ls='')
            if deformFactor > 0:
                ax.plot(*coordo[:,:2].T, c='red', lw=lw, marker='.', ls='')
        
        ax.autoscale()
        ax.axis('equal')

    elif inDim == 3:
        # in 3d space

        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
            ax.view_init(elev=105, azim=-90)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")

        if deformFactor > 0:
            # Displays only 1D or 2D elements, depending on the mesh type
            if plotDim > 1:
                # Deformed 2D mesh 
                pcDef = Poly3DCollection(coordFacesDef, edgecolor='red', linewidths=0.5, alpha=0, zorder=0)
                ax.add_collection3d(pcDef)                
                # Overlay the two meshes
                # Undeformed mesh
                # ax.scatter(x,y,z, linewidth=0, alpha=0)
                pcNonDef = Poly3DCollection(coordFaces, edgecolor=edgecolor, linewidths=0.5, alpha=0, zorder=0)
                ax.add_collection3d(pcNonDef)

            else:
                # Deformed mesh
                pc = Line3DCollection(coordFacesDef, edgecolor='red', lw=lw, antialiaseds=True, zorder=0)
                ax.add_collection3d(pc)
                # Overlay undeformed and deformed mesh
                # Undeformed mesh
                pc = Line3DCollection(coordFaces, edgecolor=edgecolor, lw=lw, antialiaseds=True, zorder=0)
                ax.add_collection3d(pc)
                # nodes
                ax.plot(*mesh.coordoGlob.T, c='black', lw=lw, marker='.', ls='')
                ax.plot(*coordo.T, c='red', lw=lw, marker='.', ls='')

        else:
            # Undeformed mesh
            # Displays only 1D or 2D elements, depending on the mesh type
            if plotDim > 1:
                pc = Poly3DCollection(coordFaces, facecolors=facecolors, edgecolor=edgecolor, linewidths=0.5, alpha=alpha, zorder=0)
            else:
                pc = Line3DCollection(coordFaces, edgecolor=edgecolor, lw=lw, antialiaseds=True, zorder=0)
                ax.plot(*coordo.T, c='black', lw=lw, marker='.', ls='')
            ax.add_collection3d(pc, zs=0, zdir='z')
            
        _ScaleChange(ax, coordo)
    
    if title == "":
        title = f"{mesh.elemType} : Ne = {mesh.Ne}, Nn = {mesh.Nn}"

    ax.set_title(title)

    tic.Tac("Display","Plot_Mesh")

    if folder != "":
        Save_fig(folder, "mesh")

    return ax

def Plot_Nodes(mesh, nodes=[], showId=False, marker='.', c='red',
               folder="", ax: plt.Axes=None) -> plt.Axes:
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
        marker type (matplotlib.markers), default '.'
    c : str, optional
        color, default 'red'
    folder : str, optional
        save folder, default "".
    ax : plt.Axes, optional
        Axis to use, default None, default None

    Returns
    -------
    plt.Axes
    """
    
    tic = Tic()
    
    from Mesh import Mesh
    mesh = cast(Mesh, mesh)

    if ax is None:
        ax = Plot_Mesh(mesh, alpha=0)
    ax.set_title("")
    
    if len(nodes) == 0:
        nodes = mesh.nodes    
    
    coordo = mesh.coordoGlob

    if mesh.inDim == 2:
        ax.plot(*coordo[nodes,:2].T, ls='', marker=marker, c=c, zorder=2.5)
        if showId:            
            [ax.text(*coordo[nodes,:2].T, str(noeud), c=c) for noeud in nodes]
    elif mesh.inDim == 3:            
        ax.plot(*coordo[nodes].T, ls='', marker=marker, c=c, zorder=2.5)
        if showId:
            [ax.text(*coordo[nodes].T, str(noeud), c=c) for noeud in nodes]

    tic.Tac("Display","Plot_Nodes")
    
    if folder != "":
        Save_fig(folder, "nodes")

    return ax

def Plot_Elements(mesh, nodes=[], dimElem: int=None, showId=False, alpha=1.0, c='red', edgecolor='black', folder="", ax: plt.Axes=None) -> plt.Axes:
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
    alpha : float, optional
        transparency of faces, by default 1.0
    c : str, optional
        color used to display faces, by default 'red
    edgecolor : str, optional
        color used to display segments, by default 'black'
    folder : str, optional
        save folder, by default ""
    ax : plt.Axes, optional
        Axis to use, default None

    Returns
    -------
    plt.Axes
    """

    tic = Tic()

    from Mesh import Mesh
    mesh = cast(Mesh, mesh)

    if dimElem is None:
        dimElem = 2 if mesh.inDim == 3 else mesh.dim

    # list of element group associated with the dimension
    list_groupElem = mesh.Get_list_groupElem(dimElem)[:1]
    if len(list_groupElem) == 0: return

    if ax is None:
        if mesh.inDim in [1,2]:
            fig, ax = plt.subplots()
            ax.autoscale()
            ax.axis('equal')        
        else:
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
            ax.view_init(elev=105, azim=-90)
            _ScaleChange(ax, mesh.coordo)

    # for each group elem
    for groupElem in list_groupElem:

        # get the elements associated with the nodes
        if len(nodes) > 0:
            elements = groupElem.Get_Elements_Nodes(nodes)
        else:
            elements = np.arange(groupElem.Ne)

        if elements.size == 0: continue

        # Construct the faces coordinates
        connect_e = groupElem.connect
        coord_n = groupElem.coordoGlob
        indexeFaces = groupElem.indexesFaces
        coordFaces_e = coord_n[connect_e[:, indexeFaces], :mesh.inDim]
        coordFaces = coordFaces_e[elements]

        # center coordinates for each elements
        center_e = np.mean(coordFaces_e, axis=1)
        
        # plot the entities associated with the tag
        if mesh.inDim in [1,2]:
            if groupElem.dim == 1:
                pc = matplotlib.collections.LineCollection(coordFaces, edgecolor=c, lw=1, zorder=2)
            else:
                pc = matplotlib.collections.PolyCollection(coordFaces, facecolors=c, edgecolor=edgecolor, lw=0.5, alpha=alpha, zorder=2)
            ax.add_collection(pc)
        elif mesh.inDim == 3:            
            pc = Poly3DCollection(coordFaces, facecolors=c, edgecolor=edgecolor, linewidths=0.5, alpha=alpha, zorder=2)
            ax.add_collection3d(pc, zdir='z')
        if showId:
            [ax.text(*center_e[element], element, zorder=25, ha='center', va='center') for element in elements]

    # ax.axis('off')

    tic.Tac("Display","Plot_Elements")
    
    if folder != "":
        Save_fig(folder, "noeuds")

    return ax

def Plot_BoundaryConditions(simu, folder="", ax: plt.Axes=None) -> plt.Axes:
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

    tic = Tic()

    from Simulations import _Simu

    simu = cast(_Simu, simu)
    coordo = simu.mesh.coordoGlob

    # get dirichlet and neumann boundary conditions
    dirchlets = simu.Bc_Dirichlet
    BoundaryConditions = dirchlets
    neumanns = simu.Bc_Neuman
    BoundaryConditions.extend(neumanns)
    displays = simu.Bc_Display # boundary conditions for display used for lagrangian boundary conditions
    BoundaryConditions.extend(displays)

    if ax is None:
        ax = Plot_Mesh(simu, alpha=0)

    for bc in BoundaryConditions:
        
        problemType = bc.problemType        
        dofsValues = bc.dofsValues
        directions = bc.directions
        nDir = len(directions)
        nodes = bc.nodes
        description = bc.description

        if problemType in ["damage","thermal"]:
            marker='o'
        elif problemType in ["displacement","beam"]:

            # get values for each direction
            sum = np.sum(dofsValues.reshape(-1, nDir), axis=0)
            values = np.round(sum, 2)
            # values will be use to choose the marker
            if len(directions) == 1:
                sign = np.sign(values[0])
                if directions[0] == 'x':
                    if sign == -1:
                        marker = '<'
                    else:
                        marker='>'
                elif directions[0] == 'y':
                    if sign == -1:
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

        # Title        
        directions_str = str(directions).replace("'","")
        title = f"{description} {directions_str}"

        lw=0
        ax.plot(*coordo[nodes,:simu.mesh.inDim].T, marker=marker, lw=lw, label=title, zorder=2.5, ls='')
    
    ax.legend()

    tic.Tac("Display","Plot_BoundaryConditions")

    if folder != "":
        Save_fig(folder, "Boundary conditions")

    return ax

def Plot_Model(obj, showId=True, folder="", alpha=1.0, ax: plt.Axes=None) -> plt.Axes:
    """Plot the model.

    Parameters
    ----------    
    obj : Simu or Mesh
        object containing the mesh
    showId : bool, optional
        show tags, by default True
    folder : str, optional
        save folder, by default ""
    alpha : float, optional
        transparency, by default 1.0
    ax : plt.Axes, optional
        Axis to use, default None

    Returns
    -------
    plt.Axes
    """

    tic = Tic()

    simu, mesh, coordo, inDim = __init_obj(obj, 0.0)

    if ax is None:
        if mesh.inDim <= 2:
            fig, ax = plt.subplots()
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.view_init(elev=105, azim=-90)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_zlabel(r"$z$")
    else:
        fig = ax.figure

    # get the group of elements for dimension 2 to 0
    listGroupElem = mesh.Get_list_groupElem(2)
    listGroupElem.extend(mesh.Get_list_groupElem(1))
    listGroupElem.extend(mesh.Get_list_groupElem(0))

    # List of collections during creation
    collections = []
    for groupElem in listGroupElem:        
        
        # Tags available by element group
        tags_e = groupElem.elementTags
        dim = groupElem.dim
        coordo = groupElem.coordoGlob[:, :inDim]
        faces = mesh.Get_dict_connect_Faces()[groupElem.elemType]
        coordoFaces = coordo[faces]
        center_e = np.mean(coordoFaces, axis=1) # center of each elements

        nColor = 0
        for tag_e in tags_e:

            nodes = groupElem.Get_Nodes_Tag(tag_e)
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
            else:
                color = (np.random.random(), np.random.random(), np.random.random())
            
            x_e = center_e[elements,0].mean()
            y_e = center_e[elements,1].mean()
            if inDim == 3:
                z_e = center_e[elements,2].mean()

            x_n = coordo[nodes,0]
            y_n = coordo[nodes,1]
            if inDim == 3:
                z_n = coordo[nodes,2]

            if inDim in [1,2]:
                # in 2D space
                if len(nodes) > 0:
                    # lines or surfaces
                    if dim == 0:
                        # plot points
                        collections.append(ax.plot(x_n, y_n, c='black', marker='.', zorder=2, label=tag_e, lw=2, ls=''))
                    elif dim == 1:
                        # plot lines
                        pc = matplotlib.collections.LineCollection(coordo_faces, lw=1.5, edgecolor='black', alpha=1, label=tag_e)
                        collections.append(ax.add_collection(pc))
                    else:
                        # plot surfaces
                        pc = matplotlib.collections.PolyCollection(coordo_faces, lw=1, alpha=alpha, facecolors=color, label=tag_e, edgecolor=color)
                        collections.append(ax.add_collection(pc))
                else:
                    # points
                    ax.plot(x_n, y_n, c='black', marker='.', zorder=2, ls='')
                    
                if showId:
                    # plot the tag on the center of the element
                    ax.text(x_e, y_e, tag_e, zorder=25)
                
            else:
                # in 3D space
                if len(nodes) > 0:
                    # lines or surfaces
                    if dim == 0:
                        # plot points
                        collections.append(ax.scatter(x_n, y_n, z_n, c='black', marker='.', zorder=2, label=tag_e, lw=2, zdir='z'))
                    elif dim == 1:
                        # plot lines
                        pc = Line3DCollection(coordo_faces, lw=1.5, edgecolor='black', alpha=1, label=tag_e)
                        # collections.append(ax.add_collection3d(pc, zs=z_e, zdir='z'))
                        collections.append(ax.add_collection3d(pc, zdir='z'))
                    elif dim == 2:
                        # plot surfaces
                        pc = Poly3DCollection(coordo_faces, lw=0, alpha=alpha, facecolors=color, label=tag_e)
                        pc._facecolors2d = color
                        pc._edgecolors2d = color                        
                        collections.append(ax.add_collection3d(pc, zdir='z'))                    
                else:
                    
                    collections.append(ax.scatter(x_n, y_n, z_n, c='black', marker='.', zorder=2, label=tag_e))

                if showId:
                    ax.text(x_e, y_e, z_e, tag_e, zorder=25)
    
    if inDim in [1, 2]:
        ax.autoscale()
        ax.axis('equal')        
    else:
        _ScaleChange(ax, coordo)

    tic.Tac("Display","Plot_Model")
    
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

def Plot_Load_Displacement(displacement: np.ndarray, forces: np.ndarray, xlabel='u', ylabel='f', folder="", ax: plt.Axes=None) -> tuple[plt.Figure, plt.Axes]:
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
    
    tic = Tic()

    # dpi = 500
    plt.savefig(path, dpi=dpi, transparent=transparent, bbox_inches='tight')   

    tic.Tac("Display","Save figure")

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

def __init_obj(obj, deformFactor: float=0.0):

    from Simulations import _Simu, Mesh

    # here we detect the nature of the object
    if isinstance(obj, _Simu):
        simu = obj
        mesh = simu.mesh
        u = simu.Results_displacement_matrix()
        coordo = mesh.coordoGlob + u * np.abs(deformFactor)
        inDim = np.max([simu.model.dim, mesh.inDim])
    elif isinstance(obj, Mesh):
        simu = None
        mesh = obj
        coordo = mesh.coordoGlob
        inDim = mesh.inDim
    else:
        raise Exception("Must be a simulation or mesh")
    
    return simu, mesh, coordo, inDim

# TODO use gmsh like x4_t1_1.msh