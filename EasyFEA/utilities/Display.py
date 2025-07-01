# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing functions used to display simulations and meshes with matplotlib (https://matplotlib.org/)."""

import platform
from typing import Union, Callable, Optional, TYPE_CHECKING, Any
import numpy as np
import pandas as pd
from enum import Enum

# Matplotlib: https://matplotlib.org/
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection, LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable  # use to do colorbarIsClose
import matplotlib.animation as animation

# utilities
from . import Folder, Tic, _types

# simulations
from ..simulations._simu import _Init_obj, _Get_values

if TYPE_CHECKING:
    from ..simulations._simu import _Simu, Mesh

# Ideas: https://www.python-graph-gallery.com/


# ----------------------------------------------
# Plot Simu or Mesh
# ----------------------------------------------
def Plot_Result(
    obj: Union["_Simu", "Mesh"],
    result: Union[str, _types.FloatArray],
    deformFactor: _types.Number = 0.0,
    coef: _types.Number = 1.0,
    nodeValues: bool = True,
    plotMesh: bool = False,
    edgecolor: str = "black",
    title: str = "",
    cmap: str = "jet",
    ncolors=256,
    clim=(None, None),
    colorbarIsClose=False,
    colorbarLabel="",
    ax: Optional[_types.Axes] = None,
    folder: str = "",
    filename: str = "",
) -> _types.Axes:
    """Plots a simulation's result.

    Parameters
    ----------
    obj : _Simu | Mesh
        simulation
    result : str | _types.FloatArray
        Result you want to display.
        Must be included in simu.Get_Results() or be a numpy array of size (Nn, Ne).
    deformFactor : float, optional
        factor used to display the deformed solution (0 means no deformations), default 0.0
    coef : float, optional
        coef to apply to the solution, by default 1.0
    nodeValues : bool, optional
        displays result to nodes otherwise displays it to elements, by default True
    plotMesh : bool, optional
        displays mesh, by default False
    edgecolor : str, optional
        Color used to plot the mesh, by default 'black'
    title: str, optional
        figure title, by default ""
    cmap: str, optional
        the color map used near the figure, by default "jet" \n
        ["jet", "seismic", "binary", "viridis"] -> https://matplotlib.org/stable/tutorials/colors/colormaps.html
    ncolors : int, optional
        number of colors for colorbar, by default 21
    clim : sequence[float], optional
        Two item color bar range for scalars. Defaults to minimum and maximum of scalars array. Example: (-1, 2), by default (None, None)
    colorbarIsClose : bool, optional
        color bar is displayed close to the figure, by default False
    colorbarLabel: str, optional
        colorbar label, by default ""
    ax: axis, optional
        Axis to use, default None, by default None
    folder : str, optional
        save folder, by default "".
    filename : str, optional
        filename, by default ""

    Returns
    -------
    Figure, Axis, colorbar
        fig, ax, cb
    """

    tic = Tic()

    simu, mesh, coordo, inDim = _Init_obj(obj, deformFactor)  # type: ignore
    plotDim = mesh.dim  # plot dimension

    # don't know how to display nodal values on lines
    nodeValues = False if plotDim == 1 else nodeValues

    # When mesh use 3D elements, results are displayed only on 2D elements.
    # To display values on 2D elements, we first need to know the values at 3D nodes.
    nodeValues = True if plotDim == 3 else nodeValues  # do not modify

    values = _Get_values(simu, mesh, result, nodeValues)

    values *= coef  # Apply coef to values

    # Builds boundary markers for the colorbar
    min, max = clim
    if min is None and max is None:
        if isinstance(result, str) and result == "damage":
            min = values.min() - 1e-12
            max = np.max([values.max() + 1e-12, 1])
            ticks = np.linspace(min, max, 11)
            # ticks = np.linspace(0,1,11) # ticks colorbar
        else:
            max = np.max(values) + 1e-12 if max is None else max
            min = np.min(values) - 1e-12 if min is None else min
            ticks = np.linspace(min, max, 11)
        levels = np.linspace(min, max, ncolors)
    else:
        ticks = np.linspace(min, max, 11)
        levels = np.linspace(min, max, ncolors)

    if ncolors != 256:
        norm = colors.BoundaryNorm(boundaries=levels, ncolors=256)
    else:
        norm = None

    if ax is not None:
        _Remove_colorbar(ax)
        ax.clear()
        fig = ax.figure
        # change the plot dimentsion if the given axes is in 3d
        inDim = 3 if ax.name == "3d" else inDim

    if inDim in [1, 2]:
        # Mesh contained in a 2D plane
        # Only designed for one element group!

        if ax is None:
            ax = Init_Axes()
            fig = ax.figure
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")

        # construct coordinates for each elements
        faces = mesh.groupElem.faces.ravel().tolist()
        connectFaces = mesh.connect[:, faces]
        elements_coordinates = coordo[connectFaces, :2]

        # Plot the mesh
        if plotMesh:
            if mesh.dim == 1:
                # mesh for 1D elements are points
                ax.plot(
                    *mesh.coord[:, :inDim].T, c=edgecolor, lw=0.1, marker=".", ls=""
                )
            else:
                # mesh for 2D elements are lines / segments
                pc = LineCollection(elements_coordinates, edgecolor=edgecolor, lw=0.5)  # type: ignore [arg-type]
                ax.add_collection(pc)

        # Plot element values
        if mesh.Ne == len(values):
            if mesh.dim == 1:
                pc = LineCollection(elements_coordinates, lw=1.5, cmap=cmap, norm=norm)  # type: ignore [arg-type]
            else:
                pc = PolyCollection(elements_coordinates, lw=0.5, cmap=cmap, norm=norm)  # type: ignore
            pc.set_clim(min, max)
            pc.set_array(values)
            ax.add_collection(pc)
            # ticks = None if ncolors != 11 else ticks

        # Plot node values
        elif mesh.Nn == len(values):
            # retrieves triangles from each face to use the trisurf function
            triangles = mesh.groupElem.triangles
            connectTri = np.reshape(mesh.connect[:, triangles], (-1, 3))
            # tripcolor, tricontour, tricontourf
            pc = ax.tricontourf(  # type: ignore [call-overload]
                coordo[:, 0],
                coordo[:, 1],
                connectTri,
                values,
                levels,
                cmap=cmap,
                vmin=min,
                vmax=max,
            )

        # scale the axis
        ax.autoscale()
        ax.axis("equal")

        if colorbarIsClose:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.1)
            # # cax = divider.add_auto_adjustable_area(use_axes=ax, pad=0.1, adjust_dirs='right')
        else:
            cax = None

        cb = plt.colorbar(pc, ax=ax, cax=cax, ticks=ticks)

    elif inDim == 3:
        # If the mesh is a 3D mesh, only the 2D elements of the mesh will be displayed.
        # A 3D mesh can contain several types of 2D element.
        # For example, when PRISM6 -> TRI3 and QUAD4 at the same time

        plotDim = 2 if plotDim == 3 else plotDim

        if ax is None:
            ax = Init_Axes(3)
            fig = ax.figure
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_zlabel(r"$z$")  # type: ignore

        # construct the face connection matrix
        connectFaces = []  # type: ignore [assignment]
        groupElems = mesh.Get_list_groupElem(plotDim)
        list_faces = _Get_list_faces(mesh, plotDim)
        for groupElem, faces in zip(groupElems, list_faces):
            connectFaces.extend(groupElem.connect[:, faces])  # type: ignore [attr-defined]
        connectFaces = np.asarray(connectFaces, dtype=int)

        elements_coordinates = coordo[connectFaces, :3]

        if nodeValues:
            # If the result is stored at nodes, we'll average the node values over the element.
            facesValues = []
            # for each group of elements, we'll calculate the value to be displayed on each element
            for groupElem in groupElems:
                values_loc = values[groupElem.connect]
                values_e = np.mean(values_loc, axis=1)
                facesValues.extend(values_e)
            facesValues = np.array(facesValues)  # type: ignore [assignment]
        else:
            facesValues = values  # type: ignore [assignment]

        # update max and min
        max = np.max([facesValues.max(), max])  # type: ignore [attr-defined]
        min = np.min([facesValues.min(), min])  # type: ignore [attr-defined]

        # Display result with or without the mesh
        if plotMesh:
            if plotDim == 1:
                ax.plot(*mesh.coordGlob.T, c="black", lw=0.1, marker=".", ls="")
                pc = Line3DCollection(
                    elements_coordinates, cmap=cmap, zorder=0, norm=norm
                )
            elif plotDim == 2:
                pc = Poly3DCollection(
                    elements_coordinates,
                    edgecolor="black",
                    linewidths=0.5,
                    cmap=cmap,
                    zorder=0,
                    norm=norm,
                )
        else:
            if plotDim == 1:
                pc = Line3DCollection(
                    elements_coordinates, cmap=cmap, zorder=0, norm=norm
                )
            if plotDim == 2:
                pc = Poly3DCollection(
                    elements_coordinates, cmap=cmap, zorder=0, norm=norm
                )

        # Colors are applied to the faces
        pc.set_array(facesValues)
        pc.set_clim(min, max)
        ax.add_collection3d(pc)
        # We set the colorbar limits and display it
        cb = fig.colorbar(pc, ax=ax, ticks=ticks)

        # Change axis scale
        _Axis_equal_3D(ax, mesh.coordGlob)

    cb.set_label(colorbarLabel)

    # Title
    # if no title has been entered, the constructed title is used
    if title == "" and isinstance(result, str):
        optionTex = result
        if isinstance(result, str):
            if result == "damage":
                optionTex = r"\phi"
            elif result == "thermal":
                optionTex = "T"
            elif "S" in result and ("_norm" not in result):
                optionFin = result.split("S")[-1]
                optionTex = f"\sigma_{'{' + optionFin + '}'}"
            elif "E" in result:
                optionFin = result.split("E")[-1]
                optionTex = f"\epsilon_{'{' + optionFin + '}'}"

        # Specify whether values are on nodes or elements
        if nodeValues:
            # loc = "^{n}"
            loc = ""
        else:
            loc = "^{e}"
        title = optionTex + loc
        ax.set_title(rf"${title}$")
    else:
        ax.set_title(f"{title}")

    tic.Tac("Display", "Plot_Result")

    # If the folder has been filled in, save the figure.
    if folder != "":
        if filename == "":
            filename = result  # type: ignore [assignment]
        Save_fig(folder, filename, transparent=False)

    return ax


def Plot_Mesh(
    obj: Union["_Simu", "Mesh"],
    deformFactor: float = 0.0,
    alpha: float = 1.0,
    facecolors: str = "c",
    edgecolor: str = "black",
    lw: float = 0.5,
    ax: Optional[_types.Axes] = None,
    folder: str = "",
    title: str = "",
) -> _types.Axes:
    """Plots the mesh.

    Parameters
    ----------
    obj : _Simu | Mesh | _GroupElem
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
    ax: _types.Axes, optional
        Axis to use, default None
    folder : str, optional
        save folder, default "".
    title: str, optional
        figure title, by default ""

    Returns
    -------
    _types.Axes
    """

    tic = Tic()

    simu, mesh, coordo, inDim = _Init_obj(obj, deformFactor)

    if ax is not None:
        inDim = 3 if ax.name == "3d" else inDim

    deformFactor = 0 if simu is None else np.abs(deformFactor)

    # Dimensions of displayed elements
    dimElem = mesh.dim
    # If the mesh is a 3D mesh, only the 2D elements of the mesh will be displayed.
    if dimElem == 3:
        dimElem = 2

    # construct the connection matrix for the faces
    list_groupElem = mesh.Get_list_groupElem(dimElem)
    list_faces = _Get_list_faces(mesh, dimElem)
    connectFaces: list = []
    for groupElem, faces in zip(list_groupElem, list_faces):
        connectFaces.extend(groupElem.connect[:, faces])
    connectFaces = np.asarray(connectFaces, dtype=int)  # type: ignore [assignment]

    # faces coordinates
    coordFacesDef = coordo[connectFaces, :inDim]
    coordFaces = mesh.coordGlob[connectFaces, :inDim]

    if title == "":
        title = f"{mesh.elemType} : Ne = {mesh.Ne}, Nn = {mesh.Nn}"

    if inDim in [1, 2]:
        # in 2d space

        if ax is None:
            ax = Init_Axes()
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_title(title)

        if deformFactor > 0:
            # Deformed mesh
            pc = LineCollection(
                coordFacesDef,  # type: ignore
                edgecolor="red",
                lw=lw,
                antialiaseds=True,
                zorder=1,  # type: ignore
            )
            ax.add_collection(pc)
            # Overlay undeformed and deformed mesh
            # Undeformed mesh
            pc = LineCollection(
                coordFaces,  # type: ignore
                edgecolor=edgecolor,
                lw=lw,
                antialiaseds=True,
                zorder=1,  # type: ignore
            )
            ax.add_collection(pc)
        else:
            # Undeformed mesh
            pc = LineCollection(coordFaces, edgecolor=edgecolor, lw=lw, zorder=1)  # type: ignore
            ax.add_collection(pc)
            if alpha > 0:
                pc = PolyCollection(
                    coordFaces,  # type: ignore
                    facecolors=facecolors,
                    edgecolor=edgecolor,
                    lw=lw,
                    zorder=1,
                    alpha=alpha,
                )
                ax.add_collection(pc)

        if mesh.dim == 1:
            # nodes
            ax.plot(*mesh.coordGlob[:, :2].T, c="black", lw=lw, marker=".", ls="")
            if deformFactor > 0:
                ax.plot(*coordo[:, :2].T, c="red", lw=lw, marker=".", ls="")

        ax.autoscale()
        if ax.name != "3d":
            ax.axis("equal")

    elif inDim == 3:
        # in 3d space

        if ax is None:
            ax = Init_Axes(3)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_zlabel(r"$z$")  # type: ignore
            ax.set_title(title)

        if deformFactor > 0:
            # Displays only 1D or 2D elements, depending on the mesh type
            if dimElem > 1:
                # Deformed 2D mesh
                pcDef = Poly3DCollection(
                    coordFacesDef, edgecolor="red", linewidths=0.5, alpha=0, zorder=0
                )
                ax.add_collection3d(pcDef)  # type: ignore
                # Overlay the two meshes
                # Undeformed mesh
                # ax.scatter(x,y,z, linewidth=0, alpha=0)
                pcNonDef = Poly3DCollection(
                    coordFaces, edgecolor=edgecolor, linewidths=0.5, alpha=0, zorder=0
                )
                ax.add_collection3d(pcNonDef)  # type: ignore

            else:
                # Deformed mesh
                pc = Line3DCollection(
                    coordFacesDef, edgecolor="red", lw=lw, antialiaseds=True, zorder=0
                )
                ax.add_collection3d(pc)  # type: ignore
                # Overlay undeformed and deformed mesh
                # Undeformed mesh
                pc = Line3DCollection(
                    coordFaces, edgecolor=edgecolor, lw=lw, antialiaseds=True, zorder=0
                )
                ax.add_collection3d(pc)  # type: ignore
                # nodes
                ax.plot(*mesh.coordGlob.T, c="black", lw=lw, marker=".", ls="")
                ax.plot(*coordo.T, c="red", lw=lw, marker=".", ls="")

        else:
            # Undeformed mesh
            # Displays only 1D or 2D elements, depending on the mesh type
            if dimElem > 1:
                pc = Poly3DCollection(
                    coordFaces,
                    facecolors=facecolors,
                    edgecolor=edgecolor,
                    linewidths=0.5,
                    alpha=alpha,
                    zorder=0,
                )
            else:
                pc = Line3DCollection(
                    coordFaces, edgecolor=edgecolor, lw=lw, antialiaseds=True, zorder=0
                )
                ax.plot(*coordo.T, c="black", lw=lw, marker=".", ls="")
            ax.add_collection3d(pc, zs=0, zdir="z")  # type: ignore

        _Axis_equal_3D(ax, coordo)  # type: ignore

    tic.Tac("Display", "Plot_Mesh")

    if folder != "":
        Save_fig(folder, "mesh")

    return ax  # type: ignore


def Plot_Nodes(
    obj,
    nodes: Optional[_types.IntArray] = None,
    showId=False,
    marker=".",
    c="red",
    ax: Optional[_types.Axes] = None,
) -> _types.Axes:
    """Plots the mesh's nodes.

    Parameters
    ----------
    obj : _Simu | Mesh | _GroupElem
        object containing the mesh
    nodes : _types.IntArray, optional
        nodes to display, default []
    showId : bool, optional
        display numbers, default False
    marker : str, optional
        marker type (matplotlib.markers), default '.'
    c : str, optional
        color, default 'red'
    ax : _types.Axes, optional
        Axis to use, default None, default None

    Returns
    -------
    _types.Axes
    """

    tic = Tic()

    mesh = _Init_obj(obj)[1]

    inDim = mesh.inDim

    if ax is None:
        ax = Init_Axes(inDim)
        ax.set_title("")
    else:
        inDim = 3 if ax.name == "3d" else inDim

    if nodes is None:
        nodes = mesh.nodes
    else:
        nodes = np.asarray(list(set(np.ravel(nodes))))

    coordo = mesh.coordGlob

    if inDim == 2:
        ax.plot(*coordo[nodes, :2].T, ls="", marker=marker, c=c, zorder=2.5)
        if showId:
            [ax.text(*coordo[node, :2].T, str(node), c=c) for node in nodes]  # type: ignore [call-arg]
        ax.axis("equal")
    elif inDim == 3:
        ax.plot(*coordo[nodes].T, ls="", marker=marker, c=c, zorder=2.5)
        if showId:
            [ax.text(*coordo[node].T, str(node), c=c) for node in nodes]  # type: ignore [call-arg]
        _Axis_equal_3D(ax, coordo)

    tic.Tac("Display", "Plot_Nodes")

    return ax


def Plot_Elements(
    obj,
    nodes=[],
    dimElem: Optional[int] = None,
    showId=False,
    alpha=1.0,
    c="red",
    edgecolor="black",
    ax: Optional[_types.Axes] = None,
) -> _types.Axes:
    """Plots the mesh's elements corresponding to the given nodes.

    Parameters
    ----------
    obj : _Simu | Mesh | _GroupElem
        object containing the mesh
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
    ax : _types.Axes, optional
        Axis to use, default None

    Returns
    -------
    _types.Axes
    """

    tic = Tic()

    mesh = _Init_obj(obj)[1]

    inDim = mesh.inDim

    if dimElem is None:
        dimElem = 2 if mesh.inDim == 3 else mesh.dim

    # list of element group associated with the dimension
    list_groupElem = mesh.Get_list_groupElem(dimElem)[:1]
    if len(list_groupElem) == 0:
        return None  # type: ignore

    if ax is None:
        ax = Init_Axes(inDim)
    else:
        inDim = 3 if ax.name == "3d" else inDim

    # for each group elem
    for groupElem in list_groupElem:
        # get the elements associated with the nodes
        if len(nodes) > 0:
            elements = groupElem.Get_Elements_Nodes(nodes)
        else:
            elements = np.arange(groupElem.Ne)

        if elements.size == 0:
            continue

        # Construct the faces coordinates
        connect_e = groupElem.connect  # connect
        coord_n = groupElem.coordGlob[:, : mesh.inDim]  # global coordinates
        faces = groupElem.faces.ravel().tolist()  # faces indexes
        coordFaces_e = coord_n[connect_e[:, faces]]  # faces coordinates
        coordFaces = coordFaces_e[elements]

        # center coordinates for each elements
        center_e = np.mean(coordFaces_e, axis=1)

        # plot the entities associated with the tag
        if mesh.inDim in [1, 2]:
            if groupElem.dim == 1:
                # 1D elements
                pc = LineCollection(coordFaces, edgecolor=c, lw=1, zorder=2)
            else:
                # 2D elements
                pc = PolyCollection(  # type: ignore [assignment]
                    coordFaces,
                    facecolors=c,
                    edgecolor=edgecolor,
                    lw=0.5,
                    alpha=alpha,
                    zorder=2,
                )
            ax.add_collection(pc)
        elif mesh.inDim == 3:
            # 2D elements
            pc = Poly3DCollection(
                coordFaces,
                facecolors=c,
                edgecolor=edgecolor,
                linewidths=0.5,
                alpha=alpha,
                zorder=2,
            )
            ax.add_collection3d(pc, zdir="z")
        if showId:
            # plot elements id's
            [
                ax.text(  # type: ignore [call-arg]
                    *center_e[element], element, zorder=25, ha="center", va="center"
                )
                for element in elements
            ]

    tic.Tac("Display", "Plot_Elements")

    if inDim < 3:
        ax.axis("equal")
    else:
        _Axis_equal_3D(ax, mesh.coord)

    return ax


def Plot_BoundaryConditions(simu, ax: Optional[_types.Axes] = None) -> _types.Axes:
    """Plots simulation's boundary conditions.

    Parameters
    ----------
    simu : _Simu
        simulation
    ax : _types.Axes, optional
        Axis to use, default None

    Returns
    -------
    _types.Axes
    """

    tic = Tic()

    simu = _Init_obj(simu)[0]

    coord = simu.mesh.coordGlob

    # get Dirichlet and Neumann boundary conditions
    dirchlets = simu.Bc_Dirichlet
    BoundaryConditions = dirchlets
    neumanns = simu.Bc_Neuman
    BoundaryConditions.extend(neumanns)
    displays = (
        simu.Bc_Display
    )  # boundary conditions for display used for lagrangian boundary conditions
    BoundaryConditions.extend(displays)

    if ax is None:
        ax = Plot_Elements(simu.mesh, dimElem=1, c="k")
        ax.set_title("Boundary conditions")

    for bc in BoundaryConditions:
        problemType = bc.problemType
        dofsValues = bc.dofsValues
        unknowns = bc.unknowns
        nDir = len(unknowns)
        nodes = list(set(list(bc.nodes)))
        description = bc.description

        if problemType in ["damage", "thermal"]:
            marker = "o"
        elif problemType in ["elastic", "beam", "hyperelastic"]:
            # get values for each direction
            sum = np.sum(dofsValues.reshape(-1, nDir), axis=0)
            values = np.round(sum, 2)
            # values will be use to choose the marker
            if len(unknowns) == 1:
                sign = np.sign(values[0])
                if unknowns[0] == "x":
                    if sign == -1:
                        marker = "<"
                    else:
                        marker = ">"
                elif unknowns[0] == "y":
                    if sign == -1:
                        marker = "v"
                    else:
                        marker = "^"
                elif unknowns[0] == "z":
                    marker = "d"
            elif len(unknowns) == 2:
                if "Connection" in description:
                    marker = "o"
                else:
                    marker = "X"
            elif len(unknowns) > 2:
                marker = "s"
        else:
            raise Exception(f"{problemType} is not implemented here")

        # Title
        unknowns_str = str(unknowns).replace("'", "")
        title = f"{description} {unknowns_str}"

        lw = 0
        if len(nodes) == simu.mesh.Nn:
            ax.plot(
                *coord[:, : simu.mesh.inDim].mean(0).T,
                marker=marker,
                lw=lw * 5,
                label=title,
                zorder=2.5,
                ls="",
            )
        else:
            ax.plot(
                *coord[nodes, : simu.mesh.inDim].T,
                marker=marker,
                lw=lw,
                label=title,
                zorder=2.5,
                ls="",
            )

    ax.legend()

    tic.Tac("Display", "Plot_BoundaryConditions")

    return ax


def Plot_Tags(
    obj, showId=False, folder="", alpha=1.0, ax: Optional[_types.Axes] = None
) -> _types.Axes:
    """Plots the mesh's elements tags (from 2d elements to points) but do not plot the 3d elements tags.

    Parameters
    ----------
    obj : _Simu | Mesh | _GroupElem
        object containing the mesh
    showId : bool, optional
        shows tags, by default False
    folder : str, optional
        saves folder, by default ""
    alpha : float, optional
        transparency, by default 1.0
    ax : _types.Axes, optional
        Axis to use, default None

    Returns
    -------
    _types.Axes
    """

    tic = Tic()

    __, mesh, coordo, inDim = _Init_obj(obj)

    # check if there is available tags in the mesh
    nTtags = [
        np.max([len(groupElem.nodeTags), len(groupElem.elementTags)])
        for groupElem in mesh.dict_groupElem.values()
    ]
    if np.max(nTtags) == 0:
        MyPrintError(
            "There is no tags available in the mesh, so don't forget to use the '_Set_PhysicalGroups()' function before meshing your geometry with in the gmsh interface."
        )
        return None  # type: ignore [return-value]

    if ax is None:
        if mesh.inDim <= 2:
            ax = Init_Axes()
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
        else:
            ax = Init_Axes(3)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_zlabel(r"$z$")
    else:
        inDim = 3 if ax.name == "3d" else inDim

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
        coordo = groupElem.coordGlob[:, :inDim]
        center_e = np.mean(coordo[groupElem.connect], axis=1)  # center of each elements
        faces_coordinates = coordo[
            groupElem.connect[:, groupElem.faces.ravel().tolist()]
        ]

        for tag_e in tags_e:
            if "nodes" in tag_e:
                pass

            nodes = groupElem.Get_Nodes_Tag(tag_e)
            elements = groupElem.Get_Elements_Tag(tag_e)
            if len(elements) == 0:
                continue

            coord_faces = faces_coordinates[elements]

            # Assign color
            if groupElem.dim in [0, 1]:
                color = "black"
            else:
                color = "tab:blue"

            x_e = np.mean(center_e[elements, 0])
            y_e = np.mean(center_e[elements, 1])
            if inDim == 3:
                z_e = np.mean(center_e[elements, 2])

            x_n = coordo[nodes, 0]
            y_n = coordo[nodes, 1]
            if inDim == 3:
                z_n = coordo[nodes, 2]

            if inDim in [1, 2]:
                # in 2D space
                if len(nodes) > 0:
                    # lines or surfaces
                    if dim == 0:
                        # plot points
                        collections.append(
                            ax.plot(
                                x_n,
                                y_n,
                                c="black",
                                marker=".",
                                zorder=2,
                                label=tag_e,
                                lw=2,
                                ls="",
                            )
                        )
                    elif dim == 1:
                        # plot lines
                        pc = LineCollection(
                            coord_faces,
                            lw=1.5,
                            edgecolor="black",
                            alpha=1,
                            label=tag_e,  # type: ignore
                        )
                        collections.append(ax.add_collection(pc))
                    else:
                        # plot surfaces
                        pc = PolyCollection(
                            coord_faces,  # type: ignore
                            facecolors=color,
                            label=tag_e,
                            edgecolor=color,
                            alpha=alpha,
                        )
                        collections.append(ax.add_collection(pc))
                else:
                    # points
                    ax.plot(x_n, y_n, c="black", marker=".", zorder=2, ls="")

                if showId:
                    # plot the tag on the center of the element
                    ax.text(x_e, y_e, tag_e, zorder=25)

            else:
                # in 3D space
                if len(nodes) > 0:
                    # lines or surfaces
                    if dim == 0:
                        # plot points
                        collections.append(
                            ax.scatter(
                                x_n,
                                y_n,
                                z_n,
                                c="black",
                                marker=".",
                                zorder=2,
                                label=tag_e,
                                lw=2,
                                zdir="z",
                            )
                        )
                    elif dim == 1:
                        # plot lines
                        pc = Line3DCollection(
                            coord_faces, lw=1.5, edgecolor="black", alpha=1, label=tag_e
                        )
                        # collections.append(ax.add_collection3d(pc, zs=z_e, zdir='z'))
                        collections.append(ax.add_collection3d(pc, zdir="z"))
                    elif dim == 2:
                        # plot surfaces
                        pc = Poly3DCollection(
                            coord_faces,
                            lw=0,
                            alpha=alpha,
                            facecolors=color,
                            label=tag_e,
                        )
                        pc._facecolors2d = color  # type: ignore [attr-defined]
                        pc._edgecolors2d = color  # type: ignore [attr-defined]
                        collections.append(ax.add_collection3d(pc, zdir="z"))
                else:
                    collections.append(
                        ax.scatter(
                            x_n, y_n, z_n, c="black", marker=".", zorder=2, label=tag_e
                        )
                    )

                if showId:
                    ax.text(x_e, y_e, z_e, tag_e, zorder=25)  # type: ignore [arg-type]

    if inDim in [1, 2]:
        ax.autoscale()
        ax.axis("equal")
    else:
        _Axis_equal_3D(ax, coordo)

    tic.Tac("Display", "Plot_Tags")

    if folder != "":
        Save_fig(folder, "geom")

    __Annotation_Event(collections, ax.figure, ax)

    return ax


def __Annotation_Event(
    collections: list, fig: Union[plt.Figure, Any], ax: _types.Axes
) -> None:
    """Creates an event to display the element tag currently active under the mouse at the bottom of the figure."""

    def Set_Message(collection, event):
        if isinstance(collection, list):
            return
        if collection.contains(event)[0]:
            toolbar = ax.figure.canvas.toolbar
            coordo = ax.format_coord(event.xdata, event.ydata)
            toolbar.set_message(f"{collection.get_label()} : {coordo}")
            # TODO get surface or length ?
            # change the title instead the toolbar message ?

    def hover(event):
        if event.inaxes == ax:
            # TODO is there a way to access the collection containing the event directly?
            [Set_Message(collection, event) for collection in collections]

    fig.canvas.mpl_connect("motion_notify_event", hover)


# ----------------------------------------------
# Plot 1D
# ----------------------------------------------
def Plot_Force_Displacement(
    force: _types.FloatArray,
    displacement: _types.FloatArray,
    xlabel="u",
    ylabel="f",
    folder="",
    ax: Optional[_types.Axes] = None,
) -> tuple[plt.Figure, _types.Axes]:  # type: ignore
    """Plots the force displacement curve.

    Parameters
    ----------
    force : _types.FloatArray
        array of values for force
    displacement : _types.FloatArray
        array of values for displacements
    xlabel : str, optional
        x-axis title, by default 'u'.
    ylabel : str, optional
        y-axis title, by default 'f' folder : str, optional
    folder : str, optional
        save folder, by default ""
    ax : _types.Axes, optional
        ax in which to plot the figure, by default None

    Returns
    -------
    tuple[plt.Figure, _types.Axes]
        returns figure and ax
    """

    if isinstance(ax, _types.Axes):  # type: ignore
        fig = ax.figure
        ax.clear()
    else:
        ax = Init_Axes()
        fig = ax.figure

    ax.plot(np.abs(displacement), np.abs(force), c="blue")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()

    if folder != "":
        Save_fig(folder, "force-displacement")

    return fig, ax  # type: ignore [return-value]


def Plot_Energy(
    simu: "_Simu",
    load: _types.FloatArray = np.empty(0),
    displacement: _types.FloatArray = np.empty(0),
    plotSolMax: bool = True,
    N: int = 200,
    folder: str = "",
) -> None:
    """Plots the energy for each iteration.

    Parameters
    ----------
    simu : _Simu
        simulation
    load : _types.FloatArray, optional
        array of values, by default np.array([])
    displacement : _types.FloatArray, optional
        array of values, by default np.array([])
    plotSolMax : bool, optional
        displays the evolution of the maximul solution over iterations. (max damage for damage simulation), by default True
    N : int, optional
        number of iterations for which energy will be calculated, by default 200
    folder : str, optional
        save folder, by default ""
    """

    simu = _Init_obj(simu)[0]  # type: ignore [assignment]

    # First we check whether the simulation can calculate energies
    if len(simu.Results_dict_Energy()) == 0:
        print("This simulation don't calculate energies.")
        return

    # Check whether it is possible to plot the force-displacement curve
    pltLoad = len(load) == len(displacement) and len(load) > 0

    # For each displacement increment we calculate the energy
    tic = Tic()

    # recover simulation results
    Niter = len(simu.results)
    if len(load) > 0:
        ecart = np.abs(Niter - len(load))
        if ecart != 0:
            Niter -= ecart
    step = np.max([1, Niter // N])
    iterations = np.arange(0, Niter, step)

    list_dict_Energy: list[dict[str, float]] = []
    times = []
    if plotSolMax:
        listSolMax: list[float] = []

    # activate the first iteration
    simu.Set_Iter(0, resetAll=True)

    for i, iteration in enumerate(iterations):
        # Update simulation at iteration i
        simu.Set_Iter(iteration)

        if plotSolMax:
            listSolMax.append(simu._Get_u_n(simu.problemType).max())  # type: ignore

        list_dict_Energy.append(simu.Results_dict_Energy())

        time = tic.Tac("PostProcessing", "Calc Energy", False)
        times.append(time)

        rmTime = Tic.Get_Remaining_Time(i, iterations.size - 1, time)

        print(f"Calc Energy {i}/{iterations.size - 1} {rmTime}     ", end="\r")
    print("\n")

    # Figure construction
    nrows = 1
    if plotSolMax:
        nrows += 1
    if pltLoad:
        nrows += 1
    axs: list[_types.Axes] = plt.subplots(nrows, 1, sharex=True)[1]

    iter_rows = iter(np.arange(nrows))

    # Retrieve the axis to be used for x-axes
    if len(displacement) > 0:
        listX = displacement[iterations]
        xlabel = "displacement"
    else:
        listX = iterations
        xlabel = "iter"

    # Transform list_dict_energy into a dataframe
    df = pd.DataFrame(list_dict_Energy)

    row: int = next(iter_rows)
    # For each energy, we plot the values
    for energie_str in df.columns:
        valeurs = df[energie_str].values
        axs[row].plot(listX, valeurs, label=energie_str)
    axs[row].legend()
    axs[row].grid()

    if plotSolMax:
        # plot max solution
        row = next(iter_rows)
        axs[row].plot(listX, listSolMax)
        axs[row].set_ylabel(r"$max(u_n)$")
        axs[row].grid()

    if pltLoad:
        # plot the loading
        row = next(iter_rows)
        axs[row].plot(listX, np.abs(load[iterations]) * 1e-3)
        axs[row].set_ylabel("load")
        axs[row].grid()

    axs[-1].set_xlabel(xlabel)

    if folder != "":
        Save_fig(folder, "Energy")

    tic.Tac("PostProcessing", "Calc Energy", False)


def Plot_Iter_Summary(simu, folder="", iterMin=None, iterMax=None) -> None:
    """Plots a summary of iterations between iterMin and iterMax.

    Parameters
    ----------
    simu : _Simu
        Simulation
    folder : str, optional
        backup folder, by default ""
    iterMin : int, optional
        lower bound, by default None
    iterMax : int, optional
        upper bound, by default None
    """

    simu = _Init_obj(simu)[0]

    # Recover simulation results
    iterations, list_label_values = simu.Results_Iter_Summary()

    if iterMax is None:
        iterMax = np.max(iterations)

    if iterMin is None:
        iterMin = np.min(iterations)

    selectionIndex = list(
        filter(
            lambda iterations: iterations >= iterMin and iterations <= iterMax,
            iterations,
        )
    )
    iterations = np.asarray(iterations)[selectionIndex]

    nbGraph = len(list_label_values)

    axs: list[_types.Axes] = plt.subplots(nrows=nbGraph, sharex=True)[1]

    for ax, label_values in zip(axs, list_label_values):
        ax.grid()
        ax.plot(iterations, label_values[1][iterations], color="blue")
        ax.set_ylabel(label_values[0])

    ax.set_xlabel("iterations")

    if folder != "":
        Save_fig(folder, "resumeConvergence")


# ----------------------------------------------
# Animation
# ----------------------------------------------
def Movie_Simu(
    simu,
    result: str,
    folder: str,
    filename="video.gif",
    N: int = 200,
    deformFactor=0.0,
    coef=1.0,
    nodeValues=True,
    plotMesh=False,
    edgecolor="black",
    fps=30,
    **kwargs,
) -> None:
    """Generates a movie from a simulation's result.

    Parameters
    ----------
    simu : _Simu
        simulation
    result : str
        result that you want to plot
    folder : str
        folder where you want to save the video
    filename : str, optional
        filename of the video with the extension (gif, mp4), by default 'video.gif'
    N : int, optional
        Maximal number of iterations displayed, by default 200
    deformFactor : int, optional
        deformation factor, by default 0.0
    coef : float, optional
        Coef to apply to the solution, by default 1.0
    nodeValues : bool, optional
        Displays result to nodes otherwise displays it to elements, by default True
    plotMesh : bool, optional
        Plot the mesh, by default False
    edgecolor : str, optional
        Color used to plot the mesh, by default 'black'
    fps : int, optional
        frames per second, by default 30
    """

    simu = _Init_obj(simu)[0]

    if simu is None:
        MyPrintError("Must give a simulation.")
        return

    Niter = len(simu.results)
    step = np.max([1, Niter // N])
    iterations = np.arange(0, Niter, step)

    ax = Init_Axes(simu.mesh.inDim)
    fig = ax.figure

    # activate the first iteration
    simu.Set_Iter(0, resetAll=True)

    def DoAnim(fig: plt.Figure, i):  # type: ignore
        simu.Set_Iter(iterations[i])
        ax = fig.axes[0]
        Plot_Result(
            simu,
            result,
            deformFactor,
            coef,
            nodeValues,
            plotMesh,
            edgecolor,
            ax=ax,
            **kwargs,
        )
        ax.set_title(f"{result} {iterations[i]:d}/{Niter - 1:d}")

    Movie_func(DoAnim, fig, iterations.size, folder, filename, fps)


def Movie_func(
    func: Callable[[plt.Figure, int], None],
    fig: Union[plt.Figure, Any],
    N: int,
    folder: str,
    filename="video.gif",
    fps=30,
    dpi=200,
    show=True,
):
    """Generates the movie for the specified function.\n
    This function will peform a loop in range(N).

    Parameters
    ----------
    func : Callable[[plt.Figure, int], None]
        The function that will use in first argument the plotter and in second argument the iter step such that.\n
        def func(fig, i) -> None
    fig : Figure
        Figure used to make the video
    N : int
        number of iteration
    folder : str
        folder where you want to save the video
    filename : str, optional
        filename of the video with the extension (eg. .gif, .mp4), by default 'video.gif'
    fps : int, optional
        frames per second, by default 30
    dpi: int, optional
        Dots per Inch, by default 200
    show: bool, optional
        shows the movie, by default True
    """

    # Name of the video in the folder where the folder is communicated
    filename = Folder.Join(folder, filename)

    if not Folder.Exists(folder):
        Folder.os.makedirs(folder)

    writer = animation.FFMpegWriter(fps)
    with writer.saving(fig, filename, dpi):  # type: ignore [arg-type]
        tic = Tic()
        for i in range(N):
            func(fig, i)  # type: ignore [arg-type]

            if show:
                plt.pause(1e-12)

            writer.grab_frame()

            time = tic.Tac("Display", "Movie_func", False)

            rmTime = Tic.Get_Remaining_Time(i, N - 1, time)

            print(f"Make_Movie {i}/{N - 1} {rmTime}    ", end="\r")


# ----------------------------------------------
# Functions
# ----------------------------------------------


def Save_fig(
    folder: str, filename: str, transparent=False, extension="pdf", dpi="figure"
) -> None:
    """Saves the current figure.

    Parameters
    ----------
    folder : str
        save folder
    filename : str
        filename
    transparent : bool, optional
        transparent background, by default False
    extension : str, optional
        extension, by default 'pdf', [pdf, png]
    dpi : str, optional
        dpi, by default 'figure'
    """

    if folder == "":
        return

    # the filename must not contain these characters
    for char in ["NUL", "\ ", ",", "/", ":", "*", "?", "<", ">", "|"]:
        filename = filename.replace(char, "")

    path = Folder.Join(folder, filename + "." + extension)

    if not Folder.Exists(folder):
        Folder.os.makedirs(folder)

    tic = Tic()

    plt.savefig(path, dpi=dpi, transparent=transparent, bbox_inches="tight")

    tic.Tac("Display", "Save figure")


def _Get_list_faces(mesh, dimElem: int) -> list[list[int]]:
    """Returns a list of faces for each element group of dimension dimElem.\n
    Faces is a list of index used to construct/plot a faces.\n
    You can go check their values for each groupElem in `EasyFEA/fem/elems/` folder"""

    mesh = _Init_obj(mesh)[1]

    list_faces: list[list[int]] = []  # list of faces
    list_len: list[int] = []  # list that store the size for each faces

    # get faces and nodes per element for each element group
    for groupElem in mesh.Get_list_groupElem(dimElem):
        list_faces.append(groupElem.faces.ravel().tolist())
        list_len.append(groupElem.faces.size)

    # make sure that faces in list_faces are at the same length
    max_len = np.max(list_len)
    # this loop make sure that faces in list_faces get the same length
    for f, faces in enumerate(list_faces.copy()):
        repeat = max_len - len(faces)
        if repeat > 0:
            faces.extend([faces[0]] * repeat)
            list_faces[f] = faces

    return list_faces


def _Remove_colorbar(ax: _types.Axes) -> None:
    """Removes the current colorbar from the axis."""
    [
        collection.colorbar.remove()
        for collection in ax.collections
        if collection.colorbar is not None
    ]


def Init_Axes(dim: int = 2, elev=105, azim=-90) -> _types.Axes:
    """Initialize 2d or 3d axes."""
    if dim == 1 or dim == 2:
        ax = plt.subplots()[1]
    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.view_init(elev=elev, azim=azim)  # type: ignore [attr-defined]
    else:
        raise ValueError("dim error")
    return ax


def _Axis_equal_3D(ax: Axes3D, coord: _types.FloatArray) -> None:
    """Changes axis size for 3D display.\n
    Center the part and make the axes the right size.

    Parameters
    ----------
    ax : _types.Axes
        Axes in which figure will be created
    coord : _types.FloatArray
        mesh coordinates
    """

    # Change axis size
    xmin = np.min(coord[:, 0])
    xmax = np.max(coord[:, 0])
    ymin = np.min(coord[:, 1])
    ymax = np.max(coord[:, 1])
    zmin = np.min(coord[:, 2])
    zmax = np.max(coord[:, 2])

    maxRange = np.max(np.abs([xmin - xmax, ymin - ymax, zmin - zmax]))
    maxRange = maxRange * 0.55

    xmid = (xmax + xmin) / 2
    ymid = (ymax + ymin) / 2
    zmid = (zmax + zmin) / 2

    ax.set_xlim([xmid - maxRange, xmid + maxRange])
    ax.set_ylim([ymid - maxRange, ymid + maxRange])
    ax.set_zlim([zmid - maxRange, zmid + maxRange])
    ax.set_box_aspect([1, 1, 1])


# ----------------------------------------------
# Print in terminal
# ----------------------------------------------


class __Colors(str, Enum):
    blue = "\033[34m"
    cyan = "\033[36m"
    white = "\033[37m"
    green = "\033[32m"
    black = "\033[30m"
    red = "\033[31m"
    yellow = "\033[33m"
    magenta = "\033[35m"


class __Sytles(str, Enum):
    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    RESET = "\33[0m"


def MyPrint(
    text: str,
    color="cyan",
    bold=False,
    italic=False,
    underLine=False,
    end: str = "",
) -> str:
    dct = dict(map(lambda item: (item.name, item.value), __Colors))

    if color not in dct:
        return MyPrint(f"Color must be in {dct.keys()}", "red")

    else:
        formatedText = ""

        if bold:
            formatedText += __Sytles.BOLD
        if italic:
            formatedText += __Sytles.ITALIC
        if underLine:
            formatedText += __Sytles.UNDERLINE

        formatedText += dct[color] + str(text)

        formatedText += __Sytles.RESET

        print(formatedText, end=end)
        return formatedText


def MyPrintError(text: str) -> str:
    return MyPrint(text, "red")


def Section(text: str, verbosity=True) -> str:
    """Creates a new section in the terminal."""
    edges = "======================="

    lengthText = len(text)

    lengthTot = 45

    edges = "=" * int((lengthTot - lengthText) / 2)

    section = f"\n\n{edges} {text} {edges}\n"

    if verbosity:
        MyPrint(section)

    return section


def Clear() -> None:
    """Clears the terminal."""
    syst = platform.system()
    if syst in ["Linux", "Darwin"]:
        Folder.os.system("clear")
    elif syst == "Windows":
        Folder.os.system("cls")
