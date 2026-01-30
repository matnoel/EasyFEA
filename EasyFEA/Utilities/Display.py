# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing functions used to display simulations and meshes with matplotlib (https://matplotlib.org/)."""

from __future__ import annotations
import platform
from typing import Union, Callable, Optional, TYPE_CHECKING, Any
import numpy as np
from enum import Enum

# utilities
from . import Folder, Tic, _types

# simulations
from ..Simulations._simu import _Init_obj, _Get_values

if TYPE_CHECKING:
    from ..Simulations._simu import _Simu
    from ..FEM._mesh import Mesh
    from ..FEM._group_elem import _GroupElem

from ._requires import Create_requires_decorator

# Matplotlib: https://matplotlib.org/
try:
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.collections import PolyCollection, LineCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # use to do colorbarIsClose
    import matplotlib.animation as animation

    Axes = Union[plt.Axes, Axes3D]
except ImportError:
    pass
requires_matplotlib = Create_requires_decorator("matplotlib")

# Ideas: https://www.python-graph-gallery.com/

# fmt: off
# tab10_colors = [colors.rgb2hex(color) for color in plt.get_cmap("tab10").colors] 
tab10_colors = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
]
# tab20_colors = [colors.rgb2hex(color) for color in plt.get_cmap("tab20").colors] 
tab20_colors = [
    "#1f77b4","#aec7e8","#ff7f0e","#ffbb78","#2ca02c",
    "#98df8a","#d62728","#ff9896","#9467bd","#c5b0d5",
    "#8c564b","#c49c94","#e377c2","#f7b6d2","#7f7f7f",
    "#c7c7c7","#bcbd22","#dbdb8d","#17becf","#9edae5",
]
# fmt: on


# ----------------------------------------------
# Plot Simu or Mesh
# ----------------------------------------------
@requires_matplotlib
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
    ax: Optional[Axes] = None,
    folder: str = "",
    filename: str = "",
) -> Axes:
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
    _types.Axis
    """

    # TODO #21: regroup function by dimElem instead of inDim

    tic = Tic()

    simu, mesh, coord, inDim = _Init_obj(obj, deformFactor)  # type: ignore
    dimElem = mesh.dim  # Dimension of displayed elements
    groupElem = mesh.groupElem

    if dimElem == 1:
        # Don't know how to display nodal values on lines
        nodeValues = False  # do not modify
    elif dimElem == 3:
        # When mesh use 3D elements, results are displayed only on 2D elements.
        # To display values on 2D elements, we first need to know the values at 3D nodes.
        nodeValues = True  # do not modify
    else:
        nodeValues

    # Get values
    values = _Get_values(simu, mesh, result, nodeValues) * coef
    # Get colorbar properties
    ticks, levels, norm, min, max = __Get_colorbar_properties(
        clim, result, values, ncolors
    )

    ax, inDim = __Get_axis(ax, inDim)

    if inDim == 3:
        # If the mesh is a 3D mesh, only the 2D elements of the mesh will be displayed.
        # A 3D mesh can contain several types of 2D element.
        # For example PRISM6 mesh use TRI3 and QUAD4 at the same time
        dimElem = 2 if dimElem == 3 else dimElem

        if dimElem == 1:
            if plotMesh:
                ax.plot(*coord.T, c=edgecolor, lw=0.1, marker=".", ls="")
            vertices = coord[groupElem.connect[:, groupElem.segments[0]]]
            pc = Line3DCollection(vertices, cmap=cmap, zorder=0, norm=norm)

        else:
            # construct the surface connection matrix
            list_connect: list[_types.IntArray] = []
            list_groupElem = mesh.Get_list_groupElem(2)
            list_surfaces = _Get_list_surfaces(mesh, 2)
            for groupElem, surfaces in zip(list_groupElem, list_surfaces):
                list_connect.extend(groupElem.connect[:, surfaces])  # type: ignore [attr-defined]
            # get surfaces coordinates
            vertices = coord[list_connect]

            # Display result with or without the mesh
            edgecolor = edgecolor if plotMesh else None
            linewidths = 0.5 if plotMesh else None
            # concat params
            params = {
                "edgecolor": edgecolor,
                "linewidths": linewidths,
                "cmap": cmap,
                "zorder": 0,
                "norm": norm,
            }
            pc = Poly3DCollection(vertices, **params)

        # get elementValues
        if nodeValues:
            # If the result is stored at nodes, we'll average the node values over the element.
            elementValues = []
            # for each group of elements, we'll calculate the value to be displayed on each element
            for groupElem in mesh.Get_list_groupElem(dimElem):
                values_loc = values[groupElem.connect]
                values_e = np.mean(values_loc, axis=1)
                elementValues.extend(values_e)
            elementValues = np.array(elementValues)  # type: ignore [assignment]
        else:
            elementValues = values  # type: ignore [assignment]

        # Colors are applied to the faces
        pc.set_array(elementValues)
        pc.set_clim(
            np.min([elementValues.min(), min]),  # type: ignore [attr-defined]
            np.max([elementValues.max(), max]),  # type: ignore [attr-defined]
        )
        ax.add_collection3d(pc)
        # We set the colorbar limits and display it
        colorbar = plt.colorbar(pc, ax=ax, ticks=ticks)

        # Change axis scale
        _Axis_equal_3D(ax, mesh.coord)

    else:

        # get vertices
        if mesh.dim == 1:
            idx = groupElem.segments[0]
        else:
            idx = groupElem.surfaces[0]
        vertices = coord[groupElem.connect[:, idx], :2]

        # Plot the mesh
        if plotMesh:
            if mesh.dim == 1:
                # mesh for 1D elements are points
                ax.plot(*coord[:, :2].T, c=edgecolor, lw=0.1, marker=".", ls="")
            else:
                # mesh for 2D elements are lines / segments
                pc = LineCollection(vertices, edgecolor=edgecolor, lw=0.5)  # type: ignore [arg-type]
                ax.add_collection(pc)

        # Plot element values
        if mesh.Ne == len(values):
            if mesh.dim == 1:
                pc = LineCollection(vertices, lw=1.5, cmap=cmap, norm=norm)  # type: ignore [arg-type]
            else:
                pc = PolyCollection(vertices, lw=0.5, cmap=cmap, norm=norm)  # type: ignore
            pc.set_clim(min, max)
            pc.set_array(values)
            ax.add_collection(pc)

        # Plot node values
        elif mesh.Nn == len(values):
            # retrieves triangles from each face to use the trisurf function
            triangles = mesh.groupElem.triangles
            triangulation = np.reshape(mesh.connect[:, triangles], (-1, 3))
            # tripcolor, tricontour, tricontourf
            pc = ax.tricontourf(  # type: ignore [call-overload]
                *coord[:, :2].T,
                triangulation,
                values,
                levels,
                cmap=cmap,
                vmin=values.min(),
                vmax=values.max(),
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
        colorbar = plt.colorbar(pc, ax=ax, cax=cax, ticks=ticks)

    colorbar.set_label(colorbarLabel)

    # Title
    # if no title has been entered, the constructed title is used
    if title == "" and isinstance(result, str):
        ax.set_title(rf"${__Get_latex_title(result, nodeValues)}$")
    else:
        ax.set_title(title)

    tic.Tac("Display", "Plot_Result")

    # If the folder has been filled in, save the figure.
    if folder != "":
        if filename == "":
            filename = result  # type: ignore [assignment]
        Save_fig(folder, filename, transparent=False)

    return ax


@requires_matplotlib
def __Get_axis(ax: Union[plt.Axes, Axes3D, None], inDim: int):
    # init Axes
    if ax is None:
        ax = Init_Axes(3) if inDim == 3 else Init_Axes(2)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        if inDim == 3:
            ax.set_zlabel(r"$z$")  # type: ignore
    else:
        _Remove_colorbar(ax)
        # change the plot dimentsion if the given axes is in 3d
        inDim = 3 if ax.name == "3d" else inDim

    return ax, inDim


@requires_matplotlib
def __Get_colorbar_properties(
    clim: tuple[int, int],
    result: Union[str, np.ndarray],
    values: np.ndarray,
    ncolors: int,
):
    """Returns ticks, levels, norm"""
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

    return ticks, levels, norm, min, max


def __Get_latex_title(result, nodeValues=True) -> str:
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
    return title


@requires_matplotlib
def Plot_Mesh(
    obj: Union["_Simu", "Mesh"],
    deformFactor: float = 0.0,
    alpha: float = 1.0,
    facecolors: str = "c",
    edgecolor: str = "black",
    lw: float = 0.5,
    ax: Optional[Axes] = None,
    folder: str = "",
    title: str = "",
) -> Axes:
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
    ax: Axes, optional
        Axis to use, default None
    folder : str, optional
        save folder, default "".
    title: str, optional
        figure title, by default ""

    Returns
    -------
    Axes
    """

    tic = Tic()

    simu, mesh, coord, inDim = _Init_obj(obj, deformFactor)
    groupElem = mesh.groupElem

    if ax is not None:
        inDim = 3 if ax.name == "3d" else inDim

    deformFactor = 0 if simu is None else np.abs(deformFactor)

    # Dimension of displayed elements
    dimElem = mesh.dim
    # If the mesh is a 3D mesh, only the 2D elements of the mesh will be displayed.
    if dimElem == 3:
        dimElem = 2

    if title == "":
        title = f"{mesh.elemType}: Ne = {mesh.Ne}, Nn = {mesh.Nn}"

    # get axis
    ax, inDim = __Get_axis(ax, inDim)
    ax.set_title(title)

    if inDim == 3:
        # in 3d space

        if dimElem == 1:

            # get segments coordinates / vertices
            segments = groupElem.connect[:, groupElem.segments[0]]
            vertices = mesh.coord[segments, :inDim]
            verticesDef = coord[segments, :inDim]

            if deformFactor > 0:
                # Deformed mesh
                pc = Line3DCollection(verticesDef, edgecolor="red", lw=lw, zorder=1)
                ax.add_collection3d(pc)  # type: ignore
                ax.plot(*coord.T, c="red", lw=lw, marker=".", ls="")

            # Undeformed mesh
            pc = Line3DCollection(vertices, edgecolor=edgecolor, lw=lw, zorder=0)
            ax.plot(*mesh.coord.T, c="black", lw=lw, marker=".", ls="")
        else:

            # construct the connection matrix for the surfaces
            list_connect: list[_types.IntArray] = []
            list_groupElem = mesh.Get_list_groupElem(dimElem)
            list_surfaces = _Get_list_surfaces(mesh, dimElem)
            for groupElem, surfaces in zip(list_groupElem, list_surfaces):
                list_connect.extend(groupElem.connect[:, surfaces])

            # get faces coordinates / vertices
            verticesDef = coord[list_connect, :inDim]
            vertices = mesh.coord[list_connect, :inDim]

            if deformFactor > 0:
                # Deformed mesh
                pcDef = Poly3DCollection(
                    verticesDef, edgecolor="red", linewidths=0.5, alpha=0, zorder=1
                )
                ax.add_collection3d(pcDef)  # type: ignore
                alpha = 0

            # Undeformed mesh
            pc = Poly3DCollection(
                vertices,
                facecolors=facecolors,
                edgecolor=edgecolor,
                linewidths=0.5,
                alpha=alpha,
                zorder=0,
            )
        ax.add_collection3d(pc, zs=0, zdir="z")  # type: ignore

        _Axis_equal_3D(ax, coord)  # type: ignore

    else:
        # in 2d space

        # get vertices
        if mesh.dim == 1:
            idx = groupElem.segments[0]
        else:
            idx = groupElem.surfaces[0]
        vertexConnect = groupElem.connect[:, idx]
        vertices = groupElem.coordGlob[vertexConnect, :2]
        verticesDef = coord[vertexConnect, :2]

        if deformFactor > 0:
            # Deformed mesh
            pc = LineCollection(
                verticesDef,  # type: ignore
                edgecolor="red",
                lw=lw,
                zorder=1,  # type: ignore
            )
            ax.add_collection(pc)
            # Overlay undeformed and deformed mesh
            # Undeformed mesh
            pc = LineCollection(
                vertices,  # type: ignore
                edgecolor=edgecolor,
                lw=lw,
                zorder=0,  # type: ignore
            )
            ax.add_collection(pc)
        else:
            # Undeformed mesh
            if facecolors != edgecolor:
                pc = LineCollection(vertices, edgecolor=edgecolor, lw=lw, zorder=1)  # type: ignore
                ax.add_collection(pc)
            else:
                edgecolor = None

            pc = PolyCollection(
                vertices,  # type: ignore
                facecolors=facecolors,
                edgecolor=edgecolor,
                lw=lw,
                zorder=1,
                alpha=alpha,
            )
            ax.add_collection(pc)

        if mesh.dim == 1:
            # nodes
            if deformFactor > 0:
                ax.plot(*coord[:, :2].T, c="red", lw=lw, marker=".", ls="")
            ax.plot(*mesh.coord[:, :2].T, c="black", lw=lw, marker=".", ls="")

        ax.autoscale()
        if ax.name != "3d":
            ax.axis("equal")

    tic.Tac("Display", "Plot_Mesh")

    if folder != "":
        Save_fig(folder, "mesh")

    return ax  # type: ignore


@requires_matplotlib
def _Plot_obj(
    obj: Union["_Simu", "Mesh", "_GroupElem"],
    alpha: float = 1.0,
    color: str = "gray",
    ax: Optional[Axes] = None,
) -> Axes:
    """Plots the mesh.

    Parameters
    ----------
    obj : _Simu | Mesh | _GroupElem
        object containing the mesh
    alpha : float, optional
        face transparency, default 1.0
    color: str, optional
        color, default 'gray'
    ax: Axes, optional
        Axis to use, default None

    Returns
    -------
    Axes
    """

    tic = Tic()

    _, mesh, coord, inDim = _Init_obj(obj)
    groupElem = mesh.groupElem

    if ax is not None:
        inDim = 3 if ax.name == "3d" else inDim

    # Dimension of displayed elements
    dimElem = mesh.dim
    # If the mesh is a 3D mesh, only the 2D elements of the mesh will be displayed.
    if dimElem == 3:
        dimElem = 2

    # get axis
    ax, inDim = __Get_axis(ax, inDim)
    inDim = np.max([inDim, 2])

    if dimElem == 1:
        segments = groupElem.connect[:, groupElem.segments[0]]
        vertices = coord[segments, :inDim]

        params = {"edgecolor": color, "lw": 0.5, "alpha": alpha}

        if inDim == 3:
            pc = Line3DCollection(vertices, **params)
            ax.add_collection3d(pc)  # type: ignore
        else:
            pc = LineCollection(vertices, **params)  # type: ignore [arg-type]
            ax.add_collection(pc)

    else:

        # construct the connection matrix for the surfaces
        list_connect: list[_types.IntArray] = []
        list_groupElem = mesh.Get_list_groupElem(dimElem)
        list_surfaces = _Get_list_surfaces(mesh, dimElem)
        for groupElem, surfaces in zip(list_groupElem, list_surfaces):
            list_connect.extend(groupElem.connect[:, surfaces])

        # get faces coordinates / vertices
        vertices = mesh.coord[list_connect, :inDim]

        params = {"facecolors": color, "alpha": alpha}

        if inDim == 3:
            pc = Poly3DCollection(vertices, **params)
            ax.add_collection3d(pc)  # type: ignore
        else:
            pc = PolyCollection(vertices, **params)  # type: ignore [arg-type]
            ax.add_collection(pc)  # type: ignore

    if inDim == 3:
        _Axis_equal_3D(ax, coord)  # type: ignore
    else:
        ax.autoscale()
        ax.axis("equal")

    tic.Tac("Display", "Plot")

    return ax  # type: ignore


@requires_matplotlib
def Plot_Nodes(
    obj,
    nodes: Optional[_types.IntArray] = None,
    showId=False,
    marker=".",
    color="red",
    ax: Optional[Axes] = None,
) -> Axes:
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
    color: str, optional
        color, default 'red'
    ax : Axes, optional
        Axis to use, default None, default None

    Returns
    -------
    Axes
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

    if nodes.size == 0:
        return ax

    coordo = mesh.coord

    if inDim == 2:
        ax.plot(*coordo[nodes, :2].T, ls="", marker=marker, c=color, zorder=2.5)
        if showId:
            [ax.text(*coordo[node, :2].T, str(node), c=color) for node in nodes]  # type: ignore [call-arg]
        ax.axis("equal")
    elif inDim == 3:
        ax.plot(*coordo[nodes].T, ls="", marker=marker, c=color, zorder=2.5)
        if showId:
            [ax.text(*coordo[node].T, str(node), c=color) for node in nodes]  # type: ignore [call-arg]
        _Axis_equal_3D(ax, coordo)

    tic.Tac("Display", "Plot_Nodes")

    return ax


@requires_matplotlib
def Plot_Elements(
    obj,
    nodes=[],
    dimElem: Optional[int] = None,
    showId=False,
    alpha=1.0,
    color="red",
    edgecolor="black",
    ax: Optional[Axes] = None,
) -> Axes:
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
    color : str, optional
        color used to display faces, by default 'red
    edgecolor : str, optional
        color used to display segments, by default 'black'
    ax : Axes, optional
        Axis to use, default None

    Returns
    -------
    Axes
    """

    tic = Tic()

    mesh = _Init_obj(obj)[1]

    inDim = mesh.inDim

    if dimElem is None:
        dimElem = 2 if mesh.inDim == 3 else mesh.dim

    ax, inDim = __Get_axis(ax, inDim)

    # list of element group associated with the dimension
    list_groupElem = mesh.Get_list_groupElem(dimElem)
    if len(list_groupElem) == 0:
        return None  # type: ignore

    # for each group elem
    for groupElem in list_groupElem:
        # get the elements associated with the nodes
        if len(nodes) > 0:
            elements = groupElem.Get_Elements_Nodes(nodes)
        else:
            elements = np.arange(groupElem.Ne)

        if elements.size == 0:
            continue

        # get params
        if groupElem.dim == 1:
            # 1D elements
            idx = groupElem.segments.ravel().tolist()
            # get params
            params = {"edgecolor": color, "lw": 1, "zorder": 2}
        else:
            # 2D elements
            idx = groupElem.surfaces.ravel().tolist()
            # get params
            params = {
                "facecolors": color,
                "edgecolor": edgecolor,
                "lw": 0.5,
                "alpha": alpha,
                "zorder": 2,
            }

        # Construct the vertices coordinates
        connect_e = groupElem.connect  # connect
        coord_n = groupElem.coordGlob[:, : mesh.inDim]  # global coordinates
        vertices_e = coord_n[connect_e[:, idx]]
        vertices = vertices_e[elements]

        # center coordinates for each elements
        center_e = np.mean(vertices_e, axis=1)

        if inDim == 3:

            if groupElem.dim == 1:
                pc = Line3DCollection(vertices, **params)
            else:
                pc = Poly3DCollection(vertices, **params)

            ax.add_collection3d(pc, zdir="z")

        else:

            if groupElem.dim == 1:
                pc = LineCollection(vertices, **params)
            else:
                pc = PolyCollection(vertices, **params)

            ax.add_collection(pc)

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


@requires_matplotlib
def Plot_BoundaryConditions(simu, ax: Optional[Axes] = None) -> Axes:
    """Plots simulation's boundary conditions.

    Parameters
    ----------
    simu : _Simu
        simulation
    ax : Axes, optional
        Axis to use, default None

    Returns
    -------
    Axes
    """

    tic = Tic()

    simu = _Init_obj(simu)[0]

    coord = simu.mesh.coord

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
        ax = Plot_Elements(simu.mesh, dimElem=1, color="k")
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
            marker = "."

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


@requires_matplotlib
def Plot_Tags(
    obj,
    showId=True,
    folder="",
    alpha=1.0,
    useColorCycler=False,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plots the mesh's elements tags (from 2d elements to points) but do not plot the 3d elements tags.

    Parameters
    ----------
    obj : _Simu | Mesh | _GroupElem
        object containing the mesh
    showId : bool, optional
        shows tags, by default True
    folder : str, optional
        saves folder, by default ""
    alpha : float, optional
        transparency, by default 1.0
    useColorCycler : bool, optional
        whether to use color cycler, by default False
    ax : Axes, optional
        Axis to use, default None

    Returns
    -------
    Axes
    """

    tic = Tic()

    _, mesh, coord, inDim = _Init_obj(obj)

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

    ax, inDim = __Get_axis(ax, inDim)
    inDim = np.max([inDim, 2])

    _Plot_obj(mesh, alpha=0.1, color="gray", ax=ax)

    colors = plt.get_cmap("tab10").colors
    colorIterator = iter(colors * np.ceil(np.sum(nTtags) / len(colors)).astype(int))

    # List of collections during creation
    collections = []
    for groupElem in mesh.dict_groupElem.values():
        # Tags available by element group
        tags_e = groupElem.elementTags
        dim = groupElem.dim
        coord = groupElem.coordGlob[:, :inDim]
        center_e = np.mean(coord[groupElem.connect], axis=1)  # center of each elements

        if groupElem.dim == 1:
            idx = groupElem.segments[0]
        else:
            idx = groupElem.surfaces.ravel().tolist()
        vertices_e = coord[groupElem.connect[:, idx]]

        for tag_e in tags_e:
            if "nodes" in tag_e:
                pass

            nodes = groupElem.Get_Nodes_Tag(tag_e)
            elements = groupElem.Get_Elements_Tag(tag_e)
            if len(elements) == 0 or len(nodes) == 0:
                continue

            vertices = vertices_e[elements]

            # Assign color
            if useColorCycler:
                color = next(colorIterator)
            elif groupElem.dim in [0, 1]:
                color = "black"
            else:
                color = "tab:cyan"

            center = np.mean(center_e[elements], axis=0)

            if dim == 0:
                # plot points
                points = ax.scatter(
                    *coord[nodes, :inDim].T,
                    c="black",  # type: ignore [misc]
                    marker=".",
                    zorder=2,
                    label=tag_e,
                    lw=2,
                )
                collections.append(points)
            elif dim == 1:
                # plot lines
                params = {
                    "lw": 1.5,
                    "edgecolor": "black",
                    "alpha": 1,
                    "label": tag_e,
                }

                if inDim == 3:
                    pc = Line3DCollection(vertices, **params)
                    collections.append(ax.add_collection3d(pc, zdir="z"))
                else:
                    pc = LineCollection(vertices, **params)  # type: ignore [arg-type]
                    collections.append(ax.add_collection(pc))

            elif dim == 2:
                # plot surfaces

                if inDim == 3:
                    pc = Poly3DCollection(
                        vertices,
                        lw=0,
                        alpha=alpha,
                        facecolors=color,
                        label=tag_e,
                    )
                    collections.append(ax.add_collection3d(pc, zdir="z"))
                else:
                    pc = PolyCollection(
                        vertices,  # type: ignore
                        facecolors=color,
                        label=tag_e,
                        alpha=alpha,
                    )
                    collections.append(ax.add_collection(pc))

            if showId:
                ax.text(*center[:inDim], tag_e, zorder=25)  # type: ignore [arg-type, call-arg]

        if inDim == 3:
            _Axis_equal_3D(ax, coord)
        else:
            ax.autoscale()
            ax.axis("equal")

    tic.Tac("Display", "Plot_Tags")

    if folder != "":
        Save_fig(folder, "geom")

    __Annotation_Event(collections, ax.figure, ax)

    return ax


@requires_matplotlib
def __Annotation_Event(
    collections: list, fig: Union[plt.Figure, Any], ax: Axes
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
@requires_matplotlib
def Plot_Force_Displacement(
    force: _types.FloatArray,
    displacement: _types.FloatArray,
    xlabel="u",
    ylabel="f",
    folder="",
    ax: Optional[Axes] = None,
) -> tuple[plt.Figure, Axes]:  # type: ignore
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
    ax : Axes, optional
        ax in which to plot the figure, by default None

    Returns
    -------
    tuple[plt.Figure, Axes]
        returns figure and ax
    """

    if isinstance(ax, Axes):  # type: ignore
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


@requires_matplotlib
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
    N = np.max([Niter, N])
    iterations = np.linspace(0, Niter - 1, N, endpoint=True, dtype=int)

    list_dict_energy: list[dict[str, float]] = []
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

        list_dict_energy.append(simu.Results_dict_Energy())

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
    axs: list[Axes] = plt.subplots(nrows, 1, sharex=True)[1]

    iter_rows = iter(np.arange(nrows))
    row: int = next(iter_rows)

    # Retrieve the axis to be used for x-axes
    if len(displacement) > 0:
        listX = displacement[iterations]
        xlabel = "displacement"
    else:
        listX = iterations
        xlabel = "iter"

    # For each energy, we plot the values
    for energy_str in list_dict_energy[0].keys():
        values = [dict_energy[energy_str] for dict_energy in list_dict_energy]
        axs[row].plot(listX, values, label=energy_str)
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


@requires_matplotlib
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

    axs: list[Axes] = plt.subplots(nrows=nbGraph, sharex=True)[1]

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
@requires_matplotlib
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
    N = np.max([Niter, N])
    iterations = np.linspace(0, Niter - 1, N, endpoint=True, dtype=int)

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


@requires_matplotlib
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

            iteration = i + 1
            rmTime = Tic.Get_Remaining_Time(iteration, N, time)

            iteration = str(iteration).zfill(len(str(N)))
            MyPrint(f"Generate movie {iteration}/{N} {rmTime}    ", end="\r")


# ----------------------------------------------
# Functions
# ----------------------------------------------


@requires_matplotlib
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


def _Get_list_surfaces(mesh, dimElem: int) -> list[list[int]]:
    """Returns a list of surfaces for each element group of dimension dimElem.\n
    Surfaces are a list of index used to construct/plot a surface.\n
    You can go check their values for each groupElem in `EasyFEA/fem/elems/` folder"""

    mesh = _Init_obj(mesh)[1]

    list_surfaces: list[list[int]] = []  # list of faces
    list_len: list[int] = []  # list that store the size for each faces

    # get faces and nodes per element for each element group
    for groupElem in mesh.Get_list_groupElem(dimElem):
        list_surfaces.append(groupElem.surfaces.ravel().tolist())
        list_len.append(groupElem.surfaces.size)

    # make sure that faces in list_faces are at the same length
    max_len = np.max(list_len)
    # this loop make sure that faces in list_faces get the same length
    for f, surfaces in enumerate(list_surfaces.copy()):
        repeat = max_len - len(surfaces)
        if repeat > 0:
            surfaces.extend([surfaces[0]] * repeat)
            list_surfaces[f] = surfaces

    return list_surfaces


@requires_matplotlib
def _Remove_colorbar(ax: Axes) -> None:
    """Removes the current colorbar from the axis."""
    [
        collection.colorbar.remove()
        for collection in ax.collections
        if collection.colorbar is not None
    ]


@requires_matplotlib
def Init_Axes(dim: int = 2, elev=105, azim=-90) -> Axes:
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


@requires_matplotlib
def _Axis_equal_3D(ax: Axes3D, coord: _types.FloatArray) -> None:
    """Changes axis size for 3D display.\n
    Center the part and make the axes the right size.

    Parameters
    ----------
    ax : Axes
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
    from .. import BUILDING_GALLERY

    if not BUILDING_GALLERY:
        syst = platform.system()
        if syst in ["Linux", "Darwin"]:
            Folder.os.system("clear")
        elif syst == "Windows":
            Folder.os.system("cls")
