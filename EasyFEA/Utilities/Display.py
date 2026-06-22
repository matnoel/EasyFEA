# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing functions used to display simulations and meshes with matplotlib (https://matplotlib.org/)."""

from __future__ import annotations
import platform
from typing import Union, Callable, Optional, TYPE_CHECKING, Any
import numpy as np
from enum import Enum
import re

# utilities
from . import Folder, Tic, _types
from ._mpi import rank0_only, MPI_COMM

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
    from matplotlib import colorbar
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
# Plot core (matplotlib analogue of PyVista.Plot)
# ----------------------------------------------
def __Get_vertices(
    mesh: "Mesh",
    coord: _types.FloatArray,
    inDim: int,
    dimElem: int,
) -> _types.FloatArray:
    """Returns the (Ne, nPts, dim) vertex array used to build a matplotlib collection.\n
    Shared by Plot and Plot_Mesh. Branches exactly as the historical code to avoid display regressions.
    """

    if inDim == 3:
        # When the mesh uses 3D elements, only the 2D surfaces are displayed.
        dimElem = 2 if dimElem == 3 else dimElem
        if dimElem == 1:
            list_connect = []
            for groupElem in mesh.Get_list_groupElem(dimElem):
                list_connect.extend(groupElem.connect[:, groupElem.segments[0]])
            vertices = coord[list_connect]
        else:
            # construct the surface connection matrix across every 2D element group
            list_connect = []
            list_groupElem = mesh.Get_list_groupElem(dimElem)
            list_surfaces = _Get_list_surfaces(mesh, dimElem)
            for groupElem, surfaces in zip(list_groupElem, list_surfaces):
                list_connect.extend(groupElem.connect[:, surfaces])  # type: ignore [attr-defined]
            vertices = coord[list_connect]
    else:
        # one or several element groups of dimension dimElem; build the polygons (or segments) following Get_list_groupElem(dimElem) order so they match the element ordering of any per-element field. When the groups mix element types (e.g. QUAD4 + TRI3) the polygons have different vertex counts, so a ragged list is returned instead of an ndarray.
        list_verts: list[_types.FloatArray] = []
        nPts: Optional[int] = None
        homogeneous = True
        for groupElem in mesh.Get_list_groupElem(dimElem):
            idx = groupElem.segments[0] if dimElem == 1 else groupElem.surfaces[0]
            verts = coord[groupElem.connect[:, idx], :2]  # (Ne_g, nPts_g, 2)
            if nPts is None:
                nPts = verts.shape[1]
            elif verts.shape[1] != nPts:
                homogeneous = False
            list_verts.append(verts)
        if homogeneous:
            vertices = np.concatenate(list_verts, axis=0)
        else:
            vertices = [poly for verts in list_verts for poly in verts]  # type: ignore [assignment]

    return vertices


@requires_matplotlib
def __Add_Collection(
    ax: Axes,
    vertices: _types.FloatArray,
    inDim: int,
    dimElem: int,
    *,
    array: Optional[_types.FloatArray] = None,
    norm=None,
    cmap: Optional[str] = None,
    facecolors=None,
    edgecolor=None,
    lw: float = 0.5,
    alpha: float = 1.0,
    zorder: float = 0.0,
    clim: Optional[tuple] = None,
):
    """Builds and adds the matplotlib collection matching ``inDim`` × ``dimElem`` to ``ax``.\n
    This is the matplotlib analogue of ``pyvista.Plotter.add_mesh``. Returns the collection.
    """

    is3D = inDim == 3
    isLine = dimElem == 1

    if isLine:
        Coll = Line3DCollection if is3D else LineCollection
    else:
        Coll = Poly3DCollection if is3D else PolyCollection

    params: dict[str, Any] = {"zorder": zorder, "lw": lw, "alpha": alpha}
    if cmap is not None:
        params["cmap"] = cmap
    if norm is not None:
        params["norm"] = norm
    if edgecolor is not None:
        params["edgecolor"] = edgecolor
    if facecolors is not None:
        # lines are colored through edgecolor; faces through facecolors
        params["edgecolor" if isLine else "facecolors"] = facecolors

    pc = Coll(vertices, **params)  # type: ignore [arg-type]

    if array is not None:
        pc.set_array(array)
        if clim is not None:
            pc.set_clim(*clim)

    if is3D:
        ax.add_collection3d(pc, zs=0, zdir="z")  # type: ignore [union-attr]
    else:
        ax.add_collection(pc)

    return pc


def _Node_to_element_values(
    mesh: "Mesh", values: _types.FloatArray, dimElem: int
) -> _types.FloatArray:
    """Averages nodal values over each element of dimension ``dimElem`` (used for 3D surface display)."""
    elementValues: list = []
    for groupElem in mesh.Get_list_groupElem(dimElem):
        elementValues.extend(np.mean(values[groupElem.connect], axis=1))
    return np.asarray(elementValues)


@requires_matplotlib
def Plot(
    obj: Union["_Simu", "Mesh", "_GroupElem"],
    result: Optional[Union[str, _types.FloatArray]] = None,
    deformFactor: _types.Number = 0.0,
    coef: _types.Number = 1.0,
    nodeValues: bool = True,
    color=None,
    plotMesh: bool = False,
    edgecolor: str = "black",
    lw: float = 0.5,
    alpha: float = 1.0,
    cmap: str = "jet",
    ncolors: int = 256,
    clim=(None, None),
    ax: Optional[Axes] = None,
    colorbarIsClose: bool = False,
    colorbarLabel: str = "",
    title: str = "",
    folder: str = "",
    filename: str = "",
) -> Axes:
    """Plots an object (simulation, mesh or group of elements) with matplotlib.

    This is the rendering core that ``Plot_Mesh`` and ``_Plot_obj`` delegate to.
    It is the matplotlib counterpart of ``PyVista.Plot``: pass ``result`` to color the object by a
    scalar field, or ``color`` to draw it with a single solid color.

    Parameters
    ----------
    obj : _Simu | Mesh | _GroupElem
        object to plot
    result : str | _types.FloatArray, optional
        Result used to color the object. Must be included in simu.Get_Results() or be a numpy array of size (Nn,) or (Ne,). When None, the object is drawn with ``color``, by default None
    deformFactor : float, optional
        factor used to display the deformed solution (0 means no deformations), default 0.0
    coef : float, optional
        coef to apply to the solution, by default 1.0
    nodeValues : bool, optional
        displays result to nodes otherwise displays it to elements, by default True
    color : str, optional
        solid color used when ``result`` is None, by default None
    plotMesh : bool, optional
        displays mesh edges, by default False
    edgecolor : str, optional
        Color used to plot the mesh, by default 'black'
    lw : float, optional
        line width, by default 0.5
    alpha : float, optional
        face transparency, by default 1.0
    cmap : str, optional
        the color map used near the figure, by default "jet" \n
        ["jet", "seismic", "binary", "viridis"] -> https://matplotlib.org/stable/tutorials/colors/colormaps.html
    ncolors : int, optional
        number of colors for colorbar, by default 256
    clim : sequence[float], optional
        Two item color bar range for scalars. Defaults to minimum and maximum of scalars array. Example: (-1, 2), by default (None, None)
    ax : axis, optional
        Axis to use, by default None
    colorbarIsClose : bool, optional
        color bar is displayed close to the figure, by default False
    colorbarLabel : str, optional
        colorbar label, by default ""
    title : str, optional
        figure title, by default ""
    folder : str, optional
        save folder, by default "".
    filename : str, optional
        filename, by default ""

    Returns
    -------
    Axes

    Examples
    --------
    Von Mises stress in MPa (elastic simulation):

    >>> from EasyFEA import Display
    >>> Display.Plot(simu, "Svm", coef=1e-6, colorbarLabel="σ_vm [MPa]")
    >>> Display.plt.show()

    Mesh only (no scalar field):

    >>> Display.Plot(mesh, color="cyan", plotMesh=True)
    >>> Display.plt.show()
    """

    tic = Tic()

    simu, mesh, coordDef, inDim = _Init_obj(obj, deformFactor)  # type: ignore
    dimElem = mesh.dim  # Dimension of displayed elements

    hasResult = result is not None

    if dimElem == 1:
        # Don't know how to display nodal values on lines
        nodeValues = False  # do not modify
    elif dimElem == 3:
        # When mesh use 3D elements, results are displayed only on 2D elements.
        # To display values on 2D elements, we first need to know the values at 3D nodes.
        nodeValues = True  # do not modify

    ax, inDim = __Get_axis(ax, inDim)

    # surface dimension actually displayed (3D meshes show their 2D skin)
    surfDim = 2 if (inDim == 3 and dimElem == 3) else dimElem

    if hasResult:
        # Get values and colorbar properties
        values = _Get_values(simu, mesh, result, nodeValues) * coef
        ticks, levels, norm, vmin, vmax = __Get_colorbar_properties(
            clim, result, values, ncolors
        )
    else:
        values = None  # type: ignore [assignment]
        norm = None

    vertices = __Get_vertices(mesh, coordDef, inDim, dimElem)

    if inDim == 3:

        if surfDim == 1 and plotMesh:
            ax.plot(*coordDef.T, c=edgecolor, lw=0.1, marker=".", ls="")

        if hasResult:
            # element values colored by the scalar field
            if nodeValues:
                elementValues = _Node_to_element_values(mesh, values, surfDim)
            else:
                elementValues = values
            edge = edgecolor if (plotMesh and surfDim == 2) else None
            pc = __Add_Collection(
                ax,
                vertices,
                inDim,
                surfDim,
                array=elementValues,
                norm=norm,
                cmap=cmap,
                edgecolor=edge,
                lw=1.5 if surfDim == 1 else 0.5,
            )
            pc.set_clim(
                np.min([elementValues.min(), vmin]),
                np.max([elementValues.max(), vmax]),
            )
            colorbar = plt.colorbar(pc, ax=ax, ticks=ticks)
            colorbar.set_label(colorbarLabel)
        else:
            # solid color
            __Add_Collection(
                ax,
                vertices,
                inDim,
                surfDim,
                facecolors=color,
                edgecolor=edgecolor if plotMesh else None,
                lw=lw,
                alpha=alpha,
            )

        _Axis_equal_3D(ax, coordDef)

    else:

        # Plot the mesh edges (for a scalar field, edges are a dedicated collection drawn
        # underneath, matching the historical scalar-field behavior)
        if plotMesh and mesh.dim == 1:
            # mesh for 1D elements are points
            ax.plot(*coordDef[:, :2].T, c=edgecolor, lw=0.1, marker=".", ls="")
        elif plotMesh and hasResult:
            # mesh for 2D elements are lines / segments (dimElem=1 for LineCollection)
            __Add_Collection(ax, vertices, inDim, 1, edgecolor=edgecolor, lw=0.5)

        if hasResult and nodeValues:
            # smooth nodal field: matplotlib has no collection equivalent -> tricontourf
            # triangulate every main-dimension element group (QUAD4 -> 2 tris, ...)
            triangulation = np.concatenate(
                [
                    np.reshape(groupElem.connect[:, groupElem.triangles], (-1, 3))
                    for groupElem in mesh.Get_list_groupElem(mesh.dim)
                ]
            )
            pc = ax.tricontourf(  # type: ignore [call-overload]
                *coordDef[:, :2].T,
                triangulation,
                values,
                levels,
                cmap=cmap,
                vmin=values.min(),
                vmax=values.max(),
                zorder=-1,
            )
        elif hasResult:
            # element values
            pc = __Add_Collection(
                ax,
                vertices,
                inDim,
                surfDim,
                array=values,
                norm=norm,
                cmap=cmap,
                lw=1.5 if surfDim == 1 else 0.5,
                clim=(vmin, vmax),
            )
        else:
            # solid color (edges live on the face collection, matching Plot_Mesh / _Plot_obj)
            __Add_Collection(
                ax,
                vertices,
                inDim,
                surfDim,
                facecolors=color,
                edgecolor=edgecolor if plotMesh else None,
                lw=lw,
                alpha=alpha,
            )

        ax.autoscale()
        ax.axis("equal")

        if hasResult:
            if colorbarIsClose:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="10%", pad=0.1)
            else:
                cax = None
            colorbar = plt.colorbar(pc, ax=ax, cax=cax, ticks=ticks)
            colorbar.set_label(colorbarLabel)

    # Title
    if title == "" and isinstance(result, str):
        ax.set_title(rf"${__Get_latex_title(result, nodeValues)}$")
    elif title != "":
        ax.set_title(title)

    tic.Tac("Display", "Plot")

    # If the folder has been filled in, save the figure.
    if folder != "":
        if filename == "":
            filename = result if isinstance(result, str) else "mesh"
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
            optionTex = rf"\sigma_{{{optionFin}}}"
        elif "E" in result:
            optionFin = result.split("E")[-1]
            optionTex = rf"\epsilon_{{{optionFin}}}"

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

    Examples
    --------
    Undeformed mesh:

    >>> import matplotlib.pyplot as plt
    >>> Display.Plot_Mesh(simu)
    >>> plt.show()

    Deformed mesh, semi-transparent faces:

    >>> Display.Plot_Mesh(simu, deformFactor=50, facecolors="white", alpha=0.5)
    >>> plt.show()
    """

    tic = Tic()

    simu, mesh, coordDef, inDim = _Init_obj(obj, deformFactor)
    coord = mesh.coord

    if ax is not None:
        inDim = 3 if ax.name == "3d" else inDim

    deformFactor = 0 if simu is None else np.abs(deformFactor)

    if title == "":
        title = str(mesh).replace("\n", ", ")

    if deformFactor == 0:
        # Undeformed mesh: the common case routes through the shared Plot core.
        ax = Plot(
            obj,
            color=facecolors,
            plotMesh=True,
            edgecolor=edgecolor,
            lw=lw,
            alpha=alpha,
            ax=ax,
            title=title,
        )
        if mesh.dim == 1:
            # 1D meshes display their nodes
            markCoord = coord if ax.name == "3d" else coord[:, :2]
            ax.plot(*markCoord.T, c="black", lw=lw, marker=".", ls="")
    else:
        # Deformed mesh: overlay the deformed (red) over the undeformed wireframe, both built
        # with the same _Get_vertices / _Add_Collection helpers used by Plot. The element
        # outlines are drawn as lines (dimElem=1) so the overlay renders identically in 2D and
        # 3D without relying on transparent faces.
        ax, inDim = __Get_axis(ax, inDim)
        ax.set_title(title)

        verticesDef = __Get_vertices(mesh, coordDef, inDim, mesh.dim)
        vertices = __Get_vertices(mesh, coord, inDim, mesh.dim)

        __Add_Collection(ax, verticesDef, inDim, 1, edgecolor="red", lw=lw, zorder=1)
        __Add_Collection(ax, vertices, inDim, 1, edgecolor=edgecolor, lw=lw, zorder=0)

        if mesh.dim == 1:
            # 1D meshes display their nodes (undeformed in black, deformed in red)
            markCoord = coord if inDim == 3 else coord[:, :2]
            markDef = coordDef if inDim == 3 else coordDef[:, :2]
            ax.plot(*markCoord.T, c="black", lw=lw, marker=".", ls="")
            ax.plot(*markDef.T, c="red", lw=lw, marker=".", ls="")

        if inDim == 3:
            _Axis_equal_3D(ax, coordDef)  # type: ignore
        else:
            ax.autoscale()
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

    return Plot(obj, color=color, alpha=alpha, ax=ax)


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

    _, mesh, coord, inDim = _Init_obj(obj)

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

    if inDim == 2:
        ax.plot(*coord[nodes, :2].T, ls="", marker=marker, c=color, zorder=2.5)
        if showId:
            [ax.text(*coord[node, :2].T, str(node), c=color) for node in nodes]  # type: ignore [call-arg]
        ax.axis("equal")
    elif inDim == 3:
        ax.plot(*coord[nodes].T, ls="", marker=marker, c=color, zorder=2.5)
        if showId:
            [ax.text(*coord[node].T, str(node), c=color) for node in nodes]  # type: ignore [call-arg]
        _Axis_equal_3D(ax, coord)

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

    _, mesh, coord, inDim = _Init_obj(obj)

    if dimElem is None:
        dimElem = 2 if inDim == 3 else mesh.dim

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
        vertices_e = coord[connect_e[:, idx], : mesh.inDim]
        vertices = vertices_e[elements]

        # center coordinates for each elements
        center_e = np.mean(vertices_e, axis=1)

        __Add_Collection(ax, vertices, inDim, groupElem.dim, **params)

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
        _Axis_equal_3D(ax, coord)

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

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> Display.Plot_BoundaryConditions(simu)
    >>> plt.show()

    Combined with mesh overlay:

    >>> ax = Display.Plot_Mesh(simu)
    >>> Display.Plot_BoundaryConditions(simu, ax=ax)
    >>> plt.show()
    """

    tic = Tic()

    simu, _, coord, _ = _Init_obj(simu)

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
        center_e = np.mean(coord[groupElem.connect], axis=1)  # center of each elements

        if groupElem.dim == 1:
            idx = groupElem.segments[0]
        else:
            idx = groupElem.surfaces.ravel().tolist()
        vertices_e = coord[groupElem.connect[:, idx], :inDim]

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
                pc = __Add_Collection(
                    ax, vertices, inDim, 1, edgecolor="black", lw=1.5, alpha=1
                )
                pc.set_label(tag_e)
                collections.append(pc)

            elif dim == 2:
                # plot surfaces
                pc = __Add_Collection(
                    ax, vertices, inDim, 2, facecolors=color, lw=0, alpha=alpha
                )
                pc.set_label(tag_e)
                collections.append(pc)

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
            coord = ax.format_coord(event.xdata, event.ydata)
            toolbar.set_message(f"{collection.get_label()} : {coord}")
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
@rank0_only
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
    Niter = simu.Niter
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


@rank0_only
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
@rank0_only
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

    simu, _, _, inDim = _Init_obj(simu)

    if simu is None:
        MyPrintError("Must give a simulation.")
        return

    Niter = simu.Niter
    N = np.max([Niter, N])
    iterations = np.linspace(0, Niter - 1, N, endpoint=True, dtype=int)

    ax = Init_Axes(inDim)
    fig = ax.figure

    # activate the first iteration
    simu.Set_Iter(0, resetAll=True)

    def DoAnim(fig: plt.Figure, i):  # type: ignore
        simu.Set_Iter(iterations[i])
        ax = fig.axes[0]
        _Remove_colorbar(ax)
        ax.clear()
        Plot(
            simu,
            result,
            deformFactor=deformFactor,
            coef=coef,
            nodeValues=nodeValues,
            plotMesh=plotMesh,
            edgecolor=edgecolor,
            ax=ax,
            **kwargs,
        )
        ax.set_title(f"{result} {iterations[i]:d}/{Niter - 1:d}")

    Movie_func(DoAnim, fig, iterations.size, folder, filename, fps)


@rank0_only
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
    filename = Folder.Join(folder, filename, mkdir=True)

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


@rank0_only
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

    # Remove invalid characters for Windows/Mac/Linux
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", filename)

    # Remove leading/trailing spaces and dots
    filename = filename.strip(". ")

    path = Folder.Join(folder, filename + "." + extension)

    Folder.os.makedirs(folder, exist_ok=True)

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


@requires_matplotlib
def _Get_colors_for_values(
    values: np.ndarray,
    vMin: float = None,
    vMax: float = None,
    cmap: str = "jet",
) -> np.ndarray:
    """
    Generates RGB colors for scalar values using a matplotlib colormap.

    Parameters
    ----------
    values : np.ndarray
        1D array of scalar values to be mapped to colors
    vMin : float, optional
        Minimum value for normalization. If None, uses the minimum of values.
    vMax : float, optional
        Maximum value for normalization. If None, uses the maximum of values.
    cmap: str, optional
        the color map used near the figure, by default "jet" \n
        ["jet", "seismic", "binary", "viridis"] -> https://matplotlib.org/stable/tutorials/colors/colormaps.html

    Returns
    -------
    np.ndarray
        Array of RGB colors with shape (N, 3) and values in range [0, 1]

    Notes
    -----
    The function normalizes input values to [0, 1] and maps them through the specified
    colormap. The alpha channel is discarded, returning only RGB components.
    """

    assert isinstance(values, np.ndarray), "values must be a numpy array"
    assert values.ndim == 1, "values must be a 1D array"

    # Determine normalization bounds
    vMin = values.min() if vMin is None else vMin
    vMax = values.max() if vMax is None else vMax

    # Normalize values to [0, 1] range
    if vMax > vMin:
        normalizedValues = (values - vMin) / (vMax - vMin)
    else:
        normalizedValues = np.zeros_like(values)

    # Apply colormap and extract RGB components
    colormap = plt.get_cmap(cmap)
    colors = colormap(normalizedValues)[:, :3]  # Discard alpha channel

    return colors


@rank0_only
@requires_matplotlib
def _Save_colorbar(
    vMin: float,
    vMax: float,
    folder: str,
    filename="colorbar",
    cmap="jet",
    orientation="vertical",
    label="",
):
    """
    Generates and save colorbar.

    Parameters
    ----------
    vMin : float, optional
        Minimum value for normalization. If None, uses the minimum of values.
    vMax : float, optional
        Maximum value for normalization. If None, uses the maximum of values.
    folder : str, optional
        save folder, by default "".
    filename : str, optional
        filename, by default "colorbar"
    cmap: str, optional
        the color map used near the figure, by default "jet" \n
        ["jet", "seismic", "binary", "viridis"] -> https://matplotlib.org/stable/tutorials/colors/colormaps.html
    orientation : str, optional
        orientation, by default "vertical"
    label : str, optional
        label, by default ""
    """

    fig = plt.figure(figsize=(1.5, 6) if orientation == "vertical" else (6, 1.5))
    ax = fig.add_axes(
        [
            0.05,
            0.05,
            0.15 if orientation == "vertical" else 0.9,
            0.9 if orientation == "vertical" else 0.15,
        ]
    )

    norm = colors.Normalize(vmin=vMin, vmax=vMax)

    cb = colorbar.ColorbarBase(
        ax, cmap=plt.get_cmap(cmap), norm=norm, orientation=orientation
    )

    # set explicit ticks
    nTicks = 5  # Number of tick marks
    tick_values = np.linspace(vMin, vMax, nTicks)
    cb.set_ticks(tick_values)

    # set label
    if label != "":
        cb.set_label(label, fontsize=12)

    path = Folder.Join(folder, filename + ".png", mkdir=True)
    plt.savefig(
        path,
        dpi=150,
        # bbox_inches="tight",
        transparent=True,
        pad_inches=0.1,
    )
    plt.close()


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


@rank0_only
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

        if end == "\r" and MPI_COMM is not None:
            end = "\n"

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
