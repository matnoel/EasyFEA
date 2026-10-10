# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing functions used to display simulations and meshes with matplotlib (https://matplotlib.org/)."""

from __future__ import annotations
from typing import Callable, TYPE_CHECKING, Any, Sequence, TypeAlias
import numpy as np
import re

from ..Utilities import Folder, Tic, _types
from ..Utilities._mpi import rank0_only
from ..Utilities.Terminal import MyPrint, MyPrintError
from ..IO._utils import _Init_obj, _Get_values
from ._utils import tab10_colors as tab10_colors  # public
from ._utils import tab20_colors as tab20_colors  # public
from ._utils import (
    PlotOptions,
    _Flatten_geoms,
    _Check_kwargs,
    _Union_bounds,
    _Gauss_points_averaged,
)

if TYPE_CHECKING:
    from typing import Unpack
    from ..Simulations._simu import _Simu
    from ..FEM._mesh import Mesh
    from ..FEM._group_elem import _GroupElem
    from ..Geoms._geom import _Geom

from ..Utilities._requires import Create_requires_decorator

# Matplotlib: https://matplotlib.org/
try:
    from matplotlib import colors
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.collections import PolyCollection, LineCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # use to do colorbarIsClose
    from matplotlib import animation

    Axes: TypeAlias = plt.Axes | Axes3D
except ImportError:
    pass
requires_matplotlib = Create_requires_decorator("matplotlib")

# Ideas: https://www.python-graph-gallery.com/


# ----------------------------------------------
# Plot core (matplotlib analogue of PyVista.Plot)
# ----------------------------------------------
def __Get_vertices(
    mesh: Mesh,
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
            list_connect: list[_types.IntArray] = []
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
        nPts: int | None = None
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
    array: _types.FloatArray | None = None,
    norm=None,
    cmap: str | None = None,
    facecolors=None,
    edgecolor=None,
    linewidth: float | None = 0.5,
    alpha: float = 1.0,
    zorder: float | None = None,
    clim: tuple | None = None,
    label: str | None = None,
    **kwargs,
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

    params: dict[str, Any] = {"lw": linewidth, "label": label, **kwargs}
    if zorder is not None:
        params["zorder"] = zorder
    if cmap is not None:
        params["cmap"] = cmap
    if norm is not None:
        params["norm"] = norm
    if edgecolor is not None:
        params["edgecolor"] = edgecolor
    if facecolors is not None:
        # lines are colored through edgecolor; faces through facecolors
        params["edgecolor" if isLine else "facecolors"] = facecolors

    # alpha sets face transparency: for a polygon with an explicit face color we apply it to the faces only (below) so the mesh edges stay opaque, otherwise matplotlib's collection-level alpha fades the edges too and the wireframe vanishes when alpha=0 (e.g. Plot_Mesh over an image). Lines and colormap-driven faces keep the collection-level alpha.
    applyFaceAlpha = (not isLine) and (array is None) and (facecolors is not None)
    if not applyFaceAlpha:
        params["alpha"] = alpha

    pc = Coll(vertices, **params)  # type: ignore [arg-type]

    if applyFaceAlpha and alpha != 1.0:
        pc.set_facecolor(colors.to_rgba_array(facecolors, alpha))  # type: ignore [arg-type]

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
    mesh: Mesh, values: _types.FloatArray, dimElem: int
) -> _types.FloatArray:
    """Averages nodal values over each element of dimension ``dimElem`` (used for 3D surface display)."""
    elementValues: list = []
    for groupElem in mesh.Get_list_groupElem(dimElem):
        elementValues.extend(np.mean(values[groupElem.connect], axis=1))
    return np.asarray(elementValues)


# matplotlib spellings of our options, refused in **kwargs
_KWARGS_ALIASES: dict[str, tuple[str, ...]] = {
    "color": ("c", "facecolor", "facecolors", "fc"),
    "edgecolor": ("edgecolors", "ec"),
    "linewidth": ("lw", "linewidths"),
    "nColors": ("levels", "norm"),
    "clim": ("vmin", "vmax"),
}


@requires_matplotlib
def Plot(
    obj: _Simu | Mesh | _GroupElem,
    result: str | _types.FloatArray | dict | None = None,
    deformFactor: _types.Number = 0.0,
    coef: _types.Number = 1.0,
    nodeValues: bool = True,
    *,
    cmap: str = "jet",
    nColors: int = 256,
    clim: tuple[float, float] | None = None,
    colorbarTitle: str | None = None,
    plotColorbar: bool = True,
    verticalColorbar: bool = True,
    color: str | None = None,
    edgecolor: str = "black",
    linewidth: float | None = 0.5,
    alpha: float = 1.0,
    plotMesh: bool = False,
    plotNodes: bool = False,
    nodeSize: float | None = None,
    title: str = "",
    label: str | None = None,
    showGrid: bool = False,
    bounds: _types.Numbers | None = None,
    ax: Axes | None = None,
    colorbarIsClose: bool = False,
    folder: str = "",
    filename: str = "",
    **kwargs,
) -> Axes:
    """Plots an object (simulation, mesh or group of elements) with matplotlib.

    This is the rendering core that ``Plot_Mesh`` and ``_Plot_obj`` delegate to.
    It is the matplotlib counterpart of ``PyVista.Plot``: pass ``result`` to color the object by a
    scalar field, or ``color`` to draw it with a single solid color.

    Parameters
    ----------
    obj : _Simu | Mesh | _GroupElem
        object to plot
    result : str | _types.FloatArray | dict, optional
        Result used to color the object. Must be included in simu.Get_Results(), be a numpy array of size (Nn,), (Ne,) or (Ne, nPg), or a per-group parameter ``{_GroupElem: scalar | (Ne_g,) | (Ne_g, nPg)}`` (Gauss points averaged). When None, the object is drawn with ``color``, by default None
    deformFactor : float, optional
        factor used to display the deformed solution (0 means no deformations), default 0.0
    coef : float, optional
        coef to apply to the solution, by default 1.0
    nodeValues : bool, optional
        displays result to nodes otherwise displays it to elements, by default True
    cmap : str, optional
        the color map used near the figure, by default "jet" \\n
        ["jet", "seismic", "binary", "viridis"] -> https://matplotlib.org/stable/tutorials/colors/colormaps.html
    nColors : int, optional
        number of colors for colorbar, by default 256
    clim : sequence[float], optional
        Two item color bar range for scalars. Defaults to minimum and maximum of scalars array. Example: (-1, 2), by default None
    colorbarTitle : str, optional
        colorbar title, by default None
    plotColorbar : bool, optional
        displays the colorbar, by default True
    verticalColorbar : bool, optional
        color bar is vertical, by default True
    color : str, optional
        solid color used when ``result`` is None, by default None
    edgecolor : str, optional
        Color used to plot the mesh, by default 'black'
    linewidth : float, optional
        line width, by default 0.5
    alpha : float, optional
        face transparency, by default 1.0
    plotMesh : bool, optional
        displays mesh edges, by default False
    plotNodes : bool, optional
        displays the nodes, by default False
    nodeSize : float, optional
        node marker size, by default None
    title : str, optional
        figure title, by default "" (the result's name)
    label : str, optional
        legend label, by default None
    showGrid : bool, optional
        shows the grid, by default False
    bounds : sequence[float], optional
        fixed view box (xmin, xmax, ymin, ymax, zmin, zmax), by default None (fits everything drawn)
    ax : axis, optional
        Axis to use, by default None
    colorbarIsClose : bool, optional
        color bar is displayed close to the figure, by default False
    folder : str, optional
        save folder, by default "".
    filename : str, optional
        filename, by default ""
    **kwargs:
        Everything matplotlib's collection (or ``tricontourf``) accepts, our options excepted

    Returns
    -------
    Axes

    Examples
    --------
    Von Mises stress in MPa (elastic simulation):

    >>> from EasyFEA import Matplotlib
    >>> Matplotlib.Plot(simu, "Svm", coef=1e-6, colorbarTitle="σ_vm [MPa]")
    >>> Matplotlib.plt.show()

    Mesh only (no scalar field):

    >>> Matplotlib.Plot(mesh, color="cyan", plotMesh=True)
    >>> Matplotlib.plt.show()
    """

    _Check_kwargs(kwargs, _KWARGS_ALIASES)

    tic = Tic()

    simu, mesh, coordDef, inDim = _Init_obj(obj, deformFactor)  # type: ignore
    dimElem = mesh.dim  # Dimension of displayed elements
    result = _Gauss_points_averaged(mesh, result)
    clim = (None, None) if clim is None else clim  # type: ignore [assignment]
    colorbarTitle = "" if colorbarTitle is None else colorbarTitle

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

    if result is not None:
        # Get values and colorbar properties
        values = _Get_values(simu, mesh, result, nodeValues) * coef
        ticks, levels, norm, vmin, vmax = __Get_colorbar_properties(
            clim, result, values, nColors  # type: ignore [arg-type]
        )
    else:
        values = None  # type: ignore [assignment]
        norm = None

    vertices = __Get_vertices(mesh, coordDef, inDim, dimElem)

    pc = None
    if inDim == 3:

        if surfDim == 1 and plotMesh:
            ax.plot(*coordDef.T, c=edgecolor, lw=0.1, marker=".", ls="")

        if values is not None:
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
                linewidth=1.5 if surfDim == 1 else 0.5,
                label=label,
                **kwargs,
            )
            pc.set_clim(
                np.min([elementValues.min(), vmin]),
                np.max([elementValues.max(), vmax]),
            )
        else:
            # solid color
            __Add_Collection(
                ax,
                vertices,
                inDim,
                surfDim,
                facecolors=color,
                edgecolor=edgecolor if plotMesh else None,
                linewidth=linewidth,
                alpha=alpha,
                label=label,
                **kwargs,
            )

    else:

        # Plot the mesh edges (for a scalar field, edges are a dedicated collection drawn
        # underneath, matching the historical scalar-field behavior)
        if plotMesh and mesh.dim == 1:
            # mesh for 1D elements are points
            ax.plot(*coordDef[:, :2].T, c=edgecolor, lw=0.1, marker=".", ls="")
        elif plotMesh and hasResult:
            # mesh for 2D elements are lines / segments (dimElem=1 for LineCollection)
            __Add_Collection(
                ax, vertices, inDim, 1, edgecolor=edgecolor, linewidth=linewidth
            )

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
                **kwargs,
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
                linewidth=1.5 if surfDim == 1 else 0.5,
                clim=(vmin, vmax),
                label=label,
                **kwargs,
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
                linewidth=linewidth,
                alpha=alpha,
                label=label,
                **kwargs,
            )

    if plotNodes:
        ax.plot(
            *coordDef[:, :inDim].T,
            c=edgecolor,
            marker=".",
            ms=nodeSize,
            ls="",
            zorder=2.5,
        )

    _Fit_view(ax, coordDef, bounds)

    if hasResult and plotColorbar and pc is not None:
        orientation = "vertical" if verticalColorbar else "horizontal"
        if colorbarIsClose and inDim < 3:
            divider = make_axes_locatable(ax)
            side = "right" if verticalColorbar else "bottom"
            cax = divider.append_axes(side, size="10%", pad=0.1)
        else:
            cax = None
        colorbar = plt.colorbar(
            pc, ax=ax, cax=cax, ticks=ticks, orientation=orientation
        )
        colorbar.set_label(colorbarTitle)

    # Title
    if title == "" and isinstance(result, str):
        ax.set_title(rf"${__Get_latex_title(result, nodeValues)}$")
    elif title != "":
        ax.set_title(title)

    if showGrid:
        ax.grid(True)

    tic.Tac("Matplotlib", "Plot")

    # If the folder has been filled in, save the figure.
    if folder != "":
        if filename == "":
            filename = result if isinstance(result, str) else "mesh"
        Save_fig(folder, filename, transparent=False)

    return ax


@requires_matplotlib
def __Get_axis(ax: plt.Axes | Axes3D | None, inDim: int):
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
    result: str | np.ndarray,
    values: np.ndarray,
    nColors: int,
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
        levels = np.linspace(min, max, nColors)
    else:
        ticks = np.linspace(min, max, 11)
        levels = np.linspace(min, max, nColors)

    if nColors != 256:
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
    obj: _Simu | Mesh,
    deformFactor: float = 0.0,
    *,
    color: str | None = "c",
    edgecolor: str = "black",
    linewidth: float | None = 0.5,
    alpha: float = 1.0,
    plotMesh: bool = True,
    plotNodes: bool = False,
    nodeSize: float | None = None,
    title: str = "",
    label: str | None = None,
    showGrid: bool = False,
    bounds: _types.Numbers | None = None,
    ax: Axes | None = None,
    folder: str = "",
) -> Axes:
    """Plots the mesh, over the undeformed one in red when deformed.

    Parameters
    ----------
    obj : _Simu | Mesh | _GroupElem
        object containing the mesh
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    color : str, optional
        face color, default 'c' (cyan)
    edgecolor : str, optional
        edgecolor, default 'black'
    linewidth : float, optional
        line width, default 0.5
    alpha : float, optional
        face transparency, default 1.0
    plotMesh : bool, optional
        displays the edges, default True
    plotNodes : bool, optional
        displays the nodes, default False
    nodeSize : float, optional
        node marker size, default None
    title : str, optional
        figure title, by default "" (the mesh's description)
    label : str, optional
        legend label, by default None
    showGrid : bool, optional
        shows the grid, by default False
    bounds : sequence[float], optional
        fixed view box (xmin, xmax, ymin, ymax, zmin, zmax), by default None (fits everything drawn)
    ax : Axes, optional
        Axis to use, default None
    folder : str, optional
        save folder, default "".

    Returns
    -------
    Axes

    Examples
    --------
    Undeformed mesh:

    >>> import matplotlib.pyplot as plt
    >>> Matplotlib.Plot_Mesh(simu)
    >>> plt.show()

    Deformed mesh, semi-transparent faces:

    >>> Matplotlib.Plot_Mesh(simu, deformFactor=50, color="white", alpha=0.5)
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
            color=color,
            plotMesh=plotMesh,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            plotNodes=plotNodes,
            nodeSize=nodeSize,
            title=title,
            label=label,
            showGrid=showGrid,
            bounds=bounds,
            ax=ax,
        )
        if mesh.dim == 1:
            # 1D meshes display their nodes
            markCoord = coord if ax.name == "3d" else coord[:, :2]
            ax.plot(*markCoord.T, c="black", lw=linewidth, marker=".", ls="")
    else:
        # Deformed mesh: overlay the deformed (red) over the undeformed wireframe, both built
        # with the same _Get_vertices / _Add_Collection helpers used by Plot. The element
        # outlines are drawn as lines (dimElem=1) so the overlay renders identically in 2D and
        # 3D without relying on transparent faces.
        ax, inDim = __Get_axis(ax, inDim)
        ax.set_title(title)

        verticesDef = __Get_vertices(mesh, coordDef, inDim, mesh.dim)
        vertices = __Get_vertices(mesh, coord, inDim, mesh.dim)

        __Add_Collection(
            ax, verticesDef, inDim, 1, edgecolor="red", linewidth=linewidth, label=label
        )
        __Add_Collection(
            ax, vertices, inDim, 1, edgecolor=edgecolor, linewidth=linewidth
        )

        if mesh.dim == 1 or plotNodes:
            # undeformed nodes in black, deformed in red
            markCoord = coord if inDim == 3 else coord[:, :2]
            markDef = coordDef if inDim == 3 else coordDef[:, :2]
            ax.plot(*markCoord.T, c="black", ms=nodeSize, marker=".", ls="")
            ax.plot(*markDef.T, c="red", ms=nodeSize, marker=".", ls="")

        _Fit_view(ax, np.concatenate((coord, coordDef)), bounds)

        if showGrid:
            ax.grid(True)

    tic.Tac("Matplotlib", "Plot_Mesh")

    if folder != "":
        Save_fig(folder, "mesh")

    return ax  # type: ignore


@requires_matplotlib
def _Plot_obj(
    obj: _Simu | Mesh | _GroupElem,
    alpha: float = 1.0,
    color: str = "gray",
    ax: Axes | None = None,
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
    nodes: _types.IntArray | None = None,
    showId: bool = False,
    *,
    deformFactor: float = 0.0,
    color: str | None = "red",
    alpha: float = 1.0,
    nodeSize: float | None = None,
    title: str = "",
    label: str | None = None,
    showGrid: bool = False,
    bounds: _types.Numbers | None = None,
    ax: Axes | None = None,
    marker: str = ".",
) -> Axes:
    """Plots the mesh's nodes.

    Parameters
    ----------
    obj : _Simu | Mesh | _GroupElem
        object containing the mesh
    nodes : _types.IntArray, optional
        nodes to display, default None (all)
    showId : bool, optional
        display numbers, default False
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    color : str, optional
        color, default 'red'
    alpha : float, optional
        transparency, default 1.0
    nodeSize : float, optional
        marker size, default None
    title : str, optional
        figure title, by default ""
    label : str, optional
        legend label, by default None
    showGrid : bool, optional
        shows the grid, by default False
    bounds : sequence[float], optional
        fixed view box (xmin, xmax, ymin, ymax, zmin, zmax), by default None (fits everything drawn)
    ax : Axes, optional
        Axis to use, default None
    marker : str, optional
        marker type (matplotlib.markers), default '.'

    Returns
    -------
    Axes
    """

    tic = Tic()

    _, mesh, coord, inDim = _Init_obj(obj, deformFactor)

    if ax is None:
        ax = Init_Axes(inDim)
    else:
        inDim = 3 if ax.name == "3d" else inDim
    inDim = max(inDim, 2)

    if nodes is None:
        nodes = mesh.nodes
    else:
        nodes = np.asarray(list(set(np.ravel(nodes))))

    if nodes.size == 0:
        return ax

    ax.plot(
        *coord[nodes, :inDim].T,
        ls="",
        marker=marker,
        c=color,
        ms=nodeSize,
        alpha=alpha,
        label=label,
        zorder=2.5,
    )
    if showId:
        [ax.text(*coord[node, :inDim].T, str(node), c=color) for node in nodes]  # type: ignore [call-arg]

    _Fit_view(ax, coord[nodes], bounds)

    if title != "":
        ax.set_title(title)
    if showGrid:
        ax.grid(True)

    tic.Tac("Matplotlib", "Plot_Nodes")

    return ax


@requires_matplotlib
def Plot_Elements(
    obj,
    nodes: _types.IntArray | None = None,
    dimElem: int | None = None,
    showId: bool = False,
    *,
    deformFactor: float = 0.0,
    color: str | None = "red",
    edgecolor: str = "black",
    linewidth: float | None = None,
    alpha: float = 1.0,
    plotMesh: bool = True,
    plotNodes: bool = False,
    nodeSize: float | None = None,
    title: str = "",
    label: str | None = None,
    showGrid: bool = False,
    bounds: _types.Numbers | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plots the mesh's elements corresponding to the given nodes.

    Parameters
    ----------
    obj : _Simu | Mesh | _GroupElem
        object containing the mesh
    nodes : _types.IntArray, optional
        node numbers, by default None (all elements)
    dimElem : int, optional
        dimension of elements, by default None
    showId : bool, optional
        display numbers, by default False
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    color : str, optional
        color used to display faces, by default 'red'
    edgecolor : str, optional
        color used to display segments, by default 'black'
    linewidth : float, optional
        line width, by default None (1 for lines, 0.5 for edges)
    alpha : float, optional
        transparency of faces, by default 1.0
    plotMesh : bool, optional
        displays the edges, by default True
    plotNodes : bool, optional
        displays the elements' nodes, by default False
    nodeSize : float, optional
        node marker size, by default None
    title : str, optional
        figure title, by default ""
    label : str, optional
        legend label, by default None
    showGrid : bool, optional
        shows the grid, by default False
    bounds : sequence[float], optional
        fixed view box (xmin, xmax, ymin, ymax, zmin, zmax), by default None (fits everything drawn)
    ax : Axes, optional
        Axis to use, default None

    Returns
    -------
    Axes
    """

    tic = Tic()

    _, mesh, coord, inDim = _Init_obj(obj, deformFactor)

    if dimElem is None:
        dimElem = 2 if inDim == 3 else mesh.dim

    ax, inDim = __Get_axis(ax, inDim)

    plotDim = np.max([inDim, 2])

    # list of element group associated with the dimension
    list_groupElem = mesh.Get_list_groupElem(dimElem)
    if len(list_groupElem) == 0:
        return None  # type: ignore

    drawn: list[_types.IntArray] = []
    # for each group elem
    for groupElem in list_groupElem:
        # get the elements associated with the nodes
        if nodes is not None and len(nodes) > 0:
            elements = groupElem.Get_Elements_Nodes(nodes)
        else:
            elements = np.arange(groupElem.Ne)

        if elements.size == 0:
            continue

        # get params
        if groupElem.dim == 1:
            # 1D elements
            idx = groupElem.segments.ravel().tolist()
            params: dict[str, Any] = {
                "edgecolor": color,
                "linewidth": 1 if linewidth is None else linewidth,
                "zorder": 2,
            }
        else:
            # 2D elements
            idx = groupElem.surfaces.ravel().tolist()
            params = {
                "facecolors": color,
                "edgecolor": edgecolor if plotMesh else None,
                "linewidth": 0.5 if linewidth is None else linewidth,
                "alpha": alpha,
                "zorder": 2,
            }
        if len(drawn) == 0:
            params["label"] = label

        # Construct the vertices coordinates
        connect_e = groupElem.connect  # connect
        vertices_e = coord[connect_e[:, idx], :plotDim]
        vertices = vertices_e[elements]
        drawn.append(connect_e[elements].ravel())

        # center coordinates for each elements
        center_e = np.mean(vertices_e, axis=1)

        __Add_Collection(ax, vertices, plotDim, groupElem.dim, **params)

        if showId:
            # plot elements id's
            [
                ax.text(  # type: ignore [call-arg]
                    *center_e[element], element, zorder=25, ha="center", va="center"
                )
                for element in elements
            ]

    tic.Tac("Matplotlib", "Plot_Elements")

    if len(drawn) > 0:
        drawnNodes = np.unique(np.concatenate(drawn))
        if plotNodes:
            ax.plot(
                *coord[drawnNodes, :plotDim].T,
                c=edgecolor,
                marker=".",
                ms=nodeSize,
                ls="",
                zorder=2.5,
            )
        _Fit_view(ax, coord[drawnNodes], bounds)

    if title != "":
        ax.set_title(title)
    if showGrid:
        ax.grid(True)

    return ax


@requires_matplotlib
def Plot_BoundaryConditions(
    simu,
    *,
    deformFactor: float = 0.0,
    alpha: float = 0.0,
    nodeSize: float | None = None,
    title: str = "Boundary conditions",
    showGrid: bool = False,
    plotLegend: bool = True,
    bounds: _types.Numbers | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plots simulation's boundary conditions.

    Parameters
    ----------
    simu : _Simu
        simulation
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    alpha : float, optional
        transparency of the mesh's faces drawn underneath, default 0.0 (edges only)
    nodeSize : float, optional
        marker size, default None
    title : str, optional
        figure title, by default "Boundary conditions"
    showGrid : bool, optional
        shows the grid, by default False
    plotLegend : bool, optional
        displays the legend, by default True
    bounds : sequence[float], optional
        fixed view box (xmin, xmax, ymin, ymax, zmin, zmax), by default None (fits everything drawn)
    ax : Axes, optional
        Axis to use, default None

    Returns
    -------
    Axes

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> Matplotlib.Plot_BoundaryConditions(simu)
    >>> plt.show()

    Combined with mesh overlay:

    >>> ax = Matplotlib.Plot_Mesh(simu)
    >>> Matplotlib.Plot_BoundaryConditions(simu, ax=ax)
    >>> plt.show()
    """

    tic = Tic()

    simu, _, coord, _ = _Init_obj(simu, deformFactor)

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
        ax = Plot_Elements(simu, dimElem=1, color="k", deformFactor=deformFactor)
        if alpha > 0:
            Plot(simu, None, deformFactor, color="gray", alpha=alpha, ax=ax)

    plotDim = np.max([simu.mesh.inDim, 2])

    for bc in BoundaryConditions:
        dofsValues = bc.dofsValues
        unknowns = bc.unknowns
        nDir = len(unknowns)
        nodes = list(set(list(bc.nodes)))
        description = bc.description

        marker = "."
        if not set(unknowns) <= {"x", "y", "z", "rx", "ry", "rz"}:
            marker = "o"
        else:
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

        # Title
        unknowns_str = str(unknowns).replace("'", "")
        label = f"{description} {unknowns_str}"

        if len(nodes) == simu.mesh.Nn:
            points = coord[:, :plotDim].mean(0, keepdims=True)
        else:
            points = coord[nodes, :plotDim]
        ax.plot(*points.T, marker=marker, ms=nodeSize, label=label, zorder=2.5, ls="")

    _Fit_view(ax, coord, bounds)

    if title != "":
        ax.set_title(title)
    if showGrid:
        ax.grid(True)
    if plotLegend:
        ax.legend()

    tic.Tac("Matplotlib", "Plot_BoundaryConditions")

    return ax


@requires_matplotlib
def Plot_Geoms(
    *geoms: _Geom | list[_Geom],
    color: str | None = None,
    linewidth: _types.Number | None = None,
    alpha: float = 1.0,
    title: str = "",
    label: str | None = None,
    showGrid: bool = True,
    plotLegend: bool = True,
    bounds: _types.Numbers | None = None,
    ax: Axes | None = None,
    ls: str | None = None,
    plotPoints: bool = True,
) -> Axes:
    """Plots geometric objects, or lists of them, on the same axis; `label` replaces each geom's name.

    Examples
    --------
    >>> from EasyFEA import Matplotlib
    >>> from EasyFEA.Geoms import Domain, Circle
    >>> domain = Domain((0, 0), (1, 1))
    >>> circle = Circle((0.5, 0.5), 0.2)
    >>> ax = Matplotlib.Plot_Geoms(domain, circle)
    """

    from ..Geoms import Point

    for geom in _Flatten_geoms(geoms):
        if isinstance(geom, Point):
            continue

        lines, points = geom.Get_coord_for_plot()

        if ax is None:
            ax = Init_Axes(2 if np.abs(lines[:, 2].max()) == 0 else 3)

        inDim = 3 if ax.name == "3d" else 2
        name = geom.name if label is None else label

        ax.plot(
            *lines[:, :inDim].T,
            color=color,
            label=name,
            lw=linewidth,
            ls=ls,
            alpha=alpha,
        )
        if plotPoints:
            ax.plot(*points[:, :inDim].T, ls="", marker=".", c="black", alpha=alpha)

        _Fit_view(ax, lines, bounds)

    if ax is None:
        ax = Init_Axes(2)

    if title != "":
        ax.set_title(title)
    if showGrid:
        ax.grid(True)
    if plotLegend:
        ax.legend()

    return ax


@requires_matplotlib
def Plot_Tags(
    obj,
    *,
    deformFactor: float = 0.0,
    alpha: float = 1.0,
    linewidth: float | None = None,
    title: str = "",
    showId: bool = True,
    showGrid: bool = False,
    plotLegend: bool = False,
    useColorCycler: bool = False,
    bounds: _types.Numbers | None = None,
    ax: Axes | None = None,
    folder: str = "",
) -> Axes:
    """Plots the mesh's elements tags (from 2d elements to points) but do not plot the 3d elements tags.

    Parameters
    ----------
    obj : _Simu | Mesh | _GroupElem
        object containing the mesh
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    alpha : float, optional
        transparency, by default 1.0
    linewidth : float, optional
        width of the tagged lines, by default None (1.5)
    title : str, optional
        figure title, by default ""
    showId : bool, optional
        writes the tags, by default True
    showGrid : bool, optional
        shows the grid, by default False
    plotLegend : bool, optional
        displays the legend, by default False (the tag under the mouse shows in the toolbar)
    useColorCycler : bool, optional
        whether to use color cycler, by default False
    bounds : sequence[float], optional
        fixed view box (xmin, xmax, ymin, ymax, zmin, zmax), by default None (fits everything drawn)
    ax : Axes, optional
        Axis to use, default None
    folder : str, optional
        saves folder, by default ""

    Returns
    -------
    Axes
    """

    tic = Tic()

    _, mesh, coord, inDim = _Init_obj(obj, deformFactor)

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

    Plot(obj, None, deformFactor, color="gray", alpha=0.1, ax=ax)

    colors = plt.get_cmap("tab10").colors  # type: ignore [attr-defined]
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
                    ax,
                    vertices,
                    inDim,
                    1,
                    edgecolor="black",
                    linewidth=1.5 if linewidth is None else linewidth,
                    alpha=1,
                    label=tag_e,
                )
                collections.append(pc)

            elif dim == 2:
                # plot surfaces
                pc = __Add_Collection(
                    ax,
                    vertices,
                    inDim,
                    2,
                    facecolors=color,
                    linewidth=0,
                    alpha=alpha,
                    label=tag_e,
                )
                collections.append(pc)

            if showId:
                ax.text(*center[:inDim], tag_e, zorder=25)  # type: ignore [arg-type, call-arg]

    _Fit_view(ax, coord, bounds)

    if title != "":
        ax.set_title(title)
    if showGrid:
        ax.grid(True)
    if plotLegend:
        ax.legend()

    tic.Tac("Matplotlib", "Plot_Tags")

    if folder != "":
        Save_fig(folder, "geom")

    __Annotation_Event(collections, ax.figure, ax)

    return ax


@requires_matplotlib
def __Annotation_Event(collections: list, fig: plt.Figure | Any, ax: Axes) -> None:
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
    simu: _Simu,
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


@rank0_only
@requires_matplotlib
def Plot_Tic_History(folder="", details=False) -> None:
    """Plots the `Tic` history, per category and, with `details`, per subcategory."""

    history = Tic.Get_History()
    if history == {}:
        return

    # Calculate total time per category
    categories = np.array(list(history.keys()))
    timesPerCategory = np.array(
        [sum(v[0] for v in history[c].values()) for c in categories]
    )

    # Sort categories by descending time
    sorted_indices = np.argsort(timesPerCategory)[::-1]
    categories = categories[sorted_indices]
    timesPerCategory = timesPerCategory[sorted_indices]

    totalTime = []
    for i, c in enumerate(categories):
        # Extract aggregated data: { text: [total_time, count] }
        subcats = history[c]
        unique_subcats = np.array(list(subcats.keys()))
        time_by_subcat = np.array([subcats[s][0] for s in unique_subcats])
        rep_by_subcat = np.array([subcats[s][1] for s in unique_subcats], dtype=int)

        totalTime.append(float(time_by_subcat.sum()))

        # Plot subcategories if needed
        if len(unique_subcats) > 1 and details and totalTime[-1] > 0:
            # Sort subcategories by descending time
            sorted_subcat_indices = np.argsort(time_by_subcat)[::-1]
            ax = plt.subplots()[1]
            _Plot_Bar(
                ax,
                unique_subcats[sorted_subcat_indices],
                time_by_subcat[sorted_subcat_indices],
                rep_by_subcat[sorted_subcat_indices],
                c,
            )
            if folder != "":
                Save_fig(folder, f"TicTac{i}_{c}")

    # Plot summary of categories
    ax = plt.subplots()[1]
    _Plot_Bar(ax, categories, timesPerCategory, [1] * len(categories), "Summary")
    if folder != "":
        Save_fig(folder, "TicTac_Summary")


@requires_matplotlib
def _Plot_Bar(
    ax: plt.Axes,
    categories: Sequence[str] | _types.AnyArray,
    times: Sequence[float] | _types.AnyArray,
    reps: Sequence[int] | _types.AnyArray,
    title: str,
) -> None:
    ax.xaxis.set_tick_params(labelbottom=False, labeltop=True, length=0)
    ax.yaxis.set_visible(False)
    ax.set_axisbelow(True)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)

    ax.grid(axis="x", lw=1.2)

    timeMax = np.max(times)
    Ncategory = len(categories)

    for i, (category, time, rep) in enumerate(zip(categories, times, reps)):
        y_pos = Ncategory - 1 - i
        ax.barh(y_pos, time, align="center", label=category)

        space = " "

        unitTime, unit = Tic.Get_time_unity(time / rep)

        if rep > 1:
            repTemps = f" ({rep} x {np.round(unitTime, 2)} {unit})"
        else:
            repTemps = f" ({np.round(unitTime, 2)} {unit})"

        category = space + category + repTemps + space

        if time / timeMax < 0.6:
            ax.text(
                time,
                y_pos,
                category,
                color="black",
                verticalalignment="center",
                horizontalalignment="left",
            )
        else:
            ax.text(
                time,
                y_pos,
                category,
                color="white",
                verticalalignment="center",
                horizontalalignment="right",
            )

    ax.set_title(title)


# ----------------------------------------------
# Animation
# ----------------------------------------------
@rank0_only
@requires_matplotlib
def Movie_simu(
    simu,
    result: str,
    folder: str,
    filename: str = "video.gif",
    N: int = 200,
    deformFactor: float = 0.0,
    coef: float = 1.0,
    nodeValues: bool = True,
    *,
    fps: int = 30,
    **kwargs: Unpack[PlotOptions],
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
    fps : int, optional
        frames per second, by default 30
    **kwargs:
        `Plot` options, e.g. `plotMesh`, `clim` or `bounds`
    """

    simu, _, _, inDim = _Init_obj(simu)

    if simu is None:
        MyPrintError("Must give a simulation.")
        return

    Niter = simu.Niter
    N = np.min([Niter, N])
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
            deformFactor,
            coef,
            nodeValues,
            ax=ax,
            **kwargs,
        )
        ax.set_title(f"{result} {iterations[i]:d}/{Niter - 1:d}")

    Movie_func(DoAnim, iterations.size, folder, filename, fps=fps, fig=fig)


@rank0_only
@requires_matplotlib
def Movie_func(
    func: Callable[[plt.Figure], None] | Callable[[plt.Figure, int], None],
    N: int,
    folder: str,
    filename: str = "video.gif",
    *,
    fps: int = 30,
    fig: plt.Figure | Any | None = None,
    dpi: int = 200,
    show: bool = True,
) -> None:
    """Generates the movie for the specified function.\\n
    This function will peform a loop in range(N).

    Parameters
    ----------
    func : Callable[[plt.Figure, int], None]
        The function that will use in first argument the figure and in second argument the iter step such that.\\n
        def func(fig, i) -> None
    N : int
        number of iteration
    folder : str
        folder where you want to save the video
    filename : str, optional
        filename of the video with the extension (eg. .gif, .mp4), by default 'video.gif'
    fps : int, optional
        frames per second, by default 30
    fig : Figure, optional
        Figure used to make the video, by default None (a new one)
    dpi: int, optional
        Dots per Inch, by default 200
    show: bool, optional
        shows the movie, by default True
    """

    if fig is None:
        fig = plt.figure()

    # Name of the video in the folder where the folder is communicated
    filename = Folder.Join(folder, filename, mkdir=True)

    writer = animation.FFMpegWriter(fps)
    with writer.saving(fig, filename, dpi):  # type: ignore [arg-type]
        tic = Tic()
        for i in range(N):
            func(fig, i)  # type: ignore [call-arg]

            if show:
                plt.pause(1e-12)

            writer.grab_frame()

            time = tic.Tac("Matplotlib", "Movie_func", False)

            iteration = i + 1
            rmTime = Tic.Get_Remaining_Time(iteration, N, time)

            iteration = str(iteration).zfill(len(str(N)))  # type: ignore [assignment]
            MyPrint(f"Generate movie {iteration}/{N} {rmTime}    ", end="\r")


# ----------------------------------------------
# Functions
# ----------------------------------------------

# view state kept on the axes
_DRAWN_BOUNDS = "_easyfea_drawn_bounds"


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

    tic.Tac("Matplotlib", "Save figure")


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
def _Fit_view(
    ax: Axes, coord: _types.FloatArray, bounds: _types.Numbers | None = None
) -> None:
    """Frames `ax` on everything drawn in it so far, `coord` included, or on the fixed `bounds`."""
    is3D = ax.name == "3d"
    if bounds is not None:
        lims = np.reshape(np.asarray(bounds, dtype=float), (-1, 2))
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
        if is3D:
            ax.set_zlim(*lims[2])  # type: ignore [union-attr]
            ax.set_box_aspect([1, 1, 1])  # type: ignore [arg-type]
        else:
            ax.set_aspect("equal", adjustable="box")
    elif is3D:
        # 3D axes do not autoscale on collections: keep the union on the axes
        drawn = _Union_bounds(getattr(ax, _DRAWN_BOUNDS, None), coord)
        setattr(ax, _DRAWN_BOUNDS, drawn)
        _Axis_equal_3D(ax, drawn)  # type: ignore [arg-type]
    else:
        ax.autoscale()
        ax.axis("equal")
