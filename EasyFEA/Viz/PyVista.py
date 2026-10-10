# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module providing an interface with PyVista (https://docs.pyvista.org/version/stable/).\n
https://docs.pyvista.org/api/plotting/plotting.html"""

from __future__ import annotations
from typing import Callable, TYPE_CHECKING, Any, Iterable
from scipy.sparse import csr_matrix
import numpy as np
from functools import singledispatch

# utilities
from ..Utilities import Folder, Terminal, Tic, _types
from ..IO._utils import _Init_obj, _Get_values
from ..IO import PyVista as _pvIO
from ._utils import (
    tab10_colors,
    PlotOptions,
    _Flatten_geoms,
    _Check_kwargs,
    _Gauss_points_averaged,
)
from .. import Geoms

# fem
from ..FEM import GroupElemFactory

if TYPE_CHECKING:
    from typing import Unpack
    from ..Simulations._simu import _Simu
    from ..FEM._mesh import Mesh
    from ..FEM._group_elem import _GroupElem

from ..Utilities._requires import Create_requires_decorator
from ..Utilities._mpi import rank0_only

try:
    import pyvista as pv
except ImportError:
    pass
requires_pyvista = Create_requires_decorator("matplotlib", "pyvista")


# pyvista spellings of our options, refused in **kwargs
_KWARGS_ALIASES: dict[str, tuple[str, ...]] = {
    "result": ("scalars",),
    "plotMesh": ("show_edges",),
    "edgecolor": ("edge_color",),
    "linewidth": ("line_width",),
    "plotNodes": ("show_vertices",),
    "nodeSize": ("point_size",),
    "alpha": ("opacity",),
    "nColors": ("n_colors",),
    "plotColorbar": ("show_scalar_bar",),
    "scalar_bar_kwargs": ("scalar_bar_args",),
    "showGrid": ("show_grid",),
}


@requires_pyvista
def Plot(
    obj: _Simu | Mesh | _GroupElem | Any,
    result: str | _types.FloatArray | dict | None = None,
    deformFactor: float = 0.0,
    coef: float = 1.0,
    nodeValues: bool = True,
    *,
    cmap: str = "jet",
    nColors: int = 256,
    clim: tuple[float, float] | None = None,
    colorbarTitle: str | None = None,
    plotColorbar: bool = True,
    verticalColorbar: bool = True,
    color: str | None = None,
    edgecolor: str = "k",
    linewidth: float | None = None,
    alpha: float = 1.0,
    plotMesh: bool = False,
    plotNodes: bool = False,
    nodeSize: float | None = None,
    title: str = "",
    label: str | None = None,
    showGrid: bool = False,
    bounds: _types.Numbers | None = None,
    plotter: pv.Plotter | None = None,
    style: str = "surface",
    scalar_bar_kwargs: dict | None = None,
    **kwargs,
):
    """Plots the object obj that can be either a simu, mesh, MultiBlock, PolyData.\\n
    If you want to plot the solution use plotter.show().

    Parameters
    ----------
    obj : _Simu | Mesh | _GroupElem | MultiBlock | PolyData | UnstructuredGrid
        The object to plot and will be transformed to a mesh
    result : str | _types.FloatArray | dict, optional
        Scalars used to “color” the mesh, an (Ne, nPg) array or a per-group parameter ``{_GroupElem: scalar | (Ne_g,) | (Ne_g, nPg)}`` (Gauss points averaged), by default None
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    coef : float, optional
        Coef to apply to the solution, by default 1.0
    nodeValues : bool, optional
        Displays result to nodes otherwise displays it to elements, by default True
    cmap : str, optional
        If a string, this is the name of the matplotlib colormap to use when mapping the scalars, by default "jet"\\n
        ["jet", "seismic", "binary"] -> https://matplotlib.org/stable/tutorials/colors/colormaps.html
    nColors : int, optional
        Number of colors to use when displaying scalars, by default 256
    clim : sequence[float], optional
        Two item color bar range for scalars. Defaults to minimum and maximum of scalars array. Example: [-1, 2], by default None
    colorbarTitle: str, optional
        colorbar title, by default None (the result's name)
    plotColorbar : bool, optional
        displays the colorbar, by default True
    verticalColorbar : bool, optional
        color bar is vertical, by default True
    color : str, optional
        Use to make the entire mesh have a single solid color, by default None
    edgecolor : str, optional
        The solid color to give the edges when plotMesh=True, by default 'k'
    linewidth : float, optional
        Thickness of lines. Only valid for wireframe and surface representations, by default None
    alpha : float | str | ndarray, optional
        Opacity of the mesh, by default 1.0
    plotMesh : bool, optional
        Shows the edges of a mesh. Does not apply to a wireframe representation, by default False
    plotNodes : bool, optional
        Shows the nodes, by default False
    nodeSize : float, optional
        Point size of the nodes plotted when plotNodes=True, by default None
    title : str, optional
        title, by default ""
    label : str, optional
        legend label, by default None
    showGrid : bool, optional
        Show the grid, by default False
    bounds : sequence[float], optional
        fixed view box (xmin, xmax, ymin, ymax, zmin, zmax), by default None (fits everything drawn)
    plotter : pv.Plotter, optional
        The pyvista plotter, by default None and create a new Plotter instance
    style : str, optional
        Visualization style of the mesh. One of the following: ['surface', 'wireframe', 'points', 'points_gaussian'], by default 'surface'
    scalar_bar_kwargs : dict, optional
        Extra scalar-bar options, merged over the ones built from ``colorbarTitle`` and ``verticalColorbar`` and taking precedence over them, by default None.
        Useful when the defaults are unreadable — a horizontal bar in a narrow subplot needs ``n_labels`` and ``fmt`` to stop the ticks overlapping::

            PyVista.Plot(simu, "uz", verticalColorbar=False,
                         scalar_bar_kwargs={"n_labels": 3, "fmt": "%.1e"})

        Note that pyvista keys scalar bars by title, so several bars in one figure need distinct ``title`` entries.
        Everything accepted by https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_scalar_bar
    **kwargs:
        Everything that can goes in add_mesh function https://docs.pyvista.org/version/stable/api/plotting/_autosummary/pyvista.Plotter.add_mesh.html#pyvista.Plotter.add_mesh, our options excepted

    Returns
    -------
    pv.Plotter
        The pyvista plotter

    Examples
    --------
    Von Mises stress in MPa (elastic simulation):

    >>> from EasyFEA import PyVista
    >>> plotter = PyVista.Plot(simu, result="Svm", coef=1e-6, colorbarTitle="σ_vm [MPa]")
    >>> plotter.show()

    With deformed mesh:

    >>> plotter = PyVista.Plot(simu, result="displacement_norm", deformFactor=100)
    >>> plotter.show()

    Mesh only (no scalar field):

    >>> plotter = PyVista.Plot(mesh)
    >>> plotter.show()
    """

    _Check_kwargs(kwargs, _KWARGS_ALIASES)

    tic = Tic()

    # initilize the obj to construct the grid
    if isinstance(obj, (pv.MultiBlock, pv.PolyData, pv.UnstructuredGrid)):
        inDim = 3
        pvMesh = obj
        result = result if result in pvMesh.array_names else None

    else:
        pvMesh = _pvMesh(obj, result, deformFactor, nodeValues)
        inDim = _Init_obj(obj)[-1]

    if pvMesh is None:
        # something do not work during the grid creation≠
        raise TypeError("Issue during UnstructuredGrid creation process")

    # apply coef to the array
    name = "array" if isinstance(result, (np.ndarray, dict)) else result
    name = None if pvMesh.n_arrays == 0 else name
    if name is not None:
        pvMesh[name] *= coef  # type: ignore [operator, call-overload]

    colorbarTitle = name if colorbarTitle is None else colorbarTitle

    if plotter is None:
        plotter = _Plotter()

    if verticalColorbar:
        pos = "position_x"
        val = 0.85
    else:
        pos = "position_y"
        val = 0.025

    # caller options win, so `scalar_bar_kwargs` can override the title, the orientation or the position as well as add to them
    scalar_bar_args = {
        "title": colorbarTitle,
        "vertical": verticalColorbar,
        pos: val,
    }
    if scalar_bar_kwargs is not None:
        scalar_bar_args.update(scalar_bar_kwargs)

    plotter.add_mesh(
        pvMesh,
        scalars=name,
        color=color,
        show_edges=plotMesh,
        edge_color=edgecolor,
        line_width=linewidth,
        show_vertices=plotNodes,
        point_size=nodeSize,
        opacity=alpha,
        style=style,  # type: ignore [arg-type]
        cmap=cmap,  # type: ignore [arg-type]
        n_colors=nColors,
        clim=clim,
        show_scalar_bar=plotColorbar if name is not None else None,
        scalar_bar_args=scalar_bar_args,  # type: ignore [arg-type]
        label=label,
        **kwargs,
    )

    _Annotate(plotter, inDim, title, showGrid, bounds)

    tic.Tac("PyVista_Interface", "Plot")

    return plotter


@requires_pyvista
def Plot_Mesh(
    obj: _Simu | Mesh | Any,
    deformFactor: float = 0.0,
    *,
    color: str | None = "cyan",
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
    plotter: pv.Plotter | None = None,
):
    """Plots the mesh.

    Parameters
    ----------
    obj : _Simu | Mesh | MultiBlock | PolyData | UnstructuredGrid
        object containing the mesh
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    color: str, optional
        face colors, default 'cyan'
    edgecolor: str, optional
        edge color, default 'black'
    linewidth: float, optional
        line width, default 0.5
    alpha : float, optional
        face opacity, default 1.0
    plotMesh : bool, optional
        shows the edges, default True
    plotNodes : bool, optional
        shows the nodes, default False
    nodeSize : float, optional
        node size, default None
    title : str, optional
        title, by default ""
    label : str, optional
        legend label, by default None
    showGrid : bool, optional
        Show the grid, by default False
    bounds : sequence[float], optional
        fixed view box (xmin, xmax, ymin, ymax, zmin, zmax), by default None (fits everything drawn)
    plotter : pv.Plotter, optional
        The pyvista plotter, by default None and create a new Plotter instance

    Returns
    -------
    pv.Plotter
        The pyvista plotter

    Examples
    --------
    Undeformed mesh:

    >>> from EasyFEA import PyVista
    >>> plotter = PyVista.Plot_Mesh(mesh)
    >>> plotter.show()

    Deformed mesh with transparency:

    >>> plotter = PyVista.Plot_Mesh(simu, deformFactor=50, alpha=0.5)
    >>> plotter.show()
    """

    plotter = Plot(
        obj,
        None,
        deformFactor,
        color=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        plotMesh=plotMesh,
        plotNodes=plotNodes,
        nodeSize=nodeSize,
        title=title,
        label=label,
        showGrid=showGrid,
        bounds=bounds,
        plotter=plotter,
    )

    return plotter


@requires_pyvista
def Plot_Nodes(
    obj: _Simu | Mesh,
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
    plotter: pv.Plotter | None = None,
):
    """Plots mesh's nodes.

    Parameters
    ----------
    obj : _Simu | Mesh
        object containing the mesh
    nodes : _types.IntArray, optional
        nodes to display, or a (n, 3) array of points, default None (all)
    showId : bool, optional
        display node numbers, default False
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    color : str, optional
        color, default 'red'
    alpha : float, optional
        opacity, default 1.0
    nodeSize : float, optional
        point size, default None
    title : str, optional
        title, by default ""
    label : str, optional
        label, by default None
    showGrid : bool, optional
        Show the grid, by default False
    bounds : sequence[float], optional
        fixed view box (xmin, xmax, ymin, ymax, zmin, zmax), by default None (fits everything drawn)
    plotter : pv.Plotter, optional
        The pyvista plotter, by default None and create a new Plotter instance

    Returns
    -------
    pv.Plotter
        The pyvista plotter
    """

    _, mesh, coord, inDim = _Init_obj(obj, deformFactor)

    if nodes is None:
        nodes = mesh.nodes
        coord = coord[nodes]
    else:
        nodes = np.asarray(nodes)

        if nodes.ndim == 1:
            if nodes.size == 0:
                Terminal.MyPrintError("The list of nodes is empty.")
                return
            if nodes.size > mesh.Nn:
                Terminal.MyPrintError("The list of nodes must be of size <= mesh.Nn")
                return
            else:
                coord = coord[nodes]
        elif nodes.ndim == 2 and nodes.shape[1] == 3:
            coord = nodes  # type: ignore [assignment]
        else:
            Terminal.MyPrintError(
                "Nodes must be either a list of nodes or a matrix of 3D vectors of dimension (n, 3)."
            )
            return

    if plotter is None:
        plotter = Plot(obj, None, deformFactor, style="wireframe", color="k")

    pvData = pv.PolyData(coord)  # type: ignore [arg-type]

    if showId:
        myLabels: list[str] = [f"{node}" for node in nodes]
        pvData["myLabels"] = myLabels  # type: ignore [type-var]
        plotter.add_point_labels(
            pvData,
            "myLabels",
            point_color=color,
            point_size=nodeSize,
            render_points_as_spheres=True,
        )
    else:
        plotter.add_mesh(
            pvData,
            color=color,
            opacity=alpha,
            point_size=nodeSize,
            label=label,
            render_points_as_spheres=True,
        )

    _Annotate(plotter, inDim, title, showGrid, bounds)

    return plotter


@requires_pyvista
def Plot_Elements(
    obj: _Simu | Mesh,
    nodes: _types.IntArray | None = None,
    dimElem: int | None = None,
    showId: bool = False,
    *,
    deformFactor: float = 0.0,
    color: str | None = "red",
    edgecolor: str = "black",
    linewidth: float | None = None,
    alpha: float = 1.0,
    plotMesh: bool = False,
    plotNodes: bool = False,
    nodeSize: float | None = None,
    title: str = "",
    label: str | None = None,
    showGrid: bool = False,
    bounds: _types.Numbers | None = None,
    plotter: pv.Plotter | None = None,
):
    """Plots the mesh elements corresponding to the given nodes.

    Parameters
    ----------
    obj : _Simu | Mesh
        object containing the mesh
    nodes : _types.IntArray, optional
        nodes used by elements, default None (all elements)
    dimElem : int, optional
        dimension of elements, by default None (mesh.dim)
    showId : bool, optional
        display numbers, by default False
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    color : str, optional
        color used to display faces, by default 'red
    edgecolor : str, optional
        color used to display segments, by default 'black'
    linewidth : float, optional
        Thickness of lines, by default None
    alpha : float, optional
        transparency of faces, by default 1.0
    plotMesh : bool, optional
        shows the edges, by default False
    plotNodes : bool, optional
        shows the nodes, by default False
    nodeSize : float, optional
        node size, by default None
    title : str, optional
        title, by default ""
    label : str, optional
        label, by default None
    showGrid : bool, optional
        Show the grid, by default False
    bounds : sequence[float], optional
        fixed view box (xmin, xmax, ymin, ymax, zmin, zmax), by default None (fits everything drawn)
    plotter : pv.Plotter, optional
        The pyvista plotter, by default None and create a new Plotter instance

    Returns
    -------
    pv.Plotter
        The pyvista plotter
    """

    _, mesh, coord, inDim = _Init_obj(obj, deformFactor)

    dimElem = mesh.dim if dimElem is None else dimElem

    if plotter is None:
        plotter = _Plotter()

    for groupElem in mesh.Get_list_groupElem(dimElem):
        # get the elements associated with the nodes
        if nodes is not None and len(nodes) > 0:
            elements = groupElem.Get_Elements_Nodes(nodes)
        else:
            elements = np.arange(groupElem.Ne)

        if elements.size == 0:
            continue

        # construct the new group element by changing the connectivity matrix
        connect = groupElem.connect[elements]
        newGroupElem = GroupElemFactory.Create(groupElem.elemType, connect, coord)

        pvGroup = _pvMesh(newGroupElem)  # type: ignore [arg-type]

        Plot(
            pvGroup,
            color=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            plotMesh=plotMesh,
            plotNodes=plotNodes,
            nodeSize=nodeSize,
            label=label,
            plotter=plotter,
        )

        if showId:
            centers = np.mean(coord[groupElem.connect[elements]], axis=1)
            pvData = pv.PolyData(centers)
            myLabels = [f"{element}" for element in elements]
            pvData["myLabels"] = myLabels  # type: ignore [type-var]
            plotter.add_point_labels(
                pvData, "myLabels", point_color="k", render_points_as_spheres=True
            )

    _Annotate(plotter, inDim, title, showGrid, bounds)

    return plotter


@requires_pyvista
def Plot_Arrows(
    obj: _Simu | Mesh,
    nodes: _types.IntArray,
    vectors: _types.FloatArray,
    deformFactor: float = 0.0,
    magnitudeCoef: float = 0.1,
    alpha: float = 1.0,
    color: str = "red",
    linewidth: float | None = None,
    label: str | None = None,
    plotter: pv.Plotter | None = None,
):
    """Plots the mesh elements corresponding to the given nodes.

    Parameters
    ----------
    obj : _Simu | Mesh
        object containing the mesh
    nodes : _types.IntArray
        mesh nodes
    vectors : _types.FloatArray
        vectors on nodes
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    magnitudeCoef : float, optional
        coef used to scale the average distance between the coordinates and the center, by default 0.1
    alpha : float, optional
        transparency of faces, by default 1.0
    color : str, optional
        color used to display faces, by default 'red
    linewidth : float, optional
        Thickness of lines, by default None
    label : str, optional
        label, by default None
    plotter : pv.Plotter, optional
        The pyvista plotter, by default None and create a new Plotter instance

    Returns
    -------
    pv.Plotter
        The pyvista plotter
    """

    _, mesh, coord, _ = _Init_obj(obj, deformFactor)

    nodes = np.asarray(nodes, dtype=int)
    assert nodes.ndim == 1
    vectors = np.asarray(vectors, dtype=float)
    assert vectors.ndim == 2 and vectors.shape[0] == nodes.size

    if plotter is None:
        plotter = _Plotter()

    magnitude = mesh._Get_realistic_vector_magnitude(magnitudeCoef)
    plotter.add_arrows(
        coord[nodes],  # type: ignore [arg-type]
        vectors,  # type: ignore [arg-type]
        magnitude,
        opacity=alpha,
        color=color,
        line_width=linewidth,
        label=label,
    )

    return plotter


@requires_pyvista
def Plot_BoundaryConditions(
    simu: _Simu,
    *,
    deformFactor: float = 0.0,
    alpha: float = 0.1,
    nodeSize: float | None = None,
    title: str = "Boundary conditions",
    showGrid: bool = False,
    plotLegend: bool = True,
    bounds: _types.Numbers | None = None,
    plotter: pv.Plotter | None = None,
):
    """Plots simulation's boundary conditions.

    Parameters
    ----------
    simu : Simu
        simulation
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    alpha : float, optional
        opacity of the mesh drawn underneath, default 0.1
    nodeSize : float, optional
        point size, default None
    title : str, optional
        title, by default "Boundary conditions"
    showGrid : bool, optional
        Show the grid, by default False
    plotLegend : bool, optional
        displays the legend, by default True
    bounds : sequence[float], optional
        fixed view box (xmin, xmax, ymin, ymax, zmin, zmax), by default None (fits everything drawn)
    plotter : pv.Plotter, optional
        The pyvista plotter, by default None and create a new Plotter instance, default None

    Returns
    -------
    pv.Plotter
        The pyvista plotter
    """

    tic = Tic()

    simu, mesh, coord, inDim = _Init_obj(simu, deformFactor)  # type: ignore [assignment]

    if simu is None:
        Terminal.MyPrintError("simu must be a _Simu object")
        return

    # get dirichlet and neumann boundary conditions
    dirchlets = simu.Bc_Dirichlet
    boundaryConditions = dirchlets
    neumanns = simu.Bc_Neuman
    boundaryConditions.extend(neumanns)
    displays = (
        simu.Bc_Display
    )  # boundary conditions for display used for lagrangian boundary conditions
    boundaryConditions.extend(displays)

    if plotter is None:
        plotter = Plot_Elements(simu, None, 1, deformFactor=deformFactor, color="k")
        Plot(simu, None, deformFactor, alpha=alpha, color="gray", plotter=plotter)

    colors = tab10_colors * np.ceil(len(boundaryConditions) / 10).astype(int)

    for bc, color in zip(boundaryConditions, colors):

        problemType = bc.problemType
        dofsValues = bc.dofsValues
        unknowns = bc.unknowns
        dofs = bc.dofs
        nodes = bc.nodes
        description = bc.description
        nDir = len(unknowns)

        availableUnknowns = simu.Get_unknowns(problemType)
        nDof = mesh.Nn * simu.Get_dof_n(problemType)

        # label
        unknowns_str = str(unknowns).replace("'", "")
        label = f"{description} {unknowns_str}"

        nodes = np.asarray(list(set(nodes)), dtype=int)

        unknowns_rot = ["rx", "ry", "rz"]

        if nDof == mesh.Nn:
            # plot points
            plotter.add_mesh(
                pv.PolyData(coord[nodes]),  # type: ignore [arg-type]
                render_points_as_spheres=False,
                point_size=nodeSize,
                label=label,
                color=color,
            )

        else:
            # will try to display as an arrow
            # if dofsValues are null, will display as points

            summedValues = csr_matrix(
                (dofsValues, (dofs, np.zeros_like(dofs))), (nDof, 1)
            )
            dofsValues = summedValues.toarray()

            # here I want to build two display vectors (translation and rotation)
            start = coord[nodes]
            vector = np.zeros_like(start)
            vectorRot = np.zeros_like(start)

            for d, direction in enumerate(unknowns):
                lines = simu.Bc_dofs_nodes(nodes, [direction], problemType)
                values = np.ravel(dofsValues[lines])
                if direction in unknowns_rot:
                    idx = unknowns_rot.index(direction)
                    vectorRot[:, idx] = values
                else:
                    idx = availableUnknowns.index(direction)
                    vector[:, idx] = values

            normVector = np.linalg.norm(vector, axis=1).max()
            if normVector > 0:
                vector = vector / normVector

            normVectorRot = np.linalg.norm(vectorRot, axis=1).max()
            if normVectorRot > 0:
                vectorRot = vectorRot / normVectorRot

            factor = mesh._Get_realistic_vector_magnitude(0.1)

            if dofs.size / nDir > simu.mesh.Nn:
                # values are applied on every nodes of the mesh
                # the plot only one arrow
                factor = mesh._Get_realistic_vector_magnitude(0.5)
                start = mesh.center
                vector = np.mean(vector, 0)
                vectorRot = np.mean(vectorRot, 0)

            # plot vector
            if normVector == 0:
                # vector is a matrix of zeros
                pvData = pv.PolyData(coord[nodes])  # type: ignore [arg-type]
                plotter.add_mesh(
                    pvData,
                    render_points_as_spheres=True,
                    point_size=nodeSize,
                    label=label,
                    color=color,
                )
            else:
                # here the arrow will end at the node coordinates
                plotter.add_arrows(
                    start - vector * factor, vector, factor, label=label, color=color  # type: ignore [arg-type]
                )

            if True in [direction in unknowns_rot for direction in unknowns]:
                # plot vectorRot
                if normVectorRot == 0:
                    # vectorRot is a matrix of zeros
                    pvData = pv.PolyData(coord[nodes])  # type: ignore [arg-type]
                    plotter.add_mesh(
                        pvData,
                        render_points_as_spheres=True,
                        point_size=nodeSize,
                        label=label,
                        color=color,
                    )
                else:
                    # here the arrow will end at the node coordinates
                    plotter.add_arrows(
                        start,  # type: ignore [arg-type]
                        vectorRot,  # type: ignore [arg-type]
                        factor / 2,
                        label=label,
                        color=color,
                    )

    if plotLegend and len(boundaryConditions) > 0:
        plotter.add_legend(bcolor="white", face="o")  # type: ignore [call-arg]

    _Annotate(plotter, inDim, title, showGrid, bounds)

    tic.Tac("PyVista_Interface", "Plot_BoundaryConditions")

    return plotter


@requires_pyvista
def Plot_Tags(
    obj,
    *,
    deformFactor: float = 0.0,
    alpha: float = 1.0,
    linewidth: float | None = None,
    title: str = "",
    showId: bool = True,
    showGrid: bool = False,
    plotLegend: bool = True,
    useColorCycler: bool = False,
    bounds: _types.Numbers | None = None,
    plotter: pv.Plotter | None = None,
) -> pv.Plotter:
    """Plots the mesh's elements tags (from 2d elements to points) but do not plot the 3d elements tags.

    Parameters
    ----------
    obj : _Simu | Mesh | _GroupElem
        object containing the mesh
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    alpha : float | str | ndarray, optional
        Opacity of the mesh, by default 1.0
    linewidth : float, optional
        width of the tagged lines, by default None (2)
    title : str, optional
        title, by default ""
    showId : bool, optional
        writes the tags, by default True
    showGrid : bool, optional
        Show the grid, by default False
    plotLegend : bool, optional
        displays the legend, by default True
    useColorCycler : bool, optional
        whether to use color cycler, by default False
    bounds : sequence[float], optional
        fixed view box (xmin, xmax, ymin, ymax, zmin, zmax), by default None (fits everything drawn)
    plotter : pv.Plotter, optional
        The pyvista plotter, by default None and create a new Plotter instance.

    Returns
    -------
    pv.Plotter
        The pyvista plotter
    """

    tic = Tic()

    __, mesh, coord, inDim = _Init_obj(obj, deformFactor)

    # check if there is available tags in the mesh
    nTtags = [
        np.max([len(groupElem.nodeTags), len(groupElem.elementTags)])
        for groupElem in mesh.dict_groupElem.values()
    ]
    if np.max(nTtags) == 0:
        Terminal.MyPrintError(
            "There is no tags available in the mesh, so don't forget to use the '_Set_PhysicalGroups()' function before meshing your geometry with in the gmsh interface."
        )
        return None  # type: ignore [return-value]

    if plotter is None:
        plotter = _Plotter()

    Plot(obj, None, deformFactor, alpha=0.1, plotter=plotter)

    if useColorCycler:
        colorIterator = iter(tab10_colors * np.ceil(np.sum(nTtags) / 10).astype(int))

    for groupElem in mesh.dict_groupElem.values():

        # groupElem's data
        tags_e = groupElem.elementTags
        dim = groupElem.dim
        center_e = np.mean(coord[groupElem.connect], axis=1)  # center of each elements
        cellType, connect = _pvIO._Get_pyvista_cell(groupElem)

        for tag_e in tags_e:
            # get nodes and elements
            nodes = groupElem.Get_Nodes_Tag(tag_e)
            elements = groupElem.Get_Elements_Tag(tag_e)
            if len(elements) == 0:
                continue

            grid = pv.UnstructuredGrid({cellType: connect[elements]}, coord)

            if useColorCycler:
                color = next(colorIterator)
            else:
                color = "k" if dim in [0, 1] else "c"

            kwargs = {
                "color": color,
                "label": tag_e,
            }

            if dim == 0:
                plotter.add_mesh(grid, render_points_as_spheres=True, **kwargs)
            elif dim == 1:
                lw = 2 if linewidth is None else linewidth
                plotter.add_mesh(grid, line_width=lw, **kwargs)
            else:
                plotter.add_mesh(grid, opacity=alpha, **kwargs)

            if showId:
                # add tags
                if dim == 0:
                    center = np.mean(coord[nodes], axis=0)
                else:
                    center = np.mean(center_e[elements], axis=0)
                plotter.add_point_labels(
                    center.reshape(1, 3), [tag_e], always_visible=True  # type: ignore [arg-type]
                )

    tic.Tac("PyVista", "Plot_Tags")

    if plotLegend:
        plotter.add_legend()  # type: ignore [call-arg]

    _Annotate(plotter, inDim, title, showGrid, bounds)

    return plotter


@requires_pyvista
def Plot_Geoms(
    *geoms: Geoms._Geom | list[Geoms._Geom],
    color: str | None = None,
    linewidth: float | None = 2,
    alpha: float = 1.0,
    title: str = "",
    label: str | None = None,
    showGrid: bool = False,
    plotLegend: bool = True,
    bounds: _types.Numbers | None = None,
    plotter: pv.Plotter | None = None,
    **kwargs,
) -> pv.Plotter:
    """Plots _Geom objects

    Parameters
    ----------
    *geoms : _Geom | list[_Geom]
        geom objects, or lists of them
    color : str, optional
        color of every geom, by default None (one color per geom)
    linewidth : float, optional
        Thickness of lines, by default 2
    alpha : float, optional
        opacity, by default 1.0
    title : str, optional
        title, by default ""
    label : str, optional
        legend label replacing each geom's name, by default None
    showGrid : bool, optional
        Show the grid, by default False
    plotLegend : bool,
        plot the legend, by default True
    bounds : sequence[float], optional
        fixed view box (xmin, xmax, ymin, ymax, zmin, zmax), by default None (fits everything drawn)
    plotter : pv.Plotter, optional
        The pyvista plotter, by default None and create a new Plotter instance
    **kwargs:
        Everything that can goes in Plot() and add_mesh function https://docs.pyvista.org/version/stable/api/plotting/_autosummary/pyvista.Plotter.add_mesh.html#pyvista.Plotter.add_mesh

    Returns
    -------
    pv.Plotter
        The pyvista plotter
    """

    geoms = _Flatten_geoms(geoms)  # type: ignore [assignment]

    if plotter is None:
        plotter = _Plotter()

    if color is None:
        colors = iter(tab10_colors * np.ceil(len(geoms) / 10).astype(int))
    else:
        colors = iter([color] * len(geoms))

    for geom, geomColor in zip(geoms, colors):

        dataSet = _pvGeom(geom)

        if dataSet is None:
            continue

        name = geom.name if label is None else label  # type: ignore [union-attr]
        dataSets = dataSet if isinstance(dataSet, list) else [dataSet]
        for d, data in enumerate(dataSets):
            Plot(
                data,
                color=geomColor,
                linewidth=linewidth,
                alpha=alpha,
                label=name if d == 0 else None,
                plotter=plotter,
                **kwargs,
            )

    if plotLegend:
        plotter.add_legend(bcolor="white", face="o")  # type: ignore [call-arg]

    _Annotate(plotter, 3, title, showGrid, bounds)

    return plotter


# ----------------------------------------------
# Movie
# ----------------------------------------------
@rank0_only
@requires_pyvista
def Movie_simu(
    simu: _Simu,
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
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    coef : float, optional
        Coef to apply to the solution, by default 1.0
    nodeValues : bool, optional
        Displays result to nodes otherwise displays it to elements, by default True
    fps : int, optional
        frames per second, by default 30
    **kwargs:
        `Plot` options, e.g. `plotMesh`, `clim` or `bounds`
    """

    simu = _Init_obj(simu)[0]  # type: ignore [assignment]

    if simu is None:
        Terminal.MyPrintError("Must give a simulation.")
        return

    Niter = simu.Niter
    N = np.min([Niter, N])
    iterations = np.linspace(0, Niter - 1, N, endpoint=True, dtype=int)

    # activates the first iteration
    simu.Set_Iter(0, resetAll=True)

    def DoAnim(plotter, i):
        simu.Set_Iter(iterations[i])
        Plot(simu, result, deformFactor, coef, nodeValues, plotter=plotter, **kwargs)

    Movie_func(DoAnim, iterations.size, folder, filename, fps=fps)


@rank0_only
@requires_pyvista
def Movie_func(
    func: Callable[[pv.Plotter, int], None],
    N: int,
    folder: str,
    filename: str = "video.gif",
    *,
    fps: int = 30,
) -> None:
    """Generates the movie for the specified function.\\n
    This function will peform a loop in range(N); the view of the first frame is kept for the others.

    Parameters
    ----------
    func : Callable[[pv.Plotter, int], None]
        The function that will use in first argument the plotter and in second argument the iter step such that.\\n
        def func(plotter, i) -> None
    N : int
        number of iteration
    folder : str
        folder where you want to save the video
    filename : str, optional
        filename of the video with the extension (gif, mp4), by default 'video.gif'
    fps : int, optional
        frames per second, by default 30
    """

    plotter = _Plotter(True)

    filename = Folder.Join(folder, filename, mkdir=True)

    if ".gif" in filename:
        plotter.open_gif(filename, fps=fps)
    else:
        plotter.open_movie(filename, framerate=fps)

    tic = Tic()
    print()

    for i in range(N):
        plotter.clear()

        func(plotter, i)

        # the view no longer moves: frames stay comparable
        setattr(plotter, __frozen_view_arg, True)

        plotter.write_frame()

        time = tic.Tac("PyVista_Interface", "Movie_func", False)

        iteration = i + 1
        rmTime = Tic.Get_Remaining_Time(iteration, N, time)

        iteration = str(iteration).zfill(len(str(N)))  # type: ignore [assignment]
        Terminal.MyPrint(f"Generate movie {iteration}/{N} {rmTime}    ", end="\r")

    print()
    plotter.close()


# ----------------------------------------------
# Functions
# ----------------------------------------------


__update_camera_arg = "_need_to_update_camera_position"
__frozen_view_arg = "_easyfea_frozen_view"


@requires_pyvista
def _Plotter(off_screen=False, add_axes=True, shape=(1, 1), linkViews=True):
    plotter = pv.Plotter(off_screen=pv.OFF_SCREEN, shape=shape)
    setattr(plotter, __update_camera_arg, True)
    if add_axes:
        plotter.add_axes()
    if linkViews:
        plotter.link_views()
    plotter.subplot(0, 0)
    return plotter


@requires_pyvista
def _Annotate(
    plotter: pv.Plotter,
    inDim: int,
    title: str = "",
    showGrid: bool = False,
    bounds: _types.Numbers | None = None,
) -> None:
    """Adds the title and grid, then frames the view on everything drawn, or on the fixed `bounds`."""
    if title != "":
        plotter.add_title(title)
    if showGrid:
        plotter.show_grid()  # type: ignore [call-arg]
    if getattr(plotter, __frozen_view_arg, False):
        return
    if getattr(plotter, __update_camera_arg, False):
        # orientation, once per plotter
        _setCameraPosition(plotter, inDim)
        setattr(plotter, __update_camera_arg, False)
    if bounds is not None:
        if inDim == 2:
            plotter.enable_parallel_projection()  # type: ignore [call-arg]
        plotter.reset_camera(bounds=bounds)  # type: ignore [call-arg]
    else:
        plotter.reset_camera()  # type: ignore [call-arg]


@requires_pyvista
def _setCameraPosition(
    plotter: pv.Plotter,
    inDim: int,
    camera_position="xy",
    roll=0,
    elevation=25,
    azimuth=10,
    bounds=None,
):
    """Sets the camera position, then controls the camera and resets the clipping range if `inDim == 3`.\n
    https://docs.pyvista.org/api/core/camera.html#controlling-camera-rotation

    Parameters
    ----------
    plotter : pv.Plotter
        pyvista plotter
    inDim : int
        dimension in which the objects lies.
    camera_position : str, optional
        camera position of the active render window., by default "xy"
    roll : int, optional
        this will spin the camera about its axis., by default 0
    elevation : int, optional
        the vertical rotation of the scene, by default 25
    azimuth : int, optional
        the azimuth of the camera, by default 10
    bounds : tuple, optional
        fixed view box (xmin, xmax, ymin, ymax, zmin, zmax) to frame the camera on, analogous to matplotlib's xlim/ylim/zlim; keeps the view steady across frames instead of refitting to the (deforming) scene bounds. by default None (fit the scene). For a 2D view only the first four entries matter; pass (…, 0, 0) for the z range.
    """
    # see
    plotter.camera_position = camera_position
    if inDim == 3:
        plotter.camera.roll = roll
        plotter.camera.elevation = elevation
        plotter.camera.azimuth = azimuth
        plotter.camera.reset_clipping_range()

    if bounds is not None:
        # Frame the camera on a fixed bounding box (like matplotlib xlim/ylim/zlim).
        # reset_camera fits the box along the current view direction and sets a
        # sensible clipping range from it; aspect ratio is preserved (VTK cannot
        # stretch axes independently), so the non-limiting axis shows a bit extra.
        if inDim == 2:
            plotter.enable_parallel_projection()  # type: ignore [call-arg]
        plotter.reset_camera(bounds=bounds)  # type: ignore [call-arg]


@requires_pyvista
def _pvMesh(
    obj: _Simu | Mesh | _GroupElem,
    result: str | _types.AnyArray | dict | None = None,
    deformFactor=0.0,
    nodeValues=True,
    clipAxis=None,
    clipCenter=None,
) -> pv.UnstructuredGrid:
    """Creates the pyvista mesh from obj (_Simu, Mesh and _GroupElem objects)"""

    simu, mesh, coord, __ = _Init_obj(obj, deformFactor)
    result = _Gauss_points_averaged(mesh, result)

    unstructuredGrid = _pvIO.EasyFEA_to_PyVista(mesh, coord, useAllElements=False)

    values = _Get_values(simu, mesh, result, nodeValues)  # type: ignore [arg-type]

    # Add the result
    if isinstance(result, str) and result != "":
        unstructuredGrid[result] = values
        unstructuredGrid.set_active_scalars(result)

    elif isinstance(result, np.ndarray):
        name = "array"  # here result is an array
        unstructuredGrid[name] = values
        unstructuredGrid.set_active_scalars(name)

    if clipAxis is not None:
        clipCenter = mesh.center if clipCenter is None else clipCenter
        unstructuredGrid = unstructuredGrid.clip(clipAxis, clipCenter)

    return unstructuredGrid


@singledispatch
def _pvGeom(geom) -> pv.DataSet | list[pv.DataSet]:
    Terminal.MyPrintError(
        "geom must be in [Point, Line, Domain, Circle, CircleArc, Contour, Points]"
    )
    return None  # type: ignore [return-value]


@_pvGeom.register
def _(line: Geoms.Line):
    return pv.Line(line.pt1.coord, line.pt2.coord)  # type: ignore [arg-type]


@_pvGeom.register
def _(circleArc: Geoms.CircleArc):
    return pv.CircularArc(
        pointa=circleArc.pt1.coord,  # type: ignore [arg-type]
        pointb=circleArc.pt2.coord,  # type: ignore [arg-type]
        center=circleArc.center.coord,  # type: ignore [arg-type]
        negative=circleArc.coef == -1,
    )


@_pvGeom.register
def _(geom: Geoms.Point):
    return pv.PolyData(geom.coord)


@_pvGeom.register
def _(geom: Geoms.Domain):
    xMin, xMax = geom.pt1.x, geom.pt2.x
    yMin, yMax = geom.pt1.y, geom.pt2.y
    zMin, zMax = geom.pt1.z, geom.pt2.z
    return pv.Box((xMin, xMax, yMin, yMax, zMin, zMax)).outline()


@_pvGeom.register
def _(geom: Geoms.Circle):
    arc1 = pv.CircularArc(
        pointa=geom.pt1.coord, pointb=geom.pt3.coord, center=geom.center.coord  # type: ignore [arg-type]
    )
    arc2 = pv.CircularArc(
        pointa=geom.pt1.coord,  # type: ignore [arg-type]
        pointb=geom.pt3.coord,  # type: ignore [arg-type]
        center=geom.center.coord,  # type: ignore [arg-type]
        negative=True,
    )
    return [arc1, arc2]


@_pvGeom.register
def _(geom: Geoms.Points):
    geoms = geom.Get_Contour().geoms[:-1]
    dataSets: list[pv.DataSet] = []
    for geom in geoms:
        newData = _pvGeom(geom)
        if isinstance(newData, Iterable):
            dataSets.extend(newData)
        else:
            dataSets.append(newData)
    return dataSets


@_pvGeom.register
def _(geom: Geoms.Contour):
    geoms = geom.geoms
    dataSets: list[pv.DataSet] = []
    for geom in geoms:
        newData = _pvGeom(geom)
        if isinstance(newData, Iterable):
            dataSets.extend(newData)
        else:
            dataSets.append(newData)
    return dataSets
