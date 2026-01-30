# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module providing an interface with PyVista (https://docs.pyvista.org/version/stable/).\n
https://docs.pyvista.org/api/plotting/plotting.html"""

from __future__ import annotations
from typing import Union, Callable, Optional, TYPE_CHECKING, Any, Iterable
from scipy.sparse import csr_matrix
import numpy as np
from functools import singledispatch

# utilities
from . import Display
from ..Simulations._simu import _Init_obj, _Get_values
from . import Folder, Tic, _types, MeshIO
from .. import Geoms

# fem
from ..FEM import GroupElemFactory

if TYPE_CHECKING:
    from ..Simulations._simu import _Simu
    from ..FEM._mesh import Mesh
    from ..FEM._group_elem import _GroupElem

from ._requires import Create_requires_decorator

try:
    import pyvista as pv
except ImportError:
    pass
requires_pyvista = Create_requires_decorator("matplotlib", "pyvista")


@requires_pyvista
def Plot(
    obj: Union[
        "_Simu",
        "Mesh",
        "_GroupElem",
        Any,
    ],
    result: Optional[Union[str, _types.FloatArray]] = None,
    deformFactor=0.0,
    coef=1.0,
    nodeValues=True,
    color=None,
    plotMesh=False,
    edgecolor="k",
    linewidth=None,
    plotNodes=False,
    point_size=None,
    alpha=1.0,
    style="surface",
    cmap="jet",
    nColors=256,
    clim=None,
    plotter: Optional[pv.Plotter] = None,
    show_grid=False,
    colorbarTitle=None,
    verticalColobar=True,
    **kwargs,
):
    """Plots the object obj that can be either a simu, mesh, MultiBlock, PolyData.\n
    If you want to plot the solution use plotter.show().

    Parameters
    ----------
    obj : _Simu | Mesh | _GroupElem | MultiBlock | PolyData | UnstructuredGrid
        The object to plot and will be transformed to a mesh
    result : Union[str,_types.FloatArray], optional
        Scalars used to “color” the mesh, by default None
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    coef : float, optional
        Coef to apply to the solution, by default 1.0
    nodeValues : bool, optional
        Displays result to nodes otherwise displays it to elements, by default True
    color : str, optional
        Use to make the entire mesh have a single solid color, by default None
    plotMesh : bool, optional
        Shows the edges of a mesh. Does not apply to a wireframe representation, by default False
    edgecolor : str, optional
        The solid color to give the edges when show_edges=True, by default 'k'
    linewidth : float, optional
        Thickness of lines. Only valid for wireframe and surface representations, by default None
    plotNodes : bool, optional
        Shows the nodes, by default False
    point_size : float, optional
        Point size of any nodes in the dataset plotted when plotNodes=True, by default None
    alpha : float | str | ndarray, optional
        Opacity of the mesh, by default 1.0
    style : str, optional
        Visualization style of the mesh. One of the following: ['surface', 'wireframe', 'points', 'points_gaussian'], by default 'surface'
    cmap : str, optional
        If a string, this is the name of the matplotlib colormap to use when mapping the scalars, by default "jet"\n
        ["jet", "seismic", "binary"] -> https://matplotlib.org/stable/tutorials/colors/colormaps.html
    nColors : int, optional
        Number of colors to use when displaying scalars, by default 256
    clim : sequence[float], optional
        Two item color bar range for scalars. Defaults to minimum and maximum of scalars array. Example: [-1, 2], by default None
    plotter : pv.Plotter, optional
        The pyvista plotter, by default None and create a new Plotter instance
    show_grid : bool, optionnal
        Show the grid, by default False
    colorbarTitle: str, optionnal
        colorbar title, by default None
    verticalColobar : bool, optionnal
        color bar is vertical, by default True
    **kwargs:
        Everything that can goes in add_mesh function https://docs.pyvista.org/version/stable/api/plotting/_autosummary/pyvista.Plotter.add_mesh.html#pyvista.Plotter.add_mesh

    Returns
    -------
    pv.Plotter
        The pyvista plotter
    """

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
    name = "array" if isinstance(result, np.ndarray) else result
    name = None if pvMesh.n_arrays == 0 else name
    if name is not None:
        pvMesh[name] *= coef

    colorbarTitle = name if colorbarTitle is None else colorbarTitle

    if plotter is None:
        plotter = _Plotter()

    if verticalColobar:
        pos = "position_x"
        val = 0.85
    else:
        pos = "position_y"
        val = 0.025

    # plot the mesh
    if not isinstance(pvMesh, list):
        pvMeshs = [pvMesh]

    for pvMesh in pvMeshs:
        plotter.add_mesh(
            pvMesh,
            scalars=name,
            color=color,
            show_edges=plotMesh,
            edge_color=edgecolor,
            line_width=linewidth,
            show_vertices=plotNodes,
            point_size=point_size,
            opacity=alpha,
            style=style,
            cmap=cmap,
            n_colors=nColors,
            clim=clim,
            scalar_bar_args={
                "title": colorbarTitle,
                "vertical": verticalColobar,
                pos: val,
            },
            **kwargs,
        )

    if hasattr(plotter, __update_camera_arg) and getattr(plotter, __update_camera_arg):
        _setCameraPosition(plotter, inDim)
        setattr(plotter, __update_camera_arg, False)

    if show_grid:
        # plotter.show_grid(fmt="%.3e")  # type: ignore [call-arg]
        plotter.show_grid()  # type: ignore [call-arg]

    tic.Tac("PyVista_Interface", "Plot")

    return plotter


@requires_pyvista
def Plot_Mesh(
    obj: Union[
        "_Simu",
        "Mesh",
        Any,
    ],
    deformFactor=0.0,
    alpha=1.0,
    color="cyan",
    edgecolor="black",
    linewidth=0.5,
    plotter: Optional[pv.Plotter] = None,
):
    """Plots the mesh.

    Parameters
    ----------
    obj : _Simu | Mesh | MultiBlock | PolyData | UnstructuredGrid
        object containing the mesh
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    alpha : float, optional
        face opacity, default 1.0
    color: str, optional
        face colors, default 'cyan'
    edgecolor: str, optional
        edge color, default 'black'
    line_width: float, optional
        line width, default 0.5
    plotter : pv.Plotter, optional
        The pyvista plotter, by default None and create a new Plotter instance

    Returns
    -------
    pv.Plotter
        The pyvista plotter
    """

    plotter = Plot(
        obj,
        deformFactor=deformFactor,
        alpha=alpha,
        color=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        plotter=plotter,
        plotMesh=True,
    )

    return plotter


@requires_pyvista
def Plot_Nodes(
    obj: Union["_Simu", "Mesh"],
    nodes: Optional[_types.IntArray] = None,
    showId=False,
    deformFactor=0,
    color="red",
    folder="",
    label=None,
    plotter: Optional[pv.Plotter] = None,
):
    """Plots mesh's nodes.

    Parameters
    ----------
    obj : _Simu | Mesh
        object containing the mesh
    nodes : _types.IntArray, optional
        nodes to display, default None
    showId : bool, optional
        display node numbers, default False
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    color : str, optional
        color, default 'red'
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

    if nodes is None:
        nodes = mesh.nodes
        coord = coord[nodes]
    else:
        nodes = np.asarray(nodes)

        if nodes.ndim == 1:
            if nodes.size == 0:
                Display.MyPrintError("The list of nodes is empty.")
                return
            if nodes.size > mesh.Nn:
                Display.MyPrintError("The list of nodes must be of size <= mesh.Nn")
                return
            else:
                coord = coord[nodes]
        elif nodes.ndim == 2 and nodes.shape[1] == 3:
            coord = nodes  # type: ignore [assignment]
        else:
            Display.MyPrintError(
                "Nodes must be either a list of nodes or a matrix of 3D vectors of dimension (n, 3)."
            )
            return

    if plotter is None:
        plotter = Plot(obj, deformFactor=deformFactor, style="wireframe", color="k")

    pvData = pv.PolyData(coord)  # type: ignore [arg-type]

    if showId:
        myLabels: list[str] = [f"{node}" for node in nodes]
        pvData["myLabels"] = myLabels  # type: ignore [assignment]
        plotter.add_point_labels(
            pvData, "myLabels", point_color=color, render_points_as_spheres=True
        )
    else:
        plotter.add_mesh(
            pvData, color=color, label=label, render_points_as_spheres=True
        )

    return plotter


@requires_pyvista
def Plot_Elements(
    obj: Union["_Simu", "Mesh"],
    nodes: Optional[_types.IntArray] = None,
    dimElem: Optional[int] = None,
    showId=False,
    deformFactor=0.0,
    alpha=1.0,
    color="red",
    edgecolor="black",
    linewidth: Optional[float] = None,
    label: Optional[str] = None,
    plotter: Optional[pv.Plotter] = None,
):
    """Plots the mesh elements corresponding to the given nodes.

    Parameters
    ----------
    obj : _Simu | Mesh
        object containing the mesh
    nodes : _types.IntArray, optional
        nodes used by elements, default None
    dimElem : int, optional
        dimension of elements, by default None (mesh.dim)
    showId : bool, optional
        display numbers, by default False
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    alpha : float, optional
        transparency of faces, by default 1.0
    color : str, optional
        color used to display faces, by default 'red
    edgecolor : str, optional
        color used to display segments, by default 'black'
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

    dimElem = mesh.dim if dimElem is None else dimElem

    if nodes is None:
        nodes = mesh.nodes
    else:
        nodes = np.asarray(nodes, dtype=int)
        if nodes.ndim != 1 or nodes.size > mesh.Nn:
            Display.MyPrintError("Nodes must be a list of nodes of size <= mesh.Nn.")
            return

    if plotter is None:
        # plotter = Plot(obj, deformFactor=deformFactor, style='wireframe', color=edge_color, line_width=line_width)
        plotter = _Plotter()

    for groupElem in mesh.Get_list_groupElem(dimElem):
        # get the elements associated with the nodes
        elements = groupElem.Get_Elements_Nodes(nodes)

        if elements.size == 0:
            continue

        # construct the new group element by changing the connectivity matrix
        connect = groupElem.connect[elements]
        newGroupElem = GroupElemFactory.Create(groupElem.elemType, connect, coord)

        pvGroup = _pvMesh(newGroupElem)  # type: ignore [arg-type]

        Plot(
            pvGroup,
            alpha=alpha,
            color=color,
            edgecolor=edgecolor,
            plotter=plotter,
            linewidth=linewidth,
            label=label,
        )

        if showId:
            centers = np.mean(coord[groupElem.connect[elements]], axis=1)
            pvData = pv.PolyData(centers)
            myLabels = [f"{element}" for element in elements]
            pvData["myLabels"] = myLabels  # type: ignore [assignment]
            plotter.add_point_labels(
                pvData, "myLabels", point_color="k", render_points_as_spheres=True
            )

    return plotter


@requires_pyvista
def Plot_Arrows(
    obj: Union["_Simu", "Mesh"],
    nodes: _types.IntArray,
    vectors: _types.FloatArray,
    deformFactor: float = 0.0,
    magnitudeCoef: float = 0.1,
    alpha: float = 1.0,
    color: str = "red",
    linewidth: Optional[float] = None,
    label: Optional[str] = None,
    plotter: Optional[pv.Plotter] = None,
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
        coord[nodes],
        vectors,
        magnitude,
        opacity=alpha,
        color=color,
        line_width=linewidth,
        label=label,
    )

    return plotter


@requires_pyvista
def Plot_BoundaryConditions(
    simu: "_Simu", deformFactor=0.0, plotter: Optional[pv.Plotter] = None
):
    """Plots simulation's boundary conditions.

    Parameters
    ----------
    simu : Simu
        simulation
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
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
        Display.MyPrintError("simu must be a _Simu object")
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
        plotter = _Plotter()
        Plot_Elements(simu, None, 1, False, deformFactor, plotter=plotter, color="k")
        # Plot(simu, deformFactor=deformFactor, plotter=plotter, color='k', style='wireframe')
        plotter.add_title("Boundary conditions")

    colors = Display.tab10_colors
    colors = colors * np.ceil(len(boundaryConditions) / len(colors)).astype(int)

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
            if np.max(vectorRot) > 0:
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
                    pvData, render_points_as_spheres=True, label=label, color=color
                )
            else:
                # here the arrow will end at the node coordinates
                plotter.add_arrows(
                    start - vector * factor, vector, factor, label=label, color=color
                )

            if True in [direction in unknowns_rot for direction in unknowns]:
                # plot vectorRot
                if normVectorRot == 0:
                    # vectorRot is a matrix of zeros
                    pvData = pv.PolyData(coord[nodes])  # type: ignore [arg-type]
                    plotter.add_mesh(
                        pvData, render_points_as_spheres=True, label=label, color=color
                    )
                else:
                    # here the arrow will end at the node coordinates
                    plotter.add_arrows(
                        start, vector, factor / 2, label=label, color=color
                    )

    plotter.add_legend(bcolor="white", face="o")  # type: ignore [call-arg]

    _setCameraPosition(plotter, inDim)
    plotter.zoom_camera(0.9)

    tic.Tac("PyVista_Interface", "Plot_BoundaryConditions")

    return plotter


@requires_pyvista
def Plot_Tags(
    obj, useColorCycler=False, plotter: Optional[pv.Plotter] = None
) -> pv.Plotter:
    """Plots the mesh's elements tags (from 2d elements to points) but do not plot the 3d elements tags.

    Parameters
    ----------
    obj : _Simu | Mesh | _GroupElem
        object containing the mesh
    useColorCycler : bool, optional
        whether to use color cycler, by default False
    plotter : pv.Plotter, optional
        The pyvista plotter, by default None and create a new Plotter instance, default None

    Returns
    -------
    pv.Plotter
        The pyvista plotter
    """

    tic = Tic()

    __, mesh, coord, inDim = _Init_obj(obj)

    # check if there is available tags in the mesh
    nTtags = [
        np.max([len(groupElem.nodeTags), len(groupElem.elementTags)])
        for groupElem in mesh.dict_groupElem.values()
    ]
    if np.max(nTtags) == 0:
        Display.MyPrintError(
            "There is no tags available in the mesh, so don't forget to use the '_Set_PhysicalGroups()' function before meshing your geometry with in the gmsh interface."
        )
        return None  # type: ignore [return-value]

    if plotter is None:
        plotter = _Plotter()

    Plot(mesh, alpha=0.1, plotter=plotter)

    if useColorCycler:
        colors = Display.tab10_colors
        colorIterator = iter(colors * np.ceil(np.sum(nTtags) / len(colors)).astype(int))

    for groupElem in mesh.dict_groupElem.values():

        # groupElem's data
        tags_e = groupElem.elementTags
        dim = groupElem.dim
        coord = groupElem.coordGlob
        center_e = np.mean(coord[groupElem.connect], axis=1)  # center of each elements

        for tag_e in tags_e:
            if "nodes" in tag_e:
                pass

            # get nodes and elements
            nodes = groupElem.Get_Nodes_Tag(tag_e)
            elements = groupElem.Get_Elements_Tag(tag_e)
            if len(elements) == 0:
                continue

            grid = MeshIO._GroupElem_to_PyVista(groupElem, elements)

            if useColorCycler:
                color = next(colorIterator)
            else:
                color = "k" if dim in [0, 1] else "c"

            if dim == 0:
                plotter.add_mesh(grid, color=color, render_points_as_spheres=True)
            elif dim == 1:
                plotter.add_mesh(grid, color=color, line_width=2)
            elif dim == 2:
                plotter.add_mesh(grid, color=color, opacity=0.5)
            else:
                plotter.add_mesh(grid, color=color, opacity=0.5)

            # add tags
            if dim == 0:
                center = np.mean(coord[nodes], axis=0)
            else:
                center = np.mean(center_e[elements], axis=0)
            plotter.add_point_labels(center.reshape(1, 3), [tag_e], always_visible=True)

    tic.Tac("PyVista", "Plot_Tags")

    return plotter


@requires_pyvista
def Plot_Geoms(
    geoms: list,
    line_width=2,
    plotLegend=True,
    plotter: Optional[pv.Plotter] = None,
    **kwargs,
) -> pv.Plotter:
    """Plots _Geom objects

    Parameters
    ----------
    geoms : list[_Geom]
        list of geom object
    plotLegend : bool,
        plot the legend, by default True
    line_width : float, optional
        Thickness of lines, by default 2
    plotter : pv.Plotter, optional
        The pyvista plotter, by default None and create a new Plotter instance
    **kwargs:
        Everything that can goes in Plot() and add_mesh function https://docs.pyvista.org/version/stable/api/plotting/_autosummary/pyvista.Plotter.add_mesh.html#pyvista.Plotter.add_mesh

    Returns
    -------
    pv.Plotter
        The pyvista plotter
    """

    if not isinstance(geoms, list):
        geoms = [geoms]

    if plotter is None:
        plotter = _Plotter()

    geoms: list[Geoms._Geom] = geoms  # type: ignore [no-redef]

    if "color" not in kwargs:
        colors = Display.tab10_colors
        colors = iter(colors * np.ceil(len(geoms) / len(colors)).astype(int))
    else:
        colors = [kwargs["color"]] * len(geoms)  # type: ignore [assignment]

    for geom, color in zip(geoms, colors):

        dataSet = _pvGeom(geom)

        if dataSet is None:
            continue

        if isinstance(dataSet, list):
            for d, data in enumerate(dataSet):
                label = geom.name if d == 0 else None
                Plot(
                    data,
                    plotter=plotter,
                    label=label,
                    color=color,
                    linewidth=line_width,
                    **kwargs,
                )
        else:
            Plot(
                dataSet,
                plotter=plotter,
                label=geom.name,
                color=color,
                linewidth=line_width,
                **kwargs,
            )

    _setCameraPosition(plotter, 3)

    if plotLegend:
        plotter.add_legend(bcolor="white", face="o")  # type: ignore [call-arg]

    return plotter


# ----------------------------------------------
# Movie
# ----------------------------------------------
@requires_pyvista
def Movie_simu(
    simu: "_Simu",
    result: str,
    folder: str,
    filename="video.gif",
    N: int = 200,
    deformFactor=0.0,
    coef=1.0,
    nodeValues=True,
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
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    coef : float, optional
        Coef to apply to the solution, by default 1.0
    nodeValues : bool, optional
        Displays result to nodes otherwise displays it to elements, by default True
    """

    simu = _Init_obj(simu)[0]  # type: ignore [assignment]

    if simu is None:
        Display.MyPrintError("Must give a simulation.")
        return

    Niter = len(simu.results)
    N = np.min([Niter, N])
    iterations = np.linspace(0, Niter - 1, N, endpoint=True, dtype=int)

    # activates the first iteration
    simu.Set_Iter(0, resetAll=True)

    def DoAnim(plotter, i):
        simu.Set_Iter(iterations[i])
        Plot(simu, result, deformFactor, coef, nodeValues, plotter=plotter, **kwargs)

    Movie_func(DoAnim, iterations.size, folder, filename)


@requires_pyvista
def Movie_func(
    func: Callable[[pv.Plotter, int], None], N: int, folder: str, filename="video.gif"
):
    """Generates the movie for the specified function.\n
    This function will peform a loop in range(N).

    Parameters
    ----------
    func : Callable[[pv.Plotter, int], None]
        The function that will use in first argument the plotter and in second argument the iter step such that.\n
        def func(plotter, i) -> None
    N : int
        number of iteration
    folder : str
        folder where you want to save the video
    filename : str, optional
        filename of the video with the extension (gif, mp4), by default 'video.gif'
    """

    plotter = _Plotter(True)

    filename = Folder.Join(folder, filename)

    if not Folder.Exists(folder):
        Folder.os.makedirs(folder)

    if ".gif" in filename:
        plotter.open_gif(filename)
    else:
        plotter.open_movie(filename)

    tic = Tic()
    print()

    for i in range(N):
        plotter.clear()

        func(plotter, i)

        plotter.write_frame()

        time = tic.Tac("PyVista_Interface", "Movie_func", False)

        iteration = i + 1
        rmTime = Tic.Get_Remaining_Time(iteration, N, time)

        iteration = str(iteration).zfill(len(str(N)))
        Display.MyPrint(f"Generate movie {iteration}/{N} {rmTime}    ", end="\r")

    print()
    plotter.close()


# ----------------------------------------------
# Functions
# ----------------------------------------------


__update_camera_arg = "_need_to_update_camera_position"


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
def _setCameraPosition(
    plotter: pv.Plotter,
    inDim: int,
    camera_position="xy",
    roll=0,
    elevation=25,
    azimuth=10,
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
    """
    # see
    plotter.camera_position = camera_position
    if inDim == 3:
        plotter.camera.roll = roll
        plotter.camera.elevation = elevation
        plotter.camera.azimuth = azimuth
        plotter.camera.reset_clipping_range()


@requires_pyvista
def _pvMesh(
    obj: Union["_Simu", "Mesh", "_GroupElem"],
    result: Optional[Union[str, _types.AnyArray]] = None,
    deformFactor=0.0,
    nodeValues=True,
    clipAxis=None,
    clipCenter=None,
) -> pv.UnstructuredGrid:
    """Creates the pyvista mesh from obj (_Simu, Mesh and _GroupElem objects)"""

    simu, mesh, coord, __ = _Init_obj(obj, deformFactor)

    unstructuredGrid = MeshIO.EasyFEA_to_PyVista(mesh, coord, useAllElements=False)

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


@requires_pyvista
@singledispatch
def _pvGeom(geom) -> Union[pv.DataSet, list[pv.DataSet]]:
    Display.MyPrintError(
        "geom must be in [Point, Line, Domain, Circle, CircleArc, Contour, Points]"
    )
    return None  # type: ignore [return-value]


@_pvGeom.register
def _(line: Geoms.Line):
    return pv.Line(line.pt1.coord, line.pt2.coord)


@_pvGeom.register
def _(circleArc: Geoms.CircleArc):
    return pv.CircularArc(
        pointa=circleArc.pt1.coord,
        pointb=circleArc.pt2.coord,
        center=circleArc.center.coord,
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
        pointa=geom.pt1.coord, pointb=geom.pt3.coord, center=geom.center.coord
    )
    arc2 = pv.CircularArc(
        pointa=geom.pt1.coord,
        pointb=geom.pt3.coord,
        center=geom.center.coord,
        negative=True,
    )
    return [arc1, arc2]


@_pvGeom.register
def _(geom: Geoms.Points):
    geoms = geom.Get_Contour().geoms
    if geom.isOpen:
        geoms = geoms[:-1]
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
