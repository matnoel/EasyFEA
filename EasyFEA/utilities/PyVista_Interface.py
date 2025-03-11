# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Module providing an interface with PyVista (https://docs.pyvista.org/version/stable/).\n
https://docs.pyvista.org/api/plotting/plotting.html"""

from typing import Union, Callable
from cycler import cycler
from scipy.sparse import csr_matrix
import pyvista as pv
import numpy as np

# utilities
from .Display import MyPrintError, _Init_obj, MyPrint
from . import Folder, Tic
from .. import Geoms
# fem
from ..fem import GroupElemFactory

def Plot(obj, result: Union[str,np.ndarray]=None, deformFactor=0.0, coef=1.0, nodeValues=True, 
                color=None, show_edges=False, edge_color='k', line_width=None,
                show_vertices=False, point_size=None, opacity=1.0,
                style='surface', cmap="jet", n_colors=256, clim=None,
                plotter: pv.Plotter=None, show_grid=False, colorbarTitle=None, verticalColobar=True, **kwargs):
    """Plots the object obj that can be either a simu, mesh, MultiBlock, PolyData.\n
    If you want to plot the solution use plotter.show().

    Parameters
    ----------
    obj : _Simu | Mesh | MultiBlock | PolyData | UnstructuredGrid
        The object to plot and will be transformed to a mesh
    result : Union[str,np.ndarray], optional
        Scalars used to “color” the mesh, by default None
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    coef : float, optional
        Coef to apply to the solution, by default 1.0
    nodeValues : bool, optional
        Displays result to nodes otherwise displays it to elements, by default True
    color : str, optional
        Use to make the entire mesh have a single solid color, by default None
    show_edges : bool, optional
        Shows the edges of a mesh. Does not apply to a wireframe representation, by default False
    edge_color : str, optional
        The solid color to give the edges when show_edges=True, by default 'k'
    line_width : float, optional
        Thickness of lines. Only valid for wireframe and surface representations, by default None
    show_vertices : bool, optional
        Shows the nodes, by default False
    point_size : float, optional
        Point size of any nodes in the dataset plotted when show_vertices=True, by default None
    opacity : float | str | ndarray, optional
        Opacity of the mesh, by default 1.0
    style : str, optional
        Visualization style of the mesh. One of the following: ['surface', 'wireframe', 'points', 'points_gaussian'], by default 'surface'
    cmap : str, optional
        If a string, this is the name of the matplotlib colormap to use when mapping the scalars, by default "jet"\n
        ["jet", "seismic", "binary"] -> https://matplotlib.org/stable/tutorials/colors/colormaps.html
    n_colors : int, optional
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
        pvMesh = _pvGrid(obj, result, deformFactor, nodeValues)
        inDim = _Init_obj(obj)[-1]

    if pvMesh is None:
        # something do not work during the grid creation≠
        return
    
    # apply coef to the array
    name = "array" if isinstance(result, np.ndarray) else result
    name = None if pvMesh.n_arrays == 0 else name
    if name != None:
        pvMesh[name] *= coef
    
    colorbarTitle = name if colorbarTitle is None else colorbarTitle

    if plotter is None:        
        plotter = _Plotter()
    
    if show_grid:
        plotter.show_grid()    

    if verticalColobar:
        pos = 'position_x'
        val = 0.85
    else:
        pos = 'position_y'
        val = 0.025

    # plot the mesh
    if not isinstance(pvMesh, list):
        pvMeshs = [pvMesh]

    for pvMesh in pvMeshs:
        plotter.add_mesh(pvMesh, scalars=name,
                        color=color,
                        show_edges=show_edges, edge_color=edge_color, line_width=line_width,
                        show_vertices=show_vertices, point_size=point_size,
                        opacity=opacity,
                        style=style, cmap=cmap, n_colors=n_colors, clim=clim,
                        scalar_bar_args={'title': colorbarTitle, 'vertical': verticalColobar, pos: val},
                        **kwargs)

    _setCameraPosition(plotter, inDim)

    tic.Tac("PyVista_Interface","Plot")

    return plotter

def Plot_Mesh(obj, deformFactor=0.0, opacity=1.0, color='cyan', edge_color='black', line_width=0.5,
              plotter: pv.Plotter=None):
    """Plots the mesh.

    Parameters
    ----------
    obj : _Simu | Mesh | MultiBlock | PolyData | UnstructuredGrid
        object containing the mesh
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    opacity : float, optional
        face opacity, default 1.0    
    color: str, optional
        face colors, default 'cyan'
    edge_color: str, optional
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

    plotter = Plot(obj, deformFactor=deformFactor, opacity=opacity, color=color, edge_color=edge_color, line_width=line_width, plotter=plotter, show_edges=True)

    return plotter

def Plot_Nodes(obj, nodes: np.ndarray=None, showId=False, deformFactor=0, color='red',
               folder="", label=None, plotter: pv.Plotter=None):
    """Plots mesh's nodes.

    Parameters
    ----------
    obj : _Simu | Mesh
        object containing the mesh
    nodes : np.ndarray, optional
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

    _, mesh, coordo, _ = _Init_obj(obj, deformFactor)   

    if nodes is None:
        nodes = mesh.nodes
        coordo = coordo[nodes]
    else:
        nodes = np.asarray(nodes)

        if nodes.ndim == 1:
            if nodes.size == 0:
                MyPrintError("The list of nodes is empty.")
                return
            if nodes.size > mesh.Nn:
                MyPrintError("The list of nodes must be of size <= mesh.Nn")
                return
            else:
                coordo = coordo[nodes]
        elif nodes.ndim == 2 and nodes.shape[1] == 3:
            coordo = nodes
        else:
            MyPrintError("Nodes must be either a list of nodes or a matrix of 3D vectors of dimension (n, 3).")
            return

    if plotter == None:
        plotter = Plot(obj, deformFactor=deformFactor, style='wireframe', color='k')

    pvData = pv.PolyData(coordo)

    if showId:
        myLabels = [f"{node}" for node in nodes]
        pvData["myLabels"] = myLabels
        plotter.add_point_labels(pvData, "myLabels", point_color=color, render_points_as_spheres=True)
    else:
        plotter.add_mesh(pvData, color=color, label=label, render_points_as_spheres=True)

    return plotter

def Plot_Elements(obj, nodes: np.ndarray=None, dimElem: int=None, showId=False,
                  deformFactor=0, opacity=1.0, color='red', edge_color='black',
                  line_width=None, label=None, plotter: pv.Plotter=None):
    """Plots the mesh elements corresponding to the given nodes.

    Parameters
    ----------
    obj : _Simu | Mesh
        object containing the mesh
    nodes : np.ndarray, optional
        nodes used by elements, default None    
    dimElem : int, optional
        dimension of elements, by default None (mesh.dim)
    showId : bool, optional
        display numbers, by default False  
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0  
    opacity : float, optional
        transparency of faces, by default 1.0
    color : str, optional
        color used to display faces, by default 'red
    edge_color : str, optional
        color used to display segments, by default 'black'    
    line_width : float, optional
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

    _, mesh, coordo, _ = _Init_obj(obj, deformFactor)
    
    dimElem = mesh.dim if dimElem == None else dimElem

    if nodes is None:
        nodes = mesh.nodes
    else:
        nodes = np.asarray(nodes)
        if nodes.ndim != 1 or nodes.size > mesh.Nn:
            MyPrintError("Nodes must be a list of nodes of size <= mesh.Nn.")
            return

    if plotter == None:
        # plotter = Plot(obj, deformFactor=deformFactor, style='wireframe', color=edge_color, line_width=line_width)
        plotter = _Plotter()
    
    for groupElem in mesh.Get_list_groupElem(dimElem):

        # get the elements associated with the nodes
        elements = groupElem.Get_Elements_Nodes(nodes)

        if elements.size == 0: continue

        # construct the new group element by changing the connectivity matrix
        gmshId = groupElem.gmshId
        connect = groupElem.connect[elements]
        nodes = groupElem.nodes
        newGroupElem = GroupElemFactory.Create(gmshId, connect, coordo, nodes)

        pvGroup = _pvGrid(newGroupElem)

        Plot(pvGroup, opacity=opacity, color=color, edge_color=edge_color, plotter=plotter, line_width=line_width, label=label)

        if showId:
            centers = np.mean(coordo[groupElem.connect[elements]], axis=1)
            pvData = pv.PolyData(centers)
            myLabels = [f"{element}" for element in elements]
            pvData["myLabels"] = myLabels
            plotter.add_point_labels(pvData, "myLabels", point_color='k', render_points_as_spheres=True)

    return plotter

def Plot_BoundaryConditions(simu, deformFactor=0.0, plotter: pv.Plotter=None):
    """Plots simulation's boundary conditions.

    Parameters
    ----------
    simu : Simu
        simulation
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0  
    plotter : pv.Plotter, optional
        The pyvista plotter, by default None and create a new Plotter instance

    Returns
    -------
    pv.Plotter
        The pyvista plotter
    """

    tic = Tic()

    
    simu, mesh, coordo, inDim = _Init_obj(simu, deformFactor)

    if simu is None:
        MyPrintError('simu must be a _Simu object')
        return

    # get dirichlet and neumann boundary conditions
    dirchlets = simu.Bc_Dirichlet
    BoundaryConditions = dirchlets
    neumanns = simu.Bc_Neuman
    BoundaryConditions.extend(neumanns)
    displays = simu.Bc_Display # boundary conditions for display used for lagrangian boundary conditions
    BoundaryConditions.extend(displays)

    if plotter == None:
        plotter = _Plotter()    
        Plot_Elements(simu, None, 1, False, deformFactor, plotter=plotter, color='k')
        # Plot(simu, deformFactor=deformFactor, plotter=plotter, color='k', style='wireframe')
        plotter.add_title('Boundary conditions')

    pv.global_theme.color_cycler = 'default' # same as matplotlib
    color_cycler = pv.global_theme.color_cycler

    for (bc, cycle) in zip(BoundaryConditions, color_cycler):

        color = cycle['color']
        
        problemType = bc.problemType
        dofsValues = bc.dofsValues
        directions = bc.directions
        dofs = bc.dofs
        nodes = bc.nodes
        description = bc.description
        nDir = len(directions)

        availableDirections = simu.Get_dofs(problemType)
        nDof = mesh.Nn * simu.Get_dof_n(problemType)

        # label        
        directions_str = str(directions).replace("'","")
        label = f"{description} {directions_str}"

        nodes = np.asarray(list(set(nodes)), dtype=int)

        rotDirections = ["rx","ry","rz"]

        if nDof == mesh.Nn:
            # plot points 
            plotter.add_mesh(pv.PolyData(coordo[nodes]), render_points_as_spheres=False, label=label, color=color)

        else:
            # will try to display as an arrow
            # if dofsValues are null, will display as points

            summedValues = csr_matrix((dofsValues, (dofs, np.zeros_like(dofs))), (nDof, 1))            
            dofsValues = summedValues.toarray()            

            # here I want to build two display vectors (translation and rotation)
            start = coordo[nodes]
            vector = np.zeros_like(start)
            vectorRot = np.zeros_like(start)

            for d, direction in enumerate(directions):
                lines = simu.Bc_dofs_nodes(nodes, [direction], problemType)
                values = np.ravel(dofsValues[lines])
                if direction in rotDirections:
                    idx = rotDirections.index(direction)
                    vectorRot[:,idx] = values
                else:
                    idx = availableDirections.index(direction)
                    vector[:,idx] = values

            normVector = np.linalg.norm(vector, axis=1).max()
            if normVector > 0:
                vector = vector/normVector

            normVectorRot = np.linalg.norm(vectorRot, axis=1).max()
            if np.max(vectorRot) > 0:
                vectorRot = vectorRot/normVectorRot
            
            # here calculate the average distance between the coordinates and the center
            center = np.mean(coordo, 0)
            dist = np.linalg.norm(coordo-center, axis=1).max()
            # use thise distance to apply a magnitude to the vectors
            factor = 1 if dist == 0 else dist*.1

            if dofs.size/nDir > simu.mesh.Nn:
                # values are applied on every nodes of the mesh
                # the plot only one arrow
                factor = dist*.5
                start = mesh.center
                vector = np.mean(vector, 0)
                vectorRot = np.mean(vectorRot, 0)
            
            # plot vector
            if normVector == 0:
                # vector is a matrix of zeros
                pvData = pv.PolyData(coordo[nodes])
                plotter.add_mesh(pvData, render_points_as_spheres=True, label=label, color=color)
            else:
                # here the arrow will end at the node coordinates
                plotter.add_arrows(start-vector*factor, vector, factor, label=label, color=color)

            if True in [direction in rotDirections for direction in directions]:
                # plot vectorRot
                if normVectorRot == 0:
                    # vectorRot is a matrix of zeros
                    pvData = pv.PolyData(coordo[nodes])
                    plotter.add_mesh(pvData, render_points_as_spheres=True, label=label, color=color)
                else:
                    # here the arrow will end at the node coordinates
                    plotter.add_arrows(start, vector, factor/2, label=label, color=color)
    
    plotter.add_legend(bcolor='white',face="o")

    _setCameraPosition(plotter, inDim)

    pv.global_theme.color_cycler = None # same as matplotlib

    tic.Tac("PyVista_Interface","Plot_BoundaryConditions")    

    return plotter

def Plot_Geoms(geoms: list, line_width=2, plotLegend=True, plotter: pv.Plotter=None, **kwargs) -> pv.Plotter:
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

    geoms: list[Geoms._Geom] = geoms

    if not "color" in kwargs.keys():
        pv.global_theme.color_cycler = 'default' # same as matplotlib
        color_cycler = pv.global_theme.color_cycler
    else:        
        color_cycler = cycler(color=[kwargs['color']])
        kwargs.pop('color')

    for geom, cycle in zip(geoms, color_cycler):

        color = cycle["color"]

        dataSet = _pvGeom(geom)

        if dataSet is None:
            continue

        if isinstance(dataSet, list):
            for d, data in enumerate(dataSet):
                label = geom.name if d == 0 else None
                Plot(data, plotter=plotter, label=label, color=color, line_width=line_width, **kwargs)
        else:
            Plot(dataSet, plotter=plotter, label=geom.name, color=color, line_width=line_width, **kwargs)

    pv.global_theme.color_cycler = None

    if plotLegend:
        plotter.add_legend(bcolor='white',face="o")

    return plotter

# ----------------------------------------------
# Movie
# ----------------------------------------------
def Movie_simu(simu, result: str, folder: str, filename='video.gif', N:int=200,
          deformFactor=0.0, coef=1.0, nodeValues=True, **kwargs) -> None:
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
    
    simu = _Init_obj(simu)[0]

    if simu is None:
        MyPrintError("Must give a simulation.")
        return

    Niter = len(simu.results)
    step = np.max([1, Niter//N])
    iterations: np.ndarray = np.arange(0, Niter, step)

    # activates the first iteration
    simu.Set_Iter(0, resetAll=True)

    def DoAnim(plotter, i):        
        simu.Set_Iter(iterations[i])
        Plot(simu, result, deformFactor, coef, nodeValues, plotter=plotter, **kwargs)

    Movie_func(DoAnim, iterations.size, folder, filename)

def Movie_func(func: Callable[[pv.Plotter, int], None], N: int, folder: str, filename='video.gif'):
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

    if '.gif' in filename:
        plotter.open_gif(filename)
    else:
        plotter.open_movie(filename)

    tic = Tic()
    print()

    for i in range(N):

        plotter.clear()

        func(plotter, i)

        plotter.write_frame()

        time = tic.Tac("PyVista_Interface",f"Movie_func", False)        

        rmTime = Tic.Get_Remaining_Time(i, N-1, time)

        MyPrint(f"Movie_func {i}/{N-1} {rmTime}    ", end='\r')

    print()
    plotter.close()    

# ----------------------------------------------
# Types
# ----------------------------------------------

DICT_CELL_TYPES: dict[str, tuple[pv.CellType, int]] = {
    # (to Pyvista, to Paraview)
    # see https://dev.pyvista.org/api/utilities/_autosummary/pyvista.celltype#pyvista.CellType
    "SEG2": (pv.CellType.LINE, 3),
    "SEG3": (pv.CellType.QUADRATIC_EDGE, 21),
    "SEG4": (pv.CellType.CUBIC_LINE, 35),
    "SEG5": (pv.CellType.HIGHER_ORDER_EDGE, 60),
    "TRI3": (pv.CellType.TRIANGLE, 5),
    "TRI6": (pv.CellType.QUADRATIC_TRIANGLE, 22),
    "TRI10": (pv.CellType.LAGRANGE_TRIANGLE, 69),
    "TRI15": (pv.CellType.LAGRANGE_TRIANGLE, 69),
    "QUAD4": (pv.CellType.QUAD, 9),
    "QUAD8": (pv.CellType.QUADRATIC_QUAD, 23),
    "QUAD9": (pv.CellType.BIQUADRATIC_QUAD, 28),
    "TETRA4": (pv.CellType.TETRA, 10),
    "TETRA10": (pv.CellType.QUADRATIC_TETRA, 24),
    "HEXA8": (pv.CellType.HEXAHEDRON, 12),
    "HEXA20": (pv.CellType.QUADRATIC_HEXAHEDRON, 25),
    "HEXA27": (pv.CellType.TRIQUADRATIC_HEXAHEDRON, 29),
    "PRISM6": (pv.CellType.WEDGE, 13),
    "PRISM15": (pv.CellType.QUADRATIC_WEDGE, 26),
    "PRISM18": (pv.CellType.BIQUADRATIC_QUADRATIC_WEDGE, 32)
}

# reorganize the connectivity order 
# because some elements in gmsh don't have the same numbering order as in vtk
# pyvista -> https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.UnstructuredGrid.celltypes.html
# vtk -> https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
# https://dev.pyvista.org/api/utilities/_autosummary/pyvista.celltype
# you can search for vtk elements on the internet
DICT_VTK_INDEXES: dict[str, np.ndarray] = {
    # https://dev.pyvista.org/api/examples/_autosummary/pyvista.examples.cells.quadratichexahedron#pyvista.examples.cells.QuadraticHexahedron
    "HEXA20": [0,1,2,3,4,5,6,7,
              8,11,13,9,16,18,19,17,10,12,14,15],
    # https://dev.pyvista.org/api/examples/_autosummary/pyvista.examples.cells.triquadratichexahedron#pyvista.examples.cells.TriQuadraticHexahedron
    "HEXA27": [0,1,2,3,4,5,6,7,
               8,11,13,9,16,18,19,17,10,12,14,15,
               22,23,21,24,20,25,26],
    "PRISM15": [0,1,2,3,4,5,
                6,9,7,12,14,13,8,10,11],
    "PRISM18": [0,1,2,3,4,5,
                6,9,7,12,14,13,8,10,11,
                15,17,16],
    # nodes 8 and 9 are switch
    "TETRA10": [0,1,2,3,4,5,6,7,9,8]
    }

# ----------------------------------------------
# Functions
# ----------------------------------------------

def _Plotter(off_screen=False, add_axes=True, shape=(1,1), linkViews=True):
    plotter = pv.Plotter(off_screen=off_screen, shape=shape)
    if add_axes:
        plotter.add_axes()
    if linkViews:
        plotter.link_views()
    plotter.subplot(0,0)
    return plotter

def _setCameraPosition(plotter: pv.Plotter, inDim: int, elevation=25, azimuth=10):
    plotter.camera_position = 'xy'
    if inDim == 3:
        plotter.camera.elevation = elevation
        plotter.camera.azimuth = azimuth
        plotter.camera.reset_clipping_range()
    # if inDim == 3:        
    #     plotter.camera.elevation += 15
    #     plotter.camera.azimuth += 5
    #     plotter.camera.reset_clipping_range()

def _pvGrid(obj, result: Union[str, np.ndarray]=None, deformFactor=0.0, nodeValues=True) -> pv.UnstructuredGrid:
    """Creates the pyvista mesh from obj (_Simu, Mesh, _GroupElem and _Geoms object)"""

    simu, mesh, coordo, __ = _Init_obj(obj, deformFactor)

    elemType = mesh.elemType
    Nn = mesh.Nn
    Ne = mesh.Ne

    if elemType not in DICT_CELL_TYPES.keys():
        MyPrintError(f"{elemType} is not implemented yet.")
        return

    # reorder gmsh idx to vtk indexes
    if mesh.elemType in DICT_VTK_INDEXES.keys():
        vtkIndexes = DICT_VTK_INDEXES[mesh.elemType]
    else:
        vtkIndexes = np.arange(mesh.nPe)
    
    if mesh.elemType in ["TRI10", "TRI15"]:
        # forced to do this because pyvista simply does not have LAGRANGE_TRIANGLE
        # do not put in DICT_VTK_INDEXES because paraview can read LAGRANGE_TRIANGLE without changing the indices
        vtkIndexes = np.reshape(mesh.groupElem.triangles, (-1, 3))
    
    connect = mesh.connect[:, vtkIndexes]
    connect = np.reshape(connect, (-1, np.shape(vtkIndexes)[-1]))

    cellType = DICT_CELL_TYPES[elemType][0]
    pvMesh = pv.UnstructuredGrid({cellType: connect}, coordo)

    # Add the result    
    if isinstance(result, str) and result != "":
        if simu != None:
            values = simu.Result(result, nodeValues)
            size = values.size

            if values is None:
                pass
            elif size % Nn == 0 or size % Ne == 0:
                if size % Nn == 0:
                    values = np.reshape(values, (Nn, -1))
                elif size % Ne == 0:
                    values = np.reshape(values, (Ne, -1))
                pvMesh[result] = values
                pvMesh.set_active_scalars(result)
            else:
                MyPrintError("Must return nodes or elements values")
                
        else:
            MyPrintError("obj must be a simulation object or result should be nodes or elements values.")

    elif isinstance(result, np.ndarray):
        values = result
        size = result.size
        name = 'array' # here result is an array

        if size % Nn == 1 or size % Ne == 1:
            MyPrintError("Must be nodes or elements values")
        else:
            if size % Ne == 0 and nodeValues:
                # elements values that we want to plot on nodes
                values = mesh.Get_Node_Values(values)
                
            elif size % Nn == 0 and not nodeValues:
                # nodes values that we want to plot on elements
                values: np.ndarray = np.mean(values[connect], 1)
            
            pvMesh[name] = values
            pvMesh.set_active_scalars(name)

    return pvMesh

def _pvGeom(geom) -> Union[pv.DataSet, list[pv.DataSet]]:

    if not isinstance(geom, (Geoms.Point, Geoms._Geom)):
        MyPrintError("Must be a point or a geometric object.")
        return None
    
    def __Line(line: Geoms.Line):
        return pv.Line(line.pt1.coord, line.pt2.coord)        
    
    def __CircleArc(circleArc: Geoms.CircleArc):
        dataSet = pv.CircularArc(circleArc.pt1.coord, circleArc.pt2.coord,
                                 circleArc.center.coord, negative=circleArc.coef==-1)
        return dataSet
    
    def __DoGeoms(geoms: list[Geoms._Geom]):
        dataSets: list[pv.DataSet] = []
        for geom in geoms:
            if isinstance(geom, Geoms.Line):
                dataSets.append(__Line(geom))
            elif isinstance(geom, Geoms.CircleArc):
                dataSets.append(__CircleArc(geom))
            elif isinstance(geom, Geoms.Points):
                dataSets.extend(__DoGeoms(geom.Get_Contour().geoms[:-1]))

        return dataSets

    if isinstance(geom, Geoms.Point):
        dataSet = pv.PolyData(geom.coord)

    elif isinstance(geom, Geoms.Line):
        dataSet = __Line(geom)

    elif isinstance(geom, Geoms.Domain):
        xMin, xMax = geom.pt1.x, geom.pt2.x
        yMin, yMax = geom.pt1.y, geom.pt2.y
        zMin, zMax = geom.pt1.z, geom.pt2.z
        dataSet = pv.Box((xMin,xMax,yMin,yMax,zMin,zMax)).outline()

    elif isinstance(geom, Geoms.Circle):
        arc1 = pv.CircularArc(geom.pt1.coord, geom.pt3.coord, geom.center.coord)
        arc2 = pv.CircularArc(geom.pt1.coord, geom.pt3.coord, geom.center.coord, negative=True)
        dataSet = [arc1, arc2]

    elif isinstance(geom, Geoms.CircleArc):
        dataSet = __CircleArc(geom)

    elif isinstance(geom, (Geoms.Points,Geoms.Contour)):
        if isinstance(geom, Geoms.Points):
            geoms = geom.Get_Contour().geoms
            if geom.isOpen:
                geoms = geoms[:-1]
        else:
            geoms = geom.geoms
        dataSet = __DoGeoms(geoms)
    else:
        MyPrintError("obj must be in [Point, Line, Domain, Circle, CircleArc, Contour, Points]")

    return dataSet