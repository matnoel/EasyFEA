"""This module is an interface to Pyvista\n
https://docs.pyvista.org/version/stable/
"""

from typing import Union, Any, Callable
import pyvista as pv
import numpy as np

from Display import myPrintError, _init_obj
import Folder
from TicTac import Tic
from scipy.sparse import csr_matrix
# pv.global_theme.allow_empty_mesh = True

def Plot(obj, result: Union[str,np.ndarray]=None, deformFactor=0.0, coef=1.0, nodeValues=True, 
                color=None, show_edges=False, edge_color='k', line_width=None,
                show_vertices=False, point_size=None, opacity=1.0,
                style='surface', cmap="jet", n_colors=256, clim=None,
                plotter: pv.Plotter=None, show_grid=False, verticalColobar=True, **kwargs):
    """Plot the object obj that can be either a simu, mesh, MultiBlock, PolyData. If you want to plot the solution use plotter.show().

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
    
    # initiate the obj to construct the grid
    if isinstance(obj, (pv.MultiBlock, pv.PolyData, pv.UnstructuredGrid)):
        inDim = 3
        pvMesh = obj        
        result = result if result in pvMesh.array_names else None
    else:
        pvMesh = _pvMesh(obj, result, deformFactor, nodeValues)
        inDim = _init_obj(obj)[-1]

    if pvMesh is None:
        # something do not work during the grid creation≠
        return
    
    # apply coef to the array
    name = "array" if isinstance(result, np.ndarray) else result
    name = None if pvMesh.n_arrays == 0 else name
    if name != None:
        pvMesh[name] *= coef

    if plotter is None:        
        plotter = _initPlotter()
    
    if show_grid:
        plotter.show_grid()    

    if verticalColobar:
        pos = 'position_x'
        val = 0.85
    else:
        pos = 'position_y'
        val = 0.025

    # plot the mesh    
    plotter.add_mesh(pvMesh, scalars=name,
                     color=color,
                     show_edges=show_edges, edge_color=edge_color, line_width=line_width,
                     show_vertices=show_vertices, point_size=point_size,
                     opacity=opacity,
                     style=style, cmap=cmap, n_colors=n_colors, clim=clim,
                     scalar_bar_args={'title': name, 'vertical': verticalColobar, pos: val},
                     **kwargs)

    _setCameraPosition(plotter, inDim)

    tic.Tac("PyVista_Interface","Plot")

    return plotter

def Plot_Mesh(obj, deformFactor=0.0, opacity=1.0, color='cyan', edge_color='black', line_width=0.5,
              plotter: pv.Plotter=None, **kwargs):
    """Plot the mesh.

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
    **kwargs:
        Everything that can goes in Plot() and add_mesh() function https://docs.pyvista.org/version/stable/api/plotting/_autosummary/pyvista.Plotter.add_mesh.html#pyvista.Plotter.add_mesh

    Returns
    -------
    pv.Plotter
        The pyvista plotter
    """

    plotter = Plot(obj, deformFactor=deformFactor, opacity=opacity, color=color, edge_color=edge_color, line_width=line_width, plotter=plotter, show_edges=True, **kwargs)

    return plotter

def Plot_Nodes(obj, nodes: np.ndarray=None, showId=False, deformFactor=0, color='red',
               folder="", plotter: pv.Plotter=None, **kwargs):
    """Plot mesh nodes.

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
    plotter : pv.Plotter, optional
        The pyvista plotter, by default None and create a new Plotter instance
    **kwargs:
        Everything that can goes in Plot() and add_mesh() function https://docs.pyvista.org/version/stable/api/plotting/_autosummary/pyvista.Plotter.add_mesh.html#pyvista.Plotter.add_mesh

    Returns
    -------
    pv.Plotter
        The pyvista plotter
    """

    simu, mesh, coordo, inDim = _init_obj(obj, deformFactor)   

    if nodes is None:
        nodes = mesh.nodes
        coordo = coordo[nodes]
    else:
        nodes = np.asarray(nodes)

        if nodes.ndim == 1:
            if nodes.size > mesh.Nn:
                myPrintError("The list of nodes must be of size <= mesh.Nn")
                return
            else:
                coordo = coordo[nodes]
        elif nodes.ndim == 2 and nodes.shape[1] == 3:
            coordo = nodes
        else:
            myPrintError("Nodes must be either a list of nodes or a matrix of 3D vectors of dimension (n, 3).")
            return

    if plotter == None:
        plotter = Plot(obj, deformFactor=deformFactor, style='wireframe', color='k')

    pvData = pv.PolyData(coordo)

    if showId:
        myLabels = [f"{node}" for node in nodes]
        pvData["myLabels"] = myLabels
        plotter.add_point_labels(pvData, "myLabels", point_color=color, render_points_as_spheres=True)
    else:
        plotter.add_mesh(pvData, color=color, render_points_as_spheres=True)

    return plotter

def Plot_Elements(obj, nodes: np.ndarray=None, dimElem: int=None, showId=False, deformFactor=0, opacity=1.0, color='red', edge_color='black', line_width=None, plotter: pv.Plotter=None, **kwargs):
    """Display mesh elements from given nodes.

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
    plotter : pv.Plotter, optional
        The pyvista plotter, by default None and create a new Plotter instance
    **kwargs:
        Everything that can goes in Plot() and add_mesh() function https://docs.pyvista.org/version/stable/api/plotting/_autosummary/pyvista.Plotter.add_mesh.html#pyvista.Plotter.add_mesh

    Returns
    -------
    pv.Plotter
        The pyvista plotter
    """

    simu, mesh, coordo, inDim = _init_obj(obj, deformFactor)
    
    dimElem = mesh.dim if dimElem == None else dimElem

    if nodes is None:
        nodes = mesh.nodes
    else:
        nodes = np.asarray(nodes)
        if nodes.ndim != 1 or nodes.size > mesh.Nn:
            myPrintError("Nodes must be a list of nodes of size <= mesh.Nn.")
            return

    if plotter == None:
        plotter = Plot(obj, deformFactor=deformFactor, style='wireframe', color=edge_color, line_width=line_width)

    from GroupElems import _GroupElem_Factory
    
    for groupElem in mesh.Get_list_groupElem(dimElem):

        # get the elements associated with the nodes
        elements = groupElem.Get_Elements_Nodes(nodes)

        if elements.size == 0: continue

        # construct the new group element by changing the connectivity matrix
        gmshId = groupElem.gmshId
        connect = groupElem.connect[elements]
        nodes = groupElem.nodes
        newGroupElem = _GroupElem_Factory.Create(gmshId, connect, coordo, nodes)

        pvGroup = _pvMesh(newGroupElem)

        Plot(pvGroup, opacity=opacity, color=color, edge_color=edge_color, plotter=plotter, **kwargs, line_width=line_width)

        if showId:
            centers = np.mean(coordo[groupElem.connect[elements]], axis=1)
            pvData = pv.PolyData(centers)
            myLabels = [f"{element}" for element in elements]
            pvData["myLabels"] = myLabels
            plotter.add_point_labels(pvData, "myLabels", point_color='k', render_points_as_spheres=True)

    return plotter

def Plot_BoundaryConditions(simu, deformFactor=0.0, plotter: pv.Plotter=None, **kwargs):
    """Plot boundary conditions.

    Parameters
    ----------
    simu : Simu
        simulation
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0  
    plotter : pv.Plotter, optional
        The pyvista plotter, by default None and create a new Plotter instance
    **kwargs:
        Everything that can goes in Plot() and add_mesh() function https://docs.pyvista.org/version/stable/api/plotting/_autosummary/pyvista.Plotter.add_mesh.html#pyvista.Plotter.add_mesh

    Returns
    -------
    pv.Plotter
        The pyvista plotter
    """

    tic = Tic()

    
    simu, mesh, coordo, inDim = _init_obj(simu, deformFactor)

    if simu is None:
        myPrintError('simu must be a _Simu object')
        return

    # get dirichlet and neumann boundary conditions
    dirchlets = simu.Bc_Dirichlet
    BoundaryConditions = dirchlets
    neumanns = simu.Bc_Neuman
    BoundaryConditions.extend(neumanns)
    displays = simu.Bc_Display # boundary conditions for display used for lagrangian boundary conditions
    BoundaryConditions.extend(displays)

    if plotter == None:
        plotter = _initPlotter()    
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


        availableDirections = simu.Get_directions(problemType)
        nDof = mesh.Nn * simu.Get_dof_n(problemType)

        # label        
        directions_str = str(directions).replace("'","")
        label = f"{description} {directions_str}"

        nodes = np.asarray(list(set(nodes)), dtype=int)

        # ici continuer en construisant
        # un vecteur sparse !!!!!!!!

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


# --------------------------------------------------------------------------------------------
# Movie
# --------------------------------------------------------------------------------------------

def Movie_simu(simu, result: str, folder: str, videoName='video.gif',
          deformFactor=0.0, coef=1.0, nodeValues=True, **kwargs) -> None:
    """Generate a movie from a simu object and a result that you want to plot.

    Parameters
    ----------
    simu : _Simu
        simulation
    result : str
        result that you want to plot
    folder : str
        folder where you want to save the video
    videoName : str, optional
        filename of the video with the extension (gif, mp4), by default 'video.gif'
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    coef : float, optional
        Coef to apply to the solution, by default 1.0
    nodeValues : bool, optional
        Displays result to nodes otherwise displays it to elements, by default True
    """
    
    simu, mesh, coordo, inDim = _init_obj(simu)

    if simu is None:
        myPrintError("Must give a simulation.")
        return

    N = len(simu.results)

    def DoAnim(plotter, n):
        simu.Set_Iter(n)
        Plot(simu, result, deformFactor, coef, nodeValues, plotter=plotter, **kwargs)

    Movie_func(DoAnim, N, folder, videoName)

def Movie_func(func: Callable[[pv.Plotter, int], None], N: int, folder: str, videoName='video.gif'):
    """Make the movie for the specified function. This function will peform a loop in range(N) and plot in pyvista with func()

    Parameters
    ----------
    func : Callable[[pv.Plotter, int], None]
        The functiion that will use in first argument the plotter and in second argument the iter step. def func(plotter, n)
    N : int
        number of iteration
    folder : str
        folder where you want to save the video
    videoName : str, optional
        filename of the video with the extension (gif, mp4), by default 'video.gif'
    """

    tic = Tic()

    plotter = _initPlotter(True)
    
    videoName = Folder.Join(folder, videoName)

    if '.gif' in videoName:
        plotter.open_gif(videoName)
    else:
        plotter.open_movie(videoName)

    for n in range(N):

        plotter.clear()

        func(plotter, n)

        plotter.write_frame()

    plotter.close()

    tic.Tac("Pyvista_Interface","Movie")

# --------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------

__dictCellTypes: dict[str, pv.CellType] = {
    "SEG2": pv.CellType.LINE,
    "SEG3": pv.CellType.QUADRATIC_EDGE,
    "SEG4": pv.CellType.CUBIC_LINE,
    "TRI3": pv.CellType.TRIANGLE,
    "TRI6": pv.CellType.QUADRATIC_TRIANGLE,
    "TRI10": pv.CellType.TRIANGLE, # there is no TRI10 elements available
    "QUAD4": pv.CellType.QUAD,
    "QUAD8": pv.CellType.QUADRATIC_QUAD,
    "TETRA4": pv.CellType.TETRA,
    "TETRA10": pv.CellType.QUADRATIC_TETRA,
    "HEXA8": pv.CellType.HEXAHEDRON,
    "HEXA20": pv.CellType.QUADRATIC_HEXAHEDRON,
    "PRISM6": pv.CellType.WEDGE,
    "PRISM15": pv.CellType.QUADRATIC_WEDGE,
}

def _initPlotter(off_screen=False,add_axes=True):
    plotter = pv.Plotter(off_screen=off_screen)
    if add_axes:
        plotter.add_axes()
    return plotter

def _pvMesh(obj, result: Union[str, np.ndarray]=None, deformFactor=0.0, nodeValues=True) -> pv.UnstructuredGrid:
    """Construct the pyvista mesh from obj (_Simu or Mesh)"""

    simu, mesh, coordo, inDim = _init_obj(obj, deformFactor)

    groupElem = mesh.groupElem
    elemType = mesh.elemType
    Nn = mesh.Nn
    Ne = mesh.Ne

    if elemType not in __dictCellTypes.keys():
        myPrintError(f"{elemType} is not implemented yet.")
        return
    
    cellType = __dictCellTypes[elemType]
    
    # reorganize the connectivity order 
    # because some elements in gmsh don't have the same numbering order as in vtk
    # pyvista -> https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.UnstructuredGrid.celltypes.html
    # vtk -> https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
    # you can search for vtk elements on the internet
    if elemType == "TRI10":
        # there is cellType available for TRI10 so i juste use the TRI10 corners
        order = np.arange(3)
    elif elemType == "HEXA20":
        order = [0,1,2,3,4,5,6,7,8,11,13,9,16,18,19,17,10,12,14,15] # dont change here
    elif elemType == "PRISM15":
        order = [0,1,2,3,4,5,6,9,7,12,14,13,8,10,11]  # dont change here
    elif elemType == "TETRA10":
        order = [0,1,2,3,4,5,6,7,9,8]  # nodes 8 and 9 are switch
    else:
        order = np.arange(mesh.nPe)
    
    connect = mesh.connect[:, order]

    pvMesh = pv.UnstructuredGrid({cellType: connect}, coordo)

    # Add the result    
    if isinstance(result, str) and result != "":
        if simu != None:
            values = simu.Result(result, nodeValues)
            size = values.size

            if values is None:
                pass
            elif size % Nn == 1 or size % Ne == 1:
                myPrintError("Must return nodes or elements values")
            else:             
                if size % Nn == 0:
                    values = np.reshape(values, (Nn, -1))
                elif size % Ne == 0:
                    values = np.reshape(values, (Ne, -1))
                pvMesh[result] = values
                pvMesh.set_active_scalars(result)            
        else:
            myPrintError("obj must be a simulation object or result should be nodes or elements values.")

    elif isinstance(result, np.ndarray):
        values = result
        size = result.size
        name = 'array' # here result is an array

        tt = result.size % Nn

        if size % Nn == 1 or size % Ne == 1:
            myPrintError("Must be nodes or elements values")
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

def _setCameraPosition(plotter: pv.Plotter, inDim: int):
    plotter.camera_position = 'xy'
    if inDim == 3:        
        plotter.camera.elevation += 25
        plotter.camera.azimuth += 10
        plotter.camera.reset_clipping_range()