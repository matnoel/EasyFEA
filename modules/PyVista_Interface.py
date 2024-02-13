"""This module is an interface to Pyvista\n
https://docs.pyvista.org/version/stable/
"""

from typing import Union, Any
import pyvista as pv
import numpy as np


from Display import myPrintError, _init_obj

# pv.global_theme.colorbar_orientation = 'horizontal'
# pv.global_theme.colorbar_orientation = 'vertical'
# pv.global_theme.allow_empty_mesh = True

def Plot(obj, result: Union[str,np.ndarray]=None, deformFactor=0.0, coef=1.0, nodeValues=True, 
                color=None, show_edges=False, edge_color='k', line_width=None,
                show_vertices=False, point_size=None, opacity=1.0,
                style='surface', cmap="jet", n_colors=256, clim=None,
                plotter: pv.Plotter=None, show_grid=False, **kwargs):
    """Plot the object obj that can be either a simu, mesh, MultiBlock, PolyData. If you want to plot the solution use plotter.show().

    Parameters
    ----------
    obj : _Simu | Mesh | MultiBlock | PolyData
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
    line_width : _type_, optional
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
    **kwargs:
        Everything that can goes in add_mesh function https://docs.pyvista.org/version/stable/api/plotting/_autosummary/pyvista.Plotter.add_mesh.html#pyvista.Plotter.add_mesh

    Returns
    -------
    pv.Plotter
        The pyvista plotter
    """
    
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

    if plotter is None:        
        plotter = _initPlotter()
    
    if show_grid:
        plotter.show_grid()

    # apply coef to the array
    name = "array" if isinstance(result, np.ndarray) else result
    name = None if pvMesh.n_arrays == 0 else name
    if name != None:
        pvMesh[name] *= coef

    # plot the mesh    
    plotter.add_mesh(pvMesh, scalars=name,
                     color=color,
                     show_edges=show_edges, edge_color=edge_color, line_width=line_width,
                     show_vertices=show_vertices, point_size=point_size,
                     opacity=opacity,
                     style=style, cmap=cmap, n_colors=n_colors, clim=clim, scalar_bar_args={'title': name},
                     **kwargs)

    plotter.camera_position = 'xy'
    if inDim == 3:        
        plotter.camera.elevation += 25
        plotter.camera.azimuth += 10
        plotter.camera.reset_clipping_range()

    return plotter

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

