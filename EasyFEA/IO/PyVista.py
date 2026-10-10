# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Conversions between EasyFEA and PyVista meshes (https://pyvista.org/)."""

from __future__ import annotations
from collections import Counter
from typing import TYPE_CHECKING
import numpy as np

from ..Utilities import _types
from ..Utilities._requires import Create_requires_decorator
from ..FEM._mesh import Mesh
from ..FEM._utils import ElemType
from ..FEM._group_elem import _GroupElem, GroupElemFactory
from ._vtk import (
    VTKCellType,
    DICT_ELEMTYPE_TO_VTK,
    DICT_VTK_TO_ELEMTYPE,
    DICT_EASYFEA_TO_VTK_INDEXES,
    DICT_VTK_TO_EASYFEA_INDEXES,
)

if TYPE_CHECKING:
    import pyvista as pv

requires_pyvista = Create_requires_decorator("matplotlib", "pyvista")


def Surface_reconstruction(mesh: Mesh) -> Mesh:
    """Reconstructs the missing surfaces in a mesh."""

    assert isinstance(mesh, Mesh), "mesh must be a EasyFEA mesh!"

    if mesh.dim != 3:
        # No need to reconstruct elements for 0D, 1D or 2D meshes
        return mesh

    # get coordinates with orphan nodes
    coordinates = mesh.coord  # DON'T remove orphan nodes!

    allConnect: list[_types.IntArray] = []
    allIds: list[tuple[int]] = []
    elemTypes: list[ElemType] = []
    Nface = 0

    # A mesh can hold several groups of elements in its dimension, prisms next to
    # hexahedrons for instance. Their faces are collected together: a face shared by two
    # groups is then created twice, just like a face shared by two elements of the same
    # group, and is recognized as an interior one below.
    for groupElem in mesh.Get_list_groupElem(mesh.dim):

        connectivity = groupElem.connect

        # get faces to access nodes in connectivity
        faces = groupElem.faces
        Nface += groupElem.Ne * len(faces)

        # loop over each indices
        for face in faces:

            # get connect for the idx
            connect = connectivity[:, face]
            allConnect.extend(connect.copy())

            # Ensure that generated IDs (tuples in this case) are unique
            connect = np.sort(connect, axis=1)

            # add unique ids
            allIds.extend([tuple(nodes) for nodes in connect])

        # a prism gives both triangles and quadrangles, and two groups can give the same
        elemTypes.extend(GroupElemFactory._Get_2d_element_types(groupElem.elemType))

    # make sure all nodes are imported
    assert len(allConnect) == Nface

    # counts the number of repetitions of each identifier
    counts = Counter(allIds)
    # get unique nodes in all created nodes
    uniqueNodes: list[_types.IntArray] = [
        allConnect[i] for i, id in enumerate(allIds) if counts[id] == 1
    ]

    # contstruct the new group of elements from the existing ones
    new_dict_groupElem: dict[ElemType, _GroupElem] = {
        elemType: groupElem
        for elemType, groupElem in mesh.dict_groupElem.items()
        if groupElem.dim != 2
    }

    # create new elements 2d elements
    for elemType in dict.fromkeys(elemTypes):

        # get connect
        nPe = GroupElemFactory.DICT_ELEMTYPE[elemType][1]
        connect = np.asarray(
            [nodes for nodes in uniqueNodes if nodes.size == nPe], dtype=int
        )

        if connect.size == 0:
            # a group can give a type of face that the boundary does not use, e.g. a prism
            # whose triangular caps are all shared with its neighbors
            continue

        # create the new group of elements
        newGroupElem = GroupElemFactory.Create(elemType, connect, coordinates)
        new_dict_groupElem[elemType] = newGroupElem

    # create the new mesh
    newMesh = Mesh(new_dict_groupElem)

    return newMesh


@requires_pyvista
def _Get_pyvista_cell(groupElem: _GroupElem) -> tuple[VTKCellType, _types.IntArray]:

    elemType = groupElem.elemType

    if elemType not in DICT_ELEMTYPE_TO_VTK:
        raise TypeError(f"{elemType} is not implemented yet.")

    # reorder gmsh idx to vtk indexes
    if elemType in DICT_EASYFEA_TO_VTK_INDEXES:
        vtkIndexes = DICT_EASYFEA_TO_VTK_INDEXES[elemType]
    elif elemType in [ElemType.TRI10, ElemType.TRI15]:
        # forced to do this because pyvista simply does not have LAGRANGE_TRIANGLE
        # do not put in DICT_VTK_INDEXES because paraview can read LAGRANGE_TRIANGLE without changing the indices
        vtkIndexes = np.reshape(groupElem.triangles, (-1, 3)).tolist()
        elemType = ElemType.TRI3
    else:
        vtkIndexes = np.arange(groupElem.nPe).tolist()

    # get groupelem connectivity
    connect = groupElem.connect[:, vtkIndexes]
    connect = np.reshape(connect, (-1, np.shape(vtkIndexes)[-1]))

    # create cellData
    cellType = DICT_ELEMTYPE_TO_VTK[elemType]

    return cellType, connect


@requires_pyvista
def EasyFEA_to_PyVista(
    mesh: Mesh,
    coord: _types.FloatArray | None = None,
    useAllElements=True,
) -> pv.UnstructuredGrid:
    """Converts EasyFEA mesh to PyVista Multiblock format.

    Parameters
    ----------
    mesh : Mesh
        EasyFEA mesh object.
    coord : _types.FloatArray, optional
        mesh coordinates, by default None
    useAllElements : bool, optional
        Use all group of elements, by default True
        Uses only the main group of elements if set to False.

    Returns
    -------
    pv.UnstructuredGrid
        pyvista mesh

    Examples
    --------
    Convert and inspect the PyVista mesh:

    >>> from EasyFEA import IO
    >>> pvMesh = IO.PyVista.EasyFEA_to_PyVista(mesh)
    >>> print(pvMesh)
    """

    import pyvista as pv

    assert isinstance(mesh, Mesh), "mesh must be a EasyFEA mesh!"

    # init dict of cell data
    dict_cellData: dict[VTKCellType, np.ndarray] = {}

    # useAllElements -> every group (all dimensions); otherwise only the groups of the main dimension.
    # The latter follows Get_list_groupElem(mesh.dim) order so the cells stay aligned with the element ordering of any per-element field (mesh.Ne concatenation order).
    if useAllElements:
        list_groupElem = list(mesh.dict_groupElem.values())
    else:
        list_groupElem = mesh.Get_list_groupElem(mesh.dim)

    for groupElem in list_groupElem:
        cellType, connect = _Get_pyvista_cell(groupElem)
        dict_cellData[cellType] = connect

    # get mesh coordinates
    if coord is None:
        coordinates = mesh.coord
    else:
        expectedShape = mesh.coord.shape
        assert coord.shape == expectedShape, f"coord must be a {expectedShape} array"
        coordinates = coord

    # get UnstructuredGrid
    pyVistaMesh = pv.UnstructuredGrid(dict_cellData, coordinates)

    return pyVistaMesh


@requires_pyvista
def _GroupElem_to_PyVista(
    groupElem: _GroupElem,
    elements: _types.IntArray | None = None,
) -> pv.UnstructuredGrid:
    """Converts EasyFEA mesh to PyVista Multiblock format.

    Parameters
    ----------
    mesh : Mesh
        EasyFEA mesh object.
    elements : _types.IntArray, optional
        mesh coordinates, by default None

    Returns
    -------
    pv.UnstructuredGrid
        pyvista mesh
    """

    import pyvista as pv

    assert isinstance(groupElem, _GroupElem), "groupElem must be a group of elements!"

    cellType, connect = _Get_pyvista_cell(groupElem)
    connect = groupElem._global_to_local_nodes[connect]

    if isinstance(elements, np.ndarray):
        assert elements.min() >= 0
        assert elements.max() < groupElem.Ne
        connect = connect[elements]

    pyVistaMesh = pv.UnstructuredGrid({cellType: connect}, groupElem.coord)

    return pyVistaMesh


@requires_pyvista
def PyVista_to_EasyFEA(pyVistaMesh: pv.UnstructuredGrid | pv.MultiBlock) -> Mesh:
    """Converts PyVista mesh to EasyFEA format.

    Parameters
    ----------
    pyVistaMesh : pv.UnstructuredGrid | pv.MultiBlock
        PyVista mesh object.

    Returns
    -------
    Mesh
        Converted EasyFEA mesh object.
    """

    import pyvista as pv

    dict_groupElem: dict[ElemType, _GroupElem] = {}

    def read_grid(grid: pv.UnstructuredGrid, part: int):

        coordinates = grid.points

        cellTypes = grid.celltypes.astype(int)

        for cellTypeId in list(set(cellTypes)):
            # get cell and element types
            cellType = VTKCellType(cellTypeId)
            elemType = DICT_VTK_TO_ELEMTYPE[cellType]

            # get connect
            connect = grid.cells_dict[cellTypeId].astype(int)
            # reorder vtk idx to gmsh/easyfea indexes
            if cellType in DICT_VTK_TO_EASYFEA_INDEXES:
                indexes = DICT_VTK_TO_EASYFEA_INDEXES[cellType]
                connect = connect[:, indexes]

            if elemType not in dict_groupElem:
                groupElem = GroupElemFactory.Create(elemType, connect, coordinates)
                groupElem.Set_Tag(groupElem.nodes, str(part))
            else:
                groupElem = dict_groupElem[elemType]
                # get previous tags
                tags = groupElem.nodeTags
                nodeTags = [groupElem.Get_Nodes_Tag(tag) for tag in tags]

                # concate new data in previous groupElem
                newNodes = np.array(list(set(connect.ravel())))
                connect = np.concat((groupElem.connect, connect), axis=0)
                groupElem = GroupElemFactory.Create(elemType, connect, coordinates)

                # add previous tags
                for nodes, tag in zip(nodeTags, tags):
                    groupElem.Set_Tag(nodes, tag)
                # add new tags
                groupElem.Set_Tag(newNodes, str(part))

            dict_groupElem[elemType] = groupElem

    if isinstance(pyVistaMesh, pv.MultiBlock):
        pyVistaMesh = pyVistaMesh.as_unstructured_grid_blocks()

        # loop over blocks
        for part in range(pyVistaMesh.n_blocks):
            grid = pyVistaMesh.get_block(part)
            if isinstance(grid, pv.UnstructuredGrid):
                read_grid(grid, part)

    elif isinstance(pyVistaMesh, pv.UnstructuredGrid):
        read_grid(pyVistaMesh, 0)
    else:
        raise TypeError("Wrong type.")

    mesh = Mesh(dict_groupElem)

    return mesh
