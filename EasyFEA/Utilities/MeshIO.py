# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module providing an interface with meshio (https://pypi.org/project/meshio/)."""

from __future__ import annotations
import re
from collections import Counter
from typing import Any, Optional, Union
import numpy as np
from enum import Enum

from . import Folder, Display, _types

from ..FEM._mesh import Mesh
from ..FEM._utils import ElemType
from ..FEM._group_elem import _GroupElem
from ..FEM._group_elem import GroupElemFactory

from .PyVista import requires_pyvista

from ._requires import Create_requires_decorator

try:
    import pyvista as pv
except ImportError:
    pass

try:
    import meshio
except ImportError:
    pass
requires_meshio = Create_requires_decorator("meshio")

# ----------------------------------------------
# TYPES
# ----------------------------------------------

DICT_ELEMTYPE_TO_MESHIO = {
    ElemType.POINT: "vertex",
    ElemType.SEG2: "line",
    ElemType.SEG3: "line3",
    ElemType.SEG4: "line4",
    ElemType.SEG5: "line5",
    ElemType.TRI3: "triangle",
    ElemType.TRI6: "triangle6",
    ElemType.TRI10: "triangle10",
    ElemType.TRI15: "triangle15",
    ElemType.QUAD4: "quad",
    ElemType.QUAD8: "quad8",
    ElemType.QUAD9: "quad9",
    ElemType.TETRA4: "tetra",
    ElemType.TETRA10: "tetra10",
    ElemType.HEXA8: "hexahedron",
    ElemType.HEXA20: "hexahedron20",
    ElemType.HEXA27: "hexahedron27",
    ElemType.PRISM6: "wedge",
    ElemType.PRISM15: "wedge15",
    ElemType.PRISM18: "wedge18",
}
"""ElemType: meshioType"""

DICT_MESHIO_TO_ELEMTYPE: dict[str, ElemType] = {
    meshio: elemType for elemType, meshio in DICT_ELEMTYPE_TO_MESHIO.items()
}
"""CellType: ElemType"""


class VTKCellType(int, Enum):
    # https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
    # Linear cells
    EMPTY_CELL = 0
    VERTEX = 1
    POLY_VERTEX = 2
    LINE = 3
    POLY_LINE = 4
    TRIANGLE = 5
    TRIANGLE_STRIP = 6
    POLYGON = 7
    PIXEL = 8
    QUAD = 9
    TETRA = 10
    VOXEL = 11
    HEXAHEDRON = 12
    WEDGE = 13
    PYRAMID = 14
    PENTAGONAL_PRISM = 15
    HEXAGONAL_PRISM = 16
    # Quadratic, isoparametric cells
    QUADRATIC_EDGE = 21
    QUADRATIC_TRIANGLE = 22
    QUADRATIC_QUAD = 23
    QUADRATIC_POLYGON = 36
    QUADRATIC_TETRA = 24
    QUADRATIC_HEXAHEDRON = 25
    QUADRATIC_WEDGE = 26
    QUADRATIC_PYRAMID = 27
    BIQUADRATIC_QUAD = 28
    TRIQUADRATIC_HEXAHEDRON = 29
    TRIQUADRATIC_PYRAMID = 37
    QUADRATIC_LINEAR_QUAD = 30
    QUADRATIC_LINEAR_WEDGE = 31
    BIQUADRATIC_QUADRATIC_WEDGE = 32
    BIQUADRATIC_QUADRATIC_HEXAHEDRON = 33
    BIQUADRATIC_TRIANGLE = 34
    # Cubic, isoparametric cell
    CUBIC_LINE = 35
    # Special class of cells formed by convex group of points
    CONVEX_POINT_SET = 41
    # Polyhedron cell (consisting of polygonal faces)
    POLYHEDRON = 42
    # Higher order cells in parametric form
    PARAMETRIC_CURVE = 51
    PARAMETRIC_SURFACE = 52
    PARAMETRIC_TRI_SURFACE = 53
    PARAMETRIC_QUAD_SURFACE = 54
    PARAMETRIC_TETRA_REGION = 55
    PARAMETRIC_HEX_REGION = 56
    # Higher order cells
    HIGHER_ORDER_EDGE = 60
    HIGHER_ORDER_TRIANGLE = 61
    HIGHER_ORDER_QUAD = 62
    HIGHER_ORDER_POLYGON = 63
    HIGHER_ORDER_TETRAHEDRON = 64
    HIGHER_ORDER_WEDGE = 65
    HIGHER_ORDER_PYRAMID = 66
    HIGHER_ORDER_HEXAHEDRON = 67
    # Arbitrary order Lagrange elements (formulated separated from generic higher order cells)
    LAGRANGE_CURVE = 68
    LAGRANGE_TRIANGLE = 69
    LAGRANGE_QUADRILATERAL = 70
    LAGRANGE_TETRAHEDRON = 71
    LAGRANGE_HEXAHEDRON = 72
    LAGRANGE_WEDGE = 73
    LAGRANGE_PYRAMID = 74
    # Arbitrary order Bezier elements (formulated separated from generic higher order cells)
    BEZIER_CURVE = 75
    BEZIER_TRIANGLE = 76
    BEZIER_QUADRILATERAL = 77
    BEZIER_TETRAHEDRON = 78
    BEZIER_HEXAHEDRON = 79
    BEZIER_WEDGE = 80
    BEZIER_PYRAMID = 81
    NUMBER_OF_CELL_TYPES = 82


DICT_ELEMTYPE_TO_VTK: dict[ElemType, VTKCellType] = {
    # (to Pyvista, to Paraview)
    # see https://dev.pyvista.org/api/utilities/_autosummary/pyvista.celltype#pyvista.CellType
    ElemType.POINT: VTKCellType.VERTEX,
    ElemType.SEG2: VTKCellType.LINE,
    ElemType.SEG3: VTKCellType.QUADRATIC_EDGE,
    ElemType.SEG4: VTKCellType.CUBIC_LINE,
    ElemType.SEG5: VTKCellType.HIGHER_ORDER_EDGE,
    ElemType.TRI3: VTKCellType.TRIANGLE,
    ElemType.TRI6: VTKCellType.QUADRATIC_TRIANGLE,
    ElemType.TRI10: VTKCellType.LAGRANGE_TRIANGLE,
    ElemType.TRI15: VTKCellType.LAGRANGE_TRIANGLE,
    ElemType.QUAD4: VTKCellType.QUAD,
    ElemType.QUAD8: VTKCellType.QUADRATIC_QUAD,
    ElemType.QUAD9: VTKCellType.BIQUADRATIC_QUAD,
    ElemType.TETRA4: VTKCellType.TETRA,
    ElemType.TETRA10: VTKCellType.QUADRATIC_TETRA,
    ElemType.HEXA8: VTKCellType.HEXAHEDRON,
    ElemType.HEXA20: VTKCellType.QUADRATIC_HEXAHEDRON,
    ElemType.HEXA27: VTKCellType.TRIQUADRATIC_HEXAHEDRON,
    ElemType.PRISM6: VTKCellType.WEDGE,
    ElemType.PRISM15: VTKCellType.QUADRATIC_WEDGE,
    ElemType.PRISM18: VTKCellType.BIQUADRATIC_QUADRATIC_WEDGE,
}
"""ElemType: CellType"""

DICT_VTK_TO_ELEMTYPE: dict[VTKCellType, ElemType] = {
    cellType: elemType for elemType, cellType in DICT_ELEMTYPE_TO_VTK.items()
}
"""CellType: ElemType"""

DICT_ELEMTYPE_TO_ENSIGHT: dict[ElemType, str] = {
    # (to Ensight)
    ElemType.POINT: "point",
    ElemType.SEG2: "bar2",
    ElemType.SEG3: "bar3",
    # ElemType.SEG4: "bar4",  # not supported by Ensight
    # ElemType.SEG5: "bar5",  # not supported by Ensight
    ElemType.TRI3: "tria3",
    ElemType.TRI6: "tria6",
    # ElemType.TRI10: "tria10", # not supported by Ensight
    # ElemType.TRI15: "tria15", # not supported by Ensight
    ElemType.QUAD4: "quad4",
    ElemType.QUAD8: "quad8",
    # ElemType.QUAD9: "quad9", # not supported by Ensight
    ElemType.TETRA4: "tetra4",
    ElemType.TETRA10: "tetra10",
    ElemType.HEXA8: "hexa8",
    ElemType.HEXA20: "hexa20",
    # ElemType.HEXA27: "hexa27", # not supported by Ensight
    ElemType.PRISM6: "wedge6",
    ElemType.PRISM15: "wedge15",
    # ElemType.PRISM18: "wedge18", # not supported by Ensight
}
"""ElemType: Ensight"""

DICT_ENSIGHT_TO_ELEMTYPE: dict[str, ElemType] = {
    ensight: elemType for elemType, ensight in DICT_ELEMTYPE_TO_ENSIGHT.items()
}
"""Ensight: ElemType"""

# ----------------------------------------------
# INDEXES
# ----------------------------------------------

# reorganize the connectivity order
# because some elements in gmsh don't have the same numbering order as in vtk
# pyvista -> https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.UnstructuredGrid.celltypes.html
# vtk -> https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
# https://dev.pyvista.org/api/utilities/_autosummary/pyvista.celltype
# you can search for vtk elements on the internet
DICT_GMSH_TO_VTK_INDEXES: dict[ElemType, list[int]] = {
    # https://dev.pyvista.org/api/examples/_autosummary/pyvista.examples.cells.quadratichexahedron#pyvista.examples.cells.QuadraticHexahedron
    # fmt: off
    ElemType.HEXA20: [
        0, 1, 2, 3, 4, 5, 6, 7,  # vertices
        8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15 # edges
    ],    
    # https://dev.pyvista.org/api/examples/_autosummary/pyvista.examples.cells.triquadratichexahedron#pyvista.examples.cells.TriQuadraticHexahedron    
    ElemType.HEXA27: [
        0, 1, 2, 3, 4, 5, 6, 7,  # vertices
        8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15,  # edges
        22, 23, 21, 24, 20, 25,  # faces
        26  # volumes
    ],
    ElemType.PRISM15: [
        0, 1, 2, 3, 4, 5, # vertices
        6, 9, 7, 12, 14, 13, 8, 10, 11 # edges
    ],
    ElemType.PRISM18: [
        0, 1, 2, 3, 4, 5, # vertices
        6, 9, 7, 12, 14, 13, 8, 10, 11, # edges
        15, 17, 16 # faces
    ],
    # nodes 8 and 9 are switch
    ElemType.TETRA10: [
        0, 1, 2, 3, # vertices
        4, 5, 6, 7, 9, 8 # faces
    ],
    # fmt: on
}
"""ElemType: list[int]"""

DICT_VTK_TO_GMSH_INDEXES: dict[VTKCellType, list[int]] = {
    DICT_ELEMTYPE_TO_VTK[elemType]: [indexes.index(i) for i in range(len(indexes))]
    for elemType, indexes in DICT_GMSH_TO_VTK_INDEXES.items()
}
"""CellType: list[int]"""

# https://ansyshelp.ansys.com/public/account/secured?returnurl=%2F%2F%2F%2F%2FViews%2FSecured%2Fcorp%2Fv242%2Fen%2Fensight_um%2FUM-C9xmlidEnSightGoldCaseFileFormat.html
DICT_GMSH_TO_ENSIGHT_INDEXES: dict[ElemType, list[int]] = {
    # fmt: off
    ElemType.SEG3: [0, 2, 1],
    # nodes 8 and 9 are switch
    ElemType.TETRA10: [
        0, 1, 2, 3, # vertices
        4, 5, 6, 7, 9, 8 # edges
    ],
    ElemType.PRISM15: [
        0, 1, 2, 3, 4, 5, # vertices
        6, 8, 12, 7, 13, 14, 9, 11, 10 # edges
    ],
    ElemType.HEXA20: [
        0, 1, 2, 3, 4, 5, 6, 7,  # vertices
        8, 11, 16, 9, 17, 10, 18, 19, 12, 15, 13, 14 # edges
    ],
    # fmt: on
}
"""ElemType: list[int]"""

DICT_ENSIGHT_TO_GMSH_INDEXES: dict[str, list[int]] = {
    DICT_ELEMTYPE_TO_ENSIGHT[elemType]: [indexes.index(i) for i in range(len(indexes))]
    for elemType, indexes in DICT_GMSH_TO_ENSIGHT_INDEXES.items()
}
"""Ensight: list[int]"""

# ----------------------------------------------
# Tools
# ----------------------------------------------


def Surface_reconstruction(mesh: Mesh) -> Mesh:
    """Reconstructs the missing surfaces in a mesh."""

    assert isinstance(mesh, Mesh), "mesh must be a EasyFEA mesh!"

    if mesh.dim != 3:
        # No need to reconstruct elements for 0D, 1D or 2D meshes
        return mesh

    useMixedElements = mesh.elemType.startswith(("PRISM"))

    # get coordinates with orphan nodes
    coordinates = mesh.coord  # DON'T remove orphan nodes!

    # get group elem data
    groupElem = mesh.groupElem
    connectivity = groupElem.connect

    # get faces to access nodes in connectivity
    faces = groupElem.faces
    Nface = faces.shape[0]

    allConnect: list[_types.IntArray] = []
    allIds: list[tuple[int]] = []

    # loop over each indices
    for face in faces:

        # get connect for the idx
        connect = connectivity[:, face]
        allConnect.extend(connect.copy())

        # Ensure that generated IDs (tuples in this case) are unique
        connect = np.sort(connect, axis=1)

        # add unique ids
        if useMixedElements:
            allIds.extend([tuple(nodes) for nodes in connect])
        else:
            allIds.extend(list(map(tuple, connect)))

    # make sure all nodes are imported
    assert len(allConnect) == groupElem.Ne * Nface

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
    for elemType in GroupElemFactory._Get_2d_element_types(mesh.elemType):

        # get connect
        nPe = GroupElemFactory.DICT_ELEMTYPE[elemType][1]
        connect = np.asarray(
            [nodes for nodes in uniqueNodes if nodes.size == nPe], dtype=int
        )

        # create the new group of elements
        newGroupElem = GroupElemFactory.Create(elemType, connect, coordinates)
        new_dict_groupElem[elemType] = newGroupElem

        pass

    # create the new mesh
    newMesh = Mesh(new_dict_groupElem)

    return newMesh


def __Get_dict_tags_converter(mesh: Mesh) -> dict[Any, int]:
    """Construct dict_tags as a dictionary with string keys and int values."""

    assert isinstance(mesh, Mesh), "mesh must be a EasyFEA mesh!"

    # get all the tags contained in the mesh
    tags = []
    [tags.extend(groupElem.nodeTags) for groupElem in mesh.dict_groupElem.values()]  # type: ignore [func-returns-value]
    tags = np.unique(tags).tolist()

    # get all int values in each tags
    dict_tags = {tag: int(re.sub(r"\D", "", tag)) for tag in tags}
    # For now, it does not import strings different from P{i}, L{i}, S{i}, V{i}.
    # It won't work for long strings.

    return dict_tags


# ----------------------------------------------
# EasyFEA to Meshio
# ----------------------------------------------


@requires_meshio
def _EasyFEA_to_Meshio(
    mesh: Mesh, dict_tags_converter: dict[Any, int] = {}, cellType: str = "tags"
) -> meshio.Mesh:
    """Converts EasyFEA mesh to meshio format.

    Parameters
    ----------
    mesh : Mesh
        EasyFEA mesh object.
    dict_tags_converter : dict[Any, int], optional
        Dictionary converting tags to integers, by default {}
    cellType : str, optional
        cell type to acces tags, by default "tags"

    Returns
    -------
    meshio.Mesh
        Converted meshio mesh object.
    """

    assert isinstance(mesh, Mesh), "mesh must be a EasyFEA mesh!"

    cells_dict: dict[str, _types.IntArray] = {}

    list_elements: list[_types.IntArray] = []

    # loop over the group elem in the mesh
    for elemType, groupElem in mesh.dict_groupElem.items():

        # get meshio type
        meshioType = DICT_ELEMTYPE_TO_MESHIO[elemType]

        # get connectivity
        connect = groupElem.connect

        # reorder gmsh/easyfea idx to vtk indexes
        if elemType in DICT_GMSH_TO_VTK_INDEXES:
            indexes = DICT_GMSH_TO_VTK_INDEXES[elemType]
            connect = connect[:, indexes]

        # set cell dict
        cells_dict[meshioType] = connect
        # get element tags
        element_tags = np.zeros(groupElem.Ne, dtype=int)

        # converts tags and make sure they are integers
        for tag, val in dict_tags_converter.items():
            assert isinstance(val, int), "dict_tags_converter values must be integers."
            # elements
            elements = groupElem.Get_Elements_Tag(tag)
            if elements.size > 0:
                element_tags[elements] = int(val)
        list_elements.append(element_tags)

    cell_data = {cellType: list_elements}

    # import in meshio
    try:
        meshioMesh = meshio.Mesh(mesh.coord, cells_dict, None, cell_data)

    except KeyError:
        raise KeyError(
            f"To support {mesh.elemType} elements, you need to install meshio using the following meshio fork (https://github.com/matnoel/meshio/tree/medit_higher_order_elements)."
        )

    return meshioMesh


@requires_meshio
def _Meshio_to_EasyFEA(meshioMesh: meshio.Mesh) -> Mesh:
    """Converts meshio mesh to EasyFEA format.

    Parameters
    ----------
    meshioMesh : meshio.Mesh
        Meshio mesh object.

    Returns
    -------
    Mesh
        Converted EasyFEA mesh object.
    """

    assert isinstance(meshioMesh, meshio.Mesh), "meshioMesh must be a meshio mesh!"

    dict_groupElem: dict[ElemType, _GroupElem] = {}

    # get coordinates
    Nn, dim = meshioMesh.points.shape
    coordinates = np.zeros((Nn, 3))
    coordinates[:, :dim] = meshioMesh.points

    for meshioType, connect in meshioMesh.cells_dict.items():

        # get associated elemType
        elemType = DICT_MESHIO_TO_ELEMTYPE[meshioType]

        # reorder vtk idx to gmsh/easyfea indexes
        cellType = DICT_ELEMTYPE_TO_VTK[elemType]
        if cellType in DICT_VTK_TO_GMSH_INDEXES:
            indexes = DICT_VTK_TO_GMSH_INDEXES[cellType]
            connect = connect[:, indexes]

        # get groupElem
        groupElem = GroupElemFactory.Create(elemType, connect, coordinates)
        dict_groupElem[elemType] = groupElem

    mesh = Mesh(dict_groupElem)

    Display.MyPrint("Successfully imported the mesh in EasyFEA.")
    print(mesh)

    # set tags
    dict_tags: dict[str, _types.IntArray] = {
        meshioType: tags
        for values in meshioMesh.cell_data_dict.values()
        for meshioType, tags in values.items()
        if np.issubdtype(tags.dtype, np.integer)
    }
    _Set_Tags(mesh, dict_tags)

    return mesh


def _Set_Tags(mesh: Mesh, dict_tags: dict[str, _types.IntArray]):
    """Set tags for nodes and elements in the EasyFEA mesh.

    Parameters
    ----------
    mesh : Mesh
        EasyFEA mesh object.
    dict_tags : dict[str, _types.IntArray]
        Dictionary of tags for elements.
    """

    assert isinstance(mesh, Mesh), "mesh must be a EasyFEA mesh!"
    assert isinstance(dict_tags, dict), "dict_tags must be a dictionnary!"

    # retrieve tags

    for elemType, tags in dict_tags.items():
        if elemType.startswith(("vertex")):
            dim = 0
            t = "P"
        elif elemType.startswith(("line")):
            dim = 1
            t = "L"
        elif elemType.startswith(("triangle", "quad")):
            dim = 2
            t = "S"
        elif elemType.startswith(("tetra", "hexahedron", "wedge")):
            dim = 3
            t = "V"
        else:
            raise Exception(f"elemType {elemType} is unknown.")

        uniqueTags = np.unique(tags)
        list_elems = [np.where(tags == tag)[0] for tag in uniqueTags]

        for groupElem in mesh.Get_list_groupElem(dim):
            for elems, tag in zip(list_elems, uniqueTags):
                if elems.max() > groupElem.Ne:
                    # We can be here when several elements are the same size.
                    # For example, in the case of a prism, there are triangles and quadrangles at the same time.
                    continue
                nodes_set = set(groupElem.connect[elems].ravel())
                nodes = np.array(list(nodes_set))
                # set tag
                groupElem.Set_Tag(nodes, t + str(tag))

            print(f"{groupElem.elemType} -> Ne = {groupElem.Ne}")


# ----------------------------------------------
# Medit
# ----------------------------------------------


@requires_meshio
def EasyFEA_to_Medit(
    mesh: Mesh,
    folder: str,
    name: str,
    dict_tags_converter: dict[str, int] = {},
    useBinary=False,
) -> str:
    """Converts EasyFEA mesh to Medit format.

    Parameters
    ----------
    mesh : Mesh
        EasyFEA mesh object.
    folder : str
        Directory to save the Medit file.
    name : str
        The name of the Medit file, without the extension.
    dict_tags_converter : dict[str, int], optional
        Dictionary converting string tags to integers (default is {}).
    useBinary : bool, optional
        Whether to save as binary (default is False).

    Returns
    -------
    str
        Path to the saved Medit file.
    """

    assert isinstance(mesh, Mesh), "mesh must be a EasyFEA mesh!"

    meshioMesh = _EasyFEA_to_Meshio(mesh, dict_tags_converter)

    extension = "meshb" if useBinary else "mesh"
    filename = Folder.Join(folder, f"{name}.{extension}", mkdir=True)

    Display.MyPrint(f"\nCreation of: {filename}\n", "green")
    meshio.medit.write(filename, meshioMesh)

    return filename


@requires_meshio
def Medit_to_EasyFEA(meditMesh: str) -> Mesh:
    """Converts Medit mesh to EasyFEA format.

    Parameters
    ----------
    meditMesh : str
        Path to the Medit mesh file.

    Returns
    -------
    Mesh
        Converted EasyFEA mesh object.
    """

    meshioMesh = meshio.medit.read(meditMesh)
    # Please note that your python's meshio must come from https://github.com/matnoel/meshio/tree/medit_higher_order_elements

    if len(meshioMesh.cells) == 0:
        Display.MyPrintError(
            f"The medit mesh:\n {meditMesh}\n does not contain any elements!"
        )
        return None  # type: ignore [return-value]

    mesh = _Meshio_to_EasyFEA(meshioMesh)

    return mesh


# ----------------------------------------------
# Gmsh
# ----------------------------------------------


@requires_meshio
def EasyFEA_to_Gmsh(mesh: Mesh, folder: str, name: str, useBinary=False) -> str:
    """Converts EasyFEA mesh to Gmsh format.

    Parameters
    ----------
    mesh : Mesh
        EasyFEA mesh object.
    folder : str
        Directory to save the Gmsh file.
    name : str
        The name of the Gmsh file, without the extension.
    useBinary : bool, optional
        Whether to save as binary (default is False).

    Returns
    -------
    str
        Path to the saved Gmsh file.
    """

    assert isinstance(mesh, Mesh), "mesh must be a EasyFEA mesh!"

    dict_tags_converter = __Get_dict_tags_converter(mesh)

    meshioMesh = _EasyFEA_to_Meshio(mesh, dict_tags_converter, "cell_tags")

    filename = Folder.Join(folder, f"{name}.msh", mkdir=True)

    Display.MyPrint(f"\nCreation of: {filename}", "green")

    meshio.gmsh.write(filename, meshioMesh, "2.2", useBinary)
    # Error with 4.1

    return filename


@requires_meshio
def Gmsh_to_EasyFEA(gmshMesh: str) -> Mesh:
    """Converts Gmsh mesh to EasyFEA format.

    Args:
        gmshMesh (str): Path to the Gmsh mesh file.

    Returns:
        Mesh: Converted EasyFEA mesh object.
    """

    meshioMesh: meshio.Mesh = meshio.gmsh.read(gmshMesh)

    if len(meshioMesh.cells) == 0:
        Display.MyPrintError(
            f"The gmsh mesh:\n {gmshMesh}\n does not contain any elements!"
        )
        return None  # type: ignore [return-value]

    mesh = _Meshio_to_EasyFEA(meshioMesh)

    return mesh


# ----------------------------------------------
# PyVista
# ----------------------------------------------


@requires_pyvista
def _Get_pyvista_cell(groupElem: _GroupElem) -> tuple[VTKCellType, _types.IntArray]:

    elemType = groupElem.elemType

    if elemType not in DICT_ELEMTYPE_TO_VTK:
        raise TypeError(f"{elemType} is not implemented yet.")

    # reorder gmsh idx to vtk indexes
    if elemType in DICT_GMSH_TO_VTK_INDEXES:
        vtkIndexes = DICT_GMSH_TO_VTK_INDEXES[elemType]
    elif elemType in ["TRI10", "TRI15"]:
        # forced to do this because pyvista simply does not have LAGRANGE_TRIANGLE
        # do not put in DICT_VTK_INDEXES because paraview can read LAGRANGE_TRIANGLE without changing the indices
        vtkIndexes = np.reshape(groupElem.triangles, (-1, 3)).tolist()
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
    mesh: Mesh, coord: Optional[_types.FloatArray] = None, useAllElements=True
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
    """

    assert isinstance(mesh, Mesh), "mesh must be a EasyFEA mesh!"

    # init dict of cell data
    dict_cellData: dict[VTKCellType, np.ndarray] = {}

    for groupElem in mesh.dict_groupElem.values():
        if not useAllElements and groupElem is not mesh.groupElem:
            continue

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
    groupElem: _GroupElem, elements: Optional[_types.IntArray] = None
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

    assert isinstance(groupElem, _GroupElem), "groupElem must be a group of elements!"

    cellType, connect = _Get_pyvista_cell(groupElem)

    if isinstance(elements, np.ndarray):
        assert elements.min() >= 0
        assert elements.max() < groupElem.Ne
        connect = connect[elements]

    pyVistaMesh = pv.UnstructuredGrid({cellType: connect}, groupElem.coordGlob)

    return pyVistaMesh


@requires_pyvista
def PyVista_to_EasyFEA(pyVistaMesh: Union[pv.UnstructuredGrid, pv.MultiBlock]) -> Mesh:
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

    dict_groupElem: dict[ElemType, _GroupElem] = {}

    def read_grid(grid: pv.UnstructuredGrid, part: int):

        coordGlob = grid.points

        cellTypes = grid.celltypes.astype(int)

        for cellTypeId in list(set(cellTypes)):
            # get cell and element types
            cellType = VTKCellType(cellTypeId)
            elemType = DICT_VTK_TO_ELEMTYPE[cellType]

            # get connect
            connect = grid.cells_dict[cellTypeId].astype(int)
            # reorder vtk idx to gmsh/easyfea indexes
            if cellType in DICT_VTK_TO_GMSH_INDEXES:
                indexes = DICT_VTK_TO_GMSH_INDEXES[cellType]
                connect = connect[:, indexes]

            if elemType not in dict_groupElem:
                groupElem = GroupElemFactory.Create(elemType, connect, coordGlob)
                groupElem.Set_Tag(groupElem.nodes, str(part))
            else:
                groupElem = dict_groupElem[elemType]
                # get previous tags
                tags = groupElem.nodeTags
                nodeTags = [groupElem.Get_Nodes_Tag(tag) for tag in tags]

                # concate new data in previous groupElem
                newNodes = np.array(list(set(connect.ravel())))
                connect = np.concat((groupElem.connect, connect), axis=0)
                groupElem = GroupElemFactory.Create(elemType, connect, coordGlob)

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
        raise TypeError("Wrond type.")

    mesh = Mesh(dict_groupElem)

    return mesh


# ----------------------------------------------
# Ensight
# ----------------------------------------------


@requires_pyvista
def _Ensight_to_PyVista(geoFile: str) -> pv.MultiBlock:
    """Converts Ensight mesh to PyVista format.

    Parameters
    ----------
    geoFile : str
        Path to the Ensight geo file.

    Returns
    -------
    Mesh
        Converted PyVista mesh object.
    """

    # create case file
    folder = Folder.Dir(geoFile)
    name = Folder.os.path.basename(geoFile).split(".geo")[0]
    caseFile = Folder.Join(folder, f"{name}.case")
    with open(caseFile, "w") as f:
        f.write("FORMAT\n")
        f.write("type: ensight\n")
        f.write("GEOMETRY\n")
        f.write(f"model: 1 {name}.geo\n")

    # import case to pyvista
    reader = pv.EnSightReader(caseFile)

    # get thepyvista Multi pyvista mesh
    pyVistaMesh = reader.read()

    # remove the created case file
    Folder.os.remove(caseFile)

    return pyVistaMesh


@requires_pyvista
def _Ensight_to_Meshio(geoFile: str) -> Mesh:
    """Converts Ensight mesh to Meshio format.

    Parameters
    ----------
    geoFile : str
        Path to the Ensight geo file.

    Returns
    -------
    Mesh
        Converted EasyFEA mesh object.
    """

    pyVistaMesh = _Ensight_to_PyVista(geoFile)

    mesh = PyVista_to_EasyFEA(pyVistaMesh)

    meshioMesh = _EasyFEA_to_Meshio(mesh, {})

    return meshioMesh


def Ensight_to_EasyFEA(geoFile: str) -> Mesh:
    """Converts Ensight mesh to EasyFEA format.

    Parameters
    ----------
    geoFile : str
        Path to the Ensight geo file.

    Returns
    -------
    Mesh
        Converted EasyFEA mesh object.
    """

    with open(geoFile, "r") as file:
        lines = file.readlines()

    dict_ensightType_data: dict[str, dict[str, _types.IntArray]] = {}

    index = 0
    while index < len(lines):

        line = lines[index].strip()

        if line == "coordinates":
            index += 1
            Nn = int(lines[index].strip())

            coordinates = np.array(
                [
                    [
                        float(value)
                        # [+-]? (get sign)
                        # \d+\.\d+ (decimal part of the number)
                        # e[+-]?\d+ (exponential part of the number)
                        for value in re.findall(r"[+-]?\d+\.\d+e[+-]?\d+", line)
                    ]
                    for line in lines[index + 1 : index + 1 + Nn]
                ],
                dtype=float,
            )
            index += 1 + Nn  # don't change

        elif line.startswith("part"):

            # get description
            index += 1
            description = lines[index].strip()
            tag = re.sub(r"\D", "", description)
            # get ensightType
            index += 1
            ensight = lines[index].strip()
            # get Ne
            index += 1
            Ne = int(lines[index].strip())
            # get connect
            connect = np.array(
                [
                    [int(value) for value in line.strip().split()]
                    for line in lines[index + 1 : index + 1 + Ne]
                ],
                dtype=int,
            )
            # start connect index from 0
            connect -= 1
            index += 1 + Ne  # don't change

            # append data
            if ensight not in dict_ensightType_data:
                dict_ensightType_data[ensight] = {tag: connect}
            else:
                dict_ensightType_data[ensight][tag] = connect

        else:
            index += 1

    # create groups of elements
    dict_groupElem: dict[ElemType, _GroupElem] = {}
    for ensight, dict_data in dict_ensightType_data.items():

        elemType = DICT_ENSIGHT_TO_ELEMTYPE[ensight]

        # import connect
        connect = np.concat(
            [connect for connect in dict_data.values()], axis=0, dtype=int
        )

        # make sur connect is unique
        unique_rows = list(set(tuple(row) for row in connect))
        connect = np.array(unique_rows, dtype=int)

        # reorder connect
        if ensight in DICT_ENSIGHT_TO_GMSH_INDEXES:
            indexes = DICT_ENSIGHT_TO_GMSH_INDEXES[ensight]
            connect = connect[:, indexes]
        # create the group of elements
        groupElem = GroupElemFactory.Create(elemType, connect, coordinates)

        # Set tags
        for tag, connect in dict_data.items():
            nodes = list(set(connect.ravel()))
            groupElem.Set_Tag(np.asarray(nodes, dtype=int), tag)

        # add group of elements
        dict_groupElem[elemType] = groupElem

    # create the mesh
    mesh = Mesh(dict_groupElem)

    return mesh


def EasyFEA_to_Ensight(mesh: Mesh, folder: str, name: str) -> str:
    """Converts EasyFEA mesh to Gmsh format.

    Parameters
    ----------
    mesh : Mesh
        EasyFEA mesh object.
    folder : str
        Directory to save the Ensight .geo file.
    name : str
        The name of the Ensight .geo file, without the extension.

    Returns
    -------
    str
        Path to the saved Ensight .geo file.
    """

    assert isinstance(mesh, Mesh), "mesh must be a EasyFEA mesh!"

    filename = Folder.Join(folder, f"{name}.geo", mkdir=True)

    Nn = mesh.coord.shape[0]

    dict_tags_converter = __Get_dict_tags_converter(mesh)
    parts = np.unique([value for value in dict_tags_converter.values()])

    def get_line(number: int, pos: int = 8):
        return f"{' '*(pos-len(str(number)))}{number}"

    # get tags with groupElem and elements
    dict_tags = {
        tag: (groupElem, groupElem.Get_Elements_Tag(tag))
        for groupElem in mesh.dict_groupElem.values()
        for tag in groupElem.elementTags
    }

    # offset to ensure that parts starts at 0
    offset = 1 if parts.min() == 0 else 0

    with open(filename, "w") as file:

        file.write("Geometry ensight6 file\n")
        file.write(f"{name}\n")
        file.write("node id assign\n")
        file.write("element id assign\n")
        file.write("coordinates\n")
        file.write(get_line(Nn) + "\n")
        np.savetxt(file, mesh.coord, fmt="%12.5e", delimiter="")

        for part in parts:

            for tag, (groupElem, elements) in dict_tags.items():

                if dict_tags_converter[tag] != part:
                    continue

                # write part (starts at 1)
                file.write(f"part{get_line(part+offset)}\n")
                # write description
                file.write(f"{groupElem.topology}_subdomain {tag}\n")
                # write ensight name
                elemType = groupElem.elemType
                file.write(f"{DICT_ELEMTYPE_TO_ENSIGHT[elemType]}\n")
                # write elements
                file.write(get_line(elements.size) + "\n")
                # write connect (starts at 1)
                connect = groupElem.connect[elements] + 1
                if elemType in DICT_GMSH_TO_ENSIGHT_INDEXES:
                    indexes = DICT_GMSH_TO_ENSIGHT_INDEXES[elemType]
                    connect = connect[:, indexes]
                np.savetxt(file, connect, fmt="%8i", delimiter="")

    return filename
