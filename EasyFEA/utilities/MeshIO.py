# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module providing an interface with meshio (https://pypi.org/project/meshio/)."""

import meshio
from typing import Any, Optional

from . import Folder, Display, _types

from ..fem import Mesh, ElemType, GroupElemFactory, _GroupElem
from .PyVista import np, pv

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

DICT_ELEMTYPE_TO_VTK: dict[ElemType, pv.CellType] = {
    # (to Pyvista, to Paraview)
    # see https://dev.pyvista.org/api/utilities/_autosummary/pyvista.celltype#pyvista.CellType
    ElemType.POINT: pv.CellType.VERTEX,
    ElemType.SEG2: pv.CellType.LINE,
    ElemType.SEG3: pv.CellType.QUADRATIC_EDGE,
    ElemType.SEG4: pv.CellType.CUBIC_LINE,
    ElemType.SEG5: pv.CellType.HIGHER_ORDER_EDGE,
    ElemType.TRI3: pv.CellType.TRIANGLE,
    ElemType.TRI6: pv.CellType.QUADRATIC_TRIANGLE,
    ElemType.TRI10: pv.CellType.LAGRANGE_TRIANGLE,
    ElemType.TRI15: pv.CellType.LAGRANGE_TRIANGLE,
    ElemType.QUAD4: pv.CellType.QUAD,
    ElemType.QUAD8: pv.CellType.QUADRATIC_QUAD,
    ElemType.QUAD9: pv.CellType.BIQUADRATIC_QUAD,
    ElemType.TETRA4: pv.CellType.TETRA,
    ElemType.TETRA10: pv.CellType.QUADRATIC_TETRA,
    ElemType.HEXA8: pv.CellType.HEXAHEDRON,
    ElemType.HEXA20: pv.CellType.QUADRATIC_HEXAHEDRON,
    ElemType.HEXA27: pv.CellType.TRIQUADRATIC_HEXAHEDRON,
    ElemType.PRISM6: pv.CellType.WEDGE,
    ElemType.PRISM15: pv.CellType.QUADRATIC_WEDGE,
    ElemType.PRISM18: pv.CellType.BIQUADRATIC_QUADRATIC_WEDGE,
}
"""ElemType: CellType"""

DICT_PYVISTA_TO_ELEMTYPE: dict[pv.CellType, ElemType] = {
    cellType: elemType for elemType, cellType in DICT_ELEMTYPE_TO_VTK.items()
}
"""CellType: ElemType"""

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
    # fmt: on
    # https://dev.pyvista.org/api/examples/_autosummary/pyvista.examples.cells.triquadratichexahedron#pyvista.examples.cells.TriQuadraticHexahedron
    # fmt: off
    ElemType.HEXA27: [
        0, 1, 2, 3, 4, 5, 6, 7,  # vertices
        8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15,  # edges
        22, 23, 21, 24, 20, 25,  # faces
        26,  # volumes
    ],
    # fmt: on
    ElemType.PRISM15: [0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11],
    ElemType.PRISM18: [0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11, 15, 17, 16],
    # nodes 8 and 9 are switch
    ElemType.TETRA10: [0, 1, 2, 3, 4, 5, 6, 7, 9, 8],
}
"""ElemType: list[int]"""

DICT_VTK_TO_GMSH_INDEXES: dict[pv.CellType, list[int]] = {
    DICT_ELEMTYPE_TO_VTK[elemType]: [order.index(i) for i in range(len(order))]
    for elemType, order in DICT_GMSH_TO_VTK_INDEXES.items()
}
"""CellType: list[int]"""

# ----------------------------------------------
# EasyFEA to Meshio
# ----------------------------------------------


def _EasyFEA_to_Meshio(
    mesh: Mesh, dict_tags_converter: dict[Any, int] = {}
) -> meshio.Mesh:
    """Convert EasyFEA mesh to meshio format.

    Parameters
    ----------
    mesh : Mesh
        EasyFEA mesh object.
    dict_tags_converter : dict[Any, int], optional
        Dictionary converting tags to integers (default is {}).

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

        # convert tags and make sure they are integers
        for tag, val in dict_tags_converter.items():
            assert isinstance(val, int), "dict_tags_converter values must be integers."
            # elements
            elements = groupElem.Get_Elements_Tag(tag)
            if elements.size > 0:
                element_tags[elements] = val
        list_elements.append(element_tags)

    cell_data = {"tags": list_elements}

    # import in meshio
    try:
        meshioMesh = meshio.Mesh(
            mesh.coordGlob[:, : mesh.inDim], cells_dict, None, cell_data
        )
    except KeyError:
        raise KeyError(
            f"To support {mesh.elemType} elements, you need to install meshio using the following meshio fork (https://github.com/matnoel/meshio/tree/medit_higher_order_elements)."
        )

    return meshioMesh


def _Meshio_to_EasyFEA(meshioMesh: meshio.Mesh) -> Mesh:
    """Convert meshio mesh to EasyFEA format.

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
        if cellType in DICT_VTK_TO_GMSH_INDEXES.keys():
            indexes = DICT_VTK_TO_GMSH_INDEXES[cellType]
            connect = connect[:, indexes]

        # get groupElem
        groupElem = GroupElemFactory._Create(elemType, connect, coordinates)
        dict_groupElem[elemType] = groupElem

    mesh = Mesh(dict_groupElem)

    Display.MyPrint("Successfully imported the mesh in EasyFEA.")
    print(mesh)

    # set tags
    dict_tags: dict[str, _types.IntArray] = {
        meshioType: tags
        for values in meshioMesh.cell_data_dict.values()
        for meshioType, tags in values.items()
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
        elif elemType.startswith(("line")):
            dim = 1
        elif elemType.startswith(("triangle", "quad")):
            dim = 2
        elif elemType.startswith(("tetra", "hexahedron", "wedge")):
            dim = 3
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

                groupElem._Set_Nodes_Tag(nodes, str(tag))
                groupElem._Set_Elements_Tag(nodes, str(tag))

            print(f"{groupElem.elemType} -> Ne = {groupElem.Ne}")


# ----------------------------------------------
# Medit
# ----------------------------------------------


def EasyFEA_to_Medit(
    mesh: Mesh,
    folder: str,
    name: str,
    dict_tags_converter: dict[str, int] = {},
    useBinary=False,
) -> str:
    """Convert EasyFEA mesh to Medit format.

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


def Medit_to_EasyFEA(meditMesh: str) -> Mesh:
    """Convert Medit mesh to EasyFEA format.

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


def EasyFEA_to_Gmsh(mesh: Mesh, folder: str, name: str, useBinary=False) -> str:
    """Convert EasyFEA mesh to Gmsh format.

    Parameters
    ----------
    mesh : Mesh
        EasyFEA mesh object.
    folder : str
        Directory to save the Gmsh file.
    name : str
        The name of the Gmsh file, without the extension.
    dict_tags_converter : dict[str, int], optional
        Dictionary converting string tags to integers (default is {}).
    useBinary : bool, optional
        Whether to save as binary (default is False).

    Returns
    -------
    str
        Path to the saved Gmsh file.
    """

    assert isinstance(mesh, Mesh), "mesh must be a EasyFEA mesh!"

    # Construct dict_tags as a dictionary with string keys and int values.
    tags = []
    [tags.extend(groupElem.nodeTags) for groupElem in mesh.dict_groupElem.values()]  # type: ignore [func-returns-value]
    tags = np.unique(tags).tolist()
    # change "L1" as 1
    dict_tags = {tag: int(tag[-1]) for tag in tags if len(tag) == 2}
    # For now, it does not import strings different from P{i}, L{i}, S{i}, V{i}.
    # It won't work for long strings.

    meshioMesh = _EasyFEA_to_Meshio(mesh, dict_tags)

    filename = Folder.Join(folder, f"{name}.msh", mkdir=True)

    Display.MyPrint(f"\nCreation of: {filename}", "green")
    meshio.gmsh.write(filename, meshioMesh, "2.2", useBinary)
    # Error with 4.1

    return filename


def Gmsh_to_EasyFEA(gmshMesh: str) -> Mesh:
    """Convert Gmsh mesh to EasyFEA format.

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


def EasyFEA_to_PyVista(
    mesh: Mesh, coord: Optional[_types.FloatArray] = None, useMainGroupElem=True
) -> pv.UnstructuredGrid:
    """Convert EasyFEA mesh to PyVista Multiblock format.

    Parameters
    ----------
    mesh : Mesh
        EasyFEA mesh object.
    coord : _types.FloatArray, optional
        mesh coordinates, by default None
    useMainGroupElem : bool, optional
        Whether to save every group of elements, by default True

    Returns
    -------
    pv.MultiBlock
        pyvista mesh
    """

    assert isinstance(mesh, Mesh), "mesh must be a EasyFEA mesh!"

    # init dict of cell data
    dict_cellData: dict[pv.CellType, np.ndarray] = {}

    for elemType, groupElem in mesh.dict_groupElem.items():
        if useMainGroupElem and groupElem is not mesh.groupElem:
            continue

        if elemType not in DICT_ELEMTYPE_TO_VTK.keys():
            Display.MyPrintError(f"{elemType} is not implemented yet.")
            continue

        # reorder gmsh idx to vtk indexes
        if elemType in DICT_GMSH_TO_VTK_INDEXES.keys():
            vtkIndexes = DICT_GMSH_TO_VTK_INDEXES[elemType]
        else:
            vtkIndexes = np.arange(groupElem.nPe).tolist()

        if elemType in ["TRI10", "TRI15"]:
            # forced to do this because pyvista simply does not have LAGRANGE_TRIANGLE
            # do not put in DICT_VTK_INDEXES because paraview can read LAGRANGE_TRIANGLE without changing the indices
            vtkIndexes = np.reshape(groupElem.triangles, (-1, 3)).tolist()

        # get groupelem connectivity
        connect = groupElem.connect[:, vtkIndexes]
        connect = np.reshape(connect, (-1, np.shape(vtkIndexes)[-1]))

        # create cellData
        cellType = DICT_ELEMTYPE_TO_VTK[elemType]
        dict_cellData[cellType] = connect

    # get mesh coordinates
    coordinates = coord if isinstance(coord, np.ndarray) else mesh.coord

    # get UnstructuredGrid
    pyVistaMesh = pv.UnstructuredGrid(dict_cellData, coordinates)

    return pyVistaMesh


def PyVista_to_EasyFEA(pyVistaMesh: pv.MultiBlock) -> Mesh:
    """Convert PyVista mesh to EasyFEA format.

    Parameters
    ----------
    pyVistaMesh : pv.MultiBlock
        PyVista mesh object.

    Returns
    -------
    Mesh
        Converted EasyFEA mesh object.
    """

    assert isinstance(pyVistaMesh, pv.MultiBlock), "pyVistaMesh must be a MultiBlock!"

    pyVistaMesh = pyVistaMesh.as_unstructured_grid_blocks()

    dict_groupElem: dict[ElemType, _GroupElem] = {}

    for index in range(pyVistaMesh.n_blocks):
        block = pyVistaMesh.get_block(index)

        if isinstance(block, pv.UnstructuredGrid):
            coordGlob = block.points

            cellTypes = block.celltypes.astype(int)

            for cellTypeId in list(set(cellTypes)):
                # get cell and element types
                cellType = pv.CellType(cellTypeId)
                elemType = DICT_PYVISTA_TO_ELEMTYPE[cellType]

                # get connect
                connect = block.cells_dict[cellTypeId].astype(int)
                # reorder vtk idx to gmsh/easyfea indexes
                if cellType in DICT_VTK_TO_GMSH_INDEXES.keys():
                    indexes = DICT_VTK_TO_GMSH_INDEXES[cellType]
                    connect = connect[:, indexes]

                groupElem = GroupElemFactory._Create(elemType, connect, coordGlob)

                dict_groupElem[elemType] = groupElem

    mesh = Mesh(dict_groupElem)

    return mesh


# ----------------------------------------------
# Ensight
# ----------------------------------------------


def Ensight_to_PyVista(geoFile: str) -> pv.MultiBlock:
    """Convert Ensight mesh to PyVista format.

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
    # reader.disable_all_cell_arrays()
    # reader.disable_all_point_arrays()

    # get the pyvista mesh
    pyVistaMesh = reader.read()

    # remove the created case file
    Folder.os.remove(caseFile)

    return pyVistaMesh


def Ensight_to_EasyFEA(geoFile: str) -> Mesh:
    """Convert Ensight mesh to EasyFEA format.

    Parameters
    ----------
    geoFile : str
        Path to the Ensight geo file.

    Returns
    -------
    Mesh
        Converted EasyFEA mesh object.
    """

    pyVistaMesh = Ensight_to_PyVista(geoFile)

    mesh = PyVista_to_EasyFEA(pyVistaMesh)

    return mesh


def Ensight_to_Meshio(geoFile: str) -> Mesh:
    """Convert Ensight mesh to Meshio format.

    Parameters
    ----------
    geoFile : str
        Path to the Ensight geo file.

    Returns
    -------
    Mesh
        Converted EasyFEA mesh object.
    """

    pyVistaMesh = Ensight_to_PyVista(geoFile)

    mesh = PyVista_to_EasyFEA(pyVistaMesh)

    meshioMesh = _EasyFEA_to_Meshio(mesh, {})

    return meshioMesh
