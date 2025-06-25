# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module providing an interface with meshio (https://pypi.org/project/meshio/)."""

import meshio
from typing import Any

from . import Folder, Display, _types

from ..fem import Mesher, Mesh, ElemType
from .PyVista import DICT_GMSH_TO_VTK, np


# ----------------------------------------------
# TYPES
# ----------------------------------------------

DICT_MESHIO_TYPES = {
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

DICT_GMSH_TO_MESHIO_INDEXES = DICT_GMSH_TO_VTK

# ----------------------------------------------
# EasyFEA to Meshio
# ----------------------------------------------


def _EasyFEA_to_Meshio(
    mesh: Mesh, cell_name: str, dict_tags_converter: dict[Any, int] = {}
) -> meshio.Mesh:
    """Convert EasyFEA mesh to meshio format.

    Args:
        mesh (Mesh): EasyFEA mesh object.
        cell_name (str): Name of the cell data.
        dict_tags_converter (dict[Any, int], optional): Dictionary converting tags to integers. Defaults to {}.

    Returns:
        meshio.Mesh: Converted meshio mesh object.
    """

    assert isinstance(mesh, Mesh), "mesh must be a EasyFEA mesh!"

    cells_dict: dict[str, _types.IntArray] = {}

    list_elements: list[_types.IntArray] = []

    for elemType, groupElem in mesh.dict_groupElem.items():

        if elemType in DICT_MESHIO_TYPES.keys():
            meshioType = DICT_MESHIO_TYPES[elemType]
        else:
            continue

        if elemType in DICT_GMSH_TO_MESHIO_INDEXES:
            vtKindexes = DICT_GMSH_TO_MESHIO_INDEXES[elemType]
        else:
            vtKindexes = np.arange(groupElem.nPe).tolist()

        cells_dict[meshioType] = groupElem.connect[:, vtKindexes]

        element_tags = np.zeros(groupElem.Ne, dtype=int)

        for tag, val in dict_tags_converter.items():
            assert isinstance(val, int), "dict_tags_converter values must be integers."
            # elements
            elements = groupElem.Get_Elements_Tag(tag)
            if elements.size > 0:
                element_tags[elements] = val

        list_elements.append(element_tags)

    cell_data = {cell_name: list_elements}

    try:
        meshio_mesh = meshio.Mesh(
            mesh.coordGlob[:, : mesh.inDim], cells_dict, None, cell_data
        )
    except KeyError:
        raise KeyError(
            f"To support {mesh.elemType} elements, you need to install meshio using the following meshio fork (https://github.com/matnoel/meshio/tree/medit_higher_order_elements)."
        )

    return meshio_mesh


def _Meshio_to_EasyFEA(
    meshio_mesh: meshio.Mesh, mesh_file: str, dict_tags: dict[str, _types.IntArray]
) -> Mesh:
    """Convert meshio mesh to EasyFEA format.

    Args:
        meshio_mesh (meshio.Mesh): Meshio mesh object.
        mesh_file (str): Path to the mesh file.
        dict_tags (dict[str, _types.IntArray]): Dictionary of tags that should be contained within `meshio_mesh.cell_data_dict`.

    Returns:
        Mesh: Converted EasyFEA mesh object.
    """

    assert isinstance(meshio_mesh, meshio.Mesh), "meshio_mesh must be a meshio mesh!"

    keys: list[str] = meshio_mesh.cell_data_dict.keys()
    is_gmsh_compatible = any(key.startswith("gmsh") for key in keys)

    if is_gmsh_compatible:
        mesh = Mesher().Mesh_Import_mesh(mesh_file)
    else:
        gmsh_mesh = Folder.Join(Folder.Dir(mesh_file), "tmp_mesh.msh")
        meshio.gmsh.write(gmsh_mesh, meshio_mesh, fmt_version="2.2", binary=True)
        # fmt_version="4.1" does not work!
        mesh = Mesher().Mesh_Import_mesh(gmsh_mesh)
        Folder.os.remove(gmsh_mesh)

    Display.MyPrint("Successfully imported the mesh in EasyFEA.")
    print(mesh)

    _Set_Tags(mesh, dict_tags)

    return mesh


def _Set_Tags(mesh: Mesh, dict_tags: dict[str, _types.IntArray]):
    """Set tags for nodes and elements in the EasyFEA mesh.

    Args:
        mesh (Mesh): EasyFEA mesh object.
        dict_tags (dict[str, _types.IntArray]): Dictionary of tags for elements.
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

    Args:
        mesh (Mesh): EasyFEA mesh object.
        folder (str): Directory to save the Medit file.
        name (str): The name of the Medit file, without the extension.
        dict_tags_converter (dict[str, int], optional): Dictionary converting string tags to integers. Defaults to {}.
        useBinary (bool, optional): Whether to save as binary. Defaults to False.

    Returns:
        str: Path to the saved Medit file.
    """

    meshio_mesh = _EasyFEA_to_Meshio(mesh, "medit.ref", dict_tags_converter)

    extension = "meshb" if useBinary else "mesh"
    filename = Folder.Join(folder, f"{name}.{extension}", mkdir=True)

    Display.MyPrint(f"\nCreation of: {filename}\n", "green")
    meshio.medit.write(filename, meshio_mesh)

    return filename


def Medit_to_EasyFEA(meditMesh: str) -> Mesh:
    """Convert Medit mesh to EasyFEA format.

    Args:
        meditMesh (str): Path to the Medit mesh file.

    Returns:
        Mesh: Converted EasyFEA mesh object.
    """

    meshio_mesh = meshio.medit.read(meditMesh)
    # Please note that your python's meshio must come from https://github.com/matnoel/meshio/tree/medit_higher_order_elements

    if len(meshio_mesh.cells) == 0:
        Display.MyPrintError(
            f"The medit mesh:\n {meditMesh}\n does not contain any elements!"
        )
        return None  # type: ignore [return-value]

    dict_tags = meshio_mesh.cell_data_dict["medit:ref"]

    mesh = _Meshio_to_EasyFEA(meshio_mesh, meditMesh, dict_tags)

    return mesh


# ----------------------------------------------
# Gmsh
# ----------------------------------------------


def EasyFEA_to_Gmsh(mesh: Mesh, folder: str, name: str, useBinary=False) -> str:
    """Convert EasyFEA mesh to Gmsh format.

    Args:
        mesh (Mesh): EasyFEA mesh object.
        folder (str): Directory to save the Gmsh file.
        name (str): The name of the Gmsh file, without the extension.
        dict_tags_converter (dict[str, int], optional): Dictionary converting string tags to integers. Defaults to {}.
        useBinary (bool, optional): Whether to save as binary. Defaults to False.

    Returns:
        str: Path to the saved Gmsh file.
    """

    # Construct dict_tags as a dictionary with string keys and int values.
    tags = []
    [tags.extend(groupElem.nodeTags) for groupElem in mesh.dict_groupElem.values()]  # type: ignore [func-returns-value]
    tags = np.unique(tags).tolist()
    # change "L1" as 1
    dict_tags = {tag: int(tag[-1]) for tag in tags if len(tag) == 2}
    # For now, it does not import strings different from P{i}, L{i}, S{i}, V{i}.
    # It won't work for long strings.

    meshio_mesh = _EasyFEA_to_Meshio(mesh, "gmsh:physical", dict_tags)

    filename = Folder.Join(folder, f"{name}.msh", mkdir=True)

    Display.MyPrint(f"\nCreation of: {filename}", "green")
    meshio.gmsh.write(filename, meshio_mesh, "2.2", useBinary)
    # Error with 4.1

    return filename


def Gmsh_to_EasyFEA(gmshMesh: str) -> Mesh:
    """Convert Gmsh mesh to EasyFEA format.

    Args:
        gmshMesh (str): Path to the Gmsh mesh file.

    Returns:
        Mesh: Converted EasyFEA mesh object.
    """

    meshio_mesh: meshio.Mesh = meshio.gmsh.read(gmshMesh)

    if len(meshio_mesh.cells) == 0:
        Display.MyPrintError(
            f"The gmsh mesh:\n {gmshMesh}\n does not contain any elements!"
        )
        return None  # type: ignore [return-value]

    dict_tags = meshio_mesh.cell_data_dict["gmsh:physical"]

    mesh = _Meshio_to_EasyFEA(meshio_mesh, gmshMesh, dict_tags)

    return mesh
