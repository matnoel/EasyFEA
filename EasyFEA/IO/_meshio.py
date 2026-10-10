# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""meshio bridge shared by Gmsh, Medit and Ensight (https://pypi.org/project/meshio/)."""

from __future__ import annotations
import re
from typing import Any, Iterable, TYPE_CHECKING
import numpy as np

from ..Utilities import Terminal, _types
from ..Utilities._requires import Create_requires_decorator
from ..FEM._mesh import Mesh
from ..FEM._utils import ElemType
from ..FEM._group_elem import _GroupElem, GroupElemFactory
from ._vtk import (
    DICT_ELEMTYPE_TO_VTK,
    DICT_EASYFEA_TO_VTK_INDEXES,
    DICT_VTK_TO_EASYFEA_INDEXES,
)

if TYPE_CHECKING:
    import meshio

requires_meshio = Create_requires_decorator("meshio")


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

DIM_TO_TAG_LETTER: dict[int, str] = {0: "P", 1: "L", 2: "S", 3: "V"}
"""Dimension: the letter Mesher._Set_PhysicalGroups prefixes its tag names with."""

TAG_LETTER_TO_DIM: dict[str, int] = {
    letter: dim for dim, letter in DIM_TO_TAG_LETTER.items()
}
"""Tag letter: dimension."""

_RE_MESHER_TAG = re.compile(rf"^([{''.join(TAG_LETTER_TO_DIM)}])(\d+)$")
"""Matches the tags Mesher creates, e.g. S3, so their number can be reused."""


def _Sort_tags(tags: Iterable[str]) -> list[str]:
    """Sorts tags on the numbers they contain, so that V2 comes before V10."""

    def key(tag: str) -> list[tuple[int, Any]]:
        # the leading 0/1 keeps numbers comparable with numbers and text with text
        return [
            (0, int(part)) if part.isdigit() else (1, part)
            for part in re.split(r"(\d+)", tag)
            if part
        ]

    return sorted(tags, key=key)


def _Get_dict_tags_converter(mesh: Mesh) -> dict[str, int]:
    """Maps every tag of the mesh to an integer, as formats holding only references need.

    A tag Mesher created, such as S3, keeps its own number, so the references already written to
    existing files stay the same and _Set_Tags rebuilds the very same name. Any other name is given
    a free number instead of having its digits stripped, which used to raise on a name without any.
    """

    assert isinstance(mesh, Mesh), "mesh must be a EasyFEA mesh!"

    tags = _Sort_tags(
        {
            tag
            for groupElem in mesh.dict_groupElem.values()
            for tag in groupElem.nodeTags
        }
    )

    dict_tags: dict[str, int] = {}
    used: set[int] = {0}  # 0 marks an untagged entity

    for tag in tags:
        match = _RE_MESHER_TAG.match(tag)
        if match:
            dict_tags[tag] = int(match.group(2))
            used.add(dict_tags[tag])

    value = 1
    for tag in tags:
        if tag in dict_tags:
            continue
        while value in used:
            value += 1
        dict_tags[tag] = value
        used.add(value)

    return dict_tags


def _Get_field_data(
    mesh: Mesh, dict_tags_converter: dict[str, int]
) -> dict[str, _types.IntArray]:
    """Builds the meshio name/number/dimension table, which gmsh writes as $PhysicalNames."""

    field_data: dict[str, _types.IntArray] = {}

    for tag, value in dict_tags_converter.items():
        dims = {
            groupElem.dim
            for groupElem in mesh.dict_groupElem.values()
            if tag in groupElem._dict_elements_tags
        }
        match = _RE_MESHER_TAG.match(tag)
        if match:
            dim = TAG_LETTER_TO_DIM[match.group(1)]
        else:
            # the most specific group carrying the tag: a surface tag on a volume mesh is 2d
            dim = min(dims) if dims else mesh.dim
        field_data[tag] = np.array([value, dim], dtype=int)

    return field_data


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

    import meshio

    assert isinstance(mesh, Mesh), "mesh must be a EasyFEA mesh!"

    cells_dict: dict[str, _types.IntArray] = {}

    list_elements: list[_types.IntArray] = []

    groupElems = list(mesh.dict_groupElem.values())

    # loop over the group elem in the mesh
    for groupElem in groupElems:

        elemType = groupElem.elemType

        # get meshio type
        meshioType = DICT_ELEMTYPE_TO_MESHIO[elemType]

        # get connectivity
        connect = groupElem.connect

        # reorder gmsh/easyfea idx to vtk indexes
        if elemType in DICT_EASYFEA_TO_VTK_INDEXES:
            indexes = DICT_EASYFEA_TO_VTK_INDEXES[elemType]
            connect = connect[:, indexes]

        # set cell dict
        cells_dict[meshioType] = connect
        # get element tags, 0 marking an element belonging to no tag
        element_tags = np.zeros(groupElem.Ne, dtype=int)

        # converts tags and make sure they are integers
        elements_tags = groupElem._dict_elements_tags
        for tag, val in dict_tags_converter.items():
            assert isinstance(val, int), "dict_tags_converter values must be integers."
            # elements, read from the group directly: Get_Elements_Tag raises on a miss
            elements = elements_tags.get(tag)
            if elements is not None:
                element_tags[elements] = int(val)
        list_elements.append(element_tags)

    cell_data = {cellType: list_elements}

    point_sets = _Get_point_sets(groupElems)

    # import in meshio
    try:
        meshioMesh = meshio.Mesh(
            mesh.coord,
            cells_dict,
            None,
            cell_data,
            field_data=_Get_field_data(mesh, dict_tags_converter),
            point_sets=point_sets,
        )

    except KeyError:
        raise KeyError(
            f"To support {mesh.elemType} elements, you need to install meshio using the following meshio fork (https://github.com/matnoel/meshio/tree/medit_higher_order_elements)."
        )

    return meshioMesh


def _Get_point_sets(groupElems: list[_GroupElem]) -> dict[str, _types.IntArray]:
    """Converts EasyFEA tags to meshio named point sets.

    A tag is a set of nodes: Set_Tag derives the elements from them, Mesh.Set_Tag hands the same
    nodes to every group, and Mesh.Save keeps only _dict_nodes_tags. Writing cell_sets as well
    would promise element sets that nothing in EasyFEA stores, so only the nodes are written.
    """

    nodes_tags = [groupElem._dict_nodes_tags for groupElem in groupElems]

    tags = _Sort_tags({tag for dict_tags in nodes_tags for tag in dict_tags})

    return {
        tag: np.unique(
            np.concatenate(
                [d[tag] for d in nodes_tags if tag in d] or [np.empty(0, dtype=int)]
            )
        ).astype(int)
        for tag in tags
    }


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

    import meshio

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
        if cellType in DICT_VTK_TO_EASYFEA_INDEXES:
            indexes = DICT_VTK_TO_EASYFEA_INDEXES[cellType]
            connect = connect[:, indexes]

        # get groupElem
        groupElem = GroupElemFactory.Create(elemType, connect, coordinates)
        dict_groupElem[elemType] = groupElem

    mesh = Mesh(dict_groupElem)

    Terminal.MyPrint("Successfully imported the mesh in EasyFEA.\n")
    print(mesh)

    # named sets keep the tags verbatim; integer refs are the fallback for formats that
    # have nothing else, such as medit
    if meshioMesh.point_sets:
        _Set_Point_Sets(mesh, meshioMesh)
    else:
        _Set_Tags(mesh, _Get_dict_tags(meshioMesh), meshioMesh.field_data)

    return mesh


CELL_DATA_TAG_KEYS = ("gmsh:physical", "medit:ref", "cell_tags", "tags")
"""cell_data keys holding references, most meaningful first."""


def _Get_dict_tags(meshioMesh: meshio.Mesh) -> dict[str, _types.IntArray]:
    """Picks the cell_data array holding the references, as meshio type: references.

    Only one array is used. Flattening every array would let a later key overwrite an earlier one
    for the same cell type, and gmsh writes a gmsh:geometrical array of zeros next to the
    gmsh:physical one, which would wipe out every reference.
    """

    cell_data_dict = meshioMesh.cell_data_dict

    def integers(values: dict[str, _types.IntArray]) -> dict[str, _types.IntArray]:
        return {
            meshioType: tags
            for meshioType, tags in values.items()
            if np.issubdtype(tags.dtype, np.integer)
        }

    for key in CELL_DATA_TAG_KEYS:
        if key in cell_data_dict:
            return integers(cell_data_dict[key])

    for key, values in cell_data_dict.items():
        if key == "gmsh:geometrical":
            continue
        dict_tags = integers(values)
        if dict_tags:
            return dict_tags

    return {}


def _Set_Point_Sets(mesh: Mesh, meshioMesh: meshio.Mesh) -> None:
    """Restores EasyFEA tags from meshio named point sets.

    field_data gives the dimension each tag belongs to, which is what says whether a tag is a
    surface or a volume one. Without it the nodes go to every group, as Mesh.Set_Tag does.
    """

    dict_dims = {
        name: int(value[1])
        for name, value in meshioMesh.field_data.items()
        if len(value) >= 2
    }

    for tag, nodes in meshioMesh.point_sets.items():
        dim = dict_dims.get(tag)
        groupElems = (
            mesh.Get_list_groupElem(dim)
            if dim is not None
            else mesh.dict_groupElem.values()
        )
        for groupElem in groupElems:
            groupElem.Set_Tag(np.asarray(nodes, dtype=int), tag)


def _Set_Tags(
    mesh: Mesh,
    dict_tags: dict[str, _types.IntArray],
    field_data: dict[str, _types.IntArray] = {},
):
    """Set tags for nodes and elements in the EasyFEA mesh.

    Parameters
    ----------
    mesh : Mesh
        EasyFEA mesh object.
    dict_tags : dict[str, _types.IntArray]
        Dictionary of tags for elements.
    field_data : dict[str, _types.IntArray], optional
        meshio name: (reference, dimension) table, gmsh's $PhysicalNames. Names found here are
        used as they are; the rest fall back to the {P,L,S,V}{reference} Mesher builds.
    """

    assert isinstance(mesh, Mesh), "mesh must be a EasyFEA mesh!"
    assert isinstance(dict_tags, dict), "dict_tags must be a dictionnary!"

    # (reference, dimension): name
    dict_names = {
        (int(value[0]), int(value[1])): name
        for name, value in field_data.items()
        if len(value) >= 2
    }

    # retrieve tags

    for elemType, tags in dict_tags.items():
        if elemType not in DICT_MESHIO_TO_ELEMTYPE:
            raise Exception(f"elemType {elemType} is unknown.")
        # (gmshId, nPe, dim, order, Nvertex, Nedge, Nface, Nvolume)
        dim = GroupElemFactory.DICT_ELEMTYPE[DICT_MESHIO_TO_ELEMTYPE[elemType]][2]
        t = DIM_TO_TAG_LETTER[dim]

        # uniqueTags = np.unique(tags)
        uniqueTags, inverse = np.unique(tags, return_inverse=True)
        list_elems = [np.where(inverse == i)[0] for i in range(len(uniqueTags))]

        for groupElem in mesh.Get_list_groupElem(dim):
            # exact match: "triangle" is a substring of "triangle6", so a linear block
            # would otherwise be applied to a quadratic group of the same dimension
            if elemType != DICT_ELEMTYPE_TO_MESHIO[groupElem.elemType]:
                continue
            for elems, tag in zip(list_elems, uniqueTags):
                name = dict_names.get((int(tag), dim))
                if name is None:
                    # writers store 0 for an element belonging to no group, so an unnamed 0
                    # would invent a tag on a mesh that never had one
                    if int(tag) == 0:
                        continue
                    name = t + str(tag)
                nodes = np.unique(groupElem.connect[elems])
                # the elements are known: recomputing them from the nodes would overlap neighbouring tags
                groupElem.Set_Tag(nodes=nodes, tag=name, elements=elems)

            print(f"{groupElem.elemType} -> Ne = {groupElem.Ne}")
