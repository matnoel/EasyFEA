# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Ensight meshes (.geo)."""

from __future__ import annotations
import re
from typing import TYPE_CHECKING
import numpy as np

from ..Utilities import Folder, _types
from ..FEM._mesh import Mesh
from ..FEM._utils import ElemType
from ..FEM._group_elem import _GroupElem, GroupElemFactory
from ._meshio import _EasyFEA_to_Meshio, _Get_dict_tags_converter
from .PyVista import requires_pyvista, PyVista_to_EasyFEA

if TYPE_CHECKING:
    import pyvista as pv


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
    ElemType.PRISM6: "penta6",
    ElemType.PRISM15: "penta15",
    # ElemType.PRISM18: "penta18", # not supported by Ensight
}
"""ElemType: Ensight"""

DICT_ENSIGHT_TO_ELEMTYPE: dict[str, ElemType] = {
    ensight: elemType for elemType, ensight in DICT_ELEMTYPE_TO_ENSIGHT.items()
}
"""Ensight: ElemType"""

# https://ansyshelp.ansys.com/public/account/secured?returnurl=%2F%2F%2F%2F%2FViews%2FSecured%2Fcorp%2Fv242%2Fen%2Fensight_um%2FUM-C9xmlidEnSightGoldCaseFileFormat.html
DICT_EASYFEA_TO_ENSIGHT_INDEXES: dict[ElemType, list[int]] = {
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

DICT_ENSIGHT_TO_EASYFEA_INDEXES: dict[str, list[int]] = {
    DICT_ELEMTYPE_TO_ENSIGHT[elemType]: [indexes.index(i) for i in range(len(indexes))]
    for elemType, indexes in DICT_EASYFEA_TO_ENSIGHT_INDEXES.items()
}
"""Ensight: list[int]"""


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

    import pyvista as pv

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


def Load_mesh(path: str) -> Mesh:
    """Converts Ensight mesh to EasyFEA format.

    Parameters
    ----------
    path : str
        Path to the Ensight geo file.

    Returns
    -------
    Mesh
        Converted EasyFEA mesh object.
    """

    with open(path, "r") as file:
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
        if ensight in DICT_ENSIGHT_TO_EASYFEA_INDEXES:
            indexes = DICT_ENSIGHT_TO_EASYFEA_INDEXES[ensight]
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


def Save_mesh(
    mesh: Mesh, folder: str, name: str, *, useBinary: bool | None = None
) -> str:
    """Converts EasyFEA mesh to Ensight format.

    Parameters
    ----------
    mesh : Mesh
        EasyFEA mesh object.
    folder : str
        Directory to save the Ensight .geo file.
    name : str
        The name of the Ensight .geo file, without the extension.
    useBinary : bool, optional
        Text only: True raises ValueError, by default None.

    Returns
    -------
    str
        Path to the saved Ensight .geo file.
    """

    assert isinstance(mesh, Mesh), "mesh must be a EasyFEA mesh!"
    if useBinary:
        raise ValueError("Ensight .geo files are written as text only.")

    filename = Folder.Join(folder, f"{name}.geo", mkdir=True)

    Nn = mesh.coord.shape[0]

    dict_tags_converter = _Get_dict_tags_converter(mesh)
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
                if elemType in DICT_EASYFEA_TO_ENSIGHT_INDEXES:
                    indexes = DICT_EASYFEA_TO_ENSIGHT_INDEXES[elemType]
                    connect = connect[:, indexes]
                np.savetxt(file, connect, fmt="%8i", delimiter="")

    return filename
