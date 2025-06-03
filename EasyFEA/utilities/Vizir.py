# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Module providing functions used to save FEM-solutions for vizir (https://pyamg.saclay.inria.fr/vizir4.html)."""

from typing import Union
import numpy as np
import io

from ..utilities import Folder, MeshIO
from ..simulations._simu import _Simu, _Init_obj, _Get_values
from ..fem._group_elems import _GroupElem
from ..fem._mesh import Mesh
from ..geoms._utils import (
    _Get_BaryCentric_Coordinates_In_Triangle,
    _Get_BaryCentric_Coordinates_In_Tetrahedron,
    _Get_BaryCentric_Coordinates_In_Segment,
)


def __Get_vizir_solution_key(groupElem: _GroupElem) -> str:

    elemType = groupElem.elemType

    if elemType.startswith("SEG"):
        keyword = "HOSolAtEdgesP"
    elif elemType.startswith("HEXA"):
        keyword = "HOSolAtHexahedraQ"
    elif elemType.startswith("PRISM"):
        keyword = "HOSolAtPrismsP"
    elif elemType.startswith("QUAD"):
        keyword = "HOSolAtQuadrilateralsQ"
    elif elemType.startswith("TETRA"):
        keyword = "HOSolAtTetrahedraP"
    elif elemType.startswith("TRI"):
        keyword = "HOSolAtTrianglesP"
    else:
        raise TypeError("Unknown element type")

    return keyword


def _Get_BaryCentric_Coordinates(groupElem: _GroupElem) -> np.ndarray:

    elemType = groupElem.elemType
    local_coords = groupElem.Get_Local_Coords()
    vertices_coords = local_coords[: groupElem.nbCorners]

    if elemType.startswith("SEG"):
        coordinates = _Get_BaryCentric_Coordinates_In_Segment(
            vertices_coords, local_coords
        )
    elif elemType.startswith("TETRA"):
        coordinates = _Get_BaryCentric_Coordinates_In_Tetrahedron(
            vertices_coords, local_coords
        )
    elif elemType.startswith("TRI"):
        coordinates = _Get_BaryCentric_Coordinates_In_Triangle(
            vertices_coords, local_coords
        )
    else:
        raise TypeError("Unknown element type")

    return coordinates


def __Get_NodesPositions(groupElem: _GroupElem) -> np.ndarray:

    elemType = groupElem.elemType
    local_coords = groupElem.Get_Local_Coords().astype(float)

    if elemType.startswith(("SEG", "TETRA", "TRI")):
        nodes_positions = _Get_BaryCentric_Coordinates(groupElem)
    elif elemType.startswith("PRISM"):
        local_coords2d = local_coords.copy()
        local_coords2d[:, 2] = 0
        nodes_positions = _Get_BaryCentric_Coordinates_In_Triangle(
            local_coords2d[:3], local_coords2d
        )
        # get z coords withn 0 and 1
        z_coords = local_coords[:, 2].reshape(-1, 1)
        z_coords -= groupElem.origin[2]
        z_coords /= z_coords.max()

        nodes_positions = np.concatenate((nodes_positions, z_coords), axis=1)
    else:
        nodes_positions = local_coords
        nodes_positions -= groupElem.origin
        nodes_positions /= nodes_positions.max()

    return nodes_positions


def __Write_RefGeomElt(
    file: io.TextIOWrapper, groupElem: _GroupElem, solutionOrder: int
) -> None:

    # get keyword
    keyword = __Get_vizir_solution_key(groupElem)

    # write ref geom element
    file.write(f"{keyword}{solutionOrder}NodesPositions\n{groupElem.nPe}\n")
    nodesPositions = __Get_NodesPositions(groupElem)
    np.savetxt(file, nodesPositions)


def __Write_Solution(
    file: io.TextIOWrapper,
    groupElem: _GroupElem,
    dofsValues: np.ndarray,
    dof_n: int,
    resultOrder: int,
) -> None:

    # get dofsValues as a (Ne, nPe, dof_n) array
    assembly_e = groupElem.Get_assembly_e(dof_n)
    assert dofsValues.ndim == 1, "dofsValues must be a 1d array"
    dofsValues_e = dofsValues[assembly_e].reshape(groupElem.Ne, groupElem.nPe, -1)

    if dof_n == 1:
        # scalar case
        solutionDim = 1
    elif 1 < dof_n <= 3:
        # vector case
        solutionDim = 2
        zeros_e = np.zeros((groupElem.Ne, groupElem.nPe, 1))
        for _ in range(3 - dof_n):
            dofsValues_e = np.concatenate((dofsValues_e, zeros_e), axis=-1)
    else:
        raise Exception("Symmetric/non-symmetric matrices are not yet implemented.")
    dofsValues_e = dofsValues_e.reshape(groupElem.Ne, -1)

    # write solution
    keyword = __Get_vizir_solution_key(groupElem)
    file.write(f"\n{keyword}{resultOrder}\n{groupElem.Ne}\n")
    file.write(f"1 {solutionDim}\n")
    file.write(f"{groupElem.order} {groupElem.nPe}\n")

    # write solution array
    np.savetxt(file, dofsValues_e)
    file.write("\n")


def __Write_solution_file(
    mesh: Mesh,
    values_n: np.ndarray,
    resultOrder: int,
    folder: str,
    filename: str,
) -> str:

    assert (
        values_n.ndim == 2 and values_n.shape[0] == mesh.Nn
    ), "values_n must be a (mesh.Nn, dof_n) array"
    dof_n = values_n.shape[1]

    list_groupElem = mesh.Get_list_groupElem()
    list_groupElem.extend(mesh.Get_list_groupElem(mesh.dim - 1))

    # init solution file
    solutionFile = Folder.Join(folder, f"{filename}.sol", mkdir=True)

    with open(solutionFile, "w") as f:

        # write first lines
        f.write("MeshVersionFormatted 2\n")
        f.write(f"Dimension 3\n\n")  # the mesh is always in a 3d space

        for groupElem in list_groupElem:

            __Write_RefGeomElt(f, groupElem, resultOrder)

            __Write_Solution(f, groupElem, values_n.ravel(), dof_n, resultOrder)

        f.write("End\n")

    return solutionFile


def Save_simu(simu: _Simu, result: str, folder: str, filename: str) -> None:

    assert isinstance(simu, _Simu)

    list_values_n: list[np.ndarray] = []

    for i in range(simu.Niter):

        # Update simulation iteration
        simu.Set_Iter(i)

        values_n = simu.Result(result, nodeValues=True).reshape(simu.mesh.Nn, -1)
        list_values_n.append(values_n)

    Save(simu.mesh, list_values_n, folder, filename)


def Save(
    mesh: Mesh,
    list_values_n: list[np.ndarray],
    folder: str,
    filename: str,
    resultOrder: int = None,
):

    # save the mesh in Medit format
    mesh_file = MeshIO.EasyFEA_to_Medit(mesh, folder, f"mesh", useBinary=True)

    # .sols and .movie files
    sols_file = open(Folder.Join(folder, f"{filename}.sols", mkdir=True), "w")
    movie_file = open(Folder.Join(folder, f"vizir.movie", mkdir=True), "w")

    # save meshes and solutions
    for i, values_n in enumerate(list_values_n):

        # get solution order
        if resultOrder is None:
            resultOrder = mesh.groupElem.order

        # save the solution
        solution_file = __Write_solution_file(
            mesh, values_n, resultOrder, folder, f"{filename}.{i}"
        )

        sols_file.write(f"{solution_file}\n")
        movie_file.write(f"{mesh_file}\t{solution_file}\n")

    sols_file.close()
    movie_file.close()

    command = f"vizir4 -in {mesh_file} -sols {sols_file.name}"

    print(command)

    return command
