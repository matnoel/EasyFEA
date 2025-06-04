# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Module providing functions used to save FEM-solutions for vizir (https://pyamg.saclay.inria.fr/vizir4.html)."""

import numpy as np
import io

from ..utilities import Folder, MeshIO, Tic, Display
from ..simulations._simu import _Simu, _Init_obj, _Get_values
from ..fem._group_elems import _GroupElem, GroupElemFactory
from ..fem._mesh import Mesh
from ..geoms._utils import (
    _Get_BaryCentric_Coordinates_In_Triangle,
    _Get_BaryCentric_Coordinates_In_Tetrahedron,
    _Get_BaryCentric_Coordinates_In_Segment,
)


def __Get_vizir_HOSolAt_key(groupElem: _GroupElem) -> str:

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


def __Write_HOSolAt_Element(file: io.TextIOWrapper, groupElem: _GroupElem) -> None:

    # get keyword
    keyword = __Get_vizir_HOSolAt_key(groupElem)

    # write ref geom element
    file.write(f"{keyword}{groupElem.order}NodesPositions\n{groupElem.nPe}\n")
    nodesPositions = __Get_NodesPositions(groupElem)
    np.savetxt(file, nodesPositions)


def __Concatenate_results(results: list[np.ndarray], types: list[int]) -> np.ndarray:

    # concatenate values_n in results as an (Nn, ...) array
    assert (
        isinstance(results[0], np.ndarray) and results[0].ndim == 2
    ), "results must be (Nn, dof_n) arrays"
    Nn = results[0].shape[0]

    values_n: np.ndarray = None
    for result_n, type in zip(results, types, strict=True):

        result_n = np.asarray(result_n, dtype=float).reshape(Nn, -1)
        dof_n = result_n.shape[1]

        if type == 1:
            # scalar case
            assert dof_n == 1
        elif type == 2:
            # vector case
            assert 1 < dof_n <= 3
            if dof_n < 3:
                # resize result_n as a (Nn, 3) array
                zeros_e = np.zeros((Nn, 3 - dof_n), dtype=float)
                result_n = np.concatenate((result_n, zeros_e), axis=1)
        else:
            raise Exception("Symmetric/non-symmetric matrices are not yet implemented.")

        # concatenate result_n in values_n
        if values_n is None:
            values_n = result_n
        else:
            values_n = np.concatenate((values_n, result_n), axis=1)

    return values_n


def __Write_HOSolAt_Solution(
    file: io.TextIOWrapper,
    groupElem: _GroupElem,
    connect: np.ndarray,
    values_n: np.ndarray,
    types: list[int],
    order: int,
) -> None:

    # get groupElem informations
    Ne = groupElem.Ne
    assert (
        isinstance(connect, np.ndarray) and connect.ndim == 2 and connect.shape[0] == Ne
    ), "connect must be a (Ne, nPe) array"
    nPe = connect.shape[1]
    if groupElem.order != order:
        assert nPe != groupElem.nPe

    # get values_n informations
    assert (
        isinstance(values_n, np.ndarray) and values_n.ndim == 2
    ), "values_n must be a (Nn, dof_n) array"
    Nn, dof_n = values_n.shape
    assert connect.max() <= Nn

    # get values_n as a (Ne, nPe, dof_n) array
    values_e = values_n[connect].reshape(Ne, nPe, dof_n)
    # get values_e as a (Ne, nPe * dof_n) array
    values_e = values_e.reshape(Ne, -1)

    # write solution
    keyword = __Get_vizir_HOSolAt_key(groupElem)
    file.write(f"\n{keyword}{order}\n{Ne}\n")
    file.write(f"{len(types)} {' '.join([str(type) for type in types])}\n")
    file.write(f"{order} {nPe}\n")

    # write solution array
    np.savetxt(file, values_e)
    file.write("\n")


SOLUTION_TYPES = [1, 2]


def __Write_sol_file(
    dict_groupElem: dict[_GroupElem, np.ndarray],
    values_n: np.ndarray,
    types: list[int],
    order: int,
    folder: str,
    filename: str,
) -> str:

    # init solution file
    solutionFile = Folder.Join(folder, f"{filename}.sol", mkdir=True)

    with open(solutionFile, "w") as f:

        # write first lines
        f.write("MeshVersionFormatted 2\n")
        f.write(f"Dimension 3\n\n")  # the mesh is always in a 3d space

        for groupElem, connect in dict_groupElem.items():

            __Write_HOSolAt_Element(f, groupElem)

            __Write_HOSolAt_Solution(f, groupElem, connect, values_n, types, order)

        f.write("End\n")

    return solutionFile


def Save_simu(
    simu: _Simu, results: list[str], types: list[int], folder: str, N: int = None
) -> str:

    assert isinstance(simu, _Simu)

    results_per_iteration: list[list[np.ndarray]] = []

    for i in range(simu.Niter):

        # Update simulation iteration
        simu.Set_Iter(i)

        list_values_n: list[np.ndarray] = []
        for result in results:
            values_n = simu.Result(result, nodeValues=True).reshape(simu.mesh.Nn, -1)
            list_values_n.append(values_n)

        results_per_iteration.append(list_values_n)

    # save the mesh in Medit format
    mesh = simu.mesh
    mesh_file = MeshIO.EasyFEA_to_Medit(mesh, folder, f"mesh", useBinary=True)

    # get dict_groupElem
    list_groupElem: list[_GroupElem] = [mesh.groupElem]
    list_groupElem.extend(mesh.Get_list_groupElem(mesh.dim - 1))
    dict_groupElem = {groupElem: groupElem.connect for groupElem in list_groupElem}

    sols_file = Save_sols(
        dict_groupElem,
        results_per_iteration,
        types,
        mesh.groupElem.order,
        folder,
        "result",
        N,
    )

    command = f"vizir4 -in {mesh_file} -sols {sols_file}"

    return command


def Save_sols(
    dict_groupElem: dict[_GroupElem, np.ndarray],
    results_per_iteration: list[list[np.ndarray]],
    types: list[int],
    order: int,
    folder: str,
    filename: str = "result",
    N: int = None,
) -> str:

    # .sols and .movie files
    sols_file = open(Folder.Join(folder, f"{filename}.sols", mkdir=True), "w")

    for type in types:
        assert type in SOLUTION_TYPES, f"{type} is not in {SOLUTION_TYPES}"

    # sample the results
    Niter = len(results_per_iteration)
    if N is None:
        N = Niter
    step = Niter // N

    tic = Tic()

    # save meshes and solutions
    for iteration in np.arange(0, Niter, step):

        values_n = __Concatenate_results(results_per_iteration[iteration], types)

        assert len(results_per_iteration[iteration]) == len(types)

        # save the solution
        solution_file = __Write_sol_file(
            dict_groupElem, values_n, types, order, folder, f"solution.{iteration}"
        )

        sols_file.write(f"{solution_file}\n")

        time = tic.Tac("Vizir", f"Save_sols", False)

        rmTime = Tic.Get_Remaining_Time(iteration, Niter - 1, time)

        Display.MyPrint(f"Save_sols {iteration}/{Niter} {rmTime}    ", end="\r")

    sols_file.close()

    return sols_file.name
