# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module providing functions used to save FEM-solutions for vizir (https://pyamg.saclay.inria.fr/vizir4.html)."""

from typing import Optional, TYPE_CHECKING
import numpy as np
import io

from ..Utilities import Folder, MeshIO, _types
from ..FEM._group_elem import GroupElemFactory
from ..FEM._utils import ElemType

from ..Geoms._utils import (
    _Get_BaryCentric_Coordinates_In_Triangle,
    _Get_BaryCentric_Coordinates_In_Tetrahedron,
    _Get_BaryCentric_Coordinates_In_Segment,
)

if TYPE_CHECKING:
    from ..FEM._mesh import Mesh
    from ..FEM._group_elem import _GroupElem
    from ..Simulations._simu import _Simu


def __Get_vizir_HOSolAt_key(groupElem: "_GroupElem") -> str:
    """Returns the appropriate keyword for a given element type.

    Parameters
    ----------
    groupElem : _GroupElem
        An object representing a group element with an `elemType` attribute.

    Returns
    -------
    str
        The keyword corresponding to the specified element type.
    """

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


def _Get_BaryCentric_Coordinates(groupElem: "_GroupElem") -> _types.FloatArray:
    """Computes the barycentric coordinates for a given group element based on its type.

    Parameters
    ----------
    groupElem : _GroupElem
        An object representing a group element.

    Returns
    -------
    _types.FloatArray
        The barycentric coordinates corresponding to the specified element type.
    """

    elemType = groupElem.elemType
    local_coords = groupElem.Get_Local_Coords()
    vertices_coords = local_coords[: groupElem.Nvertex]

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


def __Get_NodesPositions(groupElem: "_GroupElem") -> _types.FloatArray:
    """Retrieves the nodes positions for a given group element based on its type.

    Parameters
    ----------
    groupElem : _GroupElem
        An object representing a group element.

    Returns
    -------
    _types.FloatArray
        The nodes positions corresponding to the specified element type.
    """

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


def _Get_empty_groupElem(groupElem: "_GroupElem", order: int):
    """Generates an empty group element based on the specified order.

    Parameters
    ----------
    groupElem : _GroupElem
        An object representing a group element.
    order : int
        The desired order for the new group element.

    Returns
    -------
    _GroupElem
        An empty group element of the specified order.
    """

    if groupElem.order != order and groupElem.elemType is not ElemType.POINT:
        # get the new elemType
        unavailable_elemTypes = [ElemType.QUAD8, ElemType.HEXA20, ElemType.PRISM15]
        filtered_dict = {
            elemType: values
            for elemType, values in GroupElemFactory.DICT_ELEMTYPE.items()
            if elemType.startswith(groupElem.topology)
            and values[3] == order
            and elemType not in unavailable_elemTypes
        }
        assert len(filtered_dict) == 1
        elemType = next(iter(filtered_dict))

        # create empty groupElem
        emptyArray = np.empty((0), dtype=int)
        groupElem = GroupElemFactory._Create(
            filtered_dict[elemType][0], emptyArray, emptyArray
        )

    else:
        emptyArray = np.empty((0), dtype=int)
        groupElem = GroupElemFactory.Create(groupElem.elemType, emptyArray, emptyArray)

    return groupElem


def __Write_HOSolAt_Element(
    file: io.TextIOWrapper, groupElem: "_GroupElem", order: int
) -> None:
    """Writes HOSolAt Element data for a given element to a file.

    Parameters
    ----------
    file : io.TextIOWrapper
        The file object where the element data will be written.
    groupElem : _GroupElem
        An object representing a group element.
    order : int
        The order of the element for which data is being written.
    """

    # set groupElem info
    keyword = __Get_vizir_HOSolAt_key(groupElem)
    file.write(f"{keyword}{groupElem.order}NodesPositions\n")

    groupElem = _Get_empty_groupElem(groupElem, order)

    # write ref geom element
    file.write(f"{groupElem.nPe}\n")
    nodesPositions = __Get_NodesPositions(groupElem)
    np.savetxt(file, nodesPositions)


def __Write_HOSolAt_Solution(
    file: io.TextIOWrapper,
    groupElem: "_GroupElem",
    dofsValues: _types.FloatArray,
    assembly_e: _types.IntArray,
    type: int,
    order: int,
) -> None:
    """Writes HOSolAt solution data for a given element to a file.

    Parameters
    ----------
    file : io.TextIOWrapper
        The file object where the solution data will be written.
    groupElem : _GroupElem
        An object representing a group element.
    dofsValues : _types.FloatArray
        Array of degree of freedom values.
    assembly_e : _types.IntArray
        Assembly information array.
    type : int
        The type of solution being written.
    order : int
        The order of the element for which data is being written.
    """

    # assembly_e informations
    Ne = assembly_e.shape[0]
    assert (
        assembly_e.ndim == 2 and Ne == groupElem.Ne
    ), "assembly_e must be a (Ne, nPe*dof_n) array"

    # get dofsValues as a (Ne, nPe*dof_n) array
    dofsValues_e = dofsValues[assembly_e]

    # write solution
    keyword = __Get_vizir_HOSolAt_key(groupElem)
    file.write(f"\n{keyword}{groupElem.order}\n{Ne}\n")
    file.write(f"1 {type}\n")
    newGroupElem = _Get_empty_groupElem(groupElem, order)
    nPe = newGroupElem.nPe
    file.write(f"{order} {nPe}\n")

    dof_n = dofsValues_e.size // Ne // nPe

    if dof_n == 2:
        # get dofsValues as a (Ne, nPe, 2) array
        dofsValues_e = dofsValues_e.reshape(Ne, nPe, dof_n)
        # reshape dofsValues_e as a (Ne, nPe, 3) array
        dofsValues_e = np.concat((dofsValues_e, np.zeros((Ne, nPe, 1))), axis=2)
        # reshape dofsValues_e as a (Ne, nPe*3) array
        dofsValues_e = dofsValues_e.reshape(Ne, -1)

    # write solution array
    np.savetxt(file, dofsValues_e)
    file.write("\n")


SOLUTION_TYPES = [1, 2]


def _Write_solution_file(
    mesh: "Mesh",
    dofsValues: _types.FloatArray,
    assembly_e: _types.IntArray,
    type: int,
    order: int,
    folder: str,
    filename: str,
    warpVector_n: Optional[_types.FloatArray] = None,
    deformFactor: float = 1.0,
) -> str:
    """Writes a solution file for a given mesh and solution data.

    Parameters
    ----------
    mesh : Mesh
        The mesh object for which the solution is being written.
    dofsValues : _types.FloatArray
        Array of degree of freedom values.
    assembly_e : _types.IntArray
        Assembly information array.
    type : int
        The type of solution being written.
    order : int
        The order of the elements for which data is being written.
    folder : str
        The directory where the solution file will be saved.
    filename : str
        The name of the solution file.
    warpVector_n : Optional[_types.FloatArray], optional
        Warp vector values for mesh deformation.
    deformFactor : float, optional
        Deformation factor for the warp vector, default is 1.0.

    Returns
    -------
    str
        The path to the created solution file.
    """

    # init solution file
    solutionFile = Folder.Join(folder, f"{filename}.sol", mkdir=True)

    if type == 1:
        dof_n = 1
    elif type == 2:
        dof_n = mesh.inDim
    else:
        raise TypeError("type error")

    if warpVector_n is not None:
        if warpVector_n.ndim == 2:

            with open(Folder.Join(Folder.Dir(folder), "default.vizir"), "w") as f:
                # write WarpVec
                f.write(f"WarpVec\n1\n{deformFactor}\n\n")

    Nn = mesh.coord.shape[0]

    with open(solutionFile, "w") as f:
        # write first lines
        f.write("MeshVersionFormatted 2\n")
        f.write("Dimension 3\n\n")  # the mesh is always in a 3d space

        if warpVector_n is not None:
            # write SolAtVertices
            assert (
                warpVector_n.shape[0] == Nn and warpVector_n.ndim == 2
            ), "nodesValues must be a (Nn, ...) array"
            assert warpVector_n.shape[1] <= 3
            nodesValues_type = 1 if warpVector_n.shape[1] == 1 else 2
            f.write(f"SolAtVertices\n{Nn}\n1 {nodesValues_type}\n")
            np.savetxt(f, warpVector_n)
            f.write("\n")

        for groupElem in mesh.Get_list_groupElem(2):
            __Write_HOSolAt_Element(f, groupElem, order)

            if assembly_e is None or assembly_e.shape[0] != groupElem.Ne:
                assembly_e = groupElem.Get_assembly_e(dof_n)

            __Write_HOSolAt_Solution(f, groupElem, dofsValues, assembly_e, type, order)

        f.write("End\n")

    return solutionFile


def Save_simu(
    simu: "_Simu",
    results: list[str],
    types: list[int],
    folder: str,
    N: Optional[int] = None,
) -> str:
    """Saves simulation results to files and prepares a command for visualization.

    Parameters
    ----------
    simu : _Simu
        The simulation object containing the results to be saved.
    results : list[str]
        A list of result names to be saved.
    types : list[int]
        A list of types corresponding to each result.
    folder : str
        The directory where the results will be saved.
    N : Optional[int], optional
        The number of iterations to sample from the simulation. If None, all iterations are used.

    Returns
    -------
    str
        A command string for visualizing the saved results using vizir.
    """

    assert isinstance(simu, "_Simu")

    # sample the results
    Niter = simu.Niter
    if N is None:
        N = Niter
    N = np.min([Niter, N])

    # init sols files and make checks
    for result, type in zip(results, types, strict=True):  # type: ignore [call-overload]
        with open(Folder.Join(folder, f"{result}.sols", mkdir=True), "w") as file:
            # do nothing
            pass
        assert type in SOLUTION_TYPES

    for i in np.linspace(0, Niter - 1, N, endpoint=True, dtype=int):
        # Update simulation iteration
        simu.Set_Iter(i)

        for result, type in zip(results, types):
            # get dofsValues
            dofsValues = simu.Result(result, nodeValues=True).ravel()
            dof_n = dofsValues.size // simu.mesh.Nn
            assembly_e = simu.mesh.Get_assembly_e(dof_n)

            # init sols file
            with open(Folder.Join(folder, f"{result}.sols"), "a") as file:
                filename = f"{result}.{i}"

                # save the solution
                solution_file = _Write_solution_file(
                    simu.mesh,
                    dofsValues,  # type: ignore [arg-type]
                    assembly_e,
                    type,
                    simu.mesh.groupElem.order,
                    folder,
                    filename,
                    simu.Results_displacement_matrix()[:, : simu.mesh.inDim],
                    100,
                )
                file.write(solution_file + "\n")

    # save the mesh in Medit format
    mesh_file = MeshIO.EasyFEA_to_Medit(simu.mesh, folder, "mesh")

    sols_files = " ".join([f"{Folder.Join(folder, result)}.sols" for result in results])
    command = f"vizir4 -in {mesh_file} -sols {sols_files}"

    return command
