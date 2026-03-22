# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""This module allows you to save a simulation's results on Paraview (https://www.paraview.org/)."""

import numpy as np
from typing import TYPE_CHECKING

# utilities
from . import Display, Folder, Tic
from .MeshIO import DICT_GMSH_TO_VTK_INDEXES, DICT_ELEMTYPE_TO_VTK
from ._mpi import MPI_SIZE, MPI_RANK, MPI_COMM

from ..Utilities import _types

if TYPE_CHECKING:
    from ..Simulations._simu import _Simu
    from ..FEM._mesh import Mesh


# ----------------------------------------------
# Paraview
# ----------------------------------------------
def Save_simu(
    simu: "_Simu",
    folder: str,
    N: int = 200,
    details: bool = False,
    nodeFields: list[str] = [],
    elementFields: list[str] = [],
):
    """Generates the paraview (.pvd and .vtu/.pvtu files) with a simu.

    Parameters
    ----------
    simu : _Simu
        Simulation
    folder: str
        folder in which we will create the Paraview folder
    N : int, optional
        Maximal number of iterations displayed, by default 200
    details: bool, optional
        details of nodesField and elementsField used in the .vtu
    nodeFields: list, optional
        Additional nodeFields, by default []
    elementFields: list, optional
        Additional elementFields, by default []
    """
    print("\n")

    simu = Display._Init_obj(simu)[0]  # type: ignore
    meshDim = simu.mesh.dim
    Ne = simu.mesh.Ne

    results = simu.results

    Niter = len(results)
    N = np.min([Niter, N])
    iterations = np.linspace(0, Niter - 1, N, endpoint=True, dtype=int)

    folder = Folder.Join(folder, "Paraview")

    if MPI_RANK == 0 and not Folder.Exists(folder):
        Folder.os.makedirs(folder)

    additionalNodesField = nodeFields
    additionalElementsField = elementFields

    nodeFields, elementFields = simu.Results_nodeFields_elementFields(details)

    [
        nodeFields.append(n)  # type: ignore [func-returns-value]
        for n in additionalNodesField
        if simu._Results_Check_Available(n) and n not in nodeFields
    ]
    [
        elementFields.append(e)  # type: ignore [func-returns-value]
        for e in additionalElementsField
        if simu._Results_Check_Available(e) and e not in elementFields
    ]

    if len(nodeFields) == 0 and len(elementFields) == 0:
        Display.MyPrintError(
            "The simulation has no solution fields to display in paraview."
        )

    # activate the first iteration
    simu.Set_Iter(0, resetAll=True)

    pvFiles: list[str] = []
    times = []
    tic = Tic()

    for i, iter in enumerate(iterations):
        simu.Set_Iter(iter)

        # get nodeResults
        nodeResults: dict[str, _types.AnyArray] = {}
        for nodeField in nodeFields:
            array = simu.Result(nodeField, True)
            nodeField = nodeField.removesuffix("_matrix")
            nodeResults[nodeField] = array

        # get elementResults
        elementResults: dict[str, _types.AnyArray] = {}
        for elementField in elementFields:
            array = simu.Result(elementField, False)
            if meshDim == 3 and array.size / Ne == 6:
                # reorder (xx, yy, zz, yz, xz, xy)
                # to      (xx, yy, zz, xy, yz, xz)
                array = array.reshape(Ne, -1)[:, [0, 1, 2, 5, 3, 4]]
            elementResults[elementField] = array

        pvFiles.append(__Make_vtu(simu.mesh, folder, iter, nodeResults, elementResults))

        times.append(tic.Tac("Paraview", "Make vtu", False))

        iteration = i + 1
        rmTime = Tic.Get_Remaining_Time(iteration, N, times[-1])

        iteration = str(iteration).zfill(len(str(N)))
        Display.MyPrint(f"Generate paraview {iteration}/{N} {rmTime}     ", end="\r")

    print("\n")

    tic = Tic()

    if MPI_RANK == 0:
        __Make_pvd(Folder.os.path.join(folder, "simulation"), pvFiles)

    tic.Tac("Paraview", "Make pvd", False)


def _Save_mesh(
    mesh: "Mesh",
    folder: str,
    N: int,
    nodeFields: dict[str, list[_types.AnyArray]] = {},
    elementFields: dict[str, list[_types.AnyArray]] = {},
):
    """Generates the paraview (.pvd and .vtu/.pvtu files) with a mesh.

    Parameters
    ----------
    mesh : Mesh
        mesh
    folder: str
        folder in which we will create the Paraview folder
    N : int
        number of iterations
    nodeFields: dict[str, list[_types.AnyArray]], optional
        Additional nodeFields, by default {}
    elementFields: dict[str, list[_types.AnyArray]], optional
        Additional elementFields, by default {}
    """
    print("\n")

    folder = Folder.Join(folder, "Paraview")

    if MPI_RANK == 0 and not Folder.Exists(folder):
        Folder.os.makedirs(folder)

    pvFiles: list[str] = []
    times = []
    tic = Tic()

    for i in range(N):
        nodeResults = {
            nodeField: results[i] for nodeField, results in nodeFields.items()
        }
        elementResults = {
            elementField: results[i] for elementField, results in elementFields.items()
        }

        pvFiles.append(__Make_vtu(mesh, folder, i, nodeResults, elementResults))

        times.append(tic.Tac("Paraview", "Make vtu", False))

        iteration = i + 1
        rmTime = Tic.Get_Remaining_Time(iteration, N, times[-1])

        iteration = str(iteration).zfill(len(str(N)))
        Display.MyPrint(f"Generate paraview {iteration}/{N} {rmTime}     ", end="\r")

    print("\n")

    tic = Tic()

    if MPI_RANK == 0:
        __Make_pvd(Folder.os.path.join(folder, "simulation"), pvFiles)

    tic.Tac("Paraview", "Make pvd", False)


# ----------------------------------------------
# Functions
# ----------------------------------------------
def __Make_vtu(
    mesh: "Mesh",
    folder: str,
    iter: int,
    nodeResults: dict[str, _types.AnyArray],
    elementResults: dict[str, _types.AnyArray],
) -> str:
    """Generates the .vtu file for this rank and, when MPI_SIZE > 1, the .pvtu
    descriptor (rank 0 only). Returns the path to reference in the .pvd."""

    rank_folder = Folder.Rank_Dir(folder)

    if not Folder.Exists(rank_folder):
        Folder.os.makedirs(rank_folder)

    filename = Folder.Join(rank_folder, f"solution_{iter}.vtu")

    # get mesh data
    elemType = mesh.elemType
    Ne = mesh.Ne
    Nn = mesh.Nn
    nPe = mesh.groupElem.nPe
    inDim = mesh.inDim

    # reorder gmsh idx to vtk indexes
    if elemType in DICT_GMSH_TO_VTK_INDEXES:
        vtkIndexes = DICT_GMSH_TO_VTK_INDEXES[elemType]
    else:
        vtkIndexes = np.arange(nPe).tolist()
    connect = mesh.connect[:, vtkIndexes]

    paraviewType = DICT_ELEMTYPE_TO_VTK[elemType].value

    types = np.ones(Ne, dtype=int) * paraviewType

    # coordinates as a vector (e.g (x1, y1, z1,..., xn, yn, zn))
    nodes = mesh.coord.ravel()
    # connect as a vector (e.g (n1^1, n2^1, n3^1, ..., n1^e, n2^e, n3^e))
    connect = connect.ravel()

    connect_offsets = np.arange(nPe, nPe * Ne + 1, nPe, dtype=np.int32)

    endian_paraview = "LittleEndian"  # 'LittleEndian' 'BigEndian'

    bitSize = 4  # bit size

    def CalcOffset(offset, size):
        return offset + bitSize + (bitSize * size)

    with open(filename, "w") as file:
        file.write('<?pickle version="1.0" ?>\n')
        file.write(
            f'<VTKFile type="UnstructuredGrid" version="0.1" byte_order="{endian_paraview}">\n'
        )
        file.write("\t<UnstructuredGrid>\n")
        file.write(f'\t\t<Piece NumberOfPoints="{Nn}" NumberOfCells="{Ne}">\n')

        # Specify the nodes values
        file.write('\t\t\t<PointData scalars="scalar"> \n')
        offset = 0
        list_values_n: list[_types.FloatArray] = []
        nodeFields_meta: dict[str, int] = {}
        for nodeField, nodeValues in nodeResults.items():
            assert isinstance(
                nodeValues, np.ndarray
            ), "nodeValues must be a numpy array."

            dof_n = nodeValues.size // Nn

            if dof_n == 2 and inDim == 2:
                # add new array for z values
                # otherwise we won't be able to plot the deformed mesh
                nodeValues = np.concatenate(
                    (nodeValues.reshape(Nn, 2), np.zeros((Nn, 1))), axis=1
                )
                dof_n = 3

            list_values_n.append(nodeValues.ravel())
            nodeFields_meta[nodeField] = dof_n

            file.write(
                f'\t\t\t\t<DataArray type="Float32" Name="{nodeField}" NumberOfComponents="{dof_n}" format="appended" offset="{offset}" />\n'
            )
            offset = CalcOffset(offset, nodeValues.size)

        file.write("\t\t\t</PointData> \n")

        # Specify the elements values
        file.write("\t\t\t<CellData> \n")
        list_values_e: list[_types.FloatArray] = []
        elementFields_meta: dict[str, int] = {}
        for elementField, elementValues in elementResults.items():
            assert isinstance(
                elementValues, np.ndarray
            ), "elementValues must be a numpy array."

            list_values_e.append(elementValues.ravel())

            dof_n = elementValues.size // Ne
            elementFields_meta[elementField] = dof_n

            file.write(
                f'\t\t\t\t<DataArray type="Float32" Name="{elementField}" NumberOfComponents="{dof_n}" format="appended" offset="{offset}" />\n'
            )
            offset = CalcOffset(offset, elementValues.size)

        file.write("\t\t\t</CellData> \n")

        # Points / Nodes coordinates
        file.write("\t\t\t<Points>\n")
        # NumberOfComponents must be "3"
        file.write(
            f'\t\t\t\t<DataArray type="Float32" NumberOfComponents="3" format="appended" offset="{offset}" />\n'
        )
        offset = CalcOffset(offset, nodes.size)
        file.write("\t\t\t</Points>\n")

        # Elements -> Connectivity matrix
        file.write("\t\t\t<Cells>\n")
        file.write(
            f'\t\t\t\t<DataArray type="Int32" Name="connectivity" format="appended" offset="{offset}" />\n'
        )
        offset = CalcOffset(offset, connect.size)
        file.write(
            f'\t\t\t\t<DataArray type="Int32" Name="offsets" format="appended" offset="{offset}" />\n'
        )
        offset = CalcOffset(offset, connect_offsets.size)
        file.write(
            f'\t\t\t\t<DataArray type="Int8" Name="types" format="appended" offset="{offset}" />\n'
        )
        file.write("\t\t\t</Cells>\n")

        file.write("\t\t</Piece>\n")
        file.write("\t</UnstructuredGrid> \n")

        # Adding values
        file.write('\t<AppendedData encoding="raw"> \n_')

    # Add all values in binary
    with open(filename, "ab") as file:
        # Nodes values
        for nodeValues in list_values_n:
            __WriteBinary(bitSize * nodeValues.size, "uint32", file)
            __WriteBinary(nodeValues, "float32", file)

        # Elements values
        for elementValues in list_values_e:
            __WriteBinary(bitSize * elementValues.size, "uint32", file)
            __WriteBinary(elementValues, "float32", file)

        # Nodes
        __WriteBinary(bitSize * nodes.size, "uint32", file)
        __WriteBinary(nodes, "float32", file)

        # Connectivity
        __WriteBinary(bitSize * connect.size, "uint32", file)
        __WriteBinary(connect, "int32", file)

        # Offsets
        __WriteBinary(bitSize * Ne, "uint32", file)
        __WriteBinary(connect_offsets, "int32", file)

        # Element types
        __WriteBinary(types.size, "uint32", file)
        __WriteBinary(types, "int8", file)

    with open(filename, "a") as file:
        file.write("\n\t</AppendedData>\n")
        file.write("</VTKFile> \n")

    if MPI_SIZE > 1:
        MPI_COMM.Barrier()  # wait for all ranks to finish writing their .vtu

        if MPI_RANK == 0:
            piece_files = [
                Folder.Join(folder, f"Rank{r}", f"solution_{iter}.vtu")
                for r in range(MPI_SIZE)
            ]
            return __Make_pvtu(
                Folder.Join(folder, f"solution_{iter}"),
                piece_files,
                nodeFields_meta,
                elementFields_meta,
            )

    return filename


def __Make_pvd(filename: str, pvFiles: list[str] = []):
    """Makes .pvd file to link the .vtu or .pvtu files."""

    tic = Tic()

    endian_paraview = "LittleEndian"  # 'LittleEndian' 'BigEndian'

    filename = filename + ".pvd"

    dir = Folder.Dir(filename)

    with open(filename, "w") as file:
        file.write('<?pickle version="1.0" ?>\n')
        file.write(
            f'<VTKFile type="Collection" version="0.1" byte_order="{endian_paraview}">\n'
        )
        file.write("\t<Collection>\n")

        for t, pvFile in enumerate(pvFiles):
            pvFile = pvFile.replace(dir, ".")
            file.write(
                f'\t\t<DataSet timestep="{t}" group="" part="1" file="{pvFile}"/>\n'
            )

        file.write("\t</Collection>\n")
        file.write("</VTKFile>\n")

    tic.Tac("Paraview", "Make pvd", False)


def __Make_pvtu(
    filename: str,
    piece_files: list[str],
    nodeFields_meta: dict[str, int],
    elementFields_meta: dict[str, int],
) -> str:
    """Generates a .pvtu parallel descriptor file referencing per-rank .vtu pieces."""

    endian_paraview = "LittleEndian"
    pvtu_filename = filename + ".pvtu"
    dir = Folder.Dir(pvtu_filename)

    with open(pvtu_filename, "w") as file:
        file.write('<?xml version="1.0" ?>\n')
        file.write(
            f'<VTKFile type="PUnstructuredGrid" version="0.1" byte_order="{endian_paraview}">\n'
        )
        file.write('\t<PUnstructuredGrid GhostLevel="0">\n')

        file.write("\t\t<PPointData>\n")
        for name, n_comp in nodeFields_meta.items():
            file.write(
                f'\t\t\t<PDataArray type="Float32" Name="{name}" NumberOfComponents="{n_comp}"/>\n'
            )
        file.write("\t\t</PPointData>\n")

        file.write("\t\t<PCellData>\n")
        for name, n_comp in elementFields_meta.items():
            file.write(
                f'\t\t\t<PDataArray type="Float32" Name="{name}" NumberOfComponents="{n_comp}"/>\n'
            )
        file.write("\t\t</PCellData>\n")

        file.write("\t\t<PPoints>\n")
        file.write('\t\t\t<PDataArray type="Float32" NumberOfComponents="3"/>\n')
        file.write("\t\t</PPoints>\n")

        for piece_file in piece_files:
            rel = piece_file.replace(dir, ".")
            file.write(f'\t\t<Piece Source="{rel}"/>\n')

        file.write("\t</PUnstructuredGrid>\n")
        file.write("</VTKFile>\n")

    return pvtu_filename


def __WriteBinary(value, type: str, file):
    """Converts value (int or array) to binary."""

    if type not in ["uint32", "float32", "int32", "int8"]:
        raise Exception("Type not implemented")

    if type == "uint32":
        value = np.uint32(value)
    elif type == "float32":
        value = np.float32(value)
    elif type == "int32":
        value = np.int32(value)
    elif type == "int8":
        value = np.int8(value)

    file.write(value.tobytes())
