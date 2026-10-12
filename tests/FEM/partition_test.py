# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Checks the mesh partitioner without mpirun: `Mesher._Mesh_Get_Meshes` builds every rank's piece in one process."""

import numpy as np
import pytest

from EasyFEA import ElemType, Mesher, Mesh
from EasyFEA.Geoms import Domain, Line, Point

NPROCS = [2, 3, 5]
SIZE = 10.0


def Get_partitions(elemType: ElemType, Nproc: int) -> list[Mesh]:
    """`Nproc` pieces of a SIZE-long line, SIZE-wide square or slab of `elemType`, built on the private Mesher steps."""
    mesher = Mesher()
    mesher._Init_gmsh("occ")

    if elemType in ElemType.Get_1D():
        dim, meshSize = 1, 1.0
        line = Line(Point(), Point(SIZE, 0), meshSize)
        p1 = mesher._factory.addPoint(*line.pt1.coord, meshSize=meshSize)
        p2 = mesher._factory.addPoint(*line.pt2.coord, meshSize=meshSize)
        mesher._Create_Lines(line, p1, p2)
    else:
        dim = 2 if elemType in ElemType.Get_2D() else 3
        meshSize = 1.0 if dim == 2 else 2.0
        contour = Domain(Point(), Point(SIZE, SIZE), meshSize)
        mesher._Surfaces(contour, [])
        mesher._Organise_Surfaces(elemType, True, meshSize)
        if dim == 3:
            surfaces = [entity[1] for entity in mesher._factory.getEntities(2)]
            mesher._Extrude(surfaces, (0, 0, 2 * meshSize), elemType, [2])

    mesher._Set_PhysicalGroups()
    mesher._Mesh_Generate(dim, elemType)
    return mesher._Mesh_Get_Meshes(Nproc)


def Get_partitioned_data(meshes: list[Mesh]) -> list[tuple]:
    """Returns (rank, elements, ghostElements, nodes, ghostNodes) of the main group of each partition."""
    return [mesh.groupElem._Get_partitioned_data() for mesh in meshes]


@pytest.fixture(
    scope="module",
    params=ElemType.Get_1D() + ElemType.Get_2D() + ElemType.Get_3D(),
    ids=str,
)
def partitionedType(request) -> ElemType:
    return request.param


@pytest.fixture(scope="module")
def partitions(partitionedType: ElemType) -> dict[int, list[Mesh]]:
    # Nproc == 1 is the unpartitioned mesh, used as the reference to split against
    return {Nproc: Get_partitions(partitionedType, Nproc) for Nproc in [1] + NPROCS}


@pytest.mark.parametrize("Nproc", NPROCS)
def test_owned_elements_partition_the_mesh(partitions, Nproc: int):
    """Every element of the global mesh is owned by exactly one rank."""

    data = Get_partitioned_data(partitions[Nproc])
    owned = [set(elements.tolist()) for _, elements, _, _, _ in data]

    for rank, elements in enumerate(owned):
        for other in range(rank + 1, Nproc):
            assert elements.isdisjoint(
                owned[other]
            ), f"ranks {rank} and {other} share elements"

    assert set().union(*owned) == set(range(partitions[1][0].groupElem.Ne))


@pytest.mark.parametrize("Nproc", NPROCS)
def test_owned_nodes_partition_the_mesh(partitions, Nproc: int):
    """Every node of the global mesh is owned by exactly one rank."""

    data = Get_partitioned_data(partitions[Nproc])
    owned = [set(nodes.tolist()) for _, _, _, nodes, _ in data]

    for rank, nodes in enumerate(owned):
        for other in range(rank + 1, Nproc):
            assert nodes.isdisjoint(
                owned[other]
            ), f"ranks {rank} and {other} share nodes"

    assert set().union(*owned) == set(range(partitions[1][0].Nn))


@pytest.mark.parametrize("Nproc", NPROCS)
def test_ghost_elements_are_owned_elsewhere(partitions, Nproc: int):
    """A rank's ghost elements are owned by some other rank."""

    data = Get_partitioned_data(partitions[Nproc])
    owned = [set(elements.tolist()) for _, elements, _, _, _ in data]

    for rank, (_, _, ghostElements, _, _) in enumerate(data):
        others = set().union(*(owned[r] for r in range(Nproc) if r != rank))
        assert set(ghostElements.tolist()).issubset(others)


@pytest.mark.parametrize("Nproc", NPROCS)
def test_connectivity_is_owned_plus_ghost(partitions, Nproc: int):
    """Each partition holds exactly its owned and ghost elements."""

    meshes = partitions[Nproc]
    data = Get_partitioned_data(meshes)

    for mesh, (_, elements, ghostElements, _, _) in zip(meshes, data):
        assert mesh.groupElem.Ne == elements.size + ghostElements.size


@pytest.mark.parametrize("Nproc", NPROCS)
def test_no_empty_partition(partitions, Nproc: int):
    """No rank is left without elements as long as there are enough of them."""

    data = Get_partitioned_data(partitions[Nproc])
    assert Nproc <= partitions[1][0].groupElem.Ne

    for rank, (_, elements, _, _, _) in enumerate(data):
        assert elements.size > 0, f"rank {rank} owns no element"


@pytest.mark.parametrize("Nproc", NPROCS)
def test_partition_is_reproducible(partitionedType: ElemType, partitions, Nproc: int):
    """The same input gives the same split twice."""

    data = Get_partitioned_data(partitions[Nproc])
    again = Get_partitioned_data(Get_partitions(partitionedType, Nproc))

    for (_, elements, ghosts, nodes, _), (_, elements2, ghosts2, nodes2, _) in zip(
        data, again
    ):
        assert np.array_equal(np.sort(elements), np.sort(elements2))
        assert np.array_equal(np.sort(ghosts), np.sort(ghosts2))
        assert np.array_equal(np.sort(nodes), np.sort(nodes2))
