# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Checks the mesh partitioner without mpirun: `Mesher._Mesh_Get_Meshes` builds every rank's piece in one process."""

import numpy as np
import pytest

from EasyFEA import Mesher, ElemType, Mesh
from EasyFEA.Geoms import Domain, Point

NPROCS = [2, 3, 5]


def Get_partitions(Nproc: int, meshSize: float = 1.0) -> list[Mesh]:
    """Meshes a domain and returns its `Nproc` partitions."""

    mesher = Mesher()
    domain = Domain(Point(), Point(10, 10), meshSize)

    mesher._Init_gmsh("occ")
    mesher._Surfaces(domain, [])
    mesher._Set_PhysicalGroups()
    mesher._Mesh_Generate(2, ElemType.TRI3)

    return mesher._Mesh_Get_Meshes(Nproc)


def Get_partitioned_data(meshes: list[Mesh]) -> list[tuple]:
    """Returns (rank, elements, ghostElements, nodes, ghostNodes) of the main group of each partition."""
    return [mesh.groupElem._Get_partitioned_data() for mesh in meshes]


@pytest.fixture(scope="module")
def partitions() -> dict[int, list[Mesh]]:
    # Nproc == 1 is the unpartitioned mesh, used as the reference to split against
    return {Nproc: Get_partitions(Nproc) for Nproc in [1] + NPROCS}


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
def test_partition_is_reproducible(partitions, Nproc: int):
    """The same input gives the same split twice."""

    data = Get_partitioned_data(partitions[Nproc])
    again = Get_partitioned_data(Get_partitions(Nproc))

    for (_, elements, ghosts, nodes, _), (_, elements2, ghosts2, nodes2, _) in zip(
        data, again
    ):
        assert np.array_equal(np.sort(elements), np.sort(elements2))
        assert np.array_equal(np.sort(ghosts), np.sort(ghosts2))
        assert np.array_equal(np.sort(nodes), np.sort(nodes2))
