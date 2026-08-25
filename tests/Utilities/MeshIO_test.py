# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import Callable

import numpy as np
import pytest

from EasyFEA import Folder, ElemType, Mesh, MeshIO
from EasyFEA.Geoms import Line

folder_results = Folder.Results_Dir()

L = 2
H = 1


@pytest.fixture
def meshes(
    make_mesh_2D: Callable[..., Mesh], make_mesh_3D: Callable[..., Mesh]
) -> list[Mesh]:

    meshSize = H / 3
    meshes: list[Mesh] = []

    # 1d meshes
    line = Line((0, 0), (1, 0), meshSize)
    for elemType in ElemType.Get_1D():
        meshes.append(line.Mesh_1D(elemType))

    # 2d meshes
    for elemType in ElemType.Get_2D():
        meshes.append(make_mesh_2D(elemType=elemType, isOrganised=True))

    # 3d meshes
    for elemType in ElemType.Get_3D():
        meshes.append(make_mesh_3D(elemType=elemType, B=L, isOrganised=True))

    return meshes


def check_mesh(mesh1: Mesh, mesh2: Mesh):

    assert mesh1.Ne == mesh2.Ne
    assert mesh1.Nn == mesh2.Nn

    if mesh1.dim >= 1:
        diff_length = mesh1.length - mesh2.length
        assert diff_length < 1e-12

    if mesh1.dim >= 2:
        diff_area = mesh1.area - mesh2.area
        assert diff_area < 1e-12

    if mesh1.dim >= 3:
        diff_volume = mesh1.volume - mesh2.volume
        assert diff_volume < 1e-12


def get_tags(mesh: Mesh) -> list[str]:
    return sorted(
        {
            tag
            for groupElem in mesh.dict_groupElem.values()
            for tag in groupElem.nodeTags
        }
    )


def check_tags(mesh1: Mesh, mesh2: Mesh):
    """Every tag must come back, holding the very same nodes and elements.

    Nodes_Tags unions the node sets over every group and Elements_Tags reads the main group, so
    those are what a user sees; which group holds a node tag is an internal detail.
    """

    tags = get_tags(mesh1)
    assert tags == get_tags(mesh2), "tag names differ"

    for tag in tags:
        assert np.array_equal(
            np.sort(mesh1.Nodes_Tags(tag)), np.sort(mesh2.Nodes_Tags(tag))
        ), f"nodes of {tag} differ"

    for tag in mesh1.groupElem.elementTags:
        assert np.array_equal(
            np.sort(mesh1.Elements_Tags(tag)), np.sort(mesh2.Elements_Tags(tag))
        ), f"elements of {tag} differ"


class TestMeshIO:

    def test_mesh_reconstruction(self, meshes: list[Mesh]):

        for mesh in meshes:

            newMesh = MeshIO.Surface_reconstruction(mesh)

            # check surface reconstruction

            for groupElem1, groupElem2 in zip(
                mesh.Get_list_groupElem(2), newMesh.Get_list_groupElem(2)
            ):

                assert groupElem1.Ne == groupElem2.Ne

            check_mesh(mesh, newMesh)

    def test_easyfea_to_meshio(self, meshes: list[Mesh]):

        for mesh in meshes:

            meshio = MeshIO._EasyFEA_to_Meshio(mesh)
            newMesh = MeshIO._Meshio_to_EasyFEA(meshio)

            check_mesh(mesh, newMesh)
            # named sets carry the tags, so this pair loses nothing
            check_tags(mesh, newMesh)

    def test_easyfea_to_meshio_keeps_a_named_tag(self, meshes: list[Mesh]):
        """A tag that is not P{i}, L{i}, S{i} or V{i} used to raise, its digits being stripped."""

        for mesh in meshes:

            nodes = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
            for groupElem in mesh.dict_groupElem.values():
                groupElem.Set_Tag(nodes, "endocardium")

            newMesh = MeshIO._Meshio_to_EasyFEA(MeshIO._EasyFEA_to_Meshio(mesh))

            assert "endocardium" in get_tags(newMesh)
            check_tags(mesh, newMesh)

    def test_easyfea_to_gmsh(self, meshes: list[Mesh]):

        for mesh in meshes:

            filename = MeshIO.EasyFEA_to_Gmsh(mesh, folder_results, mesh.elemType.name)
            newMesh = MeshIO.Gmsh_to_EasyFEA(filename)

            check_mesh(mesh, newMesh)
            # the tags Mesher creates claim disjoint elements, so every one of them fits in the
            # single reference gmsh stores per element
            check_tags(mesh, newMesh)

    def test_easyfea_to_gmsh_keeps_a_named_tag(self, meshes: list[Mesh]):
        """$PhysicalNames carries the name, so it comes back instead of a S{ref} placeholder."""

        for mesh in meshes:

            groupElems = mesh.Get_list_groupElem(max(mesh.dim - 1, 0))
            nodes = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
            for groupElem in groupElems:
                groupElem.Set_Tag(nodes, "endocardium")

            filename = MeshIO.EasyFEA_to_Gmsh(mesh, folder_results, mesh.elemType.name)
            newMesh = MeshIO.Gmsh_to_EasyFEA(filename)

            assert "endocardium" in get_tags(newMesh)

    def test_a_mesh_without_tags_gains_none(self):
        """Writers store 0 for an element belonging to no group, which is not a tag."""

        import meshio

        meshioMesh = meshio.Mesh(
            np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
            [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))],
        )
        mesh = MeshIO._Meshio_to_EasyFEA(meshioMesh)
        assert get_tags(mesh) == []

        filename = MeshIO.EasyFEA_to_Medit(mesh, folder_results, "tagless")
        assert get_tags(MeshIO.Medit_to_EasyFEA(filename)) == []

    def test_easyfea_to_medit(self, meshes: list[Mesh]):

        for mesh in meshes:

            if mesh.elemType in ["QUAD8", "HEXA20", "PRISM15"]:
                continue

            filename = MeshIO.EasyFEA_to_Medit(mesh, folder_results, mesh.elemType.name)
            newMesh = MeshIO.Medit_to_EasyFEA(filename)

            check_mesh(mesh, newMesh)

    def test_easyfea_to_pyvista(self, meshes: list[Mesh]):

        for mesh in meshes:

            if mesh.groupElem.order >= 2:
                continue

            pyVistaMesh = MeshIO.EasyFEA_to_PyVista(mesh)
            newMesh = MeshIO.PyVista_to_EasyFEA(pyVistaMesh)

            check_mesh(mesh, newMesh)

    def test_easyfea_to_ensight(self, meshes: list[Mesh]):

        for mesh in meshes:

            if mesh.elemType not in MeshIO.DICT_ELEMTYPE_TO_ENSIGHT:
                continue

            ensightMesh = MeshIO.EasyFEA_to_Ensight(
                mesh, folder_results, mesh.elemType.name
            )
            newMesh = MeshIO.Ensight_to_EasyFEA(ensightMesh)

            check_mesh(mesh, newMesh)
