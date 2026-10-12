# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.


import numpy as np
import pytest

from EasyFEA import Folder, Mesh, IO

from _numerics import rel_error

folder_results = Folder.Results_Dir()

L = 2
H = 1


def check_mesh(mesh1: Mesh, mesh2: Mesh):

    assert mesh1.Ne == mesh2.Ne
    assert mesh1.Nn == mesh2.Nn

    if mesh1.dim >= 1:
        assert rel_error(mesh1.length, mesh2.length) < 1e-12

    if mesh1.dim >= 2:
        assert rel_error(mesh1.area, mesh2.area) < 1e-12

    if mesh1.dim >= 3:
        assert rel_error(mesh1.volume, mesh2.volume) < 1e-12


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


class TestMeshFormats:

    def test_mesh_reconstruction(self, all_elemType_meshes: list[Mesh]):

        for mesh in all_elemType_meshes:

            newMesh = IO.PyVista.Surface_reconstruction(mesh)

            # check surface reconstruction

            for groupElem1, groupElem2 in zip(
                mesh.Get_list_groupElem(2), newMesh.Get_list_groupElem(2)
            ):

                assert groupElem1.Ne == groupElem2.Ne

            check_mesh(mesh, newMesh)

    def test_easyfea_to_meshio(self, all_elemType_meshes: list[Mesh]):

        for mesh in all_elemType_meshes:

            meshio = IO._meshio._EasyFEA_to_Meshio(mesh)
            newMesh = IO._meshio._Meshio_to_EasyFEA(meshio)

            check_mesh(mesh, newMesh)
            # named sets carry the tags, so this pair loses nothing
            check_tags(mesh, newMesh)

    def test_easyfea_to_meshio_keeps_a_named_tag(self, all_elemType_meshes: list[Mesh]):
        """A tag that is not P{i}, L{i}, S{i} or V{i} used to raise, its digits being stripped."""

        for mesh in all_elemType_meshes:

            mesh = mesh.copy()  # the meshes are shared across the session

            nodes = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
            for groupElem in mesh.dict_groupElem.values():
                groupElem.Set_Tag(nodes, "endocardium")

            newMesh = IO._meshio._Meshio_to_EasyFEA(IO._meshio._EasyFEA_to_Meshio(mesh))

            assert "endocardium" in get_tags(newMesh)
            check_tags(mesh, newMesh)

    def test_easyfea_to_gmsh(self, all_elemType_meshes: list[Mesh]):

        for mesh in all_elemType_meshes:

            filename = IO.Gmsh.Save_mesh(mesh, folder_results, mesh.elemType.name)
            newMesh = IO.Gmsh.Load_mesh(filename)

            check_mesh(mesh, newMesh)
            # the tags Mesher creates claim disjoint elements, so every one of them fits in the
            # single reference gmsh stores per element
            check_tags(mesh, newMesh)

    def test_easyfea_to_gmsh_keeps_a_named_tag(self, all_elemType_meshes: list[Mesh]):
        """$PhysicalNames carries the name, so it comes back instead of a S{ref} placeholder."""

        for mesh in all_elemType_meshes:

            mesh = mesh.copy()  # the meshes are shared across the session

            groupElems = mesh.Get_list_groupElem(max(mesh.dim - 1, 0))
            nodes = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
            for groupElem in groupElems:
                groupElem.Set_Tag(nodes, "endocardium")

            filename = IO.Gmsh.Save_mesh(mesh, folder_results, mesh.elemType.name)
            newMesh = IO.Gmsh.Load_mesh(filename)

            assert "endocardium" in get_tags(newMesh)

    def test_a_mesh_without_tags_gains_none(self):
        """Writers store 0 for an element belonging to no group, which is not a tag."""

        import meshio

        meshioMesh = meshio.Mesh(
            np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
            [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))],
        )
        mesh = IO._meshio._Meshio_to_EasyFEA(meshioMesh)
        assert get_tags(mesh) == []

        filename = IO.Medit.Save_mesh(mesh, folder_results, "tagless")
        assert get_tags(IO.Medit.Load_mesh(filename)) == []

    def test_imported_tags_keep_their_elements(self):
        """The middle triangle's nodes all belong to its neighbours' tag, which must not take it."""

        import meshio

        #  0---2---4
        #   \ / \ /      refs: (0,1,2) -> 1, (1,3,2) -> 2, (2,3,4) -> 1
        #    1---3
        points = np.array([[0, 0], [1, -1], [2, 0], [3, -1], [4, 0]], dtype=float)
        triangles = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]])
        refs = np.array([1, 2, 1])
        meshioMesh = meshio.Mesh(
            points, [("triangle", triangles)], cell_data={"medit:ref": [refs]}
        )

        mesh = IO._meshio._Meshio_to_EasyFEA(meshioMesh)

        assert np.array_equal(np.sort(mesh.Elements_Tags("S1")), [0, 2])
        assert np.array_equal(mesh.Elements_Tags("S2"), [1])
        # a tag given by nodes alone still takes every element using only those nodes
        mesh.groupElem.Set_Tag(mesh.Nodes_Tags("S1"), "fromNodes")
        assert np.array_equal(np.sort(mesh.Elements_Tags("fromNodes")), [0, 1, 2])
        with pytest.raises(Exception):
            mesh.groupElem.Set_Tag(mesh.Nodes_Tags("S1"), "bad", np.array([3]))

    def test_easyfea_to_medit(self, all_elemType_meshes: list[Mesh]):

        for mesh in all_elemType_meshes:

            if mesh.elemType in ["QUAD8", "HEXA20", "PRISM15"]:
                continue

            filename = IO.Medit.Save_mesh(mesh, folder_results, mesh.elemType.name)
            newMesh = IO.Medit.Load_mesh(filename)

            check_mesh(mesh, newMesh)

    def test_easyfea_to_pyvista(self, all_elemType_meshes: list[Mesh]):

        for mesh in all_elemType_meshes:

            if mesh.groupElem.order >= 2:
                continue

            pyVistaMesh = IO.PyVista.EasyFEA_to_PyVista(mesh)
            newMesh = IO.PyVista.PyVista_to_EasyFEA(pyVistaMesh)

            check_mesh(mesh, newMesh)

    def test_easyfea_to_ensight(self, all_elemType_meshes: list[Mesh]):

        for mesh in all_elemType_meshes:

            if mesh.elemType not in IO.Ensight.DICT_ELEMTYPE_TO_ENSIGHT:
                continue

            ensightMesh = IO.Ensight.Save_mesh(mesh, folder_results, mesh.elemType.name)
            newMesh = IO.Ensight.Load_mesh(ensightMesh)

            check_mesh(mesh, newMesh)


def test_gmsh_save_simu_writes_pos(tmp_path):
    from EasyFEA import Models, Simulations
    from EasyFEA.Geoms import Domain

    contour = Domain((0, 0), (1, 1), 0.5)
    mesh = contour.Mesh_2D()
    simu = Simulations.Elastic(mesh, Models.Elastic.Isotropic(2))
    nodesLeft = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesRight = mesh.Nodes_Conditions(lambda x, y, z: x == 1)
    simu.add_dirichlet(nodesLeft, [0, 0], ["x", "y"])
    simu.add_dirichlet(nodesRight, [0.1], ["x"])
    simu.Solve()
    simu.Save_Iter()

    path = IO.Gmsh.Save_simu(simu, str(tmp_path), results=["ux"])

    assert path == str(tmp_path / "simu.pos")
    assert (tmp_path / "simu.pos").exists()


def test_ensight_save_mesh_is_text_only(tmp_path):
    from EasyFEA.Geoms import Domain

    contour = Domain((0, 0), (1, 1), 0.5)
    mesh = contour.Mesh_2D()

    with pytest.raises(ValueError):
        IO.Ensight.Save_mesh(mesh, str(tmp_path), "mesh", useBinary=True)


def test_medit_save_mesh_binary(tmp_path):
    from EasyFEA.Geoms import Domain

    contour = Domain((0, 0), (1, 1), 0.5)
    mesh = contour.Mesh_2D()

    path = IO.Medit.Save_mesh(mesh, str(tmp_path), "mesh", useBinary=True)

    assert path.endswith(".meshb")
