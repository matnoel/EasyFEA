# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import Folder, Mesher, ElemType, Mesh, MeshIO
from EasyFEA.Geoms import Points

folder_results = Folder.Results_Dir()

L = 2
H = 1


@pytest.fixture
def meshes() -> list[Mesh]:

    meshSize = H / 3

    contour = Points([(0, 0), (L, 0), (L, H), (0, H)], meshSize)
    meshes: list[Mesh] = []

    # 1d meshes
    for elemType in ElemType.Get_1D():

        mesher = Mesher()
        factory = mesher._factory

        p1 = factory.addPoint(0, 0, 0)
        p2 = factory.addPoint(1, 0, 0)
        factory.addLine(p1, p2)

        mesher._Set_PhysicalGroups()
        mesher._Mesh_Generate(1, elemType)

        meshes.append(mesher._Mesh_Get_Mesh())

    # 2d meshes
    for elemType in ElemType.Get_2D():
        mesh = Mesher().Mesh_2D(contour, [], elemType, isOrganised=True)
        meshes.append(mesh)

    # 3d meshes
    for elemType in ElemType.Get_3D():
        mesh = Mesher().Mesh_Extrude(
            contour, [], [0, 0, L], [3], elemType, isOrganised=True
        )
        meshes.append(mesh)

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

    def test_easyfea_to_gmsh(self, meshes: list[Mesh]):

        for mesh in meshes:

            filename = MeshIO.EasyFEA_to_Gmsh(mesh, folder_results, mesh.elemType.name)
            newMesh = MeshIO.Gmsh_to_EasyFEA(filename)

            check_mesh(mesh, newMesh)

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
