# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA.fem._utils import MatrixType
from EasyFEA import Mesher, ElemType, Mesh, Models, np, Simulations
from EasyFEA.Geoms import Points

L = 2
H = 1


def equal(val1, val2, tol=1e-12):
    assert np.abs(val1 - val2) / np.abs(val2) < tol


def __move_meshes(list_mesh: list[Mesh]):

    for mesh in list_mesh.copy():

        meshRot90x = mesh.copy()
        meshRot90x.Rotate(90, direction=(1, 0, 0))
        meshRot90y = mesh.copy()
        meshRot90y.Rotate(90, direction=(0, 1, 0))
        meshRot90z = mesh.copy()
        meshRot90z.Rotate(90, direction=(0, 0, 1))

        mesh.Rotate(45, direction=(1, 0, 0))
        mesh.Rotate(45, direction=(0, 1, 0))
        mesh.Translate(-1)

        list_mesh.extend([meshRot90x, meshRot90y, meshRot90z, mesh])

    return list_mesh


@pytest.fixture
def meshes_2D() -> list[Mesh]:

    meshSize = H / 3

    contour = Points([(0, 0), (L, 0), (L, H), (0, H)], meshSize)

    meshes_2D: list[Mesh] = []

    for elemType in ElemType.Get_2D():

        mesh = Mesher().Mesh_2D(contour, [], elemType, isOrganised=True)

        meshes_2D.append(mesh)

    meshes_2D = __move_meshes(meshes_2D)

    return meshes_2D


@pytest.fixture
def meshes_3D() -> list[Mesh]:

    meshSize = H / 3

    contour = Points([(0, 0), (L, 0), (L, H), (0, H)], meshSize)

    meshes_3D: list[Mesh] = []

    for elemType in ElemType.Get_3D():

        mesh = Mesher().Mesh_Extrude(
            contour, [], [0, 0, L], [3], elemType, isOrganised=True
        )

        meshes_3D.append(mesh)

    meshes_3D = __move_meshes(meshes_3D)

    return meshes_3D


class TestMesh:

    def test_construct_matrix(self):

        meshes = Mesher._Construct_2D_meshes()
        meshes.extend(Mesher._Construct_3D_meshes()[:2])

        matrixTypes = [MatrixType.rigi, MatrixType.mass]

        for mesh in meshes:

            dim = mesh.dim
            connect = mesh.connect
            elements = range(mesh.Ne)

            # Verify assembly
            assembly_e_test = np.array(
                [
                    [int(n * dim + d) for n in connect[e] for d in range(dim)]
                    for e in elements
                ]
            )
            testAssembly = np.testing.assert_array_almost_equal(
                mesh.assembly_e, assembly_e_test, verbose=False
            )
            assert testAssembly is None

            # Verify lines_e
            lines_e_test = np.array(
                [
                    [i for i in mesh.assembly_e[e] for j in mesh.assembly_e[e]]
                    for e in elements
                ]
            )
            testLines = np.testing.assert_array_almost_equal(
                lines_e_test, mesh.rowsVector_e, verbose=False
            )
            assert testLines is None

            # Verify lines_e
            colonnes_e_test = np.array(
                [
                    [j for i in mesh.assembly_e[e] for j in mesh.assembly_e[e]]
                    for e in elements
                ]
            )
            testColumns = np.testing.assert_array_almost_equal(
                colonnes_e_test, mesh.columnsVector_e, verbose=False
            )
            assert testColumns is None

            for matrixType in matrixTypes:
                mesh.Get_jacobian_e_pg(matrixType)
                mesh.Get_dN_e_pg(matrixType)
                mesh.Get_ddN_e_pg(matrixType)
                mesh.Get_B_e_pg(matrixType)

    def test_area(self, meshes_2D: list[Mesh]):

        area = L * H

        for mesh in meshes_2D:

            assert (area - mesh.area) / area < 1e-10

    def test_volume(self, meshes_3D: list[Mesh]):

        volume = L * H * L

        for mesh in meshes_3D:

            assert (volume - mesh.volume) / volume < 1e-12

    def test_load(self, meshes_3D: list[Mesh]):

        mat = Models.ElasIsot(3, 210000 * 1e6, 0.33)
        rho = 7850  # kg/m3

        volume = L * H * L
        mass = volume * rho  # kg
        F = mass * 9.81  # N
        P = 50

        for mesh in meshes_3D:

            simu = Simulations.ElasticSimu(mesh, mat)
            simu.rho = rho

            assert (mass - simu.mass) / mass < 1e-12

            simu.add_volumeLoad(mesh.nodes, [-rho * 9.81], ["z"])

            rhs = simu._Solver_Apply_Neumann(simu.problemType)

            assert (F + rhs.sum()) / F < 1e-12

            groupSurf = mesh.Get_list_groupElem(2)[-1]
            elems = groupSurf.Get_Elements_Tag("S0")
            nodes = groupSurf.Get_Nodes_Tag("S0")
            area = groupSurf.area_e[elems].sum()

            simu.Bc_Init()
            simu.add_surfLoad(nodes, [P / area], ["z"])

            rhs = simu._Solver_Apply_Neumann(simu.problemType)

            assert (P / area - rhs.sum()) / (P / area) < 1e-12

            simu.Bc_Init()
            simu.add_pressureLoad(nodes, P / area)

            rhs = simu._Solver_Apply_Neumann(simu.problemType).toarray()
            load = np.linalg.norm(rhs.reshape(-1, 3), axis=1)

            assert (P / area - load.sum()) / (P / area) < 1e-12

    def test_Evaluate_dofsValues_at_coordinates_2D(self, meshes_2D: list[Mesh]):

        for mesh in meshes_2D:

            node = np.random.randint(0, mesh.Nn, 1)

            coords = mesh.coord[node].reshape(1, 3)

            dofsValues = np.arange(mesh.Nn * 2) + 1

            values = mesh.Evaluate_dofsValues_at_coordinates(coords, dofsValues)
            equal(dofsValues.reshape(-1, 2)[node, 0], values[0, 0])
            equal(dofsValues.reshape(-1, 2)[node, 1], values[0, 1])

    def test_Evaluate_dofsValues_at_coordinates_3D(self, meshes_3D: list[Mesh]):

        for mesh in meshes_3D:

            node = np.random.randint(0, mesh.Nn, 1)

            coords = mesh.coord[node].reshape(1, 3)

            dofsValues = np.arange(mesh.Nn * 2) + 1

            values = mesh.Evaluate_dofsValues_at_coordinates(coords, dofsValues)
            equal(dofsValues.reshape(-1, 2)[node, 0], values[0, 0])
            equal(dofsValues.reshape(-1, 2)[node, 1], values[0, 1])
