# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest
import numpy as np
from scipy.spatial import cKDTree

from EasyFEA.FEM._utils import MatrixType
from EasyFEA import Mesher, ElemType, Mesh, Models, Simulations
from EasyFEA.Geoms import Points

L = 2
H = 1


def equal(val1, val2, tol=1e-11):
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

        mesh = contour.Mesh_2D([], elemType, isOrganised=True)

        meshes_2D.append(mesh)

    meshes_2D = __move_meshes(meshes_2D)

    return meshes_2D


@pytest.fixture
def meshes_3D() -> list[Mesh]:

    meshSize = H / 3

    contour = Points([(0, 0), (L, 0), (L, H), (0, H)], meshSize)

    meshes_3D: list[Mesh] = []

    for elemType in ElemType.Get_3D():

        mesh = contour.Mesh_Extrude([], [0, 0, L], [3], elemType, isOrganised=True)

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
            assembly_e = mesh.Get_assembly_e(dim)
            testAssembly = np.testing.assert_array_almost_equal(
                assembly_e, assembly_e_test, verbose=False
            )
            assert testAssembly is None

            # Verify lines_e
            lines_e_test = np.array(
                [[i for i in assembly_e[e] for j in assembly_e[e]] for e in elements]
            )
            testLines = np.testing.assert_array_almost_equal(
                lines_e_test, mesh.Get_rows_e(mesh.dim), verbose=False
            )
            assert testLines is None

            # Verify lines_e
            colonnes_e_test = np.array(
                [[j for i in assembly_e[e] for j in assembly_e[e]] for e in elements]
            )
            testColumns = np.testing.assert_array_almost_equal(
                colonnes_e_test, mesh.Get_columns_e(dim), verbose=False
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

        mat = Models.Elastic.Isotropic(3, 210000 * 1e6, 0.33)
        rho = 7850  # kg/m3

        volume = L * H * L
        mass = volume * rho  # kg
        F = mass * 9.81  # N
        P = 50

        for mesh in meshes_3D:

            simu = Simulations.Elastic(mesh, mat)
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


def _no_duplicate_coords(mesh: Mesh, atol: float = 1e-12) -> bool:
    """Returns True when no two nodes share the same position."""
    return len(cKDTree(mesh.coord).query_pairs(atol)) == 0


class TestMeshMerge:

    @pytest.mark.parametrize("elemType", ElemType.Get_2D())
    def test_adjacent_2d_area_and_nodes(self, elemType):
        """Two adjacent unit squares: area = 2, shared edge nodes merged, no duplicates."""
        h = 1 / 3
        left = Points([(0, 0), (1, 0), (1, 1), (0, 1)], h).Mesh_2D([], elemType, isOrganised=True)
        right = Points([(1, 0), (2, 0), (2, 1), (1, 1)], h).Mesh_2D([], elemType, isOrganised=True)

        merged = Mesh.Merge([left, right])

        assert abs(merged.area - 2.0) / 2.0 < 1e-10
        assert merged.Nn < left.Nn + right.Nn
        assert _no_duplicate_coords(merged)

    @pytest.mark.parametrize("elemType", ElemType.Get_3D())
    def test_adjacent_3d_volume_and_nodes(self, elemType):
        """Two adjacent unit cubes: volume = 2, shared face nodes merged, no duplicates."""
        h = 0.5
        left = Points([(0, 0), (1, 0), (1, 1), (0, 1)], h).Mesh_Extrude([], [0, 0, 1], [2], elemType, isOrganised=True)
        right = Points([(1, 0), (2, 0), (2, 1), (1, 1)], h).Mesh_Extrude([], [0, 0, 1], [2], elemType, isOrganised=True)

        merged = Mesh.Merge([left, right])

        assert abs(merged.volume - 2.0) / 2.0 < 1e-10
        assert merged.Nn < left.Nn + right.Nn
        assert _no_duplicate_coords(merged)

    def test_single_mesh_returns_itself(self):
        """Merge([mesh]) is the identity — no copy, no processing."""
        mesh = Points([(0, 0), (1, 0), (1, 1), (0, 1)], 0.4).Mesh_2D([], ElemType.QUAD4, isOrganised=True)
        assert Mesh.Merge([mesh]) is mesh

    def test_construct_unique_elements_removes_duplicates(self):
        """Merging a mesh with itself: constructUniqueElements=True keeps one copy."""
        mesh = Points([(0, 0), (1, 0), (1, 1), (0, 1)], 0.4).Mesh_2D([], ElemType.QUAD4, isOrganised=True)
        merged = Mesh.Merge([mesh, mesh], constructUniqueElements=True)
        assert merged.Ne == mesh.Ne
        assert merged.Nn == mesh.Nn

    def test_no_deduplication_keeps_all_elements(self):
        """constructUniqueElements=False preserves every element from every mesh."""
        mesh = Points([(0, 0), (1, 0), (1, 1), (0, 1)], 0.4).Mesh_2D([], ElemType.QUAD4, isOrganised=True)
        merged = Mesh.Merge([mesh, mesh], constructUniqueElements=False)
        assert merged.Ne == 2 * mesh.Ne

    def test_tolerance_merges_near_coincident_nodes(self):
        """Nodes offset by < mergePointsTol are merged; nodes beyond it are not."""
        mesh = Points([(0, 0), (1, 0), (1, 1), (0, 1)], 0.4).Mesh_2D([], ElemType.QUAD4, isOrganised=True)

        delta_inside = 1e-13   # within default 1e-12 → should merge
        delta_outside = 1e-11  # beyond default 1e-12 → should NOT merge

        def perturbed(d):
            m = mesh.copy()
            m.coord = mesh.coord + d
            return m

        merged_in = Mesh.Merge([mesh, perturbed(delta_inside)])
        assert merged_in.Nn == mesh.Nn  # all nodes collapsed

        merged_out = Mesh.Merge([mesh, perturbed(delta_outside)])
        assert merged_out.Nn == 2 * mesh.Nn  # no nodes merged

    def test_merge_points_false_skips_deduplication(self):
        """mergePoints=False concatenates coordinates without any KDTree search."""
        h = 1 / 3
        left = Points([(0, 0), (1, 0), (1, 1), (0, 1)], h).Mesh_2D([], ElemType.QUAD4, isOrganised=True)
        right = Points([(1, 0), (2, 0), (2, 1), (1, 1)], h).Mesh_2D([], ElemType.QUAD4, isOrganised=True)

        merged = Mesh.Merge([left, right], mergePoints=False)

        assert merged.Nn == left.Nn + right.Nn  # no node merging
        assert abs(merged.area - 2.0) / 2.0 < 1e-10  # geometry still correct

    def test_return_mapping_identity(self):
        """return_mapping=True on a single mesh returns the identity mapping."""
        mesh = Points([(0, 0), (1, 0), (1, 1), (0, 1)], 0.4).Mesh_2D([], ElemType.QUAD4, isOrganised=True)
        result, mapping = Mesh.Merge([mesh], return_mapping=True)

        assert result is mesh
        assert len(mapping) == 1
        np.testing.assert_array_equal(mapping[0], np.arange(mesh.Nn))

    def test_return_mapping_two_meshes(self):
        """mapping[i][j] is the index of mesh i's node j in the merged mesh."""
        h = 1 / 3
        left = Points([(0, 0), (1, 0), (1, 1), (0, 1)], h).Mesh_2D([], ElemType.QUAD4, isOrganised=True)
        right = Points([(1, 0), (2, 0), (2, 1), (1, 1)], h).Mesh_2D([], ElemType.QUAD4, isOrganised=True)

        merged, mapping = Mesh.Merge([left, right], return_mapping=True)

        assert len(mapping) == 2

        # Every node of 'left' must land at the correct coordinate in merged
        for j in range(left.Nn):
            np.testing.assert_allclose(
                merged.coord[mapping[0][j]], left.coord[j], atol=1e-12
            )

        # Every node of 'right' must land at the correct coordinate in merged
        for j in range(right.Nn):
            np.testing.assert_allclose(
                merged.coord[mapping[1][j]], right.coord[j], atol=1e-12
            )

        # Shared edge nodes from left and right must map to the same merged node
        shared_left = np.where(left.coord[:, 0] == 1.0)[0]
        shared_right = np.where(right.coord[:, 0] == 1.0)[0]
        # build a dict: merged_node → coord for each side, they must overlap
        merged_from_left = set(mapping[0][shared_left].tolist())
        merged_from_right = set(mapping[1][shared_right].tolist())
        assert len(merged_from_left & merged_from_right) > 0  # at least some shared
