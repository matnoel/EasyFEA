# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import gmsh
import numpy as np

from EasyFEA import ElemType, Simulations, IO

from Simulations._simus import LOADS
from ._meshes import check_mesh, check_tags, get_tags


def test_Save_mesh_round_trips_through_Load_mesh_with_its_tags(
    mesh_of, tmp_path, subtests
):
    for elemType in ElemType.Get_1D() + ElemType.Get_2D() + ElemType.Get_3D():
        with subtests.test(elemType=str(elemType)):
            mesh = mesh_of(elemType)

            path = IO.Gmsh.Save_mesh(mesh, str(tmp_path), "mesh")
            newMesh = IO.Gmsh.Load_mesh(path)

            assert path.endswith(".msh")
            check_mesh(mesh, newMesh)
            # Mesher's tags claim disjoint elements, so each fits in the one reference gmsh stores
            check_tags(mesh, newMesh)


def test_Save_mesh_keeps_a_named_tag(mesh_of, tmp_path, subtests):
    for elemType in ElemType.Get_1D() + ElemType.Get_2D() + ElemType.Get_3D():
        with subtests.test(elemType=str(elemType)):
            mesh = mesh_of(elemType).copy()  # session-scoped mesh
            nodes = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
            for groupElem in mesh.Get_list_groupElem(max(mesh.dim - 1, 0)):
                groupElem.Set_Tag(nodes, "endocardium")

            path = IO.Gmsh.Save_mesh(mesh, str(tmp_path), "mesh")

            newMesh = IO.Gmsh.Load_mesh(path)
            assert "endocardium" in get_tags(newMesh)
            tagged = np.sort(newMesh.Nodes_Tags("endocardium"))
            assert np.array_equal(tagged, np.sort(nodes))


def test_Save_mesh_binary_round_trips(make_mesh_2D, tmp_path):
    mesh = make_mesh_2D()

    path = IO.Gmsh.Save_mesh(mesh, str(tmp_path), "mesh", useBinary=True)

    check_mesh(mesh, IO.Gmsh.Load_mesh(path))


def test_Load_mesh_returns_None_for_a_file_without_elements(tmp_path):
    path = tmp_path / "empty.msh"
    path.write_text(
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n1\n1 0 0 0\n$EndNodes\n"
        "$Elements\n0\n$EndElements\n"
    )

    assert IO.Gmsh.Load_mesh(str(path)) is None


def read_views(path: str) -> dict[str, tuple[int, np.ndarray]]:
    """{view name: (time steps, list data of the view's single element block)}."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.open(path)
    views = {}
    for tag in gmsh.view.getTags():
        index = gmsh.view.getIndex(tag)
        name = gmsh.option.getString(f"View[{index}].Name")
        nSteps = int(gmsh.option.getNumber(f"View[{index}].NbTimeStep"))
        _, _, data = gmsh.view.getListData(tag)
        views[name] = (nSteps, np.asarray(data[0]))
    gmsh.finalize()
    return views


def test_Save_simu_writes_one_view_per_component_and_default_field(
    bar_simu: Simulations.Elastic, tmp_path
):
    path = IO.Gmsh.Save_simu(bar_simu, str(tmp_path), results=["ux"])

    assert path == str(tmp_path / "simu.pos")
    views = read_views(path)
    assert set(views) == {
        "ux",
        "displacement_0",
        "displacement_1",
        "Svm",
        "Stress_0",
        "Stress_1",
        "Stress_2",
    }
    assert {nSteps for nSteps, _ in views.values()} == {len(LOADS)}


def test_Save_simu_stores_the_corner_values_of_every_iteration(
    bar_simu: Simulations.Elastic, tmp_path
):
    mesh = bar_simu.mesh
    nCorners = mesh.groupElem.Nvertex

    path = IO.Gmsh.Save_simu(bar_simu, str(tmp_path), results=["ux"])

    _, data = read_views(path)["ux"]
    # per element: x, y, z of each corner, then the corner values of each step
    data = data.reshape(mesh.Ne, -1)[:, 3 * nCorners :].reshape(mesh.Ne, -1, nCorners)
    corners = mesh.connect[:, :nCorners]
    for i in range(len(LOADS)):
        bar_simu.Set_Iter(i)
        np.testing.assert_allclose(data[:, i], bar_simu.Result("ux")[corners])


def test_Save_simu_samples_at_most_N_iterations(
    bar_simu: Simulations.Elastic, tmp_path
):
    path = IO.Gmsh.Save_simu(bar_simu, str(tmp_path), N=2, results=["ux"])

    nSteps, _ = read_views(path)["ux"]
    assert nSteps == 2
