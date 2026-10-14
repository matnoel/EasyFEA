# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import os

import numpy as np
import pytest
from pxr import Usd, UsdGeom

from EasyFEA import ElemType, Simulations, IO

from Simulations._simus import LOADS
from ._meshes import frames_of


def points_at(stage: Usd.Stage, primPath: str, time) -> np.ndarray:
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(primPath))
    return np.asarray(mesh.GetPointsAttr().Get(time))


def test_get_triangles_splits_every_face_into_its_triangles(make_mesh_3D):
    mesh = make_mesh_3D(elemType=ElemType.HEXA8)

    triangles = IO.USD._get_triangles(mesh)

    nTriangles = sum(
        groupElem.Ne * np.size(groupElem.triangles) // 3
        for groupElem in mesh.Get_list_groupElem(2)
    )
    assert triangles.shape == (nTriangles, 3)
    area = 0.5 * np.linalg.norm(
        np.cross(
            mesh.coord[triangles[:, 1]] - mesh.coord[triangles[:, 0]],
            mesh.coord[triangles[:, 2]] - mesh.coord[triangles[:, 0]],
        ),
        axis=1,
    )
    np.testing.assert_allclose(
        area.sum(), sum(g.area for g in mesh.Get_list_groupElem(2))
    )


def test_get_lines_gives_the_element_edges_as_segments(make_mesh_2D, subtests):
    for elemType in (ElemType.TRI3, ElemType.TRI6):
        with subtests.test(elemType=str(elemType)):
            mesh = make_mesh_2D(elemType=elemType)

            lines = IO.USD._get_lines(mesh)

            length = np.linalg.norm(
                mesh.coord[lines[:, 1]] - mesh.coord[lines[:, 0]], axis=1
            ).sum()
            # each element walks its own edges
            corners = mesh.coord[mesh.connect[:, :3]][:, [0, 1, 2, 0]]
            perimeter = np.linalg.norm(np.diff(corners, axis=1), axis=2).sum()
            np.testing.assert_allclose(length, perimeter)


def test_Save_mesh_writes_the_mesh_nodes(mesh_of, tmp_path, subtests):
    for elemType in ElemType.Get_2D() + ElemType.Get_3D():
        mesh = mesh_of(elemType)
        for plotMesh in (False, True):
            with subtests.test(elemType=str(elemType), plotMesh=plotMesh):
                folder = tmp_path / f"{elemType}_{plotMesh}"

                path = IO.USD.Save_mesh(mesh, str(folder), "mesh", plotMesh=plotMesh)

                assert path == str(folder / "mesh.usdz")
                stage = Usd.Stage.Open(path)
                points = points_at(stage, "/Mesh", Usd.TimeCode.Default())
                np.testing.assert_allclose(points, mesh.coord, rtol=1e-6)
                assert stage.GetPrimAtPath("/Lines").IsValid() == plotMesh


def test_Save_mesh_animates_the_displaced_nodes(make_mesh_2D, tmp_path, subtests):
    mesh = make_mesh_2D()
    frames = frames_of(mesh)

    for plotMesh in (False, True):
        with subtests.test(plotMesh=plotMesh):
            folder = tmp_path / str(plotMesh)

            path = IO.USD.Save_mesh(
                mesh,
                str(folder),
                "mesh",
                list_displacementMatrix=frames,
                list_nodesValues_n=frames,
                plotMesh=plotMesh,
            )

            stage = Usd.Stage.Open(path)
            assert stage.GetEndTimeCode() == len(frames) - 1
            for i, frame in enumerate(frames):
                np.testing.assert_allclose(
                    points_at(stage, "/Mesh", i), mesh.coord + frame, rtol=1e-6
                )
            assert (folder / "colorbar_mesh.png").exists()


def test_Save_mesh_without_smoothAnimation_writes_one_frame_prim_per_step(
    make_mesh_2D, tmp_path, subtests
):
    mesh = make_mesh_2D()
    frames = frames_of(mesh)

    for plotMesh in (False, True):
        with subtests.test(plotMesh=plotMesh):
            path = IO.USD.Save_mesh(
                mesh,
                str(tmp_path / str(plotMesh)),
                "mesh",
                list_displacementMatrix=frames,
                smoothAnimation=False,
                plotMesh=plotMesh,
            )

            stage = Usd.Stage.Open(path)
            for i, frame in enumerate(frames):
                np.testing.assert_allclose(
                    points_at(stage, f"/Frame_{i:03d}/Mesh", Usd.TimeCode.Default()),
                    mesh.coord + frame,
                    rtol=1e-6,
                )


def test_Save_mesh_takes_one_mesh_per_frame(make_mesh_2D, tmp_path, subtests):
    meshes = [make_mesh_2D(meshSize=meshSize) for meshSize in (1 / 2, 1 / 3)]
    values = [mesh.coord[:, 0] for mesh in meshes]

    for smoothAnimation in (True, False):
        with subtests.test(smoothAnimation=smoothAnimation):
            path = IO.USD.Save_mesh(
                meshes,
                str(tmp_path / str(smoothAnimation)),
                "mesh",
                list_displacementMatrix=[np.zeros((mesh.Nn, 3)) for mesh in meshes],
                list_nodesValues_n=values,
                plotMesh=True,
                smoothAnimation=smoothAnimation,
            )

            stage = Usd.Stage.Open(path)
            for i, mesh in enumerate(meshes):
                if smoothAnimation:
                    points = points_at(stage, "/Mesh", i)
                else:
                    time = Usd.TimeCode.Default()
                    points = points_at(stage, f"/Frame_{i:03d}/Mesh", time)
                np.testing.assert_allclose(points, mesh.coord, rtol=1e-6)


def test_Save_mesh_is_binary_only(make_mesh_2D, tmp_path):
    mesh = make_mesh_2D()

    with pytest.raises(ValueError):
        IO.USD.Save_mesh(mesh, str(tmp_path), "text", useBinary=False)


def test_Save_simu_writes_one_usdz_per_component_and_the_norm(
    bar_simus: list[Simulations.Elastic], tmp_path, subtests
):
    for simu in bar_simus:
        with subtests.test(dim=simu.mesh.dim):
            folder = tmp_path / str(simu.mesh.dim)

            saved = IO.USD.Save_simu(simu, str(folder), results=["uy", "displacement"])

            assert saved == str(folder)
            components = [f"displacement_{d}" for d in range(simu.mesh.dim)]
            usdzFiles = sorted(f for f in os.listdir(folder) if f.endswith(".usdz"))
            assert usdzFiles == sorted(
                f"{name}.usdz" for name in ["uy", *components, "displacement_norm"]
            )
            stage = Usd.Stage.Open(str(folder / "uy.usdz"))
            assert stage.GetEndTimeCode() == len(LOADS) - 1
            for i in range(len(LOADS)):
                simu.Set_Iter(i)
                expected = simu.mesh.coord + simu.Results_displacement_matrix()
                np.testing.assert_allclose(
                    points_at(stage, "/Mesh", i), expected, rtol=1e-6, atol=1e-9
                )


def test_Save_simu_needs_a_simulation(make_mesh_2D, tmp_path):
    mesh = make_mesh_2D()

    assert IO.USD.Save_simu(mesh, str(tmp_path), results=["uy"]) is None  # type: ignore [arg-type]
