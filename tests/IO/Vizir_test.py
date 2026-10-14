# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from pathlib import Path

import numpy as np
import pytest

from EasyFEA import ElemType, Simulations, IO

from Simulations._simus import LOADS


def test_Get_BaryCentric_Coordinates_rebuild_the_local_coordinates(mesh_of, subtests):
    for elemType in ElemType.Get_1D() + ElemType.Get_2D() + ElemType.Get_3D():
        if elemType.topology not in ("SEG", "TRI", "TETRA"):
            continue
        with subtests.test(elemType=str(elemType)):
            groupElem = mesh_of(elemType).groupElem
            local_coords = groupElem.Get_Local_Coords()
            vertices = local_coords[: groupElem.Nvertex]

            barycentric = IO.Vizir._Get_BaryCentric_Coordinates(groupElem)

            np.testing.assert_allclose(barycentric.sum(1), 1)
            np.testing.assert_allclose(barycentric @ vertices, local_coords, atol=1e-12)


def test_Get_BaryCentric_Coordinates_rejects_a_quadrangle(make_mesh_2D):
    groupElem = make_mesh_2D(elemType=ElemType.QUAD4).groupElem

    with pytest.raises(TypeError):
        IO.Vizir._Get_BaryCentric_Coordinates(groupElem)


def test_Get_empty_groupElem_gives_the_element_of_the_requested_order(
    mesh_of, subtests
):
    cases = [
        (ElemType.TRI3, 1, ElemType.TRI3),
        (ElemType.TRI3, 2, ElemType.TRI6),
        (ElemType.TRI6, 1, ElemType.TRI3),
        (ElemType.TRI15, 3, ElemType.TRI10),
        (ElemType.QUAD4, 2, ElemType.QUAD9),
        (ElemType.QUAD8, 2, ElemType.QUAD8),
    ]
    for elemType, order, expected in cases:
        with subtests.test(elemType=str(elemType), order=order):
            groupElem = mesh_of(elemType).groupElem

            emptyGroupElem = IO.Vizir._Get_empty_groupElem(groupElem, order)

            assert emptyGroupElem.elemType == expected
            assert emptyGroupElem.Ne == 0


def read_solution(path: str, keyword: str, nRows: int, nHeader: int = 3) -> np.ndarray:
    """The `nRows` value rows after `keyword` and its `nHeader` lines (count, type, order)."""
    lines = Path(path).read_text().splitlines()
    start = lines.index(keyword) + 1 + nHeader
    return np.loadtxt(lines[start : start + nRows], ndmin=2)


def test_Write_solution_file_writes_each_element_corner_values(make_mesh_2D, tmp_path):
    mesh = make_mesh_2D()
    ux = mesh.coord[:, 0]

    path = IO.Vizir._Write_solution_file(
        mesh,
        ux,
        mesh.Get_assembly_e(1),
        1,
        1,
        str(tmp_path),
        "ux",
    )

    assert path == str(tmp_path / "ux.sol")
    values = read_solution(path, "HOSolAtTrianglesP1", mesh.Ne)
    np.testing.assert_allclose(values, ux[mesh.connect])


def test_Write_solution_file_pads_a_2D_vector_to_3_components(make_mesh_2D, tmp_path):
    mesh = make_mesh_2D()
    u = mesh.coord[:, :2].ravel()

    path = IO.Vizir._Write_solution_file(
        mesh,
        u,
        mesh.Get_assembly_e(2),
        2,
        1,
        str(tmp_path),
        "u",
    )

    values = read_solution(path, "HOSolAtTrianglesP1", mesh.Ne).reshape(mesh.Ne, -1, 3)
    np.testing.assert_allclose(values[..., :2], mesh.coord[mesh.connect, :2])
    np.testing.assert_allclose(values[..., 2], 0)


def test_Write_solution_file_writes_the_warp_vector_and_its_factor(
    make_mesh_2D, tmp_path
):
    mesh = make_mesh_2D()
    folder = tmp_path / "sols"
    warp = mesh.coord[:, :2]

    path = IO.Vizir._Write_solution_file(
        mesh,
        mesh.coord[:, 0],
        mesh.Get_assembly_e(1),
        1,
        1,
        str(folder),
        "ux",
        warp,
        5.0,
    )

    values = read_solution(path, "SolAtVertices", mesh.Nn, 2)
    np.testing.assert_allclose(values, warp)
    assert (tmp_path / "default.vizir").read_text().split() == ["WarpVec", "1", "5.0"]


def test_Write_solution_file_rejects_an_unknown_solution_type(make_mesh_2D, tmp_path):
    mesh = make_mesh_2D()

    with pytest.raises(TypeError):
        IO.Vizir._Write_solution_file(
            mesh,
            mesh.coord[:, 0],
            mesh.Get_assembly_e(1),
            3,
            1,
            str(tmp_path),
            "ux",
        )


def test_Save_simu_lists_one_solution_file_per_iteration(
    bar_simu: Simulations.Elastic, tmp_path
):
    command = IO.Vizir.Save_simu(
        bar_simu, str(tmp_path), results=["ux", "displacement"], types=[1, 2]
    )

    meshFile = tmp_path / "mesh.mesh"
    sols = [tmp_path / "ux.sols", tmp_path / "displacement.sols"]
    assert command == f"vizir4 -in {meshFile} -sols {sols[0]} {sols[1]}"
    assert meshFile.exists()
    for result, solsFile in zip(["ux", "displacement"], sols):
        assert solsFile.read_text().split() == [
            str(tmp_path / f"{result}.{i}.sol") for i in range(len(LOADS))
        ]


def test_Save_simu_samples_at_most_N_iterations(
    bar_simu: Simulations.Elastic, tmp_path
):
    IO.Vizir.Save_simu(bar_simu, str(tmp_path), N=2, results=["ux"], types=[1])

    assert (tmp_path / "ux.sols").read_text().split() == [
        str(tmp_path / f"ux.{i}.sol") for i in (0, len(LOADS) - 1)
    ]


def test_Save_simu_needs_a_type_per_result(bar_simu: Simulations.Elastic, tmp_path):
    with pytest.raises(ValueError, match="types"):
        IO.Vizir.Save_simu(bar_simu, str(tmp_path), results=["ux"])
