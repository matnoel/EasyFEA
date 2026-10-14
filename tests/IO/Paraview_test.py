# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from EasyFEA import Simulations, IO

from Simulations._simus import LOADS


def read_steps(pvdFile: str) -> list[pv.UnstructuredGrid]:
    """Every .vtu of the .pvd in timestep order, via VTK (meshio misreads raw appended data)."""
    folder = Path(pvdFile).parent
    dataSets = ET.parse(pvdFile).getroot().iter("DataSet")
    return [pv.read(folder / dataSet.attrib["file"]) for dataSet in dataSets]


def test_Save_simu_round_trips_node_and_element_fields_every_iteration(
    bar_simu: Simulations.Elastic, tmp_path
):
    pvdFile = IO.Paraview.Save_simu(
        bar_simu, str(tmp_path), results=["ux"], elementFields=["Exx"]
    )

    steps = read_steps(pvdFile)

    assert pvdFile == str(tmp_path / "Paraview" / "simulation.pvd")
    assert len(steps) == len(LOADS)
    for i, vtu in enumerate(steps):
        bar_simu.Set_Iter(i)
        np.testing.assert_allclose(vtu.point_data["ux"], bar_simu.Result("ux"))
        np.testing.assert_allclose(
            vtu.cell_data["Exx"], bar_simu.Result("Exx", nodeValues=False)
        )


def test_Save_simu_keeps_the_default_fields_alongside_requested_ones(
    bar_simu: Simulations.Elastic, tmp_path
):
    pvdFile = IO.Paraview.Save_simu(
        bar_simu, str(tmp_path), results=["ux"], elementFields=["Exx"]
    )

    vtu = read_steps(pvdFile)[-1]

    assert set(vtu.point_data.keys()) == {"displacement", "ux"}
    assert set(vtu.cell_data.keys()) == {"Svm", "Stress", "Exx"}


def test_Save_simu_samples_at_most_N_iterations_first_and_last_included(
    bar_simu: Simulations.Elastic, tmp_path
):
    pvdFile = IO.Paraview.Save_simu(bar_simu, str(tmp_path), N=2, results=["ux"])

    steps = read_steps(pvdFile)

    assert len(steps) == 2
    for vtu, i in zip(steps, [0, len(LOADS) - 1]):
        bar_simu.Set_Iter(i)
        np.testing.assert_allclose(vtu.point_data["ux"], bar_simu.Result("ux"))


def test_Save_simu_raises_naming_an_unavailable_field(
    bar_simu: Simulations.Elastic, tmp_path
):
    with pytest.raises(ValueError, match="notAField"):
        IO.Paraview.Save_simu(bar_simu, str(tmp_path), results=["notAField"])
    with pytest.raises(ValueError, match="notAField"):
        IO.Paraview.Save_simu(bar_simu, str(tmp_path), elementFields=["notAField"])


def test_Save_mesh_writes_one_step_per_given_field_value(make_mesh_2D, tmp_path):
    mesh = make_mesh_2D()
    nodeValues = [mesh.coord[:, 0] * i for i in range(3)]
    elementValues = [np.arange(mesh.Ne) * float(i) for i in range(3)]

    IO.Paraview._Save_mesh(
        mesh,
        str(tmp_path),
        3,
        {"x": nodeValues},
        {"e": elementValues},
    )

    steps = read_steps(str(tmp_path / "Paraview" / "simulation.pvd"))
    assert len(steps) == 3
    for vtu, values_n, values_e in zip(steps, nodeValues, elementValues):
        np.testing.assert_allclose(vtu.points, mesh.coord)
        np.testing.assert_allclose(vtu.point_data["x"], values_n)
        np.testing.assert_allclose(vtu.cell_data["e"], values_e)


def test_CalcOffset_adds_the_header_and_the_data_bytes():
    assert IO.Paraview.CalcOffset(0, 3, 4) == 16
    assert IO.Paraview.CalcOffset(10, 3, 8) == 38
