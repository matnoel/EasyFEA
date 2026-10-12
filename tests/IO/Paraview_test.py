# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from EasyFEA import ElemType, Models, Simulations, IO

L = 2
H = 1
LOADS = (10.0, 20.0, 30.0)


@pytest.fixture
def simu(make_mesh_2D) -> Simulations.Elastic:
    """A 2D bar in tension, solved and saved once per load."""
    mesh = make_mesh_2D(L=L, H=H, elemType=ElemType.QUAD4, isOrganised=True)
    material = Models.Elastic.Isotropic(dim=2, E=210000, v=0.3)
    simu = Simulations.Elastic(mesh, material)

    nodesLeft = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesRight = mesh.Nodes_Conditions(lambda x, y, z: x == L)
    for load in LOADS:
        simu.Bc_Init()
        simu.add_dirichlet(nodesLeft, [0, 0], ["x", "y"])
        simu.add_surfLoad(nodesRight, [load], ["x"])
        simu.Solve()
        simu.Save_Iter()

    return simu


def read_steps(pvdFile: str) -> list[pv.UnstructuredGrid]:
    """Every .vtu of the .pvd in timestep order, via VTK (meshio misreads raw appended data)."""
    folder = Path(pvdFile).parent
    dataSets = ET.parse(pvdFile).getroot().iter("DataSet")
    return [pv.read(folder / dataSet.attrib["file"]) for dataSet in dataSets]


def test_requested_element_field_is_written(simu: Simulations.Elastic, tmp_path):
    pvdFile = IO.Paraview.Save_simu(simu, str(tmp_path), elementFields=["Exx"])

    vtu = read_steps(pvdFile)[-1]

    assert "Exx" in vtu.cell_data


def test_node_and_element_fields_round_trip_every_iteration(
    simu: Simulations.Elastic, tmp_path
):
    pvdFile = IO.Paraview.Save_simu(
        simu, str(tmp_path), results=["ux"], elementFields=["Exx"]
    )

    steps = read_steps(pvdFile)

    assert len(steps) == len(LOADS)
    for i, vtu in enumerate(steps):
        simu.Set_Iter(i)
        np.testing.assert_allclose(vtu.point_data["ux"], simu.Result("ux"))
        np.testing.assert_allclose(
            vtu.cell_data["Exx"], simu.Result("Exx", nodeValues=False)
        )


def test_default_fields_are_kept_alongside_requested_ones(
    simu: Simulations.Elastic, tmp_path
):
    pvdFile = IO.Paraview.Save_simu(
        simu, str(tmp_path), results=["ux"], elementFields=["Exx"]
    )

    vtu = read_steps(pvdFile)[-1]

    assert set(vtu.point_data.keys()) == {"displacement", "ux"}
    assert set(vtu.cell_data.keys()) == {"Svm", "Stress", "Exx"}
