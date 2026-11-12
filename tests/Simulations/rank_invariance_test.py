# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Under `make test-mpi`, a simulation solved over the ranks matches the same one solved serially by a child process run with `EASYFEA_MPI=off`."""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from EasyFEA import ElemType, Models, Simulations
from EasyFEA.Geoms import Circle, Domain


def _plate_with_hole(elemType: ElemType):
    """Unit-height plate with an off-centre hole: no symmetry for a partition to hide behind."""
    domain = Domain((0, 0), (2, 1), 0.1)
    circle = Circle((0.7, 0.45), 0.3, 0.1)
    return domain.Mesh_2D([circle], elemType)


def _clamped_and_pulled(simu, pull: list[float], unknowns: list[str]):
    mesh = simu.mesh
    nodesClamped = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesPulled = mesh.Nodes_Conditions(lambda x, y, z: x == 2)
    simu.add_dirichlet(nodesClamped, [0, 0], ["x", "y"])
    simu.add_dirichlet(nodesPulled, pull, unknowns)


def _elastic() -> Simulations.Elastic:
    simu = Simulations.Elastic(
        _plate_with_hole(ElemType.TRI6), Models.Elastic.Isotropic(2, E=210e3, v=0.3)
    )
    _clamped_and_pulled(simu, [1e-3, -2e-3], ["x", "y"])
    simu.Solve()
    return simu


def _newmark() -> Simulations.Elastic:
    simu = Simulations.Elastic(
        _plate_with_hole(ElemType.QUAD4), Models.Elastic.Isotropic(2, E=210e3, v=0.3)
    )
    simu.rho = 7.8e-9
    simu.Solver_Set_Hyperbolic_Algorithm(1e-6)
    for step in range(1, 4):
        simu.Bc_Init()
        _clamped_and_pulled(simu, [1e-3 * step], ["y"])
        simu.Solve()
    return simu


def _newton() -> Simulations.HyperElastic:
    simu = Simulations.HyperElastic(
        _plate_with_hole(ElemType.TRI3), Models.HyperElastic.NeoHookean(2, K=5e3)
    )
    _clamped_and_pulled(simu, [0.4, 0.2], ["x", "y"])
    simu.Solve()
    return simu


_CASES = {"elastic": _elastic, "newmark": _newmark, "newton": _newton}


def _serial(case: str) -> np.ndarray:
    """Coordinates then displacement per node, from `case` solved serially in a child process."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("OMPI_", "PMI", "MV2_"))
    }
    env["EASYFEA_MPI"] = "off"
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "serial.npy")
        subprocess.run([sys.executable, __file__, case, path], env=env, check=True)
        return np.load(path)


@pytest.mark.mpi
@pytest.mark.parametrize("case", _CASES)
def test_Solve_over_the_ranks_matches_the_serial_solve(case):
    pytest.importorskip("petsc4py")
    simu = _CASES[case]()
    nodes = simu.mesh._Get_mpi_owned_nodes()
    coord = simu.mesh.coord[nodes, :2]
    u = simu.displacement.reshape(-1, 2)[nodes]

    serial = _serial(case)

    np.testing.assert_array_equal(serial[nodes, :2], coord)
    uSerial = serial[nodes, 2:]
    assert np.abs(u - uSerial).max() < 1e-6 * np.abs(serial[:, 2:]).max()


if __name__ == "__main__":
    case, path = sys.argv[1:]
    simu = _CASES[case]()
    np.save(path, np.hstack([simu.mesh.coord[:, :2], simu.displacement.reshape(-1, 2)]))
