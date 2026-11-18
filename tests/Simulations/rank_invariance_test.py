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
from EasyFEA.Models.InElastic import Plasticity
from EasyFEA.Models.InElastic.IsotropicHardening import Linear as LinearHardening
from EasyFEA.Models.InElastic.Yield import VonMises
from EasyFEA.Utilities import Allgather


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


def _pulled_dofs(simu, unknown: str):
    nodesPulled = simu.mesh.Nodes_Conditions(lambda x, y, z: x == 2)
    return simu.Bc_dofs_nodes(nodesPulled, [unknown])


def _elastic() -> Simulations.Elastic:
    """E heterogeneous: softer left of the hole."""
    mesh = _plate_with_hole(ElemType.TRI6)
    group = mesh.groupElem
    E_e = np.where(mesh.coord[group.connect].mean(1)[:, 0] < 0.7, 70e3, 210e3)
    simu = Simulations.Elastic(mesh, Models.Elastic.Isotropic(2, E=E_e, v=0.3))
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


def _inelastic() -> Simulations.InElastic:
    """Yields around the hole, so that the plastic strain is an internal variable that varies."""
    elastic = Models.Elastic.Isotropic(2, E=210e3, v=0.3)
    behavior = Plasticity(elastic, VonMises(250.0), LinearHardening(2000.0))
    simu = Simulations.InElastic(_plate_with_hole(ElemType.TRI3), behavior)
    for step in range(1, 3):
        simu.Bc_Init()
        _clamped_and_pulled(simu, [1e-3 * step], ["x"])
        simu.Solve()
    return simu


def _moving() -> Simulations.Elastic:
    """The known dofs change between the solves: pulled along x, then x and y, then y."""
    simu = Simulations.Elastic(
        _plate_with_hole(ElemType.TRI3), Models.Elastic.Isotropic(2, E=210e3, v=0.3)
    )
    for pull, unknowns in (
        ([1e-3], ["x"]),
        ([1e-3, 1e-3], ["x", "y"]),
        ([2e-3], ["y"]),
    ):
        simu.Bc_Init()
        _clamped_and_pulled(simu, pull, unknowns)
        simu.Solve()
    return simu


def _monolithic() -> Simulations.IncompressibleHyperElastic:
    """u and p side by side, a saddle point."""
    material = Models.HyperElastic.NeoHookean(2, K=1.0)
    simu = Simulations.IncompressibleHyperElastic(
        _plate_with_hole(ElemType.TRI6), material
    )
    _clamped_and_pulled(simu, [0.1, 0.05], ["x", "y"])
    simu.Solve()
    return simu


_CASES = {
    "elastic": _elastic,
    "newmark": _newmark,
    "newton": _newton,
    "inelastic": _inelastic,
    "moving": _moving,
    "monolithic": _monolithic,
}

# the results a gathered simulation gives, at nodes and on elements
_RESULTS = {
    "elastic": ["displacement", "Svm"],
    "newmark": ["displacement", "speed", "accel", "Svm"],
    "newton": ["displacement", "Svm"],
    "inelastic": ["displacement", "Svm", "p"],
    "moving": ["displacement", "Svm"],
    "monolithic": ["displacement", "p", "Svm"],
}


def _space_coord(order: int) -> np.ndarray:
    """Coordinates of every node id of the displacement on the order `order` elements of a TRI3 mesh, zero where this rank holds none."""
    mesh = _plate_with_hole(ElemType.TRI3)
    mesh = mesh._With_FunctionSpace(Simulations.Elastic.problemType, order)
    return mesh._Get_FunctionSpace(Simulations.Elastic.problemType).coord


def _scalars(simu) -> np.ndarray:
    """Every scalar a script reads, the same on every rank."""
    if isinstance(simu, Simulations.InElastic):
        return np.empty(0)
    scalars = list(simu.Results_dict_Energy().values())
    if isinstance(
        simu, (Simulations.HyperElastic, Simulations.IncompressibleHyperElastic)
    ):
        return np.array([*scalars, simu.Result("W")])
    K = simu.Get_K_C_M_F(simu.problemType)[0]
    u = simu.displacement
    scalars += [simu.Calc_Resultant(_pulled_dofs(simu, u_)) for u_ in ("x", "y")]
    scalars.append(simu.Calc_Energy(K, u, simu.Get_dofs(simu.problemType)))
    scalars.append(simu.Result("Wdef"))
    return np.array(scalars, dtype=float)


def _results(simu, case: str) -> dict[str, np.ndarray]:
    """The case's results at nodes, then on elements (`_e`)."""
    results = {}
    for name in _RESULTS[case]:
        results[name] = simu.Result(name)
        results[f"{name}_e"] = simu.Result(name, nodeValues=False)
    return results


def _serial(case: str) -> dict[str, np.ndarray]:
    """Coordinates then displacement per node, and the scalars, from `case` solved serially in a child process."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("OMPI_", "PMI", "MV2_"))
    }
    env["EASYFEA_MPI"] = "off"
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "serial.npz")
        subprocess.run([sys.executable, __file__, case, path], env=env, check=True)
        with np.load(path) as data:
            return dict(data)


@pytest.mark.mpi
@pytest.mark.parametrize("order", [2, 3])
def test_added_nodes_over_the_ranks_have_the_serial_ids(order):
    mesh = _plate_with_hole(ElemType.TRI3)
    mesh = mesh._With_FunctionSpace(Simulations.Elastic.problemType, order)
    space = mesh._Get_FunctionSpace(Simulations.Elastic.problemType)
    held = space.localNodes

    serial = _serial(f"space{order}")["coord"]

    assert space.Ncoords == serial.shape[0]
    np.testing.assert_allclose(space.coord[held], serial[held], rtol=0, atol=1e-12)
    owned = np.concatenate(Allgather(space.ownedNodes, mesh.comm))
    assert np.array_equal(np.sort(owned), np.arange(serial.shape[0]))


@pytest.mark.mpi
@pytest.mark.parametrize("case", _CASES)
def test_Solve_over_the_ranks_matches_the_serial_solve(case):
    pytest.importorskip("petsc4py")
    simu = _CASES[case]()
    nodes = simu.mesh._Get_mpi_owned_nodes()
    coord = simu.mesh.coord[nodes, :2]
    u = simu.displacement.reshape(-1, 2)[nodes]
    scalars = _scalars(simu)

    serial = _serial(case)

    np.testing.assert_array_equal(serial["nodes"][nodes, :2], coord)
    uSerial = serial["nodes"][nodes, 2:]
    assert np.abs(u - uSerial).max() < 1e-6 * np.abs(serial["nodes"][:, 2:]).max()
    for scalarsRank in Allgather(scalars, simu.mesh.comm):
        np.testing.assert_array_equal(scalarsRank, scalars)
    np.testing.assert_allclose(scalars, serial["scalars"], rtol=1e-6, atol=1e-12)


@pytest.mark.mpi
@pytest.mark.parametrize("case", _CASES)
def test_Gather_gives_rank_0_the_serial_results(case):
    pytest.importorskip("petsc4py")
    simu = _CASES[case]()
    comm = simu.mesh.comm
    serial = _serial(case)

    simu._Gather()

    if comm.Get_rank() != 0:
        assert simu.mesh.isPartitioned
        return
    assert not simu.mesh.isPartitioned
    for name, values in _results(simu, case).items():
        expected = serial[f"result/{name}"]
        assert values.shape == expected.shape, name
        scale = max(np.abs(expected).max(), 1e-300)
        assert np.abs(values - expected).max() < 1e-6 * scale, name


if __name__ == "__main__":
    case, path = sys.argv[1:]
    if case.startswith("space"):
        np.savez(path, coord=_space_coord(int(case[5:])))
        sys.exit()
    simu = _CASES[case]()
    nodes = np.hstack([simu.mesh.coord[:, :2], simu.displacement.reshape(-1, 2)])
    results = {f"result/{k}": v for k, v in _results(simu, case).items()}
    np.savez(path, nodes=nodes, scalars=_scalars(simu), **results)
