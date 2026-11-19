# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Under `make test-mpi`, a simulation solved over the ranks matches the same one solved serially by a child process run with `EASYFEA_MPI=off`."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from EasyFEA import ElemType, Models, Simulations
from EasyFEA.Geoms import Circle, Domain
from EasyFEA.Models.InElastic import ONE, Plasticity, _Behavior
from EasyFEA.Models.InElastic.IsotropicHardening import Linear as LinearHardening
from EasyFEA.Models.InElastic.Yield import VonMises
from EasyFEA.Utilities import Allgather, Reduce_max, Reduce_sum


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


def _pinned() -> Simulations.IncompressibleHyperElastic:
    """u prescribed on the whole boundary, so that p is free up to a constant, pinned at one node."""
    material = Models.HyperElastic.NeoHookean(2, K=1.0)
    mesh = _plate_with_hole(ElemType.TRI6)
    simu = Simulations.IncompressibleHyperElastic(mesh, material)
    nodesBoundary = mesh.Nodes_Tags(*(f"L{i}" for i in range(8)))
    simu.add_dirichlet(
        nodesBoundary,
        [lambda x, y, z: 0.02 * y**2, lambda x, y, z: 0.02 * x**2],
        ["x", "y"],
    )
    simu.Solve()
    return simu


_CASES = {
    "elastic": _elastic,
    "newmark": _newmark,
    "newton": _newton,
    "inelastic": _inelastic,
    "moving": _moving,
    "monolithic": _monolithic,
    "pinned": _pinned,
}


class _Heated(_Behavior):
    """Elastic, expanding with the temperature change T, an external variable."""

    class Externals(NamedTuple):
        T: float

    def Update(self, eps, z, dt, T):
        return self.Stress(eps, z, T=T), z

    def Stress(self, eps, z, T):
        return self.C @ (eps - 1e-5 * T * ONE)


def _history(case: str, folder: str = "", steps=(1, 1.5, 2), simu=None):
    """`case` loaded step after step, each step saved; `simu` resumed if given."""
    if simu is None:
        elastic = Models.Elastic.Isotropic(2, E=210e3, v=0.3)
        if case == "plastic":
            behavior = Plasticity(elastic, VonMises(250.0), LinearHardening(2000.0))
        else:
            behavior = _Heated(elastic)
        mesh = _plate_with_hole(ElemType.TRI3)
        simu = Simulations.InElastic(mesh, behavior, folder=folder)
    for step in steps:
        simu.Bc_Init()
        if case == "plastic":
            _clamped_and_pulled(simu, [1e-3 * step], ["x"])
        else:
            simu.Set_external(T=100.0 * step * simu.mesh.coord[:, 0])
            _clamped_and_pulled(simu, [0.0], ["x"])
        simu.Solve()
        simu.Save_Iter()
    return simu


def _held(simu) -> dict[str, np.ndarray]:
    """The rank's rows of what a script reads, ghosts included."""
    return {
        "u": simu.displacement,
        "Svm_e": simu.Result("Svm", nodeValues=False),
        "Svm": simu.Result("Svm"),
    }


_HISTORIES = ["plastic", "heated"]


# the results a gathered simulation gives, at nodes and on elements
_RESULTS = {
    "elastic": ["displacement", "Svm"],
    "newmark": ["displacement", "speed", "accel", "Svm"],
    "newton": ["displacement", "Svm"],
    "inelastic": ["displacement", "Svm", "p"],
    "moving": ["displacement", "Svm"],
    "monolithic": ["displacement", "p", "Svm"],
    "pinned": ["displacement", "p"],
}


def _space_coord(order: int) -> np.ndarray:
    """Coordinates of every node id of the displacement on the order `order` elements of a TRI3 mesh."""
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
    globalNodes = space.globalNodes

    serial = _serial(f"space{order}")["coord"]

    assert Reduce_max(int(globalNodes.max()), mesh.comm) + 1 == serial.shape[0]
    np.testing.assert_allclose(
        space.coord[held], serial[globalNodes[held]], rtol=0, atol=1e-12
    )
    owned = np.concatenate(Allgather(globalNodes[space.ownedNodes], mesh.comm))
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

    rows = serial["nodes"][simu.mesh.globalNodes[nodes]]
    np.testing.assert_array_equal(rows[:, :2], coord)
    uSerial = rows[:, 2:]
    assert np.abs(u - uSerial).max() < 1e-6 * np.abs(serial["nodes"][:, 2:]).max()
    for scalarsRank in Allgather(scalars, simu.mesh.comm):
        np.testing.assert_array_equal(scalarsRank, scalars)
    np.testing.assert_allclose(scalars, serial["scalars"], rtol=1e-6, atol=1e-12)


@pytest.mark.mpi
@pytest.mark.parametrize("case", _CASES)
def test_Solve_over_the_ranks_holds_the_rank_nodes_only(case):
    pytest.importorskip("petsc4py")
    simu = _CASES[case]()
    mesh = simu.mesh
    whole = Reduce_sum(mesh._Get_mpi_owned_nodes().size, mesh.comm)

    assert mesh.Nn < whole
    assert mesh.coord.shape == (mesh.Nn, 3)
    assert simu.displacement.shape == (mesh.Nn * 2,)
    assert simu.Result("Svm").shape == (mesh.Nn,)


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


def _globalElements(mesh) -> np.ndarray:
    """Global id of each of the rank's elements, in `Result(..., nodeValues=False)` order."""
    (groupElem,) = mesh.Get_list_groupElem(mesh.dim)
    return groupElem._globalElements


@pytest.mark.mpi
@pytest.mark.parametrize("case", _CASES)
def test_Result_over_the_ranks_gives_the_serial_rows_the_rank_holds(case):
    """Ghosts included: a ghost row is its owner's, nodal averages across the partition interfaces too."""
    pytest.importorskip("petsc4py")
    simu = _CASES[case]()
    mesh = simu.mesh
    serial = _serial(case)

    rows = {"": mesh.globalNodes, "_e": _globalElements(mesh)}
    for name, values in _results(simu, case).items():
        expected = serial[f"result/{name}"]
        rowsSerial = rows["_e" if name.endswith("_e") else ""]
        expected = expected.reshape(-1, values.size // rowsSerial.size)[rowsSerial]
        scale = max(np.abs(serial[f"result/{name}"]).max(), 1e-300)
        assert np.abs(values.reshape(expected.shape) - expected).max() < 1e-6 * scale, (
            name
        )


@pytest.mark.mpi
def test_Paraview_pieces_over_the_ranks_give_back_the_serial_fields(tmp_path):
    """The owned cells, their points by GlobalPointIds, hold the serial fields once; a point in several pieces carries one value."""
    pytest.importorskip("petsc4py")
    pv = pytest.importorskip("pyvista")
    from EasyFEA import IO

    simu = _elastic()
    simu.Save_Iter()
    comm = simu.mesh.comm
    folder = comm.bcast(str(tmp_path), root=0)
    pvdFile = IO.Paraview.Save_simu(simu, folder, results=["Svm"])
    serial = _serial("elastic")
    comm.Barrier()
    if comm.Get_rank() != 0:
        return

    u = serial["result/displacement"].reshape(-1, 2)
    svm_n = serial["result/Svm"]
    cellsSerial = {tuple(sorted(c)): e for e, c in enumerate(serial["connect"])}
    pieces = sorted(Path(pvdFile).parent.glob("Rank*/solution_0.vtu"))
    assert len(pieces) == comm.Get_size()
    svm = np.full(len(cellsSerial), np.nan)
    for piece in pieces:
        grid = pv.read(piece)
        gids = grid.point_data["GlobalPointIds"].astype(int)
        np.testing.assert_allclose(grid.points[:, :2], serial["nodes"][gids, :2])
        displacement = grid.point_data["displacement"][:, :2]
        assert np.abs(displacement - u[gids]).max() < 1e-6 * np.abs(u).max()
        assert np.abs(grid.point_data["Svm"] - svm_n[gids]).max() < 1e-6 * svm_n.max()
        owned = grid.cell_data["vtkGhostType"] == 0
        for c in np.flatnonzero(owned):
            e = cellsSerial[tuple(sorted(gids[grid.get_cell(c).point_ids]))]
            assert np.isnan(svm[e]), "an owned cell in two pieces"
            svm[e] = grid.cell_data["Svm"][c]
    expected = serial["result/Svm_e"]
    assert np.abs(svm - expected).max() < 1e-6 * np.abs(expected).max()


def _shared_folder(tmp_path) -> str:
    from EasyFEA.Utilities._mpi import MPI_COMM

    return MPI_COMM.bcast(str(tmp_path), root=0)


def _assert_held_equal(
    simu, expected: dict[str, np.ndarray], label: str, rtol: float = 0.0
):
    for name, values in _held(simu).items():
        atol = rtol * np.abs(expected[name]).max()
        np.testing.assert_allclose(
            values, expected[name], rtol=0, atol=atol, err_msg=f"{label} {name}"
        )


@pytest.mark.mpi
@pytest.mark.parametrize("case", _HISTORIES)
@pytest.mark.parametrize("inFolder", [False, True], ids=["memory", "folder"])
def test_Set_Iter_over_the_ranks_gives_back_the_rank_rows_ghosts_included(
    case, inFolder, tmp_path
):
    pytest.importorskip("petsc4py")
    folder = _shared_folder(tmp_path) if inFolder else ""
    # an earlier run's files in the folder: a read right after Save_Iter sees this run's
    _history(case, folder, steps=[0.5, 1.25])
    lives = []
    simu = _history(case, folder, steps=[])
    for step in (1, 1.5, 2):
        _history(case, simu=simu, steps=[step])
        lives.append(_held(simu))

    for i, live in enumerate(lives):
        simu.Set_Iter(i)
        _assert_held_equal(simu, live, f"iteration {i}")
        assert simu.Get_results(i).keys() == simu.Set_Iter(i).keys()


@pytest.mark.mpi
@pytest.mark.parametrize("case", _HISTORIES)
def test_Set_Iter_over_the_ranks_gives_the_serial_iterations(case, tmp_path):
    pytest.importorskip("petsc4py")
    simu = _history(case, _shared_folder(tmp_path))
    mesh = simu.mesh
    serial = _serial(f"history:{case}")

    for i in range(simu.Niter):
        simu.Set_Iter(i)
        u = simu.displacement.reshape(-1, 2)
        expected = serial[f"u{i}"][mesh.globalNodes]
        assert np.abs(u - expected).max() < 1e-6 * np.abs(expected).max()
        svm = simu.Result("Svm", nodeValues=False)
        expected = serial[f"Svm_e{i}"][_globalElements(mesh)]
        assert np.abs(svm - expected).max() < 1e-6 * np.abs(expected).max()


@pytest.mark.mpi
def test_Set_Iter_over_the_ranks_reads_an_earlier_mesh_of_an_order_2_space(
    tmp_path,
):
    pytest.importorskip("petsc4py")
    folder = _shared_folder(tmp_path)
    problemType = Simulations.Elastic.problemType
    meshes = [
        _plate_with_hole(ElemType.TRI3)._With_FunctionSpace(problemType, 2)
        for _ in range(2)
    ]
    material = Models.Elastic.Isotropic(2, E=210e3, v=0.3)
    simu = Simulations.Elastic(meshes[0], material, folder=folder)
    lives = []
    for i, mesh in enumerate(meshes):
        if i > 0:
            simu.mesh = mesh
        _clamped_and_pulled(simu, [1e-3 * (i + 1)], ["x"])
        simu.Solve()
        simu.Save_Iter()
        lives.append(simu.displacement)

    for i, live in enumerate(lives):
        simu.Set_Iter(i)
        np.testing.assert_array_equal(simu.displacement, live)


@pytest.mark.mpi
def test_Set_Iter_over_the_ranks_raises_on_every_rank_when_a_file_is_missing(
    tmp_path,
):
    pytest.importorskip("petsc4py")
    folder = _shared_folder(tmp_path)
    simu = _history("plastic", folder, steps=[1])
    comm = simu.mesh.comm
    if comm.Get_rank() == 0:
        Path(
            folder, f"Rank{comm.Get_size() - 1}", "Results", "results0.pickle"
        ).unlink()
    comm.Barrier()

    with pytest.raises(ValueError, match="No such file"):
        simu.Set_Iter(0)


@pytest.mark.mpi
def test_Save_Iter_over_the_ranks_stores_each_row_once(tmp_path):
    pytest.importorskip("petsc4py")
    folder = _shared_folder(tmp_path)
    simu = _history("plastic", folder)
    comm = simu.mesh.comm
    rank = comm.Get_rank()

    files = sorted(Path(folder, f"Rank{rank}", "Results").glob("*.pickle"))
    size = Reduce_sum(sum(f.stat().st_size for f in files), comm)
    assert len(files) == 3
    assert not Path(folder, "Results").exists()
    serial = _serial("saved:plastic")["bytes"]
    assert serial < size < serial * 1.05 + 1024 * comm.Get_size() * len(files)


@pytest.mark.mpi
def test_Set_Iter_over_the_ranks_refuses_a_file_of_another_partition(tmp_path):
    pytest.importorskip("petsc4py")
    import pickle

    folder = _shared_folder(tmp_path)
    simu = _history("plastic", folder, steps=[1])
    comm = simu.mesh.comm
    if comm.Get_rank() == 0:
        path = Path(folder, "Rank0", "Results", "results0.pickle")
        iteration = pickle.loads(path.read_bytes())
        for fields in iteration["quadrature"].values():
            for name in fields:
                fields[name] = fields[name][1:]
        path.write_bytes(pickle.dumps(iteration))
    comm.Barrier()

    with pytest.raises(ValueError, match="Rank0.*saved over another partition"):
        simu.Set_Iter(0)


@pytest.mark.mpi
@pytest.mark.parametrize("case", _HISTORIES)
@pytest.mark.parametrize("inFolder", [False, True], ids=["memory", "folder"])
def test_Load_Simu_over_the_ranks_resumes_the_run_where_it_was_saved(
    case, inFolder, tmp_path
):
    pytest.importorskip("petsc4py")
    folder = _shared_folder(tmp_path)
    whole = _history(case)
    expected = _held(whole)
    whole.Set_Iter(0)
    first = _held(whole)

    stopped = _history(case, folder if inFolder else "", steps=(1, 1.5))
    stopped.Save(folder)
    simu = Simulations.Load_Simu(folder)
    assert simu.mesh.isPartitioned
    _history(case, simu=simu, steps=[2])

    # the solver is rebuilt: round-off
    _assert_held_equal(simu, expected, "resumed", rtol=1e-10)
    assert simu.Niter == 3
    simu.Set_Iter(0)
    _assert_held_equal(simu, first, "iteration 0")
    assert not Path(folder, "Results").exists()


@pytest.mark.mpi
def test_Load_Simu_over_another_rank_count_raises(tmp_path):
    pytest.importorskip("petsc4py")
    folder = _shared_folder(tmp_path)
    simu = _history("plastic", folder, steps=[1])
    simu.Save(folder)
    comm = simu.mesh.comm
    if comm.Get_rank() == 0:
        last = Path(folder, f"Rank{comm.Get_size() - 1}")
        last.rename(Path(folder, "moved"))
    comm.Barrier()

    with pytest.raises(ValueError, match="saved over .* rank"):
        Simulations.Load_Simu(folder)


if __name__ == "__main__":
    case, path = sys.argv[1:]
    if case.startswith("space"):
        np.savez(path, coord=_space_coord(int(case[5:])))
        sys.exit()
    if case.startswith("history:"):
        simu = _history(case[8:])
        iterations = {}
        for i in range(simu.Niter):
            simu.Set_Iter(i)
            iterations[f"u{i}"] = simu.displacement.reshape(-1, 2)
            iterations[f"Svm_e{i}"] = simu.Result("Svm", nodeValues=False)
        np.savez(path, **iterations)
        sys.exit()
    if case.startswith("saved:"):
        folder = os.path.dirname(path)
        _history(case[6:], folder)
        files = Path(folder, "Results").glob("*.pickle")
        np.savez(path, bytes=sum(f.stat().st_size for f in files))
        sys.exit()
    simu = _CASES[case]()
    nodes = np.hstack([simu.mesh.coord[:, :2], simu.displacement.reshape(-1, 2)])
    results = {f"result/{k}": v for k, v in _results(simu, case).items()}
    (groupElem,) = simu.mesh.Get_list_groupElem(simu.mesh.dim)
    np.savez(
        path,
        nodes=nodes,
        scalars=_scalars(simu),
        connect=groupElem.connect,
        **results,
    )
