# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import functools
import importlib.util
import os
from typing import Callable

import numpy as np
import pytest

from EasyFEA import ElemType, Mesher, Models, Simulations
from EasyFEA.FEM import BiLinearForm, Field
from EasyFEA.Geoms import Domain, Line, Point

from . import _reference


@pytest.fixture
def thermal_simu() -> Simulations.Thermal:
    """Thermal simulation whose matrices are already assembled, so `needUpdate` starts False."""
    domain = Domain(Point(), Point(1, 1))
    mesh = domain.Mesh_2D()
    simu = Simulations.Thermal(mesh, Models.Thermal(1, 1))
    simu.Get_K_C_M_F()
    return simu


def _history_simu(folder: str = "") -> Simulations.Elastic:
    domain = Domain(Point(), Point(1, 1), meshSize=0.5)
    mesh = domain.Mesh_2D()
    mat = Models.Elastic.Isotropic(
        dim=2,
        E=210e3,
        v=0.3,
        planeStress=True,
        thickness=1.0,
    )
    return Simulations.Elastic(mesh, mat, folder=folder)


def test_assembly_clears_the_update_flag(thermal_simu):
    assert not thermal_simu.needUpdate


@pytest.mark.parametrize(
    "transform",
    [
        lambda mesh: mesh.Rotate(45, mesh.center),
        lambda mesh: mesh.Translate(dy=-10),
        lambda mesh: mesh.Symmetry(mesh.center, (1, 0)),
    ],
    ids=["rotate", "translate", "symmetry"],
)
def test_a_mesh_transform_flags_the_simulation_for_update(thermal_simu, transform):
    transform(thermal_simu.mesh)
    assert thermal_simu.needUpdate


def test_a_new_mesh_flags_the_simulation_for_update(thermal_simu):
    thermal_simu.mesh = thermal_simu.mesh.copy()
    assert thermal_simu.needUpdate


def test_history_pins_mode_per_entry(tmp_path):
    """Each Save_Iter pins its entry as a dict (folder=='') or pickle path (folder set) at write time. Get_result dispatches uniformly."""
    simu = _history_simu()
    simu.Save_Iter()
    simu.Save_Iter()
    simu.folder = str(tmp_path / "run")
    simu.Save_Iter()
    simu.Save_Iter()

    assert simu.Niter == 4
    assert all("indexMesh" in simu.Get_results(i) for i in range(simu.Niter))
    results_dir = os.path.join(simu.folder, "Results")
    assert sorted(os.listdir(results_dir)) == ["results2.pickle", "results3.pickle"]


def test_folder_at_construction(tmp_path):
    """Passing `folder=...` at construction must not crash — setter runs before __list_results exists."""
    simu = _history_simu(folder=str(tmp_path / "run"))
    simu.Save_Iter()
    assert os.path.isfile(os.path.join(simu.folder, "Results", "results0.pickle"))
    assert "indexMesh" in simu.Get_results(0)


def test_folder_can_be_unset(tmp_path):
    """`simu.folder = ''` after some on-disk Save_Iter calls is fine — old entries keep their paths; new ones go to memory."""
    simu = _history_simu(folder=str(tmp_path / "run"))
    simu.Save_Iter()
    simu.folder = ""
    simu.Save_Iter()
    assert "indexMesh" in simu.Get_results(0)
    assert "indexMesh" in simu.Get_results(1)


def _bar():
    domain = Domain(Point(), Point(2, 1), meshSize=0.5)
    return domain.Mesh_2D([], ElemType.QUAD4, isOrganised=True)


def _elastic_bar(dynamic: bool) -> Simulations.Elastic:
    mat = Models.Elastic.Isotropic(2, E=210e3, v=0.3)
    simu = Simulations.Elastic(_bar(), mat, verbosity=False)
    if dynamic:
        simu.Solver_Set_Hyperbolic_Algorithm(0.1)
    _reference.pull_bar(simu, 1e-3)
    return simu


def _hyperelastic(dynamic: bool) -> Simulations.HyperElastic:
    mat = Models.HyperElastic.NeoHookean(2, K=5e4)
    simu = Simulations.HyperElastic(_bar(), mat, verbosity=False)
    if dynamic:
        simu.Solver_Set_Hyperbolic_Algorithm(0.1)
    _reference.pull_bar(simu, 1e-3)
    return simu


def _inelastic() -> Simulations.InElastic:
    elastic = Models.Elastic.Isotropic(2, E=210e3, v=0.3)
    behavior = Models.InElastic.Maxwell(elastic, [0.3], [1.0])
    simu = Simulations.InElastic(_bar(), behavior)
    simu.dt = 0.5
    _reference.pull_bar(simu, 1e-3)
    return simu


def _heat(simu: Simulations._Simu, unknowns: list[str]) -> None:
    mesh = simu.mesh
    nodesCold = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesHot = mesh.Nodes_Conditions(lambda x, y, z: x == 2)
    simu.add_dirichlet(nodesCold, [0], unknowns)
    simu.add_dirichlet(nodesHot, [1], unknowns)


def _thermal(transient: bool) -> Simulations.Thermal:
    simu = Simulations.Thermal(_bar(), Models.Thermal(1, 1), verbosity=False)
    if transient:
        simu.Solver_Set_Parabolic_Algorithm(0.1)
    _heat(simu, ["t"])
    return simu


def _phasefield() -> Simulations.PhaseField:
    mat = Models.Elastic.Isotropic(2, E=210e3, v=0.3)
    pfm = Models.PhaseField(
        mat,
        "He",
        "AT2",
        Gc=1.0,
        l0=0.5,
    )
    simu = Simulations.PhaseField(_bar(), pfm, verbosity=False)
    _reference.pull_bar(simu, 1e-2)
    return simu


def _beam(dim: int, useTimoshenko: bool) -> Simulations.Beam:
    line = Line(Point(), Point(x=10), 1)
    domain = Domain(Point(-0.5, -0.5), Point(0.5, 0.5))
    section = domain.Mesh_2D()
    beam = Models.Beam.Isotropic(
        dim,
        line,
        section,
        210e3,
        0.3,
    )
    mesh = Mesher().Mesh_Beams([beam], elemType=ElemType.SEG2)
    structure = Models.Beam.BeamStructure([beam])
    simu = Simulations.Beam(
        mesh, structure, useTimoshenko=useTimoshenko, verbosity=False
    )
    nodesClamped = mesh.Nodes_Point(Point())
    nodesTip = mesh.Nodes_Point(Point(x=10))
    simu.add_dirichlet(nodesClamped, [0] * simu.Get_dof_n(), simu.Get_unknowns())
    loaded = [u for u in ("x", "y", "z") if u in simu.Get_unknowns()]
    simu.add_neumann(nodesTip, [-1] * len(loaded), loaded)
    return simu


def _weakforms() -> Simulations.WeakForms:
    mesh = _bar()
    field = Field(mesh.groupElem, 1)

    @BiLinearForm
    def laplacian(u: Field, v: Field):
        return u.grad.dot(v.grad)

    simu = Simulations.WeakForms(
        mesh, Models.WeakForms(field, laplacian), verbosity=False
    )
    _heat(simu, simu.Get_unknowns())
    return simu


HAS_JAX = importlib.util.find_spec("jax") is not None

SIMULATIONS: dict[str, Callable[[], Simulations._Simu]] = {
    "Elastic": lambda: _elastic_bar(False),
    "Elastic hyperbolic": lambda: _elastic_bar(True),
    "HyperElastic": lambda: _hyperelastic(False),
    "HyperElastic hyperbolic": lambda: _hyperelastic(True),
    "Thermal": lambda: _thermal(False),
    "Thermal parabolic": lambda: _thermal(True),
    "PhaseField": _phasefield,
    "Beam 1D": lambda: _beam(1, False),
    "Beam 2D": lambda: _beam(2, False),
    "Beam 3D": lambda: _beam(3, False),
    "Beam 2D Timoshenko": lambda: _beam(2, True),
    "Beam 3D Timoshenko": lambda: _beam(3, True),
    "WeakForms": _weakforms,
}
if HAS_JAX:
    SIMULATIONS["InElastic"] = _inelastic


@functools.cache
def _solved(name: str) -> Simulations._Simu:
    simu = SIMULATIONS[name]()
    simu.Solve()
    simu.Save_Iter()
    return simu


AVAILABLE_RESULTS = [
    pytest.param(name, result, id=f"{name}-{result}")
    for name, make in SIMULATIONS.items()
    for result in make().Results_Available()
]
if not HAS_JAX:
    AVAILABLE_RESULTS.append(
        pytest.param("InElastic", "", marks=pytest.mark.skip(reason="needs jax"))
    )


@pytest.mark.parametrize("nodeValues", [True, False], ids=["nodes", "elements"])
@pytest.mark.parametrize("name, result", AVAILABLE_RESULTS)
def test_every_available_result_is_computed(name: str, result: str, nodeValues: bool):
    """Per node or per element (dof vectors flattened), or a scalar."""
    simu = _solved(name)
    values = simu.Result(result, nodeValues=nodeValues)
    assert values is not None
    assert np.isfinite(values).all()
    if np.ndim(values) > 0:
        N = simu.mesh.Nn if nodeValues else simu.mesh.Ne
        assert np.shape(values)[0] % N == 0
