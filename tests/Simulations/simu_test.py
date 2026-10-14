# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""`_Simu`, its members tested once here, over every simulation where they apply."""

import importlib.util
import os
from typing import Callable

import numpy as np
import pytest

from EasyFEA import AlgoType, ElemType, Mesh, Mesher, Models, Simulations, SolverType
from EasyFEA.FEM import Field, LagrangeCondition
from EasyFEA.Geoms import Domain, Line

from _numerics import rel_error
from . import _reference
from ._simus import laplacian, laplacian_form

L, H, B = 2.0, 0.5, 1.0
E, v = 210e3, 0.3
UD = 1e-3


# ----------------------------------------------
# One simulation of each kind
# ----------------------------------------------


def _bar() -> Mesh:
    """4 QUAD4 in a row."""
    domain = Domain((0, 0), (L, H), H)
    return domain.Mesh_2D([], ElemType.QUAD4, isOrganised=True)


def _box() -> Mesh:
    domain = Domain((0, 0), (L, H), H)
    return domain.Mesh_Extrude(
        [],
        [0, 0, B],
        [1],
        ElemType.HEXA8,
        isOrganised=True,
    )


def _elastic_bar(
    dynamic: bool = False, mesh: Mesh | None = None
) -> Simulations.Elastic:
    mesh = _bar() if mesh is None else mesh
    material = Models.Elastic.Isotropic(mesh.dim, E, v, planeStress=True)
    simu = Simulations.Elastic(mesh, material)
    simu.solver = SolverType.scipy
    if dynamic:
        simu.Solver_Set_Hyperbolic_Algorithm(0.1)
    _reference.pull_bar(simu, UD)
    return simu


def _hyperelastic(dynamic: bool) -> Simulations.HyperElastic:
    material = Models.HyperElastic.NeoHookean(2, K=5e4)
    simu = Simulations.HyperElastic(_bar(), material)
    if dynamic:
        simu.Solver_Set_Hyperbolic_Algorithm(0.1)
    _reference.pull_bar(simu, UD)
    return simu


def _inelastic() -> Simulations.InElastic:
    elastic = Models.Elastic.Isotropic(2, E, v)
    behavior = Models.InElastic.Maxwell(elastic, [0.3], [1.0])
    simu = Simulations.InElastic(_bar(), behavior)
    simu.dt = 0.5
    _reference.pull_bar(simu, UD)
    return simu


def _heat(simu: Simulations._Simu, unknowns: list[str]) -> None:
    mesh = simu.mesh
    nodesCold = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesHot = mesh.Nodes_Conditions(lambda x, y, z: x == L)
    simu.add_dirichlet(nodesCold, [0], unknowns)
    simu.add_dirichlet(nodesHot, [1], unknowns)


def _thermal(transient: bool = False) -> Simulations.Thermal:
    simu = Simulations.Thermal(_bar(), Models.Thermal(1, 1))
    simu.solver = SolverType.scipy
    if transient:
        simu.Solver_Set_Parabolic_Algorithm(0.1)
    _heat(simu, ["t"])
    return simu


def _phasefield() -> Simulations.PhaseField:
    material = Models.Elastic.Isotropic(2, E, v)
    pfm = Models.PhaseField(
        material,
        "He",
        "AT2",
        Gc=1.0,
        l0=0.5,
    )
    simu = Simulations.PhaseField(_bar(), pfm)
    _reference.pull_bar(simu, 1e-2)
    return simu


def _beam(dim: int, useTimoshenko: bool = False) -> Simulations.Beam:
    line = Line((0, 0), (10, 0), 1)
    domain = Domain((-0.5, -0.5), (0.5, 0.5))
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
    simu = Simulations.Beam(mesh, structure, useTimoshenko=useTimoshenko)
    nodesClamped = mesh.Nodes_Point((0, 0))
    nodesTip = mesh.Nodes_Point((10, 0))
    simu.add_dirichlet(nodesClamped, [0] * simu.Get_dof_n(), simu.Get_unknowns())
    loaded = [u for u in ("x", "y", "z") if u in simu.Get_unknowns()]
    simu.add_neumann(nodesTip, [-1] * len(loaded), loaded)
    return simu


def _weakforms() -> Simulations.WeakForms:
    mesh = _bar()
    simu = Simulations.WeakForms(mesh, laplacian(mesh))
    _heat(simu, simu.Get_unknowns())
    return simu


HAS_JAX = importlib.util.find_spec("jax") is not None

SIMULATIONS: dict[str, Callable[[], Simulations._Simu]] = {
    "Elastic": _elastic_bar,
    "Elastic hyperbolic": lambda: _elastic_bar(True),
    "HyperElastic": lambda: _hyperelastic(False),
    "HyperElastic hyperbolic": lambda: _hyperelastic(True),
    "Thermal": _thermal,
    "Thermal parabolic": lambda: _thermal(True),
    "PhaseField": _phasefield,
    "Beam 1D": lambda: _beam(1),
    "Beam 2D": lambda: _beam(2),
    "Beam 3D": lambda: _beam(3),
    "Beam 2D Timoshenko": lambda: _beam(2, True),
    "Beam 3D Timoshenko": lambda: _beam(3, True),
    "WeakForms": _weakforms,
}
if HAS_JAX:
    SIMULATIONS["InElastic"] = _inelastic


def _smooth(simu: Simulations._Simu, problemType: str, scale: float) -> np.ndarray:
    """`scale·(1 + x/L)` times the dof index, on every node."""
    x = simu.mesh.coord[:, 0] / L
    dof_n = simu.Get_dof_n(problemType)
    return scale * np.outer(1 + x, np.arange(1, dof_n + 1)).ravel()


def _inject(simu: Simulations._Simu) -> Simulations._Simu:
    """A smooth u, 2u, 3u on every problem, damage in [0.1, 0.2]: a state no solve gave."""
    for problemType in simu.Get_problemTypes():
        u = _smooth(simu, problemType, 0.1 if problemType == "damage" else 1e-3)
        simu._Set_solutions(problemType, u, 2 * u, 3 * u)
    return simu


def _injected(name: str) -> Simulations._Simu:
    return _inject(SIMULATIONS[name]())


def _bar_displacement(simu: Simulations._Simu) -> np.ndarray:
    """The pulled bar's exact displacement, as a dof vector."""
    u = _reference.bar_displacement(
        simu.mesh.coord,
        L,
        UD,
        v,
        simu.mesh.dim,
    )
    return u.ravel()


# ----------------------------------------------
# _Simu
# ----------------------------------------------


def test_Simu_describes_its_mesh_model_solver_and_conditions():
    simu = _elastic_bar()

    text = str(simu)

    for section in ("Mesh", "Model", "solver", "Boundary Conditions", "Results"):
        assert section in text


def test_Simu_describes_its_petsc_options():
    simu = _phasefield()
    simu.solver = SolverType.petsc

    text = str(simu)

    for problemType in simu.Get_problemTypes():
        kspType, pcType, solverType = simu._Solver_Get_PETSc4Py_Options(problemType)
        assert f"({problemType}) {kspType}, {pcType}, {solverType}" in text


PROBLEM_TYPES = {
    "Elastic": ["elastic"],
    "HyperElastic": ["hyperelastic"],
    "InElastic": ["inelastic"],
    "Thermal": ["thermal"],
    "PhaseField": ["damage", "elastic"],
    "Beam": ["beam"],
    "WeakForms": ["weakForm"],
}


def _kind(name: str) -> str:
    return name.split()[0]


def test_Simu_Get_problemTypes_names_each_problem_solved(subtests):
    for name, make in SIMULATIONS.items():
        with subtests.test(simulation=name):
            assert make().Get_problemTypes() == PROBLEM_TYPES[_kind(name)]


UNKNOWNS = {
    "Elastic": ["x", "y"],
    "Elastic hyperbolic": ["x", "y"],
    "HyperElastic": ["x", "y"],
    "HyperElastic hyperbolic": ["x", "y"],
    "InElastic": ["x", "y"],
    "Thermal": ["t"],
    "Thermal parabolic": ["t"],
    "PhaseField": ["x", "y"],
    "Beam 1D": ["x"],
    "Beam 2D": ["x", "y", "rz"],
    "Beam 3D": ["x", "y", "z", "rx", "ry", "rz"],
    "Beam 2D Timoshenko": ["x", "y", "rz"],
    "Beam 3D Timoshenko": ["x", "y", "z", "rx", "ry", "rz"],
    "WeakForms": ["u"],
}


def test_Simu_Get_unknowns_of_the_default_problem(subtests):
    for name, make in SIMULATIONS.items():
        with subtests.test(simulation=name):
            assert make().Get_unknowns() == UNKNOWNS[name]


def test_Simu_Get_unknowns_of_each_problem():
    simu = _phasefield()

    assert simu.Get_unknowns("damage") == ["d"]
    with pytest.raises(ValueError):
        simu.Get_unknowns("thermal")


def test_Simu_Get_unknowns_of_a_vector_weak_form():
    plate, block = _bar(), _box()

    plateSimu = Simulations.WeakForms(plate, laplacian(plate, 2))
    blockSimu = Simulations.WeakForms(block, laplacian(block, 3))

    assert plateSimu.Get_unknowns() == ["x", "y"]
    assert blockSimu.Get_unknowns() == ["x", "y", "z"]


def test_Simu_Get_dof_n_counts_the_unknowns(subtests):
    for name, make in SIMULATIONS.items():
        with subtests.test(simulation=name):
            simu = make()
            for problemType in simu.Get_problemTypes():
                dof_n = simu.Get_dof_n(problemType)
                assert dof_n == len(simu.Get_unknowns(problemType))


def test_Simu_Get_K_C_M_F_of_conduction_conserves_constants_and_heat():
    simu = _thermal()
    simu.rho = 2.0

    K, C, M, F = simu.Get_K_C_M_F()

    assert rel_error(K @ np.ones(simu.mesh.Nn), 0.0, scale=1.0) < 1e-12
    assert rel_error(C.sum(), 2.0 * L * H) < 1e-12
    assert M.nnz == 0 and F.nnz == 0


def test_Simu_Get_K_C_M_F_clears_the_update_flag_and_returns_copies():
    simu = _thermal()

    K = simu.Get_K_C_M_F()[0]
    K.data[:] = 0

    assert not simu.needUpdate
    assert simu.Get_K_C_M_F()[0].nnz > 0 and simu.Get_K_C_M_F()[0].data.any()


def test_Simu_Calc_Energy_of_the_stiffness_is_the_strain_energy():
    """Half E·eps² over the bar, for its exact displacement."""
    simu = _elastic_bar()
    u = _bar_displacement(simu)
    K = simu.Get_K_C_M_F()[0]

    energy = simu.Calc_Energy(K, u)

    assert rel_error(energy, E * (UD / L) ** 2 / 2 * L * H) < 1e-12
    assert energy == simu.Calc_Energy(K, u, simu.Get_dofs())


def _reaction_at_x_L(simu: Simulations._Simu, unknown: str) -> float:
    nodes = simu.mesh.Nodes_Conditions(lambda x, y, z: x == L)
    dofs = simu.Bc_dofs_nodes(nodes, [unknown])
    return float(np.sum(simu.Calc_Reaction(dofs)))


def test_Simu_Calc_Reaction_balances_the_pulled_end():
    simu = _elastic_bar()
    simu.Solve()

    assert rel_error(_reaction_at_x_L(simu, "x"), E * UD / L * H) < 1e-10


def test_Simu_Calc_Reaction_adds_the_inertia_of_a_dynamic_scheme():
    """An equilibrated bar stays at rest, so its reaction is the static one."""
    simu = _elastic_bar()
    simu.Solve()
    simu.Solver_Set_Hyperbolic_Algorithm(0.1)

    simu.Solve()

    assert rel_error(_reaction_at_x_L(simu, "x"), E * UD / L * H) < 1e-10


def test_Simu_Calc_Reaction_adds_the_capacity_of_a_parabolic_scheme():
    """A steady slab conducts k·ΔT/L·H through its hot end."""
    simu = _thermal(transient=True)
    simu.Solver_Set_Elliptic_Algorithm()
    simu.Solve()
    simu.Solver_Set_Parabolic_Algorithm(0.1)

    simu.Solve()

    assert rel_error(_reaction_at_x_L(simu, "t"), 1.0 / L * H) < 1e-10


def test_Simu_Calc_Reaction_of_all_dofs_sums_to_zero_without_loads():
    simu = _elastic_bar()
    simu.Solve()

    reaction = simu.Calc_Reaction()

    assert reaction.shape == (simu.mesh.Nn * 2,)
    assert abs(reaction.sum()) < 1e-10 * np.abs(reaction).max()


def test_Simu_Calc_Reaction_is_not_implemented_when_nonlinear():
    simu = _hyperelastic(False)
    simu.Solve()

    with pytest.raises(NotImplementedError):
        simu.Calc_Reaction()


def test_Simu_Get_x0_is_the_current_solution():
    simu = _injected("Elastic")

    assert np.array_equal(simu.Get_x0(), simu.displacement)
    assert simu.displacement.any()


def test_Simu_Get_terms_of_Elastic_are_its_stiffness_mass_and_damping():
    simu = _elastic_bar()
    assert [term.slots for term in simu.Get_terms()] == [("K",), ("M",)]

    simu.Set_Rayleigh_Damping_Coefs(coefM=0.1, coefK=0.2)

    slots = [term.slots for term in simu.Get_terms()]
    assert slots == [("K",), ("M",), ("C",), ("C",)]
    assert [term.scale for term in simu.Get_terms()[2:]] == [0.2, 0.1]


def test_Simu__Construct_local_matrix_system_has_one_entry_per_group():
    domain = Domain((0, 0), (120, 13), 6.5)
    mesh = domain.Mesh_2D([], ElemType.QUAD4)
    simu = Simulations.Elastic(mesh, Models.Elastic.Isotropic(2))

    local = simu._Construct_local_matrix_system(simu.problemType)

    assert list(local) == mesh.Get_list_groupElem(2)
    for groupElem, (K_e, C_e, M_e, F_e) in local.items():
        n = groupElem.nPe * 2
        assert K_e.shape == M_e.shape == (groupElem.Ne, n, n)
        assert C_e is None and F_e is None


def test_Simu_Save_Iter_pins_memory_or_disk_per_entry(tmp_path):
    simu = _elastic_bar()
    simu.Save_Iter()
    simu.Save_Iter()
    simu.folder = str(tmp_path / "run")
    simu.Save_Iter()
    simu.Save_Iter()

    assert simu.Niter == 4
    assert all("indexMesh" in simu.Get_results(i) for i in range(simu.Niter))
    resultsDir = os.path.join(simu.folder, "Results")
    assert sorted(os.listdir(resultsDir)) == ["results2.pickle", "results3.pickle"]


def test_Simu_Save_Iter_writes_to_the_folder_given_at_construction(tmp_path):
    mesh = _bar()
    folder = str(tmp_path / "run")
    simu = Simulations.Elastic(mesh, Models.Elastic.Isotropic(2), folder=folder)

    simu.Save_Iter()

    assert os.path.isfile(os.path.join(folder, "Results", "results0.pickle"))


def test_Simu_Save_Iter_records_the_newton_convergence():
    simu = _hyperelastic(False)
    simu.Solve()

    simu.Save_Iter()

    results = simu.Get_results()
    assert results["newtonIter"] == len(results["list_norm_r"]) > 0


def test_Simu_Get_results_is_a_copy_of_a_saved_iteration():
    simu = _elastic_bar()
    simu.Save_Iter()

    simu.Get_results(0)["displacement"] = None

    assert simu.Get_results(-1)["displacement"] is not None
    with pytest.raises(AssertionError, match="iter"):
        simu.Get_results(1)


def test_Simu_Set_Iter_restores_the_saved_solution():
    simu = _injected("Elastic")
    saved = simu.displacement.copy()
    simu.Save_Iter()
    simu._Set_solutions(simu.problemType, np.zeros_like(saved))

    results = simu.Set_Iter(0)

    assert np.array_equal(simu.displacement, saved)
    assert results["indexMesh"] == 0


def test_Simu_Set_Iter_restores_the_mesh_of_the_iteration():
    simu = _thermal()
    first = simu.mesh
    simu.Save_Iter()
    simu.mesh = _box()
    simu.Save_Iter()

    simu.Set_Iter(0)

    assert simu.mesh is first


def test_Simu_Set_Iter_restores_speed_and_accel_of_a_dynamic_step():
    simu = _injected("Elastic hyperbolic")
    simu.Save_Iter()
    speed, accel = simu.speed.copy(), simu.accel.copy()
    _states(simu)

    simu.Set_Iter(0)

    assert np.array_equal(simu.speed, speed) and np.array_equal(simu.accel, accel)


def test_Simu_Results_Available_include_the_displacement_matrix(subtests):
    for name, make in SIMULATIONS.items():
        with subtests.test(simulation=name):
            assert "displacement_matrix" in make().Results_Available()


def test_Simu_Result_computes_every_available_result(subtests):
    """Per node or per element (dof vectors flattened), or a scalar."""
    for name in SIMULATIONS:
        simu = _injected(name)
        for result in simu.Results_Available():
            with subtests.test(simulation=name, result=result):
                for nodeValues, N in ((True, simu.mesh.Nn), (False, simu.mesh.Ne)):
                    values = simu.Result(result, nodeValues=nodeValues)
                    assert values is not None
                    assert np.isfinite(values).all()
                    assert np.ndim(values) == 0 or np.shape(values)[0] % N == 0


def test_Simu_Result_of_an_unknown_name_is_None():
    assert _elastic_bar().Result("unknown") is None


def test_Simu_Result_at_an_iteration_reads_that_iteration():
    simu = _thermal()
    simu.Save_Iter()
    _inject(simu)
    simu.Save_Iter()

    assert np.all(simu.Result("thermal", iter=0) == 0)
    assert np.array_equal(simu.Result("thermal", iter=1), simu.thermal)


def test_Simu_Results_Iter_Summary_is_empty_by_default():
    assert _injected("Elastic").Results_Iter_Summary() == ([], [])


def _solved_phasefield() -> Simulations.PhaseField:
    """Solved and saved: a phase-field iteration records what only its staggered solve knows."""
    simu = _phasefield()
    simu.Solve()
    simu.Save_Iter()
    return simu


def test_Simu_Results_Iter_Summary_of_PhaseField_follows_each_iteration():
    simu = _solved_phasefield()
    simu.Solve()
    simu.Save_Iter()

    iterations, labelValues = simu.Results_Iter_Summary()

    assert iterations == [0, 1]
    assert all(values.shape == (2,) for _, values in labelValues)
    damage = dict(labelValues)[r"$\phi$"]
    assert damage[1] == simu.damage.max()


def test_Simu_Results_dict_Energy_per_simulation():
    elastic = _injected("Elastic")
    phasefield = _injected("PhaseField")

    assert elastic.Results_dict_Energy() == {r"$\Psi_{elas}$": elastic.Result("Wdef")}
    energies = list(phasefield.Results_dict_Energy().values())
    assert energies[2] == energies[0] + energies[1]
    assert _injected("Thermal").Results_dict_Energy() == {}


def test_Simu_Results_displacement_matrix_pads_the_translations_to_three_columns():
    simu = _injected("Elastic")
    Nn = simu.mesh.Nn

    matrix = simu.Results_displacement_matrix()

    assert matrix.shape == (Nn, 3)
    assert np.array_equal(matrix[:, :2], simu.displacement.reshape(Nn, 2))
    assert np.all(matrix[:, 2] == 0)


def test_Simu_Results_nodeFields_elementFields_are_available_results(subtests):
    for name, make in SIMULATIONS.items():
        simu = make()
        for details in (False, True):
            with subtests.test(simulation=name, details=details):
                nodesField, elementsField = simu.Results_nodeFields_elementFields(
                    details
                )
                assert set(nodesField + elementsField) <= set(simu.Results_Available())


def test_Simu__Check_dofs_rejects_an_unknown_direction():
    simu = _elastic_bar()

    simu._Check_dofs(simu.problemType, ["x", "y"])
    with pytest.raises(AssertionError, match="z"):
        simu._Check_dofs(simu.problemType, ["z"])


def test_Simu__Check_dim_mesh_material_rejects_a_model_of_another_dimension():
    with pytest.raises(AssertionError, match="same dimensions"):
        Simulations.Elastic(_bar(), Models.Elastic.Isotropic(3))


def test_Simu_model_is_the_given_model():
    material = Models.Elastic.Isotropic(2)

    assert Simulations.Elastic(_bar(), material).model is material


def test_Simu_mass_is_rho_times_the_volume():
    rho, thickness = 7.8, 3.0
    plate = Simulations.Elastic(
        _bar(), Models.Elastic.Isotropic(2, thickness=thickness)
    )
    block = Simulations.Elastic(_box(), Models.Elastic.Isotropic(3))
    beam = _beam(2)
    for simu in (plate, block, beam):
        simu.rho = rho

    assert rel_error(plate.mass, rho * L * H * thickness) < 1e-12
    assert rel_error(block.mass, rho * L * H * B) < 1e-12
    assert rel_error(beam.mass, rho * 10 * 1) < 1e-12
    assert _beam(1).mass is not None and _thermal().mass is None


def test_Simu_center_is_the_centroid():
    plate = Simulations.Elastic(_bar(), Models.Elastic.Isotropic(2))
    block = Simulations.Elastic(_box(), Models.Elastic.Isotropic(3))

    assert rel_error(plate.center, [L / 2, H / 2, 0]) < 1e-12
    assert rel_error(block.center, [L / 2, H / 2, B / 2]) < 1e-12
    assert rel_error(_beam(2).center, [5, 0, 0]) < 1e-12
    assert _thermal().center is None


def test_Simu_folder_is_created_and_can_be_unset(tmp_path):
    simu = _elastic_bar()
    folder = str(tmp_path / "run")

    simu.folder = folder
    simu.Save_Iter()
    simu.folder = ""
    simu.Save_Iter()

    assert os.path.isdir(folder)
    assert all("indexMesh" in simu.Get_results(i) for i in range(2))
    with pytest.raises(AssertionError):
        simu.folder = None


def test_Simu_Niter_counts_the_saved_iterations():
    simu = _thermal()
    assert simu.Niter == 0

    for _ in range(3):
        simu.Save_Iter()

    assert simu.Niter == 3


def _assert_column_covers_the_lagrange_rows(field: str) -> None:
    """`field` as a CSR column is the dof vector, padded by one row per multiplier."""
    simu = _beam(2)
    nodes = simu.mesh.Nodes_Point((9, 0), (10, 0))
    simu._Bc_Add_Lagrange(_tie(simu, nodes, "x"))
    _inject(simu)
    getter = getattr(simu, field)

    values = getter(simu.problemType)
    column = getter(simu.problemType, asCsrMatrix=True)

    assert column.shape == (values.size + simu._Bc_Lagrange_dim(), 1)
    assert values.any()
    assert np.array_equal(column.toarray()[: values.size, 0], values)


def test_Simu__Get_u_n_as_a_column_covers_the_lagrange_rows():
    _assert_column_covers_the_lagrange_rows("_Get_u_n")


def test_Simu__Get_v_n_as_a_column_covers_the_lagrange_rows():
    _assert_column_covers_the_lagrange_rows("_Get_v_n")


def test_Simu__Get_a_n_as_a_column_covers_the_lagrange_rows():
    _assert_column_covers_the_lagrange_rows("_Get_a_n")


def test_Simu_Get_lb_ub_bounds_only_a_bound_constrained_damage():
    simu = _phasefield()
    assert [b.size for b in simu.Get_lb_ub(simu.ProblemTypes.damage)] == [0, 0]
    simu.phaseFieldModel.solver = simu.phaseFieldModel.SolverType.BoundConstrain
    solver = simu.solver

    lb, ub = simu.Get_lb_ub(simu.ProblemTypes.damage)

    assert lb.size == ub.size == simu.mesh.Nn and np.all(ub == 1)
    assert simu.solver == solver
    assert [b.size for b in _thermal().Get_lb_ub("thermal")] == [0, 0]


def test_Simu_problemType_is_the_first_problem_type():
    assert _elastic_bar().problemType == "elastic"


def test_Simu_algo_follows_the_solver_setters():
    simu = _thermal()
    assert simu.algo == AlgoType.elliptic

    simu.Solver_Set_Parabolic_Algorithm(0.1)
    assert simu.algo == AlgoType.parabolic
    simu.Solver_Set_Hyperbolic_Algorithm(0.1, AlgoType.hht)
    assert simu.algo == AlgoType.hht
    simu.Solver_Set_Elliptic_Algorithm()
    assert simu.algo == AlgoType.elliptic


def test_Simu_inDim_is_the_space_the_unknowns_live_in():
    assert _thermal().inDim == 2
    assert Simulations.Thermal(_box(), Models.Thermal(1)).inDim == 3
    assert _beam(1).inDim == 2 and _beam(3).inDim == 3


TRANSFORMS = {
    "rotate": lambda mesh: mesh.Rotate(45, mesh.center),
    "translate": lambda mesh: mesh.Translate(dy=-10),
    "symmetry": lambda mesh: mesh.Symmetry(mesh.center, (1, 0)),
}


def test_Simu_mesh_transform_flags_reassembly(subtests):
    for name, transform in TRANSFORMS.items():
        with subtests.test(transform=name):
            simu = _thermal()
            simu.Get_K_C_M_F()

            transform(simu.mesh)

            assert simu.needUpdate


def test_Simu_mesh_change_resets_the_conditions_and_the_solution():
    simu = _injected("Thermal")
    simu.Get_K_C_M_F()

    simu.mesh = _box()

    assert simu.needUpdate
    assert simu.Bc_Dirichlet == []
    assert np.all(simu.thermal == 0) and simu.thermal.size == simu.mesh.Nn


def test_Simu_Nmesh_counts_the_meshes_set():
    simu = _thermal()

    simu.mesh = _box()
    simu.mesh = simu.mesh

    assert simu.Nmesh == 3


def test_Simu_isGathered_in_serial():
    assert _thermal().isGathered


def test_Simu__Gather_keeps_the_mesh_in_serial():
    simu = _thermal()
    mesh = simu.mesh

    simu._Gather()

    assert simu.mesh is mesh


def test_Simu_dim_is_the_model_dimension():
    assert _elastic_bar().dim == 2
    assert _thermal().dim == 1
    assert _beam(3).dim == 3


def test_Simu_thickness_scales_only_plane_problems():
    plate = Simulations.Elastic(_bar(), Models.Elastic.Isotropic(2, thickness=3.0))
    thermal = Simulations.Thermal(_bar(), Models.Thermal(1, thickness=3.0))
    mesh = _bar()
    weakForms = Models.WeakForms(
        Field(mesh.groupElem, 1), laplacian_form(), thickness=3.0
    )

    assert plate.thickness == thermal.thickness == 3.0
    assert Simulations.WeakForms(mesh, weakForms).thickness == 3.0
    assert Simulations.Elastic(_box(), Models.Elastic.Isotropic(3)).thickness == 1.0
    assert _beam(2).thickness == 1.0


def test_Simu_Assembly_is_what_Get_K_C_M_F_returns():
    simu = _elastic_bar()

    assembled = simu.Assembly(simu.problemType)

    for matrix, expected in zip(assembled, simu.Get_K_C_M_F()):
        assert (matrix != expected).nnz == 0


def test_Simu_solver_keeps_its_value_on_an_unknown_one():
    simu = _thermal()

    simu.solver = "unknown"

    assert simu.solver == SolverType.scipy


def test_Simu_Solver_Set_Elliptic_Algorithm_solves_K_u_F():
    simu = _thermal(transient=True)
    simu.Solver_Set_Elliptic_Algorithm()

    simu.Solve()

    expected = _reference.slab_temperature(simu.mesh.coord[:, 0], L, 0, 1)
    assert rel_error(simu.thermal, expected) < 1e-12


def _insulated_heating(alpha: float) -> Simulations.Thermal:
    simu = Simulations.Thermal(_bar(), Models.Thermal(1.0, 3.0))
    simu.rho = 2.0
    simu.solver = SolverType.scipy
    simu.Solver_Set_Parabolic_Algorithm(0.1, alpha)
    simu.add_volumeLoad(simu.mesh.nodes, [12.0], ["t"])
    return simu


def test_Simu_Solver_Set_Parabolic_Algorithm_heats_an_insulated_body_exactly(
    subtests,
):
    """From rest, c·ṫ = r gives t_n = (n - 1 + alpha)·dt·r / (rho c) under the alpha scheme."""
    rate = 12.0 / (2.0 * 3.0)
    for alpha in (0.5, 1.0):
        simu = _insulated_heating(alpha)
        for n in range(1, 4):
            with subtests.test(alpha=alpha, step=n):
                simu.Solve()

                assert rel_error(simu.thermal, (n - 1 + alpha) * 0.1 * rate) < 1e-10
                assert rel_error(simu.thermalDot, rate) < 1e-10


def test_Simu_Solver_Set_Parabolic_Algorithm_rejects_a_non_positive_step():
    with pytest.raises(AssertionError, match="Time increment"):
        _thermal().Solver_Set_Parabolic_Algorithm(0.0)


def test_Simu_Solver_Set_Hyperbolic_Algorithm_keeps_an_equilibrium_at_rest(subtests):
    for algo in AlgoType.Get_Hyperbolic_Types():
        with subtests.test(algo=str(algo)):
            simu = _elastic_bar()
            static = simu.Solve().copy()

            simu.Solver_Set_Hyperbolic_Algorithm(0.1, algo, alpha=0.1)
            simu.Solve()

            assert rel_error(simu.displacement, static) < 1e-10
            assert rel_error(simu.speed, 0.0, scale=UD / 0.1) < 1e-10


DT, G = 0.1, -3.0
EXACT = lambda n: (G * (n * DT) ** 2 / 2, G * n * DT, G)  # noqa: E731
FREE_FALLS = {
    AlgoType.newmark: EXACT,
    AlgoType.midpoint: EXACT,
    AlgoType.hht: EXACT,
    AlgoType.hht_newmark: EXACT,
    AlgoType.euler_implicit: lambda n: (G * DT**2 * n * (n + 1) / 2, G * n * DT, G),
    AlgoType.euler_explicit: lambda n: (G * DT**2 * n * (n - 1) / 2, G * n * DT, G),
}


def test_Simu_Solver_Set_Hyperbolic_Algorithm_free_fall_follows_the_scheme(subtests):
    """An unsupported body under a uniform body force, from rest with its acceleration: second-order schemes are exact on a quadratic motion, the Euler ones follow their own recurrence."""
    mesh = _bar()
    zeros = np.zeros(mesh.Nn * 2)
    for algo, exact in FREE_FALLS.items():
        with subtests.test(algo=str(algo)):
            simu = Simulations.Elastic(mesh, Models.Elastic.Isotropic(2, E, v))
            simu.rho = 2.0
            simu.solver = SolverType.scipy
            simu.Solver_Set_Hyperbolic_Algorithm(DT, algo, alpha=0.1)
            simu.add_volumeLoad(mesh.nodes, [2.0 * G], ["y"])
            accel0 = np.tile([0, G], mesh.Nn)
            simu._Set_solutions(simu.problemType, zeros, zeros, accel0)

            for n in range(1, 4):
                simu.Solve()

                u, speed, accel = exact(n)
                assert rel_error(simu.Result("uy"), u, scale=abs(G) * DT**2) < 1e-10
                assert rel_error(simu.Result("vy"), speed, scale=abs(G) * DT) < 1e-10
                assert rel_error(simu.Result("ay"), accel) < 1e-10


def test_Simu_Solver_Set_Hyperbolic_Algorithm_rejects_what_its_schemes_cannot_do():
    simu = _elastic_bar()
    with pytest.raises(AssertionError, match="algo"):
        simu.Solver_Set_Hyperbolic_Algorithm(0.1, AlgoType.parabolic)
    with pytest.raises(AssertionError, match="Time increment"):
        simu.Solver_Set_Hyperbolic_Algorithm(0.0)
    with pytest.raises(AssertionError, match="1/3"):
        simu.Solver_Set_Hyperbolic_Algorithm(0.1, AlgoType.hht_newmark, alpha=0.5)
    with pytest.raises(AssertionError):
        simu.Solver_Set_Hyperbolic_Algorithm(0.1, AlgoType.hht, alpha=1.0)
    with pytest.raises(AssertionError, match="linear"):
        _hyperelastic(False).Solver_Set_Hyperbolic_Algorithm(
            0.1, AlgoType.euler_explicit
        )


def _states(simu: Simulations._Simu) -> tuple[np.ndarray, ...]:
    """Distinct u_n, v_n, a_n set on `simu`, and a next iterate u_np1."""
    n = simu.mesh.Nn * simu.Get_dof_n()
    u_n, v_n, a_n, u_np1 = np.random.default_rng(0).standard_normal((4, n))
    simu._Set_solutions(simu.problemType, u_n, v_n, a_n)
    return u_n, v_n, a_n, u_np1


def test_Simu_Get_u_v_a_is_the_solution_alone_when_static():
    simu = _injected("Thermal")

    u, speed, accel = simu.Get_u_v_a()

    assert np.array_equal(u, simu.thermal) and speed is None and accel is None


def test_Simu_Get_u_v_a_evaluates_the_newton_iterate_at_the_scheme_point(
    monkeypatch,
):
    simu = _hyperelastic(False)
    simu.Solver_Set_Hyperbolic_Algorithm(0.1, AlgoType.midpoint)
    u_n, v_n, a_n, u_np1 = _states(simu)
    monkeypatch.setattr(
        simu, "_Solver_Get_Newton_Raphson_current_solution", lambda: u_np1
    )

    u_t, _, _ = simu.Get_u_v_a()

    assert rel_error(u_t, (u_n + u_np1) / 2) < 1e-15


def test_Simu__Solver_Evaluate_u_v_a_for_time_scheme_midpoint_averages():
    simu = _elastic_bar()
    simu.Solver_Set_Hyperbolic_Algorithm(DT, AlgoType.midpoint)
    u_n, v_n, a_n, u_np1 = _states(simu)

    u_t, v_t, a_t = simu._Solver_Evaluate_u_v_a_for_time_scheme(simu.problemType, u_np1)

    v_np1 = 2 / DT * (u_np1 - u_n) - v_n
    a_np1 = 2 / DT * (v_np1 - v_n) - a_n
    assert rel_error(u_t, (u_n + u_np1) / 2) < 1e-14
    assert rel_error(v_t, (v_n + v_np1) / 2) < 1e-14
    assert rel_error(a_t, (a_n + a_np1) / 2) < 1e-14


def test_Simu__Solver_Evaluate_u_v_a_for_time_scheme_hht_shifts_every_field():
    alpha = 0.2
    simu = _elastic_bar()
    simu.Solver_Set_Hyperbolic_Algorithm(DT, AlgoType.hht, alpha=alpha)
    u_n, v_n, a_n, u_np1 = _states(simu)

    u_t, v_t, a_t = simu._Solver_Evaluate_u_v_a_for_time_scheme(simu.problemType, u_np1)
    _, v_np1, a_np1 = simu._Solver_Update_solutions(simu.problemType, u_np1)

    assert rel_error(u_t, (1 - alpha) * u_np1 + alpha * u_n) < 1e-14
    assert rel_error(v_t, (1 - alpha) * v_np1 + alpha * v_n) < 1e-14
    assert rel_error(a_t, (1 - alpha) * a_np1 + alpha * a_n) < 1e-14


def test_Simu__Solver_Evaluate_u_v_a_for_time_scheme_needs_a_time_scheme():
    simu = _elastic_bar()

    with pytest.raises(AssertionError, match="algo"):
        simu._Solver_Evaluate_u_v_a_for_time_scheme(simu.problemType, simu.displacement)


BETA, GAMMA = 0.25, 0.5
ALPHA_HHT, ALPHA_DOYEN = 0.1, 1 / 6
BETA_DOYEN, GAMMA_DOYEN = (1 + ALPHA_DOYEN) ** 2 / 4, 0.5 + ALPHA_DOYEN
COEFS = {
    "newmark": (AlgoType.newmark, 0.0, (1, GAMMA / (BETA * DT), 1 / (BETA * DT**2))),
    "hht": (
        AlgoType.hht,
        ALPHA_HHT,
        tuple(
            (1 - ALPHA_HHT) * c for c in (1, GAMMA / (BETA * DT), 1 / (BETA * DT**2))
        ),
    ),
    "midpoint": (AlgoType.midpoint, 0.0, (0.5, 1 / DT, 2 / DT**2)),
    "hht_newmark": (
        AlgoType.hht_newmark,
        ALPHA_DOYEN,
        (
            1 - ALPHA_DOYEN,
            GAMMA_DOYEN / (BETA_DOYEN * DT),
            1 / (BETA_DOYEN * DT**2),
        ),
    ),
    "euler_implicit": (AlgoType.euler_implicit, 0.0, (1, 1 / DT, 1 / DT**2)),
    "euler_explicit": (AlgoType.euler_explicit, 0.0, (0, 0, 1)),
}


def test_Simu__Solver_Get_K_C_M_coefs_for_time_scheme_are_the_chain_rule_factors(
    subtests,
):
    simu = _elastic_bar()
    for name, (algo, alpha, expected) in COEFS.items():
        with subtests.test(algo=name):
            simu.Solver_Set_Hyperbolic_Algorithm(DT, algo, alpha=alpha)

            coefs = simu._Solver_Get_K_C_M_coefs_for_time_scheme()

            assert np.allclose(coefs, expected, rtol=1e-14)


def test_Simu__Solver_Get_K_C_M_coefs_for_time_scheme_of_the_parabolic_scheme():
    simu = _thermal()
    simu.Solver_Set_Parabolic_Algorithm(DT, alpha=0.5)

    assert simu._Solver_Get_K_C_M_coefs_for_time_scheme() == (1, 1 / (0.5 * DT), 0)


def test_Simu__Solver_Set_Newton_Raphson_Algorithm_solves_a_linear_problem():
    linear = _elastic_bar()
    newton = _elastic_bar()

    newton._Solver_Set_Newton_Raphson_Algorithm()

    assert newton.isNonLinear
    assert rel_error(newton.Solve(), linear.Solve()) < 1e-10
    with pytest.raises(AssertionError, match="maxIter"):
        newton._Solver_Set_Newton_Raphson_Algorithm(maxIter=1)
    with pytest.raises(AssertionError, match="absTol"):
        newton._Solver_Set_Newton_Raphson_Algorithm(absTol=2.0)


def test_Simu_isNonLinear_per_simulation():
    assert not _elastic_bar().isNonLinear
    assert _hyperelastic(False).isNonLinear


def test_Simu_Solve_returns_the_solution(subtests):
    """On `_Simu` and on the subclasses overriding it."""
    for name in ("Elastic", "InElastic"):
        if name not in SIMULATIONS:
            continue
        with subtests.test(simulation=name):
            simu = SIMULATIONS[name]()

            u = simu.Solve()

            assert u.any() and np.array_equal(u, simu.displacement)
    with subtests.test(simulation="PhaseField"):
        simu = _phasefield()

        u, d, converged = simu.Solve()

        assert np.array_equal(u, simu.displacement)
        assert np.array_equal(d, simu.damage)
        assert u.any()
        assert converged


def test_Simu__Set_solutions_sets_the_given_fields_only():
    simu = _elastic_bar(dynamic=True)
    u_n, v_n, a_n, _ = _states(simu)

    simu._Set_solutions(simu.problemType, 2 * u_n)

    assert np.array_equal(simu.displacement, 2 * u_n)
    assert np.array_equal(simu.speed, v_n) and np.array_equal(simu.accel, a_n)
    with pytest.raises(AssertionError, match="size"):
        simu._Set_solutions(simu.problemType, u_n[1:])


def test_Simu__Solver_Solve_problemType_solves_one_problem():
    simu = _phasefield()

    u = simu._Solver_Solve_problemType(simu.ProblemTypes.elastic)

    assert np.abs(u).max() > 0 and np.array_equal(u, simu.displacement)
    assert np.all(simu.damage == 0)


def test_Simu__Solver_Update_solutions_applies_the_newmark_corrector():
    simu = _elastic_bar()
    simu.Solver_Set_Hyperbolic_Algorithm(DT)
    u_n, v_n, a_n, u_np1 = _states(simu)

    u, speed, accel = simu._Solver_Update_solutions(simu.problemType, u_np1)

    predictor = u_n + DT * v_n + DT**2 / 2 * (1 - 2 * BETA) * a_n
    expected_a = (u_np1 - predictor) / (BETA * DT**2)
    assert u is u_np1
    assert rel_error(accel, expected_a) < 1e-12
    expected_v = v_n + DT * ((1 - GAMMA) * a_n + GAMMA * expected_a)
    assert rel_error(speed, expected_v) < 1e-12


def test_Simu__Solver_Update_solutions_leaves_v_and_a_when_static():
    simu = _elastic_bar()

    assert simu._Solver_Update_solutions(simu.problemType, np.zeros(1))[1:] == (
        None,
        None,
    )


def test_Simu__Solver_Get_Newton_Raphson_current_solution_is_the_last_iterate():
    simu = _hyperelastic(False)
    simu.Solve()

    current = simu._Solver_Get_Newton_Raphson_current_solution()

    assert rel_error(current, simu.displacement) < 1e-6
    with pytest.raises(AssertionError, match="linear"):
        _elastic_bar()._Solver_Get_Newton_Raphson_current_solution()


def test_Simu__Solver_Solve_Newton_Raphson_accepts_a_zero_first_residual(recwarn):
    """An unloaded solve starts at equilibrium: the relative norm must not divide by zero."""
    mesh = _bar()
    simu = Simulations.HyperElastic(mesh, Models.HyperElastic.NeoHookean(2, K=1e3))
    nodes0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    simu.add_dirichlet(nodes0, [0, 0], simu.Get_unknowns())

    u, newtonIter, _, norms = simu._Solver_Solve_Newton_Raphson()

    assert np.all(u == 0) and newtonIter == 1 and norms == [0.0]
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_Simu__Solver_Solve_Newton_Raphson_raises_when_it_does_not_converge():
    simu = _hyperelastic(False)
    simu._Solver_Set_Newton_Raphson_Algorithm(1e-14, 1e-14, 1e-14, maxIter=2)
    simu.Bc_Init()
    _reference.pull_bar(simu, 0.5)

    with pytest.raises(AssertionError, match="did not converged"):
        simu._Solver_Solve_Newton_Raphson()


def test_Simu__Solver_Apply_Neumann_is_the_load_when_static():
    simu = _elastic_bar()
    nodes = simu.mesh.Nodes_Conditions(lambda x, y, z: x == L)
    simu.add_neumann(nodes, [5.0], ["y"])

    b = simu._Solver_Apply_Neumann(simu.problemType)

    assert b.shape == (simu.mesh.Nn * 2, 1)
    assert np.array_equal(b.toarray().ravel(), simu.Bc_vector_Neumann())


def test_Simu__Solver_Apply_Dirichlet_holds_the_known_values():
    simu = _elastic_bar()
    b = simu._Solver_Apply_Neumann(simu.problemType)

    A, x = simu._Solver_Apply_Dirichlet(simu.problemType, b, "r1")

    assert (A != simu.Get_K_C_M_F()[0]).nnz == 0
    assert np.array_equal(x.toarray().ravel(), simu.Bc_vector_Dirichlet())


INVALID_PETSC = {
    "unknown ksp": (("unknown", "gamg", "petsc"), "kspType"),
    "unknown pc": (("cg", "unknown", "petsc"), "pcType"),
    "unknown solver": (("cg", "gamg", "unknown"), "solverType"),
    "lu iterated": (("cg", "lu", "petsc"), "preonly"),
    "backend without factorization": (("cg", "gamg", "mumps"), "requires"),
}


def test_Simu__Solver_Set_PETSc4Py_Options_rejects_an_invalid_combination(subtests):
    simu = _thermal()
    for name, (options, match) in INVALID_PETSC.items():
        with subtests.test(options=name):
            with pytest.raises(ValueError, match=match):
                simu._Solver_Set_PETSc4Py_Options(*options)


def test_Simu__Solver_Set_PETSc4Py_Options_sets_one_or_every_problem():
    simu = _phasefield()

    simu._Solver_Set_PETSc4Py_Options("gmres", "bjacobi")
    simu._Solver_Set_PETSc4Py_Options("preonly", "lu", problemType="damage")

    assert simu._Solver_Get_PETSc4Py_Options("elastic") == ("gmres", "bjacobi", "petsc")
    assert simu._Solver_Get_PETSc4Py_Options("damage") == ("preonly", "lu", "petsc")


def test_Simu__Solver_Get_PETSc4Py_Options_defaults_per_problem_size():
    assert _thermal()._Solver_Get_PETSc4Py_Options() == ("cg", "gamg", "petsc")
    assert _elastic_bar()._Solver_Get_PETSc4Py_Options()[:2] == ("preonly", "lu")


def test_Simu_Get_dofs_are_every_dof_in_serial():
    simu = _phasefield()

    assert np.array_equal(simu.Get_dofs("elastic"), np.arange(simu.mesh.Nn * 2))
    assert np.array_equal(simu.Get_dofs("damage"), np.arange(simu.mesh.Nn))


# ----------------------------------------------
# Boundary conditions
# ----------------------------------------------


def _tie(simu: Simulations._Simu, nodes: np.ndarray, unknown: str) -> LagrangeCondition:
    """u(nodes[0]) - u(nodes[1]) = 0."""
    dofs = simu.Bc_dofs_nodes(nodes, [unknown])
    return LagrangeCondition(
        simu.problemType,
        nodes,
        dofs,
        [unknown],
        np.array([0.0]),
        np.array([1.0, -1.0]),
        "tied",
    )


def _slab_corners(simu: Simulations._Simu) -> np.ndarray:
    """A bottom node at x = L/4 and a top one at x = 3L/4."""
    mesh = simu.mesh
    nodeA = mesh.Nodes_Point((L / 4, 0))
    nodeB = mesh.Nodes_Point((3 * L / 4, H))
    return np.concatenate([nodeA, nodeB])


def test_Simu__Bc_Add_Lagrange_ties_two_temperatures():
    simu = _thermal()
    nodes = _slab_corners(simu)
    simu.Get_K_C_M_F()

    simu._Bc_Add_Lagrange(_tie(simu, nodes, "t"))
    simu.Solve()

    assert simu.needUpdate is False
    assert rel_error(simu.thermal[nodes[0]], simu.thermal[nodes[1]]) < 1e-12
    assert 0.25 < simu.thermal[nodes[0]] < 0.75


def test_Simu__Bc_Lagrange_dim_counts_the_multipliers():
    simu = _thermal()
    assert simu._Bc_Lagrange_dim() == 0

    simu._Bc_Add_Lagrange(_tie(simu, _slab_corners(simu), "t"))

    assert simu._Bc_Lagrange_dim() == 1 + simu.Bc_dofs_Dirichlet().size


def test_Simu_Bc_Init_clears_every_condition():
    simu = _beam(2)
    nodesTip = simu.mesh.Nodes_Point((10, 0))
    simu.add_connection(nodesTip, ["x"], "held")

    simu.Bc_Init()

    assert simu.Bc_Dirichlet == simu.Bc_Neuman == simu.Bc_Lagrange == []
    assert simu.Bc_Display == []


def test_Simu_Bc_Dirichlet_is_a_copy_of_the_conditions():
    simu = _thermal()

    simu.Bc_Dirichlet.clear()

    assert [bc.description for bc in simu.Bc_Dirichlet] == ["Dirichlet "] * 2


def test_Simu_Bc_Neuman_is_a_copy_of_the_conditions():
    simu = _beam(2)

    simu.Bc_Neuman.clear()

    assert len(simu.Bc_Neuman) == 1


def test_Simu_Bc_Lagrange_is_a_copy_of_the_conditions():
    simu = _thermal()
    simu._Bc_Add_Lagrange(_tie(simu, _slab_corners(simu), "t"))

    simu.Bc_Lagrange.clear()

    assert [bc.description for bc in simu.Bc_Lagrange] == ["tied"]


def test_Simu_Bc_Display_is_a_copy_of_the_conditions():
    simu = _thermal()
    simu._Bc_Add_Display(simu.mesh.nodes[:2], ["t"], "shown")

    simu.Bc_Display.clear()

    assert [bc.description for bc in simu.Bc_Display] == ["shown"]


def test_Simu__Bc_Add_Display_marks_the_nodes():
    simu = _elastic_bar()
    nodes = simu.mesh.nodes[:2]

    simu._Bc_Add_Display(nodes, ["y"], "shown")

    bc = simu.Bc_Display[0]
    assert np.array_equal(bc.dofs, simu.Bc_dofs_nodes(nodes, ["y"]))


def test_Simu_Bc_vector_Dirichlet_spreads_the_values_over_the_dofs():
    simu = _thermal()

    vector = simu.Bc_vector_Dirichlet()

    nodesHot = simu.mesh.Nodes_Conditions(lambda x, y, z: x == L)
    assert vector.shape == (simu.mesh.Nn,)
    assert vector.sum() == nodesHot.size and np.all(vector[nodesHot] == 1)


def test_Simu_Bc_vector_Neumann_spreads_the_loads_over_the_dofs():
    simu = _beam(2)

    vector = simu.Bc_vector_Neumann()

    assert vector.shape == (simu.mesh.Nn * 3,) and vector.sum() == -2


def test_Simu_Bc_dofs_Dirichlet_are_the_constrained_dofs():
    simu = _thermal()

    dofs = simu.Bc_dofs_Dirichlet()

    nodes = simu.mesh.Nodes_Conditions(lambda x, y, z: (x == 0) | (x == L))
    assert np.array_equal(np.sort(dofs), np.sort(nodes))


def test_Simu_Bc_values_Dirichlet_follow_their_dofs():
    simu = _thermal()

    values = simu.Bc_values_Dirichlet()

    x = simu.mesh.coord[simu.Bc_dofs_Dirichlet(), 0]
    assert np.array_equal(values, x / L)


def test_Simu_Bc_dofs_Neumann_are_the_loaded_dofs():
    simu = _beam(2)

    dofs = simu.Bc_dofs_Neumann()

    nodesTip = simu.mesh.Nodes_Point((10, 0))
    assert np.array_equal(dofs, simu.Bc_dofs_nodes(nodesTip, ["x", "y"]))


def test_Simu_Bc_values_Neumann_follow_their_dofs():
    assert np.array_equal(_beam(2).Bc_values_Neumann(), [-1.0, -1.0])


def test_Simu_Bc_dofs_known_unknown_partition_the_dofs():
    simu = _elastic_bar()

    known, unknown = simu.Bc_dofs_known_unknown(simu.problemType)

    assert np.array_equal(known, np.unique(simu.Bc_dofs_Dirichlet()))
    assert np.array_equal(np.sort(np.concatenate([known, unknown])), simu.Get_dofs())


def test_Simu_Bc_dofs_nodes_numbers_node_by_node():
    simu = _beam(2)

    dofs = simu.Bc_dofs_nodes(np.array([2, 5]), ["y", "rz"])

    assert np.array_equal(dofs, [2 * 3 + 1, 2 * 3 + 2, 5 * 3 + 1, 5 * 3 + 2])
    assert simu.Bc_dofs_nodes(np.array([], dtype=int), ["y"]).size == 0


def _box_simu(elemType: ElemType, thickness: float = 3.0) -> Simulations.Elastic:
    domain = Domain((0, 0), (L, H), H)
    if elemType in ElemType.Get_2D():
        mesh = domain.Mesh_2D([], elemType)
    else:
        mesh = domain.Mesh_Extrude([], [0, 0, B], [2], elemType)
    material = Models.Elastic.Isotropic(mesh.dim, E, v, thickness=thickness)
    simu = Simulations.Elastic(mesh, material)
    simu.solver = SolverType.scipy
    return simu


def _end_face(simu: Simulations._Simu) -> np.ndarray:
    return simu.mesh.Nodes_Conditions(lambda x, y, z: x == L)


def _total(simu: Simulations._Simu, unknown: str) -> float:
    """Sum of the Neumann loads on `unknown`."""
    index = simu.Get_unknowns().index(unknown)
    return float(simu.Bc_vector_Neumann()[index :: simu.Get_dof_n()].sum())


DIRICHLET_VALUES = {
    "constant": 2.0,
    "function": lambda x, y, z: 1 + x + 2 * y,
}


def test_Simu_add_dirichlet_imposes_the_value_exactly(subtests):
    for elemType in ElemType.Get_2D() + ElemType.Get_3D():
        simu = _box_simu(elemType)
        nodes = _end_face(simu)
        dofs = simu.Bc_dofs_nodes(nodes, ["x"])
        coord = simu.mesh.coord[nodes]
        for name, value in DIRICHLET_VALUES.items():
            with subtests.test(elemType=str(elemType), value=name):
                simu.Bc_Init()

                simu.add_dirichlet(
                    nodes, [value, 0, 0][: simu.dim], simu.Get_unknowns()
                )

                expected = value(*coord.T) if callable(value) else value
                imposed = simu.Bc_vector_Dirichlet()[dofs]
                assert np.array_equal(imposed, np.broadcast_to(expected, nodes.shape))


def test_Simu_add_dirichlet_takes_one_value_per_node():
    simu = _thermal()
    nodes = _end_face(simu)
    values = np.linspace(1, 2, nodes.size)

    simu.Bc_Init()
    simu.add_dirichlet(nodes, [values], ["t"])

    assert np.array_equal(simu.Bc_values_Dirichlet(), values)


def test_Simu_add_dirichlet_ignores_empty_nodes_and_rejects_mismatched_values():
    simu = _elastic_bar()
    simu.Bc_Init()

    simu.add_dirichlet(np.array([], dtype=int), [0], ["x"])
    assert simu.Bc_Dirichlet == []
    with pytest.raises(AssertionError, match="len"):
        simu.add_dirichlet(_end_face(simu), [0, 0], ["x"])
    with pytest.raises(AssertionError, match="problem"):
        simu.add_dirichlet(_end_face(simu), [0], ["x"], problemType="thermal")


def test_Simu_add_neumann_shares_the_force_between_the_nodes(subtests):
    for elemType in ElemType.Get_2D() + ElemType.Get_3D():
        with subtests.test(elemType=str(elemType)):
            simu = _box_simu(elemType)
            nodes = _end_face(simu)

            simu.add_neumann(nodes, [-7.0], ["y"])

            assert rel_error(_total(simu, "y"), -7.0) < 1e-12
            assert rel_error(simu.Bc_values_Neumann(), -7.0 / nodes.size) < 1e-12


LINE_LOADS = {
    "constant": (5.0, 5.0 * H),
    "function": (lambda x, y, z: 3 * y, 3 * H**2 / 2),
}


def test_Simu_add_lineLoad_sums_to_the_load_times_the_length(subtests):
    """On the edges of a 2D mesh."""
    for elemType in ElemType.Get_2D():
        simu = _box_simu(elemType)
        nodes = _end_face(simu)
        for name, (value, total) in LINE_LOADS.items():
            with subtests.test(elemType=str(elemType), value=name):
                simu.Bc_Init()

                simu.add_lineLoad(nodes, [value], ["y"])

                assert rel_error(_total(simu, "y"), total) < 1e-12


def test_Simu_add_lineLoad_interpolates_nodal_values():
    """y at the nodes, integrated along the edge: H²/2."""
    simu = _box_simu(ElemType.QUAD8)
    nodes = _end_face(simu)

    simu.add_lineLoad(nodes, [simu.mesh.coord[nodes, 1]], ["y"])

    assert rel_error(_total(simu, "y"), H**2 / 2) < 1e-12


def test_Simu_add_surfLoad_sums_to_the_pressure_times_the_area(subtests):
    for elemType in ElemType.Get_2D() + ElemType.Get_3D():
        with subtests.test(elemType=str(elemType)):
            simu = _box_simu(elemType, thickness=3.0)
            nodes = _end_face(simu)
            area = H * (3.0 if simu.dim == 2 else B)

            simu.add_surfLoad(nodes, [4.0], ["x"])

            assert rel_error(_total(simu, "x"), 4.0 * area) < 1e-12


def test_Simu_add_pressureLoad_sums_to_the_pressure_times_the_area_along_the_normal(
    subtests,
):
    for elemType in ElemType.Get_2D() + ElemType.Get_3D():
        with subtests.test(elemType=str(elemType)):
            simu = _box_simu(elemType, thickness=3.0)
            nodes = _end_face(simu)
            area = H * (3.0 if simu.dim == 2 else B)

            simu.add_pressureLoad(nodes, 4.0)

            assert rel_error(abs(_total(simu, "x")), 4.0 * area) < 1e-12
            assert abs(_total(simu, "y")) < 1e-12 * area


@pytest.mark.xfail(
    strict=True,
    reason="pressure follows the boundary elements' own normals: inward on a counterclockwise 2D contour, outward on 3D faces",
)
def test_Simu_add_pressureLoad_pushes_the_same_way_in_2d_and_3d():
    plate = _box_simu(ElemType.QUAD4)
    block = _box_simu(ElemType.HEXA8)

    for simu in (plate, block):
        simu.add_pressureLoad(_end_face(simu), 1.0)

    assert np.sign(_total(plate, "x")) == np.sign(_total(block, "x"))


def test_Simu_add_pressureLoad_ignores_a_zero_pressure_and_a_1d_simulation():
    simu = _elastic_bar()
    simu.add_pressureLoad(_end_face(simu), 0.0)
    beam = _beam(1)
    beam.Bc_Init()
    beam.add_pressureLoad(beam.mesh.nodes, 1.0)

    assert simu.Bc_Neuman == beam.Bc_Neuman == []


def test_Simu_add_volumeLoad_sums_to_the_force_times_the_volume(subtests):
    for elemType in ElemType.Get_2D() + ElemType.Get_3D():
        with subtests.test(elemType=str(elemType)):
            simu = _box_simu(elemType, thickness=3.0)
            volume = L * H * (3.0 if simu.dim == 2 else B)

            simu.add_volumeLoad(simu.mesh.nodes, [-2.0], ["y"])

            assert rel_error(_total(simu, "y"), -2.0 * volume) < 1e-12


def test_Simu__Bc_Add_Neumann_adds_the_given_dofs_and_values():
    simu = _elastic_bar()
    nodes = np.array([3, 4])
    dofs = simu.Bc_dofs_nodes(nodes, ["y"])

    simu._Bc_Add_Neumann(simu.problemType, nodes, np.array([1.0, 2.0]), dofs, ["y"])

    assert np.array_equal(simu.Bc_dofs_Neumann(), dofs)
    assert np.array_equal(simu.Bc_values_Neumann(), [1.0, 2.0])


def test_Simu__Bc_Add_Dirichlet_adds_the_given_dofs_and_values():
    simu = _thermal()
    simu.Bc_Init()
    nodes = np.array([3, 4])

    simu._Bc_Add_Dirichlet(simu.problemType, nodes, np.array([1.0, 2.0]), nodes, ["t"])

    assert np.array_equal(simu.Bc_dofs_Dirichlet(), nodes)
    assert np.array_equal(simu.Bc_values_Dirichlet(), [1.0, 2.0])


# ----------------------------------------------
# Results
# ----------------------------------------------


def test_Simu__Results_Check_Available():
    simu = _thermal()

    assert simu._Results_Check_Available("thermal")
    assert not simu._Results_Check_Available("Svm")


def test_Simu_Results_Set_Iteration_Summary_prints_it(capsys):
    _thermal().Results_Set_Iteration_Summary("step 1")

    assert "step 1" in capsys.readouterr().out


def test_Simu_Results_Get_Iteration_Summary_is_unspecified_by_default():
    assert _thermal().Results_Get_Iteration_Summary() == "Unspecified."


def test_Simu_Results_Get_Iteration_Summary_of_PhaseField_reports_the_step(capsys):
    simu = _solved_phasefield()

    simu.Results_Set_Iteration_Summary(0, 1.0, "mm", percentage=0.5)

    summary = simu.Results_Get_Iteration_Summary()
    assert "convOption = 2" in summary and "1: 1.000mm" in summary
    assert "1: 1.000mm" in capsys.readouterr().out


def test_Simu_Results_Set_Bc_Summary_is_kept_by_PhaseField():
    simu = _phasefield()

    simu.Results_Set_Bc_Summary("pulled")

    assert simu.Results_Get_Bc_Summary() == "pulled"


def test_Simu_Results_Get_Bc_Summary_is_unspecified_by_default():
    simu = _thermal()
    simu.Results_Set_Bc_Summary()

    assert simu.Results_Get_Bc_Summary() == "Unspecified."


def test_Simu_Results_Reshape_values_moves_values_between_nodes_and_elements():
    """On 16 elements and 27 nodes, so that no node size is an element one."""
    domain = Domain((0, 0), (L, H), H / 2)
    mesh = domain.Mesh_2D([], ElemType.QUAD4, isOrganised=True)
    simu = _elastic_bar(mesh=mesh)
    x_n = mesh.coord[:, 0]

    assert (
        rel_error(simu.Results_Reshape_values(np.full(mesh.Ne, 5.0), True), 5.0) < 1e-15
    )
    x_e = simu.Results_Reshape_values(x_n, False)
    assert rel_error(x_e, mesh.coord[mesh.connect, 0].mean(1)) < 1e-15
    pairs = simu.Results_Reshape_values(np.ones((mesh.Ne, 2)), True)
    assert pairs.shape == (mesh.Nn, 2)
    assert simu.Results_Reshape_values(np.ones(mesh.Nn * 2), False).shape == (
        mesh.Ne * 2,
    )
    with pytest.raises(Exception, match="Unexpected"):
        simu.Results_Reshape_values(np.ones(mesh.Nn * mesh.Ne + 1), False)


# ----------------------------------------------
# Save, Load_Simu
# ----------------------------------------------


def test_Simu_Save_writes_the_simulation_its_meshes_and_a_summary(tmp_path):
    simu = _injected("Elastic")
    simu.Save_Iter()
    folder = str(tmp_path)

    path = simu.Save(folder, additionalInfos="pulled bar")

    assert path == os.path.join(folder, "simulation.pickle")
    assert os.path.isdir(os.path.join(folder, "Meshes"))
    with open(os.path.join(folder, "summary.txt"), encoding="utf8") as file:
        summary = file.read()
    assert "version" in summary and "pulled bar" in summary


def test_Load_Simu_restores_the_saved_results(tmp_path):
    simu = _injected("Elastic")
    simu.Save_Iter()
    simu.mesh = _bar()
    simu.Save_Iter()
    simu.Save(str(tmp_path), "bar")

    loaded = Simulations.Load_Simu(str(tmp_path), "bar")

    assert loaded.Niter == 2 and loaded.Nmesh == 2
    loaded.Set_Iter(0)
    assert rel_error(loaded.displacement, simu.Get_results(0)["displacement"]) < 1e-15


def test_Load_Simu_rejects_a_missing_simulation(tmp_path):
    with pytest.raises(AssertionError, match="cannot be found"):
        Simulations.Load_Simu(str(tmp_path))
