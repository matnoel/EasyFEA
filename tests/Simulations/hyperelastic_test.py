# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np
import pytest

from EasyFEA import AlgoType, ElemType, Mesh, Models, Simulations
from EasyFEA.FEM import FeArray, Kinematics, MatrixType, Operators
from EasyFEA.Geoms import Domain

from _numerics import rel_error
from ._simus import measure
from ._fields import BAR_MESHES, bar_mesh, component

StressType = Simulations.HyperElastic.StressType
L, H = 120.0, 13.0
LMBDA, MU = 1e5, 8e4
STRAIN = 1e-4


def _free_vibration_drift(
    algo: AlgoType,
    stress: StressType | None = None,
    **stressOptions,
) -> float:
    """max ``|KE + W - E0| / E0`` of a deflected NeoHookean cantilever of 4 QUAD4 released into free vibration.

    Loose ``absTol``: gonzalez's residual has a ~1e-9 round-off floor from the ΔW − s̄·Δe cancellation.
    """
    length, h, dt, nStep = 40.0, 10.0, 0.05, 50
    domain = Domain((0, 0), (length, h), h)
    mesh = domain.Mesh_2D([], ElemType.QUAD4, isOrganised=True)
    nodesClamped = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesTip = mesh.Nodes_Conditions(lambda x, y, z: x == length)
    material = Models.HyperElastic.NeoHookean(2, K=5.0e4)
    simu = Simulations.HyperElastic(mesh, material, absTol=1e-4)

    simu.add_dirichlet(nodesClamped, [0, 0], simu.Get_unknowns())
    simu.add_dirichlet(nodesTip, [-h / 3], ["y"])
    simu.Solve()
    simu.Save_Iter()

    simu.Bc_Init()
    simu.Solver_Set_Hyperbolic_Algorithm(dt, algo=algo)
    if stress is not None:
        simu.Solver_Set_Stress(stress, **stressOptions)
    simu.add_dirichlet(nodesClamped, [0, 0], simu.Get_unknowns())

    energies = [simu.Result("W")]  # at rest: KE = 0
    M = None
    for _ in range(nStep):
        simu.Solve()
        simu.Save_Iter()
        if M is None:
            M = simu.Get_K_C_M_F()[2]  # constant mass
        energies.append(simu.Calc_Energy(M, simu.speed) + simu.Result("W"))

    energy = np.array(energies)
    return float(np.abs(energy - energy[0]).max() / abs(energy[0]))


def _clamped_block(material=None) -> Simulations.HyperElastic:
    """4 QUAD4 in a row, clamped at x=0."""
    domain = Domain((0, 0), (20.0, 5.0), 5.0)
    mesh = domain.Mesh_2D([], ElemType.QUAD4, isOrganised=True)
    nodesClamped = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    if material is None:
        material = Models.HyperElastic.NeoHookean(2, K=5.0e4)
    simu = Simulations.HyperElastic(mesh, material, absTol=1e-4)
    simu.add_dirichlet(nodesClamped, [0, 0], simu.Get_unknowns())
    return simu


def _svk_bar(mesh: Mesh, strain: float = STRAIN) -> Simulations.HyperElastic:
    """Saint-Venant-Kirchhoff bar stretched along x alone (ux = strain·x), injected: never solved."""
    material = Models.HyperElastic.SaintVenantKirchhoff(mesh.dim, LMBDA, MU)
    simu = Simulations.HyperElastic(mesh, material)
    u = np.zeros((mesh.Nn, mesh.dim))
    u[:, 0] = strain * mesh.coord[:, 0]
    simu._Set_solutions(simu.problemType, u.ravel())
    return simu


def _Exx(strain: float) -> float:
    """Green-Lagrange Exx of that stretch, its only non-zero component."""
    return strain + strain**2 / 2


# ----------------------------------------------
# HyperElastic
# ----------------------------------------------


def test_HyperElastic_autodiff_law_assembles_like_the_shipped_one(monkeypatch):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    K, K1, K2 = 200.0, 30.0, 8.0

    def ciarlet_geymonat(C):
        I1, I3 = jnp.trace(C), jnp.linalg.det(C)
        I2 = (I1**2 - jnp.trace(C @ C)) / 2
        J1, J2, J = I1 * I3 ** (-1 / 3), I2 * I3 ** (-2 / 3), jnp.sqrt(I3)
        return K1 * (J1 - 3) + K2 * (J2 - 3) + K * (J - 1 - jnp.log(J))

    domain = Domain((0, 0), (1, 1), meshSize=0.5)
    mesh = domain.Mesh_Extrude([], [0, 0, 1], [2], ElemType.HEXA8)
    u = 0.05 * np.sin(mesh.coord).ravel()

    def assembled(material) -> tuple[np.ndarray, np.ndarray]:
        simu = Simulations.HyperElastic(mesh, material)
        monkeypatch.setattr(
            simu, "_Solver_Get_Newton_Raphson_current_solution", lambda: u
        )
        K, _, _, F = simu.Get_K_C_M_F()
        return K.toarray(), F.toarray()

    K_autodiff, F_autodiff = assembled(
        Models.HyperElastic.AutoDiff(3, ciarlet_geymonat)
    )
    K_shipped, F_shipped = assembled(
        Models.HyperElastic.CiarletGeymonat(3, K1=K1, K2=K2, K=K)
    )

    assert rel_error(K_autodiff, K_shipped) < 1e-10
    assert rel_error(F_autodiff, F_shipped) < 1e-10


def test_HyperElastic_Get_stressTypes_lists_every_stress():
    assert Simulations.HyperElastic.Get_stressTypes() == list(StressType)
    assert [str(s) for s in StressType] == ["pointwise", "gonzalez", "quadrature"]


def test_HyperElastic_matrixType_sets_the_rule_results_are_sampled_at(monkeypatch):
    simu = _svk_bar(bar_mesh("TRI3"))
    monkeypatch.setattr(
        simu, "_Solver_Get_Newton_Raphson_current_solution", lambda: simu.displacement
    )
    simu.Get_K_C_M_F()
    assert not simu.needUpdate
    assert simu.matrixType == MatrixType.rigi

    simu.matrixType = MatrixType.mass

    assert simu.needUpdate
    Exx = component(simu._Calc_GreenLagrange(), simu.mesh, 0, MatrixType.mass)
    assert rel_error(Exx, _Exx(STRAIN)) < 1e-10


def test_HyperElastic_material_is_the_given_model():
    material = Models.HyperElastic.NeoHookean(2, K=1e3)

    assert _clamped_block(material).material is material


def test_HyperElastic_material_eta_enters_the_residual_once(monkeypatch):
    """Turning viscosity off changes nothing else in the assembly, so the difference of the two ``F_e`` is the viscous residual alone."""
    eta = 100.0
    rng = np.random.default_rng(4)
    domain = Domain((0, 0), (60.0, 10.0), 10.0)
    mesh = domain.Mesh_2D([], ElemType.QUAD4, isOrganised=True)
    material = Models.HyperElastic.NeoHookean(2, K=5.0e4)
    material.eta = eta
    simu = Simulations.HyperElastic(mesh, material)
    simu.Solver_Set_Hyperbolic_Algorithm(0.05, algo=AlgoType.midpoint)

    # any configuration with a non-zero velocity will do; equilibrium is not needed
    problemType = simu.problemType
    n = mesh.Nn * 2
    simu._Set_solutions(
        problemType,
        rng.standard_normal(n) * 0.05,
        rng.standard_normal(n) * 0.5,
        rng.standard_normal(n) * 0.5,
    )
    u_np1 = rng.standard_normal(n) * 0.05
    monkeypatch.setattr(
        simu, "_Solver_Get_Newton_Raphson_current_solution", lambda: u_np1
    )

    groupElem = mesh.groupElem
    F_visco = simu._Construct_local_matrix_system(problemType)[groupElem][3]
    material.eta = 0.0
    F_plain = simu._Construct_local_matrix_system(problemType)[groupElem][3]
    material.eta = eta

    u_t, v_t, _ = simu._Solver_Evaluate_u_v_a_for_time_scheme(problemType, u_np1)
    kinematics = Kinematics(groupElem, u_t, MatrixType.rigi)
    _, R_e, _ = Operators.NonLinear.KelvinVoigtDamping(material, kinematics, v_t)

    assert np.abs(R_e).max() > 0
    assert rel_error(F_plain - F_visco, R_e) < 1e-10


def _dynamic() -> tuple[Simulations.HyperElastic, np.ndarray]:
    """A Newmark simulation with u, 2u and 3u injected; returns it and u."""
    simu = _clamped_block()
    simu.Solver_Set_Hyperbolic_Algorithm(0.1)
    u = np.linspace(0, 1, simu.mesh.Nn * 2)
    simu._Set_solutions(simu.problemType, u, 2 * u, 3 * u)
    return simu, u


def test_HyperElastic_displacement_is_the_dof_vector():
    simu, u = _dynamic()

    assert np.array_equal(simu.displacement, u)


def test_HyperElastic_speed_is_the_velocity_dof_vector():
    simu, u = _dynamic()

    assert np.array_equal(simu.speed, 2 * u)


def test_HyperElastic_accel_is_the_acceleration_dof_vector():
    simu, u = _dynamic()

    assert np.array_equal(simu.accel, 3 * u)


def test_HyperElastic_Solver_Set_Stress_gonzalez_conserves_the_energy():
    drift = _free_vibration_drift(AlgoType.midpoint, StressType.gonzalez)

    assert drift < 1e-6


def test_HyperElastic_Solver_Set_Stress_energyTol_controls_the_drift():
    loose = _free_vibration_drift(
        AlgoType.midpoint,
        StressType.quadrature,
        energyTol=1e-2,
    )
    tight = _free_vibration_drift(
        AlgoType.midpoint,
        StressType.quadrature,
        energyTol=1e-8,
    )

    assert tight < loose / 100
    assert tight < 1e-6


def test_HyperElastic_Solver_Set_Stress_gonzalez_rejects_a_non_midpoint_algo():
    simu = _clamped_block()
    simu.Solver_Set_Hyperbolic_Algorithm(0.05, algo=AlgoType.newmark)

    with pytest.raises(AssertionError, match="midpoint"):
        simu.Solver_Set_Stress(StressType.gonzalez)


def test_HyperElastic_Solver_Set_Stress_gonzalez_left_stale_is_rejected_at_solve():
    """Re-selecting another scheme after enabling it must not leave gonzalez active."""
    simu = _clamped_block()
    simu.Solver_Set_Hyperbolic_Algorithm(0.05, algo=AlgoType.midpoint)
    simu.Solver_Set_Stress(StressType.gonzalez)
    simu.Solver_Set_Hyperbolic_Algorithm(0.05, algo=AlgoType.newmark)

    with pytest.raises(AssertionError, match="midpoint"):
        simu.Solve()


def test_HyperElastic_Solver_Set_Stress_quadrature_is_rejected_at_a_static_solve():
    simu = _clamped_block()
    simu.Solver_Set_Stress(StressType.quadrature)

    with pytest.raises(AssertionError, match="dynamic"):
        simu.Solve()


def test_HyperElastic__nPts_e_counts_the_quadrature_points_per_element():
    simu = _clamped_block()
    assert simu._nPts_e is None

    simu.Solver_Set_Hyperbolic_Algorithm(0.05, algo=AlgoType.midpoint)
    simu.Solver_Set_Stress(StressType.quadrature, nPoints=5)
    nodesTip = simu.mesh.Nodes_Conditions(lambda x, y, z: x == 20.0)
    simu.add_neumann(nodesTip, [-10.0], ["y"])
    simu.Solve()
    simu.Save_Iter()

    assert simu._nPts_e.shape == (simu.mesh.Ne,)
    assert np.all(simu._nPts_e == 5)
    assert np.array_equal(simu.Get_results()["nPts_e"], simu._nPts_e)


def test_HyperElastic_stressType_is_the_selected_one():
    simu = _clamped_block()
    assert simu.stressType == StressType.pointwise

    simu.Solver_Set_Hyperbolic_Algorithm(0.05, algo=AlgoType.midpoint)
    simu.Solver_Set_Stress(StressType.gonzalez)

    assert simu.stressType == StressType.gonzalez


def test_HyperElastic__Calc_W_of_a_svk_stretch_is_half_lmbda_2mu_Exx2_V(subtests):
    """W, written in the invariants of C, cancels O(lmbda) terms down to lmbda·eps²: stretched 1 % for a 1e-10 floor."""
    strain = 1e-2
    for meshName in ("TRI3", "HEXA8+PRISM6"):
        with subtests.test(mesh=meshName):
            mesh = bar_mesh(meshName)
            simu = _svk_bar(mesh, strain)
            volume = measure(mesh)
            expected = (LMBDA + 2 * MU) * _Exx(strain) ** 2 / 2 * volume

            W_e = simu._Calc_W(returnScalar=False)

            assert rel_error(simu._Calc_W(), expected) < 1e-10
            assert W_e.shape == (mesh.Ne,)
            assert rel_error(W_e.sum(), expected) < 1e-10


def test_HyperElastic__Calc_GreenLagrange_of_a_svk_stretch_is_per_group(subtests):
    for meshName in BAR_MESHES:
        with subtests.test(mesh=meshName):
            mesh = bar_mesh(meshName)
            simu = _svk_bar(mesh)

            Exx = component(simu._Calc_GreenLagrange(), mesh, 0)

            assert rel_error(Exx, _Exx(STRAIN)) < 1e-10


def test_HyperElastic__Calc_SecondPiolaKirchhoff_of_a_svk_stretch_is_lmbda_trE_2mu_E(
    subtests,
):
    Exx = _Exx(STRAIN)
    for meshName in BAR_MESHES:
        with subtests.test(mesh=meshName):
            mesh = bar_mesh(meshName)
            simu = _svk_bar(mesh)

            stress = simu._Calc_SecondPiolaKirchhoff()

            Sxx = component(stress, mesh, 0)
            Syy = component(stress, mesh, 1)
            assert rel_error(Sxx, (LMBDA + 2 * MU) * Exx) < 1e-10
            assert rel_error(Syy, LMBDA * Exx) < 1e-10


def test_HyperElastic__Calc_SecondPiolaKirchhoff_adds_the_active_stress():
    """At rest only the active stress tau·T⊗T remains, along the fiber x."""
    tau = 3.0
    mesh = bar_mesh("TRI3")
    material = Models.HyperElastic.SaintVenantKirchhoff(2, LMBDA, MU)
    nPg = mesh.groupElem.Get_gauss(MatrixType.rigi).nPg
    fibers = np.zeros((mesh.Ne, nPg, 3))
    fibers[..., 0] = 1.0
    material.Set_active_stress_vec(FeArray.asfearray(fibers))
    material.active_stress = tau
    simu = Simulations.HyperElastic(mesh, material)

    stress = simu._Calc_SecondPiolaKirchhoff()

    assert rel_error(component(stress, mesh, 0), tau) < 1e-14
    assert np.abs(component(stress, mesh, 1)).max() == 0
