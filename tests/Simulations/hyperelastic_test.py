# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Simulation-level tests for HyperElastic dynamics.

The Gonzalez energy-momentum stress (``Solver_Set_Stress(StressType.gonzalez)``, on top of
``AlgoType.midpoint``) conserves the total
energy ``KE + W`` of a free (no external load, no damping) motion to round-off,
whereas ``newmark`` drifts by orders of magnitude more — observed here only through
the public simulation interface (solve loop + energy results).
"""

import numpy as np
import pytest

from EasyFEA import ElemType, Models, Simulations, AlgoType
from EasyFEA.Geoms import Domain
from EasyFEA.FEM import MatrixType, Operators
from EasyFEA.Models.HyperElastic._state import HyperElasticState


class TestGonzalezEnergyConservation:
    """A cantilever is statically deflected, then released and integrated with no
    external load and no viscosity, so the only forces are internal (hyperelastic) +
    inertia and total energy must stay constant. ``gonzalez`` keeps ``KE + W`` flat to
    round-off; ``newmark`` drifts by orders of magnitude more.
    """

    L, h = 60.0, 10.0
    dt, nStep = 0.05, 50

    def _energy_drift(self, algo, gonzalez: bool = False) -> float:
        L, h = self.L, self.h
        mesh = Domain((0, 0), (L, h), h / 2).Mesh_2D(
            [], ElemType.QUAD4, isOrganised=True
        )
        n0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
        nL = mesh.Nodes_Conditions(lambda x, y, z: x == L)

        # NeoHookean (genuinely nonlinear): Saint-Venant-Kirchhoff's energy is quadratic,
        # so the midpoint rule is already near-conservative there — a weak showcase.
        mat = Models.HyperElastic.NeoHookean(2, K=5.0e4)
        # gonzalez's discrete-gradient residual has a round-off floor (~1e-9 relative to
        # the force scale) from the ΔW − s̄·Δe cancellation → a relative-appropriate
        # (looser absolute) Newton tolerance, as in the FreeFEM reference.
        simu = Simulations.HyperElastic(mesh, mat, absTol=1e-4, verbosity=False)

        # static preload: clamp x=0, pull the tip down
        simu.add_dirichlet(n0, [0, 0], simu.Get_unknowns())
        simu.add_dirichlet(nL, [-h / 3], ["y"])
        simu.Solve()
        simu.Save_Iter()

        # release the tip (keep the clamp) → free vibration
        simu.Bc_Init()
        simu.Solver_Set_Hyperbolic_Algorithm(self.dt, algo=algo)
        if gonzalez:
            simu.Solver_Set_Stress(simu.StressType.gonzalez)
        simu.add_dirichlet(n0, [0, 0], simu.Get_unknowns())

        pt = simu.problemType
        energies = [float(simu._Calc_W())]  # t=0: at rest, KE = 0
        M = None
        for _ in range(self.nStep):
            simu.Solve()
            simu.Save_Iter()
            if M is None:
                _, _, M, _ = simu.Get_K_C_M_F(
                    pt
                )  # constant mass, assembled after step 1
            v = simu._Get_v_n(pt)
            energies.append(0.5 * float(v @ (M @ v)) + float(simu._Calc_W()))

        E = np.array(energies)
        return float(np.abs(E - E[0]).max() / abs(E[0]))

    def test_gonzalez_conserves_energy(self):
        """``gonzalez`` keeps KE + W constant to ~round-off over the run."""
        drift = self._energy_drift(AlgoType.midpoint, gonzalez=True)
        assert drift < 1e-6, f"gonzalez energy drift {drift:.2e} (should be ~round-off)"

    def test_gonzalez_beats_newmark(self):
        """``newmark`` drifts; ``gonzalez`` drift is orders of magnitude smaller."""
        gonzalez = self._energy_drift(AlgoType.midpoint, gonzalez=True)
        newmark = self._energy_drift(AlgoType.newmark)
        assert newmark > 1e-3, f"newmark should drift measurably; got {newmark:.2e}"
        assert (
            gonzalez < newmark / 100
        ), f"gonzalez {gonzalez:.2e} should be ≪ newmark {newmark:.2e}"


class TestGonzalezRequiresMidpoint:
    """The gonzalez stress is only valid on top of ``AlgoType.midpoint``.

    The discrete gradient's conservation proof rests on the midpoint base point
    (``Δe = B(ū)·Δu`` holds exactly only for ``ū``), so pairing it with any other scheme
    would silently integrate the wrong physics — no crash, just a wrong answer.
    """

    @staticmethod
    def _simu():
        mesh = Domain((0, 0), (20.0, 5.0), 2.5).Mesh_2D(
            [], ElemType.QUAD4, isOrganised=True
        )
        n0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
        mat = Models.HyperElastic.NeoHookean(2, K=5.0e4)
        simu = Simulations.HyperElastic(mesh, mat, absTol=1e-4, verbosity=False)
        simu.add_dirichlet(n0, [0, 0], simu.Get_unknowns())
        return simu

    def test_stale_flag_is_rejected_at_solve(self):
        """Re-selecting another scheme after enabling it must not leave gonzalez active."""
        simu = self._simu()
        simu.Solver_Set_Hyperbolic_Algorithm(0.05, algo=AlgoType.midpoint)
        simu.Solver_Set_Stress(simu.StressType.gonzalez)
        # the docs tell users to re-call this when dt changes — here the algo changes too
        simu.Solver_Set_Hyperbolic_Algorithm(0.05, algo=AlgoType.newmark)
        with pytest.raises(AssertionError):
            simu.Solve()

    def test_rejects_non_midpoint_algo(self):
        """Enabling it on a non-midpoint scheme fails at once, not silently later."""
        simu = self._simu()
        simu.Solver_Set_Hyperbolic_Algorithm(0.05, algo=AlgoType.newmark)
        with pytest.raises(AssertionError):
            simu.Solver_Set_Stress(simu.StressType.gonzalez)


class TestThicknessInvariance:
    """With no external load, ``thickness`` scales the mass, the stiffness and the
    internal force uniformly, so it cancels: the free-vibration solution must be
    *identical* whatever the thickness.

    Regression guard — HyperElastic used to scale the mass (and the strain energy) by
    ``thickness`` but not the internal-force operator, so a ``thickness != 1`` silently
    made the body too heavy for its own stiffness and changed the dynamics, with no
    error raised.
    """

    L, h = 60.0, 10.0

    def _tip_trajectory(self, thickness: float, gonzalez: bool, nStep: int = 6):
        L, h = self.L, self.h
        mesh = Domain((0, 0), (L, h), h / 2).Mesh_2D(
            [], ElemType.QUAD4, isOrganised=True
        )
        n0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
        nL = mesh.Nodes_Conditions(lambda x, y, z: x == L)
        mat = Models.HyperElastic.NeoHookean(2, K=5.0e4, thickness=thickness)
        simu = Simulations.HyperElastic(mesh, mat, absTol=1e-4, verbosity=False)

        simu.add_dirichlet(n0, [0, 0], simu.Get_unknowns())
        simu.add_dirichlet(nL, [-h / 3], ["y"])
        simu.Solve()
        simu.Save_Iter()

        simu.Bc_Init()
        simu.Solver_Set_Hyperbolic_Algorithm(0.05, algo=AlgoType.midpoint)
        if gonzalez:
            simu.Solver_Set_Stress(simu.StressType.gonzalez)
        simu.add_dirichlet(n0, [0, 0], simu.Get_unknowns())

        traj = []
        for _ in range(nStep):
            simu.Solve()
            simu.Save_Iter()
            traj.append(simu.Result("uy")[nL].mean())
        return np.array(traj)

    @pytest.mark.parametrize("gonzalez", [False, True], ids=["midpoint", "gonzalez"])
    def test_free_vibration_is_thickness_invariant(self, gonzalez):
        """No external load ⇒ thickness cancels ⇒ same motion for any thickness."""
        ref = self._tip_trajectory(1.0, gonzalez)
        thick = self._tip_trajectory(5.0, gonzalez)
        relDiff = np.abs(ref - thick).max() / np.abs(ref).max()
        assert relDiff < 1e-8, f"thickness changed the motion: rel diff {relDiff:.2e}"


class TestQuadratureEnergyConservation:
    """The ``quadrature`` stress conserves ``KE + W`` up to its quadrature error, so tightening
    ``energyTol`` — the per-element energy-defect tolerance — drives the drift down. That control is
    the whole point of the adaptive rule; a broken quadrature would drift like ``pointwise``
    (~1e-2). Exercised end to end through the public ``Solver_Set_Stress`` / ``Solve`` interface.
    """

    L, h = 60.0, 10.0
    dt, nStep = 0.05, 50

    def _drift(self, **stress) -> float:
        """max ``|KE + W - E0| / E0`` over a released free vibration using the quadrature stress."""
        L, h = self.L, self.h
        mesh = Domain((0, 0), (L, h), h / 2).Mesh_2D(
            [], ElemType.QUAD4, isOrganised=True
        )
        n0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
        nL = mesh.Nodes_Conditions(lambda x, y, z: x == L)
        mat = Models.HyperElastic.NeoHookean(2, K=5.0e4)
        simu = Simulations.HyperElastic(mesh, mat, absTol=1e-4, verbosity=False)

        # static preload: clamp x=0, pull the tip down
        simu.add_dirichlet(n0, [0, 0], simu.Get_unknowns())
        simu.add_dirichlet(nL, [-h / 3], ["y"])
        simu.Solve()
        simu.Save_Iter()

        # release the tip → free vibration with the chosen quadrature rule
        simu.Bc_Init()
        simu.Solver_Set_Hyperbolic_Algorithm(self.dt, algo=AlgoType.midpoint)
        simu.Solver_Set_Stress(simu.StressType.quadrature, **stress)
        simu.add_dirichlet(n0, [0, 0], simu.Get_unknowns())

        pt = simu.problemType
        energies = [float(simu._Calc_W())]  # t=0: at rest, KE = 0
        M = None
        for _ in range(self.nStep):
            simu.Solve()
            simu.Save_Iter()
            if M is None:
                _, _, M, _ = simu.Get_K_C_M_F(pt)  # constant mass
            v = simu._Get_v_n(pt)
            energies.append(0.5 * float(v @ (M @ v)) + float(simu._Calc_W()))
        E = np.array(energies)
        return float(np.abs(E - E[0]).max() / abs(E[0]))

    def test_energyTol_controls_energy_drift(self):
        """Tightening ``energyTol`` conserves ``KE + W`` markedly better."""
        loose = self._drift(energyTol=1e-2)
        tight = self._drift(energyTol=1e-8)
        assert (
            tight < loose / 100
        ), f"tighter energyTol should conserve energy far better: loose={loose:.1e}, tight={tight:.1e}"
        assert (
            tight < 1e-6
        ), f"tight energyTol should conserve energy well; got drift {tight:.1e}"


class TestKelvinVoigtWiring:
    """The viscous residual the operator returns is exactly what reaches ``F_e``.

    ``Construct_local_matrix_system`` must subtract ``R_e`` once, with the right sign, and
    put nothing else in ``F_e``. Turning viscosity off changes nothing else in the
    assembly, so the difference of the two ``F_e`` is the viscous residual alone. The
    operator itself is checked in ``tests/FEM/operators_test.py``.
    """

    def test_residual_enters_F_e_once(self):
        L, h = 60.0, 10.0
        eta = 100.0
        rng = np.random.default_rng(4)

        mesh = Domain((0, 0), (L, h), h).Mesh_2D([], ElemType.QUAD4, isOrganised=True)
        mat = Models.HyperElastic.NeoHookean(2, K=5.0e4)
        mat.eta = eta
        simu = Simulations.HyperElastic(mesh, mat, verbosity=False)
        simu.Solver_Set_Hyperbolic_Algorithm(0.05, algo=AlgoType.midpoint)

        # Any configuration with a non-zero velocity will do; equilibrium is not needed.
        pt = simu.problemType
        n = mesh.Nn * 2
        simu._Set_solutions(
            pt,
            rng.standard_normal(n) * 0.05,
            rng.standard_normal(n) * 0.5,
            rng.standard_normal(n) * 0.5,
        )
        u_np1 = rng.standard_normal(n) * 0.05
        simu._Simu__Solver_Set_Newton_Raphson_current_solution(u_np1)

        groupElem = mesh.groupElem
        F_visco = simu.Construct_local_matrix_system(pt)[groupElem][3]
        mat.eta = 0.0
        F_plain = simu.Construct_local_matrix_system(pt)[groupElem][3]
        mat.eta = eta

        # what the operator says the viscous residual is, at the same evaluation state
        u_t, v_t, _ = simu._Solver_Evaluate_u_v_a_for_time_scheme(pt, u_np1)
        state = HyperElasticState(groupElem, u_t, MatrixType.rigi)
        _, R_e, _ = Operators.NonLinear.KelvinVoigtDamping(mat, state, v_t)

        assert (
            np.abs(R_e).max() > 0
        ), "test state is degenerate: the viscous residual is zero"
        got = F_plain - F_visco  # F_e -= R_e  =>  the difference is +R_e
        assert np.abs(got - R_e).max() < 1e-10 * np.abs(R_e).max(), (
            "F_e does not carry exactly one -R_visco: max diff "
            f"{np.abs(got - R_e).max():.3e} vs |R| {np.abs(R_e).max():.3e}"
        )
