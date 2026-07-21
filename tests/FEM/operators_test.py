# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Tests for the nonlinear FEM operators (EasyFEA.FEM.Operators.NonLinear).

These operators are normally exercised only through ``Simulations`` Newton
solves, where a wrong tangent merely slows convergence and never shows up as a
wrong answer. They are validated here in isolation: the analytic tangents are
checked against central finite differences of the element force they linearize.
"""

import numpy as np
import pytest

from EasyFEA import ElemType, MatrixType, Models
from EasyFEA.Geoms import Domain
from EasyFEA.FEM import Operators, FeArray
from EasyFEA.Models.HyperElastic._state import HyperElasticState

# Element types spanning shapes and interpolation orders (low + high order).
# Meshes are kept to a single element (or the few an organised mesh splits a
# unit cell into), so the finite-difference loop stays cheap even at high order.
ELEMS = [
    (2, ElemType.TRI3),
    (2, ElemType.QUAD4),
    (2, ElemType.TRI6),
    (2, ElemType.QUAD8),
    (3, ElemType.TETRA4),
    (3, ElemType.TETRA10),
    (3, ElemType.PRISM6),
    (3, ElemType.PRISM15),
    (3, ElemType.PRISM18),
    (3, ElemType.HEXA8),
    (3, ElemType.HEXA20),
    (3, ElemType.HEXA27),
]
ELEM_IDS = [e.name for _, e in ELEMS]

LAWS = ["SaintVenantKirchhoff", "NeoHookean", "MooneyRivlin"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _mesh(dim: int, elemType: ElemType):
    """Single-element (organised) unit-square / unit-cube mesh.

    ``meshSize`` equal to the cell size keeps the count minimal (QUAD/HEXA = 1,
    TRI = 2, TETRA = 6), so the finite-difference loop stays cheap at high order.
    """
    if dim == 2:
        return Domain((0, 0), (1, 1), 1.0).Mesh_2D([], elemType, isOrganised=True)
    return Domain((0, 0), (1, 1), 1.0).Mesh_Extrude(
        [], [0, 0, 1], [1], elemType, isOrganised=True
    )


def _material(dim: int, law: str):
    if law == "SaintVenantKirchhoff":
        return Models.HyperElastic.SaintVenantKirchhoff(dim, lmbda=1.2, mu=0.8)
    if law == "NeoHookean":
        return Models.HyperElastic.NeoHookean(dim, K=2.0)
    if law == "MooneyRivlin":
        return Models.HyperElastic.MooneyRivlin(dim, K1=0.5, K2=0.3, K=1.0)
    raise ValueError(f"unknown law {law!r}")


def _fd_tangent(force_fn, u, asse, eps=1e-6):
    """Central-difference element tangent ``K[e,:,j] = ∂(force_fn(u)[e]) / ∂u[asse[e,j]]``.

    ``force_fn(u_)`` returns the element force array ``(Ne, ndof_e)``. The
    ``ndof_e`` global dofs of an element are distinct, so perturbing one global
    dof isolates one element-local column.
    """
    Ne, ndof_e = asse.shape
    K_fd = np.zeros((Ne, ndof_e, ndof_e))
    for e in range(Ne):
        for j in range(ndof_e):
            g = asse[e, j]
            up = u.copy()
            up[g] += eps
            um = u.copy()
            um[g] -= eps
            K_fd[e, :, j] = (force_fn(up)[e] - force_fn(um)[e]) / (2 * eps)
    return K_fd


def _assert_matches(K_ana, K_fd, tol=1e-6, msg=""):
    err = np.abs(K_ana - K_fd).max()
    scale = np.abs(K_fd).max()
    assert err / scale < tol, f"{msg}: rel err {err / scale:.2e}"


# ----------------------------------------------------------------------------
# SecondPiolaKirchhoffStressTensor
# ----------------------------------------------------------------------------


class TestSecondPiolaKirchhoff:
    """Hyperelastic PK2 operator: the consistent tangent must be ``∂R/∂u``.

    The operator returns ``(K, R)`` with ``R = ∫ Bᵀ·Σ dΩ`` the internal residual
    force and ``K`` its consistent tangent (material + geometric). A tangent error
    only slows Newton convergence — the converged answer is unchanged — so it is
    invisible to simulation tests; check it directly against finite differences.
    """

    @staticmethod
    def _residual_fn(ge, mat):
        def residual(u_):
            st = HyperElasticState(ge, u_, MatrixType.rigi)
            _, R = Operators.NonLinear.SecondPiolaKirchhoffStressTensor(mat, st)
            return R

        return residual

    @pytest.mark.parametrize("dim, elemType", ELEMS, ids=ELEM_IDS)
    def test_tangent_vs_finite_difference(self, dim, elemType):
        """Tangent matches FD across element shapes / orders (Saint-Venant-Kirchhoff)."""
        rng = np.random.default_rng(2)
        mesh = _mesh(dim, elemType)
        mat = _material(dim, "SaintVenantKirchhoff")
        ge = mesh.groupElem

        u = rng.standard_normal(mesh.Nn * dim) * 0.03  # small strains: det(F) > 0
        asse = ge.Get_assembly_e(dim)

        st = HyperElasticState(ge, u, MatrixType.rigi)
        K_ana, _ = Operators.NonLinear.SecondPiolaKirchhoffStressTensor(mat, st)
        K_fd = _fd_tangent(self._residual_fn(ge, mat), u, asse)
        _assert_matches(K_ana, K_fd, msg=elemType.name)

    @pytest.mark.parametrize("law", LAWS)
    def test_tangent_vs_finite_difference_laws(self, law):
        """Each constitutive law's ``Compute_d2Wde`` is consistent with ``Compute_dWde``."""
        rng = np.random.default_rng(3)
        for dim, elemType in [(2, ElemType.QUAD4), (3, ElemType.HEXA8)]:
            mesh = _mesh(dim, elemType)
            mat = _material(dim, law)
            ge = mesh.groupElem

            u = rng.standard_normal(mesh.Nn * dim) * 0.02
            asse = ge.Get_assembly_e(dim)

            st = HyperElasticState(ge, u, MatrixType.rigi)
            K_ana, _ = Operators.NonLinear.SecondPiolaKirchhoffStressTensor(mat, st)
            K_fd = _fd_tangent(self._residual_fn(ge, mat), u, asse)
            _assert_matches(K_ana, K_fd, msg=f"{law}/{elemType.name}")

    def test_tangent_is_symmetric(self):
        """A hyperelastic (conservative) tangent without follower loads is symmetric."""
        rng = np.random.default_rng(5)
        for dim, elemType in [(2, ElemType.QUAD8), (3, ElemType.HEXA8)]:
            mesh = _mesh(dim, elemType)
            mat = _material(dim, "SaintVenantKirchhoff")
            ge = mesh.groupElem
            u = rng.standard_normal(mesh.Nn * dim) * 0.03
            st = HyperElasticState(ge, u, MatrixType.rigi)
            K, _ = Operators.NonLinear.SecondPiolaKirchhoffStressTensor(mat, st)
            asym = np.abs(K - K.transpose(0, 2, 1)).max()
            assert asym / np.abs(K).max() < 1e-10, f"{elemType.name}: asym {asym:.2e}"


# ----------------------------------------------------------------------------
# Active fiber stress
# ----------------------------------------------------------------------------


class TestActiveStress:
    """Active fiber stress ``τ·(T̂⊗T̂)`` of ``ActiveStressTensor``.

    It is strain-independent, so it carries no material tangent — but it does
    stiffen the structure geometrically, and it must stay *out* of the
    constitutive stress so that ``Compute_dWde`` remains ``∂(Compute_W)/∂e``.
    """

    TAU = 0.5

    @staticmethod
    def _material_with_fibers(dim, ge, tau=TAU):
        """Saint-Venant-Kirchhoff with unit fibers along x and an active stress ``tau``."""
        mat = _material(dim, "SaintVenantKirchhoff")
        nPg = ge.Get_N_pg(MatrixType.rigi).shape[0]
        T = np.zeros((ge.Ne, nPg, 3))
        T[..., 0] = 1.0  # fibers along x
        mat.Set_active_stress_vec(FeArray.asfearray(T))
        mat.active_stress = tau
        return mat

    @staticmethod
    def _residual_fn(ge, mat):
        def residual(u_):
            st = HyperElasticState(ge, u_, MatrixType.rigi)
            _, R = Operators.NonLinear.ActiveStressTensor(mat, st)
            return R

        return residual

    @pytest.mark.parametrize("dim, elemType", ELEMS, ids=ELEM_IDS)
    def test_geometric_tangent_vs_finite_difference(self, dim, elemType):
        """``Kgeo`` equals ``∂R/∂u`` — the whole tangent, since ``∂Σ_act/∂e = 0``."""
        rng = np.random.default_rng(6)
        mesh = _mesh(dim, elemType)
        ge = mesh.groupElem
        mat = self._material_with_fibers(dim, ge)

        u = rng.standard_normal(mesh.Nn * dim) * 0.02
        asse = ge.Get_assembly_e(dim)

        st = HyperElasticState(ge, u, MatrixType.rigi)
        Kgeo, _ = Operators.NonLinear.ActiveStressTensor(mat, st)
        K_fd = _fd_tangent(self._residual_fn(ge, mat), u, asse)
        _assert_matches(Kgeo, K_fd, msg=elemType.name)

    def test_stays_out_of_the_constitutive_stress(self):
        """``Compute_dWde`` must stay ``∂(Compute_W)/∂e``, active stress or not.

        This is the invariant every energy-based algorithm rests on — notably the
        ``GonzalezStressTensor`` discrete gradient, which pairs ``ΔW`` with ``s̄``:
        an active term in ``s̄`` but not in ``W`` would be silently cancelled by
        ``α`` and do exactly zero work.
        """
        rng = np.random.default_rng(7)
        dim, elemType = 3, ElemType.HEXA8
        mesh = _mesh(dim, elemType)
        ge = mesh.groupElem
        u = rng.standard_normal(mesh.Nn * dim) * 0.02
        st = HyperElasticState(ge, u, MatrixType.rigi)

        mat = self._material_with_fibers(dim, ge)
        S_active = mat.Compute_dWde(st)
        mat.active_stress = 0.0
        S_inert = mat.Compute_dWde(st)

        assert np.array_equal(np.asarray(S_active), np.asarray(S_inert))

    def test_inactive_returns_nothing(self):
        """``active_stress == 0`` short-circuits, so the assembly can skip it."""
        mesh = _mesh(3, ElemType.HEXA8)
        ge = mesh.groupElem
        mat = _material(3, "SaintVenantKirchhoff")  # never given a fiber direction
        st = HyperElasticState(ge, np.zeros(mesh.Nn * 3), MatrixType.rigi)
        assert Operators.NonLinear.ActiveStressTensor(mat, st) == (None, None)

    def test_requires_a_registered_direction(self):
        """A ``τ`` with no fiber direction is an error, not a silent no-op."""
        mesh = _mesh(3, ElemType.HEXA8)
        ge = mesh.groupElem
        mat = _material(3, "SaintVenantKirchhoff")
        mat.active_stress = self.TAU  # but no Set_active_stress_vec
        st = HyperElasticState(ge, np.zeros(mesh.Nn * 3), MatrixType.rigi)
        with pytest.raises(AssertionError):
            Operators.NonLinear.ActiveStressTensor(mat, st)


# ----------------------------------------------------------------------------
# Kelvin–Voigt viscosity
# ----------------------------------------------------------------------------


class TestKelvinVoigt:
    """Kelvin–Voigt (large-strain) viscous contributions of ``KelvinVoigtDamping``."""

    @pytest.mark.parametrize(
        "dim, elemType", [(2, ElemType.QUAD4), (2, ElemType.QUAD8), (3, ElemType.HEXA8)]
    )
    def test_Deta_vs_Edot(self, dim, elemType):
        """``Compute_Deta`` is ``∂Ė/∂(∇u)`` at fixed velocity.

        ``Ė = De(u)·flat(∇v)`` is bilinear in ``(∇u, ∇v)``; its identity (the
        ``1+`` part of ``De``) is removed in ``Deta``, leaving the part linear in
        ``∇u``. Contracting ``Deta`` with ``flat(∇u)`` must therefore recover ``Ė``
        minus the identity contribution ``De(0)·flat(∇v)``.
        """
        rng = np.random.default_rng(1)
        mesh = _mesh(dim, elemType)
        ge = mesh.groupElem
        n = mesh.Nn * dim
        u = rng.standard_normal(n) * 0.03
        v = rng.standard_normal(n) * 0.1

        st = HyperElasticState(ge, u, MatrixType.rigi)
        Edot = st.Compute_Edot_vec(v)  # De(u)·flat(∇v)
        Deta = st.Compute_Deta(v)  # ∂Ė/∂(∇u), built from ∇v
        grad_u = ge.Get_Gradient_e_pg(u, MatrixType.rigi)[..., :dim, :dim]
        grad_u_flat = np.reshape(grad_u, (*grad_u.shape[:2], -1))

        st0 = HyperElasticState(ge, np.zeros(n), MatrixType.rigi)
        Edot_lin = Edot - st0.Compute_Edot_vec(v)  # u-linear remainder

        diff = Edot_lin - (Deta @ grad_u_flat)
        assert np.linalg.norm(diff) / np.linalg.norm(Edot_lin) < 1e-12

    @pytest.mark.parametrize("dim, elemType", ELEMS, ids=ELEM_IDS)
    def test_geometric_tangent_vs_finite_difference(self, dim, elemType):
        """``Kgeo`` equals ``∂(C(u)·v)/∂u`` at fixed velocity.

        The viscous element force is ``F_visco(u) = C(u)·v``; the configuration
        tangent ``Kgeo`` (second return value) carries the two pieces a plain
        damping matrix omits — the geometric stiffening from ``Σ_visco`` and the
        ``∂Ė/∂u`` term — and must match a central finite difference.
        """
        rng = np.random.default_rng(0)
        mesh = _mesh(dim, elemType)
        mat = _material(dim, "SaintVenantKirchhoff")
        mat.eta = 0.7
        ge = mesh.groupElem

        n = mesh.Nn * dim
        u = rng.standard_normal(n) * 0.03  # small strains: det(F) > 0
        v = rng.standard_normal(n) * 0.1
        asse = ge.Get_assembly_e(dim)

        def Fvisco(u_):
            st = HyperElasticState(ge, u_, MatrixType.rigi)
            C, _ = Operators.NonLinear.KelvinVoigtDamping(mat, st, v)
            return np.einsum("eij,ej->ei", C, v[asse])

        st = HyperElasticState(ge, u, MatrixType.rigi)
        _, Kgeo = Operators.NonLinear.KelvinVoigtDamping(mat, st, v)
        K_fd = _fd_tangent(Fvisco, u, asse)
        _assert_matches(Kgeo, K_fd, msg=elemType.name)

    def test_damping_matrix_is_symmetric(self):
        """The damping matrix ``C = η·∫BᵀB`` is symmetric and positive semi-definite."""
        rng = np.random.default_rng(6)
        dim, elemType = 3, ElemType.HEXA8
        mesh = _mesh(dim, elemType)
        mat = _material(dim, "SaintVenantKirchhoff")
        mat.eta = 0.7
        ge = mesh.groupElem
        n = mesh.Nn * dim
        st = HyperElasticState(ge, rng.standard_normal(n) * 0.03, MatrixType.rigi)
        v = rng.standard_normal(n) * 0.1
        C, _ = Operators.NonLinear.KelvinVoigtDamping(mat, st, v)
        assert np.abs(C - C.transpose(0, 2, 1)).max() / np.abs(C).max() < 1e-10
        # BᵀB ⇒ each element matrix has no negative eigenvalues
        assert np.linalg.eigvalsh(C).min() > -1e-9 * np.abs(C).max()

    def test_inactive_returns_none(self):
        """No viscosity (``eta == 0``) or no velocity ⇒ ``(None, None)``."""
        dim, elemType = 2, ElemType.QUAD4
        mesh = _mesh(dim, elemType)
        ge = mesh.groupElem
        n = mesh.Nn * dim
        st = HyperElasticState(ge, np.zeros(n), MatrixType.rigi)

        mat = _material(dim, "SaintVenantKirchhoff")  # eta defaults to 0
        assert Operators.NonLinear.KelvinVoigtDamping(mat, st, np.zeros(n)) == (
            None,
            None,
        )

        mat.eta = 0.7  # viscous, but no velocity passed
        assert Operators.NonLinear.KelvinVoigtDamping(mat, st, None) == (None, None)


# ----------------------------------------------------------------------------
# Gonzalez energy-momentum operator
# ----------------------------------------------------------------------------


class TestGonzalez:
    """Energy-momentum operator ``GonzalezStressTensor(mat, state_n, state_mid, state_np1)``.

    Two properties define it. (1) The **discrete power balance** ``R·Δu = ΔW`` — the
    internal work of the residual over the step equals the stored-energy increment —
    which is what makes the scheme conserve ``KE + W`` exactly; its non-trivial
    content is the kinematic identity ``B(ū)·Δu = Δe`` (i.e. ``De(ū)`` is exact), so
    it breaks if ``B`` is built at the wrong configuration. (2) The consistent tangent
    ``∂R/∂u_{n+1}``; the operator returns it built for ``coefK = 0.5`` (shared with
    ``midpoint``), so the true Jacobian is ``½·K``.
    """

    @staticmethod
    def _states(ge, u_n, u_np1):
        return (
            HyperElasticState(ge, u_n, MatrixType.rigi),
            HyperElasticState(ge, (u_n + u_np1) / 2, MatrixType.rigi),
            HyperElasticState(ge, u_np1, MatrixType.rigi),
        )

    def _check_directionality(self, dim, elemType, law, seed):
        rng = np.random.default_rng(seed)
        mesh = _mesh(dim, elemType)
        mat = _material(dim, law)
        groupElem = mesh.groupElem

        u_n = rng.standard_normal(mesh.Nn * dim) * 0.02
        u_np1 = u_n + rng.standard_normal(mesh.Nn * dim) * 0.02  # Δe·Δe ≫ ε₀
        asse = groupElem.Get_assembly_e(dim)

        sn, smid, snp = self._states(groupElem, u_n, u_np1)
        _, R = Operators.NonLinear.GonzalezStressTensor(mat, sn, smid, snp)

        # internal work R·Δu, assembled over the interleaved element dofs
        du_e = (u_np1 - u_n)[asse]  # (Ne, ndof_e), same layout as R
        internal_work = float(np.einsum("ei,ei->", R, du_e))

        # ΔW = ∫ (W(u_{n+1}) − W(u_n)) dΩ — independent of the operator's assembly
        wJ = groupElem.Get_weightedJacobian_e_pg(MatrixType.rigi)
        dW = np.sum(wJ * (mat.Compute_W(snp) - mat.Compute_W(sn)))

        assert (
            abs(internal_work - dW) / abs(dW) < 1e-9
        ), f"{law}/{elemType.name}: R·Δu={internal_work:.6e} vs ΔW={dW:.6e}"

    @pytest.mark.parametrize("dim, elemType", ELEMS, ids=ELEM_IDS)
    def test_discrete_gradient_directionality(self, dim, elemType):
        """``R·Δu == ΔW`` across element shapes / orders (Saint-Venant-Kirchhoff)."""
        self._check_directionality(dim, elemType, "SaintVenantKirchhoff", seed=7)

    @pytest.mark.parametrize("law", LAWS)
    def test_discrete_gradient_directionality_laws(self, law):
        """``R·Δu == ΔW`` for every constitutive law (energy conservation is law-agnostic)."""
        for dim, elemType in [(2, ElemType.QUAD4), (3, ElemType.HEXA8)]:
            self._check_directionality(dim, elemType, law, seed=8)

    @staticmethod
    def _residual_fn(ge, mat, u_n):
        """``R(u_{n+1})`` with ``u_n`` fixed and ``ū = (u_n+u_{n+1})/2`` — for FD in ``u_{n+1}``."""

        def residual(u_np1_):
            sn = HyperElasticState(ge, u_n, MatrixType.rigi)
            smid = HyperElasticState(ge, (u_n + u_np1_) / 2, MatrixType.rigi)
            snp = HyperElasticState(ge, u_np1_, MatrixType.rigi)
            _, R = Operators.NonLinear.GonzalezStressTensor(mat, sn, smid, snp)
            return R

        return residual

    def _check_tangent(self, dim, elemType, law, seed):
        rng = np.random.default_rng(seed)
        mesh = _mesh(dim, elemType)
        mat = _material(dim, law)
        ge = mesh.groupElem

        u_n = rng.standard_normal(mesh.Nn * dim) * 0.02
        u_np1 = u_n + rng.standard_normal(mesh.Nn * dim) * 0.02  # Δe·Δe ≫ ε₀
        asse = ge.Get_assembly_e(dim)

        sn, smid, snp = self._states(ge, u_n, u_np1)
        K_ana, _ = Operators.NonLinear.GonzalezStressTensor(mat, sn, smid, snp)
        K_fd = _fd_tangent(self._residual_fn(ge, mat, u_n), u_np1, asse)
        # operator returns K for coefK = 0.5, so the true Jacobian ∂R/∂u_{n+1} = ½·K
        _assert_matches(0.5 * K_ana, K_fd, msg=f"{law}/{elemType.name}")

    @pytest.mark.parametrize("dim, elemType", ELEMS, ids=ELEM_IDS)
    def test_tangent_vs_finite_difference(self, dim, elemType):
        """``½·K == ∂R/∂u_{n+1}`` (FD) across element shapes / orders (Saint-Venant-Kirchhoff)."""
        self._check_tangent(dim, elemType, "SaintVenantKirchhoff", seed=9)

    @pytest.mark.parametrize("law", LAWS)
    def test_tangent_vs_finite_difference_laws(self, law):
        """The consistent discrete-gradient tangent holds for every constitutive law."""
        for dim, elemType in [(2, ElemType.QUAD4), (3, ElemType.HEXA8)]:
            self._check_tangent(dim, elemType, law, seed=10)

    @pytest.mark.parametrize("law", LAWS)
    def test_tangent_flag_leaves_residual_untouched(self, law):
        """``useConsistentTangent`` may only change ``K`` — never the residual.

        Energy conservation is carried entirely by the residual (the discrete gradient),
        so the approximate-tangent variant must integrate *exactly* the same physics and
        differ only in Newton's convergence rate. If a future optimisation of the
        cheap-tangent path perturbed ``R``, the scheme would silently stop conserving.
        """
        rng = np.random.default_rng(11)
        for dim, elemType in [(2, ElemType.QUAD4), (3, ElemType.HEXA8)]:
            mesh = _mesh(dim, elemType)
            mat = _material(dim, law)
            ge = mesh.groupElem

            u_n = rng.standard_normal(mesh.Nn * dim) * 0.02
            u_np1 = u_n + rng.standard_normal(mesh.Nn * dim) * 0.02  # Δe·Δe ≫ ε₀
            sn, smid, snp = self._states(ge, u_n, u_np1)

            K_con, R_con = Operators.NonLinear.GonzalezStressTensor(
                mat, sn, smid, snp, True
            )
            K_apx, R_apx = Operators.NonLinear.GonzalezStressTensor(
                mat, sn, smid, snp, False
            )

            assert np.array_equal(
                np.asarray(R_con), np.asarray(R_apx)
            ), f"{law}/{elemType.name}: residual must not depend on useConsistentTangent"
            # and the flag must actually do something to the tangent
            assert not np.allclose(
                np.asarray(K_con), np.asarray(K_apx)
            ), f"{law}/{elemType.name}: the two tangents should differ"
