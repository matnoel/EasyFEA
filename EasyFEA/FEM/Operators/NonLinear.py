# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from .._linalg import FeArray
from .._utils import MatrixType
from ...Models._utils import Project_matrix_to_vector, Project_vector_to_matrix
from ...Models.HyperElastic._state import HyperElasticState

if TYPE_CHECKING:
    from .._group_elem import _GroupElem
    from ...Models.HyperElastic._laws import _HyperElastic


# ----------------------------------------------------------------------------
# Shared block-assembly kinematics
# ----------------------------------------------------------------------------
# These helpers back both the hyperelastic and Kelvin–Voigt operators: the
# block gradient / strain-displacement operator, the geometric-tangent stress
# block, and the (xi,...,xn,yi,...) → (xi,yi,zi,...) dof permutation are the
# same machinery regardless of which stress drives them.


def einsum(*args):
    return np.asarray(np.einsum(*args, optimize=True))


_BLOCK_GRAD_B_ATTR = "_block_grad_B_cache"


def __block_grad_B(
    state: "HyperElasticState",
) -> tuple["FeArray", "FeArray"]:
    """Block gradient operator ``grad`` and ``B = De(u)·grad``.

    ``grad`` maps nodal dofs (laid out ``xi,...,xn,yi,...,yn,...``) to the flat displacement gradient; ``B`` is the nonlinear (Green-Lagrange) strain- displacement operator. Shared by :func:`SecondPiolaKirchhoffStressTensor` and :func:`KelvinVoigtDamping`.

    Within one assembly the same ``state`` is handed to both operators, so the
    result is memoized **on the state object** (not in any module-level
    container): the cache lives and dies with that transient state, which both
    avoids rebuilding the identical ``grad``/``B`` twice and cannot accumulate.
    """
    cached = getattr(state, _BLOCK_GRAD_B_ATTR, None)
    if cached is not None:
        return cached

    groupElem = state.groupElem
    matrixType = state.matrixType
    dN_e_pg = groupElem.Get_dN_e_pg(matrixType)
    De_e_pg = state.Compute_De()

    Ne, nPg = dN_e_pg.shape[:2]
    nPe = groupElem.nPe
    dim = groupElem.dim
    nCols = De_e_pg.shape[-1]  # = dim², the flat-grad dimension

    grad_e_pg = FeArray.zeros(Ne, nPg, nCols, dim * nPe)
    rows = np.arange(nCols).reshape((dim, dim))
    cols = np.arange(dim * nPe).reshape(dN_e_pg._shape)
    for i in range(dim):
        grad_e_pg._assemble(rows[i], cols[i], value=dN_e_pg)

    B_e_pg = De_e_pg @ grad_e_pg
    result = (grad_e_pg, B_e_pg)
    setattr(state, _BLOCK_GRAD_B_ATTR, result)
    return result


def __geometric_tangent(
    wJ_e_pg: "FeArray",
    state: "HyperElasticState",
    dWde_e_pg: "FeArray",
) -> np.ndarray:
    r"""Geometric (initial-stress) tangent ``∫ gradᵀ · Sig · grad dΩ``.

    Returned in component-major (``xi,...,xn,yi,...,yn,...``) layout to match the material tangent before the shared reorder.

    Exploits the block structure ``Sig = I_dim ⊗ dWde``: with ``dWde`` the ``dim×dim`` PK2 (from the Kelvin-Mandel vector) and ``dN`` the cartesian shape-function gradients, the dense ``gradᵀ·Sig·grad`` collapses to a block-diagonal Kronecker product::

        g   = ∫ dNᵀ · dWde · dN dΩ            (Ne, nPe, nPe)
        Kgeo = g ⊗ I_dim                     (Ne, dim·nPe, dim·nPe)

    i.e. ``Kgeo[j·nPe+a, k·nPe+b] = δ_{jk} · g[a,b]``. This avoids building the dense ``(Ne, nPg, dim², dim²)`` ``Sig`` and the ``dim²``-wide contraction.
    """
    groupElem = state.groupElem
    Ne, dim, nPe = groupElem.Ne, groupElem.dim, groupElem.nPe
    sig_e_pg = Project_vector_to_matrix(dWde_e_pg)  # (Ne, nPg, dim, dim)
    dN_e_pg = groupElem.Get_dN_e_pg(state.matrixType)  # (Ne, nPg, dim, nPe)
    g_e = einsum("ep,epab,epac,epcd->ebd", wJ_e_pg, dN_e_pg, sig_e_pg, dN_e_pg)
    return einsum("eab,jk->ejakb", g_e, np.eye(dim)).reshape(Ne, dim * nPe, dim * nPe)


def __reorder(dim: int, nPe: int) -> np.ndarray:
    """Permutation from ``(xi,...,xn,yi,...,yn,...)`` to ``(xi,yi,zi,...,xn,yn,zn)``."""
    return np.arange(0, nPe * dim).reshape(-1, nPe).T.ravel()


def __reorder_dofs(dim: int, nPe: int, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Reorder local arrays from component-major to interleaved dof layout.

    Applies the :func:`__reorder` permutation — ``(xi,...,xn,yi,...,yn,...)`` → ``(xi,yi,zi,...,xn,yn,zn)`` — to each local array by its rank: a vector ``(Ne, nPe·dim)`` (residual / force) or a matrix ``(Ne, nPe·dim, nPe·dim)`` (tangent / mass / damping). Returns the reordered arrays in the given order, so every nonlinear operator shares this one permutation site instead of open-coding the fancy indexing.
    """
    perm = __reorder(dim, nPe)
    ri, rj = perm[:, None], perm[None, :]
    reordered = [None] * len(arrays)
    for i, array in enumerate(arrays):
        if array.ndim == 2:  # (Ne, ndof) vector
            reordered[i] = array[:, perm]
        elif array.ndim == 3:  # (Ne, ndof, ndof) matrix
            reordered[i] = array[:, ri, rj]
        else:
            raise ValueError(
                f"each array must be (Ne, ndof) or (Ne, ndof, ndof); got ndim {array.ndim}."
            )
    return tuple(reordered)


def __second_piola_block(
    wJ_e_pg: "FeArray",
    state: "HyperElasticState",
    dWde_e_pg: "FeArray",
    d2Wde_e_pg: "FeArray",
) -> tuple[np.ndarray, np.ndarray]:
    r"""Residual and material+geometric d2Wde for a Kelvin-Mandel dWde / d2Wde sampled at ``state`` — the shared core of the hyperelastic dWde operators::

        R_e     = ∫ Bᵀ · dWde dΩ
        K_block = ∫ Bᵀ · d2Wde · B dΩ  +  ∫ gradᵀ (I ⊗ dWde) grad dΩ

    Both component-major (``xi,...,xn,yi,...``), before the shared reorder. The fused einsum contracts the strain and Gauss-point axes in one pass, avoiding the per-Gauss ``(Ne, nPg, ndof, ndof)`` intermediate a chained ``Bᵀ @ d2Wde @ B`` would build (summation order differs from the matmul chain, so results match only to ~1e-14 relative). :func:`SecondPiolaKirchhoffStressTensor` feeds the constitutive ``(dWde, d2Wde)``; :func:`GonzalezStressTensor` feeds the discrete-gradient ``(Ŝ, ℂ̄)`` at the midpoint state.
    """
    _, B_e_pg = __block_grad_B(state)
    A_lin = einsum("ep,epji,epjk,epkl->eil", wJ_e_pg, B_e_pg, d2Wde_e_pg, B_e_pg)
    A_geo = __geometric_tangent(wJ_e_pg, state, dWde_e_pg)
    residual_e = einsum("ep,epi,epij->ej", wJ_e_pg, dWde_e_pg, B_e_pg)
    return A_lin + A_geo, residual_e


def SecondPiolaKirchhoffStressTensor(
    material: "_HyperElastic",
    state: "HyperElasticState",
) -> tuple[np.ndarray, np.ndarray]:
    """Tangent and residual for a hyperelastic constitutive law.

    Returns ``(K_e, R_e)`` in ``(xi,yi,zi,...,xn,yn,zn)``.

    The operator pulls

    - ``De_e_pg`` from ``state.Compute_De()`` — kinematic operator,
    - ``dWde_e_pg`` from ``material.Compute_dWde(state)`` — PK2 in Kelvin-Mandel vector form (strictly ``∂W/∂e``; the non-conservative stresses have their own operators),
    - ``d2Wde_e_pg`` from ``material.Compute_d2Wde(state)`` — consistent tangent in Kelvin-Mandel matrix form,

    and assembles::

        B_e_pg  = De · grad                       (strain-displacement)
        Sig_e_pg = block(P(dWde_e_pg))            (geometric tangent kernel)

        K_e = ∫ Bᵀ · d2Wde · B dΩ  +  ∫ gradᵀ · Sig · grad dΩ
        R_e = ∫ Bᵀ · dWde dΩ

    where ``P(·)`` is the Kelvin-Mandel vector → symmetric matrix projection.

    Parameters
    ----------
    material
        Hyperelastic constitutive law — supplies ``Compute_dWde(state)`` and ``Compute_d2Wde(state)``.
    state
        Hyperelastic state — owns the mesh and the current displacement.

    Returns
    -------
    A_e : ndarray of shape ``(Ne, nPe·dim, nPe·dim)``
        Consistent tangent — sum of the linear (material) and nonlinear (geometric) pieces.
    r_e : ndarray of shape ``(Ne, nPe·dim)``
        Internal residual force.
    """

    groupElem = state.groupElem
    matrixType = state.matrixType
    wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(matrixType)
    nPe = groupElem.nPe
    dim = groupElem.dim

    tangent_e, residual_e = __second_piola_block(
        wJ_e_pg,
        state,
        material.Compute_dWde(state),
        material.Compute_d2Wde(state),
    )

    if dim == 2:
        thickness = material.thickness
        tangent_e *= thickness
        residual_e *= thickness

    return __reorder_dofs(dim, nPe, tangent_e, residual_e)


def GonzalezStressTensor(
    material: "_HyperElastic",
    state_n: "HyperElasticState",
    state_mid: "HyperElasticState",
    state_np1: "HyperElasticState",
    useConsistentTangent: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Energy-conserving (Gonzalez / Simo-Tarnow) tangent and residual.

    Returns ``(K_e, R_e)`` in ``(xi,yi,zi,...,xn,yn,zn)`` for the **discrete-gradient** stress::

        Ŝ = s̄ + α Δe ,   α = (ΔW − s̄·Δe) / (Δe·Δe)   (α = 0 when Δe·Δe ≤ ε₀)

        R_e = ∫ B_midᵀ · Ŝ dΩ

    with ``s̄ = S(E(ū))`` the midpoint PK2, ``Δe = E(u_{n+1}) − E(u_n)`` and ``ΔW = W(u_{n+1}) − W(u_n)``. By construction ``Ŝ:Δe = ΔW``, and since ``Δe = B(ū)·Δu`` exactly, ``F_int·Δu = ΔW``: total energy is conserved for any stored energy. Kelvin-Mandel form makes the double contractions plain dot products, so no ``halfCrossed``-type correction is needed.

    Tangent — the **consistent** Jacobian ``∂R_e/∂u_{n+1}``, built for :attr:`~EasyFEA.AlgoType.midpoint`'s ``coefK = 0.5``::

        coefK · K_e = ½[ ∫ B_midᵀ ℂ̄ B_mid dΩ + A_geo(state_mid, Ŝ) ]   # midpoint block, raw
                    + α ∫ B_midᵀ B_{n+1} dΩ  +  ∫ (B_midᵀ Δe) ⊗ g dΩ    # corrections, pre-doubled

    The midpoint block is returned raw so ``coefK`` supplies its ``∂ū/∂u_{n+1} = ½`` chain factor, while the discrete-gradient corrections are genuine ``∂/∂u_{n+1}`` terms and are pre-doubled to survive it; ``g = ∂α/∂u_{n+1}`` and both corrections vanish where ``Δe·Δe ≤ ε₀``. The rank-1 term makes ``K_e`` non-symmetric.

    Parameters
    ----------
    material
        Hyperelastic constitutive law — supplies ``Compute_W`` / ``Compute_dWde`` / ``Compute_d2Wde``.
    state_n, state_mid, state_np1
        Hyperelastic states at ``u_n``, ``ū`` and ``u_{n+1}`` (same group / matrix type).
    useConsistentTangent
        If False, keep only the midpoint block: same residual, linear Newton convergence.

    Returns
    -------
    K_e : ndarray of shape ``(Ne, nPe·dim, nPe·dim)``
        Consistent tangent, built for ``coefK = 0.5``.
    R_e : ndarray of shape ``(Ne, nPe·dim)``
        Discrete-gradient internal residual force.
    """

    eps0 = 1e-10

    groupElem = state_mid.groupElem
    matrixType = state_mid.matrixType
    wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(matrixType)
    nPe = groupElem.nPe
    dim = groupElem.dim

    # --- 1. discrete-gradient stress  Ŝ = s̄ + α Δe ---
    s_mid = material.Compute_dWde(state_mid)  # s̄   (Ne, nPg, d)
    C_mid = material.Compute_d2Wde(state_mid)  # ℂ̄   (Ne, nPg, d, d)
    # Kelvin-Mandel strain increment Δe = E(u_{n+1}) − E(u_n), sliced to `d` so it
    # shares s̄'s basis; the √2 shear factor makes s̄:ΔE = s̄·Δe a plain FeArray dot.
    E_n = Project_matrix_to_vector(state_n.Compute_GreenLagrange())
    E_np1 = Project_matrix_to_vector(state_np1.Compute_GreenLagrange())
    dE = state_mid._Slice_Vector(E_np1 - E_n)

    # numerator N = ΔW − s̄·Δe
    N = (material.Compute_W(state_np1) - material.Compute_W(state_n)) - s_mid.dot(dE)
    dEdE = dE.dot(dE)  # Δe·Δe
    # guard the vanishing denominator: α = 0 where Δe·Δe ≤ ε₀ (invD = 0 there)
    inv_dEdE = np.divide(1.0, dEdE, out=np.zeros_like(dEdE), where=dEdE > eps0)
    alpha = N * inv_dEdE  # (Ne, nPg), N / Δe·Δe
    S_hat = s_mid + alpha * dE  # scalar-field α broadcasts over Δe

    # --- 2. residual + midpoint material/geometric block (shared with SPK) ---
    # Built RAW so it rides coefK = 0.5's ∂ū/∂u_{n+1} = ½ chain factor exactly like
    # SecondPiolaKirchhoffStressTensor; the geometric block carries the full Ŝ.
    tangent_e, residual_e = __second_piola_block(wJ_e_pg, state_mid, S_hat, C_mid)

    if useConsistentTangent:
        # --- 3. discrete-gradient tangent corrections  (∂α/∂u_{n+1}) ---
        # Genuine ∂/∂u_{n+1} terms (no ½ chain factor) → pre-doubled to survive coefK = 0.5;
        # they vanish where α = 0 (invD = 0). The rank-1 term makes K non-symmetric.
        _, B_mid = __block_grad_B(state_mid)  # cache hit (built in phase 2)
        _, B_np1 = __block_grad_B(state_np1)
        s_np1 = material.Compute_dWde(state_np1)  # S(E(u_{n+1}))
        v = C_mid @ dE  # ℂ̄ : Δe
        # dof-covector g = ∂α/∂u_{n+1}
        g = inv_dEdE * (B_np1.T @ s_np1 - 0.5 * (B_mid.T @ v) - B_np1.T @ s_mid) - (
            N * inv_dEdE * inv_dEdE * 2.0
        ) * (B_np1.T @ dE)
        tangent_e += 2.0 * (
            # α ∫ B_midᵀ B_{n+1}
            einsum("ep,ep,epji,epjk->eik", wJ_e_pg, alpha, B_mid, B_np1)
            # rank-1 ∫ (B_midᵀΔe)⊗g
            + einsum("ep,epi,epj->eij", wJ_e_pg, B_mid.T @ dE, g)
        )

    if dim == 2:
        thickness = material.thickness
        tangent_e *= thickness
        residual_e *= thickness

    return __reorder_dofs(dim, nPe, tangent_e, residual_e)


@lru_cache(maxsize=None)
def __clenshaw_curtis(nPoints: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    r"""Clenshaw-Curtis nodes and weights on ``[0, 1]``, using ``nPoints`` points.

    Nodes are the Chebyshev extrema ``cos(kπ/n)`` (``n = nPoints - 1``) rescaled to the unit interval and returned in **increasing** order; weights come from the direct closed form (Trefethen, *Spectral Methods in MATLAB*, ``clencurt``) and sum to 1. ``nPoints = 1`` is the midpoint convention (single node at ``½``).

    The classical rules fall out of the same formula — ``nPoints = 1, 2, 3`` are the midpoint, trapezoid and Simpson rules — so nothing downstream special-cases them, and more points converge spectrally for a smooth integrand. The endpoints are *exactly* ``0`` and ``1``, which :func:`TimeQuadratureStressTensor` relies on to reuse the two genuine end states rather than interpolate a strain there.

    Returns tuples rather than arrays on purpose: the result is cached, and a mutable array handed out repeatedly could be modified in place by a caller and silently corrupt every later call. Cached because the same handful of rules is requested at every Newton iteration.
    """
    assert nPoints >= 1, f"nPoints must be >= 1 (got {nPoints})."
    if nPoints == 1:
        return (0.5,), (1.0,)

    n = nPoints - 1
    theta = np.pi * np.arange(n + 1) / n
    x = np.cos(theta)
    w = np.zeros(n + 1)
    ii = np.arange(1, n)
    v = np.ones(n - 1)
    if n % 2 == 0:
        w[0] = w[n] = 1.0 / (n**2 - 1)
        for k in range(1, n // 2):
            v -= 2 * np.cos(2 * k * theta[ii]) / (4 * k**2 - 1)
        v -= np.cos(n * theta[ii]) / (n**2 - 1)
    else:
        w[0] = w[n] = 1.0 / n**2
        for k in range(1, (n - 1) // 2 + 1):
            v -= 2 * np.cos(2 * k * theta[ii]) / (4 * k**2 - 1)
    w[ii] = 2 * v / n

    # [-1, 1] -> [0, 1], ascending. `cos(π/2)` is 6e-17 rather than 0, so snap the centre
    # node to an exact ½ (it is a meaningful special value on the path).
    nodes = 0.5 * x[::-1] + 0.5
    nodes = np.where(np.abs(nodes - 0.5) < 1e-12, 0.5, nodes)

    return tuple(nodes), tuple(0.5 * w[::-1])


class _StrainPathState(HyperElasticState):
    """A point of the segment ``C(s) = C_n + s (C_{n+1} - C_n)`` — the strain path integrated by :func:`TimeQuadratureStressTensor`.

    No displacement field produces such a strain, so only the constitutive response is defined: a law reads a state through the invariants ``I1…I8``, which all descend from :meth:`Compute_C`. The kinematic quantities raise.

    :meth:`_sliced` builds a holder over an explicit ``C`` — used by the per-element adaptive path to evaluate the response on the still-refining subset of elements. ``Ne``/``nPg`` therefore come from the stored ``C`` (:meth:`_GetDims`), not from the group, so the two constructors stay consistent whether ``C`` spans the whole block or a subset.
    """

    def __init__(
        self,
        state_n: HyperElasticState,
        state_np1: HyperElasticState,
        s: float,
    ):
        """``s = 0`` sits at ``state_n``, ``s = 1`` at ``state_np1``."""
        assert state_n.groupElem is state_np1.groupElem, "states must share their group"
        assert (
            state_n.matrixType == state_np1.matrixType
        ), "states must share matrixType"

        # the displacement only fixes the solution dimension, it does not generate the strain
        super().__init__(state_n.groupElem, state_n.displacement, state_n.matrixType)

        C_n = state_n.Compute_C()
        self.__C_e_pg = C_n + s * (state_np1.Compute_C() - C_n)

    @classmethod
    def _sliced(cls, template: HyperElasticState, C_e_pg) -> "_StrainPathState":
        """Holder over an explicit ``C_e_pg`` (typically an element subset of a path state). ``template`` lends its group / displacement / matrixType — used only for the scalar ``dim`` — while ``Ne``/``nPg`` come from ``C_e_pg``."""
        obj = cls.__new__(cls)
        HyperElasticState.__init__(
            obj, template.groupElem, template.displacement, template.matrixType
        )
        obj.__C_e_pg = C_e_pg
        return obj

    def _GetDims(self) -> tuple[int, int, int]:
        # Ne, nPg from the stored C (so a sliced holder sizes its arrays to the subset); dim is
        # a scalar property of the discretisation, taken from the borrowed displacement/group.
        Ne, nPg = self.__C_e_pg.shape[:2]
        dim = self.displacement.size // self.groupElem.Ncoords
        return Ne, nPg, dim

    def Compute_C(self):
        return self.__C_e_pg

    def Compute_F(self):
        raise NotImplementedError("an interpolated strain has no deformation gradient")

    def Compute_De(self):
        raise NotImplementedError(
            "an interpolated strain has no strain-displacement operator"
        )


def __AdaptiveTimeQuadratureStressTensor(
    material: "_HyperElastic",
    state_n: "HyperElasticState",
    state_np1: "HyperElasticState",
    tol: float,
    maxPoints: int,
) -> tuple["FeArray", "FeArray", int]:
    r"""Per-element adaptive strain-path quadrature — the ``tol``-driven path of :func:`TimeQuadratureStressTensor`.

    Each element refines along the nested chain ``1, 3, 5, 9, …`` until *its own* energy defect is within ``tol``, then freezes — so a low-strain element stops at one point while a high-strain one keeps refining. The test is the **integrated** relative error over the element, ``∫_Ωe |S:Δe − ΔW| dΩ ≤ tol · ∫_Ωe |ΔW| dΩ``, evaluated as the Gauss-point sums ``Σ_p V_(ep) |S:Δe − ΔW| ≤ tol · Σ_p V_(ep) |ΔW|`` with ``V_(ep)`` the Gauss point's volume (weight × Jacobian) — so the ``V_(ep)`` factor makes each side a genuine *integral over the element*, **not** a pointwise energy-density comparison. This L1 (absolute) form is the tightest simple bound on the element's actual per-step energy drift ``|∫_Ωe (S:Δe − ΔW) dΩ|`` (triangle inequality, no volume factor), and taking ``|·|`` before summing makes it safe against sign cancellation between Gauss points — a well-resolved region cannot mask a coarse one. Energy-safe because ``S:Δe = ΔW`` holds per Gauss point.

    Only the still-active elements are evaluated at each level — the rule is applied to that subset via :meth:`_StrainPathState._sliced` — so the constitutive cost tracks the hard elements, not the mesh. (Nodes shared with a coarser level are re-evaluated rather than cached; the active set shrinks fast, so that stays cheap and keeps the loop plain.) Returns ``(dWde_quad, d2Wde_quad)`` — each row carrying its element's accepted rule — and ``nPts_e``, the point count each element accepted.
    """
    groupElem = state_n.groupElem  # state_n and state_np1 share the group
    dim = groupElem.dim
    wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(
        state_n.matrixType
    )  # V_(ep), ref config

    C_n = state_n.Compute_C()
    dC = state_np1.Compute_C() - C_n
    dW = material.Compute_W(state_np1) - material.Compute_W(state_n)  # ΔW per point
    dE = state_np1.Compute_GreenLagrange() - state_n.Compute_GreenLagrange()
    dE_vec = Project_matrix_to_vector(dE[..., :dim, :dim])  # Δe (Kelvin-Mandel)
    refW = einsum("ep,ep->e", wJ_e_pg, np.abs(dW))  #  Σ_p V |ΔW| (Ne,)
    Ne, nPg, ncomp = *wJ_e_pg.shape[:2], dE_vec.shape[-1]

    def at(
        s, e
    ):  # constitutive-state holder at strain-path point s, restricted to elements e
        return _StrainPathState._sliced(state_n, (C_n + s * dC)[e])

    dWde_quad = FeArray.zeros(Ne, nPg, ncomp)
    d2Wde_quad = FeArray.zeros(Ne, nPg, ncomp, ncomp)
    activeElements = np.arange(Ne)  # elements still refining
    nPts_e = np.zeros(Ne, dtype=int)  # accepted point count per element (diagnostic)
    nPts = 1
    while activeElements.size:
        nodes, weights = __clenshaw_curtis(nPts)
        S = sum(
            (
                w * material.Compute_dWde(at(s, activeElements))
                for s, w in zip(nodes, weights)
            ),
            0.0,
        )
        defect = einsum("epi,epi->ep", S, dE_vec[activeElements]) - dW[activeElements]
        next_nPts = 3 if nPts == 1 else 2 * nPts - 1  # next level in the chain
        # accept an element once its own energy defect is within tol (all of them at the last level)
        isAccepted = (next_nPts > max(maxPoints, 1)) | (
            einsum("ep,ep->e", wJ_e_pg[activeElements], np.abs(defect))
            <= tol * refW[activeElements]
        )
        if isAccepted.any():
            acceptedElems = activeElements[isAccepted]  # elements accepting this rule
            dWde_quad[acceptedElems] = S[isAccepted]
            # their tangent only: Σ_k 2 w_k s_k d2Wde, s=0 drops out (∂e/∂u = s B)
            d2Wde_quad[acceptedElems] = sum(
                (
                    (2.0 * w * s) * material.Compute_d2Wde(at(s, acceptedElems))
                    for s, w in zip(nodes, weights)
                    if s
                ),
                0.0,
            )
            nPts_e[acceptedElems] = nPts  # record each frozen element's accepted rule
            activeElements = activeElements[~isAccepted]
        nPts = next_nPts

    return dWde_quad, d2Wde_quad, nPts_e


def TimeQuadratureStressTensor(
    material: "_HyperElastic",
    state_n: "HyperElasticState",
    state_mid: "HyperElasticState",
    state_np1: "HyperElasticState",
    nPoints: int,
    tol: Optional[float] = None,
    maxPoints: int = 33,
) -> tuple[np.ndarray, np.ndarray, int]:
    r"""Tangent and residual for the PK2 stress **averaged along the strain path** of the step.

    Returns ``(K_e, R_e)`` in ``(xi,yi,zi,...,xn,yn,zn)``. Where :func:`SecondPiolaKirchhoffStressTensor` samples the stress at one configuration, this integrates it along the straight segment joining the two end strains and contracts the result with the **midpoint** operator::

        e(s) = e_n + s Δe ,   Δe = e_{n+1} - e_n ,   s ∈ [0, 1]

        S_quad = ∫₀¹ ∂W/∂e(e(s)) ds  ≈  Σ_k w_k ∂W/∂e(e(s_k))
        R_e    = ∫ B(ū)ᵀ · S_quad dΩ

    ``(s_k, w_k)`` is the Clenshaw-Curtis rule on ``nPoints`` points (:func:`__clenshaw_curtis`); ``1, 2, 3`` are the midpoint, trapezoid and Simpson rules. Intermediate nodes are :class:`_StrainPathState`; ``s = 0, 1`` reuse the end states.

    **Adaptive (per-element) mode.** With ``tol`` set, the rule is chosen *element by element* by :func:`__AdaptiveTimeQuadratureStressTensor`: each element walks the nested chain ``1, 3, 5, 9, 17, 33`` (up to ``maxPoints``) and freezes once *its own* energy defect is within ``tol``, so points are spent only where the step is nonlinear — a low-strain element may stop at a single midpoint while a stiff one keeps refining. The defect ``S_quad:Δe − ΔW`` is the quadrature error of the discrete-gradient identity ``S_quad:Δe = ΔW`` (with ``ΔW`` *known* from the endpoints), so the test is absolute — no consecutive-difference guess. It is the **integrated** relative error over the element, ``∫_Ωe |S_quad:Δe − ΔW| dΩ ≤ tol · ∫_Ωe |ΔW| dΩ`` — the integrals are the volume-weighted Gauss-point sums ``Σ_p V_(ep)···``, *not* a pointwise energy-density comparison — so ``tol`` reads as "conserve energy to this relative tolerance". Taking the absolute value before summing bounds the element's actual energy drift directly and guards against sign cancellation between Gauss points. Since each level is scored on its own (no comparison to a coarser one) the test accepts the coarsest rule directly: the ``1``-point midpoint is exact for a linear energy integrand, so a quadratic ``W`` converges at a single point. The tangent uses each element's accepted rule, so residual and tangent stay consistent.

    Since ``Δe = B(ū)·Δu`` exactly and ``de/ds = Δe`` is constant along the segment, the fundamental theorem of calculus gives ``S_quad:Δe = ΔW`` once the ``s``-integral is exact — a **discrete gradient**. The energy defect is therefore just the quadrature error, which Clenshaw-Curtis drives down spectrally; a quadratic ``W`` is exact at every rule. Note ``nPoints = 1`` is the average-strain stress ``S(½(e_n + e_{n+1}))``, *not* the midpoint-displacement stress of :func:`SecondPiolaKirchhoffStressTensor`.

    Tangent — only ``e_{n+1}`` depends on ``u_{n+1}``, and ``∂e(s_k)/∂u_{n+1} = s_k B_{n+1}``, so every node contracts with the same ``B_{n+1}`` and the constitutive tensors collapse into one weighted sum::

        coefK · K_e = ½ A_geo(state_mid, S_quad)                    # raw
                    + ∫ B(ū)ᵀ [ Σ_k w_k s_k ℂ(e(s_k)) ] B_{n+1} dΩ  # pre-doubled

    Built for :attr:`~EasyFEA.AlgoType.midpoint`'s ``coefK = 0.5`` as in :func:`GonzalezStressTensor`; the doubled weights sum to 1 whatever ``nPoints``. Pairing ``B(ū)`` with ``B_{n+1}`` makes ``K_e`` non-symmetric.

    Parameters
    ----------
    material
        Hyperelastic constitutive law — supplies ``Compute_dWde(state)`` and ``Compute_d2Wde(state)``.
    state_n, state_mid, state_np1
        Hyperelastic states at ``u_n``, ``ū`` and ``u_{n+1}`` (same group / matrix type).
    nPoints
        Number of Clenshaw-Curtis points for the fixed rule (``tol is None``). Ignored when
        adaptive, which always starts from 1 (the midpoint).
    tol
        If set, refine adaptively (see above) until the relative energy defect
        ``‖S_quad:Δe − ΔW‖ / ‖ΔW‖`` falls below ``tol``. ``None`` (default) keeps the fixed
        ``nPoints`` rule.
    maxPoints
        Adaptive only: cap on the number of points, by default 33. Refinement stops here even
        if ``tol`` is not met.

    Returns
    -------
    K_e : ndarray of shape ``(Ne, nPe·dim, nPe·dim)``
        Consistent tangent, built for ``coefK = 0.5``.
    R_e : ndarray of shape ``(Ne, nPe·dim)``
        Internal residual force.
    nPts_e : ndarray of shape ``(Ne,)``
        Clenshaw-Curtis points each element used — constant ``nPoints`` when fixed, per-element
        when adaptive.
    """

    groupElem = state_mid.groupElem
    wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(state_mid.matrixType)
    nPe = groupElem.nPe
    dim = groupElem.dim

    _, B_mid = __block_grad_B(state_mid)
    _, B_np1 = __block_grad_B(state_np1)

    if tol is None:
        # Fixed rule: one Clenshaw-Curtis rule for the whole block.
        dWde_quad = 0.0  # Σ_k w_k dWde(e(s_k))
        d2Wde_quad = (
            0.0  # Σ_k 2 w_k s_k d2Wde(e(s_k)); s=0 drops out (∂e/∂u_{n+1}=s B=0)
        )
        for s, w in zip(*__clenshaw_curtis(int(nPoints))):
            if s == 0.0:
                state = state_n
            elif s == 1.0:
                state = state_np1
            else:
                state = _StrainPathState(state_n, state_np1, s)
            dWde_quad += w * material.Compute_dWde(state)
            if s != 0.0:
                d2Wde_quad += (2.0 * w * s) * material.Compute_d2Wde(state)
        nPts_e = np.full(groupElem.Ne, int(nPoints))  # every element uses the same rule
    else:
        # Adaptive: refine per element on the *energy defect*. This stress exists so that
        # S_quad : Δe = ΔW exactly (a discrete gradient, for energy conservation); the only error
        # that matters is how far the quadrature of ∫ ∂W/∂e·Δe ds still is from the true ΔW, which
        # is *known* from the endpoints — so the test is absolute (each level scored on its own).
        # Refining element-by-element rather than the whole block spends points only where the
        # material is nonlinear over the step (see __AdaptiveTimeQuadratureStressTensor).
        dWde_quad, d2Wde_quad, nPts_e = __AdaptiveTimeQuadratureStressTensor(
            material, state_n, state_np1, tol, int(maxPoints)
        )

    # not __second_piola_block: that pairs one B with itself, while here the tangent is
    # differentiated at u_{n+1} but tested against B(ū) — hence the non-symmetry.
    residual_e = einsum("ep,epi,epij->ej", wJ_e_pg, dWde_quad, B_mid)
    tangent_e = einsum(
        "ep,epji,epjk,epkl->eil", wJ_e_pg, B_mid, d2Wde_quad, B_np1
    ) + __geometric_tangent(wJ_e_pg, state_mid, dWde_quad)

    if dim == 2:
        thickness = material.thickness
        tangent_e *= thickness
        residual_e *= thickness

    K_e, R_e = __reorder_dofs(dim, nPe, tangent_e, residual_e)
    return K_e, R_e, nPts_e


def ActiveStressTensor(
    material: "_HyperElastic",
    state: "HyperElasticState",
) -> tuple[np.ndarray, np.ndarray]:
    r"""Active-stress contributions ``(Kgeo_e, R_e)`` for the fiber stress :math:`\Sigma_{act} = \tau \, \hat{T} \otimes \hat{T}`.

    A contractile stress of magnitude ``τ`` (``material.active_stress``) along the unit fiber direction ``T̂``, typical of cardiac mechanics. It is **not** a derivative of the stored energy ``W``, so it is assembled here instead of being folded into ``material.Compute_dWde`` — the same separation Kelvin–Voigt viscosity gets in :func:`KelvinVoigtDamping`, and for the same reason: it keeps ``Compute_dWde == ∂(Compute_W)/∂e``, which every energy-based algorithm relies on. In particular :func:`GonzalezStressTensor` builds its discrete gradient from ``ΔW`` and ``s̄``; had ``s̄`` carried the active stress while ``ΔW`` did not, ``α`` would silently cancel the active contribution and the active stress would do exactly zero work.

    Since ``Σ_act`` is independent of ``e`` there is **no material tangent**; it does however stiffen the structure geometrically::

        R_e    = ∫ Bᵀ · Σ_act dΩ                    internal force
        Kgeo_e = ∫ gradᵀ (I ⊗ Σ_act) grad dΩ        = ∂R_e/∂u

    The simulation adds both to the elastic ``K_e`` / residual, so ``Kgeo_e`` rides ``coefK`` exactly like the elastic geometric tangent.

    Parameters
    ----------
    material
        Hyperelastic constitutive law — supplies ``active_stress`` and ``Compute_active_stress(state)``.
    state
        Hyperelastic state at the evaluation point of the time scheme (the midpoint state ``ū`` for :attr:`~EasyFEA.AlgoType.midpoint`), owning the mesh and that displacement.

    Returns
    -------
    tuple
        ``(None, None)`` when ``material.active_stress == 0``. Otherwise ``Kgeo_e`` of shape ``(Ne, nPe·dim, nPe·dim)`` and ``R_e`` of shape ``(Ne, nPe·dim)``, reordered to ``(xi, yi, zi, ..., xn, yn, zn)``.
    """
    if material.active_stress == 0.0:
        return None, None  # type: ignore [return-value]

    groupElem = state.groupElem
    wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(state.matrixType)
    nPe = groupElem.nPe
    dim = groupElem.dim

    sig_e_pg = material.Compute_active_stress(state)

    _, B_e_pg = __block_grad_B(state)
    residual_e = einsum("ep,epi,epij->ej", wJ_e_pg, sig_e_pg, B_e_pg)
    Kgeo_e = __geometric_tangent(wJ_e_pg, state, sig_e_pg)

    if dim == 2:
        thickness = material.thickness
        Kgeo_e *= thickness
        residual_e *= thickness

    return __reorder_dofs(dim, nPe, Kgeo_e, residual_e)


def KelvinVoigtDamping(
    material: "_HyperElastic",
    state: "HyperElasticState",
    velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Kelvin–Voigt viscous element contributions (Kgeo_e, R_e, C_e) for the
    large-strain viscous force F_visco(u) = C(u)·v, with Σ_visco = η·Ė
    (Green-Lagrange strain rate of velocity) and B = De(u)·grad.

    Ordered ``(tangent, residual, extra)`` like the rest of this module.

    - Kgeo_e — the configuration tangent ∂(C·v)/∂u at fixed velocity (geometric
      stiffening from Σ_visco plus the ∂Ė/∂u term); the simulation adds it to
      K_e so it rides coefK.
    - R_e = thickness · η · ∫ Bᵀ Ė dΩ — the viscous residual, which the simulation
      subtracts from F_e.
    - C_e = thickness · η · ∫ Bᵀ B dΩ — the damping matrix; the simulation puts it
      in slot 2 of (K, C, M, F), where it rides the coefC·C tangent.

    Parameters
    ----------
    material
        Hyperelastic constitutive law — supplies the viscosity eta.
    state
        Hyperelastic state — owns the mesh and the current displacement.
    velocity
        Velocity field (same (xi, yi, zi, ...) layout as the displacement), or
        None for a quasi-static evaluation.

    Returns
    -------
    tuple
        (None, None, None) when material.eta == 0 or velocity is None. Kgeo_e and C_e
        are (Ne, nPe·dim, nPe·dim) and R_e is (Ne, nPe·dim), all reordered to
        (xi, yi, zi, ..., xn, yn, zn).
    """
    if material.eta == 0.0 or velocity is None:
        return None, None, None  # type: ignore [return-value]

    groupElem = state.groupElem
    matrixType = state.matrixType
    wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(matrixType)
    nPe = groupElem.nPe
    dim = groupElem.dim
    thickness = material.thickness if dim == 2 else 1

    grad_e_pg, B_e_pg = __block_grad_B(state)
    Beta_e_pg = state.Compute_Deta(velocity) @ grad_e_pg
    sig_e_pg = material.eta * state.Compute_Edot_vec(velocity)  # Σ_visco = η·Ė

    # damping matrix C = thickness · η · ∫ Bᵀ B (fused einsum, see SPK above)
    subscripts = "ep,epji,epjl->eil"
    C_e = thickness * material.eta * einsum(subscripts, wJ_e_pg, B_e_pg, B_e_pg)

    # viscous residual ∫ Bᵀ Σ_visco — same contraction as the active stress
    residual_e = thickness * einsum("ep,epji,epj->ei", wJ_e_pg, B_e_pg, sig_e_pg)

    # configuration tangent ∂(C·v)/∂u = geometric (∫ gradᵀ Sig grad) + material-like
    # (η ∫ Bᵀ (∂Ė/∂u)) pieces
    A_mat = material.eta * einsum(subscripts, wJ_e_pg, B_e_pg, Beta_e_pg)
    A_geo = __geometric_tangent(wJ_e_pg, state, sig_e_pg)
    Kgeo_e = thickness * (A_mat + A_geo)

    return __reorder_dofs(dim, nPe, Kgeo_e, residual_e, C_e)


def __skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric "cross-product" matrix: ``S(v) · w = v × w``.

    Input ``v`` of shape ``(..., 3)`` → output ``(..., 3, 3)``.

    ::

        S(v) = |  0   -v_2   v_1 |
               | v_2    0   -v_0 |
               |-v_1   v_0    0  |
    """
    zero = np.zeros_like(v[..., 0])
    return np.stack(
        [
            np.stack([zero, -v[..., 2], v[..., 1]], axis=-1),
            np.stack([v[..., 2], zero, -v[..., 0]], axis=-1),
            np.stack([-v[..., 1], v[..., 0], zero], axis=-1),
        ],
        axis=-2,
    )


def FollowingPressure(
    groupElem: "_GroupElem",
    u: np.ndarray,
    pressure: Union[float, np.ndarray],
    elements: Optional[np.ndarray] = None,
    matrixType: "MatrixType" = MatrixType.rigi,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Follower-pressure contribution on a 2D surface group in a 3D mesh.

    The load tracks the deformed normal ``n = ∂x/∂r × ∂x/∂s`` with ``x = X + u``, so its Jacobian feeds a non-symmetric tangent.

    Returned ``(K_e, R_e)`` are contributions to global ``K`` and ``R(u) = R_internal − F_follower`` — same convention as PK2:

    ```
    K_e = -∂F_follower/∂u    → slot K
    R_e = -F_follower(u)     → slot F as -R_e (= +F_follower in b)
    ```

    Outside ``elements`` the returned arrays are exact zero so the surface connectivity can scatter ``(Ne_surf, ...)`` uniformly.
    """
    assert groupElem.dim in [1, 2], "groupElem must be 1D or 2D."

    dim = 3
    Ne, nPe = groupElem.Ne, groupElem.nPe
    ndof = nPe * dim

    K_e = np.zeros((Ne, ndof, ndof))
    R_e = np.zeros((Ne, ndof))

    if elements is None:
        active = np.arange(Ne)
    else:
        active = np.asarray(elements, dtype=int).ravel()
        if active.size == 0:
            return K_e, R_e
    Ne_a = active.size

    if np.isscalar(pressure) and float(pressure) == 0.0:
        return K_e, R_e

    gauss = groupElem.Get_gauss(matrixType)
    weights = gauss.weights

    # Reference-frame shape functions and r-/s-derivatives
    N_pg = groupElem.Get_N_pg(matrixType)[:, 0, :]  # (nPg, nPe)
    dN_pg = groupElem.Get_dN_pg(matrixType)  # (nPg, 2, nPe)
    dNr_pg = dN_pg[:, 0, :]
    dNs_pg = dN_pg[:, 1, :]

    # Deformed node coordinates x = X + u (Ne_a, nPe, 3)
    connect = groupElem.connect
    connect_local = groupElem._global_to_local_nodes[connect]
    X_e = groupElem.coord[connect_local][active]
    u_e = u.reshape(-1, dim)[connect][active]
    x_e = X_e + u_e

    # Deformed tangents dxdr_e_pg = ∂x/∂r, dxds_e_pg = ∂x/∂s at Gauss points
    dxdr_e_pg = einsum("pn,enc->epc", dNr_pg, x_e)  # (Ne_a, nPg, 3)
    dxds_e_pg = einsum("pn,enc->epc", dNs_pg, x_e)  # (Ne_a, nPg, 3)
    n_e_pg = np.cross(dxdr_e_pg, dxds_e_pg)  # area-weighted deformed normal

    # F[e, i, c] = Σ_p w·p · φ_i · n_c   (component-major, then reorder)
    factor = weights[None, :] * pressure  # (1, nPg)
    F_active = einsum("ep,pn,epc->enc", factor, N_pg, n_e_pg).reshape(
        Ne_a, dim * nPe
    )  # (xi, yi, zi, ...)

    # Chain rule on n = a × b (with a = ∂x/∂r, b = ∂x/∂s) gives, at each Gauss point, the 3×3 matrix
    #     ∂n / ∂u_{j, :} = ∂φ_j/∂s · S(a)  −  ∂φ_j/∂r · S(b)
    # where S(v) is the skew-symmetric "cross-product" matrix:
    #     S(v) = | 0    −v_2   v_1 |     so that  S(v) · w = v × w.
    #            | v_2   0    −v_0 |
    #            |−v_1   v_0   0   |
    # The local tangent is then a Kronecker product over components × nodes:
    #     K_e_pg  =  factor · [ S(a) ⊗ (φ ⊗ ∂φ/∂sᵀ)  −  S(b) ⊗ (φ ⊗ ∂φ/∂rᵀ) ]
    # i.e. block (c_a, c_b) carries S(a)[c_a,c_b]·φ_iᵀ∂φ_j/∂s − S(b)[c_a,c_b]·φ_iᵀ∂φ_j/∂r.
    S_a = __skew(dxdr_e_pg)  # S(a),  shape (Ne_a, nPg, 3, 3)
    S_b = __skew(dxds_e_pg)  # S(b),  shape (Ne_a, nPg, 3, 3)
    K_active = (
        einsum("ep,epcd,pi,pj->ecidj", factor, S_a, N_pg, dNs_pg)
        - einsum("ep,epcd,pi,pj->ecidj", factor, S_b, N_pg, dNr_pg)
    ).reshape(Ne_a, dim * nPe, dim * nPe)

    # component-major → interleaved (xi, yi, zi, ...)
    (K_active,) = __reorder_dofs(dim, nPe, K_active)

    K_e[active] = -K_active
    R_e[active] = F_active

    return K_e, R_e


def PenaltyContact(
    groupElem: "_GroupElem",
    penalty: float,
    gap_e_pg: FeArray,
    normal_e_pg: FeArray,
    elements: Optional[np.ndarray] = None,
    matrixType: "MatrixType" = MatrixType.mass,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Penalty-contact tangent/residual on a contact surface group.

    Integrated over ``groupElem`` — the contact surface group of the deformable body, so it assembles onto the body dofs — from a precomputed signed normal gap ``gap_e_pg`` and outward unit normal ``normal_e_pg`` at its integration points. Contact is active only where ``gap < 0``.

    The operator is agnostic to **how** the gap / normal are obtained — by projecting the deformed contact-surface Gauss points onto a rigid obstacle surface (:meth:`_GroupElem._Get_gap_and_normal`), by an analytic signed-distance obstacle (plane, sphere, …), etc. They only have to be sampled at the contact-surface Gauss points of ``matrixType``, the same rule used here for the surface integral.

    With penalty ``εₙ``, signed gap ``gₙ``, outward normal ``n`` and test / trial fields ``v`` / ``u``, the weak-form contributions are::

        R_e = εₙ ∫_Γ  ⟨-gₙ⟩ (v·n) dΓ       slot F: outward force, grows with penetration ⟨-gₙ⟩
        K_e = εₙ ∫_Γc      (u·n)(v·n) dΓ    slot K: tangent ∂R/∂u on the active set Γc (gₙ < 0)

    where ``⟨·⟩`` is the Macaulay bracket. Linearising the ramp ``⟨-gₙ⟩`` collapses it to the active-set restriction ``Γc``, so ``K_e`` is :func:`~EasyFEA.FEM.Operators.Bilinear.MassAlongNormal` scaled by ``εₙ`` where contact is active (the small change-of-normal / closest-point curvature terms are dropped).

    Returned ``(K_e, R_e)`` follow the slot convention of :func:`FollowingPressure` — ``K_e`` → slot K, ``R_e`` → slot F (the force pushing the body out of the obstacle). Outside ``elements`` both are exact zero.

    Parameters
    ----------
    groupElem : _GroupElem
        Contact surface group (1D edges in 2D, 2D faces in 3D).
    penalty : float
        Penalty stiffness ``εₙ``.
    gap_e_pg : FeArray
        Signed normal gap at the contact-surface ``matrixType`` Gauss points, shape ``(Ne_a, nPg)`` (negative under penetration).
    normal_e_pg : FeArray
        Outward unit normal at the same Gauss points, shape ``(Ne_a, nPg, 3)``. Must share its ``nPg`` with ``gap_e_pg``.
    elements : np.ndarray, optional
        Active (contact) element indices ``gap_e_pg``/``normal_e_pg`` were computed for, by default all.
    matrixType : MatrixType, optional
        Integration scheme for the surface integral; ``gap_e_pg`` / ``normal_e_pg`` must be sampled with the same one, by default ``MatrixType.mass``.
    """
    assert groupElem.dim in [1, 2], "groupElem must be a 1D or 2D boundary group."
    assert isinstance(gap_e_pg, FeArray) and isinstance(
        normal_e_pg, FeArray
    ), "gap_e_pg and normal_e_pg must be FeArrays."
    assert (
        gap_e_pg.shape[1] == normal_e_pg.shape[1]
    ), "gap_e_pg and normal_e_pg must share the same nPg."

    dim = groupElem.inDim  # ambient (world) dimension
    Ne, nPe = groupElem.Ne, groupElem.nPe
    ndof = nPe * dim

    K_e = np.zeros((Ne, ndof, ndof))
    R_e = np.zeros((Ne, ndof))

    if elements is None:
        active = np.arange(Ne)
    else:
        active = np.asarray(elements, dtype=int).ravel()
        if active.size == 0:
            return K_e, R_e

    if penalty == 0.0:
        return K_e, R_e

    # surface integration measure and shape functions
    wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(matrixType)[active]  # (Ne_a, nPg)
    N_pg = groupElem.Get_N_pg(matrixType)[:, 0, :]  # (nPg, nPe)
    assert (
        gap_e_pg.shape[1] == wJ_e_pg.shape[1]
    ), "gap_e_pg / normal_e_pg must be sampled at the `matrixType` Gauss points."

    # precomputed gap / outward normal at the active Gauss points
    normal_e_pg = normal_e_pg[..., :dim]  # (Ne_a, nPg, dim)

    # active set: penetration only (gap < 0)
    pen_e_pg = np.where(gap_e_pg < 0, -gap_e_pg, 0.0)  # ⟨-gₙ⟩ ≥ 0
    H_e_pg = (gap_e_pg < 0).astype(float)  # active-set indicator

    # The einsum node/component index order (...i,c...) yields the interleaved
    # (xi, yi, zi, ...) dof layout directly, so no reorder is needed.
    factor = penalty * wJ_e_pg  # (Ne_a, nPg)

    # R_e = +εₙ ∫ ⟨-gₙ⟩ Nᵢ n dΓ   (force pushing the body out → slot F)
    R_active = einsum(
        "ep,ep,pi,epc->eic",
        factor,
        pen_e_pg,
        N_pg,
        normal_e_pg,
    ).reshape(active.size, nPe * dim)

    # K_e = +εₙ ∫ H Nᵢ Nⱼ (n⊗n) dΓ   (tangent → slot K)
    K_active = einsum(
        "ep,ep,pi,pj,epc,epd->eicjd",
        factor,
        H_e_pg,
        N_pg,
        N_pg,
        normal_e_pg,
        normal_e_pg,
    ).reshape(active.size, nPe * dim, nPe * dim)

    K_e[active] = K_active
    R_e[active] = R_active

    return K_e, R_e
