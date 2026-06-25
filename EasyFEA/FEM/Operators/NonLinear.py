# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from .._linalg import FeArray
from .._utils import MatrixType
from ...Models._utils import Project_vector_to_matrix

if TYPE_CHECKING:
    from .._group_elem import _GroupElem
    from ...Models.HyperElastic._laws import _HyperElastic
    from ...Models.HyperElastic._state import HyperElasticState


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
    state: "HyperElasticState",
    stress_e_pg: "FeArray",
    wJ_e_pg: "FeArray",
) -> np.ndarray:
    r"""Geometric (initial-stress) tangent ``∫ gradᵀ · Sig · grad dΩ``.

    Returned in component-major (``xi,...,xn,yi,...,yn,...``) layout to match the material tangent before the shared reorder.

    Exploits the block structure ``Sig = I_dim ⊗ sig``: with ``sig`` the ``dim×dim`` PK2 (from the Kelvin-Mandel vector) and ``dN`` the cartesian shape-function gradients, the dense ``gradᵀ·Sig·grad`` collapses to a block-diagonal Kronecker product::

        g   = ∫ dNᵀ · sig · dN dΩ            (Ne, nPe, nPe)
        Kgeo = g ⊗ I_dim                     (Ne, dim·nPe, dim·nPe)

    i.e. ``Kgeo[j·nPe+a, k·nPe+b] = δ_{jk} · g[a,b]``. This avoids building the dense ``(Ne, nPg, dim², dim²)`` ``Sig`` and the ``dim²``-wide contraction.
    """
    groupElem = state.groupElem
    Ne, dim, nPe = groupElem.Ne, groupElem.dim, groupElem.nPe
    sig_e_pg = np.asarray(Project_vector_to_matrix(stress_e_pg))  # (Ne, nPg, dim, dim)
    dN_e_pg = np.asarray(groupElem.Get_dN_e_pg(state.matrixType))  # (Ne, nPg, dim, nPe)
    g_e = einsum("ep,epab,epac,epcd->ebd", wJ_e_pg, dN_e_pg, sig_e_pg, dN_e_pg)
    return einsum("eab,jk->ejakb", g_e, np.eye(dim)).reshape(Ne, dim * nPe, dim * nPe)


def __reorder(dim: int, nPe: int) -> np.ndarray:
    """Permutation from ``(xi,...,xn,yi,...,yn,...)`` to ``(xi,yi,zi,...,xn,yn,zn)``."""
    return np.arange(0, nPe * dim).reshape(-1, nPe).T.ravel()


def SecondPiolaKirchhoffStressTensor(
    material: "_HyperElastic",
    state: "HyperElasticState",
) -> tuple[np.ndarray, np.ndarray]:
    """Tangent and residual for a hyperelastic constitutive law.

    Returns ``(K_e, R_e)`` in ``(xi,yi,zi,...,xn,yn,zn)``.

    The operator pulls

    - ``De_e_pg`` from ``state.Compute_De()`` — kinematic operator,
    - ``dWde_e_pg`` from ``material.Compute_dWde(state)`` — PK2 in Kelvin-Mandel vector form (any active-stress /  viscous fold-in is the material's responsibility),
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

    dWde_e_pg = material.Compute_dWde(state)
    d2Wde_e_pg = material.Compute_d2Wde(state)

    _, B_e_pg = __block_grad_B(state)

    # linear (material) tangent + nonlinear (geometric) tangent.
    # A single fused einsum contracts the strain indices and the Gauss-point
    # (integration) axis in one pass — it avoids materializing the per-Gauss
    # (Ne, nPg, ndof, ndof) intermediate that the chained `B.T @ d2W @ B` builds.
    # Summation order differs from the matmul chain, so results match only to
    # floating-point round-off (~1e-14 relative), not bit-for-bit.
    A_lin = einsum("ep,epji,epjk,epkl->eil", wJ_e_pg, B_e_pg, d2Wde_e_pg, B_e_pg)
    A_geo = __geometric_tangent(state, dWde_e_pg, wJ_e_pg)
    tangent_e = A_lin + A_geo

    # residual
    residual_e = einsum("ep,epi,epij->ej", wJ_e_pg, dWde_e_pg, B_e_pg)

    # reorder xi,...,xn,yi,...,yn,zi,...,zn to xi,yi,zi,...,xn,yn,zn
    reorder = __reorder(dim, nPe)
    residual_e = residual_e[:, reorder]
    tangent_e = tangent_e[:, reorder[:, None], reorder[None, :]]

    return tangent_e, residual_e


def KelvinVoigtDamping(
    material: "_HyperElastic",
    state: "HyperElasticState",
    velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Kelvin–Voigt viscous element contributions (C_e, Kgeo_e) for the
    large-strain viscous force F_visco(u) = C(u)·v, with Σ_visco = η·Ė
    (Green-Lagrange strain rate of velocity) and B = De(u)·grad.

    - C_e = thickness · η · ∫ Bᵀ B dΩ — the damping matrix; the simulation puts
      it in slot 2 of (K, C, M, F) (residual b -= C @ v_t, time-scheme history,
      and the coefC·C tangent).
    - Kgeo_e — the configuration tangent ∂(C·v)/∂u at fixed velocity (geometric
      stiffening from Σ_visco plus the ∂Ė/∂u term); the simulation adds it to
      K_e so it rides coefK.

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
        (None, None) when material.eta == 0 or velocity is None. Both matrices
        are (Ne, nPe·dim, nPe·dim) reordered to (xi, yi, zi, ..., xn, yn, zn).
    """
    if material.eta == 0.0 or velocity is None:
        return None, None  # type: ignore [return-value]

    groupElem = state.groupElem
    matrixType = state.matrixType
    wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(matrixType)
    nPe = groupElem.nPe
    dim = groupElem.dim
    thickness = material.thickness if dim == 2 else 1

    grad_e_pg, B_e_pg = __block_grad_B(state)
    Beta_e_pg = state.Compute_Deta(velocity) @ grad_e_pg

    # damping matrix C = thickness · η · ∫ Bᵀ B (fused einsum, see SPK above)
    subscripts = "ep,epji,epjl->eil"
    C_e = thickness * material.eta * einsum(subscripts, wJ_e_pg, B_e_pg, B_e_pg)

    # configuration tangent ∂(C·v)/∂u = geometric (∫ gradᵀ Sig grad) + material-like
    # (η ∫ Bᵀ (∂Ė/∂u)) pieces
    A_mat = material.eta * einsum(subscripts, wJ_e_pg, B_e_pg, Beta_e_pg)
    A_geo = __geometric_tangent(
        state, material.eta * state.Compute_Edot_vec(velocity), wJ_e_pg
    )
    Kgeo_e = thickness * (A_mat + A_geo)

    reorder = __reorder(dim, nPe)
    ri, rj = reorder[:, None], reorder[None, :]
    C_e = C_e[:, ri, rj]
    Kgeo_e = Kgeo_e[:, ri, rj]
    return C_e, Kgeo_e


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
    dxdr_e_pg = np.einsum("pn,enc->epc", dNr_pg, x_e)  # (Ne_a, nPg, 3)
    dxds_e_pg = np.einsum("pn,enc->epc", dNs_pg, x_e)  # (Ne_a, nPg, 3)
    n_e_pg = np.cross(dxdr_e_pg, dxds_e_pg)  # area-weighted deformed normal

    # F[e, i, c] = Σ_p w·p · φ_i · n_c   (component-major, then reorder)
    factor = weights[None, :] * pressure  # (1, nPg)
    F_active = np.einsum("ep,pn,epc->enc", factor, N_pg, n_e_pg).reshape(
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
        np.einsum("ep,epcd,pi,pj->ecidj", factor, S_a, N_pg, dNs_pg)
        - np.einsum("ep,epcd,pi,pj->ecidj", factor, S_b, N_pg, dNr_pg)
    ).reshape(Ne_a, dim * nPe, dim * nPe)

    # component-major → interleaved (xi, yi, zi, ...)
    reorder = np.arange(dim * nPe).reshape(dim, nPe).T.ravel()
    K_active = K_active[:, reorder[:, None], reorder[None, :]]

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

    .. math::
        R_c = -\varepsilon_n \int_\Gamma \langle -g_n \rangle \, N_i \, \mathbf{n} \, d\Gamma, \qquad
        K_c = \varepsilon_n \int_\Gamma H(-g_n) \, N_i N_j \, (\mathbf{n} \otimes \mathbf{n}) \, d\Gamma

    Returned ``(K_e, R_e)`` follow the slot convention of :func:`FollowingPressure`: ``K_e`` → slot K (tangent ``∂R/∂u``), ``R_e`` → slot F (``= -R_c``, the force pushing the body out of the obstacle). Outside ``elements`` both are exact zero.

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
