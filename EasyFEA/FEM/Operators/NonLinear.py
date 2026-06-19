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


def __block_grad_B(
    state: "HyperElasticState",
) -> tuple["FeArray", "FeArray"]:
    """Block gradient operator ``grad`` and ``B = De(u)·grad``.

    ``grad`` maps nodal dofs (laid out ``xi,...,xn,yi,...,yn,...``) to the flat displacement gradient; ``B`` is the nonlinear (Green-Lagrange) strain- displacement operator. Shared by :func:`SecondPiolaKirchhoffStressTensor` and :func:`KelvinVoigtDamping`.
    """
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
    return grad_e_pg, B_e_pg


def __block_stress(stress_e_pg: "FeArray", dim: int) -> "FeArray":
    """Geometric-tangent stress block ``block(P(stress_vec))``.

    Places the ``dim×dim`` symmetric stress (from the Kelvin-Mandel vector ``stress_e_pg``) on each of the ``dim`` diagonal blocks, so that ``∫ gradᵀ · Sig · grad`` yields the geometric (initial-stress) tangent — i.e. ``Sig = I_dim ⊗ sig``.

    Built on a ``(Ne, nPg, dim, dim, dim, dim)`` view with basic-slice writes to
    the diagonal blocks (``Sig[i,k,i,l] = sig[k,l]``), then reshaped to
    ``(Ne, nPg, dim², dim²)`` — ~3× faster than fancy-indexed block assembly.
    """
    sig_e_pg = np.asarray(Project_vector_to_matrix(stress_e_pg))  # (Ne, nPg, dim, dim)
    Ne, nPg = sig_e_pg.shape[:2]
    Sig = np.zeros((Ne, nPg, dim, dim, dim, dim))
    for i in range(dim):
        Sig[:, :, i, :, i, :] = sig_e_pg
    return FeArray.asfearray(Sig.reshape(Ne, nPg, dim * dim, dim * dim))


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

    grad_e_pg, B_e_pg = __block_grad_B(state)
    Sig_e_pg = __block_stress(dWde_e_pg, dim)

    # linear (material) tangent + nonlinear (geometric) tangent
    A_lin = (wJ_e_pg * B_e_pg.T @ d2Wde_e_pg @ B_e_pg).integrate()
    A_geo = (wJ_e_pg * grad_e_pg.T @ Sig_e_pg @ grad_e_pg).integrate()
    tangent_e = A_lin + A_geo

    # residual
    residual_e = (wJ_e_pg * dWde_e_pg.T @ B_e_pg).integrate()

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
    Sig_e_pg = __block_stress(material.eta * state.Compute_Edot_vec(velocity), dim)

    # damping matrix C = thickness · η · ∫ Bᵀ B
    C_e = thickness * material.eta * (wJ_e_pg * B_e_pg.T @ B_e_pg).integrate()

    # configuration tangent ∂(C·v)/∂u = geometric (∫ gradᵀ Sig grad) + material-like
    # (η ∫ Bᵀ (∂Ė/∂u)) pieces
    A_mat = material.eta * (wJ_e_pg * B_e_pg.T @ Beta_e_pg).integrate()
    A_geo = (wJ_e_pg * grad_e_pg.T @ Sig_e_pg @ grad_e_pg).integrate()
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
