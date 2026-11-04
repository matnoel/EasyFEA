# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import TYPE_CHECKING

import numpy as np

from .._fearray import FeArray, TensorProd
from .._utils import MatrixType
from .._kelvin_mandel import Vector_to_Matrix
from .._kinematics import Kinematics

if TYPE_CHECKING:
    from .._group_elem import _GroupElem
    from ...Models.HyperElastic._laws import _HyperElastic


# ----------------------------------------------------------------------------
# Shared block-assembly kinematics
# ----------------------------------------------------------------------------
# These helpers back both the hyperelastic and Kelvin–Voigt operators: the
# block gradient / strain-displacement operator and the geometric-tangent stress
# block are the same machinery regardless of which stress drives them.


from ._utils import einsum

_BLOCK_GRAD_B_ATTR = "_block_grad_B_cache"


def __block_grad_B(
    kinematics: "Kinematics",
) -> tuple["FeArray", "FeArray"]:
    """Block gradient operator ``grad`` and ``B = De(u)·grad``.

    ``grad`` maps nodal dofs (laid out ``xi,yi,zi,...,xn,yn,zn``) to the flat displacement gradient; ``B`` is the nonlinear (Green-Lagrange) strain- displacement operator. Shared by :func:`SecondPiolaKirchhoffStressTensor` and :func:`KelvinVoigtDamping`.

    Within one assembly the same ``kinematics`` is handed to both operators, so the
    result is memoized **on the kinematics object** (not in any module-level
    container): the cache lives and dies with that transient kinematics, which both
    avoids rebuilding the identical ``grad``/``B`` twice and cannot accumulate.
    """
    cached = getattr(kinematics, _BLOCK_GRAD_B_ATTR, None)
    if cached is not None:
        return cached

    groupElem = kinematics.groupElem
    matrixType = kinematics.matrixType
    dN_e_pg = groupElem.Get_dN_e_pg(matrixType)
    De_e_pg = kinematics.Compute_De()

    Ne, nPg = dN_e_pg.shape[:2]
    nPe = groupElem.nPe
    dim = groupElem.dim
    nCols = De_e_pg.shape[-1]  # = dim², the flat-grad dimension

    grad_e_pg = FeArray.zeros(Ne, nPg, nCols, dim * nPe)
    rows = np.arange(nCols).reshape((dim, dim))
    cols = np.arange(dim * nPe).reshape(nPe, dim).T
    for i in range(dim):
        grad_e_pg._assemble(rows[i], cols[i], value=dN_e_pg)

    B_e_pg = De_e_pg @ grad_e_pg
    result = (grad_e_pg, B_e_pg)
    setattr(kinematics, _BLOCK_GRAD_B_ATTR, result)
    return result


def __add_geometric_tangent(
    K_e: np.ndarray,
    wJ_e_pg: FeArray.FeArrayALike,
    kinematics: "Kinematics",
    dWde_e_pg: FeArray.FeArrayALike,
) -> None:
    r"""Adds the geometric (initial-stress) tangent ``∫ gradᵀ · Sig · grad dΩ`` to `K_e`, in place.

    Exploits the block structure ``Sig = I_dim ⊗ dWde``: with ``dWde`` the ``dim×dim`` PK2 (from the Kelvin-Mandel vector) and ``dN`` the cartesian shape-function gradients, the dense ``gradᵀ·Sig·grad`` collapses to a block-diagonal Kronecker product::

        g   = ∫ dNᵀ · dWde · dN dΩ            (Ne, nPe, nPe)
        Kgeo = g ⊗ I_dim                     (Ne, dim·nPe, dim·nPe)

    i.e. ``Kgeo[a·dim+j, b·dim+k] = δ_{jk} · g[a,b]``. This avoids building the dense ``(Ne, nPg, dim², dim²)`` ``Sig`` and the ``dim²``-wide contraction.
    """
    groupElem = kinematics.groupElem
    dim = groupElem.dim
    sig_e_pg = Vector_to_Matrix(dWde_e_pg)  # (Ne, nPg, dim, dim)
    dN_e_pg = groupElem.Get_dN_e_pg(kinematics.matrixType)  # (Ne, nPg, dim, nPe)
    g_e = einsum("ep,epab,epac,epcd->ebd", wJ_e_pg, dN_e_pg, sig_e_pg, dN_e_pg)
    for j in range(dim):
        K_e[:, j::dim, j::dim] += g_e


def __second_piola_block(
    wJ_e_pg: FeArray.FeArrayALike,
    kinematics: "Kinematics",
    dWde_e_pg: FeArray.FeArrayALike,
    d2Wde_e_pg: FeArray.FeArrayALike,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Residual and material+geometric d2Wde for a Kelvin-Mandel dWde / d2Wde sampled at ``kinematics`` — the shared core of the hyperelastic dWde operators::

        R_e     = ∫ Bᵀ · dWde dΩ
        K_block = ∫ Bᵀ · d2Wde · B dΩ  +  ∫ gradᵀ (I ⊗ dWde) grad dΩ

    Both in ``(xi,yi,zi,...,xn,yn,zn)``. The fused einsum contracts the strain and Gauss-point axes in one pass, avoiding the per-Gauss ``(Ne, nPg, ndof, ndof)`` intermediate a chained ``Bᵀ @ d2Wde @ B`` would build (summation order differs from the matmul chain, so results match only to ~1e-14 relative).
    """
    _, B_e_pg = __block_grad_B(kinematics)
    K_e = einsum("ep,epji,epjk,epkl->eil", wJ_e_pg, B_e_pg, d2Wde_e_pg, B_e_pg)
    __add_geometric_tangent(K_e, wJ_e_pg, kinematics, dWde_e_pg)
    residual_e = einsum("ep,epi,epij->ej", wJ_e_pg, dWde_e_pg, B_e_pg)
    return K_e, residual_e


def SecondPiolaKirchhoffStressTensor(
    material: "_HyperElastic",
    kinematics: "Kinematics",
) -> tuple[np.ndarray, np.ndarray]:
    """Tangent and residual for a hyperelastic constitutive law.

    Returns ``(K_e, R_e)`` in ``(xi,yi,zi,...,xn,yn,zn)``.

    The operator pulls

    - ``De_e_pg`` from ``kinematics.Compute_De()`` — kinematic operator,
    - ``dWde_e_pg`` from ``material.Compute_dWde(kinematics)`` — PK2 in Kelvin-Mandel vector form (strictly ``∂W/∂e``; the non-conservative stresses have their own operators),
    - ``d2Wde_e_pg`` from ``material.Compute_d2Wde(kinematics)`` — consistent tangent in Kelvin-Mandel matrix form,

    and assembles::

        B_e_pg  = De · grad                       (strain-displacement)
        Sig_e_pg = block(P(dWde_e_pg))            (geometric tangent kernel)

        K_e = ∫ Bᵀ · d2Wde · B dΩ  +  ∫ gradᵀ · Sig · grad dΩ
        R_e = ∫ Bᵀ · dWde dΩ

    where ``P(·)`` is the Kelvin-Mandel vector → symmetric matrix projection.

    Parameters
    ----------
    material
        Hyperelastic constitutive law — supplies ``Compute_dWde(kinematics)`` and ``Compute_d2Wde(kinematics)``.
    kinematics
        Hyperelastic kinematics — owns the mesh and the current displacement.

    Returns
    -------
    A_e : ndarray of shape ``(Ne, nPe·dim, nPe·dim)``
        Consistent tangent — sum of the linear (material) and nonlinear (geometric) pieces.
    r_e : ndarray of shape ``(Ne, nPe·dim)``
        Internal residual force.
    """

    groupElem = kinematics.groupElem
    matrixType = kinematics.matrixType
    wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(matrixType)

    return __second_piola_block(
        wJ_e_pg,
        kinematics,
        material.Compute_dWde(kinematics),
        material.Compute_d2Wde(kinematics),
    )


def ActiveStressTensor(
    material: "_HyperElastic",
    kinematics: "Kinematics",
) -> tuple[np.ndarray, np.ndarray]:
    r"""Active-stress contributions ``(Kgeo_e, R_e)`` for the fiber stress :math:`\Sigma_{act} = \tau \, \hat{T} \otimes \hat{T}`.

    A contractile stress of magnitude ``τ`` (``material.active_stress``) along the unit fiber direction ``T̂``, typical of cardiac mechanics. It is **not** a derivative of the stored energy ``W``, so it is assembled here instead of being folded into ``material.Compute_dWde`` — the same separation Kelvin–Voigt viscosity gets in :func:`KelvinVoigtDamping`, and for the same reason: it keeps ``Compute_dWde == ∂(Compute_W)/∂e``, which every energy-based algorithm relies on.

    Since ``Σ_act`` is independent of ``e`` there is **no material tangent**; it does however stiffen the structure geometrically::

        R_e    = ∫ Bᵀ · Σ_act dΩ                    internal force
        Kgeo_e = ∫ gradᵀ (I ⊗ Σ_act) grad dΩ        = ∂R_e/∂u

    The simulation adds both to the elastic ``K_e`` / residual, so ``Kgeo_e`` rides ``coefK`` exactly like the elastic geometric tangent.

    Parameters
    ----------
    material
        Hyperelastic constitutive law — supplies ``active_stress`` and ``Compute_active_stress(kinematics)``.
    kinematics
        Hyperelastic kinematics at the evaluation point of the time scheme (the midpoint kinematics ``ū`` for :attr:`~EasyFEA.AlgoType.midpoint`), owning the mesh and that displacement.

    Returns
    -------
    tuple
        ``(None, None)`` when ``material.active_stress`` is nowhere non-zero. Otherwise ``Kgeo_e`` of shape ``(Ne, nPe·dim, nPe·dim)`` and ``R_e`` of shape ``(Ne, nPe·dim)``, in ``(xi, yi, zi, ..., xn, yn, zn)``.
    """
    # `np.all` rather than `== 0.0`: the magnitude may be an (Ne, nPg) field, and a partly-active
    # body still has to be assembled
    if np.all(material.active_stress == 0.0):
        return None, None  # type: ignore [return-value]

    groupElem = kinematics.groupElem
    wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(kinematics.matrixType)
    ndof = groupElem.nPe * groupElem.dim

    sig_e_pg = material.Compute_active_stress(kinematics)

    _, B_e_pg = __block_grad_B(kinematics)
    R_e = einsum("ep,epi,epij->ej", wJ_e_pg, sig_e_pg, B_e_pg)
    K_e = np.zeros((groupElem.Ne, ndof, ndof))
    __add_geometric_tangent(K_e, wJ_e_pg, kinematics, sig_e_pg)

    return K_e, R_e


def MixedIncompressibility(
    kinematics: "Kinematics",
    pGroupElem: "_GroupElem",
    p_n: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Tangents and residuals of the constraint ``∫ p (1 - J) dΩ`` of an incompressible u/p formulation.

    With p > 0 in compression and ``S_p = -p J C⁻¹``::

        K_uu = -∫ Bᵀ p ∂²J/∂E² B dΩ  +  ∫ gradᵀ (I ⊗ S_p) grad dΩ
        R_u  = ∫ Bᵀ S_p dΩ
        K_up = -∫ Bᵀ ∂J/∂E N_p dΩ
        K_pu = K_upᵀ
        R_p  = ∫ N_pᵀ (1 - J) dΩ

    Parameters
    ----------
    kinematics
        Hyperelastic kinematics, at an explicit rule, ``MatrixType.Order(n)``, that both groups share.
    pGroupElem
        p's group.
    p_n
        Pressure by node id of ``pGroupElem``.

    Returns
    -------
    tuple
        ``(K_uu, R_u, K_up, K_pu, R_p)``, u's DOFs in ``(xi, yi, zi, ..., xn, yn, zn)``, p's in ``pGroupElem``'s node order.
    """
    groupElem = kinematics.groupElem
    matrixType = kinematics.matrixType
    wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(matrixType)
    N_pg = pGroupElem.Get_N_pg(matrixType)[:, 0]
    if N_pg.shape[0] != wJ_e_pg.shape[1]:
        raise ValueError(
            f"p's {pGroupElem.elemType} has {N_pg.shape[0]} points at "
            f"{matrixType}, u's {groupElem.elemType} {wJ_e_pg.shape[1]}: build the "
            "kinematics at an explicit rule, MatrixType.Order(n), that both share."
        )

    J_e_pg = kinematics.Compute_J()
    # ∂J/∂I3 = 1/(2J), ∂I3/∂E = 2 ∂I3/∂C
    dJdE_e_pg = kinematics.Compute_dI3dC() / J_e_pg
    d2JdE_e_pg = (
        2 * kinematics.Compute_d2I3dC() - TensorProd(dJdE_e_pg, dJdE_e_pg)
    ) / J_e_pg
    p_e_pg = pGroupElem.Interpolate_e_pg(p_n, matrixType)

    K_uu, R_u = __second_piola_block(
        wJ_e_pg,
        kinematics,
        -p_e_pg * dJdE_e_pg,
        -p_e_pg * d2JdE_e_pg,
    )

    _, B_e_pg = __block_grad_B(kinematics)
    K_up = -einsum("ep,epji,epj,pk->eik", wJ_e_pg, B_e_pg, dJdE_e_pg, N_pg)
    R_p = einsum("ep,ep,pk->ek", wJ_e_pg, 1 - J_e_pg, N_pg)
    return K_uu, R_u, K_up, K_up.transpose(0, 2, 1), R_p


def KelvinVoigtDamping(
    material: "_HyperElastic",
    kinematics: "Kinematics",
    velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Kelvin–Voigt viscous element contributions (Kgeo_e, R_e, C_e) for the
    large-strain viscous force F_visco(u) = C(u)·v, with Σ_visco = η·Ė
    (Green-Lagrange strain rate of velocity) and B = De(u)·grad.

    Ordered ``(tangent, residual, extra)`` like the rest of this module.

    - Kgeo_e — the configuration tangent ∂(C·v)/∂u at fixed velocity (geometric
      stiffening from Σ_visco plus the ∂Ė/∂u term); the simulation adds it to
      K_e so it rides coefK.
    - R_e = η · ∫ Bᵀ Ė dΩ — the viscous residual, which the simulation
      subtracts from F_e.
    - C_e = η · ∫ Bᵀ B dΩ — the damping matrix; the simulation puts it
      in slot 2 of (K, C, M, F), where it rides the coefC·C tangent.

    Parameters
    ----------
    material
        Hyperelastic constitutive law — supplies the viscosity eta.
    kinematics
        Hyperelastic kinematics — owns the mesh and the current displacement.
    velocity
        Velocity field (same (xi, yi, zi, ...) layout as the displacement), or
        None for a quasi-static evaluation.

    Returns
    -------
    tuple
        (None, None, None) when material.eta == 0 or velocity is None. Kgeo_e and C_e
        are (Ne, nPe·dim, nPe·dim) and R_e is (Ne, nPe·dim), all in
        (xi, yi, zi, ..., xn, yn, zn).
    """
    if material.eta == 0.0 or velocity is None:
        return None, None, None  # type: ignore [return-value]

    groupElem = kinematics.groupElem
    matrixType = kinematics.matrixType
    wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(matrixType)

    grad_e_pg, B_e_pg = __block_grad_B(kinematics)
    Beta_e_pg = kinematics.Compute_Deta(velocity) @ grad_e_pg
    sig_e_pg = material.eta * kinematics.Compute_Edot_vec(velocity)  # Σ_visco = η·Ė

    # damping matrix C = η · ∫ Bᵀ B (fused einsum, see SPK above)
    subscripts = "ep,epji,epjl->eil"
    C_e = material.eta * einsum(subscripts, wJ_e_pg, B_e_pg, B_e_pg)

    # viscous residual ∫ Bᵀ Σ_visco — same contraction as the active stress
    R_e = einsum("ep,epji,epj->ei", wJ_e_pg, B_e_pg, sig_e_pg)

    # configuration tangent ∂(C·v)/∂u = geometric (∫ gradᵀ Sig grad) + material-like
    # (η ∫ Bᵀ (∂Ė/∂u)) pieces
    K_e = material.eta * einsum(subscripts, wJ_e_pg, B_e_pg, Beta_e_pg)
    __add_geometric_tangent(K_e, wJ_e_pg, kinematics, sig_e_pg)

    return K_e, R_e, C_e


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
    pressure: float | np.ndarray,
    elements: np.ndarray | None = None,
    matrixType: "MatrixType" = MatrixType.mass,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Follower-pressure contribution on a 2D surface group in a 3D mesh.

    The load tracks the deformed normal ``n = ∂x/∂r × ∂x/∂s`` with ``x = X + u``, so its Jacobian feeds a non-symmetric tangent.

    Returned ``(K_e, R_e)`` follow the same convention as every other nonlinear operator — ``R`` is the **internal** force, which the assembly subtracts — so the term is declared ``"KR"``:

    ```
    R_e = -F_follower(u)     → slot R, subtracted: the residual of A·Δu = -R(u)
    K_e = +∂R_e/∂u           → slot K, non-symmetric because the normal follows u
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

    if np.isscalar(pressure) and pressure == 0.0:
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

    # F[e, i, c] = Σ_p w·p · φ_i · n_c
    factor = weights[None, :] * pressure  # (1, nPg)
    F_active = einsum("ep,pn,epc->enc", factor, N_pg, n_e_pg).reshape(
        Ne_a, dim * nPe
    )  # (xi, yi, zi, ...)

    # Chain rule on n = a × b (with a = ∂x/∂r, b = ∂x/∂s) gives, at each Gauss point, the 3×3 matrix
    #     ∂n / ∂u_{j, :} = ∂φ_j/∂s · S(a)  -  ∂φ_j/∂r · S(b)
    # where S(v) is the skew-symmetric "cross-product" matrix:
    #     S(v) = | 0    −v_2   v_1 |     so that  S(v) · w = v × w.
    #            | v_2   0    −v_0 |
    #            |−v_1   v_0   0   |
    # The local tangent is then a Kronecker product over components × nodes:
    #     K_e_pg  =  factor · [ S(a) ⊗ (φ ⊗ ∂φ/∂sᵀ)  -  S(b) ⊗ (φ ⊗ ∂φ/∂rᵀ) ]
    # i.e. block (c_a, c_b) carries S(a)[c_a,c_b]·φ_iᵀ∂φ_j/∂s - S(b)[c_a,c_b]·φ_iᵀ∂φ_j/∂r.
    S_a = __skew(dxdr_e_pg)  # S(a),  shape (Ne_a, nPg, 3, 3)
    S_b = __skew(dxds_e_pg)  # S(b),  shape (Ne_a, nPg, 3, 3)
    K_active = (
        einsum("ep,epcd,pi,pj->eicjd", factor, S_a, N_pg, dNs_pg)
        - einsum("ep,epcd,pi,pj->eicjd", factor, S_b, N_pg, dNr_pg)
    ).reshape(Ne_a, dim * nPe, dim * nPe)

    K_e[active] = -K_active
    R_e[active] = -F_active  # internal force: the assembly subtracts it

    return K_e, R_e


def PenaltyContact(
    groupElem: "_GroupElem",
    penalty: float,
    gap_e_pg: FeArray,
    normal_e_pg: FeArray,
    elements: np.ndarray | None = None,
    matrixType: "MatrixType" = MatrixType.mass,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Penalty-contact tangent/residual on a contact surface group.

    Integrated over ``groupElem`` — the contact surface group of the deformable body, so it assembles onto the body dofs — from a precomputed signed normal gap ``gap_e_pg`` and outward unit normal ``normal_e_pg`` at its integration points. Contact is active only where ``gap < 0``.

    The operator is agnostic to **how** the gap / normal are obtained — by projecting the deformed contact-surface Gauss points onto a rigid obstacle surface (:meth:`_GroupElem._Get_gap_and_normal`), by an analytic signed-distance obstacle (plane, sphere, …), etc. They only have to be sampled at the contact-surface Gauss points of ``matrixType``, the same rule used here for the surface integral.

    With penalty ``εₙ``, signed gap ``gₙ``, outward normal ``n`` and test / trial fields ``v`` / ``u``, the weak-form contributions are::

        R_e = -εₙ ∫_Γ  ⟨-gₙ⟩ (v·n) dΓ      slot R: **internal** force, subtracted by the assembly
        K_e = +εₙ ∫_Γc      (u·n)(v·n) dΓ   slot K: tangent ∂R/∂u on the active set Γc (gₙ < 0)

    where ``⟨·⟩`` is the Macaulay bracket. Linearising the ramp ``⟨-gₙ⟩`` collapses it to the active-set restriction ``Γc``, so ``K_e`` is :func:`~EasyFEA.FEM.Operators.Bilinear.MassAlongNormal` scaled by ``εₙ`` where contact is active (the small change-of-normal / closest-point curvature terms are dropped).

    Returned ``(K_e, R_e)`` follow the same convention as every other nonlinear operator, so the term is declared ``"KR"``: the assembly subtracts ``R_e``, leaving the outward force that pushes the body out of the obstacle. Outside ``elements`` both are exact zero.

    Parameters
    ----------
    groupElem : _GroupElem
        Contact surface group (1D edges in 2D, 2D faces in 3D).
    penalty : float
        Penalty stiffness ``εₙ``.
    gap_e_pg : FeArray
        Signed normal gap at the contact-surface ``matrixType`` Gauss points, shape ``(Ne, nPg)`` — full-group, negative under penetration. Rows outside ``elements`` are never read, so an expensive projection may leave them at zero.
    normal_e_pg : FeArray
        Outward unit normal at the same Gauss points, shape ``(Ne, nPg, 3)``. Must share its ``nPg`` with ``gap_e_pg``.
    elements : np.ndarray, optional
        Candidate contact elements of ``groupElem``, by default all. Outside them the returned arrays are exact zero.
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

    # precomputed gap / outward normal, restricted to the candidate elements
    gap_e_pg = gap_e_pg[active]  # (Ne_a, nPg)
    normal_e_pg = normal_e_pg[active][..., :dim]  # (Ne_a, nPg, dim)

    # active set: penetration only (gap < 0)
    pen_e_pg = np.where(gap_e_pg < 0, -gap_e_pg, 0.0)  # ⟨-gₙ⟩ ≥ 0
    H_e_pg = (gap_e_pg < 0).astype(float)  # active-set indicator

    # The einsum node/component index order (...i,c...) yields the interleaved
    # (xi, yi, zi, ...) dof layout directly, so no reorder is needed.
    factor = penalty * wJ_e_pg  # (Ne_a, nPg)

    # R_e = -εₙ ∫ ⟨-gₙ⟩ Nᵢ n dΓ   (internal force → slot R, subtracted; the outward push is what survives)
    R_active = -einsum(
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
