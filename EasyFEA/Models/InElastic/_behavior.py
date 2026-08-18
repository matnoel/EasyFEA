# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""Materials whose stress depends on the strain history.

A :class:`Behavior` is declared by a free energy and, once it can flow, a yield surface:

.. math::
    \Sig = \dpartial{\psi}{\Eps} \qquad A = -\dpartial{\psi}{z}

where ``z`` holds the internal variables. Everything the solver needs — the stress, the
consistent tangent, the evolution of ``z`` — follows from ``psi`` and ``f``.

``Models.Elastic`` is closed-form and total; a ``Behavior`` is incremental and carries state.
With no internal variables it degenerates to :math:`\Sig = \Crm : \Eps`, which is the same
answer ``Models.Elastic`` gives.
"""

from enum import Enum
from typing import NamedTuple, Optional, Sequence, Union

import numpy as np

from .._utils import _IModel, ModelType
from ..Elastic._laws import _Elastic
from .IsotropicHardening import IsotropicHardening, Linear
from .KinematicHardening import KinematicHardening
from .ViscoPlastic import RateLaw
from .ViscoElastic import Maxwell
from .Yield import YieldSurface
from . import _spectral
from ...FEM._linalg import FeArray, TensorProd
from ...Utilities import _params, _types, Tic

# in-plane components [xx, yy, xy] of a 2D field inside the 3D (6,) Kelvin vector, and the
# out-of-plane index condensed out under plane stress
IDX_2D = [0, 1, 5]
ZZ = 2


class Slot(str, Enum):
    """Names of the internal variables a behavior can carry.

    A ``str`` enum, so ``simu.Result(Slot.p)`` and ``simu.Result("alpha")`` are the same
    call. A repeated mechanism appends an index: the second Maxwell branch is ``eps_v1``.
    """

    eps_p = "eps_p"
    """plastic strain"""
    p = "p"
    """accumulated plastic strain (Lemaitre-Chaboche notation)"""
    alpha = "alpha"
    """kinematic hardening variable; the back-stress is X = (2/3) C alpha"""
    eps_v = "eps_v"
    """viscoelastic branch strain; one per Maxwell branch, indexed"""

    def __str__(self) -> str:
        return self.name


class StateLayout(NamedTuple):
    """Where each internal variable sits in the packed ``(Ne, nPg, n)`` state."""

    slots: dict[str, slice]
    n: int

    @staticmethod
    def From(sizes: dict[str, int]) -> "StateLayout":
        slots, start = {}, 0
        for name, size in sizes.items():
            slots[name] = slice(start, start + size)
            start += size
        return StateLayout(slots, start)


class Behavior(_IModel):
    """Elasticity, and later a yield surface, hardening and a rate law.

    The elastic model must be 3D: the state lives in 6D Kelvin whatever the problem dimension,
    so that plane stress and plane strain are handled here rather than by the elastic law.
    """

    dim: int = _params.ParameterInValues([2, 3])
    thickness: float = _params.PositiveScalarParameter()
    planeStress: bool = _params.BoolParameter()
    """the 2D model uses the plane-stress assumption (otherwise plane strain)"""

    # local solver settings
    _tol: float = 1e-10
    _maxIter: int = 20
    _planeStress_tol: float = 1e-8  # relative to the yield scale

    def __init__(
        self,
        dim: int,
        elastic: _Elastic,
        yieldSurface: Optional[YieldSurface] = None,
        hardening: Optional[IsotropicHardening] = None,
        kinematic: Union[
            KinematicHardening,
            Sequence[KinematicHardening],
            None,
        ] = None,
        rate: Optional[RateLaw] = None,
        branches: Sequence[Maxwell] = (),
        thickness: float = 1.0,
        planeStress: bool = False,
        solver: str = "auto",
    ):
        """Creates a behavior.

        Parameters
        ----------
        dim : int
            dimension (2 or 3)
        elastic : _Elastic
            elastic response; must be 3D. Any law works — only its ``C`` is read.
        yieldSurface : YieldSurface, optional
            the surface, e.g. ``Yield.VonMises(sigma_y)``; ``None`` (default) stays elastic
        hardening : IsotropicHardening, optional
            stored isotropic hardening energy; ``None`` (default) is perfect plasticity
        kinematic : KinematicHardening | Sequence[KinematicHardening], optional
            kinematic hardening; one component, or several superposed as
            ``KinematicHardening.Chaboche(...)``. ``None`` (default) leaves the surface
            centred at the origin.
        rate : RateLaw, optional
            viscoplastic flow rate; ``None`` (default) is rate-independent plasticity
        branches : Sequence[Maxwell], optional
            viscoelastic Maxwell branches, by default none
        thickness : float, optional
            thickness (2D only), by default 1.0
        planeStress : bool, optional
            2D plane-stress assumption (otherwise plane strain), by default False
        """
        # what each piece must be
        assert isinstance(elastic, _Elastic), "elastic must be an elastic model"
        assert yieldSurface is None or isinstance(
            yieldSurface, YieldSurface
        ), "yieldSurface must be a YieldSurface"
        assert hardening is None or isinstance(
            hardening, IsotropicHardening
        ), "hardening must be an IsotropicHardening"
        kinematics: tuple[KinematicHardening, ...] = (
            ()
            if kinematic is None
            else (
                (kinematic,)
                if isinstance(kinematic, KinematicHardening)
                else tuple(kinematic)
            )
        )
        assert all(
            isinstance(component, KinematicHardening) for component in kinematics
        ), "kinematic must be a KinematicHardening, or a sequence of them"
        assert rate is None or isinstance(rate, RateLaw), "rate must be a RateLaw"
        assert all(
            isinstance(branch, Maxwell) for branch in branches
        ), "branches must be Maxwell branches"

        # nothing evolves until the material yields
        needsSurface = {"hardening": hardening, "kinematic": kinematic, "rate": rate}
        given = [name for name, piece in needsSurface.items() if piece is not None]
        assert not given or yieldSurface is not None, (
            f"{', '.join(given)} only act once the material yields, "
            "so a yieldSurface is needed too"
        )

        # parameter ranges
        assert elastic.dim == 3, "the elastic model must be 3D (the state is 6D Kelvin)"
        assert not (planeStress and dim == 3), "plane stress is a 2D-only assumption"
        assert all(
            branch.tau > 0 for branch in branches
        ), "every branch tau must be > 0"
        assert all(branch.g > 0 for branch in branches), "every branch g must be > 0"
        assert (
            sum(branch.g for branch in branches) < 1.0
        ), "the branch stiffness fractions must sum to less than 1"

        self.dim = dim
        self.__elastic = elastic
        self.__yield = yieldSurface
        self.__hardening = hardening if hardening is not None else Linear(0.0)
        self.__kinematic = kinematics
        self.__rate = rate
        self.__branches = tuple(branches)
        self.thickness = thickness
        self.planeStress = planeStress

        sizes: dict[str, int] = {}
        if yieldSurface is not None:
            sizes[Slot.eps_p] = 6
            sizes[Slot.p] = 1
        for i in range(len(kinematics)):
            sizes[f"{Slot.alpha}{i}"] = 6
        for i in range(len(self.__branches)):
            sizes[f"{Slot.eps_v}{i}"] = 6
        self.__layout = StateLayout.From(sizes)

        # the local problem collapses to one scalar when the surface is quadratic and nothing
        # else evolves; the decomposition it runs in is built here, once, not per Gauss point
        self.solver = solver
        self.__eigen = (
            _spectral.Build(*elastic.Get_sqrt_C_S(), yieldSurface.P)
            if self.__Is_reducible()
            else None
        )

    def __Is_reducible(self) -> bool:
        """Whether the spectral return applies: quadratic surface, homogeneous C, nothing else."""
        return (
            self.solver != "newton"
            and self.__yield is not None
            and self.__yield.P is not None
            and np.ndim(self.C) == 2
            and not self.__kinematic
            and not self.__branches
        )

    # --------------------------------------------------------------------------
    # Model interface
    # --------------------------------------------------------------------------

    @property
    def modelType(self) -> ModelType:
        # a Behavior solves for displacement, so it reports the displacement problem type
        return ModelType.elastic

    @property
    def elastic(self) -> _Elastic:
        """Elastic model supplying the 3D Kelvin stiffness."""
        return self.__elastic

    @property
    def layout(self) -> StateLayout:
        """Packed layout of the internal variables."""
        return self.__layout

    @property
    def C(self) -> _types.FloatArray:
        """3D elastic stiffness in Kelvin-Mandel notation (6, 6)."""
        return self.__elastic.C

    @property
    def coef(self) -> float:
        """Kelvin-Mandel coefficient, used when projecting result fields."""
        return np.sqrt(2)

    @property
    def isHeterogeneous(self) -> bool:
        return self.__elastic.isHeterogeneous

    @property
    def simplification(self) -> str:
        if self.dim == 3:
            return "3D"
        return "Plane Stress" if self.planeStress else "Plane Strain"

    def __str__(self) -> str:
        text = f"{type(self).__name__}:"
        text += f"\nelastic = {type(self.__elastic).__name__}"
        if self.dim == 2:
            text += f"\nthickness = {self.thickness:.2e} ({self.simplification})"
        return text

    def State_zeros(self, Ne: int, nPg: int) -> FeArray.FeArrayALike:
        """A zeroed packed state — the virgin material."""
        return FeArray.zeros(Ne, nPg, self.__layout.n, dtype=float)

    # --------------------------------------------------------------------------
    # Free energy
    # --------------------------------------------------------------------------

    def _C_e_pg(self, Ne: int, nPg: int) -> FeArray.FeArrayALike:
        return FeArray.broadcast(self.C, Ne, nPg, tensor_ndim=2)

    def Compute_back_stress(self, z_e_pg: FeArray) -> FeArray.FeArrayALike:
        r"""Back-stress :math:`X = \dpartial{\psi}{\beta}`, zero without kinematic hardening."""
        X_e_pg: FeArray.FeArrayALike = 0.0
        for i, component in enumerate(self.__kinematic):
            slot = self.__layout.slots[f"{Slot.alpha}{i}"]
            X_e_pg = X_e_pg + component.X(z_e_pg[..., slot])
        return X_e_pg

    def __State(
        self, eps_e_pg: FeArray.FeArrayALike, z_e_pg: Optional[FeArray]
    ) -> FeArray:
        """The given state, or the virgin one — so the potentials can be asked about bare strain."""
        if z_e_pg is not None:
            return z_e_pg
        Ne, nPg = np.shape(eps_e_pg)[:2]
        return self.State_zeros(Ne, nPg)

    def Compute_psi(
        self, eps_e_pg: FeArray.FeArrayALike, z_e_pg: Optional[FeArray] = None
    ) -> FeArray.FeArrayALike:
        r"""Free energy: the springs, the Maxwell branches and the stored hardening, in 3D Kelvin."""
        z_e_pg = self.__State(eps_e_pg, z_e_pg)
        eel_e_pg = self.Compute_elastic_strain(eps_e_pg, z_e_pg)
        Ne, nPg = eel_e_pg.shape[:2]
        C_e_pg = self._C_e_pg(Ne, nPg)

        g_eq = 1.0 - sum(b.g for b in self.__branches)
        psi_e_pg = 0.5 * g_eq * eel_e_pg.dot(C_e_pg @ eel_e_pg)

        for i, branch in enumerate(self.__branches):
            d = eel_e_pg - z_e_pg[..., self.__layout.slots[f"{Slot.eps_v}{i}"]]  # type: ignore[index]
            psi_e_pg = psi_e_pg + 0.5 * branch.g * d.dot(C_e_pg @ d)

        slot = self.__layout.slots.get(Slot.p)
        if slot is not None:
            psi_e_pg = psi_e_pg + self.__hardening.psi(z_e_pg[..., slot][..., 0])  # type: ignore[index]

        for i, component in enumerate(self.__kinematic):
            slot = self.__layout.slots[f"{Slot.alpha}{i}"]
            psi_e_pg = psi_e_pg + component.psi(z_e_pg[..., slot])  # type: ignore[index]
        return psi_e_pg

    def Compute_elastic_strain(
        self, eps_e_pg: FeArray.FeArrayALike, z_e_pg: Optional[FeArray] = None
    ) -> FeArray.FeArrayALike:
        r""":math:`\Eps^e = \Eps - \Eps^p`, in 3D Kelvin."""
        slot = self.__layout.slots.get(Slot.eps_p)
        if slot is None:
            return eps_e_pg
        return eps_e_pg - self.__State(eps_e_pg, z_e_pg)[..., slot]

    def Compute_sigma(
        self, eps_e_pg: FeArray.FeArrayALike, z_e_pg: Optional[FeArray] = None
    ) -> FeArray.FeArrayALike:
        r""":math:`\Sig = \dpartial{\psi}{\Eps} = \Crm : \Eps^e - \sum_i g_i \Crm : \Eps^v_i`, in 3D Kelvin."""
        z_e_pg = self.__State(eps_e_pg, z_e_pg)
        eel_e_pg = self.Compute_elastic_strain(eps_e_pg, z_e_pg)
        Ne, nPg = eel_e_pg.shape[:2]
        C_e_pg = self._C_e_pg(Ne, nPg)

        sig_e_pg = C_e_pg @ eel_e_pg
        for i, branch in enumerate(self.__branches):
            slot = self.__layout.slots[f"{Slot.eps_v}{i}"]
            sig_e_pg = sig_e_pg - branch.g * (C_e_pg @ z_e_pg[..., slot])  # type: ignore[index]
        return sig_e_pg

    def Compute_stress(
        self, eps_e_pg: FeArray.FeArrayALike, z_e_pg: Optional[FeArray] = None
    ) -> FeArray.FeArrayALike:
        r"""Stress at the given state, in the model dimension.

        Reads the state; it does not advance it. :meth:`Integrate` is the one that flows,
        relaxes and moves time on — calling it to *read* a stress would step a
        rate-dependent material forward again.
        """
        eps6_e_pg = self.Compute_strain_6d(eps_e_pg, z_e_pg, 0.0)
        sig6_e_pg = self.Compute_sigma(eps6_e_pg, z_e_pg)
        if self.dim == 3:
            return sig6_e_pg
        return sig6_e_pg[..., IDX_2D]

    # --------------------------------------------------------------------------
    # Kinematics — 2D/3D handled once
    # --------------------------------------------------------------------------

    def Compute_strain_6d(
        self,
        eps_e_pg: FeArray.FeArrayALike,
        zOld_e_pg: Optional[FeArray] = None,
        dt: float = 0.0,
    ) -> FeArray:
        r"""The 3D Kelvin strain the material actually sees.

        Plane strain leaves ``eps_zz = 0``. Plane stress solves it so that ``sig_zz = 0``,
        which is a *solve* rather than a formula once the material can flow.
        """
        eps_e_pg = FeArray.asfearray(eps_e_pg)
        if self.dim == 3:
            return eps_e_pg

        eps6_e_pg = FeArray.zeros(*eps_e_pg.shape[:2], 6, dtype=float)
        eps6_e_pg[..., IDX_2D] = eps_e_pg
        if not self.planeStress:
            return eps6_e_pg

        if zOld_e_pg is None:
            zOld_e_pg = self.State_zeros(*eps_e_pg.shape[:2])
        return self.__Plane_stress_strain(eps6_e_pg, zOld_e_pg, dt)

    def __Plane_stress_strain(
        self, eps6_e_pg: FeArray, zOld_e_pg: FeArray, dt: float
    ) -> FeArray:
        r"""Solves ``eps_zz`` so that ``sig_zz = 0``, through the material's own response.

        Linear while elastic, so one step converges. Once the material flows the elastic
        closed form is wrong — it leaves a large ``sig_zz`` behind — hence a Newton driven
        by the algorithmic ``C[zz, zz]``.
        """
        eps6_e_pg = eps6_e_pg.copy()
        # sig_zz cannot be driven below the stress equivalent of the inner solve's accuracy:
        # that solves strains to _tol, which is a stress of about C[zz, zz] * _tol. Asking for
        # less than that stalls rather than converges.
        scale = self.__yield.scale if self.__yield is not None else 1.0
        floor = 10.0 * self._tol * float(np.max(self.C[..., ZZ, ZZ]))
        tol = max(self._planeStress_tol * max(scale, 1.0), floor)
        for _ in range(self._maxIter):
            sig6_e_pg, C6_e_pg, _, _ = self.__Integrate_3d(eps6_e_pg, zOld_e_pg, dt)
            r_e_pg = sig6_e_pg[..., ZZ]
            if np.max(np.abs(r_e_pg)) < tol:
                break
            eps_zz = eps6_e_pg[..., ZZ] - r_e_pg / C6_e_pg[..., ZZ, ZZ]
            eps6_e_pg[..., ZZ] = eps_zz
        else:
            raise AssertionError(
                f"plane-stress sig_zz = 0 did not converge in {self._maxIter} iterations "
                f"(max |sig_zz| = {np.max(np.abs(r_e_pg)):.3e})"
            )
        return eps6_e_pg

    def __Integrate_3d(
        self, eps6_e_pg: FeArray, zOld_e_pg: FeArray, dt: float
    ) -> tuple[FeArray, FeArray, FeArray, _types.FloatArray]:
        """The 3D response: the local solve, or plain elasticity when there is no state."""
        Ne, nPg = eps6_e_pg.shape[:2]
        C6_e_pg = self._C_e_pg(Ne, nPg)
        if self.__layout.n == 0:
            return (
                self.Compute_sigma(eps6_e_pg, zOld_e_pg),
                C6_e_pg,
                zOld_e_pg,
                np.ones((Ne, nPg), dtype=bool),
            )
        if self.__eigen is not None:
            return self.__Spectral(eps6_e_pg, zOld_e_pg, C6_e_pg, dt)
        return self.__Flow(eps6_e_pg, zOld_e_pg, C6_e_pg, dt)

    def __Spectral(
        self, eps6_e_pg: FeArray, zOld_e_pg: FeArray, C_e_pg: FeArray, dt: float
    ) -> tuple[FeArray, FeArray, FeArray, _types.FloatArray]:
        """One scalar unknown per Gauss point; see :mod:`._spectral`."""
        layout = self.__layout
        P, A = layout.slots[Slot.eps_p], layout.slots[Slot.p]

        epsP_e_pg = zOld_e_pg[..., P]
        pOld_e_pg = zOld_e_pg[..., A][..., 0]
        sigTr_e_pg = C_e_pg @ (eps6_e_pg - epsP_e_pg)

        res = _spectral.Solve(
            self.__eigen,
            sigTr_e_pg,
            pOld_e_pg,
            self.__hardening,
            self.__yield.scale,
            self.__rate,
            dt,
            self._tol,
            self._maxIter,
        )

        z_e_pg = FeArray.zeros(*eps6_e_pg.shape[:2], layout.n)
        # eps - C^-1 sigma is the plastic strain, with no flow direction to reconstruct
        Cinv_e_pg = FeArray.broadcast(
            self.__eigen.Cinv, *eps6_e_pg.shape[:2], tensor_ndim=2
        )
        z_e_pg[..., P] = eps6_e_pg - Cinv_e_pg @ res.sig
        z_e_pg[..., A.start] = pOld_e_pg + res.dGamma

        C_alg = _spectral.Tangent(self.__eigen, res, C_e_pg)
        converged = np.ones(eps6_e_pg.shape[:2], dtype=bool)
        return res.sig, C_alg, z_e_pg, converged

    def __Condense(self, C_e_pg: FeArray) -> FeArray:
        """Static condensation of the zz row and column, giving the in-plane tangent."""
        C_in = C_e_pg[..., IDX_2D, :][..., :, IDX_2D]
        c_iz = C_e_pg[..., IDX_2D, ZZ]
        c_zi = C_e_pg[..., ZZ, :][..., IDX_2D]
        c_zz = C_e_pg[..., ZZ, ZZ]
        return C_in - TensorProd(c_iz, c_zi) / c_zz

    # --------------------------------------------------------------------------
    # The local solve
    # --------------------------------------------------------------------------

    def __Residual(
        self,
        eps6_e_pg: FeArray,
        u_e_pg: FeArray,
        zOld_e_pg: FeArray,
        C_e_pg: FeArray,
        dt: float,
    ) -> tuple[FeArray, FeArray, Optional[FeArray], Optional[FeArray]]:
        r"""``r(u) = 0``, over the state *increments* followed by :math:`\Delta\gamma`.

        .. math::
            r_{v,i} &= \Delta\Eps^v_i - \frac{\dt}{\tau_i}(\Eps^e - \Eps^v_i) \\
            r_p &= \Delta\Eps^p - \Delta\gamma\, N \\
            r_\alpha &= \Delta\alpha - \Delta\gamma \\
            r_f &= f(\Sig, R) - \phi^{-1}(\Delta\gamma/\dt)

        The unknowns are increments, as in MFront's implicit DSL, so every row is a change and
        the committed state never appears on both sides of a subtraction. The values the laws
        are evaluated at are ``zOld + du``.

        The rate term is absent without a rate law, which recovers ``f = 0``.

        Returns ``(r, sig, N, dNdSig)`` — the last two are reused by the Jacobian rather than
        recomputed.
        """
        layout = self.__layout
        nz = layout.n
        z_e_pg = zOld_e_pg + u_e_pg[..., :nz]

        eel_e_pg = self.Compute_elastic_strain(eps6_e_pg, z_e_pg)
        sig_e_pg = self.Compute_sigma(eps6_e_pg, z_e_pg)
        # the surface is read at the shifted stress: kinematic hardening moves its centre
        xi_e_pg = sig_e_pg - self.Compute_back_stress(z_e_pg)

        r_e_pg = FeArray.zeros(u_e_pg.shape)

        for i, branch in enumerate(self.__branches):
            slot = layout.slots[f"{Slot.eps_v}{i}"]
            r_e_pg[..., slot] = u_e_pg[..., slot] - (dt / branch.tau) * (
                eel_e_pg - z_e_pg[..., slot]
            )

        N_e_pg = dNdSig_e_pg = None
        if self.__yield is not None:
            P, A = layout.slots[Slot.eps_p], layout.slots[Slot.p]
            alpha_e_pg = z_e_pg[..., A][..., 0]
            dG_e_pg = u_e_pg[..., nz]
            R_e_pg = self.__hardening.R(alpha_e_pg)
            N_e_pg = self.__yield.N(xi_e_pg, R_e_pg)
            dNdSig_e_pg = self.__yield.dNdSig(xi_e_pg)

            r_e_pg[..., P] = u_e_pg[..., P] - dG_e_pg * N_e_pg
            r_e_pg[..., A.start] = u_e_pg[..., A][..., 0] - dG_e_pg
            r_e_pg[..., nz] = self.__yield.f(xi_e_pg, R_e_pg)
            if self.__rate is not None:
                r_e_pg[..., nz] -= self.__rate.inverse(u_e_pg[..., nz] / dt)

            for i, component in enumerate(self.__kinematic):
                B = self.__layout.slots[f"{Slot.alpha}{i}"]
                r_e_pg[..., B] = u_e_pg[..., B] - dG_e_pg * (
                    N_e_pg - component.recall * z_e_pg[..., B]
                )

        return r_e_pg, sig_e_pg, N_e_pg, dNdSig_e_pg

    def __Jacobian(
        self,
        u_e_pg: FeArray,
        zOld_e_pg: FeArray,
        N_e_pg: Optional[FeArray],
        dNdSig_e_pg: Optional[FeArray],
        C_e_pg: FeArray,
        dt: float,
    ) -> tuple[FeArray, FeArray]:
        r"""``(dr/du, dr/deps)``, assembled by chain rule from the pieces' own derivatives.

        Nothing is differentiated numerically: ``dNdSig`` comes from the yield surface, ``dR``
        from the hardening, and the stress sensitivities are read off
        :math:`\Sig = \Crm : (\Eps - \Eps^p) - \sum_i g_i \Crm : \Eps^v_i`.
        """
        layout = self.__layout
        nz, nu = layout.n, u_e_pg.shape[-1]
        Ne, nPg = u_e_pg.shape[:2]
        I6 = np.eye(6)
        # dr/du is the same matrix whether the unknown is the state or its increment, since the
        # committed state is a constant -- but the two rows below read a state *value*
        z_e_pg = zOld_e_pg + u_e_pg[..., :nz]

        J_e_pg = FeArray.zeros(Ne, nPg, nu, nu)
        D_e_pg = FeArray.zeros(Ne, nPg, nu, 6)

        for i, branch in enumerate(self.__branches):
            slot = layout.slots[f"{Slot.eps_v}{i}"]
            theta = dt / branch.tau
            J_e_pg[..., slot, slot] = (1.0 + theta) * I6
            D_e_pg[..., slot, :] = -theta * I6
            if Slot.eps_p in layout.slots:
                J_e_pg[..., slot, layout.slots[Slot.eps_p]] = theta * I6

        if self.__yield is not None:
            P, A = layout.slots[Slot.eps_p], layout.slots[Slot.p]
            dG_e_pg = u_e_pg[..., nz, None, None]
            alpha_e_pg = z_e_pg[..., A][..., 0]

            NC_e_pg = N_e_pg @ C_e_pg
            dNdSig_C = dNdSig_e_pg @ C_e_pg
            dR_e_pg = self.__hardening.dR(alpha_e_pg)

            J_e_pg[..., P, P] = I6 + dG_e_pg * dNdSig_C
            J_e_pg[..., P, nz] = -N_e_pg
            J_e_pg[..., A.start, A.start] = 1.0
            J_e_pg[..., A.start, nz] = -1.0
            J_e_pg[..., nz, P] = -NC_e_pg
            # f = phi - sigma_y - R, so df/dp = -dR
            J_e_pg[..., nz, A.start] = -dR_e_pg
            if self.__rate is not None:
                dInv_e_pg = self.__rate.dinverse(u_e_pg[..., nz] / dt)
                J_e_pg[..., nz, nz] = -dInv_e_pg / dt

            D_e_pg[..., P, :] = -dG_e_pg * dNdSig_C
            D_e_pg[..., nz, :] = NC_e_pg

            # each back-stress shifts xi by -k_j alpha_j, so component i is coupled to every
            # component j through the flow direction: dxi/dalpha_j = -k_j
            for i, component in enumerate(self.__kinematic):
                Bi = layout.slots[f"{Slot.alpha}{i}"]
                J_e_pg[..., P, Bi] = component.modulus * dG_e_pg * dNdSig_e_pg
                J_e_pg[..., nz, Bi] = -component.modulus * N_e_pg
                # r_alpha_i = alpha_i - alpha_i_n - dG (N - recall_i alpha_i)
                J_e_pg[..., Bi, P] = dG_e_pg * dNdSig_C
                J_e_pg[..., Bi, nz] = -(N_e_pg - component.recall * z_e_pg[..., Bi])
                D_e_pg[..., Bi, :] = -dG_e_pg * dNdSig_C
                for j, other in enumerate(self.__kinematic):
                    Bj = layout.slots[f"{Slot.alpha}{j}"]
                    block = other.modulus * dG_e_pg * dNdSig_e_pg
                    if i == j:
                        block = block + I6 * (1.0 + dG_e_pg * component.recall)
                    J_e_pg[..., Bi, Bj] = block

            # the branches move sigma, which the flow direction and the surface both see
            for i, branch in enumerate(self.__branches):
                slot = layout.slots[f"{Slot.eps_v}{i}"]
                J_e_pg[..., P, slot] = branch.g * dG_e_pg * dNdSig_C
                J_e_pg[..., nz, slot] = -branch.g * NC_e_pg

        return J_e_pg, D_e_pg

    @staticmethod
    def __Pin(
        J_e_pg: FeArray,
        D_e_pg: Optional[FeArray],
        r_e_pg: FeArray,
        mask_e_pg: FeArray,
        row: int,
        value: FeArray,
    ) -> None:
        """Replaces one equation by ``u[row] = u[row] - value`` at the masked points."""
        Jm = J_e_pg[mask_e_pg]
        Jm[:, row, :] = 0.0
        Jm[:, row, row] = 1.0
        J_e_pg[mask_e_pg] = Jm

        rm = r_e_pg[mask_e_pg]
        rm[:, row] = value
        r_e_pg[mask_e_pg] = rm

        if D_e_pg is not None:
            Dm = D_e_pg[mask_e_pg]
            Dm[:, row, :] = 0.0
            D_e_pg[mask_e_pg] = Dm

    def __Freeze(
        self,
        J_e_pg: FeArray,
        D_e_pg: Optional[FeArray],
        r_e_pg: FeArray,
        u_e_pg: FeArray,
        active_e_pg: FeArray,
    ) -> None:
        """A point that is not flowing holds ``dGamma = 0``.

        Viscous branches keep evolving either way — relaxation does not need a yield surface.
        """
        if self.__yield is None:
            return
        nz = self.__layout.n

        idle = ~active_e_pg
        if idle.any():
            self.__Pin(J_e_pg, D_e_pg, r_e_pg, idle, nz, u_e_pg[idle][:, nz])

    def __Norm(self, r_e_pg: FeArray, active_e_pg: FeArray) -> FeArray:
        r"""Residual size per point, on one scale.

        Most rows are strains; the yield row is a stress, so it is divided by the surface
        scale. Mixing the two unscaled makes the line search meaningless — the stress row
        dominates by orders of magnitude and no step ever looks like an improvement.
        """
        rn_e_pg = np.abs(r_e_pg).copy()
        if self.__yield is not None:
            nz = self.__layout.n
            rn_e_pg[..., nz] = np.where(
                active_e_pg, rn_e_pg[..., nz] / max(self.__yield.scale, 1.0), 0.0
            )
        return np.max(rn_e_pg, axis=-1)

    def __Converged(self, r_e_pg: FeArray, active_e_pg: FeArray) -> FeArray:
        """Every residual row small, on the scale set by :meth:`__Norm`."""
        return self.__Norm(r_e_pg, active_e_pg) < self._tol

    def __Bound(self, u_e_pg: FeArray) -> FeArray:
        r"""Holds the unknowns inside their admissible range: ``dGamma >= 0``."""
        u_e_pg = u_e_pg.copy()
        if self.__yield is not None:
            nz = self.__layout.n
            u_e_pg[..., nz] = np.maximum(u_e_pg[..., nz], 0.0)
        return u_e_pg

    def __Flow(
        self, eps6_e_pg: FeArray, zOld_e_pg: FeArray, C_e_pg: FeArray, dt: float
    ) -> tuple[FeArray, FeArray, FeArray, FeArray]:
        """Newton on the residual at every Gauss point, then the tangent from its Jacobian."""
        layout = self.__layout
        Ne, nPg = eps6_e_pg.shape[:2]
        nz = layout.n
        nu = nz + (1 if self.__yield is not None else 0)

        # start from the committed state: nothing has flowed or relaxed yet
        u = FeArray.zeros(Ne, nPg, nu)

        r, sig, N, dNdSig = self.__Residual(eps6_e_pg, u, zOld_e_pg, C_e_pg, dt)

        if self.__yield is not None:
            active = r[..., nz] > 0.0  # f(sig_trial, R_n) > 0
            if self.__rate is not None:
                # dinverse is unbounded at zero flow, so start from the explicit rate estimate
                # rather than from dGamma = 0, where Newton would not move
                u[..., nz] = np.where(active, dt * self.__rate.rate(r[..., nz]), 0.0)
                r, sig, N, dNdSig = self.__Residual(eps6_e_pg, u, zOld_e_pg, C_e_pg, dt)
        else:
            active = FeArray.zeros(Ne, nPg, dtype=bool)

        converged = self.__Converged(r, active)
        for _ in range(self._maxIter):
            if converged.all():
                break
            J, _ = self.__Jacobian(u, zOld_e_pg, N, dNdSig, C_e_pg, dt)
            self.__Freeze(J, None, r, u, active)
            u = self.__Bound(u - np.linalg.solve(J, r[..., None])[..., 0])
            r, sig, N, dNdSig = self.__Residual(eps6_e_pg, u, zOld_e_pg, C_e_pg, dt)
            converged = self.__Converged(r, active)

        z_e_pg = (zOld_e_pg + u[..., :nz]).copy()

        J, D = self.__Jacobian(u, zOld_e_pg, N, dNdSig, C_e_pg, dt)
        self.__Freeze(J, D, r, u, active)
        # dz/deps = -inv(dr/dz)(dr/deps), then C_alg = dsig/deps
        dudeps = -np.linalg.solve(J, D)

        C_alg = C_e_pg.copy()
        if Slot.eps_p in layout.slots:
            C_alg = C_alg - C_e_pg @ dudeps[..., layout.slots[Slot.eps_p], :]
        for i, branch in enumerate(self.__branches):
            C_alg = C_alg - branch.g * (
                C_e_pg @ dudeps[..., layout.slots[f"{Slot.eps_v}{i}"], :]
            )

        return sig, C_alg, z_e_pg, converged

    # --------------------------------------------------------------------------
    # Integration
    # --------------------------------------------------------------------------

    def Integrate(
        self,
        eps_e_pg: FeArray,
        zOld_e_pg: Optional[FeArray] = None,
        dt: float = 0.0,
        epsOld_e_pg: Optional[FeArray] = None,
        fields: Optional[dict[str, FeArray]] = None,
        withTangent: bool = True,
    ) -> tuple[FeArray, Optional[FeArray], FeArray, FeArray]:
        """Give it the total strain, it gives back the stress. At every Gauss point.

        Pure: it reads the committed state, it never writes it. The caller decides when the new
        state becomes the committed one.

        Parameters
        ----------
        eps_e_pg : FeArray
            Total strain ``(Ne, nPg, nstrain)``, Kelvin-Mandel, in the model dimension.
        zOld_e_pg : FeArray, optional
            Packed state ``(Ne, nPg, n)`` committed at the last converged step; zeros by default.
        dt : float, optional
            Time increment; required by a rate-dependent behavior, ignored otherwise.
        epsOld_e_pg : FeArray, optional
            Total strain ``(Ne, nPg, nstrain)`` at the last converged step, supplied by the
            solver and never stored --
            the state stays exactly the history variables. Only local sub-stepping needs it, so
            leaving it out costs that and nothing else. See MFront's `eto`/`deto`, Abaqus'
            STRAN/DSTRAN and NEML's `e_n`.
        fields : dict[str, FeArray], optional
            External fields (temperature, ...). Not read yet — the seam for thermo-mechanical
            coupling.
        withTangent : bool, optional
            Build the consistent tangent, by default True.

        Returns
        -------
        sigma : FeArray
            Stress ``(Ne, nPg, nstrain)``, Kelvin-Mandel.
        C_alg : FeArray or None
            Consistent tangent ``(Ne, nPg, nstrain, nstrain)``; None if withTangent is False.
        z : FeArray
            Trial state ``(Ne, nPg, n)`` — the caller commits it only once the global step
            converges.
        converged : FeArray
            Boolean ``(Ne, nPg)``.
        """
        assert (
            not fields
        ), "external fields are not read yet (thermo-mechanical coupling)"
        assert self.__rate is None or dt > 0.0, (
            "a rate-dependent behavior needs a positive time increment; "
            "set `simu.dt` or pass `dt=` to Integrate"
        )

        tic = Tic()
        eps_e_pg = FeArray.asfearray(eps_e_pg)
        Ne, nPg = eps_e_pg.shape[:2]
        if zOld_e_pg is None:
            zOld_e_pg = self.State_zeros(Ne, nPg)

        eps6_e_pg = self.Compute_strain_6d(eps_e_pg, zOld_e_pg, dt)

        sig6_e_pg, C6alg_e_pg, z_e_pg, converged_e_pg = self.__Integrate_3d(
            eps6_e_pg, zOld_e_pg, dt
        )

        sig_e_pg = sig6_e_pg
        C_e_pg: Optional[FeArray] = C6alg_e_pg if withTangent else None
        if self.dim == 2:
            sig_e_pg = sig6_e_pg[..., IDX_2D]
            if C_e_pg is not None:
                C_e_pg = (
                    self.__Condense(C6alg_e_pg)
                    if self.planeStress
                    else C6alg_e_pg[..., IDX_2D, :][..., :, IDX_2D]
                )

        tic.Tac("Matrix", "Behavior integrate", False)
        return sig_e_pg, C_e_pg, z_e_pg, converged_e_pg
