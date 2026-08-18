# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information

from enum import Enum

import numpy as np
from typing import Union, Optional, TYPE_CHECKING

# utilities
from ..Utilities import Terminal, _types

# fem
if TYPE_CHECKING:
    from ..FEM import Mesh
from ..FEM import MatrixType, Operators
from ..Utilities._cache import cache_computed_values

# models
from ..Models import ModelType, Project_Kelvin, Result_strain_or_stress_field_e

if TYPE_CHECKING:
    from ..Models.HyperElastic._laws import _HyperElastic
from ..Models.HyperElastic._state import HyperElasticState

# simu
from ._simu import _Simu, AlgoType


class HyperElastic(_Simu):
    r"""Finite-strain (large-deformation) hyperelastic simulation, total-Lagrangian framework.

    Solves the nonlinear static or dynamic equilibrium of a hyperelastic body with a Newton-Raphson scheme, with optional Kelvin-Voigt viscosity and a fiber active stress (e.g. for cardiac mechanics). Material behavior is supplied by a hyperelastic model (Saint-Venant-Kirchhoff, Neo-Hookean, Mooney-Rivlin, Holzapfel-Ogden, …).

    Weak form:

    .. math::
        R(\ub; \vb) = \int_{\Omega_0} \boldsymbol{\Sigma}(\ub) : \Drm_\ub \eb(\ub) \cdot \vb \, \dO +
        \int_{\Omega_0} \rho \, \ddot{\ub} \cdot \vb \, \dO  - \int_{\partial\Omega_0^t} \tb\cdot\vb \, \dS - \int_{\Omega_0} \fb\cdot\vb \, \dO \quad \forall \, \vb \in V

    where :math:`\boldsymbol{\Sigma} := J \, \Fb^{-1} \cdot \Sig \cdot \Fb^{-T}`, is the second Piola Kirchhoff stress tensor (PK2), :math:`\eb := \frac{1}{2} \left( \Cb - \boldsymbol{1} \right) = \frac{1}{2} \left( \Fb^T \cdot \Fb - \boldsymbol{1} \right)` is the Green-Lagrange strain tensor and :math:`\Fb := \boldsymbol{1} +  \grad \ub` the deformation gradient.

    The total PK2 stress combines the elastic response with two optional contributions:

    .. math::
        \boldsymbol{\Sigma}(\ub, \dot{\ub}) = \dpartial{W}{\eb}(\ub) + \tau \, \hat{\Tb} \otimes \hat{\Tb} + \eta \, \dot{\eb}(\ub, \dot{\ub})

    Only the first term derives from the stored energy; the other two are non-conservative and are therefore assembled by their own operators rather than folded into ``material.Compute_dWde``, which stays exactly :math:`\partial W / \partial \eb`.

    - **Active stress** :math:`\tau \, \hat{\Tb} \otimes \hat{\Tb}`: a contractile stress of magnitude :math:`\tau` (``material.active_stress``) acting along the unit fiber direction :math:`\hat{\Tb}`, registered once with ``material.Set_active_stress_vec``. It is strain-independent and typically used for cardiac mechanics, where only :math:`\tau` is updated between time steps. Delivered by :func:`Operators.NonLinear.ActiveStressTensor` (internal force + geometric tangent, no material tangent).
    - **Kelvin-Voigt viscosity** :math:`\eta \, \dot{\eb}`: a rate-dependent stress proportional to the Green-Lagrange strain rate :math:`\dot{\eb}` (:math:`\eta` = ``material.eta``), active only in dynamic simulations where a velocity field is available. It is delivered through a damping matrix (and a configuration tangent), mirroring Rayleigh damping in :class:`Elastic`.

    This non linear problem is solve using the newton rapshon algorithm:

    .. math::
        A(\ub; \vb, \wb) \, \Delta \ub = - R(\ub; \vb) \quad \forall \, (\vb, \wb) \in \Vc \times \Wc,

    where the tangent :math:`A(\ub; \vb, \wb)` is defined :math:`\forall \, (\vb, \wb) \in \Vc \times \Wc` as:

    .. math::
        A(\ub; \vb, \wb) &=
        \dpartial{R(\ub; \vb)}{\ub} \cdot \wb \\ &=
        \int_{\Omega_0} \Drm_\ub \eb(\ub) \cdot \wb : \dNpartial{2}{W}{\eb}(\ub) : \Drm_\ub \eb(\vb) \, \dO +
        \int_{\Omega_0} \dpartial{W}{\eb}(\ub) : \Drm_\ub^2 \eb(\vb, \wb) \, \dO

    The implemented hyperelastic laws are available :ref:`here <models-hyperelastic>` and were constructed by the :ref:`ComputeHyperelasticLaws` script.
    """

    class StressType(str, Enum):
        """Which PK2 stress the internal force uses. All three solve the same continuous problem; they differ in how the stress is sampled over a step, hence in whether the discrete total energy is conserved."""

        pointwise = "pointwise"
        r"""Default. :math:`\Srm(\eb(\ub^t))` at the time scheme's evaluation state, assembled by :func:`~EasyFEA.FEM.Operators.NonLinear.SecondPiolaKirchhoffStressTensor`. Energy drifts."""
        gonzalez = "gonzalez"
        r"""Energy-momentum discrete gradient :math:`\hat{\Srm} = \bar{\Srm} + \alpha \Delta \eb`, assembled by :func:`~EasyFEA.FEM.Operators.NonLinear.GonzalezStressTensor`. Conserves :math:`\mathrm{KE} + W` exactly, for any law, from one stress evaluation."""
        quadrature = "quadrature"
        r"""Strain-path average :math:`\int_0^1 \dpartial{W}{\eb}(\eb^n + s \Delta \eb) \, \drm s`, assembled by :func:`~EasyFEA.FEM.Operators.NonLinear.TimeQuadratureStressTensor`. Conservation is exact only up to the quadrature error, which falls spectrally with ``nPoints``."""

        def __str__(self) -> str:
            return self.name

    @staticmethod
    def Get_stressTypes() -> list["HyperElastic.StressType"]:
        """Returns the available stresses."""
        return list(HyperElastic.StressType)

    def __init__(
        self,
        mesh: "Mesh",
        model: "_HyperElastic",
        folder: str = "",
        absTol: float = 1e-6,
        relTol: float = 1e-10,
        incTol: float = 1e-11,
        maxIter: int = 20,
        verbosity: bool = False,
    ):
        """Creates a hyperelastic simulation.

        Parameters
        ----------
        mesh : Mesh
            The mesh used.
        model : _HyperElas
            The hyperelatic model used.
        folder : str, optional
            save folder, by default "".
        absTol : float, optional
            absolute tolerance, by default 1e-6
        relTol : float, optional
            relative tolerance, by default 1e-10
        incTol : float, optional
            incremental tolerance, by default 1e-11
        maxIter : int, optional
            maximum iteration, by default 20
        verbosity : bool, optional
            If True, iterative solvers can be used. Defaults to True.

        WARNING
        -------
        2D simulations are conducted under the **plane strain** assumption.
        """

        super().__init__(mesh, model, folder, verbosity)

        self._Solver_Set_Newton_Raphson_Algorithm(
            absTol=absTol,
            relTol=relTol,
            incTol=incTol,
            maxIter=maxIter,
        )

        self.Solver_Set_Stress()

    # --------------------------------------------------------------------------
    # General
    # --------------------------------------------------------------------------

    def Get_problemTypes(self):
        return [ModelType.hyperelastic]

    def Get_unknowns(self, problemType=None) -> list[str]:
        dict_unknowns = {2: ["x", "y"], 3: ["x", "y", "z"]}
        return dict_unknowns[self.dim]

    def Get_dof_n(self, problemType=None) -> int:
        return self.dim

    @property
    def material(self) -> "_HyperElastic":
        """hyperelastic material"""
        return self.model  # type: ignore [return-value]

    @property
    def displacement(self) -> _types.FloatArray:
        """Displacement vector field.\n
        [uxi, uyi, uzi, ...]"""
        return self._Get_u_n(self.problemType)

    @property
    def speed(self) -> _types.FloatArray:
        """Velocity vector field.\n
        2D [vxi, vyi, ...]\n
        3D [vxi, vyi, vzi, ...]"""
        return self._Get_v_n(self.problemType)

    @property
    def accel(self) -> _types.FloatArray:
        """Acceleration vector field.\n
        2D [axi, ayi, ...]\n
        3D [axi, ayi, azi, ...]"""
        return self._Get_a_n(self.problemType)

    # --------------------------------------------------------------------------
    # Solve
    # --------------------------------------------------------------------------

    def Get_x0(self, problemType=None):
        return self.displacement

    def Solver_Set_Stress(
        self,
        stressType: "HyperElastic.StressType" = StressType.pointwise,
        nPoints: int = 3,
        useConsistentTangent: bool = True,
        energyTol: Optional[float] = None,
    ) -> None:
        r"""Selects the stress used by the internal force. Call **after** :py:meth:`~EasyFEA.Simulations._Simu.Solver_Set_Hyperbolic_Algorithm`.

        None of these is a time scheme: the discretization stays exactly whatever
        :py:meth:`~EasyFEA.Simulations._Simu.Solver_Set_Hyperbolic_Algorithm` selected (same
        :math:`\urm/\vrm/\arm` update, same :math:`\mathrm{coef}_\Krm/\mathrm{coef}_\Crm/\mathrm{coef}_\Mrm`).
        Only the stress in the residual changes, which is what decides whether the discrete
        total energy :math:`\mathrm{KE} + W` is conserved. See :class:`StressType` for the three
        options and the operators for their construction.

        Energy conservation for both non-default stresses rests on the midpoint identity
        :math:`\Delta \eb = \Brm(\bar{\ub}) \cdot \Delta \ub`, so it holds only under
        :attr:`~EasyFEA.AlgoType.midpoint`. ``gonzalez`` is intrinsically a midpoint discrete gradient and is
        rejected off midpoint. ``quadrature`` also runs under any dynamic scheme — it scales its tangent by
        that scheme's ``coefK`` and stays Newton-consistent, but conserves energy only at midpoint.

        Parameters
        ----------
        stressType : HyperElastic.StressType, optional
            Which stress to use, by default ``pointwise``. Calling with no argument restores
            that default.
        nPoints : int, optional
            ``quadrature`` only: number of Clenshaw-Curtis points, by default 3 (Simpson).
            ``1`` and ``2`` are the midpoint and trapezoid rules; more converges spectrally.
            When ``energyTol`` is set this is the starting (minimum) level instead of a fixed count.
        useConsistentTangent : bool, optional
            ``gonzalez`` only: if False, drop the discrete-gradient corrections from the
            tangent. Same residual and same exact conservation, but Newton converges linearly
            — a diagnostic, to measure what the consistent tangent is worth.
        energyTol : float, optional
            ``quadrature`` only: if set, the rule is refined adaptively *element by element*
            along the nested Clenshaw-Curtis chain ``1, 3, 5, 9, ...`` (capped at 33) — each
            element stops once its own **integrated** relative energy defect
            ``∫|S_quad:Δe − ΔW| dΩ ≤ energyTol · ∫|ΔW| dΩ`` over that element is met (a volume
            integral over its Gauss points, not a pointwise density). So energy is conserved to
            ``energyTol`` while points are spent only where the step is nonlinear. ``None``
            (default) keeps the fixed ``nPoints`` rule.
        """
        stressType = HyperElastic.StressType(stressType)

        if stressType == HyperElastic.StressType.gonzalez:
            # gonzalez is the midpoint energy-momentum stress: its discrete gradient is built on ū and
            # conserves energy only there — intrinsically midpoint-only.
            assert self.algo == AlgoType.midpoint, (
                f"the 'gonzalez' stress requires AlgoType.midpoint (got {self.algo}). "
                "Call Solver_Set_Hyperbolic_Algorithm(dt, algo=AlgoType.midpoint) first."
            )
        # quadrature works with any dynamic scheme (its tangent is scaled by coefK); the dynamic-scheme
        # requirement is checked at assembly, so the algo need not be set before this call.
        assert nPoints >= 1, f"nPoints must be >= 1 (got {nPoints})."

        self.__stressParams = (stressType, nPoints, useConsistentTangent, energyTol)
        # diagnostic: per-element quadrature-point counts, last assembly
        self.__nPts_e = None

    def __Solver_Get_Stress_Params(
        self,
    ) -> tuple["HyperElastic.StressType", int, bool, Optional[float]]:
        """Returns (stressType, nPoints, useConsistentTangent, energyTol) internal-force props."""
        return self.__stressParams

    @property
    def stressType(self) -> "HyperElastic.StressType":
        """Stress used by the internal force — see :py:meth:`Solver_Set_Stress`."""
        return self.__Solver_Get_Stress_Params()[0]

    @cache_computed_values
    def __Mass_e(self, groupElem, rho, thickness, dim):
        """Constant element mass matrix ``thickness · ∫ρ N·N``."""
        return thickness * Operators.Bilinear.UV(groupElem, rho, dof_n=dim)

    def Construct_local_matrix_system(
        self,
        problemType,
        matrixType: MatrixType = MatrixType.rigi,
    ):
        """Returns ``{groupElem: (K_e, C_e, M_e, F_e)}`` for the current Newton iterate.

        Newton solves ``A(u)·Δu = -R(u)`` with ``A = coefK·K + coefC·C + coefM·M``. Per group of elements: ``K_e`` is the elastic tangent (plus the viscous configuration tangent ``Kgeo_e``), ``C_e`` the Kelvin–Voigt damping matrix, ``M_e`` the mass matrix (dynamic schemes only) and ``F_e = -R_e`` the complete residual — internal force together with the ``C_e·v_t + M_e·a_t`` inertia and damping terms.
        """
        dim = self.dim
        thickness = self.material.thickness if dim == 2 else 1
        isDynamic = self.algo in AlgoType.Get_Hyperbolic_Types()

        # current Newton-Raphson iterate; for dynamic schemes also capture the velocity
        # (Kelvin–Voigt viscosity) and the acceleration (inertia), both needed by F_e.
        displacement = self._Solver_Get_Newton_Raphson_current_solution()
        velocity = accel = None
        # Both non-default stresses are built from the step endpoints (u_n, u_{n+1}) on top of the scheme's
        # base point u_t (ū at midpoint), so both need u_{n+1} kept before the scheme evaluation overwrites
        # `displacement` with u_t. Re-checked here (not only in the setter) so that re-calling
        # Solver_Set_Hyperbolic_Algorithm with another algo can't leave a stale selection.
        stressType, nPoints, useConsistentTangent, energyTol = (
            self.__Solver_Get_Stress_Params()
        )
        isPointwise = stressType == HyperElastic.StressType.pointwise
        if stressType == HyperElastic.StressType.gonzalez:
            # gonzalez is the midpoint energy-momentum stress: its discrete gradient Ŝ = S̄ + α Δe is built
            # on ū and conserves energy only there — it is intrinsically midpoint-only.
            assert (
                self.algo == AlgoType.midpoint
            ), f"the 'gonzalez' stress requires AlgoType.midpoint (got {self.algo})."
        elif stressType == HyperElastic.StressType.quadrature:
            # quadrature builds a consistent tangent for any dynamic scheme via its `coefK = ∂u_t/∂u_{n+1}`;
            # energy is conserved only at midpoint (coefK = 0.5) — see TimeQuadratureStressTensor.
            assert (
                isDynamic
            ), f"the 'quadrature' stress requires a dynamic (hyperbolic) time scheme (got {self.algo})."
        u_np1 = displacement
        if isDynamic:
            displacement, velocity, accel = self._Solver_Evaluate_u_v_a_for_time_scheme(
                problemType, displacement
            )
        if not isPointwise:
            u_n = self._Get_u_n(problemType)

        errDetF = "det(F) < 0 - reduce load steps"

        out = {}
        list_nPts_e = []  # per-element point counts across groups (quadrature only)
        for groupElem in self.mesh.Get_list_groupElem():
            state = HyperElasticState(groupElem, displacement, matrixType)

            # invalid-element guard
            assert state.Compute_J().min() > 0, errDetF

            if isPointwise:
                # elastic tangent + residual; Newton: A(u) Δu = -R(u) = -F(u) + b
                K_e, residual_e = Operators.NonLinear.SecondPiolaKirchhoffStressTensor(
                    self.material, state
                )
            else:
                # `state` is the midpoint state ū, so only the two endpoint states are
                # built here; both energy-conserving stresses take the same three.
                state_n = HyperElasticState(groupElem, u_n, matrixType)
                state_np1 = HyperElasticState(groupElem, u_np1, matrixType)
                assert state_np1.Compute_J().min() > 0, errDetF

                hyperElasticStates = (state_n, state, state_np1)

                if stressType == HyperElastic.StressType.gonzalez:
                    K_e, residual_e = Operators.NonLinear.GonzalezStressTensor(
                        self.material,
                        *hyperElasticStates,
                        useConsistentTangent,
                    )
                elif stressType == HyperElastic.StressType.quadrature:
                    # coefK = ∂u_t/∂u_{n+1} scales the tangent for the active scheme (0.5 at midpoint).
                    coefK = self._Solver_Get_K_C_M_coefs_for_time_scheme()[0]
                    K_e, residual_e, nPts_e = (
                        Operators.NonLinear.TimeQuadratureStressTensor(
                            self.material,
                            *hyperElasticStates,
                            coefK,
                            nPoints,
                            energyTol,
                        )
                    )
                    list_nPts_e.append(nPts_e)
                else:
                    raise NotImplementedError

            # Active fiber stress τ·(T̂⊗T̂): a non-conservative stress, so it is
            # added here rather than inside Compute_dWde — that keeps the
            # constitutive stress a true ∂W/∂e, which the gonzalez discrete
            # gradient depends on. It contributes an internal force and a
            # geometric tangent (no material tangent: it is E-independent).
            if np.any(self.material.active_stress != 0.0):
                Kgeo_e, R_act_e = Operators.NonLinear.ActiveStressTensor(
                    self.material, state
                )
                K_e += Kgeo_e
                residual_e += R_act_e

            F_e = -residual_e

            # Kelvin–Voigt viscosity: C_e is the damping matrix (slot 2, so it rides coefC
            # in A), R_visco_e its residual, and Kgeo_e the configuration tangent
            # ∂(C·v)/∂u, added to K_e so it rides coefK.
            C_e = None
            if self.material.eta != 0 and velocity is not None:
                Kgeo_e, R_visco_e, C_e = Operators.NonLinear.KelvinVoigtDamping(
                    self.material, state, velocity
                )
                K_e += Kgeo_e
                F_e -= R_visco_e

            # mass matrix and the inertia residual — dynamic schemes only
            M_e = None
            if isDynamic:
                # M_e = thickness · ∫ρ N·N is constant across the solve, so it is cached (see __Mass_e)
                # instead of rebuilt every Newton iteration. An array ρ is unhashable, so it cannot key the
                # cache and is recomputed directly.
                if isinstance(self.rho, np.ndarray):
                    M_e = thickness * Operators.Bilinear.UV(
                        groupElem, self.rho, dof_n=dim
                    )
                else:
                    M_e = self.__Mass_e(groupElem, self.rho, thickness, dim)
                F_e -= np.einsum(
                    "eij,ej->ei",
                    M_e,
                    groupElem.Locates_sol_e(accel, dim),
                    optimize=True,
                )

            out[groupElem] = (K_e, C_e, M_e, F_e)

        # per-element Clenshaw-Curtis point counts this assembly, or None if the quadrature
        # stress did not run (pointwise / gonzalez)
        self.__nPts_e = np.concatenate(list_nPts_e) if list_nPts_e else None

        return out

    # --------------------------------------------------------------------------
    # Iterations
    # --------------------------------------------------------------------------

    def Save_Iter(self, iter=None):

        if iter is None:
            iter = {}

        iter["displacement"] = self.displacement
        if self.algo in AlgoType.Get_Hyperbolic_Types():
            iter["speed"] = self._Get_v_n(self.problemType)
            iter["accel"] = self._Get_a_n(self.problemType)
        if self.__nPts_e is not None:  # per-element point counts, quadrature only
            iter["nPts_e"] = self.__nPts_e

        return super().Save_Iter(iter)

    def Set_Iter(self, iter=-1, resetAll=False):
        results = super().Set_Iter(iter)

        if results is None:
            return

        u = results["displacement"]

        if (
            self.algo in AlgoType.Get_Hyperbolic_Types()
            and "speed" in results
            and "accel" in results
        ):
            v = results["speed"]
            a = results["accel"]
        else:
            v = np.zeros_like(u)
            a = np.zeros_like(u)

        self._Set_solutions(self.problemType, u, v, a)

        return results

    # --------------------------------------------------------------------------
    # Results
    # --------------------------------------------------------------------------

    def __indexResult(self, result: str) -> int:
        if len(result) <= 2:
            "Case were ui, vi or ai"
            if "x" in result:
                return 0
            elif "y" in result:
                return 1
            elif "z" in result:
                return 2
            else:
                raise ValueError("result error")
        else:
            raise ValueError("result error")

    def Results_Available(self) -> list[str]:
        results = []
        dim = self.dim

        results.extend(["displacement", "displacement_norm", "displacement_matrix"])
        # only under a time scheme: a static solve has no velocity to report, and
        # `Results_nodeFields_elementFields` gates them the same way
        isDynamic = self.algo in AlgoType.Get_Hyperbolic_Types()
        if isDynamic:
            results.extend(["speed", "speed_norm"])
            results.extend(["accel", "accel_norm"])

        if dim == 2:
            results.extend(["ux", "uy"])
            if isDynamic:
                results.extend(["vx", "vy"])
                results.extend(["ax", "ay"])
            results.extend(["Sxx", "Syy", "Sxy"])
            results.extend(["Exx", "Eyy", "Exy"])

        elif dim == 3:
            results.extend(["ux", "uy", "uz"])
            if isDynamic:
                results.extend(["vx", "vy", "vz"])
                results.extend(["ax", "ay", "az"])
            results.extend(["Sxx", "Syy", "Szz", "Syz", "Sxz", "Sxy"])
            results.extend(["Exx", "Eyy", "Ezz", "Eyz", "Exz", "Exy"])

        results.extend(["Svm", "Piola-Kirchhoff", "Evm", "Green-Lagrange"])

        results.extend(["W", "W_e"])

        return results

    def Result(
        self, result: str, nodeValues: bool = True, iter: Optional[int] = None
    ) -> Union[_types.FloatArray, float]:
        if iter is not None:
            self.Set_Iter(iter)

        if not self._Results_Check_Available(result):
            return None  # type: ignore [return-value]

        # begin cases ----------------------------------------------------

        Nn = self.mesh.Nn

        values = None

        if result in ["ux", "uy", "uz"]:
            values_n = self.displacement.reshape(Nn, -1)
            values = values_n[:, self.__indexResult(result)]

        elif result == "displacement":
            values = self.displacement

        elif result == "displacement_norm":
            val_n = self.displacement.reshape(Nn, -1)
            values = np.linalg.norm(val_n, axis=1)

        elif result == "displacement_matrix":
            values = self.Results_displacement_matrix()

        elif result in ["vx", "vy", "vz"]:
            values_n = self.speed.reshape(Nn, -1)
            values = values_n[:, self.__indexResult(result)]

        elif result == "speed":
            values = self.speed

        elif result == "speed_norm":
            val_n = self.speed.reshape(Nn, -1)
            values = np.linalg.norm(val_n, axis=1)

        elif result in ["ax", "ay", "az"]:
            values_n = self.accel.reshape(Nn, -1)
            values = values_n[:, self.__indexResult(result)]

        elif result == "accel":
            values = self.accel

        elif result == "accel_norm":
            val_n = self.accel.reshape(Nn, -1)
            values = np.linalg.norm(val_n, axis=1)

        elif result in ["W"]:
            return self._Calc_W()

        elif result == "W_e":
            values = self._Calc_W(False)

        elif result in ["Green-Lagrange", "Piola-Kirchhoff"] or (
            ("S" in result or "E" in result) and ("_norm" not in result)
        ):
            # Green-Lagrange (E) and second Piola-Kirchhoff (S), group by group.
            # "Green-Lagrange" and "Piola-Kirchhoff" carry no upper-case E or S, so they have to
            # be matched by name: they are what Results_nodeFields_elementFields hands to Paraview.

            isStress = result == "Piola-Kirchhoff" or "S" in result
            res = (
                result
                if result in ["Green-Lagrange", "Piola-Kirchhoff"]
                else result[-2:]
            )

            def field_e_pg(groupElem):
                return (
                    self._Calc_SecondPiolaKirchhoff(groupElem=groupElem)
                    if isStress
                    else self._Calc_GreenLagrange(groupElem=groupElem)
                )

            values = Result_strain_or_stress_field_e(
                field_e_pg=field_e_pg,
                list_groupElem=self.mesh.Get_list_groupElem(),
                result=res,
                coef=self.material.coef,
            )

        else:
            Terminal.MyPrintError(f"The result '{result}' is not implemented yet.")
            return None  # type: ignore [return-value]

        # end cases ----------------------------------------------------

        return self.Results_Reshape_values(values, nodeValues)

    def _Calc_W(self, returnScalar=True, matrixType=MatrixType.rigi):
        r"""Computes the hyperelastic strain energy.

        .. math:: W = \int_{\Omega_0} W(\eb(\ub)) \, \dO

        Parameters
        ----------
        returnScalar : bool, optional
            If True returns the total energy as a float, otherwise the per-element energy (Ne,), by default True.
        matrixType : MatrixType, optional
            integration scheme, by default MatrixType.rigi.
        """
        thickness = self.material.thickness if self.dim == 2 else 1

        # strain energy density W integrated group by group (each main-dimension
        # group may have its own element type / number of Gauss points)
        list_W = []
        for groupElem in self.mesh.Get_list_groupElem(self.dim):
            state = HyperElasticState(groupElem, self.displacement, matrixType)
            wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(matrixType)
            W_e_pg = wJ_e_pg * self.material.Compute_W(state)
            list_W.append(thickness * W_e_pg.integrate())

        W_e = np.concatenate(list_W)

        return float(W_e.sum()) if returnScalar else W_e

    def _Calc_GreenLagrange(self, groupElem=None, matrixType=MatrixType.rigi):
        if groupElem is None:
            groupElem = self.mesh.groupElem
        hyperElasticState = HyperElasticState(groupElem, self.displacement, matrixType)
        return Project_Kelvin(hyperElasticState.Compute_GreenLagrange(), 2)

    def _Calc_SecondPiolaKirchhoff(self, groupElem=None, matrixType=MatrixType.rigi):
        if groupElem is None:
            groupElem = self.mesh.groupElem
        hyperElasticState = HyperElasticState(groupElem, self.displacement, matrixType)
        # total PK2 = elastic ∂W/∂e + the active fiber stress (reported as one field,
        # even though the two are assembled by separate operators)
        S_e_pg = self.material.Compute_dWde(hyperElasticState)
        if np.any(self.material.active_stress != 0.0):
            S_e_pg = S_e_pg + self.material.Compute_active_stress(hyperElasticState)
        return S_e_pg

    def Results_Iter_Summary(
        self,
    ) -> tuple[list[int], list[tuple[str, _types.FloatArray]]]:
        list_label_values = []

        iterations = list(range(self.Niter))
        results = [self.Get_results(i) for i in iterations]

        iter["newtonIter"] = self.__newtonIter
        iter["timeIter"] = self.__timeIter

        newtonIter, timeIter, list_norm_r = zip(
            *(
                (
                    result["convIter"],
                    result["timeIter"],
                )
                for result in results
            )
        )

        list_label_values = [
            ("newtonIter", np.array(newtonIter)),
            ("timeIter", np.array(timeIter)),
        ]

        return iterations, list_label_values

    def Results_dict_Energy(self):
        return super().Results_dict_Energy()

    def Results_displacement_matrix(self) -> _types.FloatArray:
        Nn = self.mesh.Nn
        coord = self.displacement.reshape((Nn, -1))
        dim = coord.shape[1]

        displacement_matrix = np.zeros((Nn, 3))
        displacement_matrix[:, :dim] = coord

        return displacement_matrix

    def Results_nodeFields_elementFields(self, details=False):
        nodesField = ["displacement"]
        if details:
            elementsField = ["Green-Lagrange", "Piola-Kirchhoff"]
        else:
            elementsField = ["Piola-Kirchhoff"]
        if self.algo in AlgoType.Get_Hyperbolic_Types():
            nodesField.extend(["speed", "accel"])
        return nodesField, elementsField
