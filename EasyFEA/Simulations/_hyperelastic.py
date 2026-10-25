# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information

import numpy as np
from typing import TYPE_CHECKING

# utilities
from ..Utilities import Terminal, _types

# fem
if TYPE_CHECKING:
    from ..FEM import Mesh
from ..FEM import MatrixType, FeArray, Kinematics, Operators

# models
from ..FEM._kelvin_mandel import Tensor_to_Kelvin
from ._utils import Result_strain_or_stress_field_e, _Field_per_groupElem

if TYPE_CHECKING:
    from ..Models.HyperElastic._laws import _HyperElastic

# simu
from ..FEM import _GroupElem
from ._simu import _Simu, AlgoType
from ._terms import Term
from ._problem_type import ProblemType


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

    problemType = ProblemType("hyperelastic")

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

        self.matrixType = MatrixType.rigi

    # --------------------------------------------------------------------------
    # General
    # --------------------------------------------------------------------------

    def Get_unknowns(self, problemType=None) -> list[str]:
        dict_unknowns = {2: ["x", "y"], 3: ["x", "y", "z"]}
        return dict_unknowns[self.dim]

    def Get_dof_n(self, problemType=None) -> int:
        return self.dim

    @property
    def matrixType(self) -> MatrixType:
        """Integration rule the tangent and internal force are built at, ``MatrixType.rigi`` by default. Raise it when a field the material reads — fibers, sheets, an active stress — is sampled at a finer rule, so both are evaluated at the same points."""
        return self.__matrixType

    @matrixType.setter
    def matrixType(self, value: MatrixType) -> None:
        self.__matrixType = value
        self.Need_Update()

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

    def Get_terms(self, problemType=None) -> list[Term]:
        r"""Terms of ``A(u)·Δu = -R(u)`` with ``A = coefK·K + coefC·C + coefM·M``.

        The elastic tangent and internal force come as one ``"KR"`` term; the active fiber stress and the Kelvin–Voigt viscosity add their own; the mass matrix is a plain ``"M"`` whose residual :math:`-\Mrm_e \, \arm_t` the fold supplies. The ``R`` slot throughout: these operators return the **internal** force, which reaches the right-hand side negated.
        """
        if problemType is None:
            problemType = self.problemType

        _, v_t, _ = self.Get_u_v_a(problemType)

        terms = [Term("KR", self.__Stress)]

        if np.any(self.material.active_stress != 0.0):
            terms.append(Term("KR", self.__ActiveStress))

        if self.material.eta != 0 and v_t is not None:
            terms.append(Term("KRC", self.__Viscosity, v_t=v_t))

        if self.algo in AlgoType.Get_Hyperbolic_Types():
            terms.append(
                Term(
                    "M",
                    Operators.Bilinear.UV,
                    coef=self.rho,
                    dof_n=self.dim,
                    constant=True,
                )
            )

        return terms

    def __Kinematics(self, groupElem: _GroupElem, u: _types.FloatArray) -> Kinematics:
        """Hyperelastic kinematics of `groupElem` at `u`, guarded against an inverted element."""
        kinematics = Kinematics(groupElem, u, self.matrixType)
        assert kinematics.Compute_J().min() > 0, "det(F) < 0 - reduce load steps"
        return kinematics

    def __Stress(
        self, groupElem: _GroupElem, u: _types.FloatArray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Elastic tangent and internal force at the time scheme's evaluation displacement `u`."""
        return Operators.NonLinear.SecondPiolaKirchhoffStressTensor(
            self.material, self.__Kinematics(groupElem, u)
        )

    def __ActiveStress(
        self, groupElem: _GroupElem, u: _types.FloatArray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Active fiber stress τ·(T̂⊗T̂): internal force + geometric tangent, no material tangent."""
        return Operators.NonLinear.ActiveStressTensor(
            self.material, self.__Kinematics(groupElem, u)
        )

    def __Viscosity(
        self, groupElem: _GroupElem, u: _types.FloatArray, v_t: _types.FloatArray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Kelvin–Voigt viscosity: configuration tangent, damping matrix, viscous residual."""
        return Operators.NonLinear.KelvinVoigtDamping(
            self.material, self.__Kinematics(groupElem, u), v_t
        )

    # --------------------------------------------------------------------------
    # Iterations
    # --------------------------------------------------------------------------

    def _Get_dof_names(self) -> dict[str, ProblemType]:
        return dict.fromkeys(("displacement", "speed", "accel"), self.problemType)

    def Save_Iter(self, iter=None):

        if iter is None:
            iter = {}

        iter["displacement"] = self.displacement
        if self.algo in AlgoType.Get_Hyperbolic_Types():
            iter["speed"] = self._Get_v_n(self.problemType)
            iter["accel"] = self._Get_a_n(self.problemType)

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
        self, result: str, nodeValues: bool = True, iter: int | None = None
    ) -> _types.FloatArray | float:
        if iter is not None:
            self.Set_Iter(iter)

        if not self._Results_Check_Available(result):
            return None  # type: ignore [return-value]

        # begin cases ----------------------------------------------------

        # node values (Nn, dim) of the displacement, speed and acceleration, through the space
        space = self._Get_FunctionSpace()
        u_n = space.Get_nodeValues(self.displacement, self.dim)
        v_n = space.Get_nodeValues(self.speed, self.dim)
        a_n = space.Get_nodeValues(self.accel, self.dim)

        values = None

        if result in ["ux", "uy", "uz"]:
            values = u_n[:, self.__indexResult(result)]

        elif result == "displacement":
            values = u_n.ravel()

        elif result == "displacement_norm":
            values = np.linalg.norm(u_n, axis=1)

        elif result == "displacement_matrix":
            values = self.Results_displacement_matrix()

        elif result in ["vx", "vy", "vz"]:
            values = v_n[:, self.__indexResult(result)]

        elif result == "speed":
            values = v_n.ravel()

        elif result == "speed_norm":
            values = np.linalg.norm(v_n, axis=1)

        elif result in ["ax", "ay", "az"]:
            values = a_n[:, self.__indexResult(result)]

        elif result == "accel":
            values = a_n.ravel()

        elif result == "accel_norm":
            values = np.linalg.norm(a_n, axis=1)

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

            field = (
                self._Calc_SecondPiolaKirchhoff()
                if isStress
                else self._Calc_GreenLagrange()
            )
            values = Result_strain_or_stress_field_e(field, res)

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
            kinematics = Kinematics(groupElem, self.displacement, matrixType)
            wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(matrixType)
            W_e_pg = wJ_e_pg * self.material.Compute_W(kinematics)
            list_W.append(thickness * W_e_pg.integrate())

        W_e = np.concatenate(list_W)

        return float(W_e.sum()) if returnScalar else W_e

    def _Calc_GreenLagrange(
        self, matrixType: MatrixType | None = None
    ) -> FeArray.FeArrayALike | dict:
        """Green-Lagrange strain in Kelvin-Mandel form, at ``self.matrixType`` by default: an ``FeArray`` on one group, ``{groupElem: FeArray}`` on several."""
        if matrixType is None:
            matrixType = self.matrixType
        return _Field_per_groupElem(
            lambda groupElem: Tensor_to_Kelvin(
                Kinematics(
                    groupElem, self.displacement, matrixType
                ).Compute_GreenLagrange(),
                2,
            ),
            self.mesh.Get_list_groupElem(),
        )

    def _Calc_SecondPiolaKirchhoff(
        self, matrixType=None
    ) -> FeArray.FeArrayALike | dict:
        """Total second Piola-Kirchhoff stress ``∂W/∂e`` + active stress, at ``self.matrixType`` by default: an ``FeArray`` on one group, ``{groupElem: FeArray}`` on several."""
        if matrixType is None:
            matrixType = self.matrixType

        def S_e_pg(groupElem: _GroupElem) -> FeArray.FeArrayALike:
            kinematics = Kinematics(groupElem, self.displacement, matrixType)
            # reported as one field, even though the two are assembled by separate operators
            S_e_pg = self.material.Compute_dWde(kinematics)
            if np.any(self.material.active_stress != 0.0):
                S_e_pg = S_e_pg + self.material.Compute_active_stress(kinematics)
            return S_e_pg

        return _Field_per_groupElem(S_e_pg, self.mesh.Get_list_groupElem())

    def Results_Iter_Summary(
        self,
    ) -> tuple[list[int], list[tuple[str, _types.FloatArray]]]:
        return super().Results_Iter_Summary()

    def Results_dict_Energy(self):
        return super().Results_dict_Energy()

    def Results_displacement_matrix(self) -> _types.FloatArray:
        Nn = self.mesh.Nn
        coord = self._Get_FunctionSpace().Get_nodeValues(self.displacement, self.dim)
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
