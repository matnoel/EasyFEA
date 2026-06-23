# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information

import numpy as np
from typing import Union, Optional, TYPE_CHECKING

# utilities
from ..Utilities import Display, _types

# fem
if TYPE_CHECKING:
    from ..FEM import Mesh
from ..FEM import MatrixType, Operators

# models
from ..Models import ModelType, Project_Kelvin, Result_strain_or_stress_field_e

if TYPE_CHECKING:
    from ..Models.HyperElastic._laws import _HyperElastic
from ..Models.HyperElastic._state import HyperElasticState

# simu
from ._simu import _Simu, AlgoType


class HyperElastic(_Simu):
    r"""Finite-strain (large-deformation) hyperelastic simulation, total-Lagrangian framework.

    Solves the nonlinear static or dynamic equilibrium of a hyperelastic body with a Newton-Raphson scheme, with optional Kelvin-Voigt viscosity and a fiber active stress (e.g. for cardiac mechanics). Material behaviour is supplied by a hyperelastic model (Saint-Venant-Kirchhoff, Neo-Hookean, Mooney-Rivlin, Holzapfel-Ogden, …).

    Weak form:

    .. math::
        R(\ub; \vb) = \int_{\Omega_0} \boldsymbol{\Sigma}(\ub) : \Drm_\ub \eb(\ub) \cdot \vb \, \dO +
        \int_{\Omega_0} \rho \, \ddot{\ub} \cdot \vb \, \dO  - \int_{\partial\Omega_0^t} \tb\cdot\vb \, \dS - \int_{\Omega_0} \fb\cdot\vb \, \dO \quad \forall \, \vb \in V

    where :math:`\boldsymbol{\Sigma} := J \, \Fb^{-1} \cdot \Sig \cdot \Fb^{-T}`, is the second Piola Kirchhoff stress tensor (PK2), :math:`\eb := \frac{1}{2} \left( \Cb - \boldsymbol{1} \right) = \frac{1}{2} \left( \Fb^T \cdot \Fb - \boldsymbol{1} \right)` is the Green-Lagrange strain tensor and :math:`\Fb := \boldsymbol{1} +  \grad \ub` the deformation gradient.

    The total PK2 stress combines the elastic response with two optional contributions:

    .. math::
        \boldsymbol{\Sigma}(\ub, \dot{\ub}) = \dpartial{W}{\eb}(\ub) + \tau \, \hat{\Tb} \otimes \hat{\Tb} + \eta \, \dot{\eb}(\ub, \dot{\ub})

    - **Active stress** :math:`\tau \, \hat{\Tb} \otimes \hat{\Tb}`: a contractile stress of magnitude :math:`\tau` (``material.active_stress``) acting along the unit fiber direction :math:`\hat{\Tb}`, registered once with ``material.Set_active_stress_vec``. It is strain-independent and typically used for cardiac mechanics, where only :math:`\tau` is updated between time steps.
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

    def __init__(
        self,
        mesh: "Mesh",
        model: "_HyperElastic",
        folder: str = "",
        tolConv=1e-5,
        maxIter=20,
        verbosity=False,
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
        tolConv : float, optional
            threshold used to check convergence, by default 1e-5
        maxIter : int, optional
            Maximum iterations for convergence, by default 20
        verbosity : bool, optional
            If True, iterative solvers can be used. Defaults to True.

        WARNING
        -------
        2D simulations are conducted under the **plane strain** assumption.
        """

        super().__init__(mesh, model, folder, verbosity)

        self._Solver_Set_Newton_Raphson_Algorithm(tolConv=tolConv, maxIter=maxIter)

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

    # --------------------------------------------------------------------------
    # Solve
    # --------------------------------------------------------------------------

    def Get_x0(self, problemType=None):
        return self.displacement

    def Construct_local_matrix_system(
        self,
        problemType,
        matrixType: MatrixType = MatrixType.rigi,
    ):
        """Returns ``{groupElem: (K_e, C_e, M_e, F_e)}`` for the current Newton iterate.

        Newton solves ``A(u)·Δu = -R(u)``; the simulation assembles ``A = coefK·K + coefC·C + coefM·M`` and ``b -= C @ v_t``. Per group of elements: ``K_e`` is the elastic tangent (plus the viscous configuration tangent ``Kgeo_e``), ``F_e = -R_e`` the residual, ``C_e`` the Kelvin–Voigt damping matrix and ``M_e`` the mass matrix (dynamic schemes only).
        """
        dim = self.dim
        thickness = self.material.thickness if dim == 2 else 1
        isDynamic = self.algo in AlgoType.Get_Hyperbolic_Types()

        # current Newton-Raphson iterate; for dynamic schemes also capture the
        # velocity, for Kelvin–Voigt viscosity.
        displacement = self._Solver_Get_Newton_Raphson_current_solution()
        velocity = None
        if isDynamic:
            displacement, velocity, _ = self._Solver_Evaluate_u_v_a_for_time_scheme(
                problemType, displacement
            )

        out = {}
        for groupElem in self.mesh.Get_list_groupElem():
            state = HyperElasticState(groupElem, displacement, matrixType)

            # invalid-element guard
            assert state.Compute_J().min() > 0, "det(F) < 0 - reduce load steps"

            # elastic tangent + residual; Newton: A(u) Δu = -R(u) = -F(u) + b
            K_e, residual_e = Operators.NonLinear.SecondPiolaKirchhoffStressTensor(
                self.material, state
            )
            F_e = -residual_e

            # Kelvin–Voigt viscosity: C_e is the damping matrix (slot 2, rides
            # coefC, carries the viscous residual b -= C @ v_t); Kgeo_e is the
            # configuration tangent ∂(C·v)/∂u, added to K_e so it rides coefK.
            C_e = None
            if self.material.eta != 0 and velocity is not None:
                C_e, Kgeo_e = Operators.NonLinear.KelvinVoigtDamping(
                    self.material, state, velocity
                )
                K_e = K_e + Kgeo_e

            # mass matrix — only assembled for dynamic schemes
            M_e = None
            if isDynamic:
                M_e = thickness * Operators.Bilinear.UV(groupElem, self.rho, dof_n=dim)

            out[groupElem] = (K_e, C_e, M_e, F_e)

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
        # results.extend(["speed", "speed_norm"])
        # results.extend(["accel", "accel_norm"])

        if dim == 2:
            results.extend(["ux", "uy"])
            # results.extend(["vx", "vy"])
            # results.extend(["ax", "ay"])
            results.extend(["Sxx", "Syy", "Sxy"])
            results.extend(["Exx", "Eyy", "Exy"])

        elif dim == 3:
            results.extend(["ux", "uy", "uz"])
            # results.extend(["vx", "vy", "vz"])
            # results.extend(["ax", "ay", "az"])
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

        # elif result in ["vx", "vy", "vz"]:
        #     values_n = self.speed.reshape(Nn, -1)
        #     values = values_n[:,self.__indexResult(result)]

        # elif result == "speed":
        #     values = self.speed

        # elif result == "speed_norm":
        #     val_n = self.speed.reshape(Nn, -1)
        #     values = np.linalg.norm(val_n, axis=1)

        # elif result in ["ax", "ay", "az"]:
        #     values_n = self.accel.reshape(Nn, -1)
        #     values = values_n[:,self.__indexResult(result)]

        # elif result == "accel":
        #     values = self.accel

        # elif result == "accel_norm":
        #     val_n = self.accel.reshape(Nn, -1)
        #     values = np.linalg.norm(val_n, axis=1)

        elif result in ["W"]:
            return self._Calc_W()

        elif result == "W_e":
            values = self._Calc_W(False)

        elif ("S" in result or "E" in result) and ("_norm" not in result):
            # Green-Lagrange (E) and second Piola-Kirchhoff (S), group by group

            isStress = "S" in result
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
            Display.MyPrintError(f"The result '{result}' is not implemented yet.")
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
        return self.material.Compute_dWde(hyperElasticState)

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
