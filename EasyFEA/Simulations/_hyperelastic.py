# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information

import numpy as np
from typing import Union, Optional, TYPE_CHECKING

# utilities
from ..Utilities import Display, _types

# fem
if TYPE_CHECKING:
    from ..FEM import Mesh
from ..FEM import MatrixType, FeArray

# models
from ..Models import (
    ModelType,
    Reshape_variable,
    Result_in_Strain_or_Stress_field,
    Project_Kelvin,
)

if TYPE_CHECKING:
    from ..Models.HyperElastic._laws import _HyperElastic
from ..Models.HyperElastic._state import HyperElasticState

# simu
from ._simu import _Simu, AlgoType


class HyperElastic(_Simu):
    r"""Hyperelastic simulation.

    Weak form:

    .. math::
        R(\ub; \vb) = \int_{\Omega_0} \boldsymbol{\Sigma}(\ub) : \Drm_\ub \eb(\ub) \cdot \vb \, \dO +
        \int_{\Omega_0} \rho \, \ddot{\ub} \cdot \vb \, \dO  - \int_{\partial\Omega_0^t} \tb\cdot\vb \, \dS - \int_{\Omega_0} \fb\cdot\vb \, \dO \quad \forall \, \vb \in V

    where :math:`\boldsymbol{\Sigma} := J \, \Fb^{-1} \cdot \Sig \cdot \Fb^{-T}`, is the second Piola Kirchhoff stress tensor (PK2), :math:`\eb := \frac{1}{2} \left( \Cb - \boldsymbol{1} \right) = \frac{1}{2} \left( \Fb^T \cdot \Fb - \boldsymbol{1} \right)` is the Green-Lagrange strain tensor and :math:`\Fb := \boldsymbol{1} +  \grad \ub` the deformation gradient.

    This non linear problem is solve using the newton rapshon algorithm:

    .. math::
        A(\ub; \vb, \wb) \, \Delta \ub = - R(\ub; \vb) \quad \forall \, (\vb, \wb) \in \Vc \times \Wc,

    where the tangent :math:`A(\ub; \vb, \wb)` is defined :math:`\forall \, (\vb, \wb) \in \Vc \times \Wc` as:

    .. math::
        A(\ub; \vb, \wb) &=
        \dpartial{R(\ub; \vb)}{\ub} \cdot \wb \\ &=
        \int_{\Omega_0} \Drm_\ub \eb(\ub) \cdot \wb : \dNpartial{2}{W}{\eb}(\ub) : \Drm_\ub \eb(\vb) \, \dO +
        \int_{\Omega_0} \dpartial{W}{\eb}(\ub) : \Drm_\ub^2 \eb(\vb, \wb) \, \dO

    The implemented hyperelastic laws are available :ref:`here <models-hyperelastic>` and where constructed by the :ref:`ComputeHyperelasticLaws` script.
    """

    # TODO: add math

    def __init__(
        self,
        mesh: "Mesh",
        model: "_HyperElastic",
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

        super().__init__(mesh, model, verbosity)

        self._Solver_Set_Newton_Raphson_Algorithm(tolConv=tolConv, maxIter=maxIter)

        # Set solver petsc4py options, even if petsc4py is unavailable.
        self._Solver_Set_PETSc4Py_Options(pcType="lu")

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

    def Construct_local_matrix_system(self, problemType):
        # data
        mesh = self.mesh
        dim = self.dim
        thickness = self.material.thickness if dim == 2 else 1

        # get the current newton raphson displacement (updated via u += delta_u)
        displacement = self._Solver_Get_Newton_Raphson_current_solution()

        if self.algo in AlgoType.Get_Hyperbolic_Types():
            # here update the displacement according to the time scheme
            displacement = self._Solver_Evaluate_u_v_a_for_time_scheme(
                problemType, displacement
            )[0]

        # get the hyperelastic state
        hyperElasticState = HyperElasticState(mesh, displacement, MatrixType.rigi)

        # check if there is any invalid element
        J_e_pg = hyperElasticState.Compute_J()
        assert J_e_pg.min() > 0, "Warning: det(F) < 0 - reduce load steps"

        # ------------------------------
        # Compute tangent and residual
        # ------------------------------
        tangent_e, residual_e = self.material.Compute_Tangent_and_Residual(
            hyperElasticState
        )

        # Here we solve:
        # K(u) Δu = - R(u)
        #         = - (F(u) - b)
        #         = - F(u) + b
        K_e = tangent_e
        F_e = -residual_e

        # ------------------------------
        # Compute Mass
        # ------------------------------
        if self.algo in AlgoType.Get_Hyperbolic_Types():
            matrixType = MatrixType.mass
            N_pg = FeArray.asfearray(mesh.Get_N_vector_pg(matrixType)[np.newaxis])
            wJ_e_pg = mesh.Get_weightedJacobian_e_pg(matrixType)

            rho_e_pg = Reshape_variable(self.rho, *wJ_e_pg.shape[:2])

            M_e = thickness * (rho_e_pg * wJ_e_pg * N_pg.T @ N_pg).sum(axis=1)
        else:
            M_e = None

        return K_e, None, M_e, F_e

    # --------------------------------------------------------------------------
    # Iterations
    # --------------------------------------------------------------------------

    def Save_Iter(self):
        iter = super().Save_Iter()

        iter["displacement"] = self.displacement
        if self.algo in AlgoType.Get_Hyperbolic_Types():
            iter["speed"] = self._Get_v_n(self.problemType)
            iter["accel"] = self._Get_a_n(self.problemType)

        self._results.append(iter)

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
            # Green-Lagrange and second Piola-Kirchhoff for each element and gauss point

            # Element average
            if "S" in result:
                values_e_pg = self._Calc_SecondPiolaKirchhoff()
            elif "E" in result:
                values_e_pg = self._Calc_GreenLagrange()
            else:
                raise Exception("Wrong option")

            res = (
                result
                if result in ["Green-Lagrange", "Piola-Kirchhoff"]
                else result[-2:]
            )

            coef = self.material.coef
            values = Result_in_Strain_or_Stress_field(values_e_pg, res, coef).mean(1)

        if not isinstance(values, np.ndarray):
            Display.MyPrintError("This result option is not implemented yet.")
            return None  # type: ignore [return-value]

        # end cases ----------------------------------------------------

        return self.Results_Reshape_values(values, nodeValues)

    def _Calc_W(self, returnScalar=True, matrixType=MatrixType.rigi):
        wJ_e_pg = self.mesh.Get_weightedJacobian_e_pg(matrixType)
        if self.dim == 2:
            wJ_e_pg = self.material.thickness
        W_e_pg = self.material.Compute_W(self.mesh, self.displacement, matrixType)

        if returnScalar:
            return (wJ_e_pg * W_e_pg).sum()
        else:
            return (wJ_e_pg * W_e_pg).sum(1)

    def _Calc_GreenLagrange(self, matrixType=MatrixType.rigi):
        hyperElasticState = HyperElasticState(self.mesh, self.displacement, matrixType)
        return Project_Kelvin(hyperElasticState.Compute_GreenLagrange(), 2)

    def _Calc_SecondPiolaKirchhoff(self, matrixType=MatrixType.rigi):
        hyperElasticState = HyperElasticState(self.mesh, self.displacement, matrixType)
        return self.material.Compute_dWde(hyperElasticState)

    def Results_Iter_Summary(
        self,
    ) -> tuple[list[int], list[tuple[str, _types.FloatArray]]]:
        list_label_values = []

        results = self.results
        iterations = list(range(len(results)))

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
