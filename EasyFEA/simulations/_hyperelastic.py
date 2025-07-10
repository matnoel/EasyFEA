# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information

from scipy import sparse
import numpy as np
from typing import Union, Optional, TYPE_CHECKING
import pandas as pd

# utilities
from ..utilities import Tic, Display, _types
from ..simulations.Solvers import Solve_simu

# fem
if TYPE_CHECKING:
    from ..fem import Mesh
from ..fem import MatrixType, FeArray
from ..fem._linalg import Det

# materials
from ..models import (
    ModelType,
    Project_vector_to_matrix,
    Result_in_Strain_or_Stress_field,
    Project_Kelvin,
)

if TYPE_CHECKING:
    from ..models._hyperelastic_laws import _HyperElas
from ..models._hyperelastic import HyperElastic

# simu
from ._simu import _Simu, AlgoType


class HyperElasticSimu(_Simu):
    def __init__(
        self,
        mesh: "Mesh",
        model: "_HyperElas",
        verbosity=False,
        useNumba=True,
        useIterativeSolvers=True,
    ):
        """Creates a simulation.

        Parameters
        ----------
        mesh : Mesh
            The mesh used.
        model : _HyperElas
            The hyperelatic model used.
        verbosity : bool, optional
            If True, the simulation can write in the terminal. Defaults to False.
        useNumba : bool, optional
            If True and numba is installed numba can be used. Defaults to True.
        useIterativeSolvers : bool, optional
            If True, iterative solvers can be used. Defaults to True.
        """

        super().__init__(mesh, model, verbosity, useNumba, useIterativeSolvers)

        assert model.dim == 3, "For the moment, the simulation is only available in 3D."

        # init
        self.Solver_Set_Elliptic_Algorithm()

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
    def material(self) -> "_HyperElas":
        """hyperelastic material"""
        return self.model  # type: ignore [return-value]

    @property
    def displacement(self) -> _types.FloatArray:
        """Displacement vector field.\n
        [uxi, uyi, uzi, ...]"""
        return self._Get_u_n(self.problemType)

    def _Bc_Integration_scale(self, groupElem, elements, values_e_p, matrixType):
        # TODO to validate
        # return values_e_p

        values_e_p = FeArray.asfearray(values_e_p)
        u = self.displacement

        if groupElem.dim == 3:
            # dont scale for 3D elements
            coef_e_pg = 1
        else:
            # 1D or 2D elements

            # compute F and J
            grad_e_pg = groupElem.Get_Gradient_e_pg(u, matrixType)[elements]
            F_e_pg = np.eye(3) + grad_e_pg
            J_e_pg = Det(F_e_pg)
            coef_e_pg = J_e_pg

            # displacementMatrix = self.Results_displacement_matrix()
            # normal_e = groupElem._Get_sysCoord_e(displacementMatrix)[elements,:,groupElem.dim]
            # normal_e = FeArray.asfearray(normal_e[:,np.newaxis])
            # coef_e_pg = J_e_pg * Norm(Inv(F_e_pg).T @ normal_e, axis=-1)

        scaled_e_pg = coef_e_pg * values_e_p

        return np.asarray(scaled_e_pg)

    # --------------------------------------------------------------------------
    # Solve
    # --------------------------------------------------------------------------

    def Get_K_C_M_F(self, problemType=None):
        if self.needUpdate:
            self.Assembly()
            self.Need_Update(False)

        size = self.__K.shape[0]
        initcsr = sparse.csr_matrix((size, size))

        return self.__K.copy(), initcsr, initcsr, self.__F.copy()

    def Get_x0(self, problemType=None):
        return self.displacement

    def __Solve_hyperelastic(self):
        # compute delta_u
        delta_u = Solve_simu(self, self.problemType)
        # The new delta_u indicates that u will be updated,
        # which is why we must update the matrices.
        self.Need_Update()

        return delta_u

    def _Solver_problemType_is_incremental(self, problemType):
        return True

    def Solve(self, tolConv=1.0e-5, maxIter=20) -> _types.FloatArray:
        """Solves the hyperelastic problem using the newton raphson algorithm.

        Parameters
        ----------
        tolConv : float, optional
            threshold used to check convergence, by default 1e-5
        maxIter : int, optional
            Maximum iterations for convergence, by default 20

        Returns
        -------
        _types.FloatArray
            u_np1: displacement vector field
        """

        u, Niter, timeIter, list_res = self._Solver_Solve_NewtonRaphson(
            self.__Solve_hyperelastic, tolConv, maxIter
        )

        # save iter parameters
        self.__Niter = Niter
        self.__timeIter = timeIter
        self.__list_res = list_res

        return u

    def Assembly(self):
        # Data
        mesh = self.mesh
        Ndof = mesh.Nn * self.dim

        # Additional dimension linked to the use of lagrange coefficients
        Ndof += self._Bc_Lagrange_dim(self.problemType)

        K_e, F_e = self.__Construct_Local_Matrix()

        tic = Tic()

        linesVector_e = mesh.rowsVector_e.ravel()
        columnsVector_e = mesh.columnsVector_e.ravel()

        # Assembly
        self.__K = sparse.csr_matrix(
            (K_e.ravel(), (linesVector_e, columnsVector_e)), shape=(Ndof, Ndof)
        )
        """Kglob matrix for the displacement problem (Ndof, Ndof)"""

        rows = mesh.assembly_e.ravel()
        self.__F = sparse.csr_matrix(
            (F_e.ravel(), (rows, np.zeros_like(rows))), shape=(Ndof, 1)
        )
        """Fglob vector for the displacement problem (Ndof, 1)"""

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.spy(self.__K)
        # plt.show()

        tic.Tac("Matrix", "Assembly Ku and Fu", self._verbosity)

        return self.__K, self.__F

    def __Construct_Local_Matrix(
        self,
    ) -> tuple[FeArray.FeArrayALike, FeArray.FeArrayALike]:
        # data
        mat = self.material
        mesh = self.mesh
        Ne = mesh.Ne
        nPe = mesh.nPe
        dim = self.dim

        # get mesh data
        matrixType = MatrixType.rigi
        wJ_e_pg = mesh.Get_weightedJacobian_e_pg(matrixType)
        dN_e_pg = mesh.Get_dN_e_pg(matrixType)
        nPg = wJ_e_pg.shape[1]

        # get hyperelastic data
        displacement = self.displacement

        # check if there is any invalid element
        J_e_pg = HyperElastic.Compute_J(mesh, displacement, matrixType)
        assert J_e_pg.min() > 0, "Warning: det(F) < 0 - reduce load steps"

        # get hyper elastic matrices
        De_e_pg = HyperElastic.Compute_De(mesh, displacement, matrixType)
        dWde_e_pg = mat.Compute_dWde(mesh, displacement, matrixType)
        d2Wde_e_pg = mat.Compute_d2Wde(mesh, displacement, matrixType)

        # TODO Add HyperElastic.Compute_B_e_pg() and HyperElastic.Compute_Sig_e_pg()
        # init matrices
        grad_e_pg = FeArray.zeros(Ne, nPg, 9, dim * nPe)
        Sig_e_pg = FeArray.zeros(Ne, nPg, 9, 9)
        sig_e_pg = Project_vector_to_matrix(dWde_e_pg)

        rows = np.arange(9).reshape(3, -1)
        cols = np.arange(dim * nPe).reshape(3, -1)
        for i in range(dim):
            grad_e_pg._assemble(rows[i], cols[i], value=dN_e_pg)  # type: ignore [attr-defined]
            Sig_e_pg._assemble(rows[i], rows[i], value=sig_e_pg)  # type: ignore [attr-defined]

        B_e_pg = De_e_pg @ grad_e_pg

        # stiffness
        K1_e = (wJ_e_pg * B_e_pg.T @ d2Wde_e_pg @ B_e_pg).sum(1)
        K2_e = (wJ_e_pg * grad_e_pg.T @ Sig_e_pg @ grad_e_pg).sum(1)
        K_e = K1_e + K2_e

        # source
        F_e = -(wJ_e_pg * dWde_e_pg.T @ B_e_pg).sum(1)

        # reorder xi,...,xn,yi,...yn,zi,...,zn to xi,yi,zi,...,xn,yx,zn
        reorder = np.arange(0, nPe * dim).reshape(-1, nPe).T.ravel()
        F_e = F_e[:, reorder]
        K_e = K_e[:, reorder][:, :, reorder]

        return K_e, F_e

    # --------------------------------------------------------------------------
    # Iterations
    # --------------------------------------------------------------------------

    def Save_Iter(self):
        iter = super().Save_Iter()

        # convergence informations
        iter["Niter"] = self.__Niter
        iter["timeIter"] = self.__timeIter
        iter["list_res"] = self.__list_res

        iter["displacement"] = self.displacement
        if self.algo in AlgoType.Get_Hyperbolic_Types():
            iter["speed"] = self.speed
            iter["accel"] = self.accel

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
                S_e_pg = self._Calc_SecondPiolaKirchhoff()
                val_e = S_e_pg.mean(1)
            elif "E" in result:
                E_e_pg = self._Calc_GreenLagrange()
                val_e = E_e_pg.mean(1)
            else:
                raise Exception("Wrong option")

            res = (
                result
                if result in ["Green-Lagrange", "Piola-Kirchhoff"]
                else result[-2:]
            )

            values = Result_in_Strain_or_Stress_field(val_e, res, self.material.coef)

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
        return Project_Kelvin(
            HyperElastic.Compute_GreenLagrange(self.mesh, self.displacement), 2
        )

    def _Calc_SecondPiolaKirchhoff(self, matrixType=MatrixType.rigi):
        return self.material.Compute_dWde(self.mesh, self.displacement, matrixType)

    def Results_Iter_Summary(
        self,
    ) -> tuple[list[int], list[tuple[str, _types.FloatArray]]]:
        list_label_values = []

        resultats = self.results
        df = pd.DataFrame(resultats)
        iterations = np.arange(df.shape[0]).tolist()

        damageMaxIter = np.array([np.max(damage) for damage in df["damage"].values])
        list_label_values.append((r"$\phi$", damageMaxIter))

        convIter = df["convIter"].values
        list_label_values.append(("convIter", convIter))

        nombreIter = df["Niter"].values
        list_label_values.append(("Niter", nombreIter))

        tempsIter = df["timeIter"].values
        list_label_values.append(("time", tempsIter))

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
