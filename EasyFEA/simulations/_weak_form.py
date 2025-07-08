# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import Union, Optional, TYPE_CHECKING
from scipy import sparse

# utilities
from ..utilities import Tic, _types
from ..simulations.Solvers import Solve_simu

# fem
if TYPE_CHECKING:
    from ..fem import Mesh

# materials
from .. import Models
from ..models import ModelType

# simu
from ._simu import _Simu
from .Solvers import AlgoType


class WeakFormSimu(_Simu):
    def __init__(
        self,
        mesh: "Mesh",
        model: Models.WeakFormManager,
        verbosity=False,
        useNumba=True,
        useIterativeSolvers=True,
    ):
        """Creates a thermal simulation.

        Parameters
        ----------
        mesh : Mesh
            The mesh used.
        model : WeakFormManager
            The model used.
        verbosity : bool, optional
            If True, the simulation can write in the terminal. Defaults to False.
        useNumba : bool, optional
            If True, numba can be used. Defaults to True.
        useIterativeSolvers : bool, optional
            If True, iterative solvers can be used. Defaults to True.
        """

        assert isinstance(
            model, Models.WeakFormManager
        ), "model must be a weakf form manager"
        super().__init__(mesh, model, verbosity, useNumba, useIterativeSolvers)

        # init
        self.Solver_Set_Elliptic_Algorithm()

    def _Check_dim_mesh_material(self) -> None:
        pass

    def Get_unknowns(self, problemType=None) -> list[str]:
        return [str(d) for d in range(self.Get_dof_n(problemType))]

    def Get_dof_n(self, problemType=None) -> int:
        return self.weakFormManager.field.dof_n

    def Results_nodeFields_elementFields(
        self, details=False
    ) -> tuple[list[str], list[str]]:
        nodesField = ["u", "v", "a"]
        elementsField: list[str] = []
        return nodesField, elementsField

    def Get_problemTypes(self) -> list[ModelType]:
        return [ModelType.weakForm]

    @property
    def weakFormManager(self) -> Models.WeakFormManager:
        """Weak form manager."""
        return self.model  # type: ignore [return-value]

    @property
    def u(self) -> _types.FloatArray:
        """node field u."""
        return self._Get_u_n(self.problemType)

    @property
    def v(self) -> _types.FloatArray:
        """node field v = dudt"""
        return self._Get_v_n(self.problemType)

    @property
    def a(self) -> _types.FloatArray:
        """node field a = d2udt2"""
        return self._Get_a_n(self.problemType)

    def Get_x0(self, problemType=None):
        return self.u

    def Get_K_C_M_F(
        self, problemType=None
    ) -> tuple[
        sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix
    ]:
        if self.needUpdate:
            self.Assembly()
            self.Need_Update(False)

        return self.__K.copy(), self.__C.copy(), self.__M.copy(), self.__F.copy()

    def Assembly(self) -> None:
        """Construct the matrix system for the thermal problem in stationary or transient regime."""

        # Data
        weakFormManager = self.weakFormManager
        mesh = self.mesh
        field = weakFormManager.field
        Ndof = mesh.Nn * field.dof_n

        # Additional dimension linked to the use of lagrange coefficients
        NdofLagr = self._Bc_Lagrange_dim(self.problemType)

        initCsrMatrix = sparse.csr_matrix(
            (Ndof + NdofLagr, Ndof + NdofLagr), dtype=float
        )

        tic = Tic()

        computeK = weakFormManager.computeK

        if computeK is None:
            K = initCsrMatrix
        else:
            K = computeK._assemble(field)
            if NdofLagr > 0:
                K = K.tolil()
                K.resize((Ndof + NdofLagr, Ndof + NdofLagr))
                K = K.tocsr()

        tic.Tac("Matrix", "Assemble K", self._verbosity)

        computeC = weakFormManager.computeC

        if computeC is None:
            C = initCsrMatrix
        else:
            C = computeC._assemble(field)
            if NdofLagr > 0:
                C = C.tolil()
                C.resize((Ndof + NdofLagr, Ndof + NdofLagr))
                C = C.tocsr()

        tic.Tac("Matrix", "Assemble C", self._verbosity)

        computeM = weakFormManager.computeM

        if computeM is None:
            M = initCsrMatrix
        else:
            M = computeC._assemble(field)
            if NdofLagr > 0:
                M = M.tolil()
                M.resize((Ndof + NdofLagr, Ndof + NdofLagr))
                M = M.tocsr()

        tic.Tac("Matrix", "Assemble M", self._verbosity)

        computeF = weakFormManager.computeF

        if computeF is None:
            F = sparse.csr_matrix((Ndof + NdofLagr, 1), dtype=float)
        else:
            F = computeF._assemble(field)
            if NdofLagr > 0:
                F = F.tolil()
                F.resize((Ndof + NdofLagr, Ndof + NdofLagr))
                F = F.tocsr()

        tic.Tac("Matrix", "Assemble F", self._verbosity)

        self.__K = K
        self.__C = C
        self.__M = M
        self.__F = F

    def Solve(self):

        # solve u
        self.__isNonLinear = False
        u = Solve_simu(self, self.problemType)

        # update and set solutions
        u, v, a = self._Solver_Update_solutions(self.problemType, u)
        self._Set_solutions(self.problemType, u, v, a)

    def __Solve_delta_u(self):
        # compute delta_u
        delta_u = Solve_simu(self, self.problemType)
        # The new delta_u indicates that u will be updated,
        # which is why we must update the matrices.
        self.Need_Update()

        return delta_u

    def Solve_NonLinear(self, tolConv=1.0e-5, maxIter=20) -> _types.FloatArray:
        """Solves the problem using the newton raphson algorithm.

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

        # solve u
        self.__isNonLinear = True
        u, Niter, timeIter, list_res = self._Solver_Solve_NewtonRaphson(
            self.__Solve_delta_u, tolConv, maxIter
        )

        # save iter parameters
        self.__Niter = Niter
        self.__timeIter = timeIter
        self.__list_res = list_res

        return u

    def _Solver_problemType_is_non_linear(self, problemType):
        return self.__isNonLinear

    def Save_Iter(self):
        iter = super().Save_Iter()

        if self.__isNonLinear:
            # convergence informations
            iter["Niter"] = self.__Niter
            iter["timeIter"] = self.__timeIter
            iter["list_res"] = self.__list_res

        if self.algo == AlgoType.elliptic:
            iter["u"] = self.u

        elif self.algo == AlgoType.parabolic:
            iter["u"] = self.u
            iter["v"] = self.v

        elif self.algo in AlgoType.Get_Hyperbolic_Types():
            iter["u"] = self.u
            iter["v"] = self.v
            iter["a"] = self.a

        else:
            raise TypeError("Unknown algo type.")

        self._results.append(iter)

    def Set_Iter(self, iter: int = -1, resetAll=False) -> dict:
        results = super().Set_Iter(iter)

        if results is None:
            return

        if self.algo == AlgoType.elliptic:
            u = results["u"]
            self._Set_solutions(self.problemType, u)

        elif self.algo == AlgoType.parabolic:
            u = results["u"]
            v = results["v"]
            self._Set_solutions(self.problemType, u, v)

        elif self.algo in AlgoType.Get_Hyperbolic_Types():
            u = results["u"]
            v = results["v"]
            a = results["a"]
            self._Set_solutions(self.problemType, u, v, a)

        else:
            raise TypeError("Unknown algo type.")

        return results

    def Results_Available(self) -> list[str]:
        options = []
        options.extend(["u", "v", "a", "displacement_matrix"])
        return options

    def Result(
        self, result: str, nodeValues: bool = True, iter: Optional[int] = None
    ) -> Union[_types.FloatArray, float]:
        if iter is not None:
            self.Set_Iter(iter)

        if not self._Results_Check_Available(result):
            return None  # type: ignore [return-value]

        # begin cases ----------------------------------------------------

        if result == "u":
            values = self.u

        elif result == "v":
            values = self.v

        elif result == "a":
            values = self.a

        elif result == "displacement_matrix":
            values = self.Results_displacement_matrix()

        # end cases ----------------------------------------------------

        return self.Results_Reshape_values(values, nodeValues)

    def Results_Iter_Summary(
        self,
    ) -> tuple[list[int], list[tuple[str, _types.FloatArray]]]:
        return super().Results_Iter_Summary()

    def Results_dict_Energy(self) -> dict[str, float]:
        return super().Results_dict_Energy()

    def Results_displacement_matrix(self) -> _types.FloatArray:
        return super().Results_displacement_matrix()
