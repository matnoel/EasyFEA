# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from typing import Union
import numpy as np
from scipy import sparse

# utilities
from ..utilities import Tic
# fem
from ..fem import Mesh, MatrixType
# materials
from .. import Materials
from ..materials import ModelType, Reshape_variable
# simu
from ._simu import _Simu
from .Solvers import AlgoType

class ThermalSimu(_Simu):

    def __init__(self, mesh: Mesh, model: Materials.Thermal, verbosity=False, useNumba=True, useIterativeSolvers=True):
        """Creates a thermal simulation.

        Parameters
        ----------
        mesh : Mesh
            The mesh used.
        model : IModel
            The model used.
        verbosity : bool, optional
            If True, the simulation can write in the terminal. Defaults to False.
        useNumba : bool, optional
            If True, numba can be used. Defaults to True.
        useIterativeSolvers : bool, optional
            If True, iterative solvers can be used. Defaults to True.
        """

        assert isinstance(model, Materials.Thermal), "model must be a thermal model"
        super().__init__(mesh, model, verbosity, useNumba, useIterativeSolvers)

        # init
        self.Solver_Set_Elliptic_Algorithm()
    
    def Get_dofs(self, problemType=None) -> list[str]:
        return ["t"]
    
    def Get_dof_n(self, problemType=None) -> int:
        return 1

    def Results_nodesField_elementsField(self, details=False) -> tuple[list[str], list[str]]:
        nodesField = ["thermal", "thermalDot"]
        elementsField = []
        return nodesField, elementsField
    
    def Get_problemTypes(self) -> list[ModelType]:
        return [ModelType.thermal]

    @property
    def thermalModel(self) -> Materials.Thermal:
        """Thermal simulation model."""
        return self.model

    @property
    def thermal(self) -> np.ndarray:
        """Scalar temperature field.\n
        [ti, ....]"""
        return self._Get_u_n(self.problemType)

    @property
    def thermalDot(self) -> np.ndarray:
        """Time derivative of the scalar temperature field.\n
        [d(ti)/dt, ....]"""
        return self._Get_v_n(self.problemType)

    def Get_x0(self, problemType=None):
        if self.thermal.size != self.mesh.Nn:
            return np.zeros(self.mesh.Nn)
        else:
            return self.thermal

    def Get_K_C_M_F(self, problemType=None) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        if self.needUpdate:
            self.Assembly()
            self.Need_Update(False)
        size = self.__Kt.shape[0]
        initcsr = sparse.csr_matrix((size, size))
        return self.__Kt.copy(), self.__Ct.copy(), initcsr, self.__Ft.copy()

    def __Construct_Thermal_Matrix(self) -> tuple[np.ndarray, np.ndarray]:

        thermalModel = self.thermalModel

        # Data
        k = thermalModel.k
        rho = self.rho
        c = thermalModel.c

        matrixType=MatrixType.rigi

        mesh = self.mesh

        jacobian_e_pg = mesh.Get_jacobian_e_pg(matrixType)
        weight_pg = mesh.Get_weight_pg(matrixType)
        N_e_pg = mesh.Get_N_pg(matrixType)
        D_e_pg = mesh.Get_dN_e_pg(matrixType)
        Ne = mesh.Ne
        nPg = weight_pg.size

        k_e_pg = Reshape_variable(k, Ne, nPg)

        Kt_e = np.einsum('ep,p,epji,ep,epjk->eik', jacobian_e_pg, weight_pg, D_e_pg, k_e_pg, D_e_pg, optimize="optimal")

        rho_e_pg = Reshape_variable(rho, Ne, nPg)
        c_e_pg = Reshape_variable(c, Ne, nPg)

        Ct_e = np.einsum('ep,p,pji,ep,ep,pjk->eik', jacobian_e_pg, weight_pg, N_e_pg, rho_e_pg, c_e_pg, N_e_pg, optimize="optimal")

        if self.dim == 2:
            thickness = thermalModel.thickness
            Kt_e *= thickness
            Ct_e *= thickness

        return Kt_e, Ct_e

    def Assembly(self) -> None:
        """Construct the matrix system for the thermal problem in stationary or transient regime."""

        # Data
        mesh = self.mesh
        Ndof = mesh.Nn
        linesScalar_e = mesh.linesScalar_e.ravel()
        columnsScalar_e = mesh.columnsScalar_e.ravel()

        # Additional dimension linked to the use of lagrange coefficients
        Ndof += self._Bc_Lagrange_dim(self.problemType)
        
        # Calculate elementary matrices
        Kt_e, Ct_e = self.__Construct_Thermal_Matrix()
        
        tic = Tic()

        self.__Kt = sparse.csr_matrix((Kt_e.ravel(), (linesScalar_e, columnsScalar_e)), shape = (Ndof, Ndof))
        """Kglob for thermal problem (Ndof, Ndof)"""
        
        self.__Ft = sparse.csr_matrix((Ndof, 1))
        """Fglob vector for thermal problem (Ndof, 1)."""

        self.__Ct = sparse.csr_matrix((Ct_e.ravel(), (linesScalar_e, columnsScalar_e)), shape = (Ndof, Ndof))
        """Mglob for thermal problem (Ndof, Ndof)"""

        tic.Tac("Matrix","Assembly Kt, Mt and Ft", self._verbosity)

    def Save_Iter(self):

        iter = super().Save_Iter()
        
        iter['thermal'] = self.thermal

        if self.algo == AlgoType.parabolic:
            iter['thermalDot'] = self.thermalDot
            
        self._results.append(iter)

    def Set_Iter(self, iter: int=-1, resetAll=False) -> dict:
        
        results = super().Set_Iter(iter)

        if results is None: return

        problemType = self.problemType

        self._Set_u_n(problemType, results["thermal"])

        if self.algo == AlgoType.parabolic and "thermalDot" in results:
            self._Set_v_n(problemType, results["thermalDot"])
        else:
            self._Set_v_n(problemType, np.zeros_like(self.thermal))

        return results

    def Results_Available(self) -> list[str]:
        options = []
        options.extend(["thermal", "thermalDot", "displacement_matrix"])
        return options
        
    def Result(self, result: str, nodeValues=True, iter=None) -> Union[np.ndarray, float, None]:

        if iter != None:
            self.Set_Iter(iter)
        
        if not self._Results_Check_Available(result): return None

        # begin cases ----------------------------------------------------

        if result == "thermal":
            values = self.thermal

        elif result == "thermalDot":
            values = self.thermalDot

        elif result == "displacement_matrix":
            values = self.Results_displacement_matrix()

        # end cases ----------------------------------------------------
        
        return self.Results_Reshape_values(values, nodeValues)

    def Results_Iter_Summary(self) -> list[tuple[str, np.ndarray]]:
        return super().Results_Iter_Summary()

    def Results_dict_Energy(self) -> list[tuple[str, float]]:
        return super().Results_dict_Energy()
    
    def Results_displacement_matrix(self) -> np.ndarray:
        return super().Results_displacement_matrix()