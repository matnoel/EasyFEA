# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information

from scipy import sparse
import numpy as np

# utilities
from ..utilities import Tic
from ..utilities._linalg import Transpose
# fem
from ..fem import Mesh, MatrixType, FeArray
# materials
from ..materials import ModelType, Reshape_variable, Project_vector_to_matrix
from ..materials._hyperelastic_laws import _HyperElas
from ..materials._hyperelastic import HyperElastic
# simu
from ._simu import _Simu
from .Solvers import AlgoType

class HyperElasticSimu(_Simu):

    def __init__(self, mesh: Mesh, model: _HyperElas, verbosity=True, useNumba=True, useIterativeSolvers=True):
        """Creates a simulation.

        Parameters
        ----------
        mesh : Mesh
            The mesh used.
        model : _HyperElas
            The hyperelatic model used.
        verbosity : bool, optional
            If True, the simulation can write in the terminal. Defaults to True.
        useNumba : bool, optional
            If True and numba is installed numba can be used. Defaults to True.
        useIterativeSolvers : bool, optional
            If True, iterative solvers can be used. Defaults to True.
        """

        super().__init__(mesh, model, verbosity, useNumba, useIterativeSolvers)

        # init
        self.Solver_Set_Elliptic_Algorithm()

    # --------------------------------------------------------------------------
    # General
    # --------------------------------------------------------------------------

    def Get_problemTypes(self):
        return [ModelType.hyperelastic]
    
    def Get_dofs(self, problemType=None) -> list[str]:
        dict_dim_directions = {
            2 : ["x", "y"],
            3 : ["x", "y", "z"]
        }
        return dict_dim_directions[self.dim]
    
    def Get_dof_n(self, problemType=None) -> int:
        return self.dim
    
    @property
    def material(self) -> _HyperElas:
        """hyperelastic material"""
        return self.model
    
    # --------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------- 

    def Get_K_C_M_F(self, problemType=None):

        self.Assembly()

        size = self.mesh.Nn * self.dim
        initcsr = sparse.csr_matrix((size, size))
        initcsrF = sparse.csr_matrix((size, 1))

        return initcsr, initcsr, initcsr, initcsrF
    
    def Get_x0(self, problemType=None):
        return super().Get_x0(problemType)
    
    def Assembly(self):

        u0 = np.zeros(self.mesh.Nn * self.dim)        

        K_e, F_e = self.__Construct_Local_Matrix(u0)

        return ModuleNotFoundError
    
    def __Construct_Local_Matrix(self, u: np.ndarray) -> tuple[FeArray, FeArray]:

        # data
        mat = self.material
        mesh = self.mesh
        Ne = mesh.Ne
        nPe = mesh.nPe
        dim = self.dim

        # get mesh data
        matrixType = MatrixType.rigi
        weightedJacobian_e_pg = mesh.Get_weightedJacobian_e_pg(matrixType)
        dN_e_pg = mesh.Get_dN_e_pg(matrixType)
        nPg = weightedJacobian_e_pg.shape[1]

        # get hyperelastic data
        De_e_pg = HyperElastic.Compute_De(mesh, u, matrixType)
        dWde_e_pg = mat.Compute_dWde(mesh, u, matrixType) 
        d2Wde_e_pg = mat.Compute_d2Wde(mesh, u, matrixType)        

        # init matrices
        grad_e_pg = FeArray.zeros(Ne, nPg, 9, dim*nPe)
        Sig_e_pg = FeArray.zeros(Ne, nPg, 9, 9)
        sig_e_pg = Project_vector_to_matrix(dWde_e_pg)
        
        rows = np.arange(9).reshape(3, -1)
        cols = np.arange(dim*nPe).reshape(3, -1)
        for i in range(dim):
            grad_e_pg._assemble(rows[i], cols[i], value=dN_e_pg)
            Sig_e_pg._assemble(rows[i], rows[i], value=sig_e_pg)

        B_e_pg = De_e_pg @ grad_e_pg

        # stiffness
        K1_e = (weightedJacobian_e_pg * B_e_pg.T @ d2Wde_e_pg @ B_e_pg).sum(1)
        K2_e = (weightedJacobian_e_pg * grad_e_pg.T @ Sig_e_pg @ grad_e_pg).sum(1)
        K_e = K1_e + K2_e
        
        # source
        F_e = - (weightedJacobian_e_pg * dWde_e_pg.T @ B_e_pg).sum(1)

        return K_e, F_e

    # --------------------------------------------------------------------------
    # Iterations
    # --------------------------------------------------------------------------

    def Save_Iter(self):
        return super().Save_Iter()
    
    def Set_Iter(self, iter = -1, resetAll=False):
        return super().Set_Iter(iter, resetAll)

    # --------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------- 

    def Results_Available(self):
        return super().Results_Available()
    
    def Result(self, option, nodeValues=True, iter=None):
        return super().Result(option, nodeValues, iter)
    
    def Results_Iter_Summary(self):
        return super().Results_Iter_Summary()
    
    def Results_dict_Energy(self):
        return super().Results_dict_Energy()
    
    def Results_displacement_matrix(self):
        return super().Results_displacement_matrix()
    
    def Results_nodesField_elementsField(self, details=False):
        return super().Results_nodesField_elementsField(details)