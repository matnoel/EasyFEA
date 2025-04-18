# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information

from scipy import sparse
import scipy.sparse.linalg as sla
import numpy as np

# utilities
from ..utilities import Tic
# fem
from ..fem import Mesh, MatrixType, FeArray
# materials
from ..materials import ModelType, Project_vector_to_matrix
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
    
    @property
    def displacement(self) -> np.ndarray:
        """Displacement vector field.\n    
        [uxi, uyi, uzi, ...]"""
        return self._Get_u_n(self.problemType)
    
    # --------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------- 

    def Get_K_C_M_F(self, problemType=None):

        if self.needUpdate:
            self.Assembly()
            self.Need_Update(False)

        size = self.mesh.Nn * self.dim
        initcsr = sparse.csr_matrix((size, size))

        return self.__K.copy(), initcsr, initcsr, self.__F.copy()
    
    def Get_x0(self, problemType=None):
        return super().Get_x0(problemType)
    
    def _Solve_hyperelastic(self):
        
        self._Solver_Solve(self.problemType)
        self.Need_Update()

        return self._Get_u_n(self.problemType)
    
    def Solve(self, tolConv=1.e-3, maxIter=5) -> tuple[np.ndarray, bool]:
        """Solves the hyperelastic problem using the newton raphson algorithm.

        Parameters
        ----------
        tolConv : float, optional
            threshold used to check convergence, by default 1e-3
        maxIter : int, optional
            Maximum iterations for convergence, by default 5

        Returns
        -------
        np.ndarray, bool
            u_np1, converged

            such that:\n
            - u_np1: displacement vector field\n
            - converged: the solution has converged\n
        """

        assert 0 < tolConv < 1 , "tolConv must be between 0 and 1."
        assert maxIter > 1 , "Must be > 1."

        Niter = 0
        converged = False

        tic = Tic()

        u = self.displacement

        while not converged and Niter < maxIter:
                    
            Niter += 1

            # solve here
            delta_u = self._Solve_hyperelastic()

            u += delta_u
            self._Set_u_n(self.problemType, u)

            r = self._Solver_Apply_Neumann(self.problemType)
            norm_r = sla.norm(r)

            converged = norm_r < tolConv

            pass

        timeIter = tic.Tac("Resolution hyperelastic", "Hyperelastic iteration", False)

        # # save solve config
        # self.__tolConv = tolConv
        # self.__convOption = convOption
        # self.__maxIter = maxIter
        # # save iter parameters
        # self.__Niter = Niter
        # self.__convIter = convIter
        # self.__timeIter = timeIter
            
        return u, converged
    
    def Assembly(self):

        # Data
        mesh = self.mesh        
        Ndof = mesh.Nn*self.dim

        # Additional dimension linked to the use of lagrange coefficients
        Ndof += self._Bc_Lagrange_dim(self.problemType)
                        
        K_e, F_e = self.__Construct_Local_Matrix()
        
        tic = Tic()

        linesVector_e = mesh.linesVector_e.ravel()
        columnsVector_e = mesh.columnsVector_e.ravel()

        # Assembly
        self.__K = sparse.csr_matrix((K_e.ravel(), (linesVector_e, columnsVector_e)), shape=(Ndof, Ndof))
        """Kglob matrix for the displacement problem (Ndof, Ndof)"""
        
        rows = mesh.assembly_e.ravel()
        self.__F = sparse.csr_matrix((F_e.ravel(), (rows, np.zeros_like(rows))), shape=(Ndof, 1))
        """Fglob vector for the displacement problem (Ndof, 1)"""

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.spy(self.__Ku)
        # plt.show()

        tic.Tac("Matrix","Assembly Ku and Fu", self._verbosity)

        return self.__K, self.__F
    
    def __Construct_Local_Matrix(self) -> tuple[FeArray, FeArray]:


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
        displacement = self.displacement
        De_e_pg = HyperElastic.Compute_De(mesh, displacement, matrixType)
        dWde_e_pg = mat.Compute_dWde(mesh, displacement, matrixType) 
        d2Wde_e_pg = mat.Compute_d2Wde(mesh, displacement, matrixType)        

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

        # reorder xi,...,xn,yi,...yn,zi,...,zn to xi,yi,zi,...,xn,yx,zn
        reorder = np.arange(0, nPe*dim).reshape(nPe, -1).T.ravel()
        F_e = F_e[:,reorder]
        K_e = K_e[:,reorder,:][:,:,reorder]

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