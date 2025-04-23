# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information

from scipy import sparse
import scipy.sparse.linalg as sla
import numpy as np
from typing import Union

# utilities
from ..utilities import Tic, Display
# fem
from ..fem import Mesh, MatrixType, FeArray
# materials
from ..materials import ModelType, Project_vector_to_matrix, Result_in_Strain_or_Stress_field, Project_Kelvin
from ..materials._hyperelastic_laws import _HyperElas
from ..materials._hyperelastic import HyperElastic
# simu
from ._simu import _Simu

class HyperElasticSimu(_Simu):

    def __init__(self, mesh: Mesh, model: _HyperElas, verbosity=False, useNumba=True, useIterativeSolvers=True):
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

        # init
        self.Solver_Set_Elliptic_Algorithm()

    # --------------------------------------------------------------------------
    # General
    # --------------------------------------------------------------------------

    def Get_problemTypes(self):
        return [ModelType.hyperelastic]
    
    def Get_unknowns(self, problemType=None) -> list[str]:
        dict_unknowns = {
            2 : ["x", "y"],
            3 : ["x", "y", "z"]
        }
        return dict_unknowns[self.dim]
    
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
    
    def Solve(self, tolConv=1.e-5, maxIter=20) -> tuple[np.ndarray, bool]:
        """Solves the hyperelastic problem using the newton raphson algorithm.

        Parameters
        ----------
        tolConv : float, optional
            threshold used to check convergence, by default 1e-3
        maxIter : int, optional
            Maximum iterations for convergence, by default 20

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
        problemType = self.problemType

        tic = Tic()

        # init u
        u = self.Bc_vector_Dirichlet(problemType)
        self._Set_u_n(problemType, u)
        # u = self.displacement

        # apply new dirichlet bc conditions
        previous_dirichlet = self.Bc_Dirichlet
        previous_neumann = self.Bc_Neuman
        self.Bc_Init()
        for bc in previous_dirichlet:
            self._Bc_Add_Dirichlet(problemType, bc.nodes, bc.dofsValues*0, bc.dofs, bc.unknowns)
        for bc in previous_neumann:
            self._Bc_Add_Neumann(problemType, bc.nodes, bc.dofsValues, bc.dofs, bc.unknowns)

        list_res = []

        while not converged and Niter < maxIter:
                    
            Niter += 1

            # solve here
            delta_u = self._Solve_hyperelastic()

            # uptate new displacement
            u += delta_u
            self._Set_u_n(problemType, u)

            # check convergence
            r = self._Solver_Apply_Neumann(problemType)
            norm_r = sla.norm(r)

            if Niter == 1:
                converged = norm_r < tolConv
            else:
                res = np.abs(list_res[-1] - norm_r)/list_res[-1]
                converged = res < tolConv
            list_res.append(norm_r)

        timeIter = tic.Tac("Resolution hyperelastic", "Hyperelastic iteration", False)

        assert converged, f"Newton raphson algorithm did not converged in {Niter} iterations."

        # # save solve config
        # self.__tolConv = tolConv
        # self.__convOption = convOption
        # self.__maxIter = maxIter
        # # save iter parameters
        # self.__Niter = Niter
        # self.__convIter = convIter
        # self.__timeIter = timeIter
            
        return u
    
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
        # plt.spy(self.__K)
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
        reorder = np.arange(0, nPe*dim).reshape(-1, nPe).T.ravel()
        F_e = F_e[:,reorder]
        K_e = K_e[:,reorder][:,:,reorder]

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

    def __indexResult(self, result: str) -> int:
        
        if len(result) <= 2:
            "Case were ui, vi or ai"
            if "x" in result:
                return 0
            elif "y" in result:
                return 1
            elif "z" in result:
                return 2

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
        
        results.extend(["W","W_e"])

        return results
    
    def Result(self, result: str, nodeValues=True, iter=None) -> Union[np.ndarray, float, None]:

        if iter != None:
            self.Set_Iter(iter)
        
        if not self._Results_Check_Available(result): return None

        # begin cases ----------------------------------------------------

        Nn = self.mesh.Nn

        values = None

        if result in ["ux", "uy", "uz"]:
            values_n = self.displacement.reshape(Nn, -1)
            values = values_n[:,self.__indexResult(result)]

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
        
        elif ("S" in result or "E" in result) and (not "_norm" in result):
            # Green-Lagrange and second Piola-Kirchhoff for each element and gauss point

            # Element average
            if "S" in result:
                S_e_pg = self._Calc_PiolaKirchhoff()
                val_e = S_e_pg.mean(1)
            elif "E" in result:
                E_e_pg = self._Calc_GreenLagrange()
                val_e = E_e_pg.mean(1)
            else:
                raise Exception("Wrong option")
            
            res = result if result in ["Green-Lagrange", "Piola-Kirchhoff"] else result[-2:]
            
            values = Result_in_Strain_or_Stress_field(val_e, res, self.material.coef)

        if not isinstance(values, np.ndarray):
            Display.MyPrintError("This result option is not implemented yet.")
            return

        # end cases ----------------------------------------------------
        
        return self.Results_Reshape_values(values, nodeValues)
    
    def _Calc_W(self, returnScalar=True, matrixType=MatrixType.rigi):

        weightedJacobian_e_pg = self.mesh.Get_weightedJacobian_e_pg(matrixType)
        if self.dim == 2:
            weightedJacobian_e_pg *= self.material.thickness
        W_e_pg = self.material.Compute_W(self.mesh, self.displacement, matrixType)

        if returnScalar:
            return (weightedJacobian_e_pg * W_e_pg).sum()
        else:
            return (weightedJacobian_e_pg * W_e_pg).sum(1)
    
    def _Calc_GreenLagrange(self, matrixType=MatrixType.rigi):

        return Project_Kelvin(HyperElastic.Compute_GreenLagrange(self.mesh, self.displacement), 2)

    def _Calc_PiolaKirchhoff(self, matrixType=MatrixType.rigi):

        return self.material.Compute_dWde(self.mesh, self.displacement, matrixType)
    
    def Results_Iter_Summary(self):
        return super().Results_Iter_Summary()
    
    def Results_dict_Energy(self):
        return super().Results_dict_Energy()
    
    def Results_displacement_matrix(self) -> np.ndarray:

        Nn = self.mesh.Nn
        coord = self.displacement.reshape((Nn,-1))
        dim = coord.shape[1]

        if dim == 1:
            # Here we add two columns
            coord = np.append(coord, np.zeros((Nn,1)), axis=1)
            coord = np.append(coord, np.zeros((Nn,1)), axis=1)
        elif dim == 2:
            # Here we add 1 column
            coord = np.append(coord, np.zeros((Nn,1)), axis=1)

        return coord
    
    def Results_nodesField_elementsField(self, details=False):
        return super().Results_nodesField_elementsField(details)