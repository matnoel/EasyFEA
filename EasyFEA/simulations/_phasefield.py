# Copyright (C) 2021-2024 Université Gustave Eiffel. All rights reserved.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from typing import Union
import numpy as np
from scipy import sparse
import pandas as pd

# utilities
from ..utilities import Display, Tic
from ..utilities._observers import Observable
# fem
from ..fem import Mesh, MatrixType
# materials
from .. import Materials
from ..materials import ModelType, _IModel, Reshape_variable, Result_in_Strain_or_Stress_field
# simu
from ._simu import _Simu

class PhaseFieldSimu(_Simu):

    def __init__(self, mesh: Mesh, model: Materials.PhaseField, verbosity=False, useNumba=True, useIterativeSolvers=True):
        """
        Creates a damage simulation.

        Parameters
        ----------
        mesh : Mesh
            The mesh used.
        model : PhaseField
            The model used.
        verbosity : bool, optional
            If True, the simulation can write to the console. Defaults to False.
        useNumba : bool, optional
            If True, numba can be used. Defaults to True.
        useIterativeSolvers : bool, optional
            If True, iterative solvers can be used. Defaults to True.
        """

        assert isinstance(model, Materials.PhaseField), "model must be a phase field model"
        super().__init__(mesh, model, verbosity, useNumba, useIterativeSolvers)
        
        # Init internal variable
        self.__psiP_e_pg = []
        self.__old_psiP_e_pg = [] # old positive elastic energy density psiPlus(e, pg, 1) to use the miehe history field
        self.Solver_Set_Elliptic_Algorithm()

        self.Need_Update()

        self.phaseFieldModel.material._Add_observer(self)

        self.__resumeLoading = ""

    def Results_nodesField_elementsField(self, details=False) -> tuple[list[str], list[str]]:
        if details:
            nodesField = ["displacement_matrix", "damage"]
            elementsField = ["Stress", "Strain", "psiP"]
        else:
            nodesField = ["displacement_matrix", "damage"]
            elementsField = ["Stress"]
        return nodesField, elementsField

    def Get_dofs(self, problemType=None) -> list[str]:        
        if problemType == ModelType.damage:
            return ["d"]
        elif problemType in [ModelType.elastic, None]:
            _dict_dim_directions_displacement = {
                2 : ["x", "y"],
                3 : ["x", "y", "z"]
            }
            return _dict_dim_directions_displacement[self.dim]
    
    def Get_problemTypes(self) -> list[ModelType]:
        return [ModelType.damage, ModelType.elastic]

    def Get_lb_ub(self, problemType: ModelType) -> tuple[np.ndarray, np.ndarray]:
        
        if problemType == ModelType.damage:
            solveur = self.phaseFieldModel.solver
            if solveur == "BoundConstrain":
                lb = self.damage
                lb[np.where(lb>=1)] = 1-np.finfo(float).eps
                ub = np.ones(lb.shape)
            else:
                lb, ub = np.array([]), np.array([])
        else:
            lb, ub = np.array([]), np.array([])
            
        return lb, ub

    def Get_dof_n(self, problemType=None) -> int:        
        if problemType == ModelType.damage:
            return 1
        elif problemType in [ModelType.elastic, None]:
            return self.dim

    @property
    def phaseFieldModel(self) -> Materials.PhaseField:
        """Damage model"""
        return self.model

    @property
    def displacement(self) -> np.ndarray:
        """Displacement vector field.\n
        2D [uxi, uyi, ...]\n
        3D [uxi, uyi, uzi, ...]"""
        return self._Get_u_n(ModelType.elastic)

    @property
    def damage(self) -> np.ndarray:
        """Damage scalar field.\n
        [di, ...]"""
        return self._Get_u_n(ModelType.damage)
    
    def Bc_dofs_nodes(self, nodes: np.ndarray, directions: list[str], problemType=ModelType.elastic) -> np.ndarray:
        return super().Bc_dofs_nodes(nodes, directions, problemType)

    def add_dirichlet(self, nodes: np.ndarray, values: np.ndarray, directions: list[str], problemType=ModelType.elastic, description=""):        
        return super().add_dirichlet(nodes, values, directions, problemType, description)
    
    def add_lineLoad(self, nodes: np.ndarray, values: list, directions: list[str], problemType=ModelType.elastic, description=""):
        return super().add_lineLoad(nodes, values, directions, problemType, description)

    def add_surfLoad(self, nodes: np.ndarray, values: list, directions: list[str], problemType=ModelType.elastic, description=""):
        return super().add_surfLoad(nodes, values, directions, problemType, description)
    
    def add_pressureLoad(self, nodes: np.ndarray, magnitude: float, problemType=ModelType.elastic, description="") -> None:
        return super().add_pressureLoad(nodes, magnitude, problemType, description)
        
    def add_neumann(self, nodes: np.ndarray, values: list, directions: list[str], problemType=ModelType.elastic, description=""):
        return super().add_neumann(nodes, values, directions, problemType, description)

    def Get_K_C_M_F(self, problemType=None) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        
        if problemType==None:
            problemType = ModelType.elastic

        # here always update to the last state
        if problemType == ModelType.elastic:
            if not self.__updatedDisplacement:
                self.__Assembly_displacement()
                self.__updatedDisplacement = True
            size = self.__Ku.shape[0]
            initcsr = sparse.csr_matrix((size, size))
            return self.__Ku.copy(), initcsr, initcsr, self.__Fu.copy()
        else:
            if not self.__updatedDamage:
                self.__Assembly_damage()
                self.__updatedDamage = True
            size = self.__Kd.shape[0]
            initcsr = sparse.csr_matrix((size, size))
            return self.__Kd.copy(), initcsr, initcsr, self.__Fd.copy()

    def _Update(self, observable: Observable, event: str) -> None:
        if isinstance(observable, _IModel):
            if event == 'The model has been modified' and not self.needUpdate:
                self.Need_Update()
        elif isinstance(observable, Mesh):
            if event == 'The mesh has been modified':
                self._Check_dim_mesh_material()
                self.Need_Update()
        else:
            Display.MyPrintError("Notification not yet implemented")

    @property
    def needUpdate(self) -> bool:
        return not self.__updatedDamage or not self.__updatedDisplacement

    def Need_Update(self, value=True) -> None:        
        # the following functions help to avoid assembling matrices too many times
        self.__updatedDamage = not value
        """The matrix system associated with the damage problem is updated."""
        self.__updatedDisplacement = not value
        """The matrix system associated with the displacement problem is updated."""

    def Get_x0(self, problemType=None):
        
        if problemType == ModelType.damage:
            if self.damage.size != self.mesh.Nn:
                return np.zeros(self.mesh.Nn)
            else:
                return self.damage
        elif problemType in [ModelType.elastic, None]:
            if self.displacement.size != self.mesh.Nn*self.dim:
                return np.zeros(self.mesh.Nn*self.dim)
            else:
                return self.displacement

    def Assembly(self) -> None:
        self.__Assembly_damage()
        self.__Assembly_displacement()
    
    def Solve(self, tolConv=1.0, maxIter=500, convOption=2) -> tuple[np.ndarray, np.ndarray, sparse.csr_matrix, bool]:
        """Solving the damage problem with the staggered scheme.

        Parameters
        ----------
        tolConv : float, optional
            tolerance between old and new damage, by default 1.0
        maxIter : int, optional
            Maximum number of iterations to reach convergence, by default 500
        convOption : int, optional
            0 -> convergence on damage np.max(np.abs(d_kp1-dk)) equivalent normInf(d_kp1-dk)\n
            1 -> convergence on crack energy np.abs(psi_crack_kp1 - psi_crack_k)/psi_crack_kp1 \n
            2 -> convergence on total energy np.abs(psi_tot_kp1 - psi_tot_k)/psi_tot_kp1

        Returns
        -------
        np.ndarray, np.ndarray, int, float
            u_np1, d_np1, Kglob, convergence

            such that :\n
            u_np1 : displacement vector field
            d_np1 : damage scalar field
            Kglob : displacement stiffness matrix
            convergence: the solution has converged
        """

        assert tolConv > 0 and tolConv <= 1 , "tolConv must be between 0 and 1."
        assert maxIter > 1 , "Must be > 1."

        Niter = 0
        convergence = False
        dn = self.damage

        solver = self.phaseFieldModel.solver
        regu = self.phaseFieldModel.regularization

        tic = Tic()

        while not convergence and Niter <= maxIter:
                    
            Niter += 1
            if convOption == 0:                    
                d_n = self.damage
            elif convOption == 1:
                psi_n = self._Calc_Psi_Crack()
            elif convOption == 2:
                psi_n = self._Calc_Psi_Crack() + self._Calc_Psi_Elas()
            elif convOption == 3:
                d_n = self.damage
                u_n = self.displacement

            # Damage
            d_np1 = self.__Solve_damage()
            self.__updatedDisplacement = False # new damage -> new displacement matrices

            # Displacement            
            u_np1 = self.__Solve_displacement()
            self.__updatedDamage = False # new displacement -> new damage matrices

            if convOption == 0:                
                convIter = np.max(np.abs(d_np1 - d_n))

            elif convOption in [1,2]:
                psi_np1 = self._Calc_Psi_Crack()
                if convOption == 2:
                   psi_np1 += self._Calc_Psi_Elas()

                if psi_np1 == 0:
                    convIter = np.abs(psi_np1 - psi_n)
                else:
                    convIter = np.abs(psi_np1 - psi_n)/psi_np1

            elif convOption == 3:
                # eq (25) Pech 2022 10.1016/j.engfracmech.2022.108591
                diffU = np.abs(u_np1 - u_n); diffU[u_np1 != 0] *= 1/np.abs(u_np1[u_np1 != 0])
                diffD = np.abs(d_np1 - d_n); diffD[d_np1 != 0] *= 1/np.abs(d_np1[d_np1 != 0])
                convU = np.sum(diffU)
                convD = np.sum(diffD)
                convIter = np.max([convD, convU])

            # Convergence condition
            if tolConv == 1:
                convergence = True
            elif convOption == 3:
                convergence = (convD <= tolConv) and (convU <= tolConv*0.999)
            else:
                convergence = convIter <= tolConv
                
        solverTypes = Materials.PhaseField.SolverType

        if solver in [solverTypes.History, solverTypes.BoundConstrain]:
            d_np1 = d_np1            
        elif solver == solverTypes.HistoryDamage:
            oldAndNewDamage = np.zeros((d_np1.shape[0], 2))
            oldAndNewDamage[:, 0] = dn
            oldAndNewDamage[:, 1] = d_np1
            d_np1 = np.max(oldAndNewDamage, 1)

        else:
            raise Exception("Solveur phase field unknown")

        timeIter = tic.Tac("Resolution phase field", "Phase Field iteration", False)

        self.__Niter = Niter
        self.__convIter = convIter
        self.__timeIter = timeIter

        Kglob = self.__Ku.copy()
            
        return u_np1, d_np1, Kglob, convergence


    def __Construct_Displacement_Matrix(self) -> np.ndarray:
        """Construct the elementary stiffness matrices for the displacement problem."""

        matrixType=MatrixType.rigi

        # Data
        mesh = self.mesh
        
        # Recovers matrices to work with        
        B_dep_e_pg = mesh.Get_B_e_pg(matrixType)
        leftDepPart = mesh.Get_leftDispPart(matrixType) # -> jacobian_e_pg * weight_pg * B_dep_e_pg'

        d = self.damage
        u = self.displacement

        phaseFieldModel = self.phaseFieldModel
        
        # Calculates the deformation required for the split
        Epsilon_e_pg = self._Calc_Epsilon_e_pg(u, matrixType)

        # Split of the behavior law
        cP_e_pg, cM_e_pg = phaseFieldModel.Calc_C(Epsilon_e_pg)

        tic = Tic()
        
        # Damage : c = g(d) * cP + cM
        g_e_pg = phaseFieldModel.get_g_e_pg(d, mesh, matrixType)
        cP_e_pg = np.einsum('ep,epij->epij', g_e_pg, cP_e_pg, optimize='optimal')

        c_e_pg = cP_e_pg + cM_e_pg
        
        # Elemental stiffness matrix
        Ku_e = np.sum(leftDepPart @ c_e_pg @ B_dep_e_pg, axis=1)

        if self.dim == 2:
            thickness = self.phaseFieldModel.thickness
            Ku_e *= thickness
        
        tic.Tac("Matrix","Construction Ku_e", self._verbosity)

        return Ku_e
 
    def __Assembly_displacement(self) -> sparse.csr_matrix:
        """Construct the displacement problem."""

        # Data
        mesh = self.mesh        
        nDof = mesh.Nn*self.dim
        
        nDof += self._Bc_Lagrange_dim(ModelType.elastic)

        Ku_e = self.__Construct_Displacement_Matrix()

        tic = Tic()

        linesVector_e = mesh.linesVector_e.ravel()
        columnsVector_e = mesh.columnsVector_e.ravel()

        # Assembly
        self.__Ku = sparse.csr_matrix((Ku_e.ravel(), (linesVector_e, columnsVector_e)), shape=(nDof, nDof))
        """Kglob matrix for the displacement problem (nDof, nDof)"""
        
        self.__Fu = sparse.csr_matrix((nDof, 1))
        """Fglob vector for the displacement problem (nDof, 1)"""

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.spy(self.__Ku)
        # plt.show()        

        tic.Tac("Matrix","Assembly Ku and Fu", self._verbosity)
        return self.__Ku

        # # Here, we always want the matrices to be updated with the latest damage or displacement results.
        # # That's why we don't say the matrices have been updated

    def __Solve_displacement(self) -> np.ndarray:
        """Solving the displacement problem."""
            
        self._Solver_Solve(ModelType.elastic)
       
        return self.displacement

    # ------------------------------------------- PROBLEME ENDOMMAGEMENT ------------------------------------------- 

    def __Calc_psiPlus_e_pg(self):
        """Calculation of the positive energy density.
        For each gauss point of all mesh elements, we calculate psi+.
        """

        phaseFieldModel = self.phaseFieldModel
        
        u = self.displacement
        d = self.damage

        testu = isinstance(u, np.ndarray) and (u.shape[0] == self.mesh.Nn*self.dim )
        testd = isinstance(d, np.ndarray) and (d.shape[0] == self.mesh.Nn )

        assert testu or testd, "Dimension problem."

        Epsilon_e_pg = self._Calc_Epsilon_e_pg(u, MatrixType.mass)
        # here the mass term is important otherwise we under-integrate

        # Energy calculation
        psiP_e_pg, psiM_e_pg = phaseFieldModel.Calc_psi_e_pg(Epsilon_e_pg)

        if phaseFieldModel.solver == "History":
            # Get the old history field
            old_psiPlus_e_pg = self.__old_psiP_e_pg.copy()
            
            if isinstance(old_psiPlus_e_pg, list) and len(old_psiPlus_e_pg) == 0:
                # No damage available yet
                old_psiPlus_e_pg = np.zeros_like(psiP_e_pg)
            
            if old_psiPlus_e_pg.shape != psiP_e_pg.shape:
                # the mesh has been changed, the value must be recalculated
                # here I do nothing
                old_psiPlus_e_pg = np.zeros_like(psiP_e_pg)

            inc_H = psiP_e_pg - old_psiPlus_e_pg

            elements, gaussPoints = np.where(inc_H < 0)

            psiP_e_pg[elements, gaussPoints] = old_psiPlus_e_pg[elements, gaussPoints]

            # new = np.linalg.norm(psiP_e_pg)
            # old = np.linalg.norm(self.__old_psiP_e_pg)
            # assert new >= old, "Erreur"
            
        self.__psiP_e_pg = psiP_e_pg

        return self.__psiP_e_pg
    
    def __Construct_Damage_Matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Construct the elementary matrices for the damage problem."""

        phaseFieldModel = self.phaseFieldModel

        # Data
        k = phaseFieldModel.k
        PsiP_e_pg = self.__Calc_psiPlus_e_pg()
        r_e_pg = phaseFieldModel.get_r_e_pg(PsiP_e_pg)
        f_e_pg = phaseFieldModel.get_f_e_pg(PsiP_e_pg)

        matrixType=MatrixType.mass

        mesh = self.mesh
        Ne = mesh.Ne
        nPg = r_e_pg.shape[1]

        # K * Laplacien(d) + r * d = F        
        ReactionPart_e_pg = mesh.Get_ReactionPart_e_pg(matrixType) # -> jacobian_e_pg * weight_pg * Nd_pg' * Nd_pg
        DiffusePart_e_pg = mesh.Get_DiffusePart_e_pg(matrixType, phaseFieldModel.A) # -> jacobian_e_pg, weight_pg, Bd_e_pg', A, Bd_e_pg
        SourcePart_e_pg = mesh.Get_SourcePart_e_pg(matrixType) # -> jacobian_e_pg, weight_pg, Nd_pg'
        
        tic = Tic()

        # Part that involves the reaction term r ->  jacobian_e_pg * weight_pg * r_e_pg * Nd_pg' * Nd_pg
        K_r_e = np.einsum('ep,epij->eij', r_e_pg, ReactionPart_e_pg, optimize='optimal')

        # The part that involves diffusion K -> jacobian_e_pg, weight_pg, k, Bd_e_pg', Bd_e_pg
        k_e_pg = Reshape_variable(k, Ne, nPg)
        K_K_e = np.einsum('ep,epij->eij', k_e_pg, DiffusePart_e_pg, optimize='optimal')
        
        # Source part Fd_e -> jacobian_e_pg, weight_pg, f_e_pg, Nd_pg'
        Fd_e = np.einsum('ep,epij->eij', f_e_pg, SourcePart_e_pg, optimize='optimal')
    
        Kd_e = K_r_e + K_K_e

        if self.dim == 2:
            # THICKNESS not used in femobject !
            thickness = phaseFieldModel.thickness
            Kd_e *= thickness
            Fd_e *= thickness
        
        tic.Tac("Matrix","Construc Kd_e and Fd_e", self._verbosity)        

        return Kd_e, Fd_e

    def __Assembly_damage(self) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Construct the damage problem."""
       
        # Data
        mesh = self.mesh
        nDof = mesh.Nn
        linesScalar_e = mesh.linesScalar_e.ravel()
        columnsScalar_e = mesh.columnsScalar_e.ravel()

        # Additional dimension linked to the use of lagrange coefficients        
        nDof += self._Bc_Lagrange_dim(ModelType.damage)
        
        # Calculating elementary matrix
        Kd_e, Fd_e = self.__Construct_Damage_Matrix()

        # Assemblage
        tic = Tic()        

        self.__Kd = sparse.csr_matrix((Kd_e.ravel(), (linesScalar_e, columnsScalar_e)), shape = (nDof, nDof))
        """Kglob for damage problem (Nn, Nn)"""
        
        lignes = mesh.connect.ravel()
        self.__Fd = sparse.csr_matrix((Fd_e.ravel(), (lignes,np.zeros(len(lignes)))), shape = (nDof,1))
        """Fglob for damage problem (Nn, 1)"""        

        tic.Tac("Matrix","Assembly Kd and Fd", self._verbosity)

        # # Here, we always want the matrices to be updated with the latest damage or displacement results.
        # # That's why we don't say the matrices have been updated

        return self.__Kd, self.__Fd
    
    def __Solve_damage(self) -> np.ndarray:
        """Solving the damage problem."""
        
        self._Solver_Solve(ModelType.damage)

        return self.damage

    def Save_Iter(self):

        iter = super().Save_Iter()

        # convergence information        
        iter["Niter"] = self.__Niter
        iter["timeIter"] = self.__timeIter
        iter["convIter"] = self.__convIter
        
        if self.phaseFieldModel.solver == self.phaseFieldModel.SolverType.History:
            # update old history field for next resolution
            self.__old_psiP_e_pg = self.__psiP_e_pg
            
        iter["displacement"] = self.displacement
        iter["damage"] = self.damage

        self._results.append(iter)

    def Set_Iter(self, iter=-1) -> list[dict]:

        results = super().Set_Iter(iter)

        if results is None: return

        damageType = ModelType.damage
        self._Set_u_n(damageType, results[damageType])

        displacementType = ModelType.elastic
        self._Set_u_n(displacementType, results["displacement"])

        # damage and displacement field will change thats why we need to update the assembled matrices
        self.__updatedDamage = False
        self.__updatedDisplacement = False

        if self.phaseFieldModel.solver == self.phaseFieldModel.SolverType.History:
            # It's really useful to do this otherwise when we calculate psiP there will be a problem
            self.__old_psiP_e_pg = []
            self.__old_psiP_e_pg = self.__Calc_psiPlus_e_pg() # update psi+ with the current state

        return results

    def Results_Available(self) -> list[str]:

        results = []
        dim = self.dim

        results.extend(["displacement", "displacement_norm", "displacement_matrix"])
        
        if dim == 2:
            results.extend(["ux", "uy"])
            results.extend(["Sxx", "Syy", "Sxy"])
            results.extend(["Exx", "Eyy", "Exy"])

        elif dim == 3:
            results.extend(["ux", "uy", "uz"])
            results.extend(["Sxx", "Syy", "Szz", "Syz", "Sxz", "Sxy"])
            results.extend(["Exx", "Eyy", "Ezz", "Eyz", "Exz", "Exy"])
        
        results.extend(["Svm","Stress","Evm","Strain"])

        results.extend(["damage","psiP","Psi_Crack"])
        results.extend(["Wdef"])

        return results
    
    def Result(self, result: str, nodeValues=True, iter=None) -> Union[np.ndarray, float, None]:
        
        if iter != None:
            self.Set_Iter(iter)
        
        if not self._Results_Check_Available(result): return None

        # begin cases ----------------------------------------------------

        Nn = self.mesh.Nn

        values = None

        if result in ["Wdef"]:
            return self._Calc_Psi_Elas()

        elif result == "Wdef_e":
            values = self._Calc_Psi_Elas(returnScalar=False)

        elif result == "Psi_Crack":
            return self._Calc_Psi_Crack()

        if result == "psiP":
            values_e_pg = self.__Calc_psiPlus_e_pg()
            values = np.mean(values_e_pg, axis=1)

        if result == "damage":
            values = self.damage

        elif result in ["ux", "uy", "uz"]:
            values_n = self.displacement.reshape(Nn, -1)
            values = values_n[:,self.__indexResult(result)]

        elif result == "displacement":
            values = self.displacement
        
        elif result == "displacement_norm":
            val_n = self.displacement.reshape(Nn, -1)
            values = np.linalg.norm(val_n, axis=1)

        elif result == "displacement_matrix":
            values = self.Results_displacement_matrix()
        
        elif ("S" in result or "E" in result) and (not "_norm" in result):
            # Strain and Stress calculation part

            coef = self.phaseFieldModel.material.coef

            displacement = self.displacement
            # Strain and stress for each element and gauss point
            Epsilon_e_pg = self._Calc_Epsilon_e_pg(displacement)
            Sigma_e_pg = self._Calc_Sigma_e_pg(Epsilon_e_pg)

            # Element average
            if "S" in result and result != "Strain":
                val_e = Sigma_e_pg.mean(1)
            elif "E" in result or result == "Strain":
                val_e = Epsilon_e_pg.mean(1)
            else:
                raise Exception("Wrong option")
            
            res = result if result in ["Strain", "Stress"] else result[-2:]
            
            values = Result_in_Strain_or_Stress_field(val_e, res, coef)

        if not isinstance(values, np.ndarray):
            Display.MyPrintError("This result option is not implemented yet.")
            return

        # end cases ----------------------------------------------------
        
        return self.Results_Reshape_values(values, nodeValues)

    def __indexResult(self, resultat: str) -> int:

        dim = self.dim

        if len(resultat) <= 2:
            if "x" in resultat:
                return 0
            elif "y" in resultat:
                return 1
            elif "z" in resultat:
                return 1

    def _Calc_Psi_Elas(self) -> float:
        """Calculation of the kinematically admissible deformation energy, damaged or not.
        Wdef = 1/2 int_Omega jacobian * weight * Sig : Eps dOmega thickness"""

        tic = Tic()

        u = self.displacement.reshape(-1,1)
        Ku = self.Get_K_C_M_F(ModelType.elastic)[0]
        Wdef = 1/2 * float(u.T @ Ku @ u)

        tic.Tac("PostProcessing","Calc Psi Elas",False)
        
        return Wdef

    def _Calc_Psi_Crack(self) -> float:
        """Calculating crack energy."""

        tic = Tic()
        
        d = self.damage.reshape(-1,1)        
        Kd = self.Get_K_C_M_F(ModelType.damage)[0]
        Psi_Crack = 1/2 * float(d.T @ Kd @ d)

        tic.Tac("PostProcessing","Calc Psi Crack",False)

        return Psi_Crack

    def _Calc_Epsilon_e_pg(self, sol: np.ndarray, matrixType=MatrixType.rigi):
        """Builds epsilon for each element and each gauss point.\n
        2D : [Exx Eyy sqrt(2)*Exy]\n
        3D : [Exx Eyy Ezz sqrt(2)*Eyz sqrt(2)*Exz sqrt(2)*Exy]

        Parameters
        ----------
        sol : np.ndarray
            Displacement vector

        Returns
        -------
        np.ndarray
            Deformations stored at elements and gauss points (Ne,pg,(3 or 6))
        """
        
        tic = Tic()        
        u_e = sol[self.mesh.assembly_e]
        B_dep_e_pg = self.mesh.Get_B_e_pg(matrixType)
        Epsilon_e_pg = np.einsum('epij,ej->epi', B_dep_e_pg, u_e, optimize='optimal')            
        
        tic.Tac("Matrix", "Epsilon_e_pg", False)

        return Epsilon_e_pg

    def _Calc_Sigma_e_pg(self, Epsilon_e_pg: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Calculating stresses from strains.\n
        2D : [Sxx Syy sqrt(2)*Sxy]\n
        3D : [Sxx Syy Szz sqrt(2)*Syz sqrt(2)*Sxz sqrt(2)*Sxy]

        Parameters
        ----------
        Epsilon_e_pg : np.ndarray
            Deformations stored at elements and gauss points (Ne,pg,(3 or 6))

        Returns
        -------
        np.ndarray
            Returns damaged or undamaged constraints (Ne,pg,(3 or 6))
        """

        assert Epsilon_e_pg.shape[0] == self.mesh.Ne
        assert Epsilon_e_pg.shape[1] == self.mesh.Get_nPg(matrixType)

        d = self.damage

        phaseFieldModel = self.phaseFieldModel

        SigmaP_e_pg, SigmaM_e_pg = phaseFieldModel.Calc_Sigma_e_pg(Epsilon_e_pg)
        
        # Damage : Sig = g(d) * SigP + SigM
        g_e_pg = phaseFieldModel.get_g_e_pg(d, self.mesh, matrixType)
        
        tic = Tic()
        
        SigmaP_e_pg = np.einsum('ep,epi->epi', g_e_pg, SigmaP_e_pg, optimize='optimal')

        Sigma_e_pg = SigmaP_e_pg + SigmaM_e_pg
            
        tic.Tac("Matrix", "Sigma_e_pg", False)

        return Sigma_e_pg

    def Results_Set_Bc_Summary(self, loadMax: float, listInc: list, listTreshold: list, listOption: list):
        assert len(listInc) == len(listTreshold) and len(listInc) == len(listOption), "Must be the same dimension."
        
        resume = 'Loading :'
        resume += f'\n\tload max = {loadMax:.3}'

        for inc, treshold, option in zip(listInc, listTreshold, listOption):

            resume += f'\n\tinc = {inc} -> {option} < {treshold:.4e}'
        
        self.__resumeLoading = resume

        return self.__resumeLoading

    def Results_Get_Bc_Summary(self) -> str:
        return self.__resumeLoading

    def Results_Set_Iteration_Summary(self, iter: int, load: float, uniteLoad: str, percentage=0.0, remove=False) -> str:
        """Builds the iteration summary for the damage problem

        Parameters
        ----------
        iter : int
            iteration
        load : float
            loading
        uniteLoad : str
            loading unit
        percentage : float, optional
            percentage of simualtion performed, by default 0.0
        remove : bool, optional
            removes line from terminal after display, by default False
        """

        d = self.damage

        nombreIter = self.__Niter
        dincMax = self.__convIter
        timeIter = self.__timeIter

        min_d = d.min()
        max_d = d.max()
        summaryIter = f"{iter:4} : {load:4.3f} {uniteLoad}, [{min_d:.2e}; {max_d:.2e}], {nombreIter}:{timeIter:4.3f} s, tol={dincMax:.2e}  "
        
        if remove:
            end='\r'
        else:
            end=''

        if percentage > 0:
            timeLeft = (1/percentage-1)*timeIter*iter            
            timeCoef, unite = Tic.Get_time_unity(timeLeft)
            # Adds percentage and estimated time remaining
            summaryIter = summaryIter + f"{percentage*100:3.2f} % -> {timeCoef:3.2f} {unite}  "

        Display.MyPrint(summaryIter, end=end)

        self.__resumeIter = summaryIter

    def Results_Get_Iteration_Summary(self) -> str: 
        return self.__resumeIter

    def Results_dict_Energy(self) -> dict[str, float]:
        PsiElas = self._Calc_Psi_Elas()
        PsiCrack = self._Calc_Psi_Crack()
        dict_Energie = {
            r"$\Psi_{elas}$": PsiElas,
            r"$\Psi_{crack}$": PsiCrack,
            r"$\Psi_{tot}$": PsiCrack+PsiElas
            }
        return dict_Energie

    def Results_Iter_Summary(self) -> list[tuple[str, np.ndarray]]:
        
        list_label_values = []
        
        resultats = self.results
        df = pd.DataFrame(resultats)
        iterations = np.arange(df.shape[0])
        
        damageMaxIter = np.array([np.max(damage) for damage in df["damage"].values])
        list_label_values.append((r"$\phi$", damageMaxIter))

        tolConvergence = df["convIter"].values
        list_label_values.append(("converg", tolConvergence))

        nombreIter = df["Niter"].values
        list_label_values.append(("Niter", nombreIter))

        tempsIter = df["timeIter"].values
        list_label_values.append(("time", tempsIter))
        
        return iterations, list_label_values
    
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