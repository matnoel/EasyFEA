# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from typing import Union, Callable
import numpy as np
from scipy import sparse

# utilities
from ..utilities import Folder, Display, Tic
# fem
from ..fem import Mesh, MatrixType, Mesher
# materials
from .. import Materials
from ..materials import ModelType, Reshape_variable, Result_in_Strain_or_Stress_field
# simu
from ._simu import _Simu
from .Solvers import AlgoType

class ElasticSimu(_Simu):

    def __init__(self, mesh: Mesh, model: Materials._Elas, verbosity=False, useNumba=True, useIterativeSolvers=True):
        """Creates a elastic simulation.

        Parameters
        ----------
        mesh : Mesh
            The mesh used.
        model : _Elas
            The elastic model (or material) used.
        verbosity : bool, optional
            If True, the simulation can write in the terminal. Defaults to False.
        useNumba : bool, optional
            If True, numba can be used. Defaults to True.
        useIterativeSolvers : bool, optional
            If True, iterative solvers can be used. Defaults to True.
        """

        assert isinstance(model, Materials._Elas), "model must be a elastic model"
        super().__init__(mesh, model, verbosity, useNumba, useIterativeSolvers)

        # init
        self.Set_Rayleigh_Damping_Coefs()
        self.Solver_Set_Elliptic_Algorithm()    

    def Results_nodesField_elementsField(self, details=False) -> tuple[list[str], list[str]]:
        nodesField = ["displacement_matrix"]
        if details:            
            elementsField = ["Stress", "Strain"]
        else:            
            elementsField = ["Stress"]
        if self.algo == AlgoType.hyperbolic: nodesField.extend(["speed", "accel"])
        return nodesField, elementsField
    
    def Get_dofs(self, problemType=None) -> list[str]:
        dict_dim_directions = {
            2 : ["x", "y"],
            3 : ["x", "y", "z"]
        }
        return dict_dim_directions[self.dim]
    
    def Get_problemTypes(self) -> list[ModelType]:
        return [ModelType.elastic]
        
    def Get_dof_n(self, problemType=None) -> int:
        return self.dim

    @property
    def material(self) -> Materials._Elas:
        """elastic material"""
        return self.model

    @property
    def displacement(self) -> np.ndarray:
        """Displacement vector field.\n
        2D [uxi, uyi, ...]\n
        3D [uxi, uyi, uzi, ...]"""
        return self._Get_u_n(self.problemType)

    @property
    def speed(self) -> np.ndarray:
        """Velocity vector field.\n
        2D [vxi, vyi, ...]\n
        3D [vxi, vyi, vzi, ...]"""
        return self._Get_v_n(self.problemType)

    @property
    def accel(self) -> np.ndarray:
        """Acceleration vector field.\n
        2D [axi, ayi, ...]\n
        3D [axi, ayi, azi, ...]"""
        return self._Get_a_n(self.problemType)

    def __Construct_Local_Matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Computes the elementary stiffness matrices for the elastic problem."""

        matrixType=MatrixType.rigi
        
        mesh = self.mesh; Ne = mesh.Ne
        jacobian_e_pg = mesh.Get_jacobian_e_pg(matrixType)
        weight_pg = mesh.Get_weight_pg(matrixType)
        nPg = weight_pg.size

        N_pg = mesh.Get_N_vector_pg(matrixType)
        rho = self.rho
        
        B_dep_e_pg = mesh.Get_B_e_pg(matrixType)
        leftDepPart = mesh.Get_leftDispPart(matrixType) # -> jacobian_e_pg * weight_pg * B_dep_e_pg'

        mat = self.material

        tic = Tic()
        
        matC = mat.C

        # Stifness
        matC = Reshape_variable(matC, Ne, nPg)
        Ku_e = np.sum(leftDepPart @ matC @ B_dep_e_pg, axis=1)
        
        # Mass
        rho_e_pg = Reshape_variable(rho, Ne, nPg)
        Mu_e = np.einsum(f'ep,p,pdi,ep,pdj->eij', jacobian_e_pg, weight_pg, N_pg, rho_e_pg, N_pg, optimize="optimal")

        if self.dim == 2:
            thickness = self.material.thickness
            Ku_e *= thickness
            Mu_e *= thickness
        
        tic.Tac("Matrix","Construct Ku_e and Mu_e", self._verbosity)

        return Ku_e, Mu_e

    def Get_K_C_M_F(self, problemType=None) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        if self.needUpdate:
            self.Assembly()
            self.Need_Update(False)

        Cu = self.__coefK * self.__Ku + self.__coefM * self.__Mu
        
        return self.__Ku.copy(), Cu, self.__Mu.copy(), self.__Fu.copy()
 
    def Assembly(self) -> None:

        # Data
        mesh = self.mesh        
        Ndof = mesh.Nn*self.dim

        # Additional dimension linked to the use of lagrange coefficients
        Ndof += self._Bc_Lagrange_dim(self.problemType)
                        
        Ku_e, Mu_e = self.__Construct_Local_Matrix()
        
        tic = Tic()

        linesVector_e = mesh.linesVector_e.ravel()
        columnsVector_e = mesh.columnsVector_e.ravel()

        # Assembly
        self.__Ku = sparse.csr_matrix((Ku_e.ravel(), (linesVector_e, columnsVector_e)), shape=(Ndof, Ndof))
        """Kglob matrix for the displacement problem (Ndof, Ndof)"""

        # Here I'm initializing Fu because I'd have to calculate the volumetric forces in __Construct_Local_Matrix.
        self.__Fu = sparse.csr_matrix((Ndof, 1))
        """Fglob vector for the displacement problem (Ndof, 1)"""

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.spy(self.__Ku)
        # plt.show()

        self.__Mu = sparse.csr_matrix((Mu_e.ravel(), (linesVector_e, columnsVector_e)), shape=(Ndof, Ndof))
        """Mglob matrix for the displacement problem (Ndof, Ndof)"""

        tic.Tac("Matrix","Assembly Ku, Mu and Fu", self._verbosity)

    def Set_Rayleigh_Damping_Coefs(self, coefM=0.0, coefK=0.0):
        """Sets damping coefficients."""
        self.__coefM = coefM
        self.__coefK = coefK    

    def Get_x0(self, problemType=None):
        algo = self.algo
        if self.displacement.size != self.mesh.Nn*self.dim:
            return np.zeros(self.mesh.Nn*self.dim)
        elif algo == AlgoType.elliptic:
            return self.displacement
        elif algo == AlgoType.hyperbolic:
            return self.accel
    
    def Save_Iter(self):
        
        iter = super().Save_Iter()

        iter['displacement'] = self.displacement
        if self.algo == AlgoType.hyperbolic:
            iter["speed"] = self.speed
            iter["accel"] = self.accel

        self._results.append(iter)
    
    def Set_Iter(self, iter: int=-1, resetAll=False) -> dict:
        
        results = super().Set_Iter(iter)

        if results is None: return

        problemType = self.problemType

        self._Set_u_n(self.problemType, results["displacement"])

        if self.algo == AlgoType.hyperbolic and "speed" in results and "accel" in results:
            self._Set_v_n(problemType, results["speed"])
            self._Set_a_n(problemType, results["accel"])
        else:
            initZeros = np.zeros_like(self.displacement)
            self._Set_v_n(problemType, initZeros)
            self._Set_a_n(problemType, initZeros)

        return results

    def Results_Available(self) -> list[str]:

        results = []
        dim = self.dim

        results.extend(["displacement", "displacement_norm", "displacement_matrix"])
        results.extend(["speed", "speed_norm"])
        results.extend(["accel", "accel_norm"])
        
        if dim == 2:
            results.extend(["ux", "uy"])
            results.extend(["vx", "vy"])
            results.extend(["ax", "ay"])
            results.extend(["Sxx", "Syy", "Sxy"])
            results.extend(["Exx", "Eyy", "Exy"])

        elif dim == 3:
            results.extend(["ux", "uy", "uz"])
            results.extend(["vx", "vy", "vz"])
            results.extend(["ax", "ay", "az"])
            results.extend(["Sxx", "Syy", "Szz", "Syz", "Sxz", "Sxy"])
            results.extend(["Exx", "Eyy", "Ezz", "Eyz", "Exz", "Exy"])

        results.extend(["Svm", "Stress","Evm", "Strain"])
        
        results.extend(["Wdef","Wdef_e","ZZ1","ZZ1_e"])

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
            
        elif result == "ZZ1":
            return self._Calc_ZZ1()[0]

        elif result == "ZZ1_e":
            values = self._Calc_ZZ1()[1]

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

        elif result in ["vx", "vy", "vz"]:
            values_n = self.speed.reshape(Nn, -1)
            values = values_n[:,self.__indexResult(result)]

        elif result == "speed":
            values = self.speed
        
        elif result == "speed_norm":
            val_n = self.speed.reshape(Nn, -1)
            values = np.linalg.norm(val_n, axis=1)

        elif result in ["ax", "ay", "az"]:
            values_n = self.accel.reshape(Nn, -1)
            values = values_n[:,self.__indexResult(result)]
        
        elif result == "accel":
            values = self.accel
        
        elif result == "accel_norm":
            val_n = self.accel.reshape(Nn, -1)
            values = np.linalg.norm(val_n, axis=1)
        
        elif ("S" in result or "E" in result) and (not "_norm" in result):
            # Strain and Stress calculation part

            coef = self.material.coef

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

    def _Calc_Psi_Elas(self, returnScalar=True, smoothedStress=False, matrixType=MatrixType.rigi) -> float:
        """Computes the kinematically admissible deformation energy.
        Wdef = 1/2 int_Ω Sig : Eps dΩ"""

        tic = Tic()
        
        sol_u  = self.displacement

        mesh = self.mesh
        
        Epsilon_e_pg = self._Calc_Epsilon_e_pg(sol_u, matrixType)
        jacobian_e_pg = mesh.Get_jacobian_e_pg(matrixType)
        weight_pg = mesh.Get_weight_pg(matrixType)
        N_pg = mesh.Get_N_pg(matrixType)

        if self.dim == 2:
            ep = self.material.thickness
        else:
            ep = 1

        Sigma_e_pg = self._Calc_Sigma_e_pg(Epsilon_e_pg, matrixType)

        if smoothedStress:
            Sigma_n = mesh.Get_Node_Values(np.mean(Sigma_e_pg, 1))

            Sigma_n_e = mesh.Locates_sol_e(Sigma_n)
            Sigma_e_pg = np.einsum('eni,pjn->epi',Sigma_n_e, N_pg)

        if returnScalar:

            Wdef = 1/2 * np.einsum(',ep,p,epi,epi->', ep, jacobian_e_pg, weight_pg, Sigma_e_pg, Epsilon_e_pg, optimize='optimal')
            Wdef = float(Wdef)

        else:

            Wdef = 1/2 * np.einsum(',ep,p,epi,epi->e', ep, jacobian_e_pg, weight_pg, Sigma_e_pg, Epsilon_e_pg, optimize='optimal')

        tic.Tac("PostProcessing","Calc Psi Elas",False)
        
        return Wdef
    
    def _Calc_ZZ1(self) -> tuple[float, np.ndarray]:
        """Computes the ZZ1 error.\n
        For more details, [F.Pled, Vers une stratégie robuste ... ingénierie mécanique] page 20/21\n
        Returns the global error and the error on each element.

        Returns
        -------
        error, error_e
        """

        W_e = self._Calc_Psi_Elas(False)
        Welas = np.sum(W_e)

        Ws_e: np.ndarray = self._Calc_Psi_Elas(False, True)
        Ws = np.sum(Ws_e)

        error_e: np.ndarray = np.abs(Ws_e-W_e).ravel()/Welas

        error: float = np.abs(Welas-Ws)/Welas

        return error, error_e

    def _Calc_Epsilon_e_pg(self, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes strain field from the displacement vector field.\n
        2D : [Exx Eyy sqrt(2)*Exy]\n
        3D : [Exx Eyy Ezz sqrt(2)*Eyz sqrt(2)*Exz sqrt(2)*Exy]

        Parameters
        ----------
        u : np.ndarray
            displacement vector (Ndof)

        Returns
        -------
        np.ndarray
            Computed strain field (Ne,pg,(3 or 6))
        """

        tic = Tic()        
        u_e = u[self.mesh.assembly_e]
        B_dep_e_pg = self.mesh.Get_B_e_pg(matrixType)
        Epsilon_e_pg: np.ndarray = np.einsum('epij,ej->epi', B_dep_e_pg, u_e, optimize='optimal')
        
        tic.Tac("Matrix", "Epsilon_e_pg", False)

        return Epsilon_e_pg
                    
    def _Calc_Sigma_e_pg(self, Epsilon_e_pg: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes stress field from strain field.\n
        2D : [Sxx Syy sqrt(2)*Sxy]\n
        3D : [Sxx Syy Szz sqrt(2)*Syz sqrt(2)*Sxz sqrt(2)*Sxy]

        Parameters
        ----------
        Epsilon_e_pg : np.ndarray
            Strain field (Ne,pg,(3 or 6))

        Returns
        -------
        np.ndarray
            Computed stress field (Ne,pg,(3 or 6))
        """
        Ne = Epsilon_e_pg.shape[0]
        nPg = Epsilon_e_pg.shape[1]

        assert Ne == self.mesh.Ne
        assert nPg == self.mesh.Get_nPg(matrixType)

        tic = Tic()

        c = self.material.C
        
        c_e_p = Reshape_variable(c, Ne, nPg)

        Sigma_e_pg = c_e_p @ Epsilon_e_pg[:,:,:,np.newaxis]
        Sigma_e_pg: np.ndarray = Sigma_e_pg.reshape((Ne,nPg,-1))
            
        tic.Tac("Matrix", "Sigma_e_pg", False)

        return Sigma_e_pg

    def __indexResult(self, result: str) -> int:
        
        if len(result) <= 2:
            "Case were ui, vi or ai"
            if "x" in result:
                return 0
            elif "y" in result:
                return 1
            elif "z" in result:
                return 2

    def Results_dict_Energy(self) -> dict[str, float]:
        dict_energy = {
            r"$\Psi_{elas}$": self._Calc_Psi_Elas()
            }
        return dict_energy

    def Results_Get_Iteration_Summary(self) -> str:        

        summary = ""

        if not self._Results_Check_Available("Wdef"):
            return
        
        Wdef = self.Result("Wdef")
        summary += f"\nW def = {Wdef:.2f}"
        
        Svm = self.Result("Svm", nodeValues=False)
        summary += f"\n\nSvm max = {Svm.max():.2f}"

        Evm = self.Result("Evm", nodeValues=False)
        summary += f"\n\nEvm max = {Evm.max()*100:3.2f} %"

        # Affichage des déplacements
        dx = self.Result("ux", nodeValues=True)
        summary += f"\n\nUx max = {dx.max():.2e}"
        summary += f"\nUx min = {dx.min():.2e}"

        dy = self.Result("uy", nodeValues=True)
        summary += f"\n\nUy max = {dy.max():.2e}"
        summary += f"\nUy min = {dy.min():.2e}"

        if self.dim == 3:
            dz = self.Result("uz", nodeValues=True)
            summary += f"\n\nUz max = {dz.max():.2e}"
            summary += f"\nUz min = {dz.min():.2e}"

        return summary

    def Results_Iter_Summary(self) -> tuple[list[int], list[tuple[str, np.ndarray]]]:
        return super().Results_Iter_Summary()

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
    
# ----------------------------------------------
# Other functions
# ----------------------------------------------
def Mesh_Optim_ZZ1(DoSimu: Callable[[str], ElasticSimu], folder: str, threshold: float=1e-2, iterMax: int=20, coef: float=1/2) -> ElasticSimu:
    """Optimizes the mesh using ZZ1 error criterion.

    Parameters
    ----------
    DoSimu : Callable[[str], Displacement]
        Function that runs a simulation and takes a *.pos file as argument for mesh optimization. The function must return a Displacement simulation.
    folder : str
        Folder in which .pos files are created and then deleted.
    threshold : float, optional
        targeted error, by default 1e-2
    iterMax : int, optional
        Maximum number of iterations, by default 20
    coef : float, optional
        mesh size division ratio, by default 1/2

    Returns
    -------
    Displacement
        Displacement simulation
    """

    i = -1
    error = 1
    optimGeom = None
    # max=1
    while error >= threshold and i <= iterMax:

        i += 1

        # perform the simulation
        simu = DoSimu(optimGeom)
        assert isinstance(simu, ElasticSimu), 'DoSimu function must return a Displacement simulation'
        # get the current mesh
        mesh = simu.mesh

        if i > 0:
            # remove previous *.pos file
            Folder.os.remove(optimGeom)
        
        # Calculate the error with the ZZ1 method
        error, error_e = simu._Calc_ZZ1()

        print(f'error = {error*100:.3f} %')

        # calculate the new mesh size for the associated error
        meshSize_n = mesh.Get_New_meshSize_n(error_e, coef)

        # build the *.pos file that will be used to refine the mesh
        optimGeom = Mesher().Create_posFile(mesh.coord, meshSize_n, folder, f"pos{i}")
        
    if Folder.Exists(optimGeom):
        # remove last *.pos file
        Folder.os.remove(optimGeom)

    return simu