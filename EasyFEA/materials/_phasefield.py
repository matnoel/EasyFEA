# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from typing import Union
from enum import Enum

# utilities
import numpy as np
from ..utilities import Numba, Tic
# fem
from ..fem import Mesh, FeArray
# others
from ._utils import _IModel, ModelType, Reshape_variable, Project_vector_to_matrix, Project_matrix_to_vector
from ..utilities import _params
from ..utilities._linalg import Trace, TensorProd, Det, Inv, Norm

# ----------------------------------------------
# Elasticity
# ----------------------------------------------

from ._linear_elastic_laws import _Elas, Elas_Isot, Elas_IsotTrans, Elas_Anisot

# ----------------------------------------------
# Phase field
# ----------------------------------------------

class PhaseField(_IModel):
    """PhaseField class."""

    class ReguType(str, Enum):
        """Regularization models."""
        AT1 = "AT1"
        AT2 = "AT2"

        def __str__(self) -> str:
            return self.name

    class SplitType(str, Enum):
        """Split models."""

        # Isotropic
        Bourdin = "Bourdin" # [Bourdin 2000] DOI : 10.1016/S0022-5096(99)00028-9
        Amor = "Amor" # [Amor 2009] DOI : 10.1016/j.jmps.2009.04.011
        Miehe = "Miehe" # [Miehe 2010] DOI : 10.1016/j.cma.2010.04.011

        # Anisotropic
        He = "He" # [He Shao 2019] DOI : 10.1115/1.4042217
        Stress = "Stress" # Miehe in stress
        Zhang = "Zhang" # [Zhang 2020] DOI : 10.1016/j.cma.2019.112643

        # spectral decomposition in strain
        AnisotStrain = "AnisotStrain"
        AnisotStrain_PM = "AnisotStrain_PM"
        AnisotStrain_MP = "AnisotStrain_MP"
        AnisotStrain_NoCross = "AnisotStrain_NoCross"

        # spectral decomposition in stress
        AnisotStress = "AnisotStress"
        AnisotStress_PM = "AnisotStress_PM"
        AnisotStress_MP = "AnisotStress_MP"
        AnisotStress_NoCross = "AnisotStress_NoCross"

        def __str__(self) -> str:
            return self.name

    __SPLITS_ISOT = [SplitType.Amor, SplitType.Miehe, SplitType.Stress]
    __SPLITS_ANISOT = [SplitType.Bourdin, SplitType.He, SplitType.Zhang,
                    SplitType.AnisotStrain, SplitType.AnisotStrain_PM, SplitType.AnisotStrain_MP, SplitType.AnisotStrain_NoCross,
                    SplitType.AnisotStress, SplitType.AnisotStress_PM, SplitType.AnisotStress_MP, SplitType.AnisotStress_NoCross]
    
    class SolverType(str, Enum):
        """Solver used to manage crack irreversibility."""
        History = "History"
        HistoryDamage = "HistoryDamage"
        BoundConstrain = "BoundConstrain"

        def __str__(self) -> str:
            return self.name

    def __init__(self, material: _Elas, split: SplitType, regularization: ReguType, Gc: Union[float,np.ndarray], l0: float, solver=SolverType.History, A=None):
        """Creates a phase-field model.

        Parameters
        ----------
        material : _Elas
            Elastic material (Elas_Isot, Elas_IsotTrans, Elas_Anisot)
        split : SplitType
            split used to decompose the elastic energy density (see PhaseField_Model.get_splits())
        regularization : RegularizationType
            AT1 or AT2 crack regularization model
        Gc : float | np.ndarray
            critical energy release rate in J.m^-2
        l0 : float | np.ndarray
            half crack width
        solver : SolverType, optional
            solver used to manage crack irreversibility, by default History (see SolverType)        
        A : np.ndarray, optional
            matrix characterizing the weak anisotropy in the crack surface density function.
        """

        assert isinstance(material, _Elas), "Must be a displacement model (Elas_Isot, Elas_IsotTrans, Elas_Anisot)"
        # Material object cannot be changed by another _Elas object
        self.__material = material

        self.split = split
        
        self.regularization = regularization
        
        self.Gc = Gc

        self.l0 = l0        

        self.solver = solver

        self.A = A

        self.useNumba = False

    @property
    def modelType(self) -> ModelType:
        return ModelType.damage

    @property
    def dim(self) -> int:
        return self.__material.dim

    @property
    def thickness(self) -> float:
        return self.__material.thickness
    
    def __str__(self) -> str:
        text = str(self.__material)
        text += f'\n\n{type(self).__name__} :'
        text += f'\nsplit : {self.__split}'
        text += f'\nregularization : {self.__regularization}'
        text += f'\nGc : {self.__Gc:.4e}'
        text += f'\nl0 : {self.__l0:.4e}'
        return text
    
    @property
    def isHeterogeneous(self) -> bool:
        return isinstance(self.__Gc, np.ndarray)
    
    @staticmethod
    def Get_splits() -> list[SplitType]:
        """Returns splits available"""
        return list(PhaseField.SplitType)    
    
    @staticmethod
    def Get_regularisations() -> list[ReguType]:
        """Returns regularizations available"""
        __regularizations = list(PhaseField.ReguType)
        return __regularizations    

    @staticmethod
    def Get_solvers() -> list[SolverType]:
        """Returns available solvers used to manage crack irreversibility"""
        return list(PhaseField.SolverType)

    @property
    def k(self) -> float:
        """get diffusion therm"""

        Gc = self.__Gc
        l0 = self.__l0
        
        # J/m
        if self.__regularization == self.ReguType.AT1:
            k = 3/4 * Gc * l0 
        elif self.__regularization == self.ReguType.AT2:
            k = Gc * l0        

        return k

    def Get_r_e_pg(self, PsiP_e_pg: np.ndarray) -> FeArray:
        """Returns reaction therm"""

        Gc = self.Gc
        if self.isHeterogeneous:
            Gc = Reshape_variable(Gc, *PsiP_e_pg.shape[:2])
        else:
            FeArray.asfearray(Gc, True)
        l0 = self.__l0

        # J/m3
        if self.__regularization == self.ReguType.AT1:
            r = 2 * PsiP_e_pg
        elif self.__regularization == self.ReguType.AT2:
            r = 2 * PsiP_e_pg + (Gc/l0)
        
        return r

    def Get_f_e_pg(self, PsiP_e_pg: np.ndarray) -> FeArray:
        """Returns source therm"""

        Gc = self.Gc
        if self.isHeterogeneous:
            Gc = Reshape_variable(Gc, *PsiP_e_pg.shape[:2])
        else:
            Gc = FeArray.asfearray(Gc, True)
        l0 = self.__l0
        
        # J/m3
        if self.__regularization == self.ReguType.AT1:
            f = 2 * PsiP_e_pg - ( (3*Gc) / (8*l0) )            
            absF = np.abs(f)
            f = (f+absF)/2
        elif self.__regularization == self.ReguType.AT2:
            f = 2 * PsiP_e_pg
        
        return f

    def Get_g_e_pg(self, d_n: np.ndarray, mesh: Mesh, matrixType: str, k_res=1e-12) -> FeArray:
        """Returns degradation function"""

        d_e_n = mesh.Locates_sol_e(d_n, asFeArray=True)
        Nd_pg = FeArray.asfearray(mesh.Get_N_pg(matrixType)[np.newaxis,:,0])

        d_e_pg = Nd_pg @ d_e_n

        if self.__regularization in self.Get_regularisations():
            g_e_pg: np.ndarray = (1-d_e_pg)**2 + k_res
        else:
            raise Exception("Not implemented.")

        assert mesh.Ne == g_e_pg.shape[0]
        assert mesh.Get_nPg(matrixType) == g_e_pg.shape[1]
        
        return FeArray.asfearray(g_e_pg)
    
    @property
    def A(self) -> np.ndarray:
        """matrix characterizing the weak anisotropy in the crack surface density function"""
        return self.__A.copy()
    
    @A.setter
    def A(self, array: np.ndarray) -> None:
        dim = self.dim
        if not isinstance(array, np.ndarray):
            array = np.eye(dim)
        shape = (dim, dim)
        assert array.shape[-2:] == shape, f"Must an array of dimension {shape}"
        self.Need_Update()
        self.__A = array

    @property
    def split(self) -> str:
        """split used to decompose the elastic energy density"""
        return self.__split
    
    @split.setter
    def split(self, value: str) -> None:
        splits = self.Get_splits()
        assert value in splits, f"Must be included in {splits}"
        if not isinstance(self.material, Elas_Isot):
            # check that if the material is not a isotropic material you cant pick a isotoprpic split
            assert not value in PhaseField.__SPLITS_ISOT, "These splits are only implemented for Elas_Isot material"
        self.Need_Update()
        self.__split =  value

    @property
    def regularization(self) -> str:
        """crack regularization model"""
        return self.__regularization
    
    @regularization.setter
    def regularization(self, value: str) -> None:
        types = self.Get_regularisations()
        assert value in types, f"Must be included in {types}"
        self.Need_Update()
        self.__regularization = value
    
    @property
    def material(self) -> _Elas:
        """elastic material"""
        return self.__material

    @property
    def solver(self):
        """solver used to manage crack irreversibility"""
        return self.__solver
    
    @solver.setter
    def solver(self, value: str):        
        solvers = self.Get_solvers()
        assert value in solvers, f"Must be included in {solvers}"
        self.Need_Update()
        self.__solver = value

    @property
    def Gc(self) -> Union[float, np.ndarray]:
        """critical energy release rate (e.g. J/m^2)"""
        if isinstance(self.__Gc, np.ndarray):
            return self.__Gc.copy()
        else:
            return self.__Gc
    
    @Gc.setter
    def Gc(self, value: Union[float, np.ndarray]):
        _params.CheckIsPositive(value)
        self.Need_Update()
        self.__Gc = value

    @property
    def l0(self) -> float:
        """half crack width"""
        return self.__l0
    
    @l0.setter
    def l0(self, value: float):
        _params.CheckIsPositive(value)
        assert isinstance(value, (int, float)), 'l0 must be a homogeneous parameter'
        self.Need_Update()
        self.__l0 = value

    @property
    def c_w(self):
        """scaling parameter for accurate dissipation of crack energy"""
        if self.__regularization == self.ReguType.AT1:
            c_w = 8/3
        elif self.__regularization == self.ReguType.AT2:
            c_w = 2
        return c_w
            
    def Calc_psi_e_pg(self, Epsilon_e_pg: np.ndarray) -> tuple[FeArray, FeArray]:
        """Computes the elastic energy densities.\n

        psiP_e_pg = 1/2 SigmaP_e_pg * Epsilon_e_pg\n
        psiM_e_pg = 1/2 SigmaM_e_pg * Epsilon_e_pg\n
        Such as :\n
        SigmaP_e_pg = cP_e_pg * Epsilon_e_pg\n
        SigmaM_e_pg = cM_e_pg * Epsilon_e_pg       
        """

        Epsilon_e_pg = FeArray.asfearray(Epsilon_e_pg)

        SigmaP_e_pg, SigmaM_e_pg = self.Calc_Sigma_e_pg(Epsilon_e_pg)

        tic = Tic()

        psiP_e_pg = np.sum(1/2 * Epsilon_e_pg * SigmaP_e_pg, -1)
        psiM_e_pg = np.sum(1/2 * Epsilon_e_pg * SigmaM_e_pg, -1)

        tic.Tac("Matrix", "psiP_e_pg and psiM_e_pg", False)

        return psiP_e_pg, psiM_e_pg

    def Calc_Sigma_e_pg(self, Epsilon_e_pg: np.ndarray) -> tuple[FeArray, FeArray]:
        """Computes the Stress field using the strains and the split such that:\n
        
        SigmaP_e_pg = cP_e_pg * Epsilon_e_pg\n
        SigmaM_e_pg = cM_e_pg * Epsilon_e_pg

        Parameters
        ----------
        Epsilon_e_pg : np.ndarray
            strains field (e, p, D)

        Returns
        -------
        FeArray
            SigmaP_e_pg, SigmaM_e_pg: positive and negative stress fields (e, p, D)
        """       

        Epsilon_e_pg = FeArray.asfearray(Epsilon_e_pg)

        Ne, nPg, dim = Epsilon_e_pg.shape[:3]
        
        cP_e_pg, cM_e_pg = self.Calc_C(Epsilon_e_pg)

        tic = Tic()
        
        Epsilon_e_pg = Epsilon_e_pg.reshape((Ne,nPg,dim,1))

        SigmaP_e_pg = np.reshape(cP_e_pg @ Epsilon_e_pg, (Ne,nPg,-1))
        SigmaM_e_pg = np.reshape(cM_e_pg @ Epsilon_e_pg, (Ne,nPg,-1))

        tic.Tac("Matrix", "SigmaP_e_pg and SigmaM_e_pg", False)

        return SigmaP_e_pg, SigmaM_e_pg
    
    def Calc_C(self, Epsilon_e_pg: FeArray, verif=False) -> tuple[FeArray, FeArray]:
        """Computes the splited stifness matrices for the given strain field.

        Parameters
        ----------
        Epsilon_e_pg : FeArray
            strains field (e, p, D)

        Returns
        -------
        FeArray
            cP_e_pg, cM_e_pg: positive and negative stifness matrices (e, p, D, D)
        """

        Ne, nPg = Epsilon_e_pg.shape[:2]

        if self.__split == self.SplitType.Bourdin:
            cP_e_pg, cM_e_pg = self.__Split_Bourdin(Ne, nPg)

        elif self.__split == self.SplitType.Amor:
            cP_e_pg, cM_e_pg = self.__Split_Amor(Epsilon_e_pg)

        elif self.__split == self.SplitType.Miehe or "Strain" in self.__split:
            cP_e_pg, cM_e_pg = self.__Split_Strain(Epsilon_e_pg, verif=verif)
        
        elif self.__split == self.SplitType.Zhang or "Stress" in self.__split:
            cP_e_pg, cM_e_pg = self.__Split_Stress(Epsilon_e_pg, verif=verif)

        elif self.__split == self.SplitType.He:
            cP_e_pg, cM_e_pg = self.__Split_He(Epsilon_e_pg, verif=verif)

        return cP_e_pg, cM_e_pg

    def __Split_Bourdin(self, Ne: int, nPg: int):
        """[Bourdin 2000] DOI : 10.1016/S0022-5096(99)00028-9"""

        tic = Tic()

        C = self.__material.C
        if self.isHeterogeneous:
            C_e_pg = Reshape_variable(C, Ne, nPg)
        else:
            C_e_pg = FeArray.asfearray(C, True)

        cP_e_pg = C_e_pg
        cM_e_pg = np.zeros_like(cP_e_pg)

        tic.Tac("Split",f"cP_e_pg and cM_e_pg", False)

        return cP_e_pg, cM_e_pg

    def __Split_Amor(self, Epsilon_e_pg: np.ndarray):
        """[Amor 2009] DOI : 10.1016/j.jmps.2009.04.011"""

        assert isinstance(self.__material, Elas_Isot), f"Implemented only for Elas_Isot material."
        
        tic = Tic()
        
        material = self.__material
        

        Rp_e_pg, Rm_e_pg = self.__Rp_Rm(Epsilon_e_pg)

        dim = material.dim

        if dim == 2:
            I = np.array([1,1,0]).reshape((3,1))
            size = 3
        else:
            I = np.array([1,1,1,0,0,0]).reshape((6,1))
            size = 6

        IxI = I @ I.T

        mu = material.get_mu()
        bulk = material.get_bulk()

        if material.isHeterogeneous:        
            Ne, nPg = Epsilon_e_pg.shape[:2]
            mu = Reshape_variable(mu, Ne, nPg)
            bulk = Reshape_variable(bulk, Ne, nPg)

        cP_e_pg = bulk * (Rp_e_pg * IxI) + 2*mu * (np.eye(size) - 1/dim * IxI)
        cM_e_pg = bulk * (Rm_e_pg * IxI)

        tic.Tac("Split",f"cP_e_pg and cM_e_pg", False)

        return cP_e_pg, cM_e_pg

    def __Rp_Rm(self, vector_e_pg: FeArray):
        """Returns Rp_e_pg, Rm_e_pg"""

        Ne, nPg = vector_e_pg.shape[:2]

        dim = self.__material.dim

        trace = np.zeros((Ne, nPg))

        trace = vector_e_pg[:,:,0] + vector_e_pg[:,:,1]

        if dim == 3:
            trace += vector_e_pg[:,:,2]

        if not isinstance(trace, FeArray):
            trace = FeArray.asfearray(trace)

        Rp_e_pg = (1+np.sign(trace))/2
        Rm_e_pg = (1+np.sign(-trace))/2

        return Rp_e_pg, Rm_e_pg
    
    def __Split_Strain(self, Epsilon_e_pg: FeArray, verif=False):
        """Computes the stifness matrices for strain based splits."""

        material = self.__material
        dim = material.dim

        projP_e_pg, projM_e_pg = self.__Spectral_Decomposition(Epsilon_e_pg, verif)

        tic = Tic()

        if self.__split == self.SplitType.Miehe:
            # [Miehe 2010] DOI : 10.1016/j.cma.2010.04.011
            
            assert isinstance(self.__material, Elas_Isot), f"Implemented only for Elas_Isot material"

            # Compute Rp and Rm
            Rp_e_pg, Rm_e_pg = self.__Rp_Rm(Epsilon_e_pg)
            
            # Compute IxI
            if dim == 2:
                I = np.array([1,1,0]).reshape((3,1))
            elif dim == 3:
                I = np.array([1,1,1,0,0,0]).reshape((6,1))
            IxI = I @ I.T

            # Compute stifness matrices
            mu = self.__material.get_mu()
            lamb = self.__material.get_lambda()

            if material.isHeterogeneous:
                Ne, nPg = Epsilon_e_pg.shape[:2]
                mu = Reshape_variable(mu, Ne, nPg)                
                lamb = Reshape_variable(lamb, Ne, nPg)

            cP_e_pg = lamb * (Rp_e_pg * IxI) + 2 * mu * projP_e_pg
            cM_e_pg = lamb * (Rm_e_pg * IxI) + 2 * mu * projM_e_pg
        
        elif "Strain" in self.__split:

            # here don't use numba if behavior is heterogeneous
            if self.useNumba and not self.isHeterogeneous:
                # Faster (x2) but not available for heterogeneous material (memory issues)
                Cpp, Cpm, Cmp, Cmm = Numba.Get_Anisot_C(projP_e_pg, material.C, projM_e_pg)
                Cpp, Cpm, Cmp, Cmm = FeArray._asfearrays(Cpp, Cpm, Cmp, Cmm)

            else:
                C_e_pg = FeArray.asfearray(material.C, True)

                projPTC = projP_e_pg.T @ C_e_pg
                projMTc = projM_e_pg.T @ C_e_pg
                
                Cpp = projPTC @ projP_e_pg
                Cpm = projPTC @ projM_e_pg
                Cmm = projMTc @ projM_e_pg
                Cmp = projMTc @ projP_e_pg
            
            if self.__split == self.SplitType.AnisotStrain:

                cP_e_pg = Cpp + Cpm + Cmp
                cM_e_pg = Cmm 

            elif self.__split == self.SplitType.AnisotStrain_PM:
                
                cP_e_pg = Cpp + Cpm
                cM_e_pg = Cmm + Cmp

            elif self.__split == self.SplitType.AnisotStrain_MP:
                
                cP_e_pg = Cpp + Cmp
                cM_e_pg = Cmm + Cpm

            elif self.__split == self.SplitType.AnisotStrain_NoCross:
                
                cP_e_pg = Cpp
                cM_e_pg = Cmm + Cpm + Cmp
            
        else:
            raise Exception("Unknown split.")

        tic.Tac("Split",f"cP_e_pg and cM_e_pg", False)

        return cP_e_pg, cM_e_pg
    
    def __Split_Stress(self, Epsilon_e_pg: np.ndarray, verif=False):
        """Computes the stifness matrices for stress based splits."""

        # Recover stresses        
        material = self.__material

        Ne, nPg = Epsilon_e_pg.shape[:2]

        C = material.C
        if self.isHeterogeneous:
            C_e_pg = Reshape_variable(C, Ne, nPg)
        else:
            C_e_pg = FeArray.asfearray(C, True)

        Sigma_e_pg = C_e_pg @ Epsilon_e_pg

        # Compute projectors such that SigmaP = Pp : Sigma and SigmaM = Pm : Sigma
        projP_e_pg, projM_e_pg = self.__Spectral_Decomposition(Sigma_e_pg, verif)

        tic = Tic()

        if self.__split == self.SplitType.Stress:
        
            assert isinstance(material, Elas_Isot)

            E = material.E
            v = material.v
            mu = material.get_mu()

            if material.isHeterogeneous:
                E = Reshape_variable(E, Ne, nPg)
                v = Reshape_variable(v, Ne, nPg)
                mu = Reshape_variable(mu, Ne, nPg)

            dim = self.dim

            # Compute Rp and Rm
            Rp_e_pg, Rm_e_pg = self.__Rp_Rm(Sigma_e_pg)
            
            # Compute IxI
            if dim == 2:
                I = np.array([1,1,0]).reshape((3,1))
            else:
                I = np.array([1,1,1,0,0,0]).reshape((6,1))
            IxI = I.dot(I.T)

            if dim == 2:
                if material.planeStress:
                    sP_e_pg = ((1+v)/E * projP_e_pg) - (v/E * Rp_e_pg * IxI)
                    sM_e_pg = ((1+v)/E * projM_e_pg) - (v/E * Rm_e_pg * IxI) 
                else:
                    sP_e_pg = ((1+v)/E * projP_e_pg) - (v*(1+v)/E * Rp_e_pg * IxI) 
                    sM_e_pg = ((1+v)/E * projM_e_pg) - (v*(1+v)/E * Rm_e_pg * IxI)

            elif dim == 3:
                sP_e_pg = (1/(2*mu) * projP_e_pg) - (v/E * Rp_e_pg * IxI)
                sM_e_pg = (1/(2*mu) * projM_e_pg) - (v/E * Rm_e_pg * IxI)

            if self.useNumba and not material.isHeterogeneous:
                # Faster
                cP_e_pg, cM_e_pg = Numba.Get_Cp_Cm_Stress(material.C, sP_e_pg, sM_e_pg)
                cP_e_pg, cM_e_pg = FeArray._asfearrays(cP_e_pg, cM_e_pg)
            else:
                cP_e_pg = C_e_pg.T @ sP_e_pg @ C_e_pg
                cM_e_pg = C_e_pg.T @ sM_e_pg @ C_e_pg
        
        elif self.__split == self.SplitType.Zhang or "Stress" in self.__split:
            
            Cp_e_pg = projP_e_pg @ C_e_pg
            Cm_e_pg = projM_e_pg @ C_e_pg

            if self.__split == self.SplitType.Zhang:
                # [Zhang 2020] DOI : 10.1016/j.cma.2019.112643
                cP_e_pg = Cp_e_pg
                cM_e_pg = Cm_e_pg
            
            else:
                # Compute Cp and Cm
                S = material.S

                if self.useNumba and not material.isHeterogeneous:
                    # Faster
                    Cpp, Cpm, Cmp, Cmm = Numba.Get_Anisot_C(Cp_e_pg, S, Cm_e_pg)
                    Cpp, Cpm, Cmp, Cmm = FeArray._asfearrays(Cpp, Cpm, Cmp, Cmm)
                else:
                    S_e_pg = Reshape_variable(S, Ne, nPg)

                    ps = Cp_e_pg.T @ S_e_pg
                    ms = Cm_e_pg.T @ S_e_pg
                    
                    Cpp = ps @ Cp_e_pg
                    Cpm = ps @ Cm_e_pg
                    Cmm = ms @ Cm_e_pg
                    Cmp = ms @ Cp_e_pg
                
                if self.__split == self.SplitType.AnisotStress:

                    cP_e_pg = Cpp + Cpm + Cmp
                    cM_e_pg = Cmm

                elif self.__split == self.SplitType.AnisotStress_PM:
                    
                    cP_e_pg = Cpp + Cpm
                    cM_e_pg = Cmm + Cmp

                elif self.__split == self.SplitType.AnisotStress_MP:
                    
                    cP_e_pg = Cpp + Cmp
                    cM_e_pg = Cmm + Cpm

                elif self.__split == self.SplitType.AnisotStress_NoCross:
                    
                    cP_e_pg = Cpp
                    cM_e_pg = Cmm + Cpm + Cmp
            
                else:
                    raise Exception("Unknown split.")

        tic.Tac("Split",f"cP_e_pg and cM_e_pg", False)

        return cP_e_pg, cM_e_pg

    def __Split_He(self, Epsilon_e_pg: np.ndarray, verif=False):
        """[He Shao 2019] DOI : 10.1115/1.4042217"""
            
        # Here the material is supposed to be homogeneous
        material = self.__material

        Ne, nPg = Epsilon_e_pg.shape[:2]

        C = material.C

        tic = Tic()
        sqrtC, inv_sqrtC = material.Get_sqrt_C_S()
        # inv(sqrtC) = sqrtS
        tic.Tac("Split",f"sqrt C and S", False)
        
        if material.isHeterogeneous:
            sqrtC = Reshape_variable(sqrtC, Ne, nPg)
            inv_sqrtC = Reshape_variable(inv_sqrtC, Ne, nPg)
        else:
            sqrtC = FeArray.asfearray(sqrtC, True)
            inv_sqrtC = FeArray.asfearray(inv_sqrtC, True)
        
        if verif:
            # check that C^1/2 * C^1/2 = C
            diff_C = sqrtC @ sqrtC - C
            test_C = Norm(diff_C, axis=(-2,-1))/Norm(C, axis=(-2,-1))
            assert np.max(test_C) < 1e-12

        # compute new "strain" field        
        Epsilont_e_pg = sqrtC @ Epsilon_e_pg

        # Compute projectors
        projPt_e_pg, projMt_e_pg = self.__Spectral_Decomposition(Epsilont_e_pg, verif)

        tic = Tic()

        projP_e_pg = inv_sqrtC @ projPt_e_pg @ sqrtC
        projM_e_pg = inv_sqrtC @ projMt_e_pg @ sqrtC

        tic.Tac("Split",f"proj Tild to proj", False)

        cP_e_pg = C @ projP_e_pg
        cM_e_pg = C @ projM_e_pg

        tic.Tac("Split",f"cP_e_pg and cM_e_pg", False)

        if verif:

            vector_e_pg = Epsilon_e_pg.copy()

            vectorP = projP_e_pg @ vector_e_pg
            vectorM = projM_e_pg @ vector_e_pg

            # Check orthogonality E+:C:E-
            mat = C.copy()
            ortho_vP_vM = np.abs(vectorP @ mat @ vectorM)
            ortho_vM_vP = np.abs(vectorM @ mat @ vectorP)
            ortho_v_v = np.abs(vector_e_pg @ mat @ vector_e_pg)
            if np.min(ortho_v_v) > 0:
                vertifOrthoEpsPM = np.max(ortho_vP_vM/ortho_v_v)
                assert vertifOrthoEpsPM < 1e-12
                vertifOrthoEpsMP = np.max(ortho_vM_vP/ortho_v_v)
                assert vertifOrthoEpsMP < 1e-12
            # Et+:Et- = 0 already checked in spectral decomposition

            # Rounding errors in the construction of 3D eigen projectors see [Remark M]
            tol = 1e-12 if self.dim == 2 else 1e-10
            
            # Check that vector_e_pg = vectorP_e_pg + vectorM_e_pg
            diff_vect = vector_e_pg - (vectorP + vectorM)
            if np.min(Norm(vector_e_pg, axis=-1)) > 0:
                test_vect = Norm(diff_vect, axis=-1) / Norm(vector_e_pg, axis=-1)
                assert np.max(test_vect) < tol, f"{np.max(test_vect):.3e}"

        return cP_e_pg, cM_e_pg

    def _Eigen_values_vectors_projectors(self, vector_e_pg: FeArray, verif=False) -> tuple[FeArray, list[FeArray], list[FeArray]]:
        """Computes the eigen values and eigen projectors of a second-order tensor (as a vector)."""

        dim = self.__material.dim

        coef = self.__material.coef
        Ne, nPg = vector_e_pg.shape[:2]

        tic = Tic()

        # Initialize the second-order tensor [e,pg,dim,dim]
        matrix_e_pg = Project_vector_to_matrix(vector_e_pg, coef)

        tic.Tac("Split", "vector_e_pg -> matrix_e_pg", False)

        normalize = lambda M: M / Norm(M, axis=(-2,-1))

        if self.dim == 2:
            # invariants of the strain tensor [e,pg]
            det_e_pg = Det(matrix_e_pg)

            tr_e_pg = Trace(matrix_e_pg)

            # Eigenvalue calculations [e,pg]
            delta = tr_e_pg**2 - (4*det_e_pg)
                        
            eigs_e_pg = FeArray.zeros(Ne, nPg, 2)
            eigs_e_pg[:,:,0] = (tr_e_pg - np.sqrt(delta))/2
            eigs_e_pg[:,:,1] = (tr_e_pg + np.sqrt(delta))/2

            tic.Tac("Split", "Eigenvalues", False)
            
            # m1 = (matrice_e_pg - v2*I)/(v1-v2)
            v1_m_v2 = eigs_e_pg[:,:,0] - eigs_e_pg[:,:,1]
            
            # element identification and gauss points where vp1 != vp2
            elems, pdgs = np.where(eigs_e_pg[:,:,0] != eigs_e_pg[:,:,1])
            
            # m1 and m2 [e,pg,dim,dim]
            M1 = FeArray.zeros(Ne,nPg,2,2)
            M1[:,:,0,0] = 1
            if elems.size > 0:
                v1_m_v2[v1_m_v2==0] = 1 # to avoid dividing by 0
                m1_tot = (matrix_e_pg - eigs_e_pg[:,:,1] * np.eye(2) ) / v1_m_v2
                M1[elems, pdgs] = m1_tot[elems, pdgs]
            M2 = np.eye(2) - M1

            tic.Tac("Split", "Eigenprojectors", False)
        
        elif self.dim == 3:

            version = 'invariants' # 'invariants', 'eigh'

            if version == 'eigh':

                valnum, vectnum = np.linalg.eigh(matrix_e_pg)

                tic.Tac("Split", "np.linalg.eigh", False)

                func_Mi = lambda mi: TensorProd(mi, mi, ndim=1)

                M1 = func_Mi(vectnum[:,:,:,0])
                M2 = func_Mi(vectnum[:,:,:,1])
                M3 = func_Mi(vectnum[:,:,:,2])
                
                eigs_e_pg = valnum

                tic.Tac("Split", "Eigenvalues and eigenprojectors", False)

            elif version == 'invariants':

                # [Q.-C. He Closed-form coordinate-free]

                det_e_pg = Det(matrix_e_pg)
                # det_e_pg = np.linalg.det(matrix_e_pg)
                # test_det = det_e_pg - np.linalg.det(matrix_e_pg) 

                # Invariants
                I1_e_pg = Trace(matrix_e_pg)
                trace_mat_mat = Trace(matrix_e_pg @ matrix_e_pg)
                I2_e_pg = 1/2 * (I1_e_pg**2 - trace_mat_mat)
                I3_e_pg = det_e_pg

                tic.Tac("Split", "Invariants", False)

                g_e_pg = I1_e_pg**2 - 3*I2_e_pg
                
                g_neq_0 = g_e_pg != 0
                # g_neq_0 = (g_e_pg >= tol0) & (g_e_pg <= -tol0)
                # g_neq_0 = np.logical_not(np.isclose(g_e_pg, 0, atol=tol0))
                
                if False in g_neq_0:
                    arg = 1/2 * (2*I1_e_pg**3 - 9*I1_e_pg*I2_e_pg + 27*I3_e_pg) # -1 <= arg <= 1
                    arg[g_neq_0] =  arg[g_neq_0] / g_e_pg[g_neq_0]**(3/2)
                else:
                    arg = (2*I1_e_pg**3 - 9*I1_e_pg*I2_e_pg + 27*I3_e_pg) / (2 * g_e_pg**(3/2))

                theta = 1/3 * np.arccos(arg) # Lode's angle such that 0 <= theta <= pi/3

                # -------------------------------------
                # Init eigenvalues an eigenprojectors for case 4
                # 𝜖1 = 𝜖2 = 𝜖3 ⇐⇒ 𝑔 = 0.
                # -------------------------------------
                val1_e_pg: np.ndarray = I1_e_pg/3
                val2_e_pg: np.ndarray = I1_e_pg/3
                val3_e_pg: np.ndarray = I1_e_pg/3

                # Init proj matrices
                M1 = np.zeros_like(matrix_e_pg); M1[:,:,0,0] = 1
                # M2 = np.zeros_like(matrix_e_pg); M2[:,:,1,1] = 1
                M3 = np.zeros_like(matrix_e_pg); M3[:,:,2,2] = 1

                tic.Tac("Split", "proj case 4", False)

                I_e_pg = np.zeros_like(matrix_e_pg)
                I_e_pg[:,:,0,0] = 1; I_e_pg[:,:,1,1] = 1; I_e_pg[:,:,2,2] = 1

                I_rg = 1/3 * ((I1_e_pg - g_e_pg**(1/2)) * np.eye(3))

                # -------------------------------------
                # 2. Two maximum eigenvalues
                # 𝜖1 < 𝜖2 = 𝜖3 ⇐⇒ 𝑔 ≠ 0, 𝜃 = 𝜋∕3.
                # arg = -1
                # -------------------------------------              

                test2 = g_neq_0 & (theta == np.pi/3)

                case2 = list(set(np.ravel(np.where(test2)[0])))

                if len(case2) > 0:

                    val1_e_pg[case2,:] += - 2/3 * g_e_pg[case2,:]**(1/2)
                    val2_e_pg[case2,:] += 1/3 * g_e_pg[case2,:]**(1/2) 
                    val3_e_pg[case2,:] += 1/3 * g_e_pg[case2,:]**(1/2)

                    M1[case2,:] = g_e_pg[case2,:]**(-1/2) * (I_rg - matrix_e_pg)[case2,:]
                    # M2[case2,:] = 1/2 * (I_e_pg - M1)[case2,:]
                    M3[case2,:] = 1/2 * (I_e_pg - M1)[case2,:]

                    tic.Tac("Split", "proj case 2", False)

                # -------------------------------------
                # 3. Two minimum eigenvalues
                # 𝜖1 = 𝜖2 < 𝜖3 ⇐⇒ 𝑔 ≠ 0, 𝜃 = 0.
                # arg = 1
                # -------------------------------------
                
                test3 = g_neq_0 & (theta == 0)
                
                case3 = list(set(np.ravel(np.where(test3)[0])))

                if len(case3) > 0:

                    val1_e_pg[case3,:] += - 1/3 * g_e_pg[case3,:]**(1/2)
                    val2_e_pg[case3,:] += - 1/3 * g_e_pg[case3,:]**(1/2) 
                    val3_e_pg[case3,:] += 2/3 * g_e_pg[case3,:]**(1/2)

                    M3[case3,:] = g_e_pg[case3,:]**(-1/2) * (matrix_e_pg - I_rg)[case3,:]
                    # M1[case3,:] = 1/2 * (I_e_pg - M3)[case3,:]
                    M2[case3,:] = 1/2 * (I_e_pg - M3)[case3,:]

                    tic.Tac("Split", "proj case 3", False)

                # -------------------------------------
                # 1. Three distinct eigenvalues
                # 𝜖1 < 𝜖2 < 𝜖3 ⇐⇒ 𝑔 ≠ 0, 𝜃 ≠ 0, 𝜃 ≠ 𝜋∕3.
                # -------------------------------------
                
                test1 = g_neq_0 & (theta != 0) & (theta != np.pi/3)

                case1 = list(set(np.ravel(np.where(test1)[0])))

                case1 = np.setdiff1d(case1, np.union1d(case2, case3))

                if len(case1) > 0:

                    val1_e_pg[case1,:] += (2/3 * g_e_pg**(1/2) * np.cos(2*np.pi/3 + theta))[case1,:]
                    val2_e_pg[case1,:] += (2/3 * g_e_pg**(1/2) * np.cos(2*np.pi/3 - theta))[case1,:]
                    val3_e_pg[case1,:] += (2/3 * g_e_pg**(1/2) * np.cos(theta))[case1,:]

                    e1_I = val1_e_pg * np.eye(3)
                    e2_I = val2_e_pg * np.eye(3)
                    e3_I = val3_e_pg * np.eye(3)

                    M1[case1,:] = ((matrix_e_pg - e2_I) @ (matrix_e_pg - e3_I)) / ((val1_e_pg - val2_e_pg) * (val1_e_pg - val3_e_pg))
                    # M2[case1,:] = (matrix_e_pg - e1_I) @ (matrix_e_pg - e3_I) / ((val2_e_pg - val1_e_pg) * (val2_e_pg - val3_e_pg))
                    M3[case1,:] = (matrix_e_pg - e1_I) @ (matrix_e_pg - e2_I) / ((val3_e_pg - val1_e_pg) * (val3_e_pg - val2_e_pg))

                    tic.Tac("Split", "proj case 1", False)

                # -------------------------------------
                # merge values in eigs_e_pg
                # -------------------------------------
                eigs_e_pg = FeArray.zeros(Ne, nPg, 3)
                eigs_e_pg[:,:,0] = val1_e_pg
                eigs_e_pg[:,:,1] = val2_e_pg
                eigs_e_pg[:,:,2] = val3_e_pg                

                M1 = normalize(M1)
                # M2 = normalize(M2)
                M3 = normalize(M3)

                M2 = FeArray.asfearray(I_e_pg - (M1 + M3))

                # M1 = I_e_pg - (M2 + M3)

                tic.Tac("Split", "Eigenprojectors", False)

        # transform eigenbases in the form of a vector [e,pg,3] or [e,pg,6].
        if dim == 2:
            # [x, y, xy]
            m1 = Project_matrix_to_vector(M1)
            m2 = Project_matrix_to_vector(M2)

            list_m = [m1, m2]
            list_M = [M1, M2]

        elif dim == 3:
            # [x, y, z, yz, xz, xy]
            m1 = Project_matrix_to_vector(M1)
            m2 = Project_matrix_to_vector(M2)
            m3 = Project_matrix_to_vector(M3)

            list_m = [m1, m2, m3]
            list_M = [M1, M2, M3]

        tic.Tac("Split", "Eigenvectors", False)        
        
        if verif:
            
            valnum, vectnum = np.linalg.eigh(matrix_e_pg)
            valnum, vectnum = FeArray._asfearrays(valnum, vectnum)

            func_Mi = lambda mi: TensorProd(mi, mi, ndim=1)

            M1_num = func_Mi(vectnum[:,:,:,0]); M1_num = normalize(M1_num)
            M2_num = func_Mi(vectnum[:,:,:,1]); M2_num = normalize(M2_num)

            matrix = eigs_e_pg[:,:,0] * M1 + eigs_e_pg[:,:,1] * M2
            matrix_eig = valnum[:,:,0] * M1_num + valnum[:,:,1] * M2_num
            
            if dim == 3:
                M3_num = func_Mi(vectnum[:,:,:,2]); M3_num = normalize(M3_num)
                matrix += eigs_e_pg[:,:,2] * M3
                matrix_eig += valnum[:,:,2] * M3_num

            # check if the eigen values are correct
            if valnum.max() > 0:
                diff_val = eigs_e_pg - valnum                    
                test_val = Norm(diff_val, axis=-1)/Norm(valnum, axis=-1)
                assert np.max(test_val) < 1e-12, f"Error in eigenvalues ({np.max(test_val):.3e})."

            # Check that: E1*M1 + E2*M2 + E3*M3
            if matrix_e_pg.max() > 0:
                # matrix
                diff_matrix = matrix - matrix_e_pg
                test_diff = Norm(diff_matrix, axis=(-2,-1))/Norm(matrix_e_pg, axis=(-2,-1))
                assert np.max(test_diff) < 1e-12, f"matrix != matrix_e_pg -> {np.max(test_diff):.3e}"                
                # matrix_eig
                diff_matrix_eig = matrix_eig - matrix_e_pg
                test_diff_eig = Norm(diff_matrix_eig, axis=(-2,-1))/Norm(matrix_e_pg, axis=(-2,-1))
                assert np.max(test_diff_eig) < 1e-12, f"matrix != matrix_e_pg -> {np.max(test_diff_eig):.3e}"

            if np.max(matrix) > 0:
                test_eig = Norm(matrix_eig - matrix, axis=(-2,-1))/Norm(matrix, axis=(-2,-1))
                assert np.max(test_eig) < 1e-12, f"matrix_eig != matrix -> {np.max(test_eig):.3e}"

            # [Remark M]
            # Rounding errors in the construction of 3D eigen projectors.
            # The identification of eigenvalues works, but there are errors for the eigenprojectors.
            # The problem is that we can't find the same eigen projectors as np.linal.eigh,
            # there must be rounding errors for eigen projectors
            # only occurs in 3D !!!
            tol = 1e-12 if dim == 2 else 1e-10
            
            def Checks_Ma(Ma, Mb, tol=tol):
                diff_M = Ma - Mb
                test_M = Norm(diff_M, axis=(-2,-1))/Norm(Ma, axis=(-2,-1))
                assert np.max(test_M) < tol, f"Error in eigenprojectors ({np.max(test_M):.3e})."

            Checks_Ma(M1, M1_num)
            Checks_Ma(M2, M2_num)
            if dim == 3:
                Checks_Ma(M3, M3_num, 1e-12)

            # check orthogonality between M1 and M2
            test_M1_M2 = np.abs(M1.ddot(M2))
            assert np.max(test_M1_M2) < tol, f"Orthogonality M1 : M2 not verified -> {np.max(test_M1_M2):.3e}"

            if dim == 3:
                # check orthogonality between M1 and M3
                test_M1_M3 = np.abs(M1.ddot(M3))
                assert np.max(test_M1_M3) < 1e-12, f"Orthogonality M1 : M3 not verified -> {np.max(test_M1_M3):.3e}"
                # check orthogonality between M2 and M3
                test_M2_M3 = np.abs(M2.ddot(M3))
                assert np.max(test_M2_M3) < 1e-12, f"Orthogonality M2 : M3 not verified -> {np.max(test_M2_M3):.3e}"

        return eigs_e_pg, list_m, list_M
    
    def __Spectral_Decomposition(self, vector_e_pg: FeArray, verif=False) -> tuple[FeArray, FeArray]:
        """Computes spectral projectors projP and projM such that:\n

        In 2D:
        ------

        vector_e_pg = [1 1 sqrt(2)] \n
        
        vectorP = projP • vector -> [1, 1, sqrt(2)]\n
        vectorM = projM • vector -> [1, 1, sqrt(2)]\n

        In 3D:
        ------

        vector_e_pg = [1 1 1 sqrt(2) sqrt(2) sqrt(2)] \n
        
        vectorP = projP • vector -> [1 1 1 sqrt(2) sqrt(2) sqrt(2)]\n
        vectorM = projM • vector -> [1 1 1 sqrt(2) sqrt(2) sqrt(2)]\n

        returns projP, projM
        """

        useNumba = self.useNumba        

        dim = self.__material.dim        

        Ne, nPg = vector_e_pg.shape[:2]
        
        # compute eigenvalues, eigenvectors and eigenprojectors
        val_e_pg, list_m, list_M = self._Eigen_values_vectors_projectors(vector_e_pg, verif)

        tic = Tic()
        
        # compute positive and negative parts of the eigenvalues [e,pg,2].
        valp = (val_e_pg+np.abs(val_e_pg))/2
        valm = (val_e_pg-np.abs(val_e_pg))/2
        
        # compute of di [e,pg,2].
        dvalp = np.heaviside(val_e_pg, 0.5)
        dvalm = np.heaviside(-val_e_pg, 0.5)

        if dim == 2:
            # eigenvectors
            m1, m2 = list_m[0], list_m[1]

            # elements and pdgs where eigenvalues 1 and 2 are different
            v1_m_v2 = val_e_pg[...,0] - val_e_pg[...,1] # val1 - val2
            v1_m_v2[v1_m_v2 == 0] = 1

            # compute BetaP [e,pg,1].
            # Caution: make sure you put a copy here, otherwise the Beta modification will change dvalp at the same time!
            BetaP = dvalp[...,0].copy()
            BetaP = (valp[...,0] - valp[...,1]) / v1_m_v2
            
            # compute BetaM [e,pg,1].
            BetaM = dvalm[...,0].copy()
            BetaM = (valm[...,0] - valm[...,1]) / v1_m_v2
            
            # compute gammap and gammam
            gammap = dvalp - BetaP
            gammam = dvalm - BetaM

            tic.Tac("Split", "Betas and gammas", False)

            # compute mixmi [e,pg,3,3] or [e,pg,6,6].
            m1xm1 = TensorProd(m1, m1, ndim=1)
            m2xm2 = TensorProd(m2, m2, ndim=1)

            if useNumba:
                # Faster
                projP, projM = Numba.Get_projP_projM_2D(BetaP, gammap, BetaM, gammam, m1, m2)
                projP, projM = FeArray._asfearrays(projP, projM)
            
            else:
                # Projector P such that EpsP = projP • Eps
                projP = (BetaP * np.eye(3)) + (gammap[...,0] * m1xm1) + (gammap[...,1] * m2xm2)

                # Projector M such that EpsM = projM • Eps
                projM = (BetaM * np.eye(3)) + (gammam[...,0] * m1xm1) + (gammam[...,1] * m2xm2)

            tic.Tac("Split", "projP and projM", False)

        elif dim == 3:
            m1, m2, m3 = list_m[0], list_m[1], list_m[2]

            M1, M2, M3 = list_M[0], list_M[1], list_M[2]            

            coef = np.sqrt(2)

            thetap = dvalp.copy()/2
            thetam = dvalm.copy()/2

            v1_m_v2 = val_e_pg[...,0] - val_e_pg[...,1]; v1_m_v2[v1_m_v2 == 0]=1
            thetap[...,0] = (valp[...,0] - valp[...,1]) / (2*v1_m_v2)
            thetam[...,0] = (valm[...,0] - valm[...,1]) / (2*v1_m_v2)

            v1_m_v3 = val_e_pg[...,0] - val_e_pg[...,2]; v1_m_v3[v1_m_v3 == 0]=1
            thetap[...,1] = (valp[...,0] - valp[...,2]) / (2*v1_m_v3)
            thetam[...,1] = (valm[...,0] - valm[...,2]) / (2*v1_m_v3)

            v2_m_v3 = val_e_pg[...,1] - val_e_pg[...,2]; v2_m_v3[v2_m_v3 == 0]=1
            thetap[...,2] = (valp[...,1] - valp[...,2]) / (2*v2_m_v3)
            thetam[...,2] = (valm[...,1] - valm[...,2]) / (2*v2_m_v3)

            tic.Tac("Split", "thetap and thetam", False)

            if useNumba:
                # Much faster (approx. 2x faster)

                G12_ij, G13_ij, G23_ij = Numba.Get_G12_G13_G23(M1, M2, M3)

                tic.Tac("Split", "Gab", False)

                list_mi = [m1, m2, m3]
                list_Gab = [G12_ij, G13_ij, G23_ij]

                projP, projM = Numba.Get_projP_projM_3D(dvalp, dvalm, thetap, thetam, list_mi, list_Gab)
                projP, projM = FeArray._asfearrays(projP, projM)
            
            else:

                def __Construction_Gij(Ma, Mb):

                    Gij = np.zeros((Ne, nPg, 6, 6))

                    part1 = lambda Ma, Mb: np.einsum('...ik,...jl->...ijkl', Ma, Mb, optimize='optimal')
                    part2 = lambda Ma, Mb: np.einsum('...il,...jk->...ijkl', Ma, Mb, optimize='optimal')

                    Gijkl = part1(Ma, Mb) + part2(Ma, Mb) + part1(Mb, Ma) + part2(Mb, Ma)

                    listI = [0]*6; listI.extend([1]*6); listI.extend([2]*6); listI.extend([1]*6); listI.extend([0]*12)
                    listJ = [0]*6; listJ.extend([1]*6); listJ.extend([2]*18); listJ.extend([1]*6)
                    listK = [0,1,2,1,0,0]*6
                    listL = [0,1,2,2,2,1]*6
                    
                    columns = np.arange(0,6, dtype=int).reshape((1,6)).repeat(6,axis=0).ravel()
                    lines = np.sort(columns)

                    # # builds a str matrix to check whether the indexes are good or not
                    # ma = np.zeros((6,6), dtype=np.object0)
                    # for lin,col,i,j,k,l in zip(lines, columns, listI, listJ, listK, listL):
                    #     text = f"{i+1}{j+1}{k+1}{l+1}"
                    #     ma[lin,col] = text
                    #     pass

                    Gij[:,:,lines, columns] = Gijkl[:,:,listI,listJ,listK,listL]                    
                    
                    Gij[:,:,:3,3:6] = Gij[:,:,:3,3:6] * coef
                    Gij[:,:,3:6,:3] = Gij[:,:,3:6,:3] * coef
                    Gij[:,:,3:6,3:6] = Gij[:,:,3:6,3:6] * 2

                    return FeArray.asfearray(Gij)

                G12 = __Construction_Gij(M1, M2)
                G13 = __Construction_Gij(M1, M3)
                G23 = __Construction_Gij(M2, M3)

                tic.Tac("Split", "Gab", False)

                m1xm1 = TensorProd(m1, m1, ndim=1)
                m2xm2 = TensorProd(m2, m2, ndim=1)
                m3xm3 = TensorProd(m3, m3, ndim=1)

                tic.Tac("Split", "mixmi", False)

                projP = (dvalp[:,:,0] * m1xm1) + (dvalp[:,:,1] * m2xm2) + (dvalp[:,:,2] * m3xm3) \
                      + (thetap[:,:,0] * G12) + (thetap[:,:,1] * G13) + (thetap[:,:,2] * G23)
                
                projM = (dvalm[:,:,0] * m1xm1) + (dvalm[:,:,1] * m2xm2) + (dvalm[:,:,2] * m3xm3) \
                      + (thetam[:,:,0] * G12) + (thetam[:,:,1] * G13) + (thetam[:,:,2] * G23)

            tic.Tac("Split", "projP and projM", False)

        if verif:

            vectorP = projP @ vector_e_pg
            vectorM = projM @ vector_e_pg

            # check orthogonality
            ortho_vP_vM = np.abs(vectorP @ vectorM)
            ortho_vM_vP = np.abs(vectorM @ vectorP)
            ortho_v_v = np.abs(vector_e_pg @ vector_e_pg)
            if np.min(ortho_v_v) > 0:
                verif_PM = np.max(ortho_vP_vM/ortho_v_v)
                assert verif_PM <= 1e-12, f"{verif_PM:.3e}"
                verif_MP = np.max(ortho_vM_vP/ortho_v_v)
                assert verif_MP <= 1e-12, f"{verif_MP:.3e}"

            # check that: vector_e_pg = vectorP_e_pg + vectorM_e_pg            
            diff_vect = vector_e_pg - (vectorP + vectorM)

            # Rounding errors in the construction of 3D eigen projectors see [Remark M]
            tol = 1e-12 if dim == 2 else 1e-11

            if np.max(Norm(vector_e_pg, axis=-1)) > 0:                
                test_vect = Norm(diff_vect, axis=-1)/Norm(vector_e_pg, axis=-1)
                assert np.max(test_vect) < tol, f"vector_e_pg != vectorP_e_pg + vectorM_e_pg -> {np.max(test_vect):.3e}"
            
        return projP, projM