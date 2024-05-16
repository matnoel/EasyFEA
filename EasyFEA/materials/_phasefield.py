# Copyright (C) 2021-2024 Université Gustave Eiffel. All rights reserved.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from abc import ABC, abstractmethod, abstractproperty
from typing import Union
from enum import Enum

from scipy.linalg import sqrtm

# utilities
from .. import np
from ..utilities import Tic, Numba_Interface
# fem
from ..fem import Mesh
# others
from ._utils import (_IModel, ModelType,
                     Reshape_variable, Heterogeneous_Array,
                     Tensor_Product,
                     KelvinMandel_Matrix, Project_Kelvin,
                     Result_in_Strain_or_Stress_field,
                     Get_Pmat, Apply_Pmat)

# ----------------------------------------------
# Elasticity
# ----------------------------------------------

from ._elastic import _Elas, Elas_Isot, Elas_IsotTrans, Elas_Anisot

# ----------------------------------------------
# Phase field
# ----------------------------------------------

class PhaseField(_IModel):

    class ReguType(str, Enum):
        """Available crack regularization"""
        AT1 = "AT1"
        AT2 = "AT2"

        def __str__(self) -> str:
            return self.name

    class SplitType(str, Enum):
        """Available splits"""

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
        """Solver used to manage crack irreversibility"""
        History = "History"
        HistoryDamage = "HistoryDamage"
        BoundConstrain = "BoundConstrain"

        def __str__(self) -> str:
            return self.name

    def __init__(self, material: _Elas, split: SplitType, regularization: ReguType, Gc: Union[float,np.ndarray], l0: float, solver=SolverType.History, A=None):
        """Creation of a gradient damage model

        Parameters
        ----------
        material : _Elas
            Elastic behavior (Elas_Isot, Elas_IsotTrans, Elas_Anisot)
        split : SplitType
            Split of elastic energy density (see PhaseField_Model.get_splits())
        regularization : RegularizationType
            AT1 or AT2 crack regularization model
        Gc : float | np.ndarray
            Critical energy restitution rate in J.m^-2
        l0 : float | np.ndarray
            Half crack width
        solver : SolverType, optional
            Solver used to manage crack irreversibility, by default History (see SolverType)        
        A : np.ndarray, optional
            Matrix characterizing the direction of model anisotropy for crack energy
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
        return isinstance(self.Gc, np.ndarray)
    
    @staticmethod
    def Get_splits() -> list[SplitType]:
        """splits available"""
        return list(PhaseField.SplitType)    
    
    @staticmethod
    def Get_regularisations() -> list[ReguType]:
        """regularizations available"""
        __regularizations = list(PhaseField.ReguType)
        return __regularizations    

    @staticmethod
    def Get_solvers() -> list[SolverType]:
        """Available solvers used to manage crack irreversibility"""
        __solveurs = list(PhaseField.SolverType)
        return __solveurs

    @property
    def k(self) -> float:
        """diffusion therm"""

        Gc = self.__Gc
        l0 = self.__l0
        
        # J/m
        if self.__regularization == self.ReguType.AT1:
            k = 3/4 * Gc * l0 
        elif self.__regularization == self.ReguType.AT2:
            k = Gc * l0        

        return k

    def get_r_e_pg(self, PsiP_e_pg: np.ndarray) -> np.ndarray:
        """reaction therm"""

        Gc = Reshape_variable(self.__Gc, PsiP_e_pg.shape[0], PsiP_e_pg.shape[1])
        l0 = self.__l0

        # J/m3
        if self.__regularization == self.ReguType.AT1:
            r = 2 * PsiP_e_pg
        elif self.__regularization == self.ReguType.AT2:
            r = 2 * PsiP_e_pg + (Gc/l0)
        
        return r

    def get_f_e_pg(self, PsiP_e_pg: np.ndarray) -> np.ndarray:
        """source therm"""

        Gc = Reshape_variable(self.__Gc, PsiP_e_pg.shape[0], PsiP_e_pg.shape[1])
        l0 = self.__l0
        
        # J/m3
        if self.__regularization == self.ReguType.AT1:
            f = 2 * PsiP_e_pg - ( (3*Gc) / (8*l0) )            
            absF = np.abs(f)
            f = (f+absF)/2
        elif self.__regularization == self.ReguType.AT2:
            f = 2 * PsiP_e_pg
        
        return f

    def get_g_e_pg(self, d_n: np.ndarray, mesh: Mesh, matrixType: str, k_residu=1e-12) -> np.ndarray:
        """Degradation function"""

        d_e_n = mesh.Locates_sol_e(d_n)
        Nd_pg = mesh.Get_N_pg(matrixType)

        d_e_pg = np.einsum('pij,ej->ep', Nd_pg, d_e_n, optimize='optimal')        

        if self.__regularization in self.Get_regularisations():
            g_e_pg: np.ndarray = (1-d_e_pg)**2 + k_residu
        else:
            raise Exception("Not implemented")

        assert mesh.Ne == g_e_pg.shape[0]
        assert mesh.Get_nPg(matrixType) == g_e_pg.shape[1]
        
        return g_e_pg
    
    @property
    def A(self) -> np.ndarray:
        """Matrix characterizing the direction of model anisotropy for crack energy"""
        return self.__A
    
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
        """Split of elastic energy density"""
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
        """Crack regularization model ["AT1", "AT2"]"""
        return self.__regularization
    
    @regularization.setter
    def regularization(self, value: str) -> None:
        types = self.Get_regularisations()
        assert value in types, f"Must be included in {types}"
        self.Need_Update()
        self.__regularization = value
    
    @property
    def material(self) -> _Elas:
        """displacement model"""
        return self.__material

    @property
    def solver(self):
        """Solver used to manage crack irreversibility"""
        return self.__solver
    
    @solver.setter
    def solver(self, value: str):        
        solvers = self.Get_solvers()
        assert value in solvers, f"Must be included in {solvers}"
        self.Need_Update()
        self.__solver = value

    @property
    def Gc(self) -> Union[float, np.ndarray]:
        """Critical energy release rate [J/m^2]"""
        return self.__Gc
    
    @Gc.setter
    def Gc(self, value: Union[float, np.ndarray]):
        self._Test_Sup0(value)
        self.Need_Update()
        self.__Gc = value

    @property
    def l0(self) -> float:
        """Half crack width"""
        return self.__l0
    
    @l0.setter
    def l0(self, value: float):
        self._Test_Sup0(value)
        assert isinstance(value, (int, float)), 'l0 must be a homogeneous parameter'
        self.Need_Update()
        self.__l0 = value

    @property
    def c0(self):
        """Scaling parameter for accurate dissipation of crack energy"""
        if self.__regularization == self.ReguType.AT1:
            c0 = 8/3
        elif self.__regularization == self.ReguType.AT2:
            c0 = 2
        return c0
            
    def Calc_psi_e_pg(self, Epsilon_e_pg: np.ndarray):
        """Calculation of elastic energy density\n
        psiP_e_pg = 1/2 SigmaP_e_pg * Epsilon_e_pg\n
        psiM_e_pg = 1/2 SigmaM_e_pg * Epsilon_e_pg\n
        Such as :\n
        SigmaP_e_pg = cP_e_pg * Epsilon_e_pg\n
        SigmaM_e_pg = cM_e_pg * Epsilon_e_pg       
        """

        SigmaP_e_pg, SigmaM_e_pg = self.Calc_Sigma_e_pg(Epsilon_e_pg)

        tic = Tic()

        psiP_e_pg = np.sum(1/2 * Epsilon_e_pg * SigmaP_e_pg, -1)
        psiM_e_pg = np.sum(1/2 * Epsilon_e_pg * SigmaM_e_pg, -1)

        tic.Tac("Matrix", "psiP_e_pg and psiM_e_pg", False)

        return psiP_e_pg, psiM_e_pg

    def Calc_Sigma_e_pg(self, Epsilon_e_pg: np.ndarray):
        """Calcul la contrainte en fonction de la deformation et du split\n
        Ici on calcul :\n
        SigmaP_e_pg = cP_e_pg * Epsilon_e_pg \n
        SigmaM_e_pg = cM_e_pg * Epsilon_e_pg

        Parameters
        ----------
        Epsilon_e_pg : np.ndarray
            deformations stored at elements and gauss points

        Returns
        -------
        np.ndarray
            SigmaP_e_pg, SigmaM_e_pg : constraints stored at elements and Gauss points
        """       

        Ne, nPg, dim = Epsilon_e_pg.shape[:3]
        
        cP_e_pg, cM_e_pg = self.Calc_C(Epsilon_e_pg)

        tic = Tic()
        
        Epsilon_e_pg = Epsilon_e_pg.reshape((Ne,nPg,dim,1))

        SigmaP_e_pg = np.reshape(cP_e_pg @ Epsilon_e_pg, (Ne,nPg,-1))
        SigmaM_e_pg = np.reshape(cM_e_pg @ Epsilon_e_pg, (Ne,nPg,-1))

        tic.Tac("Matrix", "SigmaP_e_pg and SigmaM_e_pg", False)

        return SigmaP_e_pg, SigmaM_e_pg
    
    def Calc_C(self, Epsilon_e_pg: np.ndarray, verif=False):
        """Calculating the behavior law.

        Parameters
        ----------
        Epsilon_e_pg : np.ndarray
            deformations stored at elements and gauss points

        Returns
        -------
        np.ndarray
            Returns cP_e_pg, cM_e_pg
        """

        Ne, nPg = Epsilon_e_pg.shape[:2]

        if self.__split == self.SplitType.Bourdin:
            cP_e_pg, cM_e_pg = self.__Split_Bourdin(Ne, nPg)

        elif self.__split == self.SplitType.Amor:
            cP_e_pg, cM_e_pg = self.__Split_Amor(Epsilon_e_pg)

        elif self.__split == self.SplitType.Miehe or "Strain" in self.__split:
            cP_e_pg, cM_e_pg = self.__Split_Miehe(Epsilon_e_pg, verif=verif)
        
        elif self.__split == self.SplitType.Zhang or "Stress" in self.__split:
            cP_e_pg, cM_e_pg = self.__Split_Stress(Epsilon_e_pg, verif=verif)

        elif self.__split == self.SplitType.He:
            cP_e_pg, cM_e_pg = self.__Split_He(Epsilon_e_pg, verif=verif)

        return cP_e_pg, cM_e_pg

    def __Split_Bourdin(self, Ne: int, nPg: int):
        """[Bourdin 2000] DOI : 10.1016/S0022-5096(99)00028-9"""

        tic = Tic()
        c = self.__material.C
        c_e_pg = Reshape_variable(c, Ne, nPg)

        cP_e_pg = c_e_pg
        cM_e_pg = np.zeros_like(cP_e_pg)
        tic.Tac("Split",f"cP_e_pg and cM_e_pg", False)

        return cP_e_pg, cM_e_pg

    def __Split_Amor(self, Epsilon_e_pg: np.ndarray):
        """[Amor 2009] DOI : 10.1016/j.jmps.2009.04.011"""

        assert isinstance(self.__material, Elas_Isot), f"Implemented only for Elas_Isot material"
        
        tic = Tic()
        
        material = self.__material
        
        Ne, nPg = Epsilon_e_pg.shape[:2]

        bulk = material.get_bulk()
        mu = material.get_mu()

        Rp_e_pg, Rm_e_pg = self.__Rp_Rm(Epsilon_e_pg)

        dim = material.dim

        if dim == 2:
            Ivect = np.array([1,1,0]).reshape((3,1))
            size = 3
        else:
            Ivect = np.array([1,1,1,0,0,0]).reshape((6,1))
            size = 6

        IxI = np.array(Ivect.dot(Ivect.T))

        spherP_e_pg = np.einsum('ep,ij->epij', Rp_e_pg, IxI, optimize='optimal')
        spherM_e_pg = np.einsum('ep,ij->epij', Rm_e_pg, IxI, optimize='optimal')

        # Deviatoric projector
        Pdev = np.eye(size) - 1/dim * IxI

        # einsum faster than with resizing (no need to try with numba)
        if material.isHeterogeneous:
            mu_e_pg = Reshape_variable(mu, Ne, nPg)
            bulk_e_pg = Reshape_variable(bulk, Ne, nPg)

            devPart_e_pg = np.einsum("ep,ij->epij",2*mu_e_pg, Pdev, optimize="optimal")        
    
            cP_e_pg = np.einsum('ep,epij->epij', bulk_e_pg, spherP_e_pg, optimize='optimal')  + devPart_e_pg
            cM_e_pg = np.einsum('ep,epij->epij', bulk_e_pg, spherM_e_pg, optimize='optimal')

        else:
            devPart = 2*mu * Pdev

            cP_e_pg = bulk * spherP_e_pg  + devPart
            cM_e_pg = bulk * spherM_e_pg

        tic.Tac("Split",f"cP_e_pg and cM_e_pg", False)

        return cP_e_pg, cM_e_pg

    def __Rp_Rm(self, vecteur_e_pg: np.ndarray):
        """Returns Rp_e_pg, Rm_e_pg"""

        Ne, nPg = vecteur_e_pg.shape[:2]

        dim = self.__material.dim

        trace = np.zeros((Ne, nPg))

        trace = vecteur_e_pg[:,:,0] + vecteur_e_pg[:,:,1]

        if dim == 3:
            trace += vecteur_e_pg[:,:,2]

        Rp_e_pg = (1+np.sign(trace))/2
        Rm_e_pg = (1+np.sign(-trace))/2

        return Rp_e_pg, Rm_e_pg
    
    def __Split_Miehe(self, Epsilon_e_pg: np.ndarray, verif=False):
        """[Miehe 2010] DOI : 10.1016/j.cma.2010.04.011"""

        dim = self.__material.dim
        useNumba = self.useNumba

        projP_e_pg, projM_e_pg = self.__Spectral_Decomposition(Epsilon_e_pg, verif)

        Ne, nPg = Epsilon_e_pg.shape[:2]

        isHeterogene = self.__material.isHeterogeneous

        tic = Tic()

        if self.__split == self.SplitType.Miehe:
            
            assert isinstance(self.__material, Elas_Isot), f"Implemented only for Elas_Isot material"

            # Calculating Rp and Rm
            Rp_e_pg, Rm_e_pg = self.__Rp_Rm(Epsilon_e_pg)
            
            # Calculation IxI
            if dim == 2:
                I = np.array([1,1,0]).reshape((3,1))
            elif dim == 3:
                I = np.array([1,1,1,0,0,0]).reshape((6,1))
            IxI = I.dot(I.T)

            # Calculation of spherical part
            spherP_e_pg = np.einsum('ep,ij->epij', Rp_e_pg, IxI, optimize='optimal')
            spherM_e_pg = np.einsum('ep,ij->epij', Rm_e_pg, IxI, optimize='optimal')

            # Calculation of the behavior law
            lamb = self.__material.get_lambda()
            mu = self.__material.get_mu()

            if isHeterogene:
                lamb_e_pg = Reshape_variable(lamb, Ne, nPg)
                mu_e_pg = Reshape_variable(mu, Ne, nPg)

                funcMult = lambda ep, epij: np.einsum('ep,epij->epij', ep, epij, optimize='optimal')

                cP_e_pg = funcMult(lamb_e_pg,spherP_e_pg) + funcMult(2*mu_e_pg, projP_e_pg)
                cM_e_pg = funcMult(lamb_e_pg,spherM_e_pg) + funcMult(2*mu_e_pg, projM_e_pg)

            else:
                cP_e_pg = lamb*spherP_e_pg + 2*mu*projP_e_pg
                cM_e_pg = lamb*spherM_e_pg + 2*mu*projM_e_pg
        
        elif "Strain" in self.__split:
            
            c = self.__material.C
            
            # here don't use numba if behavior is heterogeneous
            if useNumba and not isHeterogene:
                # Faster (x2) but not usable if heterogeneous (memory problem)
                Cpp, Cpm, Cmp, Cmm = Numba_Interface.Get_Anisot_C(projP_e_pg, c, projM_e_pg)

            else:
                # Here we don't use einsum, otherwise it's much longer
                c_e_pg = Reshape_variable(c, Ne, nPg)

                pc = np.transpose(projP_e_pg, [0,1,3,2]) @ c_e_pg
                mc = np.transpose(projM_e_pg, [0,1,3,2]) @ c_e_pg
                
                Cpp = pc @ projP_e_pg
                Cpm = pc @ projM_e_pg
                Cmm = mc @ projM_e_pg
                Cmp = mc @ projP_e_pg
            
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
            raise Exception("Unknown split")

        tic.Tac("Split",f"cP_e_pg and cM_e_pg", False)

        return cP_e_pg, cM_e_pg
    
    def __Split_Stress(self, Epsilon_e_pg: np.ndarray, verif=False):
        """Construct Cp and Cm for the split in stress"""

        # Recovers stresses        
        material = self.__material
        c = material.C

        shape_c = len(c.shape)

        Ne, nPg = Epsilon_e_pg.shape[:2]

        isHeterogene = material.isHeterogeneous

        if shape_c == 2:
            indices = ''
        elif shape_c == 3:
            indices = 'e'
        elif shape_c == 4:
            indices = 'ep'

        Sigma_e_pg = np.einsum(f'{indices}ij,epj->epi', c, Epsilon_e_pg, optimize='optimal')

        # Construct projectors such that SigmaP = Pp : Sigma and SigmaM = Pm : Sigma
        projP_e_pg, projM_e_pg = self.__Spectral_Decomposition(Sigma_e_pg, verif)

        tic = Tic()

        if self.__split == self.SplitType.Stress:
        
            assert isinstance(material, Elas_Isot)

            E = material.E
            v = material.v

            c = material.C

            dim = self.dim

            # Calculates Rp and Rm
            Rp_e_pg, Rm_e_pg = self.__Rp_Rm(Sigma_e_pg)
            
            # Calcul IxI
            if dim == 2:
                I = np.array([1,1,0]).reshape((3,1))
            else:
                I = np.array([1,1,1,0,0,0]).reshape((6,1))
            IxI = I.dot(I.T)

            RpIxI_e_pg = np.einsum('ep,ij->epij', Rp_e_pg, IxI, optimize='optimal')
            RmIxI_e_pg = np.einsum('ep,ij->epij', Rm_e_pg, IxI, optimize='optimal')
            
            def funcMult(a, epij: np.ndarray, indices=indices):
                return np.einsum(f'{indices},epij->epij', a, epij, optimize='optimal')

            if dim == 2:
                if material.planeStress:
                    sP_e_pg = funcMult((1+v)/E, projP_e_pg) - funcMult(v/E, RpIxI_e_pg)
                    sM_e_pg = funcMult((1+v)/E, projM_e_pg) - funcMult(v/E, RmIxI_e_pg) 
                else:
                    sP_e_pg = funcMult((1+v)/E, projP_e_pg) - funcMult(v*(1+v)/E, RpIxI_e_pg) 
                    sM_e_pg = funcMult((1+v)/E, projM_e_pg) - funcMult(v*(1+v)/E, RmIxI_e_pg) 
            elif dim == 3:
                mu = material.get_mu()

                if isinstance(mu, (float, int)):
                    ind = ''
                elif len(mu.shape) == 1:
                    ind = 'e'
                elif len(mu.shape) == 2:
                    ind = 'ep'                    

                sP_e_pg = funcMult(1/(2*mu), projP_e_pg, ind) - funcMult(v/E, RpIxI_e_pg) 
                sM_e_pg = funcMult(1/(2*mu), projM_e_pg, ind) - funcMult(v/E, RmIxI_e_pg) 
            
            useNumba = self.useNumba
            if useNumba and not isHeterogene:
                # Faster
                cP_e_pg, cM_e_pg = Numba_Interface.Get_Cp_Cm_Stress(c, sP_e_pg, sM_e_pg)
            else:
                if isHeterogene:
                    c_e_pg = Reshape_variable(c, Ne, nPg)
                    cT = np.transpose(c_e_pg, [0,1,3,2])

                    cP_e_pg = cT @ sP_e_pg @ c_e_pg
                    cM_e_pg = cT @ sM_e_pg @ c_e_pg
                else:
                    cT = c.T
                    cP_e_pg = np.einsum('ij,epjk,kl->epil', cT, sP_e_pg, c, optimize='optimal')
                    cM_e_pg = np.einsum('ij,epjk,kl->epil', cT, sM_e_pg, c, optimize='optimal')
        
        elif self.__split == self.SplitType.Zhang or "Stress" in self.__split:
            
            Cp_e_pg = np.einsum(f'epij,{indices}jk->epik', projP_e_pg, c, optimize='optimal')
            Cm_e_pg = np.einsum(f'epij,{indices}jk->epik', projM_e_pg, c, optimize='optimal')

            if self.__split == self.SplitType.Zhang:
                # [Zhang 2020] DOI : 10.1016/j.cma.2019.112643
                cP_e_pg = Cp_e_pg
                cM_e_pg = Cm_e_pg
            
            else:
                # Builds Cp and Cm
                S = material.S
                if self.useNumba and not isHeterogene:
                    # Faster
                    Cpp, Cpm, Cmp, Cmm = Numba_Interface.Get_Anisot_C(Cp_e_pg, S, Cm_e_pg)
                else:
                    # Here we don't use einsum, otherwise it's much longer
                    s_e_pg = Reshape_variable(S, Ne, nPg)

                    ps = np.transpose(Cp_e_pg, [0,1,3,2]) @ s_e_pg
                    ms = np.transpose(Cm_e_pg, [0,1,3,2]) @ s_e_pg
                    
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
                    raise Exception("Unknown split")

        tic.Tac("Split",f"cP_e_pg and cM_e_pg", False)

        return cP_e_pg, cM_e_pg

    def __Split_He(self, Epsilon_e_pg: np.ndarray, verif=False):
            
        # Here the material is supposed to be homogeneous
        material = self.__material

        C = material.C        
        
        assert not material.isHeterogeneous, "He decomposition has not been implemented for heterogeneous materials"
        # for heterogeneous materials how to make sqrtm ?
        sqrtC = sqrtm(C)
        
        if verif :
            # Verif C^1/2 * C^1/2 = C
            testC = np.dot(sqrtC, sqrtC) - C
            assert np.linalg.norm(testC)/np.linalg.norm(C) < 1e-12

        inv_sqrtC = np.linalg.inv(sqrtC)

        # Get epsilon tild
        Epsilont_e_pg = np.einsum('ij,epj->epi', sqrtC, Epsilon_e_pg, optimize='optimal')

        # Calculates projectors
        projPt_e_pg, projMt_e_pg = self.__Spectral_Decomposition(Epsilont_e_pg, verif)

        tic = Tic()

        # projP_e_pg = inv_sqrtC @ (projPt_e_pg @ sqrtC)
        # projM_e_pg = inv_sqrtC @ (projMt_e_pg @ sqrtC)
        
        # faster
        projP_e_pg = np.einsum('ij,epjk,kl->epil', inv_sqrtC, projPt_e_pg, sqrtC, optimize='optimal')
        projM_e_pg = np.einsum('ij,epjk,kl->epil', inv_sqrtC, projMt_e_pg, sqrtC, optimize='optimal')

        tic.Tac("Split",f"proj Tild to proj", False)

        # cP_e_pg = np.einsum('epji,jk,epkl->epil', projP_e_pg, C, projP_e_pg, optimize='optimal')
        # cM_e_pg = np.einsum('epji,jk,epkl->epil', projM_e_pg, C, projM_e_pg, optimize='optimal')

        # faster
        cP_e_pg = np.transpose(projP_e_pg, (0,1,3,2)) @ C @ projP_e_pg
        cM_e_pg = np.transpose(projM_e_pg, (0,1,3,2)) @ C @ projM_e_pg

        tic.Tac("Split",f"cP_e_pg and cM_e_pg", False)

        if verif:
            vector_e_pg = Epsilon_e_pg.copy()
            mat = C.copy()
            
            vectorP = np.einsum('epij,epj->epi', projP_e_pg, vector_e_pg, optimize='optimal')
            vectorM = np.einsum('epij,epj->epi', projM_e_pg, vector_e_pg, optimize='optimal')
            
            # Et+:Et- = 0 already checked in spectral decomposition
            
            # Check that vector_e_pg = vectorP_e_pg + vectorM_e_pg
            decomp = vector_e_pg-(vectorP + vectorM)
            if np.linalg.norm(vector_e_pg) > 0:
                verifDecomp = np.linalg.norm(decomp)/np.linalg.norm(vector_e_pg)
                assert verifDecomp < 1e-12

            # Orthogonalité E+:C:E-
            ortho_vP_vM = np.abs(np.einsum('epi,ij,epj->ep',vectorP, mat, vectorM, optimize='optimal'))
            ortho_vM_vP = np.abs(np.einsum('epi,ij,epj->ep',vectorM, mat, vectorP, optimize='optimal'))
            ortho_v_v = np.abs(np.einsum('epi,ij,epj->ep', vector_e_pg, mat, vector_e_pg, optimize='optimal'))
            if ortho_v_v.min() > 0:
                vertifOrthoEpsPM = np.max(ortho_vP_vM/ortho_v_v)
                assert vertifOrthoEpsPM < 1e-12
                vertifOrthoEpsMP = np.max(ortho_vM_vP/ortho_v_v)
                assert vertifOrthoEpsMP < 1e-12

        return cP_e_pg, cM_e_pg

    def __Eigen_values_vectors_projectors(self, vector_e_pg: np.ndarray, verif=False) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:

        dim = self.__material.dim

        coef = self.__material.coef
        Ne, nPg = vector_e_pg.shape[:2]

        tic = Tic()

        # Reconstructs the strain tensor [e,pg,dim,dim]
        matrix_e_pg = np.zeros((Ne,nPg,dim,dim))
        for d in range(dim):
            matrix_e_pg[:,:,d,d] = vector_e_pg[:,:,d]
        if dim == 2:
            # [x, y, xy]
            # xy
            matrix_e_pg[:,:,0,1] = vector_e_pg[:,:,2]/coef
            matrix_e_pg[:,:,1,0] = vector_e_pg[:,:,2]/coef
        else:
            # [x, y, z, yz, xz, xy]
            # yz
            matrix_e_pg[:,:,1,2] = vector_e_pg[:,:,3]/coef
            matrix_e_pg[:,:,2,1] = vector_e_pg[:,:,3]/coef
            # xz
            matrix_e_pg[:,:,0,2] = vector_e_pg[:,:,4]/coef
            matrix_e_pg[:,:,2,0] = vector_e_pg[:,:,4]/coef
            # xy
            matrix_e_pg[:,:,0,1] = vector_e_pg[:,:,5]/coef
            matrix_e_pg[:,:,1,0] = vector_e_pg[:,:,5]/coef

        tic.Tac("Split", "vector_e_pg -> matrix_e_pg", False)

        # trace_e_pg = np.trace(matrix_e_pg, axis1=2, axis2=3)
        trace_e_pg = np.einsum('epii->ep', matrix_e_pg, optimize='optimal')

        if self.dim == 2:
            # invariants of the strain tensor [e,pg]

            a_e_pg = matrix_e_pg[:,:,0,0]
            b_e_pg = matrix_e_pg[:,:,0,1]
            c_e_pg = matrix_e_pg[:,:,1,0]
            d_e_pg = matrix_e_pg[:,:,1,1]
            det_e_pg = (a_e_pg*d_e_pg)-(c_e_pg*b_e_pg)

            tic.Tac("Split", "Invariants", False)

            # Eigenvalue calculations [e,pg]
            delta = trace_e_pg**2 - (4*det_e_pg)
            eigs_e_pg = np.zeros((Ne,nPg,2))
            eigs_e_pg[:,:,0] = (trace_e_pg - np.sqrt(delta))/2
            eigs_e_pg[:,:,1] = (trace_e_pg + np.sqrt(delta))/2

            tic.Tac("Split", "Eigenvalues", False)
            
            # Constants for calculating m1 = (matrice_e_pg - v2*I)/(v1-v2)
            v2I = np.einsum('ep,ij->epij', eigs_e_pg[:,:,1], np.eye(2), optimize='optimal')
            v1_m_v2 = eigs_e_pg[:,:,0] - eigs_e_pg[:,:,1]
            
            # element identification and gauss points where vp1 != vp2
            # elements, pdgs = np.where(v1_m_v2 != 0)
            elems, pdgs = np.where(eigs_e_pg[:,:,0] != eigs_e_pg[:,:,1])
            
            # construction of eigenbases m1 and m2 [e,pg,dim,dim]
            M1 = np.zeros((Ne,nPg,2,2))
            M1[:,:,0,0] = 1
            if elems.size > 0:
                v1_m_v2[v1_m_v2==0] = 1 # to avoid dividing by 0
                m1_tot = np.einsum('epij,ep->epij', matrix_e_pg-v2I, 1/v1_m_v2, optimize='optimal')
                M1[elems, pdgs] = m1_tot[elems, pdgs]
            M2 = np.eye(2) - M1

            tic.Tac("Split", "Eigenprojectors", False)
        
        elif self.dim == 3:

            def __Normalize(M1, M2, M3):
                M1 = np.einsum('epij,ep->epij', M1, 1/np.linalg.norm(M1, axis=(2,3)), optimize='optimal')
                M2 = np.einsum('epij,ep->epij', M2, 1/np.linalg.norm(M2, axis=(2,3)), optimize='optimal')
                M3 = np.einsum('epij,ep->epij', M3, 1/np.linalg.norm(M3, axis=(2,3)), optimize='optimal')

                return M1, M2, M3

            version = 'invariants' # 'invariants', 'eigh'

            if version == 'eigh':

                valnum, vectnum = np.linalg.eigh(matrix_e_pg)

                tic.Tac("Split", "np.linalg.eigh", False)

                func_Mi = lambda mi: np.einsum('epi,epj->epij', mi, mi, optimize='optimal')

                M1 = func_Mi(vectnum[:,:,:,0])
                M2 = func_Mi(vectnum[:,:,:,1])
                M3 = func_Mi(vectnum[:,:,:,2])
                
                eigs_e_pg = valnum

                tic.Tac("Split", "Eigenvalues and eigenprojectors", False)

            elif version == 'invariants':

                # [Q.-C. He Closed-form coordinate-free]

                a11_e_pg = matrix_e_pg[:,:,0,0]; a12_e_pg = matrix_e_pg[:,:,0,1]; a13_e_pg = matrix_e_pg[:,:,0,2]
                a21_e_pg = matrix_e_pg[:,:,1,0]; a22_e_pg = matrix_e_pg[:,:,1,1]; a23_e_pg = matrix_e_pg[:,:,1,2]
                a31_e_pg = matrix_e_pg[:,:,2,0]; a32_e_pg = matrix_e_pg[:,:,2,1]; a33_e_pg = matrix_e_pg[:,:,2,2]

                det_e_pg = a11_e_pg * ((a22_e_pg*a33_e_pg)-(a32_e_pg*a23_e_pg)) - a12_e_pg * ((a21_e_pg*a33_e_pg)-(a31_e_pg*a23_e_pg)) + a13_e_pg * ((a21_e_pg*a32_e_pg)-(a31_e_pg*a22_e_pg))            

                # Invariants
                I1_e_pg = trace_e_pg
                # mat_mat = np.einsum('epij,epjk->epik', matrice_e_pg, matrice_e_pg, optimize='optimal')
                mat_mat = matrix_e_pg @ matrix_e_pg
                trace_mat_mat = np.einsum('epii->ep', mat_mat, optimize='optimal')
                I2_e_pg = (trace_e_pg**2 - trace_mat_mat)/2
                I3_e_pg = det_e_pg

                tic.Tac("Split", "Invariants", False)

                h = I1_e_pg**2 - 3*I2_e_pg                

                racine_h = np.sqrt(h)
                racine_h_ij = racine_h.reshape((Ne, nPg, 1, 1)).repeat(3, axis=2).repeat(3, axis=3)            
                
                arg = (2*I1_e_pg**3 - 9*I1_e_pg*I2_e_pg + 27*I3_e_pg)/2 # -1 <= arg <= 1
                arg[h != 0] *= 1/h[h != 0]**(3/2)

                phi = np.arccos(arg)/3 # Lode's angle such that 0 <= theta <= pi/3

                filtreNot0 = h != 0
                elemsNot0 = list(set(np.ravel(np.where(filtreNot0)[0])))

                elemsMin = list(set(np.ravel(np.where(arg == 1)[0]))) # positions of double minimum eigenvalue            
                elemsMax = list(set(np.ravel(np.where(arg == -1)[0]))) # positions of double maximum eigenvalue

                elemsNot0 = np.setdiff1d(elemsNot0, elemsMin)
                elemsNot0 = np.setdiff1d(elemsNot0, elemsMax)                

                # Init eigen values
                E1_e_pg: np.ndarray = I1_e_pg/3 + 2/3 * racine_h * np.cos(2*np.pi/3 + phi)
                E2_e_pg: np.ndarray = I1_e_pg/3 + 2/3 * racine_h * np.cos(2*np.pi/3 - phi)
                E3_e_pg: np.ndarray = I1_e_pg/3 + 2/3 * racine_h * np.cos(phi)

                eigs_e_pg = (I1_e_pg/3).reshape((Ne, nPg, 1)).repeat(3, axis=2)
                if elemsNot0.size > 0:
                    eigs_e_pg[elemsNot0, :, 0] = E1_e_pg[elemsNot0]
                    eigs_e_pg[elemsNot0, :, 1] = E2_e_pg[elemsNot0]
                    eigs_e_pg[elemsNot0, :, 2] = E3_e_pg[elemsNot0]

                tic.Tac("Split", "Eigenvalues", False)

                # Init proj matrices
                M1 = np.zeros_like(matrix_e_pg); M1[:,:,0,0] = 1
                M2 = np.zeros_like(matrix_e_pg); M2[:,:,1,1] = 1
                M3 = np.zeros_like(matrix_e_pg); M3[:,:,2,2] = 1

                eye3 = np.zeros_like(matrix_e_pg)
                eye3[:,:,0,0] = 1; eye3[:,:,1,1] = 1; eye3[:,:,2,2] = 1
                I_rg = np.einsum('ep,epij->epij', I1_e_pg - racine_h, eye3/3, optimize='optimal')

                # 4. Three equal eigenvalues
                # 𝜖1 = 𝜖2 = 𝜖3 ⇐⇒ 𝑔 = 0.
                # do nothing because 𝜖1 = 𝜖2 = 𝜖3 = I1_e_pg/3

                # 2. Two maximum eigenvalues
                # 𝜖1 < 𝜖2 = 𝜖3 ⇐⇒ 𝑔 ≠ 0, 𝜃 = 𝜋∕3.
                
                elems2 = list(set(np.ravel(np.where(filtreNot0 & (E1_e_pg<E2_e_pg) & (E2_e_pg==E3_e_pg))[0])))
                M1[elems2] = ((I_rg[elems2] - matrix_e_pg[elems2])/racine_h_ij[elems2])
                M2[elems2] = M3[elems2] = (eye3[elems2] - M1[elems2])/2

                # 3. Two minimum eigenvalues
                # 𝜖1 = 𝜖2 < 𝜖3 ⇐⇒ 𝑔 ≠ 0, 𝜃 = 0.
                
                elems3 = list(set(np.ravel(np.where(filtreNot0 & (E1_e_pg==E2_e_pg) & (E2_e_pg<E3_e_pg))[0])))
                M3[elems3] = ((matrix_e_pg[elems3] - I_rg[elems3])/racine_h_ij[elems3])
                M1[elems3] = M2[elems3] = (eye3[elems3] - M3[elems3])/2

                # 1. Three distinct eigenvalues
                # 𝜖1 < 𝜖2 < 𝜖3 ⇐⇒ 𝑔 ≠ 0, 𝜃 ≠ 0, 𝜃 ≠ 𝜋∕3.
                
                elems1 = list(set(np.ravel(np.where(filtreNot0 & (E1_e_pg<E2_e_pg) & (E2_e_pg<E3_e_pg))[0])))

                E1_ij = E1_e_pg.reshape((Ne,nPg,1,1)).repeat(3, axis=2).repeat(3, axis=3)[elems1]
                E2_ij = E2_e_pg.reshape((Ne,nPg,1,1)).repeat(3, axis=2).repeat(3, axis=3)[elems1]
                E3_ij = E3_e_pg.reshape((Ne,nPg,1,1)).repeat(3, axis=2).repeat(3, axis=3)[elems1]

                matr1 = matrix_e_pg[elems1]
                eye3_1 = eye3[elems1]

                M1[elems1] = ((matr1 - E2_ij*eye3_1)/(E1_ij-E2_ij)) @ ((matr1 - E3_ij*eye3_1)/(E1_ij-E3_ij))
                M2[elems1] = ((matr1 - E1_ij*eye3_1)/(E2_ij-E1_ij)) @ ((matr1 - E3_ij*eye3_1)/(E2_ij-E3_ij))
                M3[elems1] = ((matr1 - E1_ij*eye3_1)/(E3_ij-E1_ij)) @ ((matr1 - E2_ij*eye3_1)/(E3_ij-E2_ij))

                M1, M2, M3 = __Normalize(M1, M2, M3)

                tic.Tac("Split", "Eigenprojectors", False)

        # Passing eigenbases in the form of a vector [e,pg,3] or [e,pg,6].
        if dim == 2:
            # [x, y, xy]
            m1 = np.zeros((Ne,nPg,3)); m2 = np.zeros_like(m1)
            m1[:,:,0] = M1[:,:,0,0];   m2[:,:,0] = M2[:,:,0,0]
            m1[:,:,1] = M1[:,:,1,1];   m2[:,:,1] = M2[:,:,1,1]            
            m1[:,:,2] = M1[:,:,0,1]*coef;   m2[:,:,2] = M2[:,:,0,1]*coef

            list_m = [m1, m2]

            list_M = [M1, M2]

        elif dim == 3:
            # [x, y, z, yz, xz, xy]
            m1 = np.zeros((Ne,nPg,6)); m2 = np.zeros_like(m1);  m3 = np.zeros_like(m1)
            m1[:,:,0] = M1[:,:,0,0];   m2[:,:,0] = M2[:,:,0,0]; m3[:,:,0] = M3[:,:,0,0]
            m1[:,:,1] = M1[:,:,1,1];   m2[:,:,1] = M2[:,:,1,1]; m3[:,:,1] = M3[:,:,1,1]
            m1[:,:,2] = M1[:,:,2,2];   m2[:,:,2] = M2[:,:,2,2]; m3[:,:,2] = M3[:,:,2,2]
            
            m1[:,:,3] = M1[:,:,1,2]*coef;   m2[:,:,3] = M2[:,:,1,2]*coef;   m3[:,:,3] = M3[:,:,1,2]*coef
            m1[:,:,4] = M1[:,:,0,2]*coef;   m2[:,:,4] = M2[:,:,0,2]*coef;   m3[:,:,4] = M3[:,:,0,2]*coef
            m1[:,:,5] = M1[:,:,0,1]*coef;   m2[:,:,5] = M2[:,:,0,1]*coef;   m3[:,:,5] = M3[:,:,0,1]*coef

            list_m = [m1, m2, m3]

            list_M = [M1, M2, M3]

        tic.Tac("Split", "Eigenvectors", False)        
        
        if verif:
            
            valnum, vectnum = np.linalg.eigh(matrix_e_pg)

            func_Mi = lambda mi: np.einsum('epi,epj->epij', mi, mi, optimize='optimal')
            func_ep_epij = lambda ep, epij : np.einsum('ep,epij->epij', ep, epij, optimize='optimal')

            M1_num = func_Mi(vectnum[:,:,:,0])
            M2_num = func_Mi(vectnum[:,:,:,1])

            matrix = func_ep_epij(eigs_e_pg[:,:,0], M1) + func_ep_epij(eigs_e_pg[:,:,1], M2)

            matrix_eig = func_ep_epij(valnum[:,:,0], M1_num) + func_ep_epij(valnum[:,:,1], M2_num)
            
            if dim == 3:                
                M3_num = func_Mi(vectnum[:,:,:,2])
                matrix = matrix + func_ep_epij(eigs_e_pg[:,:,2], M3)
                matrix_eig = matrix_eig + func_ep_epij(valnum[:,:,2], M3_num)

            # check if the default values are correct
            if valnum.max() > 0:
                ecartVal = eigs_e_pg - valnum                    
                testval = np.linalg.norm(ecartVal)/np.linalg.norm(valnum)
                assert testval <= 1e-12, "Error in the calculation of eigenvalues."

            # check if clean spotlights are correct
            def erreur_Mi_Minum(Mi, mi_num):
                Mi_num = np.einsum('epi,epj->epij', mi_num, mi_num, optimize='optimal')
                ecart = Mi_num-Mi
                erreur = np.linalg.norm(ecart)/np.linalg.norm(Mi)
                assert erreur <= 1e-10, "Error in the calculation of eigenprojectors."

            erreur_Mi_Minum(M1, vectnum[:,:,:,0])
            erreur_Mi_Minum(M2, vectnum[:,:,:,1])
            if dim == 3:
                erreur_Mi_Minum(M3, vectnum[:,:,:,2])

            # Verification that matrix = E1*M1 + E2*M2 + E3*M3
            if matrix_e_pg.max() > 0:
                ecart_matrix = matrix - matrix_e_pg
                errorMatrix = np.linalg.norm(ecart_matrix)/np.linalg.norm(matrix_e_pg)
                assert errorMatrix <= 1e-10, "matrice != E1*M1 + E2*M2 + E3*M3 != matrix_e_pg"                

            if matrix.max() > 0:
                erreurMatriceNumMatrice = np.linalg.norm(matrix_eig - matrix)/np.linalg.norm(matrix)
                assert erreurMatriceNumMatrice <= 1e-10, "matrice != matrice_num"

            # ortho test between M1 and M2
            verifOrtho_M1M2 = np.einsum('epij,epij->ep', M1, M2, optimize='optimal')
            textTest = "Orthogonality not verified"
            assert np.abs(verifOrtho_M1M2).max() <= 1e-9, textTest

            if dim == 3:
                verifOrtho_M1M3 = np.einsum('epij,epij->ep', M1, M3, optimize='optimal')
                assert np.abs(verifOrtho_M1M3).max() <= 1e-9, textTest
                verifOrtho_M2M3 = np.einsum('epij,epij->ep', M2, M3, optimize='optimal')
                assert np.abs(verifOrtho_M2M3).max() <= 1e-9, textTest

        return eigs_e_pg, list_m, list_M
    
    def __Spectral_Decomposition(self, vector_e_pg: np.ndarray, verif=False):
        """Calculate projP and projM such that:\n

        vector_e_pg = [1 1 sqrt(2)] \n
        
        vectorP = projP : vector -> [1, 1, sqrt(2)]\n
        vectorM = projM : vector -> [1, 1, sqrt(2)]\n

        returns projP, projM
        """

        useNumba = self.useNumba        

        dim = self.__material.dim        

        Ne, nPg = vector_e_pg.shape[:2]
        
        # recovery of eigenvalues, eigenvectors and eigenprojectors
        val_e_pg, list_m, list_M = self.__Eigen_values_vectors_projectors(vector_e_pg, verif)

        tic = Tic()
        
        # Recovery of the positive and negative parts of the eigenvalues [e,pg,2].
        valp = (val_e_pg+np.abs(val_e_pg))/2
        valm = (val_e_pg-np.abs(val_e_pg))/2
        
        # Calculation of di [e,pg,2].
        dvalp = np.heaviside(val_e_pg, 0.5)
        dvalm = np.heaviside(-val_e_pg, 0.5)

        if dim == 2:
            # eigenvectors
            m1, m2 = list_m[0], list_m[1]

            # elements and pdgs where eigenvalues 1 and 2 are different
            elems, pdgs = np.where(val_e_pg[:,:,0] != val_e_pg[:,:,1])

            v1_m_v2 = val_e_pg[:,:,0] - val_e_pg[:,:,1] # val1 - val2

            # Calculation of BetaP [e,pg,1].
            BetaP = dvalp[:,:,0].copy() # make sure you put copy here otherwise when Beta modification modifies dvalp at the same time!
            BetaP[elems,pdgs] = (valp[elems,pdgs,0]-valp[elems,pdgs,1])/v1_m_v2[elems,pdgs]
            
            # Calculation of BetaM [e,pg,1].
            BetaM = dvalm[:,:,0].copy()
            BetaM[elems,pdgs] = (valm[elems,pdgs,0]-valm[elems,pdgs,1])/v1_m_v2[elems,pdgs]
            
            # calc gammap and gammam
            gammap = dvalp - np.repeat(BetaP.reshape((Ne,nPg,1)),2, axis=2)
            gammam = dvalm - np.repeat(BetaM.reshape((Ne,nPg,1)), 2, axis=2)

            tic.Tac("Split", "Betas and gammas", False)

            if useNumba:
                # Faster
                projP, projM = Numba_Interface.Get_projP_projM_2D(BetaP, gammap, BetaM, gammam, m1, m2)

            else:
                # Calculation of mixmi [e,pg,3,3] or [e,pg,6,6].
                m1xm1 = np.einsum('epi,epj->epij', m1, m1, optimize='optimal')
                m2xm2 = np.einsum('epi,epj->epij', m2, m2, optimize='optimal')

                matriceI = np.eye(3)
                # Projector P such that vecteur_e_pg = projP_e_pg : vecteur_e_pg
                BetaP_x_matriceI = np.einsum('ep,ij->epij', BetaP, matriceI, optimize='optimal')
                gamma1P_x_m1xm1 = np.einsum('ep,epij->epij', gammap[:,:,0], m1xm1, optimize='optimal')
                gamma2P_x_m2xm2 = np.einsum('ep,epij->epij', gammap[:,:,1], m2xm2, optimize='optimal')
                projP = BetaP_x_matriceI + gamma1P_x_m1xm1 + gamma2P_x_m2xm2

                # Projector M such that EpsM = projM : Eps
                BetaM_x_matriceI = np.einsum('ep,ij->epij', BetaM, matriceI, optimize='optimal')
                gamma1M_x_m1xm1 = np.einsum('ep,epij->epij', gammam[:,:,0], m1xm1, optimize='optimal')
                gamma2M_x_m2xm2 = np.einsum('ep,epij->epij', gammam[:,:,1], m2xm2, optimize='optimal')
                projM = BetaM_x_matriceI + gamma1M_x_m1xm1 + gamma2M_x_m2xm2

            tic.Tac("Split", "projP and projM", False)

        elif dim == 3:
            m1, m2, m3 = list_m[0], list_m[1], list_m[2]

            M1, M2, M3 = list_M[0], list_M[1], list_M[2]            

            coef = np.sqrt(2)

            thetap = dvalp.copy()/2
            thetam = dvalm.copy()/2

            funcFiltreComp = lambda vi, vj: vi != vj
            
            elems, pdgs = np.where(funcFiltreComp(val_e_pg[:,:,0], val_e_pg[:,:,1]))
            v1_m_v2 = val_e_pg[elems,pdgs,0]-val_e_pg[elems,pdgs,1]
            thetap[elems, pdgs, 0] = (valp[elems,pdgs,0]-valp[elems,pdgs,1])/(2*v1_m_v2)
            thetam[elems, pdgs, 0] = (valm[elems,pdgs,0]-valm[elems,pdgs,1])/(2*v1_m_v2)

            elems, pdgs = np.where(funcFiltreComp(val_e_pg[:,:,0], val_e_pg[:,:,2]))
            v1_m_v3 = val_e_pg[elems,pdgs,0]-val_e_pg[elems,pdgs,2]
            thetap[elems, pdgs, 1] = (valp[elems,pdgs,0]-valp[elems,pdgs,2])/(2*v1_m_v3)
            thetam[elems, pdgs, 1] = (valm[elems,pdgs,0]-valm[elems,pdgs,2])/(2*v1_m_v3)

            elems, pdgs = np.where(funcFiltreComp(val_e_pg[:,:,1], val_e_pg[:,:,2]))
            v2_m_v3 = val_e_pg[elems,pdgs,1]-val_e_pg[elems,pdgs,2]
            thetap[elems, pdgs, 2] = (valp[elems,pdgs,1]-valp[elems,pdgs,2])/(2*v2_m_v3)
            thetam[elems, pdgs, 2] = (valm[elems,pdgs,1]-valm[elems,pdgs,2])/(2*v2_m_v3)

            tic.Tac("Split", "thetap and thetam", False)

            if useNumba:
                # Much faster (approx. 2x faster)

                G12_ij, G13_ij, G23_ij = Numba_Interface.Get_G12_G13_G23(M1, M2, M3)

                tic.Tac("Split", "Gab", False)

                list_mi = [m1, m2, m3]
                list_Gab = [G12_ij, G13_ij, G23_ij]

                projP, projM = Numba_Interface.Get_projP_projM_3D(dvalp, dvalm, thetap, thetam, list_mi, list_Gab)
            
            else:

                def __Construction_Gij(Ma, Mb):

                    Gij = np.zeros((Ne, nPg, 6, 6))

                    part1 = lambda Ma, Mb: np.einsum('epik,epjl->epijkl', Ma, Mb, optimize='optimal')
                    part2 = lambda Ma, Mb: np.einsum('epil,epjk->epijkl', Ma, Mb, optimize='optimal')

                    Gijkl = part1(Ma, Mb) + part2(Ma, Mb) + part1(Mb, Ma) + part2(Mb, Ma)

                    listI = [0]*6; listI.extend([1]*6); listI.extend([2]*6); listI.extend([1]*6); listI.extend([0]*12)
                    listJ = [0]*6; listJ.extend([1]*6); listJ.extend([2]*18); listJ.extend([1]*6)
                    listK = [0,1,2,1,0,0]*6
                    listL = [0,1,2,2,2,1]*6
                    
                    columns = np.arange(0,6, dtype=int).reshape((1,6)).repeat(6,axis=0).ravel()
                    lines = np.sort(columns)

                    # # build a matrix to check whether the indexes are good or not
                    # ma = np.zeros((6,6), dtype=np.object0)
                    # for lin,col,i,j,k,l in zip(lines, columns, listI, listJ, listK, listL):
                    #     text = f"{i+1}{j+1}{k+1}{l+1}"
                    #     ma[lin,col] = text
                    #     pass

                    Gij[:,:,lines, columns] = Gijkl[:,:,listI,listJ,listK,listL]                    
                    
                    Gij[:,:,:3,3:6] = Gij[:,:,:3,3:6] * coef
                    Gij[:,:,3:6,:3] = Gij[:,:,3:6,:3] * coef
                    Gij[:,:,3:6,3:6] = Gij[:,:,3:6,3:6] * 2

                    return Gij

                G12 = __Construction_Gij(M1, M2)
                G13 = __Construction_Gij(M1, M3)
                G23 = __Construction_Gij(M2, M3)

                tic.Tac("Split", "Gab", False)

                m1xm1 = np.einsum('epi,epj->epij', m1, m1, optimize='optimal')
                m2xm2 = np.einsum('epi,epj->epij', m2, m2, optimize='optimal')
                m3xm3 = np.einsum('epi,epj->epij', m3, m3, optimize='optimal')

                tic.Tac("Split", "mixmi", False)

                func = lambda ep, epij: np.einsum('ep,epij->epij', ep, epij, optimize='optimal')
                # func = lambda ep, epij: ep[:,:,np.newaxis,np.newaxis].repeat(epij.shape[2], axis=2).repeat(epij.shape[3], axis=3) * epij

                projP = func(dvalp[:,:,0], m1xm1) + func(dvalp[:,:,1], m2xm2) + func(dvalp[:,:,2], m3xm3) + func(thetap[:,:,0], G12) + func(thetap[:,:,1], G13) + func(thetap[:,:,2], G23)
                projM = func(dvalm[:,:,0], m1xm1) + func(dvalm[:,:,1], m2xm2) + func(dvalm[:,:,2], m3xm3) + func(thetam[:,:,0], G12) + func(thetam[:,:,1], G13) + func(thetam[:,:,2], G23)

            tic.Tac("Split", "projP and projM", False)

        if verif:
            # Verification of decomposition and orthogonality
            # projector in [1; 1; 1]
            vectorP = np.einsum('epij,epj->epi', projP, vector_e_pg, optimize='optimal')
            vectorM = np.einsum('epij,epj->epi', projM, vector_e_pg, optimize='optimal')
            
            # Decomposition vector_e_pg = vectorP_e_pg + vectorM_e_pg
            decomp = vector_e_pg-(vectorP + vectorM)
            if np.linalg.norm(vector_e_pg) > 0:                
                verifDecomp = np.linalg.norm(decomp)/np.linalg.norm(vector_e_pg)
                assert verifDecomp <= 1e-12, "vector_e_pg != vectorP_e_pg + vectorM_e_pg"

            # Orthogonality
            ortho_vP_vM = np.abs(np.einsum('epi,epi->ep',vectorP, vectorM, optimize='optimal'))
            ortho_vM_vP = np.abs(np.einsum('epi,epi->ep',vectorM, vectorP, optimize='optimal'))
            ortho_v_v = np.abs(np.einsum('epi,epi->ep', vector_e_pg, vector_e_pg, optimize='optimal'))
            if ortho_v_v.min() > 0:
                vertifOrthoEpsPM = np.max(ortho_vP_vM/ortho_v_v)
                assert vertifOrthoEpsPM <= 1e-12
                vertifOrthoEpsMP = np.max(ortho_vM_vP/ortho_v_v)
                assert vertifOrthoEpsMP <= 1e-12
            
        return projP, projM