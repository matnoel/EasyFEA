# Copyright (C) 2021-2024 Université Gustave Eiffel. All rights reserved.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from abc import ABC, abstractmethod, abstractproperty
from typing import Union

# utilities
from .. import np
# others
from ..Geoms import As_Coordinates
from ._utils import (_IModel, ModelType,
                     Reshape_variable, Heterogeneous_Array,
                     Tensor_Product,
                     KelvinMandel_Matrix, Project_Kelvin,
                     Result_in_Strain_or_Stress_field,
                     Get_Pmat, Apply_Pmat)

# ----------------------------------------------
# Elasticity
# ----------------------------------------------

class _Elas(_IModel, ABC):
    """Displacement class model.\n
    Elas_Isot, Elas_IsotTrans and Elas_Anisot inherit from _Elas
    """
    def __init__(self, dim: int, thickness: float, planeStress: bool):
        
        self.__dim = dim

        if dim == 2:
            assert thickness > 0 , "Must be greater than 0"
            self.__thickness = thickness

        self.__planeStress = planeStress if dim == 2 else False
        """2D simplification type"""

        self.useNumba = False

    @property
    def modelType(self) -> ModelType:
        return ModelType.elastic

    @property
    def dim(self) -> int:
        return self.__dim

    @property
    def thickness(self) -> float:
        if self.__dim == 2:
            return self.__thickness
        else:
            return 1.0

    @property
    def planeStress(self) -> bool:
        """The model uses plane stress simplification"""
        return self.__planeStress
    
    @planeStress.setter
    def planeStress(self, value: bool) -> None:
        if isinstance(value, bool):
            if self.__planeStress != value:
                self.Need_Update()
            self.__planeStress = value

    @property
    def simplification(self) -> str:
        """Simplification used for the model"""
        if self.__dim == 2:
            return "Plane Stress" if self.planeStress else "Plane Strain"
        else:
            return "3D"

    @abstractmethod
    def _Update(self) -> None:
        """Update the C and S behavior law"""
        pass

    # Model
    @staticmethod
    def Available_Laws():
        laws = [Elas_Isot, Elas_IsotTrans, Elas_Anisot]
        return laws

    @property
    def coef(self) -> float:
        """Coef linked to kelvin mandel -> sqrt(2)"""
        return np.sqrt(2)

    @property
    def C(self) -> np.ndarray:
        """Behaviour for Lame's law in Kelvin Mandel\n
        In 2D: C -> C: Epsilon = Sigma [Sxx, Syy, sqrt(2)*Sxy]\n
        In 3D: C -> C: Epsilon = Sigma [Sxx, Syy, Szz, sqrt(2)*Syz, sqrt(2)*Sxz, sqrt(2)*Sxy].
        """
        if self.needUpdate:
            self._Update()
            self.Need_Update(False)
        return self.__C.copy()

    @C.setter
    def C(self, array: np.ndarray):
        assert isinstance(array, np.ndarray), "must be an array"
        shape = (3, 3) if self.dim == 2 else (6, 6)
        assert array.shape[-2:] == shape, f"With dim = {self.dim} array must be a {shape} matrix"
        self.__C = array

    @property
    def isHeterogeneous(self) -> bool:
        return len(self.C.shape) > 2

    @property
    def S(self) -> np.ndarray:
        """Behaviour for Hooke's law in Kelvin Mandel\n
        In 2D: S -> S : Sigma = Epsilon [Exx, Eyy, sqrt(2)*Exy]\n
        In 3D: S -> S: Sigma = Epsilon [Exx, Eyy, Ezz, sqrt(2)*Eyz, sqrt(2)*Exz, sqrt(2)*Exy].
        """
        if self.needUpdate:
            self._Update()
            self.Need_Update(False)
        return self.__S.copy()
    
    @S.setter
    def S(self, array: np.ndarray):
        assert isinstance(array, np.ndarray), "must be an array"
        shape = (3, 3) if self.dim == 2 else (6, 6)
        assert array.shape[-2:] == shape, f"With dim = {self.dim} array must be a {shape} matrix"
        self.__S = array

    @abstractmethod
    def Walpole_Decomposition(self) -> tuple[np.ndarray, np.ndarray]:
        """Walpole's decomposition such as C = sum(ci * Ei).\n
        # Use Kelvin mandel notation ! \n
        return ci, Ei"""
        return np.array([]), np.array([])

# ----------------------------------------------
# Isotropic
# ----------------------------------------------

class Elas_Isot(_Elas):

    def __str__(self) -> str:
        text = f"{type(self).__name__}:"
        text += f"\nE = {self.E:.2e}, v = {self.v}"
        if self.__dim == 2:
            text += f"\nplaneStress = {self.planeStress}"
            text += f"\nthickness = {self.thickness:.2e}"
        return text

    def __init__(self, dim: int, E=210000.0, v=0.3, planeStress=True, thickness=1.0):
        """Isotropic elastic material.

        Parameters
        ----------
        dim : int
            Dimension of 2D or 3D simulation
        E : float|np.ndarray, optional
            Young modulus
        v : float|np.ndarray, optional
            Poisson ratio ]-1;0.5]
        planeStress : bool, optional
            Plane Stress, by default True
        thickness : float, optional
            thickness, by default 1.0
        """       

        # Checking values
        assert dim in [2,3], "Must be dimension 2 or 3"
        self.__dim = dim
        
        self.E=E
        self.v=v

        _Elas.__init__(self, dim, thickness, planeStress)

    def _Update(self) -> None:
        C, S = self._Behavior(self.dim)
        self.C = C
        self.S = S

    @property
    def E(self) -> Union[float,np.ndarray]:
        """Young modulus"""
        return self.__E
    
    @E.setter
    def E(self, value):
        self._Test_Sup0(value)
        self.Need_Update()
        self.__E = value

    @property
    def v(self) -> Union[float,np.ndarray]:
        """Poisson coefficient"""
        return self.__v
    
    @v.setter
    def v(self, value: float):
        self._Test_In(value)
        self.Need_Update()
        self.__v = value

    def get_lambda(self):

        E=self.E
        v=self.v
        
        l = E*v/((1+v)*(1-2*v))

        if self.__dim == 2 and self.planeStress:
            l = E*v/(1-v**2)
        
        return l
    
    def get_mu(self):
        """Shear coefficient"""
        
        E=self.E
        v=self.v

        mu = E/(2*(1+v))

        return mu
    
    def get_bulk(self):
        """Bulk modulus"""

        E=self.E
        v=self.v

        mu = self.get_mu()
        l = self.get_lambda()
        
        bulk = l + 2*mu/self.dim        

        return bulk

    def _Behavior(self, dim:int=None):
        """"Builds behavior matrices in kelvin mandel\n
        
        In 2D:
        -----

        C -> C : Epsilon = Sigma [Sxx Syy sqrt(2)*Sxy]\n
        S -> S : Sigma = Epsilon [Exx Eyy sqrt(2)*Exy]

        In 3D:
        -----

        C -> C : Epsilon = Sigma [Sxx Syy Szz sqrt(2)*Syz sqrt(2)*Sxz sqrt(2)*Sxy]\n
        S -> S : Sigma = Epsilon [Exx Eyy Ezz sqrt(2)*Eyz sqrt(2)*Exz sqrt(2)*Exy]

        """

        if dim == None:
            dim = self.__dim
        else:
            assert dim in [2,3]

        E=self.E
        v=self.v

        mu = self.get_mu()
        l = self.get_lambda()

        dtype = object if True in [isinstance(p, np.ndarray) for p in [E, v]] else float

        if dim == 2:

            # Careful here because lambda changes according to 2D simplification.

            cVoigt = np.array([ [l + 2*mu, l, 0],
                                [l, l + 2*mu, 0],
                                [0, 0, mu]], dtype=dtype)

            # if self.contraintesPlanes:
            #     # C = np.array([  [4*(mu+l), 2*l, 0],
            #     #                 [2*l, 4*(mu+l), 0],
            #     #                 [0, 0, 2*mu+l]]) * mu/(2*mu+l)

            #     cVoigt = np.array([ [1, v, 0],
            #                         [v, 1, 0],
            #                         [0, 0, (1-v)/2]]) * E/(1-v**2)
                
            # else:
            #     cVoigt = np.array([ [l + 2*mu, l, 0],
            #                         [l, l + 2*mu, 0],
            #                         [0, 0, mu]])

            #     # C = np.array([  [1, v/(1-v), 0],
            #     #                 [v/(1-v), 1, 0],
            #     #                 [0, 0, (1-2*v)/(2*(1-v))]]) * E*(1-v)/((1+v)*(1-2*v))

        elif dim == 3:
            
            cVoigt = np.array([ [l+2*mu, l, l, 0, 0, 0],
                                [l, l+2*mu, l, 0, 0, 0],
                                [l, l, l+2*mu, 0, 0, 0],
                                [0, 0, 0, mu, 0, 0],
                                [0, 0, 0, 0, mu, 0],
                                [0, 0, 0, 0, 0, mu]], dtype=dtype)
            
        cVoigt = Heterogeneous_Array(cVoigt)
        
        c = KelvinMandel_Matrix(dim, cVoigt)

        s = np.linalg.inv(c)

        return c, s
    
    def Walpole_Decomposition(self) -> tuple[np.ndarray, np.ndarray]:

        c1 = self.get_bulk()
        c2 = self.get_mu()

        Ivect = np.array([1,1,1,0,0,0])
        Isym = np.eye(6)

        E1 = 1/3 * Tensor_Product(Ivect, Ivect)
        E2 = Isym - E1

        if not self.isHeterogeneous:
            C = self.C
            # only test if the material is heterogeneous
            test_C = np.linalg.norm((3*c1*E1  + 2*c2*E2) - C)/np.linalg.norm(C)
            assert test_C <= 1e-12

        ci = np.array([c1, c2])
        Ei = np.array([3*E1, 2*E2])

        return ci, Ei

# ----------------------------------------------
# Transversely isotropic 
# ----------------------------------------------

class Elas_IsotTrans(_Elas):

    def __str__(self) -> str:
        text = f"{type(self).__name__}:"
        text += f"\nEl = {self.El:.2e}, Et = {self.Et:.2e}, Gl = {self.Gl:.2e}"
        text += f"\nvl = {self.vl}, vt = {self.vt}"
        text += f"\naxis_l = {np.array_str(self.axis_l, precision=3)}"
        text += f"\naxis_t = {np.array_str(self.axis_t, precision=3)}"
        if self.__dim == 2:
            text += f"\nplaneStress = {self.planeStress}"
            text += f"\nthickness = {self.thickness:.2e}"
        return text

    def __init__(self, dim: int, El: float, Et: float, Gl: float, vl: float, vt: float, axis_l=[1,0,0], axis_t=[0,1,0], planeStress=True, thickness=1.0):
        """Transverse isotropic elastic material. More details Torquato 2002 13.3.2 (iii) : http://link.springer.com/10.1007/978-1-4757-6355-3

        Parameters
        ----------
        dim : int
            Dimension of 2D or 3D simulation
        El : float
            Longitudinal Young modulus
        Et : float
            Transverse Young modulus
        Gl : float
            Longitudinal shear modulus
        vl : float
            Longitudinal Poisson ratio
        vt : float
            Transverse Poisson ratio
        axis_l : np.ndarray, optional
            Longitudinal axis, by default np.array([1,0,0])
        axis_t : np.ndarray, optional
            Transverse axis, by default np.array([0,1,0])
        planeStress : bool, optional
            Plane Stress, by default True
        thickness : float, optional
            thickness, by default 1.0
        """

        # Checking values
        assert dim in [2,3], "Must be dimension 2 or 3"
        self.__dim = dim

        self.El=El
        self.Et=Et
        self.Gl=Gl
        self.vl=vl
        self.vt=vt

        axis_l = As_Coordinates(axis_l)
        axis_t = As_Coordinates(axis_t)
        assert axis_l.size == 3 and len(axis_l.shape) == 1, 'axis_l must be a 3D vector'
        assert axis_t.size == 3 and len(axis_t.shape) == 1, 'axis_t must be a 3D vector'
        assert axis_l @ axis_t <= 1e-12, 'axis1 and axis2 must be perpendicular'
        self.__axis_l = axis_l
        self.__axis_t = axis_t

        _Elas.__init__(self, dim, thickness, planeStress)

    @property
    def Gt(self) -> Union[float,np.ndarray]:
        """Transverse shear modulus"""
        
        Et = self.Et
        vt = self.vt

        Gt = Et/(2*(1+vt))

        return Gt

    @property
    def El(self) -> Union[float,np.ndarray]:
        """Longitudinal Young modulus"""
        return self.__El

    @El.setter
    def El(self, value: Union[float,np.ndarray]):
        self._Test_Sup0(value)
        self.Need_Update()
        self.__El = value

    @property
    def Et(self) -> Union[float,np.ndarray]:
        """Transverse Young modulus"""
        return self.__Et
    
    @Et.setter
    def Et(self, value: Union[float,np.ndarray]):
        self._Test_Sup0(value)
        self.Need_Update()
        self.__Et = value

    @property
    def Gl(self) -> Union[float,np.ndarray]:
        """Longitudinal shear modulus"""
        return self.__Gl

    @Gl.setter
    def Gl(self, value: Union[float,np.ndarray]):
        self._Test_Sup0(value)
        self.Need_Update()
        self.__Gl = value

    @property
    def vl(self) -> Union[float,np.ndarray]:
        """Longitudinal Poisson ratio"""
        return self.__vl

    @vl.setter
    def vl(self, value: Union[float,np.ndarray]):
        # -1<vt<1
        # -1<vl<0.5
        # Torquato 328
        self._Test_In(value, -1, 1)
        self.Need_Update()
        self.__vl = value
    
    @property
    def vt(self) -> Union[float,np.ndarray]:
        """Transverse Poisson ratio"""
        return self.__vt

    @vt.setter
    def vt(self, value: Union[float,np.ndarray]):
        # -1<vt<1
        # -1<vl<0.5
        # Torquato 328
        self._Test_In(value)
        self.Need_Update()
        self.__vt = value

    @property
    def kt(self) -> Union[float,np.ndarray]:
        # Torquato 2002 13.3.2 (iii)
        El = self.El
        Et = self.Et
        vtt = self.vt
        vtl = self.vl
        kt = El*Et/((2*(1-vtt)*El)-(4*vtl**2*Et))

        return kt
    
    @property
    def axis_l(self) -> np.ndarray:
        """Longitudinal axis"""
        return self.__axis_l.copy()
    
    @property
    def axis_t(self) -> np.ndarray:
        """Transversal axis"""
        return self.__axis_t.copy()
    
    @property
    def _useSameAxis(self) -> bool:
        testAxis_l = np.linalg.norm(self.axis_l-np.array([1,0,0])) <= 1e-12
        testAxis_t = np.linalg.norm(self.axis_t-np.array([0,1,0])) <= 1e-12
        if testAxis_l and testAxis_t:
            return True
        else:
            return False

    def _Update(self) -> None:
        C, S = self._Behavior(self.dim)
        self.C = C
        self.S = S

    def _Behavior(self, dim: int=None, P: np.ndarray=None):
        """"Constructs behavior matrices in kelvin mandel\n
        
        In 2D:
        -----

        C -> C : Epsilon = Sigma [Sxx Syy sqrt(2)*Sxy]\n
        S -> S : Sigma = Epsilon [Exx Eyy sqrt(2)*Exy]

        In 3D:
        -----

        C -> C : Epsilon = Sigma [Sxx Syy Szz sqrt(2)*Syz sqrt(2)*Sxz sqrt(2)*Sxy]\n
        S -> S : Sigma = Epsilon [Exx Eyy Ezz sqrt(2)*Eyz sqrt(2)*Exz sqrt(2)*Exy]

        """

        if dim == None:
            dim = self.__dim
        
        if not isinstance(P, np.ndarray):
            P = Get_Pmat(self.__axis_l, self.__axis_t)

        useSameAxis = self._useSameAxis

        El = self.El
        Et = self.Et
        vt = self.vt
        vl = self.vl
        Gl = self.Gl
        Gt = self.Gt

        kt = self.kt
        
        dtype = object if isinstance(kt, np.ndarray) else float

        # Mandel softness and stiffness matrix in the material base
        # [11, 22, 33, sqrt(2)*23, sqrt(2)*13, sqrt(2)*12]

        material_sM = np.array([[1/El, -vl/El, -vl/El, 0, 0, 0],
                      [-vl/El, 1/Et, -vt/Et, 0, 0, 0],
                      [-vl/El, -vt/Et, 1/Et, 0, 0, 0],
                      [0, 0, 0, 1/(2*Gt), 0, 0],
                      [0, 0, 0, 0, 1/(2*Gl), 0],
                      [0, 0, 0, 0, 0, 1/(2*Gl)]], dtype=dtype)
        
        material_sM = Heterogeneous_Array(material_sM)

        material_cM = np.array([[El+4*vl**2*kt, 2*kt*vl, 2*kt*vl, 0, 0, 0],
                      [2*kt*vl, kt+Gt, kt-Gt, 0, 0, 0],
                      [2*kt*vl, kt-Gt, kt+Gt, 0, 0, 0],
                      [0, 0, 0, 2*Gt, 0, 0],
                      [0, 0, 0, 0, 2*Gl, 0],
                      [0, 0, 0, 0, 0, 2*Gl]], dtype=dtype)
        
        material_cM = Heterogeneous_Array(material_cM)

        # # Verify that C = S^-1#
        # assert np.linalg.norm(material_sM - np.linalg.inv(material_cM)) < 1e-10        
        # assert np.linalg.norm(material_cM - np.linalg.inv(material_sM)) < 1e-10

        # Performs a base change to orient the material in space
        global_sM = Apply_Pmat(P, material_sM)
        global_cM = Apply_Pmat(P, material_cM)
        
        # verification that if the axes do not change, the same behavior law is obtained
        test_diff_c = global_cM - material_cM
        if useSameAxis: assert(np.linalg.norm(test_diff_c)<1e-12)

        # verification that if the axes do not change, the same behavior law is obtained
        test_diff_s = global_sM - material_sM
        if useSameAxis: assert np.linalg.norm(test_diff_s) < 1e-12
        
        c = global_cM
        s = global_sM

        if dim == 2:
            x = np.array([0,1,5])

            shape = c.shape
            
            if self.planeStress == True:
                if len(shape) == 2:
                    s = global_sM[x,:][:,x]
                elif len(shape) == 3:
                    s = global_sM[:,x,:][:,:,x]
                elif len(shape) == 4:
                    s = global_sM[:,:,x,:][:,:,:,x]
                    
                c = np.linalg.inv(s)
            else:                
                if len(shape) == 2:
                    c = global_cM[x,:][:,x]
                elif len(shape) == 3:
                    c = global_cM[:,x,:][:,:,x]
                elif len(shape) == 4:
                    c = global_cM[:,:,x,:][:,:,:,x]
                
                s = np.linalg.inv(c)

                # testS = np.linalg.norm(s-s2)/np.linalg.norm(s2)            
        
        return c, s

    def Walpole_Decomposition(self) -> tuple[np.ndarray, np.ndarray]:

        El = self.El
        Et = self.Et
        Gl = self.Gl
        vl = self.vl
        kt = self.kt
        Gt = self.Gt

        c1 = El + 4*vl**2*kt
        c2 = 2*kt
        c3 = 2*np.sqrt(2)*kt*vl
        c4 = 2*Gt
        c5 = 2*Gl

        n = self.axis_l
        p = Tensor_Product(n,n)
        q = np.eye(3) - p
        
        E1 = Project_Kelvin(Tensor_Product(p,p))
        E2 = Project_Kelvin(1/2 * Tensor_Product(q,q))
        E3 = Project_Kelvin(1/np.sqrt(2) * Tensor_Product(p,q))
        E4 = Project_Kelvin(1/np.sqrt(2) * Tensor_Product(q,p))
        E5 = Project_Kelvin(Tensor_Product(q,q,True) - 1/2*Tensor_Product(q,q))
        I = Project_Kelvin(Tensor_Product(np.eye(3),np.eye(3),True))
        E6 = I - E1 - E2 - E5

        if not self.isHeterogeneous:
            P = Get_Pmat(self.axis_l, self.axis_t)
            C, S = self._Behavior(3, P)
            # only test if the material is heterogeneous
            diff_C = C - (c1*E1 + c2*E2 + c3*(E3+E4) + c4*E5 + c5*E6)
            test_C = np.linalg.norm(diff_C)/np.linalg.norm(C)
            assert test_C <= 1e-12

        ci = np.array([c1, c2, c3, c4, c5])
        Ei = np.array([E1, E2, E3+E4, E5, E6])

        return ci, Ei

# ----------------------------------------------
# Anisotropic
# ----------------------------------------------

class Elas_Anisot(_Elas):
    
    def __str__(self) -> str:
        text = f"\n{type(self).__name__}):"
        text += f"\n{self.C}"
        text += f"\naxis1 = {np.array_str(self.__axis1, precision=3)}"
        text += f"\naxis2 = {np.array_str(self.__axis2, precision=3)}"
        if self.__dim == 2:
            text += f"\nplaneStress = {self.planeStress}"
            text += f"\nthickness = {self.thickness:.2e}"
        return text

    def __init__(self, dim: int, C: np.ndarray, useVoigtNotation:bool, axis1: np.ndarray=(1,0,0), axis2: np.ndarray=(0,1,0), thickness=1.0):
        """Anisotropic elastic material.

        Parameters
        ----------
        dim : int
            dimension
        C : np.ndarray
            stiffness matrix in anisotropy basis
        useVoigtNotation : bool
            behavior law uses voigt notation
        axis1 : np.ndarray, optional
            axis1 vector, by default (1,0,0)
        axis2 : np.ndarray
            axis2 vector, by default (0,1,0)
        thickness: float, optional
            material thickness, by default 1.0

        Returns
        -------
        Elas_Anisot
            Anisotropic behavior law
        """

        # Checking values
        assert dim in [2,3], "Must be dimension 2 or 3"
        self.__dim = dim

        axis1 = As_Coordinates(axis1)
        axis2 = As_Coordinates(axis2)
        assert axis1.size == 3 and len(axis1.shape) == 1, 'axis1 must be a 3D vector'
        assert axis2.size == 3 and len(axis2.shape) == 1, 'axis2 must be a 3D vector'
        assert axis1 @ axis2 <= 1e-12, 'axis1 and axis2 must be perpendicular'
        self.__axis1 = axis1
        self.__axis2 = axis2

        # here planeStress is set to False because we just know the C matrix
        _Elas.__init__(self, dim, thickness, False)

        self.Set_C(C, useVoigtNotation)

    def _Update(self) -> None:
        # doesn't do anything here, because we use Set_C to update the laws.
        return super()._Update()

    def Set_C(self, C: np.ndarray, useVoigtNotation=True, update_S=True):
        """Update C and S behavior law

        Parameters
        ----------
        C : np.ndarray
           Behavior law for Lamé's law
        useVoigtNotation : bool, optional
            Behavior law uses Kevin Mandel's notation, by default True
        update_S : bool, optional
            Updates the compliance matrix, by default True
        """

        self.Need_Update()

        dim = 2 if C.shape[0] == 3 else 3
        
        C_mandelP = self._Behavior(C, useVoigtNotation)
        self.C = C_mandelP
        
        if update_S:
            S_mandelP = np.linalg.inv(C_mandelP)
            self.S = S_mandelP
    
    def _Behavior(self, C: np.ndarray, useVoigtNotation: bool) -> np.ndarray:

        shape = C.shape
        assert (shape[-2], shape[-1]) in [(3,3), (6,6)], 'C must be a (3,3) or (6,6) matrix'        
        dim = 3 if C.shape[-1] == 6 else 2
        testSym = np.linalg.norm(C.T - C)/np.linalg.norm(C)
        assert testSym <= 1e-12, "The matrix is not symmetrical."

        # Application of coef if necessary
        if useVoigtNotation:
            C_mandel = KelvinMandel_Matrix(dim, C)
        else:
            C_mandel = C.copy()

        # set to 3D
        idx = np.array([0,1,5])
        if dim == 2:
            if len(shape)==2:
                C_mandel_global = np.zeros((6,6))
                for i, I in enumerate(idx):
                    for j, J in enumerate(idx):
                        C_mandel_global[I,J] = C_mandel[i,j]
            if len(shape)==3:
                C_mandel_global = np.zeros((shape[0],6,6))
                for i, I in enumerate(idx):
                    for j, J in enumerate(idx):
                        C_mandel_global[:,I,J] = C_mandel[:,i,j]
            elif len(shape)==4:
                C_mandel_global = np.zeros((shape[0],shape[1],6,6))
                for i, I in enumerate(idx):
                    for j, J in enumerate(idx):
                        C_mandel_global[:,:,I,J] = C_mandel[:,:,i,j]
        else:
            C_mandel_global = C
        
        P = Get_Pmat(self.__axis1, self.__axis2)

        C_mandelP_global = Apply_Pmat(P, C_mandel_global)

        if self.__dim == 2:
            if len(shape)==2:
                C_mandelP = C_mandelP_global[idx,:][:,idx]
            if len(shape)==3:
                C_mandelP = C_mandelP_global[:,idx,:][:,:,idx]
            elif len(shape)==4:
                C_mandelP = C_mandelP_global[:,:,idx,:][:,:,:,idx]
            
        else:
            C_mandelP = C_mandelP_global

        return C_mandelP
    
    @property
    def axis1(self) -> np.ndarray:
        """axis1 vector"""
        return self.__axis1.copy()
    
    @property
    def axis2(self) -> np.ndarray:
        """axis2 vector"""
        return self.__axis2.copy()
    
    def Walpole_Decomposition(self) -> tuple[np.ndarray, np.ndarray]:
        return super().Walpole_Decomposition()