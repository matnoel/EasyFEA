"""Module for creating behavior models. Such as elastic models or damage models."""

from abc import ABC, abstractmethod, abstractproperty
from typing import List, Union
from enum import Enum
import numpy as np

from TicTac import Tic
from Mesh import Mesh, GroupElem
import CalcNumba as CalcNumba
import Display as Display
from Geom import Line, Point, normalize_vect

from scipy.linalg import sqrtm

class ModelType(str, Enum):
    """Physical models available"""

    displacement = "displacement"
    damage = "damage"
    thermal = "thermal"
    beam = "beam"

class IModel(ABC):
    """Model interface"""

    @abstractproperty
    def modelType(self) -> ModelType:
        """Model type"""
        pass
    
    @abstractproperty
    def dim(self) -> int:
        """model dimension"""
        pass
    
    @abstractproperty
    def thickness(self) -> float:
        """thickness to be used in the model"""
        pass

    @property
    def useNumba(self) -> bool:
        """Returns whether the model can use numba functions"""
        try:
            return self.__useNumba
        except AttributeError:
            self.__useNumba = False
            return self.__useNumba

    @useNumba.setter
    def useNumba(self, value: bool):
        self.__useNumba = value

    @property
    def needUpdate(self) -> bool:
        """The model needs to be updated"""
        return self.__needUpdate

    def Need_Update(self, value=True):
        """Indicates whether the model needs to be updated"""
        self.__needUpdate = value
    
    @property
    def isHeterogeneous(self) -> bool:
        """Indicates whether the model has heterogeneous parameters"""
        return False
    
    @staticmethod
    def _Test_Sup0(value: Union[float,np.ndarray]):
        errorText = "Must be > 0!"
        if isinstance(value, (float, int)):
            assert value > 0.0, errorText
        if isinstance(value, np.ndarray):
            assert value.min() > 0.0, errorText

    @staticmethod
    def _Test_In(value: Union[float,np.ndarray], bInf=-1, bSup=0.5):
        errorText = f"Must be between ]{bInf};{bSup}["
        if isinstance(value, (float, int)):
            assert value > bInf and value < bSup, errorText
        if isinstance(value, np.ndarray):
            assert value.min() > bInf and value.max() < bSup, errorText

class _Displacement_Model(IModel, ABC):
    """Class of elastic behavior laws
    (Elas_Isot, Elas_IsotTrans, Elas_Anisot ...)
    """
    def __init__(self, dim: int, thickness: float, planeStress: bool):
        
        self.__dim = dim

        if dim == 2:
            assert thickness > 0 , "Must be greater than 0"
            self.__thickness = thickness

        self.__planeStress = planeStress if dim == 2 else False
        """2D simplification type"""

        self.Need_Update()

    @property
    def modelType(self) -> ModelType:
        return ModelType.displacement

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
        try:
            return self.__planeStress
        except AttributeError:
            # do this in case the object was created in old instances
            if isinstance(self, Elas_Isot):
                self.__planeStress = self._Elas_Isot__planeStress
            elif isinstance(self, Elas_IsotTrans):
                self.__planeStress = self._Elas_IsotTrans__planeStress
            elif isinstance(self, Elas_Anisot):
                self.__planeStress = self._Elas_Anisot__planeStress
            return self.__planeStress
    
    @planeStress.setter
    def planeStress(self, value: bool) -> None:
        if isinstance(value, bool):
            if self.__planeStress == value: self.Need_Update()
            self.__planeStress = value

    @property
    def simplification(self) -> str:
        """Simplification used for the model"""
        if self.__dim == 2:
            return "Plane Stress" if self.planeStress else "Plane Strain"
        else:
            return "3D"

    @abstractmethod
    def _Update(self):
        """Update the C and S behavior law"""
        pass

    # Model
    @staticmethod
    def get_behaviorLaws():
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
        self.__C = array

    @property
    def isHeterogeneous(self) -> bool:
        return len(self.__C.shape) > 2

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
        self.__S = array

    @abstractmethod
    def Walpole_Decomposition(self) -> tuple[np.ndarray, np.ndarray]:
        """Walpole's decomposition such as C = sum(ci * Ei).\n
        # Use Kelvin mandel notation ! \n
        return ci, Ei"""
        return np.array([]), np.array([])
    
class Elas_Isot(_Displacement_Model):

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

        _Displacement_Model.__init__(self, dim, thickness, planeStress)

        self._Update()

    def _Update(self):
        try:
            C, S = self._Behavior(self.dim)
        except ValueError:
            raise Exception(str(_erroDim))
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

        E1 = 1/3 * TensorProduct(Ivect, Ivect)
        E2 = Isym - E1

        if not self.isHeterogeneous:
            C = self.C
            # only test if the material is heterogeneous
            test_C = np.linalg.norm((3*c1*E1  + 2*c2*E2) - C)/np.linalg.norm(C)
            assert test_C <= 1e-12

        ci = np.array([c1, c2])
        Ei = np.array([3*E1, 2*E2])

        return ci, Ei

class Elas_IsotTrans(_Displacement_Model):

    def __str__(self) -> str:
        text = f"{type(self).__name__}:"
        text += f"\nEl = {self.El:.2e}, Et = {self.Et:.2e}, Gl = {self.Gl:.2e}"
        text += f"\nvl = {self.vl}, vt = {self.vt}"
        text += f"\naxis_l = {np.array_str(self.__axis_l, precision=3)}"
        text += f"\naxis_t = {np.array_str(self.__axis_t, precision=3)}"
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

        axis_l = np.asarray(axis_l)
        axis_t = np.asarray(axis_t)
        assert axis_l.size == 3 and len(axis_l.shape) == 1, 'axis_l must be a 3D vector'
        assert axis_t.size == 3 and len(axis_t.shape) == 1, 'axis_t must be a 3D vector'
        assert axis_l @ axis_t <= 1e-12, 'axis1 and axis2 must be perpendicular'
        self.__axis_l = axis_l
        self.__axis_t = axis_t

        _Displacement_Model.__init__(self, dim, thickness, planeStress)

        self._Update()

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
        # try:
        #     C, S = self._Behavior(self.dim)
        # except ValueError:
        #     raise Exception(str(_erroDim))
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
        p = TensorProduct(n,n)
        q = np.eye(3) - p
        
        E1 = Project_Kelvin(TensorProduct(p,p))
        E2 = Project_Kelvin(1/2 * TensorProduct(q,q))
        E3 = Project_Kelvin(1/np.sqrt(2) * TensorProduct(p,q))
        E4 = Project_Kelvin(1/np.sqrt(2) * TensorProduct(q,p))
        E5 = Project_Kelvin(TensorProduct(q,q,True) - 1/2*TensorProduct(q,q))
        I = Project_Kelvin(TensorProduct(np.eye(3),np.eye(3),True))
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

class Elas_Anisot(_Displacement_Model):
    
    def __str__(self) -> str:
        text = f"\n{type(self).__name__}):"
        text += f"\n{self.C}"
        text += f"\naxis1 = {np.array_str(self.__axis1, precision=3)}"
        text += f"\naxis2 = {np.array_str(self.__axis2, precision=3)}"
        if self.__dim == 2:
            text += f"\nplaneStress = {self.planeStress}"
            text += f"\nthickness = {self.thickness:.2e}"
        return text

    def __init__(self, dim: int, C: np.ndarray, useVoigtNotation:bool, axis1: np.ndarray=(1,0,0), axis2: np.ndarray=(0,1,0), planeStress=True, thickness=1.0):
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
        planeStress : bool, optional
            2D simplification, by default True
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

        axis1 = np.asarray(axis1)
        axis2 = np.asarray(axis2)
        assert axis1.size == 3 and len(axis1.shape) == 1, 'axis1 must be a 3D vector'
        assert axis2.size == 3 and len(axis2.shape) == 1, 'axis2 must be a 3D vector'
        assert axis1 @ axis2 <= 1e-12, 'axis1 and axis2 must be perpendicular'
        self.__axis1 = axis1
        self.__axis2 = axis2

        _Displacement_Model.__init__(self, dim, thickness, planeStress)

        self.Set_C(C, useVoigtNotation)

    def _Update(self):
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

class _Beam_Model(IModel):

    # Number of beams createds
    __nBeam = -1

    @property
    def modelType(self) -> ModelType:
        return ModelType.beam
    
    @property
    def dim(self):
        return self.__dim
    
    @property
    def thickness(self) -> float:
        return 1.0
    
    @property
    def area(self) -> float:
        """Cross-section area."""
        return self.__section.area
    
    @property
    def Iy(self) -> float:        
        """Squared moment of the cross-section around y-axis.\n
        int_S z^2 dS"""
        # here z is the x axis of the section thats why we use x
        func = lambda x,y,z: x**2
        Iy: float = np.sum([grp.Integrate_e(func).sum()
                            for grp in self.__section.Get_list_groupElem(2)])        
        return Iy
    
    @property
    def Iz(self) -> float:        
        """Squared moment of the cross-section around z-axis.\n
        int_S y^2 dS"""
        func = lambda x,y,z: y**2
        Iz: float = np.sum([grp.Integrate_e(func).sum()
                            for grp in self.__section.Get_list_groupElem(2)])        
        return Iz
    
    @property    
    def J(self) -> float:
        """Polar area moment of inertia (Iy + Iz)."""        
        func = lambda x,y,z: x**2+y**2
        J: float = np.sum([grp.Integrate_e(func).sum()
                            for grp in self.__section.Get_list_groupElem(2)])
        return J

    def __init__(self, dim: int, line: Line, section: Mesh, yAxis: Union[list,tuple,np.ndarray]=(0,1,0)):
        """Creating a beam.

        Parameters
        ----------
        dim : int
            Beam dimension (1D, 2D or 3D)
        line : Line
            Line characterizing the neutral fiber.
        section : Mesh
            Beam cross-section
        yAxis: tuple|list|array, optional
            vertical cross-beam axis, by default (0,1,0)
        """

        _Beam_Model.__nBeam += 1
        self.__name = f"beam{_Beam_Model.__nBeam}"
        
        self.__dim: int = dim
        self.__line: Line = line

        assert section.inDim == 2, 'The cross-beam section must be contained in the (x,y) plane.'
        # make sure that the section is centered in (0,0)
        section.translate(*section.center)
        Iyz = section.groupElem.Integrate_e(lambda x,y,z: y*z).sum()
        assert Iyz <= 1e-12, 'The section must have at least 1 symetry axis'
        self.__section: Mesh = section

        self.yAxis = yAxis

    @property
    def line(self) -> Line:
        """Average fiber line of the beam."""
        return self.__line
    
    @property
    def section(self) -> Mesh:
        """Beam cross-section in (x,y) plane"""
        return self.__section
    
    @property
    def xAxis(self) -> np.ndarray:
        """Perpendicular cross-beam axis (fiber)."""
        return self.__line.unitVector
    
    @property
    def yAxis(self) -> np.ndarray:
        """Vertical cross-beam axis."""
        return self.__yAxis.copy()
    
    @yAxis.setter
    def yAxis(self, value):

        # set y axis
        xAxis = self.xAxis
        yAxis = normalize_vect(Point._getCoord(value))

        # check that the yaxis is not colinear to the fiber axis
        crossProd = np.cross(xAxis, yAxis)
        if np.linalg.norm(crossProd) <= 1e-12:
            # Choose x-axis
            yAxis = normalize_vect(np.cross([0,0,1], xAxis))
            print(f"The beam's vertical axis has been selected incorrectly (collinear with the beam x-axis).\nAxis {np.array_str(yAxis, precision=3)} has been assigned for {self.name}.")
        else:
            # get the horizontal direction of the beam
            zAxis = normalize_vect(np.cross(xAxis, yAxis))
            # then make sure that x,y,z are orthogonal
            yAxis = normalize_vect(np.cross(zAxis, xAxis))

        self.__yAxis: np.ndarray = yAxis
    
    @property
    def name(self) -> str:
        """Beam name"""
        return self.__name

    @property
    def dof_n(self) -> int:
        """Degrees of freedom per node
        1D -> [u1, . . . , un]\n
        2D -> [u1, v1, rz1, . . ., un, vn, rzn]\n
        3D -> [u1, v1, w1, rx1, ry1, rz1, . . ., u2, v2, w2, rx2, ry2, rz2]"""
        if self.__dim == 1:
            return 1 # u
        elif self.__dim == 2:
            return 3 # u v rz
        elif self.__dim == 3:
            return 6 # u v w rx ry rz
        return self.__dim
    
    @abstractmethod
    def Get_D(self) -> np.ndarray:
        """Returns a matrix characterizing the beam's stiffness behavior"""
        return
    
    def __str__(self) -> str:
        text = ""
        text += f"\n{self.name}:"        
        text += f"\n  area = {self.__section.area:.2}, Iz = {self.__section.Iz:.2}, Iy = {self.__section.Iy:.2}, J = {self.__section.J:.2}"

        return text
    
    def _Calc_P(self) -> np.ndarray:
        """P matrix use to transform beam coordinates to global coordinates.\n
        [ix, jx, kx\n
        iy, jy, ky\n
        iz, jz, kz]\n
        coordo(x,y,z) = P • coordo(i,j,k)
        """
        line = self.line
        
        i = line.get_unitVector(line.pt1, line.pt2)
        j = self.yAxis
        k = normalize_vect(np.cross(i,j))

        J = np.array([i,j,k]).T
        return J
    
class Beam_Elas_Isot(_Beam_Model):

    def __init__(self, dim: int, line: Line, section: Mesh, E: float, v:float, yAxis: Union[list,tuple,np.ndarray]=(0,1,0)):
        """Construction of an isotropic elastic beam.

        Parameters
        ----------
        dim : int
            Beam dimension (1D, 2D or 3D)
        line : Line
            Line characterizing the neutral fiber.
        section : Mesh
            Beam cross-section
        E : float
            Young module
        v : float
            Poisson ratio
        yAxis: tuple|list|array, optional
            vertical cross-beam axis, by default (0,1,0)
        """

        _Beam_Model.__init__(self, dim, line, section, yAxis)
        
        IModel._Test_Sup0(E)        
        self.__E = E

        IModel._Test_In(v, -1, 0.5)
        self.__v = v

    @property
    def E(self) -> float:
        """Young modulus"""
        return self.__E

    @property
    def v(self) -> float:
        """Poisson ratio"""
        return self.__v
    
    def Get_D(self) -> np.ndarray:

        dim = self.dim
        section = self.section
        A = section.area
        Iy = self.Iy
        Iz = self.Iz
        J = self.J
        
        E = self.__E
        v = self.__v
        
        if dim == 1:
            # u = [u1, . . . , un]
            D = np.diag([E*A])
        elif dim == 2:
            # u = [u1, v1, rz1, . . . , un, vn, rzn]
            D = np.diag([E*A, E*Iz])
        elif dim == 3:
            # u = [u1, v1, w1, rx1, ry1 rz1, . . . , un, vn, wn, rxn, ryn rzn]
            mu = E/(2*(1+v))
            D = np.diag([E*A, mu*J, E*Iy, E*Iz])

        return D

class Beam_Structure(IModel):

    @property
    def modelType(self) -> ModelType:
        return ModelType.beam
    
    @property
    def dim(self) -> int:
        """Model dimensions  \n
        1D -> tension compression \n
        2D -> tension compression + bending + flexion \n
        3D -> all
        """
        return self.__dim
    
    @property
    def thickness(self) -> float:
        """The beam structure can have several beams and therefore different sections.
        You need to look at the section of the beam you are interested in."""
        return None
    
    @property
    def areas(self) -> list[float]:
        """beams areas"""
        return [beam.area for beam in self.__listBeam]

    def __init__(self, listBeam: list[_Beam_Model]) -> None:
        """Construct a beam structure

        Parameters
        ----------
        listBeam : list[_Beam_Model]
            Beam list
        """

        dims = [beam.dim for beam in listBeam]        

        assert np.unique(dims, return_counts=True)[1] == len(listBeam), "The structure must use identical beams dimensions."

        self.__dim: int = dims[0]

        self.__listBeam: list[_Beam_Model] = listBeam

        self.__dof_n = listBeam[0].dof_n

    @property
    def listBeam(self) -> list[_Beam_Model]:
        return self.__listBeam
    
    @property
    def nBeam(self) -> int:
        """Number of beams in the structure"""
        return len(self.__listBeam)
    
    @property
    def dof_n(self) -> int:
        """Degrees of freedom per node
        1D -> [u1, . . . , un]\n
        2D -> [u1, v1, rz1, . . ., un, vn, rzn]\n
        3D -> [u1, v1, w1, rx1, ry1, rz1, . . ., u2, v2, w2, rx2, ry2, rz2]
        """
        return self.__dof_n        

    def Calc_D_e_pg(self, groupElem: GroupElem, matrixType: str) -> np.ndarray:

        if groupElem.dim != 1: return

        listBeam = self.__listBeam
        list_D = [beam.Get_D() for beam in listBeam]

        Ne = groupElem.Ne
        nPg = groupElem.Get_gauss(matrixType).nPg
        # Construction of D_e_pg :
        D_e_pg = np.zeros((Ne, nPg, list_D[0].shape[0], list_D[0].shape[0]))                
        
        # For each beam, we will construct the law of behavior on the associated nodes.
        for beam, D in zip(listBeam, list_D):
            
            # recovers elements
            elems = groupElem.Get_Elements_Tag(beam.name)
            D_e_pg[elems] = D

        return D_e_pg
    
    def Get_axis_e(self, groupElem: GroupElem) -> tuple[np.ndarray, np.ndarray]:
        """Get the fiber and cross bar vertical axis on every elements.\n
        return xAxis_e, yAxis_e"""

        if groupElem.dim != 1: return

        beams = self.__listBeam

        Ne = groupElem.Ne

        xAxis_e = np.zeros((Ne,3), dtype=float)
        yAxis_e = np.zeros((Ne,3), dtype=float)

        for beam in beams:            
            elems = groupElem.Get_Elements_Tag(beam.name)
            xAxis_e[elems] = beam.xAxis
            yAxis_e[elems] = beam.yAxis

        return xAxis_e, yAxis_e

class PhaseField_Model(IModel):

    class RegularizationType(str, Enum):
        """Crack regularization"""
        AT1 = "AT1"
        AT2 = "AT2"

    class SplitType(str, Enum):
        """Usable splits"""

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

    # Phase field
    __splits_Isot = [SplitType.Amor, SplitType.Miehe, SplitType.Stress]
    __split_Anisot = [SplitType.Bourdin, SplitType.He, SplitType.Zhang,
                    SplitType.AnisotStrain, SplitType.AnisotStrain_PM, SplitType.AnisotStrain_MP, SplitType.AnisotStrain_NoCross,
                    SplitType.AnisotStress, SplitType.AnisotStress_PM, SplitType.AnisotStress_MP, SplitType.AnisotStress_NoCross]
    
    class SolverType(str, Enum):
        """Solver used to manage crack irreversibility"""
        History = "History"
        HistoryDamage = "HistoryDamage"
        BoundConstrain = "BoundConstrain"


    def __init__(self, material: _Displacement_Model, split: SplitType, regularization: RegularizationType, Gc: Union[float,np.ndarray], l0: Union[float,np.ndarray], solver=SolverType.History, A=None):
        """Creation of a gradient damage model

        Parameters
        ----------
        material : _Displacement_Model
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
    
        assert isinstance(material, _Displacement_Model), "Must be a model of displacement"
        self.__material = material

        assert split in PhaseField_Model.get_splits(), f"Must be included in {PhaseField_Model.get_splits()}"
        if not isinstance(material, Elas_Isot):
            assert not split in PhaseField_Model.__splits_Isot, "These splits are only implemented for Elas_Isot"
        self.__split =  split
        """Split of elastic energy density"""
        
        assert regularization in PhaseField_Model.get_regularisations(), f"Must be included in {PhaseField_Model.get_regularisations()}"
        self.__regularization = regularization
        """Crack regularization model ["AT1", "AT2"]"""
        
        self.Gc = Gc

        assert l0 > 0, "Must be greater than 0"
        self.__l0 = l0
        """Half crack width"""

        self.__solver = solver
        """Solver used to manage crack irreversibility"""

        if not isinstance(A, np.ndarray):
            self.__A = np.eye(self.dim)
        else:
            dim = self.dim
            assert A.shape[-2] == dim and A.shape[-1] == dim, "Wrong dimension"
            self.__A = A

        self.__useNumba = True
        """Whether or not to use numba functions"""

        self.Need_Split_Update()

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
        text += f'\nregularisation : {self.__regularization}'
        text += f'\nGc : {self.__Gc:.4e}'
        text += f'\nl0 : {self.__l0:.4e}'
        return text
    
    @property
    def isHeterogeneous(self) -> bool:
        return isinstance(self.Gc, np.ndarray)
    
    @staticmethod
    def get_splits() -> List[SplitType]:
        """splits available"""
        return list(PhaseField_Model.SplitType)    
    
    @staticmethod
    def get_regularisations() -> List[RegularizationType]:
        """regularizations available"""
        __regularizations = list(PhaseField_Model.RegularizationType)
        return __regularizations    

    @staticmethod
    def get_solvers() -> List[SolverType]:
        """Available solvers used to manage crack irreversibility"""
        __solveurs = list(PhaseField_Model.SolverType)
        return __solveurs

    @property
    def k(self) -> float:
        """diffusion therm"""

        Gc = self.__Gc
        l0 = self.__l0

        k = Gc * l0

        if self.__regularization == PhaseField_Model.RegularizationType.AT1:
            k = 3/4 * k

        return k

    def get_r_e_pg(self, PsiP_e_pg: np.ndarray) -> np.ndarray:
        """reaction therm"""

        Gc = Reshape_variable(self.__Gc, PsiP_e_pg.shape[0], PsiP_e_pg.shape[1])
        
        l0 = self.__l0        
        r = 2 * PsiP_e_pg

        if self.__regularization == PhaseField_Model.RegularizationType.AT2:
            r = r + (Gc/l0)
        
        return r

    def get_f_e_pg(self, PsiP_e_pg: np.ndarray) -> np.ndarray:
        """source therm"""

        Gc = Reshape_variable(self.__Gc, PsiP_e_pg.shape[0], PsiP_e_pg.shape[1])
        l0 = self.__l0

        f = 2 * PsiP_e_pg

        if self.__regularization == PhaseField_Model.RegularizationType.AT1:
            f = f - ( (3*Gc) / (8*l0) )            
            absF = np.abs(f)
            f = (f+absF)/2
        
        return f

    def get_g_e_pg(self, d_n: np.ndarray, mesh: Mesh, matrixType: str, k_residu=1e-12) -> np.ndarray:
        """Degradation function"""

        d_e_n = mesh.Locates_sol_e(d_n)
        Nd_pg = mesh.Get_N_pg(matrixType)

        d_e_pg = np.einsum('pij,ej->ep', Nd_pg, d_e_n, optimize='optimal')        

        if self.__regularization in PhaseField_Model.get_regularisations():
            g_e_pg = (1-d_e_pg)**2 + k_residu
        else:
            raise Exception("Not implemented")

        assert mesh.Ne == g_e_pg.shape[0]
        assert mesh.Get_nPg(matrixType) == g_e_pg.shape[1]
        
        return g_e_pg
    
    @property
    def A(self) -> np.ndarray:
        """Matrix characterizing the direction of model anisotropy for crack energy"""
        return self.__A

    @property
    def split(self) -> str:
        """Split of elastic energy density"""
        return self.__split

    @property
    def regularization(self) -> str:
        """Crack regularization model ["AT1", "AT2"]"""
        return self.__regularization
    
    @property
    def material(self) -> _Displacement_Model:
        """displacement model"""
        return self.__material

    @property
    def solver(self):
        """Solver used to manage crack irreversibility"""
        return self.__solver

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
    def l0(self):
        """Half crack width"""
        return self.__l0

    @property
    def c0(self):
        """Scaling parameter for accurate dissipation of crack energy"""
        if self.__regularization == PhaseField_Model.RegularizationType.AT1:
            c0 = 8/3
        elif self.__regularization == PhaseField_Model.RegularizationType.AT2:
            c0 = 2
        return c0
    
    @property
    def useNumba(self) -> bool:
        return self.__useNumba
    
    @useNumba.setter
    def useNumba(self, val: bool):
        self.__useNumba = val
            
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

        tic.Tac("Matrix", "psiP_e_pg et psiM_e_pg", False)

        return psiP_e_pg, psiM_e_pg

    def Calc_Sigma_e_pg(self, Epsilon_e_pg: np.ndarray):
        """Calcul la contrainte en fonction de la deformation et du split\n
        Ici on calcul :\n
        SigmaP_e_pg = cP_e_pg * Epsilon_e_pg \n
        SigmaM_e_pg = cM_e_pg * Epsilon_e_pg

        Parameters
        ----------
        Epsilon_e_pg : np.ndarray
            deformations stockées aux elements et Points de Gauss

        Returns
        -------
        np.ndarray
            SigmaP_e_pg, SigmaM_e_pg : les contraintes stockées aux elements et Points de Gauss
        """       

        Ne = Epsilon_e_pg.shape[0]
        nPg = Epsilon_e_pg.shape[1]
        dim = Epsilon_e_pg.shape[2]
        
        cP_e_pg, cM_e_pg = self.Calc_C(Epsilon_e_pg)

        tic = Tic()
        
        Epsilon_e_pg = Epsilon_e_pg.reshape((Ne,nPg,dim,1))

        SigmaP_e_pg = np.reshape(cP_e_pg @ Epsilon_e_pg, (Ne,nPg,-1))
        SigmaM_e_pg = np.reshape(cM_e_pg @ Epsilon_e_pg, (Ne,nPg,-1))

        tic.Tac("Matrix", "SigmaP_e_pg et SigmaM_e_pg", False)

        return SigmaP_e_pg, SigmaM_e_pg

    def Need_Split_Update(self):
        """Initialize the dictionary that stores the decomposition of the behavior law"""
        self.__dict_cP_e_pg_And_cM_e_pg = {}
    
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

        # Here we make sure that we only go through 1 iteration to avoid making the calculations several times.
        # As it happens, this doesn't work - the damage doesn't evolve
        # Here we run 2 iterations
        # Once to calculate the energy (psiP), i.e. to build a mass matrix, and once to calculate K_u, i.e. a rigi matrix, so we have to go through it twice

        Ne = Epsilon_e_pg.shape[0]
        nPg = Epsilon_e_pg.shape[1]

        key = f"({Ne}, {nPg})"

        if key in self.__dict_cP_e_pg_And_cM_e_pg:
            # If the key is filled in, the stored solution is retrieved

            cP_e_pg = self.__dict_cP_e_pg_And_cM_e_pg[key][0]
            cM_e_pg = self.__dict_cP_e_pg_And_cM_e_pg[key][1]
        
        else:

            if self.__split == PhaseField_Model.SplitType.Bourdin:
                cP_e_pg, cM_e_pg = self.__Split_Bourdin(Ne, nPg)

            elif self.__split == PhaseField_Model.SplitType.Amor:
                cP_e_pg, cM_e_pg = self.__Split_Amor(Epsilon_e_pg)

            elif self.__split == PhaseField_Model.SplitType.Miehe or "Strain" in self.__split:
                cP_e_pg, cM_e_pg = self.__Split_Miehe(Epsilon_e_pg, verif=verif)
            
            elif self.__split == PhaseField_Model.SplitType.Zhang or "Stress" in self.__split:
                cP_e_pg, cM_e_pg = self.__Split_Stress(Epsilon_e_pg, verif=verif)

            elif self.__split == PhaseField_Model.SplitType.He:
                cP_e_pg, cM_e_pg = self.__Split_He(Epsilon_e_pg, verif=verif)
            
            else: 
                raise Exception("Split inconnue")

            self.__dict_cP_e_pg_And_cM_e_pg[key] = (cP_e_pg, cM_e_pg)

        return cP_e_pg, cM_e_pg

    def __Split_Bourdin(self, Ne: int, nPg: int):
        """[Bourdin 2000] DOI : 10.1016/S0022-5096(99)00028-9"""

        tic = Tic()
        c = self.__material.C
        c_e_pg = Reshape_variable(c, Ne, nPg)

        cP_e_pg = c_e_pg
        cM_e_pg = np.zeros_like(cP_e_pg)
        tic.Tac("Split",f"cP_e_pg et cM_e_pg", False)

        return cP_e_pg, cM_e_pg

    def __Split_Amor(self, Epsilon_e_pg: np.ndarray):
        """[Amor 2009] DOI : 10.1016/j.jmps.2009.04.011"""

        assert isinstance(self.__material, Elas_Isot), f"Implemented only for Elas_Isot material"
        
        tic = Tic()
        
        material = self.__material
        
        Ne = Epsilon_e_pg.shape[0]
        nPg = Epsilon_e_pg.shape[1]

        bulk = material.get_bulk()
        mu = material.get_mu()

        Rp_e_pg, Rm_e_pg = self.__Rp_Rm(Epsilon_e_pg)

        dim = material.dim

        if dim == 2:
            Ivect = np.array([1,1,0]).reshape((3,1))
            taille = 3
        else:
            Ivect = np.array([1,1,1,0,0,0]).reshape((6,1))
            taille = 6

        IxI = np.array(Ivect.dot(Ivect.T))

        spherP_e_pg = np.einsum('ep,ij->epij', Rp_e_pg, IxI, optimize='optimal')
        spherM_e_pg = np.einsum('ep,ij->epij', Rm_e_pg, IxI, optimize='optimal')

        # Deviatoric projector
        Pdev = np.eye(taille) - 1/dim * IxI

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

        tic.Tac("Split",f"cP_e_pg et cM_e_pg", False)

        return cP_e_pg, cM_e_pg

    def __Rp_Rm(self, vecteur_e_pg: np.ndarray):
        """Returns Rp_e_pg, Rm_e_pg"""

        Ne = vecteur_e_pg.shape[0]
        nPg = vecteur_e_pg.shape[1]

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
        useNumba = self.__useNumba

        projP_e_pg, projM_e_pg = self.__Spectral_Decomposition(Epsilon_e_pg, verif)

        Ne = Epsilon_e_pg.shape[0]
        nPg = Epsilon_e_pg.shape[1]

        isHeterogene = self.__material.isHeterogeneous

        tic = Tic()

        if self.__split == PhaseField_Model.SplitType.Miehe:
            
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
                Cpp, Cpm, Cmp, Cmm = CalcNumba.Get_Anisot_C(projP_e_pg, c, projM_e_pg)

            else:
                # Here we don't use einsum, otherwise it's much longer
                c_e_pg = Reshape_variable(c, Ne, nPg)

                pc = np.transpose(projP_e_pg, [0,1,3,2]) @ c_e_pg
                mc = np.transpose(projM_e_pg, [0,1,3,2]) @ c_e_pg
                
                Cpp = pc @ projP_e_pg
                Cpm = pc @ projM_e_pg
                Cmm = mc @ projM_e_pg
                Cmp = mc @ projP_e_pg
            
            if self.__split == PhaseField_Model.SplitType.AnisotStrain:

                cP_e_pg = Cpp + Cpm + Cmp
                cM_e_pg = Cmm 

            elif self.__split == PhaseField_Model.SplitType.AnisotStrain_PM:
                
                cP_e_pg = Cpp + Cpm
                cM_e_pg = Cmm + Cmp

            elif self.__split == PhaseField_Model.SplitType.AnisotStrain_MP:
                
                cP_e_pg = Cpp + Cmp
                cM_e_pg = Cmm + Cpm

            elif self.__split == PhaseField_Model.SplitType.AnisotStrain_NoCross:
                
                cP_e_pg = Cpp
                cM_e_pg = Cmm + Cpm + Cmp
            
        else:
            raise Exception("Unknown split")

        tic.Tac("Split",f"cP_e_pg et cM_e_pg", False)

        return cP_e_pg, cM_e_pg
    
    def __Split_Stress(self, Epsilon_e_pg: np.ndarray, verif=False):
        """Construct Cp and Cm for the split in stress"""

        # Recovers stresses        
        material = self.__material
        c = material.C

        shape_c = len(c.shape)

        Ne = Epsilon_e_pg.shape[0]
        nPg = Epsilon_e_pg.shape[1]

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

        if self.__split == PhaseField_Model.SplitType.Stress:
        
            assert isinstance(material, Elas_Isot)

            E = material.E
            v = material.v

            c = material.C

            dim = self.dim

            # Calcul Rp et Rm
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
            
            useNumba = self.__useNumba
            if useNumba and not isHeterogene:
                # Faster
                cP_e_pg, cM_e_pg = CalcNumba.Get_Cp_Cm_Stress(c, sP_e_pg, sM_e_pg)
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
        
        elif self.__split == PhaseField_Model.SplitType.Zhang or "Stress" in self.__split:
            
            Cp_e_pg = np.einsum(f'epij,{indices}jk->epik', projP_e_pg, c, optimize='optimal')
            Cm_e_pg = np.einsum(f'epij,{indices}jk->epik', projM_e_pg, c, optimize='optimal')

            if self.__split == PhaseField_Model.SplitType.Zhang:
                # [Zhang 2020] DOI : 10.1016/j.cma.2019.112643
                cP_e_pg = Cp_e_pg
                cM_e_pg = Cm_e_pg
            
            else:
                # Builds Cp and Cm
                S = material.S
                if self.__useNumba and not isHeterogene:
                    # Faster
                    Cpp, Cpm, Cmp, Cmm = CalcNumba.Get_Anisot_C(Cp_e_pg, S, Cm_e_pg)
                else:
                    # Here we don't use einsum, otherwise it's much longer
                    s_e_pg = Reshape_variable(S, Ne, nPg)

                    ps = np.transpose(Cp_e_pg, [0,1,3,2]) @ s_e_pg
                    ms = np.transpose(Cm_e_pg, [0,1,3,2]) @ s_e_pg
                    
                    Cpp = ps @ Cp_e_pg
                    Cpm = ps @ Cm_e_pg
                    Cmm = ms @ Cm_e_pg
                    Cmp = ms @ Cp_e_pg
                
                if self.__split == PhaseField_Model.SplitType.AnisotStress:

                    cP_e_pg = Cpp + Cpm + Cmp
                    cM_e_pg = Cmm

                elif self.__split == PhaseField_Model.SplitType.AnisotStress_PM:
                    
                    cP_e_pg = Cpp + Cpm
                    cM_e_pg = Cmm + Cmp

                elif self.__split == PhaseField_Model.SplitType.AnisotStress_MP:
                    
                    cP_e_pg = Cpp + Cmp
                    cM_e_pg = Cmm + Cpm

                elif self.__split == PhaseField_Model.SplitType.AnisotStress_NoCross:
                    
                    cP_e_pg = Cpp
                    cM_e_pg = Cmm + Cpm + Cmp
            
                else:
                    raise Exception("Unknown split")

        tic.Tac("Split",f"cP_e_pg et cM_e_pg", False)

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

        # On calcule les nouveaux vecteurs
        Epsilont_e_pg = np.einsum('ij,epj->epi', sqrtC, Epsilon_e_pg, optimize='optimal')

        # On calcule les projecteurs
        projPt_e_pg, projMt_e_pg = self.__Spectral_Decomposition(Epsilont_e_pg, verif)

        tic = Tic()        

        projPt_e_pg_x_sqrtC = np.einsum('epij,jk->epik', projPt_e_pg, sqrtC, optimize='optimal')
        projMt_e_pg_x_sqrtC = np.einsum('epij,jk->epik', projMt_e_pg, sqrtC, optimize='optimal')
        
        projP_e_pg = np.einsum('ij,epjk->epik', inv_sqrtC, projPt_e_pg_x_sqrtC, optimize='optimal')
        projM_e_pg = np.einsum('ij,epjk->epik', inv_sqrtC, projMt_e_pg_x_sqrtC, optimize='optimal')

        projPT_e_pg =  np.transpose(projP_e_pg, (0,1,3,2))
        projMT_e_pg = np.transpose(projM_e_pg, (0,1,3,2))

        cP_e_pg = np.einsum('epij,jk,epkl->epil', projPT_e_pg, C, projP_e_pg, optimize='optimal')
        cM_e_pg = np.einsum('epij,jk,epkl->epil', projMT_e_pg, C, projM_e_pg, optimize='optimal')

        # projP_e_pg = inv_sqrtC @ (projPt_e_pg @ sqrtC)
        # projM_e_pg = inv_sqrtC @ (projMt_e_pg @ sqrtC)
        
        # projPT_e_pg =  np.transpose(projP_e_pg, (0,1,3,2))
        # projMT_e_pg = np.transpose(projM_e_pg, (0,1,3,2))

        # cP_e_pg = projPT_e_pg @ C @ projP_e_pg
        # cM_e_pg = projMT_e_pg @ C @ projM_e_pg


        tic.Tac("Split",f"cP_e_pg et cM_e_pg", False)

        if verif:
            vecteur_e_pg = Epsilon_e_pg.copy()
            mat = C.copy()    

            # Verification de la décomposition et de l'orthogonalité            
            vecteurP = np.einsum('epij,epj->epi', projP_e_pg, vecteur_e_pg, optimize='optimal')
            vecteurM = np.einsum('epij,epj->epi', projM_e_pg, vecteur_e_pg, optimize='optimal')
            
            # Et+:Et- = 0 deja dans vérifié dans decomp spec
            
            # Décomposition vecteur_e_pg = vecteurP_e_pg + vecteurM_e_pg
            decomp = vecteur_e_pg-(vecteurP + vecteurM)
            if np.linalg.norm(vecteur_e_pg) > 0:
                verifDecomp = np.linalg.norm(decomp)/np.linalg.norm(vecteur_e_pg)
                assert verifDecomp < 1e-12

            # Orthogonalité E+:C:E-
            ortho_vP_vM = np.abs(np.einsum('epi,ij,epj->ep',vecteurP, mat, vecteurM, optimize='optimal'))
            ortho_vM_vP = np.abs(np.einsum('epi,ij,epj->ep',vecteurM, mat, vecteurP, optimize='optimal'))
            ortho_v_v = np.abs(np.einsum('epi,ij,epj->ep', vecteur_e_pg, mat, vecteur_e_pg, optimize='optimal'))
            if ortho_v_v.min() > 0:
                vertifOrthoEpsPM = np.max(ortho_vP_vM/ortho_v_v)
                assert vertifOrthoEpsPM < 1e-12
                vertifOrthoEpsMP = np.max(ortho_vM_vP/ortho_v_v)
                assert vertifOrthoEpsMP < 1e-12

        return cP_e_pg, cM_e_pg

    def __Eigen_values_vectors_projectors(self, vecteur_e_pg: np.ndarray, verif=False) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:

        dim = self.__material.dim

        coef = self.__material.coef
        Ne = vecteur_e_pg.shape[0]
        nPg = vecteur_e_pg.shape[1]

        tic = Tic()

        # Reconstructs the strain tensor [e,pg,dim,dim]
        matrix_e_pg = np.zeros((Ne,nPg,dim,dim))
        for d in range(dim):
            matrix_e_pg[:,:,d,d] = vecteur_e_pg[:,:,d]
        if dim == 2:
            # [x, y, xy]
            # xy
            matrix_e_pg[:,:,0,1] = vecteur_e_pg[:,:,2]/coef
            matrix_e_pg[:,:,1,0] = vecteur_e_pg[:,:,2]/coef
        else:
            # [x, y, z, yz, xz, xy]
            # yz
            matrix_e_pg[:,:,1,2] = vecteur_e_pg[:,:,3]/coef
            matrix_e_pg[:,:,2,1] = vecteur_e_pg[:,:,3]/coef
            # xz
            matrix_e_pg[:,:,0,2] = vecteur_e_pg[:,:,4]/coef
            matrix_e_pg[:,:,2,0] = vecteur_e_pg[:,:,4]/coef
            # xy
            matrix_e_pg[:,:,0,1] = vecteur_e_pg[:,:,5]/coef
            matrix_e_pg[:,:,1,0] = vecteur_e_pg[:,:,5]/coef

        tic.Tac("Split", "vecteur_e_pg -> matrice_e_pg", False)

        # trace_e_pg = np.trace(matrice_e_pg, axis1=2, axis2=3)
        trace_e_pg = np.einsum('epii->ep', matrix_e_pg, optimize='optimal')

        if self.dim == 2:
            # invariants of the strain tensor [e,pg]

            a_e_pg = matrix_e_pg[:,:,0,0]
            b_e_pg = matrix_e_pg[:,:,0,1]
            c_e_pg = matrix_e_pg[:,:,1,0]
            d_e_pg = matrix_e_pg[:,:,1,1]
            determinant_e_pg = (a_e_pg*d_e_pg)-(c_e_pg*b_e_pg)

            tic.Tac("Split", "Invariants", False)

            # Eigenvalue calculations [e,pg]
            delta = trace_e_pg**2 - (4*determinant_e_pg)
            val_e_pg = np.zeros((Ne,nPg,2))
            val_e_pg[:,:,0] = (trace_e_pg - np.sqrt(delta))/2
            val_e_pg[:,:,1] = (trace_e_pg + np.sqrt(delta))/2

            tic.Tac("Split", "Eigenvalues", False)
            
            # Constants for calculating m1 = (matrice_e_pg - v2*I)/(v1-v2)
            v2I = np.einsum('ep,ij->epij', val_e_pg[:,:,1], np.eye(2), optimize='optimal')
            v1_m_v2 = val_e_pg[:,:,0] - val_e_pg[:,:,1]
            
            # element identification and gauss points where vp1 != vp2
            # elements, pdgs = np.where(v1_m_v2 != 0)
            elems, pdgs = np.where(val_e_pg[:,:,0] != val_e_pg[:,:,1])
            
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
                
                val_e_pg = valnum

                tic.Tac("Split", "Eigenvalues and eigenprojectors", False)

            elif version == 'invariants':

                # [Q.-C. He Closed-form coordinate-free]

                a11_e_pg = matrix_e_pg[:,:,0,0]; a12_e_pg = matrix_e_pg[:,:,0,1]; a13_e_pg = matrix_e_pg[:,:,0,2]
                a21_e_pg = matrix_e_pg[:,:,1,0]; a22_e_pg = matrix_e_pg[:,:,1,1]; a23_e_pg = matrix_e_pg[:,:,1,2]
                a31_e_pg = matrix_e_pg[:,:,2,0]; a32_e_pg = matrix_e_pg[:,:,2,1]; a33_e_pg = matrix_e_pg[:,:,2,2]

                determinant_e_pg = a11_e_pg * ((a22_e_pg*a33_e_pg)-(a32_e_pg*a23_e_pg)) - a12_e_pg * ((a21_e_pg*a33_e_pg)-(a31_e_pg*a23_e_pg)) + a13_e_pg * ((a21_e_pg*a32_e_pg)-(a31_e_pg*a22_e_pg))            

                # Invariants
                I1_e_pg = trace_e_pg
                # mat_mat = np.einsum('epij,epjk->epik', matrice_e_pg, matrice_e_pg, optimize='optimal')
                mat_mat = matrix_e_pg @ matrix_e_pg
                trace_mat_mat = np.einsum('epii->ep', mat_mat, optimize='optimal')
                I2_e_pg = (trace_e_pg**2 - trace_mat_mat)/2
                I3_e_pg = determinant_e_pg

                tic.Tac("Split", "Invariants", False)

                h = I1_e_pg**2 - 3*I2_e_pg                

                racine_h = np.sqrt(h)
                racine_h_ij = racine_h.reshape((Ne, nPg, 1, 1)).repeat(3, axis=2).repeat(3, axis=3)            
                
                arg = (2*I1_e_pg**3 - 9*I1_e_pg*I2_e_pg + 27*I3_e_pg)/2 # -1 <= arg <= 1
                arg[h != 0] *= 1/h[h != 0]**(3/2)

                phi = np.arccos(arg)/3 # Lode's angle such that 0 <= theta <= pi/3

                filtreNot0 = h != 0
                elemsNot0 = np.unique(np.where(filtreNot0)[0])

                elemsMin = np.unique(np.where(arg == 1)[0]) # positions of double minimum eigenvalue            
                elemsMax = np.unique(np.where(arg == -1)[0]) # positions of double maximum eigenvalue

                elemsNot0 = np.setdiff1d(elemsNot0, elemsMin)
                elemsNot0 = np.setdiff1d(elemsNot0, elemsMax)                

                # Initialisation des valeurs propres
                E1 = I1_e_pg/3 + 2/3 * racine_h * np.cos(2*np.pi/3 + phi)
                E2 = I1_e_pg/3 + 2/3 * racine_h * np.cos(2*np.pi/3 - phi)
                E3 = I1_e_pg/3 + 2/3 * racine_h * np.cos(phi)

                val_e_pg = (I1_e_pg/3).reshape((Ne, nPg, 1)).repeat(3, axis=2)
                val_e_pg[elemsNot0, :, 0] = E1[elemsNot0]
                val_e_pg[elemsNot0, :, 1] = E2[elemsNot0]
                val_e_pg[elemsNot0, :, 2] = E3[elemsNot0]

                tic.Tac("Split", "Eigenvalues", False)

                # Initialisation des projecteurs propres
                M1 = np.zeros_like(matrix_e_pg); M1[:,:,0,0] = 1
                M2 = np.zeros_like(matrix_e_pg); M2[:,:,1,1] = 1
                M3 = np.zeros_like(matrix_e_pg); M3[:,:,2,2] = 1

                eye3 = np.zeros_like(matrix_e_pg)
                eye3[:,:,0,0] = 1; eye3[:,:,1,1] = 1; eye3[:,:,2,2] = 1
                I_rg = np.einsum('ep,epij->epij', I1_e_pg - racine_h, eye3/3, optimize='optimal')

                # 4. Three equal eigenvalues
                # 𝜖1 = 𝜖2 = 𝜖3 ⇐⇒ 𝑔 = 0.
                # ici ne fait rien car déja initialisé 𝜖1 = 𝜖2 = 𝜖3 = I1_e_pg/3

                # 2. Two maximum eigenvalues
                # 𝜖1 < 𝜖2 = 𝜖3 ⇐⇒ 𝑔 ≠ 0, 𝜃 = 𝜋∕3.
                
                elems2 = np.unique(np.where(filtreNot0 & (E1<E2) & (E2==E3))[0])
                M1[elems2] = ((I_rg[elems2] - matrix_e_pg[elems2])/racine_h_ij[elems2])
                M2[elems2] = M3[elems2] = (eye3[elems2] - M1[elems2])/2

                # 3. Two minimum eigenvalues
                # 𝜖1 = 𝜖2 < 𝜖3 ⇐⇒ 𝑔 ≠ 0, 𝜃 = 0.
                
                elems3 = np.unique(np.where(filtreNot0 & (E1==E2) & (E2<E3))[0])
                M3[elems3] = ((matrix_e_pg[elems3] - I_rg[elems3])/racine_h_ij[elems3])
                M1[elems3] = M2[elems3] = (eye3[elems3] - M3[elems3])/2

                # 1. Three distinct eigenvalues
                # 𝜖1 < 𝜖2 < 𝜖3 ⇐⇒ 𝑔 ≠ 0, 𝜃 ≠ 0, 𝜃 ≠ 𝜋∕3.
                
                elems1 = np.unique(np.where(filtreNot0 & (E1<E2) & (E2<E3))[0])                

                E1_ij = E1.reshape((Ne,nPg,1,1)).repeat(3, axis=2).repeat(3, axis=3)[elems1]
                E2_ij = E2.reshape((Ne,nPg,1,1)).repeat(3, axis=2).repeat(3, axis=3)[elems1]
                E3_ij = E3.reshape((Ne,nPg,1,1)).repeat(3, axis=2).repeat(3, axis=3)[elems1]

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

            matrix = func_ep_epij(val_e_pg[:,:,0], M1) + func_ep_epij(val_e_pg[:,:,1], M2)

            matrix_eig = func_ep_epij(valnum[:,:,0], M1_num) + func_ep_epij(valnum[:,:,1], M2_num)
            
            if dim == 3:                
                M3_num = func_Mi(vectnum[:,:,:,2])
                matrix = matrix + func_ep_epij(val_e_pg[:,:,2], M3)
                matrix_eig = matrix_eig + func_ep_epij(valnum[:,:,2], M3_num)

            # check if the default values are correct
            if valnum.max() > 0:
                ecartVal = val_e_pg - valnum                    
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

        return val_e_pg, list_m, list_M
    
    def __Spectral_Decomposition(self, vector_e_pg: np.ndarray, verif=False):
        """Calculate projP and projM such that:\n

        vector_e_pg = [1 1 sqrt(2)] \n
        
        vectorP = projP : vector -> [1, 1, sqrt(2)]\n
        vectorM = projM : vector -> [1, 1, sqrt(2)]\n

        returns projP, projM
        """

        useNumba = self.__useNumba        

        dim = self.__material.dim        

        Ne = vector_e_pg.shape[0]
        nPg = vector_e_pg.shape[1]
        
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

            # Calculation of Beta Plus [e,pg,1].
            BetaP = dvalp[:,:,0].copy() # make sure you put copy here otherwise when Beta modification modifies dvalp at the same time!
            BetaP[elems,pdgs] = (valp[elems,pdgs,0]-valp[elems,pdgs,1])/v1_m_v2[elems,pdgs]
            
            # Calculation of Beta Moin [e,pg,1].
            BetaM = dvalm[:,:,0].copy()
            BetaM[elems,pdgs] = (valm[elems,pdgs,0]-valm[elems,pdgs,1])/v1_m_v2[elems,pdgs]
            
            # Calcul de Beta Moin [e,pg,1].
            gammap = dvalp - np.repeat(BetaP.reshape((Ne,nPg,1)),2, axis=2)
            gammam = dvalm - np.repeat(BetaM.reshape((Ne,nPg,1)), 2, axis=2)

            tic.Tac("Split", "Betas and gammas", False)

            if useNumba:
                # Faster
                projP, projM = CalcNumba.Get_projP_projM_2D(BetaP, gammap, BetaM, gammam, m1, m2)

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

                G12_ij, G13_ij, G23_ij = CalcNumba.Get_G12_G13_G23(M1, M2, M3)

                tic.Tac("Split", "Gab", False)

                list_mi = [m1, m2, m3]
                list_Gab = [G12_ij, G13_ij, G23_ij]

                projP, projM = CalcNumba.Get_projP_projM_3D(dvalp, dvalm, thetap, thetam, list_mi, list_Gab)
            
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
                    
                    colonnes = np.arange(0,6, dtype=int).reshape((1,6)).repeat(6,axis=0).reshape(-1)
                    lignes = np.sort(colonnes)

                    # # ici je construit une matrice pour verfier que les numéros sont bons
                    # ma = np.zeros((6,6), dtype=np.object0)
                    # for lin,col,i,j,k,l in zip(lignes, colonnes, listI, listJ, listK, listL):
                    #     text = f"{i+1}{j+1}{k+1}{l+1}"
                    #     ma[lin,col] = text
                    #     pass

                    Gij[:,:,lignes, colonnes] = Gijkl[:,:,listI,listJ,listK,listL]                    
                    
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

class Thermal_Model(IModel):

    __modelType = ModelType.thermal

    @property
    def modelType(self) -> ModelType:
        return Thermal_Model.__modelType
    
    @property
    def dim(self) -> int:
        return self.__dim

    @property
    def thickness(self) -> float:        
        return self.__thickness
    
    def __str__(self) -> str:
        text = f'\n{type(self).__name__} :'
        text += f'\nconduction thermique (k)  : {self.__k}'
        text += f'\ncapacité thermique massique (c) : {self.__c}'
        return text

    def __init__(self, dim:int, k: float, c=0.0, thickness=1.0):
        """Building a thermal model

        Parameters
        ----------
        dim : int
            model dimension
        k : float
            thermal conduction [W m^-1]
        c : float, optional
            mass heat capacity [J K^-1 kg^-1], by default 0.0
        thickness : float, optional
            thickness of part, by default 1.0
        """
        assert dim in [1,2,3]
        self.__dim = dim

        self.__k = k

        # ThermalModel Anisot with different diffusion coefficients for each direction! k becomes a matrix

        self.__c = c
        
        assert thickness > 0, "Must be greater than 0"
        self.__thickness = thickness

        self.Need_Update()

    @property
    def k(self) -> Union[float,np.ndarray]:
        """thermal conduction [W m^-1]"""
        return self.__k

    @property
    def c(self) -> Union[float,np.ndarray]:
        """specific heat capacity [J K^-1 kg^-1]"""
        return self.__c
    
    @property
    def isHeterogeneous(self) -> bool:
        return isinstance(self.k, np.ndarray) or isinstance(self.c, np.ndarray)


# --------------------------------------------------------------------------------------------
# FUNCTIONS
# --------------------------------------------------------------------------------------------
    
_erroDim = "Pay attention to the dimensions of the material constants.\nIf the material constants are in arrays, these arrays must have the same dimension."

def Reshape_variable(variable: Union[int,float,np.ndarray], Ne: int, nPg: int):
    """Reshape the variable so that it is in the form ep.."""

    if isinstance(variable, (int,float)):
        return np.ones((Ne, nPg)) * variable
    
    elif isinstance(variable, np.ndarray):
        shape = variable.shape
        if len(shape) == 1:
            if shape[0] == Ne:
                variable = variable[:,np.newaxis].repeat(nPg, axis=1)
                return variable
            elif shape[0] == nPg:
                variable = variable[np.newaxis].repeat(Ne, axis=0)
                return variable
            else:
                raise Exception("The variable entered must be of dimension (e) or (p)")

        if len(shape) == 2:
            if shape == (Ne, nPg):
                return variable
            else:
                variable = variable[np.newaxis, np.newaxis]
                variable = variable.repeat(Ne, axis=0)
                variable = variable.repeat(nPg, axis=1)
                return variable
            
        elif len(shape) == 3:
            if shape[0] == Ne:
                variable = variable[:, np.newaxis].repeat(nPg, axis=1)
                return variable
            elif shape[0] == nPg:
                variable = variable[np.newaxis].repeat(Ne, axis=0)
                return variable
            else:
                raise Exception("The variable entered must be of dimension (eij) or (pij)")

def Heterogeneous_Array(array: np.ndarray):
    """Build a heterogeneous array"""

    dimI, dimJ = array.shape
    
    shapes = [np.shape(array[i,j]) for i in range(dimI) for j in range(dimJ) if len(np.shape(array[i,j]))>0]
    if len(shapes) > 0:
        idx = np.argmax([len(shape) for shape in shapes])
        shape = shapes[idx]
    else:
        shape = ()

    shapeNew = list(shape); shapeNew.extend(array.shape)

    newArray = np.zeros(shapeNew)
    def SetMat(i,j):
        values = array[i,j]
        if isinstance(values, (int, float)):
            values = np.ones(shape) * values
        if len(shape) == 0:
            newArray[i,j] = values
        elif len(shape) == 1:
            newArray[:,i,j] = values
        elif len(shape) == 2:
            newArray[:,:,i,j] = values
        else:
            raise Exception("The material constants must be of maximum dimension (Ne, nPg)")
    [SetMat(i,j) for i in range(dimI) for j in range(dimJ)]

    return newArray

def TensorProduct(A: np.ndarray, B: np.ndarray, symmetric=False) -> np.ndarray:
    """Do the tensor product.

    Parameters
    ----------
    A : np.ndarray
        array A
    B : np.ndarray
        array B 
    symmetric : bool, optional
        do symmetric product, by default False

    Returns
    -------
    np.ndarray
        the calculated tensor product
    """
        
    sizeA = A.size
    sizeB = B.size

    assert sizeA is sizeB, "A and B must have the same dimensions"

    dim = len(A.shape)

    assert dim in [1,2], "A and B must be vectors (i) or matrices (ij)"

    if dim == 1:
        # Ai Bj
        res = np.einsum('i,j->ij',A,B)
    elif dim == 2:
        if symmetric:
            # 1/2 * (Aij Bjl + Ail Bjk)
            res = 1/2 * (np.einsum('ik,jl->ijkl',A,B)+np.einsum('il,jk->ijkl',A,B))
        else:
            # Aij Bkl
            res = np.einsum('ij,kl->ijkl', A, B)
    else:
        raise "Not implemented"
    
    return res

def KelvinMandel_Matrix(dim: int, Matrix: np.ndarray) -> np.ndarray:
    """Apply these coefficients to the matrix.
    \nif 2D:
    \n
    [1,1,r2]\n
    [1,1,r2]\n
    [r2, r2, 2]]\n

    \nif 3D:
    \n
    [1,1,1,r2,r2,r2]\n
    [1,1,1,r2,r2,r2]\n
    [1,1,1,r2,r2,r2]\n
    [r2,r2,r2,2,2,2]\n
    [r2,r2,r2,2,2,2]\n
    [r2,r2,r2,2,2,2]]\n
    """

    r2 = np.sqrt(2)

    if dim == 2:            
        transform = np.array([  [1,1,r2],
                                [1,1,r2],
                                [r2, r2, 2]])
    elif dim == 3:
        transform = np.array([  [1,1,1,r2,r2,r2],
                                [1,1,1,r2,r2,r2],
                                [1,1,1,r2,r2,r2],
                                [r2,r2,r2,2,2,2],
                                [r2,r2,r2,2,2,2],
                                [r2,r2,r2,2,2,2]])
    else:
        raise Exception("Not implemented")

    newMatrix = Matrix * transform

    return newMatrix

def Project_Kelvin(A: np.ndarray) -> np.ndarray:
    """Project the tensor A in Kelvin Mandel notation.

    Parameters
    ----------
    A : np.ndarray
        tensor A

    Returns
    -------
    np.ndarray
        Projected tensor
    """
    
    shapeA = A.shape
    assert np.std(shapeA) == 0, "Must have the same number of indices in all dimensions."
    orderA = len(shapeA)

    e = np.array([[1,6,5],[6,2,4],[5,4,3]]) - 1
    kron = lambda a,b: 1 if a==b else 0        

    if orderA == 2:
        # Aij -> AI
        assert shapeA == (3,3), "Must be a (3,3) array"
        
        A_I = np.zeros(6)
        def add(i,j) -> None:
            A_I[e[i,j]] = np.sqrt((2-kron(i,j))) * A[i,j]

        [add(i,j) for i in range(3) for j in range(3)]

        res = A_I

    elif orderA == 4:
        # Aijkl -> AIJ
        assert shapeA == (3,3,3,3), "Must be a (3,3,3,3) array"
        
        A_IJ = np.zeros((6,6))
        def add(i,j,k,l) -> None:
            A_IJ[e[i,j],e[k,l]] = np.sqrt((2-kron(i,j))*(2-kron(k,l))) * A[i,j,k,l]

        [add(i,j,k,l) for i in range(3) for j in range(3) for k in range(3) for l in range(3)]

        res = A_IJ

    else:
        raise "Not implemented"

    return res

def Result_in_Strain_or_Stress_field(field_e: np.ndarray, result:str, coef=np.sqrt(2)) -> np.ndarray:
    """Extracts a specific result from a 2D or 3D strain or stress field.

    Parameters
    ----------
    field_e : np.ndarray
        Strain or stress field in each element.
    result : str
        Desired result/value to extract:\n
            2D: [xx, yy, xy, vm, Strain, Stress] \n
            3D: [xx, yy, zz, yz, xz, xy, vm, Strain, Stress] \n
    coef : float, optional
        Coefficient used to scale cross components in the field (e.g., xy/coef in dim=2, or yz/coef, xz/coef, xy/coef if dim=3).

    Returns
    -------
    np.ndarray
        The extracted field corresponding to the specified result.
    """

    field_e = np.asarray(field_e)

    Ne = field_e.shape[0]

    if field_e.shape == (Ne, 3):
        dim = 2
    elif field_e.shape == (Ne, 6):
        dim = 3
    else:
        raise Exception("field_e must be of shape (Ne, 3) or (Ne, 6)")    
    
    if dim == 2:

        xx_e = field_e[:,0]
        yy_e = field_e[:,1]
        xy_e = field_e[:,2]/coef
        
        val_vm_e = np.sqrt(xx_e**2+yy_e**2-xx_e*yy_e+3*xy_e**2)

        if "xx" in result:
            result_e = xx_e
        elif "yy" in result:
            result_e = yy_e
        elif "xy" in result:
            result_e = xy_e
        elif "vm" in result:
            result_e = val_vm_e
        elif result == "Strain" or result == "Stress":
            result_e = np.append(result_e, val_vm_e.reshape((Ne,1)), axis=1)
        else:
            raise Exception("result must be in [xx, yy, xy, vm, Strain, Stress]")

    elif dim == 3:

        xx_e = field_e[:,0]
        yy_e = field_e[:,1]
        zz_e = field_e[:,2]
        yz_e = field_e[:,3]/coef
        xz_e = field_e[:,4]/coef
        xy_e = field_e[:,5]/coef

        val_vm_e = np.sqrt(((xx_e-yy_e)**2+(yy_e-zz_e)**2+(zz_e-xx_e)**2+6*(xy_e**2+yz_e**2+xz_e**2))/2)

        if "xx" in result:
            result_e = xx_e
        elif "yy" in result:
            result_e = yy_e
        elif "zz" in result:
            result_e = zz_e
        elif "yz" in result:
            result_e = yz_e
        elif "xz" in result:
            result_e = xz_e
        elif "xy" in result:
            result_e = xy_e
        elif "vm" in result:
            result_e = val_vm_e
        elif result == "Strain" or result == "Stress":
            result_e = np.append(result_e, val_vm_e.reshape((Ne,1)), axis=1)
        else:
            raise Exception("result must be in [xx, yy, zz, yz, xz, xy, vm, Strain, Stress]")
        
    return result_e

def Get_Pmat(axis_1: np.ndarray, axis_2: np.ndarray, useMandel=True):
    """Construct Pmat to pass from the material coordinates (x,y,z) to the global coordinate (X,Y,Z) such that:\n

    if useMandel:\n
        return [Pm]\n
    else:\n
        return [Ps], [Pe]\n
    
    In mandel's notation\n
        Sig & Eps en [11, 22, 33, sqrt(2)*23, sqrt(2)*13, sqrt(2)*12]\n
        [C_global] = [Pm] * [C_material] * [Pm]' & [C_material] = [Pm]' * [C_global] * [Pm]\n
        [S_global] = [Pm] * [S_material] * [Pm]' & [S_material] = [Pm]' * [S_global] * [Pm]\n
        Sig_global = [Pm] * Sig_material & Sig_material = [Pm]' * Sig_global\n
        Eps_global = [Pm] * Eps_material & Eps_material = [Pm]' * Eps_global\n

    Or in voigt's notation
        Sig [S11, S22, S33, S23, S13, S12]\n
        Eps [E11, E22, E33, 2*E23, 2*E13, 2*E12]\n
        [C_global] = [Ps] * [C_material] * [Ps]' & [C_material] = [Pe]' * [C_global] * [Pe]\n
        S_global = [Pe] * [S_material] * [Pe]' & [S_material] = [Ps]' * S_global * [Ps]\n
        Sig_global = [Ps] * Sig_material & Sig_material = [Pe]' * Sig_global\n
        Eps_global = [Pe] * Eps_material & Eps_material = [Ps]' * Eps_global\n

    P matrices are orhogonal -> inv(P) = transpose(P)\n

    Here we use "Chevalier 1988 : Comportements élastique et viscoélastique des composites"
    """

    axis_1 = np.asarray(axis_1)
    axis_2 = np.asarray(axis_2)

    dim = axis_1.shape[-1]
    assert dim in [2,3], "Must be a 2d or 3d vector"
    shape1 = axis_1.shape
    shape2 = axis_2.shape
    assert len(shape1) <= 3, "Must be a numpy array of shape (i), (e,i) or (e,p,i)"
    assert len(shape2) <= 3, "Must be a numpy array of shape (i), (e,i) or (e,p,i)"
    assert shape1 == shape2, "axis_1 and axis_2 must be the same size"    

    # get the indices and transpose
    if len(shape1) == 1:
        id=''
        transposeP = (0,1) # (dim*2,dim*2) -> (dim*2,dim*2)
    elif len(shape1) == 2:
        id='e'
        axis_1 = axis_1.transpose((1,0)) # (e,dim) -> (dim,e)
        axis_2 = axis_2.transpose((1,0))
        transposeP = (2,0,1) # (dim,dim,e) -> (e,dim,dim)
    elif len(shape1) == 3:
        id='ep'
        axis_1 = axis_1.transpose((2,0,1)) # (e,p,dim) -> (dim,e,p)
        axis_2 = axis_2.transpose((2,0,1))
        transposeP = (2,3,0,1) # (dim,dim,e,p) -> (e,p,dim,dim)

    # normalize thoses vectors
    axis_1 = np.einsum(f'i{id},{id}->i{id}', axis_1, np.linalg.norm(axis_1, axis=0), optimize='optimal')
    axis_2 = np.einsum(f'i{id},{id}->i{id}', axis_2, np.linalg.norm(axis_2, axis=0), optimize='optimal')

    # Detection of whether the 2 vectors are perpendicular
    dotProd = np.einsum(f'i{id},i{id}->{id}', axis_1, axis_2, optimize='optimal')    
    assert np.linalg.norm(dotProd) <= 1e-12, 'Must give perpendicular axes'    
    
    if dim == 2:
        p11, p12 = axis_1
        p21, p22 = axis_2
    elif dim == 3:
        # construct z-axis
        axis_3 = np.cross(axis_1, axis_2, axis=0)
        p11, p12, p13 = axis_1
        p21, p22, p23 = axis_2
        p31, p32, p33 = axis_3        

    if len(shape1) == 1:            
        p = np.zeros((dim,dim))
    elif len(shape1) == 2:
        p = np.zeros((dim,dim,shape1[0]))
    elif len(shape1) == 3:
        p = np.zeros((dim,dim,shape1[0],shape1[1]))

    # apply vectors
    p[:,0] = axis_1
    p[:,1] = axis_2    
    if dim == 3:
        p[:,2] = axis_3

    D1 = p**2

    if dim == 2:

        A = np.array([[p11*p21],
                      [p12*p22]])
        
        B = np.array([[p11*p12, p21*p22]])

        D2 = np.array([[p11*p22 + p21*p12]])

    elif dim == 3:

        A = np.array([[p21*p31, p11*p31, p11*p21],
                        [p22*p32, p12*p32, p12*p22],
                        [p23*p33, p13*p33, p13*p23]])
        
        B = np.array([[p12*p13, p22*p23, p32*p33],
                        [p11*p13, p21*p23, p31*p33],
                        [p11*p12, p21*p22, p31*p32]])

        D2 = np.array([[p22*p33 + p32*p23, p12*p33 + p32*p13, p12*p23 + p22*p13],
                        [p21*p33 + p31*p23, p11*p33 + p31*p13, p11*p23 + p21*p13],
                        [p21*p32 + p31*p22, p11*p32 + p31*p12, p11*p22 + p21*p12]])

    if useMandel:
        cM = np.sqrt(2)
        Pmat = np.concatenate((np.concatenate((D1, cM*A), axis=1),
                               np.concatenate((cM*B, D2), axis=1)), axis=0)

        Pmat = np.transpose(Pmat,transposeP)    
        return Pmat
    else:
        Ps = np.concatenate((np.concatenate((D1, 2*A), axis=1),
                              np.concatenate((B, D2), axis=1)), axis=0).transpose(transposeP)
        
        Pe = np.concatenate((np.concatenate((D1, A), axis=1),
                              np.concatenate((2*B, D2), axis=1)), axis=0).transpose(transposeP)
        
        return Ps, Pe

def Apply_Pmat(P: np.ndarray, Matrix: np.ndarray, toGlobal=True) -> np.ndarray:
    """Apply change base matrix.\n
    WARNING: P must be in Kelvin mandel notation

    Parameters
    ----------
    P : np.ndarray
        P in mandel notation obtained with get_Pmat
    Matrix : np.ndarray
        3x3 or 6x6 matrix
    toGlobal : bool, optional
        sets wheter matrix is C or S, by default True\n
    toGlobal : bool, optional
        sets wheter you want to get matrix in global or material coordinates, by default True\n
        if toGlobal:\n
            Matrix_global = P * C_material * P'\n
        else:\n
            Matrix_material = P' * Matrix_global * P

    Returns
    -------
    np.ndarray
        new matrix
    """    
    assert isinstance(Matrix, np.ndarray), 'Matrix must be an array'
    assert Matrix.shape[-2:] == P.shape[-2:], 'Must give an matrix of shape (e,dim,dim) or (e,p,dim,dim) or (dim,dim)'
            
    # Get P indices
    pDim = P.ndim
    if pDim == 2:
        pi = ''
    elif pDim == 3:
        pi = 'e'
    elif pDim == 4:
        pi = 'ep'

    # Get P last indices
    if toGlobal:
        i1 = 'ij'
        id2 = 'lk'
    else:            
        i1 = 'ji'
        id2 = 'kl'

    # Get matrix indices
    matDim = Matrix.ndim
    if matDim == 2:
        mi = ''            
    elif matDim == 3:
        mi = 'e'
    elif matDim == 4:
        mi = 'ep'
    else:
        raise Exception("The matrix must be of dimension (ij) or (eij) or (epij)")
    
    ii = mi if matDim > pDim else pi
    matrice_P = np.einsum(f'{pi}{i1},{mi}jk,{pi}{id2}->{ii}il',P, Matrix, P, optimize='optimal')
    
    return matrice_P