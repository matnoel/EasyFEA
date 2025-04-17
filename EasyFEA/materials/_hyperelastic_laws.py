# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Hyperelastic laws."""

from abc import ABC, abstractmethod
import numpy as np

# utilities
from ..fem import Mesh, MatrixType, FeArray
from ._hyperelastic import HyperElastic
# others
from ._utils import _IModel, ModelType
from ..utilities import _params

# ----------------------------------------------
# Hyper Elastic
# ----------------------------------------------

class _HyperElas(_IModel, ABC):
    """HyperElasticit material.\n
    _, _ and _ inherit from _HyperElas class.
    """
    def __init__(self, dim: int, thickness: float):
        
        assert dim in [2,3], "Must be dimension 2 or 3"
        self.__dim = dim
        
        if dim == 2:
            assert thickness > 0 , "Must be greater than 0"
            self.__thickness = thickness

        self.useNumba = False

    @property
    def modelType(self) -> ModelType:
        return ModelType.hyperelastic

    @property
    def dim(self) -> int:
        return self.__dim

    @property
    def thickness(self) -> float:
        if self.__dim == 2:
            return self.__thickness
        else:
            return 1.0
   
    # @abstractmethod
    # def _Update(self) -> None:
    #     """Updates the constitutives laws by updating the C stiffness and S compliance matrices. in Kelvin Mandel notation"""
    #     pass

    # Model
    @staticmethod
    def Available_Laws():
        laws = [NeoHookean]
        return laws

    @property
    def isHeterogeneous(self) -> bool:
        return False

    @abstractmethod
    def Compute_W(self, mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> FeArray:
        """Computes the quadratic energy W(u).
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [u1, v1, w1, . . ., uN, vN, wN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi
        
        Returns
        -------
        np.ndarray
            We_e_pg of shape (Ne, pg)
        """
        
        return None

    @abstractmethod
    def Compute_dWde(self, mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> FeArray:
        """Computes the second Piola-Kirchhoff tensor Σ(u).
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [u1, v1, w1, . . ., uN, vN, wN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi
        
        Returns
        -------
        np.ndarray
            Σ_e_pg of shape (Ne, pg, 6)

        Σxx, Σyy, Σzz, sqrt(2) Σyz, sqrt(2) Σxz, sqrt(2) Σxy
        """
        return None

    @abstractmethod
    def Compute_d2Wde(self, mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> FeArray:
        """Computes dΣde.
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [u1, v1, w1, . . ., uN, vN, wN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi
        
        Returns
        -------
        np.ndarray
            dΣde_e_pg of shape (Ne, pg, 6, 6)
        """
        return None
    
# ----------------------------------------------
# Neo-Hookean
# ----------------------------------------------

class NeoHookean(_HyperElas):

    def __init__(self, dim: int, K: float, thickness=1.):
        """Creates an Neo-Hookean material.

        Parameters
        ----------
        dim : int
            dimension (e.g 2 or 3)
        K : float|np.ndarray, optional
            Bulk modulus
        thickness : float, optional
            thickness, by default 1.0
        """

        _HyperElas.__init__(self, dim, thickness)

        self.K = K

    @property
    def K(self):
        """Bulk modulus"""
        return self.__K

    @K.setter
    def K(self, value):
        _params.CheckIsPositive(value)
        self.Need_Update()
        self.__K = value

    def Compute_W(self, mesh, u, matrixType=MatrixType.rigi) -> FeArray:

        K = self.K
        I1 = HyperElastic.Compute_I1(mesh, u, matrixType)
        
        W = K * (I1 - 3)

        return W

    def Compute_dWde(self, mesh, u, matrixType=MatrixType.rigi) -> FeArray:
        
        K = self.K
        dI1dC = HyperElastic.Compute_dI1dC()
        
        dW = K * dI1dC

        return 2 * dW
    
    def Compute_d2Wde(self, mesh, u, matrixType=MatrixType.rigi) -> FeArray:
        
        K = self.K
        d2I1dC = HyperElastic.Compute_d2I1dC()

        d2W = 4 * (K * d2I1dC)

        return d2W

# ----------------------------------------------
# Mooney-Rivlin
# ----------------------------------------------

class MooneyRivlin(_HyperElas):

    def __init__(self, dim: int, K1: float, K2: float, thickness=1.):
        """Creates an Mooney-Rivlin material.

        Parameters
        ----------
        dim : int
            dimension (e.g 2 or 3)
        K1 : float|np.ndarray, optional
            Kappa1
        K2 : float|np.ndarray, optional
            Kappa2 -> Neo-Hoolkean if K2=0
        thickness : float, optional
            thickness, by default 1.0
        """

        _HyperElas.__init__(self, dim, thickness)

        self.K1 = K1
        self.K2 = K2

    @property
    def K1(self):
        """Kappa1"""
        return self.__K1

    @K1.setter
    def K1(self, value):
        _params.CheckIsPositive(value)
        self.Need_Update()
        self.__K1 = value

    @property
    def K2(self):
        """Kappa2"""
        return self.__K2

    @K2.setter
    def K2(self, value):
        assert value >= 0
        self.Need_Update()
        self.__K2 = value

    def Compute_W(self, mesh, u, matrixType=MatrixType.rigi) -> FeArray:

        K1 = self.K1
        K2 = self.K2
        I1 = HyperElastic.Compute_I1(mesh, u, matrixType)
        I2 = HyperElastic.Compute_I2(mesh, u, matrixType)
        
        W = K1*(I1 - 3) + K2*(I2 - 3)

        return W

    def Compute_dWde(self, mesh, u, matrixType=MatrixType.rigi) -> FeArray:
        
        K1 = self.K1
        K2 = self.K2
        dI1dC = HyperElastic.Compute_dI1dC()
        dI2dC = HyperElastic.Compute_dI2dC(mesh, u, matrixType)
        
        dW = 2 * (K1 * dI1dC + K2 * dI2dC)

        return dW
    
    def Compute_d2Wde(self, mesh, u, matrixType=MatrixType.rigi) -> FeArray:

        K1 = self.K1
        K2 = self.K2
        d2I1dC = HyperElastic.Compute_d2I1dC()
        d2I2dC = HyperElastic.Compute_d2I2dC()

        d2W = 4 * (K1 * d2I1dC + K2 * d2I2dC)

        return d2W
    
# ----------------------------------------------
# Saint-Venant-Kirchhoff
# ----------------------------------------------

class SaintVenantKirchhoff(_HyperElas):

    def __init__(self, dim: int, lmbda: float, mu: float, thickness=1.):
        """Creates Saint-Venant-Kirchhoff material.

        Parameters
        ----------
        dim : int
            dimension (e.g 2 or 3)
        lmbda : float|np.ndarray, optional
            Lame's first parameter
        mu : float|np.ndarray, optional
            Shear modulus
        thickness : float, optional
            thickness, by default 1.0
        """

        _HyperElas.__init__(self, dim, thickness)

        self.lmbda = lmbda
        self.mu = mu

    @property
    def lmbda(self):
        """Lame's first parameter"""
        return self.__lmbda

    @lmbda.setter
    def lmbda(self, value):
        self.Need_Update()
        self.__lmbda = value

    @property
    def mu(self):
        """Shear modulus"""
        return self.__mu

    @mu.setter
    def mu(self, value):
        _params.CheckIsPositive(value)
        self.Need_Update()
        self.__mu = value

    def Compute_W(self, mesh, u, matrixType=MatrixType.rigi) -> FeArray:

        lmbda = self.lmbda
        mu = self.mu
        I1 = HyperElastic.Compute_I1(mesh, u, matrixType)
        I2 = HyperElastic.Compute_I2(mesh, u, matrixType)
               
        W = lmbda*(I1**2 - 6*I1 + 9)/8 + mu*(I1**2 - 2*I1 - 2*I2 + 3)/4
        
        # # same as
        # e = HyperElastic.Compute_e(mesh, u, matrixType)
        # from ..utilities._linalg import Trace
        # We = lmbda/2 * Trace(e)**2 + mu * Trace(e @ e)
        # diff = np.abs(W - We)

        return W

    def Compute_dWde(self, mesh, u, matrixType=MatrixType.rigi) -> FeArray:

        lmbda = self.lmbda
        mu = self.mu
        I1 = HyperElastic.Compute_I1(mesh, u, matrixType)
        dI1dC = HyperElastic.Compute_dI1dC()
        dI2dC = HyperElastic.Compute_dI2dC(mesh, u, matrixType)

        I1 = I1[:,:,np.newaxis].repeat(6, -1)
        dW = 2 * (lmbda*(2*I1 - 6)/8 + mu*(2*I1 - 2)/4 * dI1dC - mu/2 * dI2dC)
        
        return dW

    def Compute_d2Wde(self, mesh, u, matrixType=MatrixType.rigi) -> FeArray:

        lmbda = self.lmbda
        mu = self.mu
        I1 = HyperElastic.Compute_I1(mesh, u, matrixType)
        dI1dC = HyperElastic.Compute_dI1dC() 
        d2I1dC = HyperElastic.Compute_d2I1dC()
        d2I2dC = HyperElastic.Compute_d2I2dC()

        d2W = 4 * (lmbda*(2*I1 - 6)/8 + mu*(2*I1 - 2)/4 * d2I1dC - mu/2 * d2I2dC) \
            + 4 * (lmbda/4 + mu/2 * dI1dC.T @ dI1dC)
        
        return d2W

# # ----------------------------------------------
# # Ciarlet Geymonat
# # ----------------------------------------------

# class CiarletGeymonat(_HyperElas):

#     def __init__(self, dim: int, K: float, K1: float, K2: float, thickness=1.):
#         """Creates Saint-Venant-Kirchhoff material.

#         Parameters
#         ----------
#         dim : int
#             dimension (e.g 2 or 3)
#         K : float|np.ndarray, optional
#             Bulk modulus
#         K1 : float|np.ndarray, optional
#             Kappa1
#         K2 : float|np.ndarray, optional
#             Kappa2 -> Neo-Hoolkean if K2=0
#         thickness : float, optional
#             thickness, by default 1.0
#         """

#         _HyperElas.__init__(self, dim, thickness)

#         self.K = K
#         self.K1 = K1
#         self.K2 = K2

#     @property
#     def K(self):
#         """Bulk modulus"""
#         return self.__K

#     @K.setter
#     def K(self, value):
#         _params.CheckIsPositive(value)
#         self.Need_Update()
#         self.__K = value

#     @property
#     def K1(self):
#         """Kappa1"""
#         return self.__K1

#     @K1.setter
#     def K1(self, value):
#         _params.CheckIsPositive(value)
#         self.Need_Update()
#         self.__K1 = value

#     @property
#     def K2(self):
#         """Kappa2"""
#         return self.__K2

#     @K2.setter
#     def K2(self, value):
#         assert value >= 0
#         self.Need_Update()
#         self.__K2 = value 

#     def Compute_W(self, mesh, u, matrixType=MatrixType.rigi):

#         K = self.K
#         K1 = self.K1
#         K2 = self.K2

#         I1 = HyperElastic.Compute_I1(mesh, u, matrixType)
#         I2 = HyperElastic.Compute_I2(mesh, u, matrixType)
#         I3 = HyperElastic.Compute_I2(mesh, u, matrixType)
               
#         W = K*K2*(I2/I3**(2/3) - 3)*(np.sqrt(I3) - np.log(np.sqrt(I3)) - 1) + K1*(I1/I3**(1/3) - 3)
       
#         return W

#     def Compute_dWde(self, mesh, u, matrixType=MatrixType.rigi):

#         K = self.K
#         K1 = self.K1
#         K2 = self.K2

#         I1 = HyperElastic.Compute_I1(mesh, u, matrixType)
#         I2 = HyperElastic.Compute_I2(mesh, u, matrixType)
#         I3 = HyperElastic.Compute_I2(mesh, u, matrixType)

#         dI1dC = HyperElastic.Compute_dI1dC()
#         dI2dC = HyperElastic.Compute_dI2dC(mesh, u, matrixType)
#         dI3dC = HyperElastic.Compute_dI3dC(mesh, u, matrixType)
               
#         dW = 2 * (K1/I3**(1/3) * dI1dC + K*K2*(np.sqrt(I3) - 1)/I3**(2/3) * dI2dC + -I1*K1/(3*I3**(4/3)) - 2*I2*K*K2*(np.sqrt(I3) - 1)/(3*I3**(5/3)) - 1/(2*I3) + K*K2*(I2/I3**(2/3) - 3)/(2*np.sqrt(I3)) * dI3dC)
         
#         return dW

#     def Compute_d2Wde(self, mesh, u, matrixType=MatrixType.rigi):

#         K = self.K
#         K1 = self.K1
#         K2 = self.K2

#         I1 = HyperElastic.Compute_I1(mesh, u, matrixType)
#         I2 = HyperElastic.Compute_I2(mesh, u, matrixType)
#         I3 = HyperElastic.Compute_I2(mesh, u, matrixType)

#         dI1dC = HyperElastic.Compute_dI1dC(mesh, u, matrixType)
#         dI2dC = HyperElastic.Compute_dI2dC(mesh, u, matrixType)
#         dI3dC = HyperElastic.Compute_dI3dC(mesh, u, matrixType)

#         d2I1dC = HyperElastic.Compute_d2I1dC()
#         d2I2dC = HyperElastic.Compute_d2I2dC()
#         d2I3dC = HyperElastic.Compute_d2I3dC(mesh, u, matrixType)

#         d2W = 4 * (K1/I3**(1/3) * d2I1dC + K*K2*(sqrt(I3) - 1)/I3**(2/3) * d2I2dC + -I1*K1/(3*I3**(4/3)) - 2*I2*K*K2*(sqrt(I3) - 1)/(3*I3**(5/3)) - 1/(2*I3) + K*K2*(I2/I3**(2/3) - 3)/(2*sqrt(I3)) * d2I3dC) + 4 * ( + -K1/(3*I3**(4/3)) * dI1dC.T @ dI3dC + -2*K*K2*(sqrt(I3) - 1)/(3*I3**(5/3)) + K*K2/(2*I3**(7/6)) * dI2dC.T @ dI3dC-K1/(3*I3**(4/3)) * dI3dC.T @ dI1dC + -2*K*K2*(sqrt(I3) - 1)/(3*I3**(5/3)) + K*K2/(2*I3**(7/6)) * dI3dC.T @ dI2dC + 4*I1*K1/(9*I3**(7/3)) + 10*I2*K*K2*(sqrt(I3) - 1)/(9*I3**(8/3)) - 2*I2*K*K2/(3*I3**(13/6)) + 1/(2*I3**2) - K*K2*(I2/I3**(2/3) - 3)/(4*I3**(3/2)) * dI3dC.T @ dI3dC)

#         return None