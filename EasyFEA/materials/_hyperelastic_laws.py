# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Hyperelastic laws."""

from abc import ABC, abstractmethod
import numpy as np

# utilities
from ..fem import Mesh, MatrixType
from ._hyperelastic import HyperElastic
# others
from ._utils import _IModel, ModelType
from ..utilities import _params
from ..utilities._linalg import Trace

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
    def Compute_We(self, mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes the quadratic energy We(u).
        
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
    def Compute_dWede(self, mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
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
            Σ_e_pg of shape (Ne, pg, 3, 3)

        Σxx Σxy Σxz \n
        Σyx Σyy Σyz \n
        Σzx Σzy Σzz
        """
        return None

    @abstractmethod
    def Compute_d2Wede(self, mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
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

    def Compute_We(self, mesh, u, matrixType=MatrixType.rigi):

        K = self.K
        I1 = HyperElastic.Compute_I1(mesh, u, matrixType)
        
        We = K * (I1 - 3)

        return We

    def Compute_dWede(self, mesh, u, matrixType=MatrixType.rigi):
        
        K = self.K
        dI1dC = HyperElastic.Compute_dI1dC(mesh, u, matrixType)        
        
        dWedC = K * dI1dC

        return 2 * dWedC
    
    def Compute_d2Wede(self, mesh, u, matrixType=MatrixType.rigi):
        return super().Compute_d2Wede(mesh, u, matrixType)

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
        _params.CheckIsPositive(value)
        self.Need_Update()
        self.__K2 = value

    def Compute_We(self, mesh, u, matrixType=MatrixType.rigi):

        K1 = self.K1
        K2 = self.K2
        I1 = HyperElastic.Compute_I1(mesh, u, matrixType)
        I2 = HyperElastic.Compute_I2(mesh, u, matrixType)
        
        We = K1 * (I1 - 3) + K2 * (I2 - 3)

        return We

    def Compute_dWede(self, mesh, u, matrixType=MatrixType.rigi):
        
        K1 = self.K1
        K2 = self.K2
        dI1dC = HyperElastic.Compute_dI1dC(mesh, u, matrixType)
        dI2dC = HyperElastic.Compute_dI2dC(mesh, u, matrixType)
        
        dWedC = K1 * dI1dC + K2 * dI2dC

        return 2 * dWedC
    
    def Compute_d2Wede(self, mesh, u, matrixType=MatrixType.rigi):
        return super().Compute_d2Wede(mesh, u, matrixType)

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

    def Compute_We(self, mesh, u, matrixType=MatrixType.rigi):

        lmbda = self.lmbda
        mu = self.mu
        I1 = HyperElastic.Compute_I1(mesh, u, matrixType)
        I2 = HyperElastic.Compute_I2(mesh, u, matrixType)
               
        We = lmbda/8 * (I1 - 3)**2 + mu/4 * (I1**2 - 2*I1 - 2*I2 + 3)
        
        # same as
        e = HyperElastic.Compute_e(mesh, u, matrixType)
        We = lmbda/2 * Trace(e)**2 + mu * Trace(e @ e)

        return We

    def Compute_dWede(self, mesh, u, matrixType=MatrixType.rigi):

        lmbda = self.lmbda
        mu = self.mu
        I1 = HyperElastic.Compute_I1(mesh, u, matrixType)
        I2 = HyperElastic.Compute_I2(mesh, u, matrixType)
        dI1dC = HyperElastic.Compute_dI1dC(mesh, u, matrixType)
        dI2dC = HyperElastic.Compute_dI2dC(mesh, u, matrixType)
        
        dWedI1 = lmbda/4 * (I1 - 3) - mu/2
        dWedI2 = mu/2 * (I2 - 1)

        dWedC = dWedI1 * dI1dC + dWedI2 * dI2dC
        
        return 2 * dWedC

    def Compute_d2Wede(self, mesh, u, matrixType=MatrixType.rigi):
        return super().Compute_d2Wede(mesh, u, matrixType)