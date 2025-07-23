# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Hyperelastic laws."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union

# utilities
from ..fem import Mesh, MatrixType, FeArray
from ._hyperelastic import HyperElastic

# others
from ._utils import _IModel, ModelType
from ..utilities import _params, _types

# ----------------------------------------------
# Hyper Elastic
# ----------------------------------------------


class _HyperElas(_IModel, ABC):
    """HyperElastic material.\n
    NeoHookean, MooneyRivlin and SaintVenantKirchhoff inherit from _HyperElas class.
    """

    def __init__(self, dim: int, thickness: float):
        assert dim in [2, 3], "Must be dimension 2 or 3"
        self.__dim = dim

        if dim == 2:
            assert thickness > 0, "Must be greater than 0"
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

    @property
    def coef(self) -> float:
        """kelvin mandel coef -> sqrt(2)"""
        return np.sqrt(2)

    # Model
    @staticmethod
    def Available_Laws():
        laws = [NeoHookean, MooneyRivlin, SaintVenantKirchhoff]
        return laws

    @property
    def isHeterogeneous(self) -> bool:
        return False

    @abstractmethod
    def Compute_W(
        self, mesh: Mesh, u: _types.FloatArray, matrixType=MatrixType.rigi
    ) -> FeArray:
        """Computes the quadratic energy W(u).

        Parameters
        ----------
        mesh : Mesh
            mesh
        u : _types.FloatArray
            discretized displacement field [u1, v1, w1, . . ., uN, vN, wN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        FeArray
            We_e_pg of shape (Ne, pg)
        """

        return None  # type: ignore [return-value]

    @abstractmethod
    def Compute_dWde(
        self, mesh: Mesh, u: _types.FloatArray, matrixType=MatrixType.rigi
    ) -> FeArray:
        """Computes the second Piola-Kirchhoff tensor Σ(u).

        Parameters
        ----------
        mesh : Mesh
            mesh
        u : _types.FloatArray
            discretized displacement field [u1, v1, w1, . . ., uN, vN, wN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        FeArray
            Σ_e_pg of shape (Ne, pg, 6)

        Σxx, Σyy, Σzz, sqrt(2) Σyz, sqrt(2) Σxz, sqrt(2) Σxy
        """
        return None  # type: ignore [return-value]

    @abstractmethod
    def Compute_d2Wde(
        self, mesh: Mesh, u: _types.FloatArray, matrixType=MatrixType.rigi
    ) -> FeArray:
        """Computes dΣde.

        Parameters
        ----------
        mesh : Mesh
            mesh
        u : _types.FloatArray
            discretized displacement field [u1, v1, w1, . . ., uN, vN, wN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        FeArray
            dΣde_e_pg of shape (Ne, pg, 6, 6)
        """
        return None  # type: ignore [return-value]


# ----------------------------------------------
# Neo-Hookean
# ----------------------------------------------


class NeoHookean(_HyperElas):
    def __init__(self, dim: int, K: Union[float, _types.FloatArray], thickness=1.0):
        """Creates an Neo-Hookean material.

        Parameters
        ----------
        dim : int
            dimension (e.g 2 or 3)
        K : float|_types.FloatArray, optional
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
        I3 = HyperElastic.Compute_I3(mesh, u, matrixType)

        W = K * (I1 / I3 ** (1 / 3) - 3)

        return W

    def Compute_dWde(self, mesh, u, matrixType=MatrixType.rigi) -> FeArray:
        K = self.K
        I1 = HyperElastic.Compute_I1(mesh, u, matrixType)
        I3 = HyperElastic.Compute_I3(mesh, u, matrixType)

        dWdI1 = K / I3 ** (1 / 3)
        dWdI3 = -I1 * K / (3 * I3 ** (4 / 3))
        dI1dC = HyperElastic.Compute_dI1dC()
        dI3dC = HyperElastic.Compute_dI3dC(mesh, u, matrixType)

        dWdI1 = K / I3 ** (1 / 3)
        dWdI3 = -I1 * K / (3 * I3 ** (4 / 3))
        dW = 2 * (dWdI1 * dI1dC + dWdI3 * dI3dC)

        return dW

    def Compute_d2Wde(self, mesh, u, matrixType=MatrixType.rigi) -> FeArray:
        K = self.K
        I1 = HyperElastic.Compute_I1(mesh, u, matrixType)
        I3 = HyperElastic.Compute_I3(mesh, u, matrixType)

        dI1dC = HyperElastic.Compute_dI1dC()[..., np.newaxis]
        dI3dC = HyperElastic.Compute_dI3dC(mesh, u, matrixType)[..., np.newaxis]
        d2I1dC = HyperElastic.Compute_d2I1dC()
        d2I3dC = HyperElastic.Compute_d2I3dC(mesh, u, matrixType)

        dWdI1 = K / I3 ** (1 / 3)
        d2WdI1I3 = -K / (3 * I3 ** (4 / 3))
        dWdI3 = -I1 * K / (3 * I3 ** (4 / 3))
        d2WdI3I1 = -K / (3 * I3 ** (4 / 3))
        d2WdI3I3 = 4 * I1 * K / (9 * I3 ** (7 / 3))

        d2W = 4 * (dWdI1 * d2I1dC + dWdI3 * d2I3dC) + 4 * (
            d2WdI1I3 * dI1dC @ dI3dC.T
            + d2WdI3I1 * dI3dC @ dI1dC.T
            + d2WdI3I3 * dI3dC @ dI3dC.T
        )

        return d2W


# ----------------------------------------------
# Mooney-Rivlin
# ----------------------------------------------


class MooneyRivlin(_HyperElas):
    def __init__(
        self,
        dim: int,
        K1: Union[float, _types.FloatArray],
        K2: Union[float, _types.FloatArray],
        thickness=1.0,
    ):
        """Creates an Mooney-Rivlin material.

        Parameters
        ----------
        dim : int
            dimension (e.g 2 or 3)
        K1 : float|_types.FloatArray, optional
            Kappa1
        K2 : float|_types.FloatArray, optional
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
        I3 = HyperElastic.Compute_I3(mesh, u, matrixType)

        W = K1 * (I1 / I3 ** (1 / 3) - 3) + K2 * (I2 / I3 ** (2 / 3) - 3)

        return W

    def Compute_dWde(self, mesh, u, matrixType=MatrixType.rigi) -> FeArray:
        K1 = self.K1
        K2 = self.K2
        I1 = HyperElastic.Compute_I1(mesh, u, matrixType)
        I2 = HyperElastic.Compute_I2(mesh, u, matrixType)
        I3 = HyperElastic.Compute_I3(mesh, u, matrixType)

        dI1dC = HyperElastic.Compute_dI1dC()
        dI2dC = HyperElastic.Compute_dI2dC(mesh, u, matrixType)
        dI3dC = HyperElastic.Compute_dI3dC(mesh, u, matrixType)

        dWdI1 = K1 / I3 ** (1 / 3)
        dWdI2 = K2 / I3 ** (2 / 3)
        dWdI3 = -I1 * K1 / (3 * I3 ** (4 / 3)) - 2 * I2 * K2 / (3 * I3 ** (5 / 3))

        dW = 2 * (dWdI1 * dI1dC + dWdI2 * dI2dC + dWdI3 * dI3dC)

        return dW

    def Compute_d2Wde(self, mesh, u, matrixType=MatrixType.rigi) -> FeArray:
        K1 = self.K1
        K2 = self.K2
        I1 = HyperElastic.Compute_I1(mesh, u, matrixType)
        I2 = HyperElastic.Compute_I2(mesh, u, matrixType)
        I3 = HyperElastic.Compute_I3(mesh, u, matrixType)

        dI1dC = HyperElastic.Compute_dI1dC()[..., np.newaxis]
        dI2dC = HyperElastic.Compute_dI2dC(mesh, u, matrixType)[..., np.newaxis]
        dI3dC = HyperElastic.Compute_dI3dC(mesh, u, matrixType)[..., np.newaxis]

        d2I1dC = HyperElastic.Compute_d2I1dC()
        d2I2dC = HyperElastic.Compute_d2I2dC()
        d2I3dC = HyperElastic.Compute_d2I3dC(mesh, u, matrixType)

        dWdI1 = K1 / I3 ** (1 / 3)
        d2WdI1I3 = -K1 / (3 * I3 ** (4 / 3))
        dWdI2 = K2 / I3 ** (2 / 3)
        d2WdI2I3 = -2 * K2 / (3 * I3 ** (5 / 3))
        dWdI3 = -I1 * K1 / (3 * I3 ** (4 / 3)) - 2 * I2 * K2 / (3 * I3 ** (5 / 3))
        d2WdI3I1 = -K1 / (3 * I3 ** (4 / 3))
        d2WdI3I2 = -2 * K2 / (3 * I3 ** (5 / 3))
        d2WdI3I3 = 4 * I1 * K1 / (9 * I3 ** (7 / 3)) + 10 * I2 * K2 / (
            9 * I3 ** (8 / 3)
        )

        d2W = 4 * (dWdI1 * d2I1dC + dWdI2 * d2I2dC + dWdI3 * d2I3dC) + 4 * (
            d2WdI1I3 * dI1dC @ dI3dC.T
            + d2WdI2I3 * dI2dC @ dI3dC.T
            + d2WdI3I1 * dI3dC @ dI1dC.T
            + d2WdI3I2 * dI3dC @ dI2dC.T
            + d2WdI3I3 * dI3dC @ dI3dC.T
        )

        return d2W


# ----------------------------------------------
# Saint-Venant-Kirchhoff
# ----------------------------------------------


class SaintVenantKirchhoff(_HyperElas):
    def __init__(
        self,
        dim: int,
        lmbda: Union[float, _types.FloatArray],
        mu: Union[float, _types.FloatArray],
        thickness=1.0,
    ):
        """Creates Saint-Venant-Kirchhoff material.

        Parameters
        ----------
        dim : int
            dimension (e.g 2 or 3)
        lmbda : float|_types.FloatArray, optional
            Lame's first parameter
        mu : float|_types.FloatArray, optional
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

        W = lmbda * (I1**2 - 6 * I1 + 9) / 8 + mu * (I1**2 - 2 * I1 - 2 * I2 + 3) / 4

        return W

    def Compute_dWde(self, mesh, u, matrixType=MatrixType.rigi) -> FeArray:
        lmbda = self.lmbda
        mu = self.mu
        I1 = HyperElastic.Compute_I1(mesh, u, matrixType)

        dI1dC = HyperElastic.Compute_dI1dC()
        dI2dC = HyperElastic.Compute_dI2dC(mesh, u, matrixType)

        dWdI1 = 2 * I1 * (lmbda / 8 + mu / 4) - 3 * lmbda / 4 - mu / 2
        dWdI2 = -mu / 2

        dW = 2 * (dWdI1 * dI1dC + dWdI2 * dI2dC)

        return dW

    def Compute_d2Wde(self, mesh, u, matrixType=MatrixType.rigi) -> FeArray:
        lmbda = self.lmbda
        mu = self.mu
        I1 = HyperElastic.Compute_I1(mesh, u, matrixType)

        dI1dC = HyperElastic.Compute_dI1dC()[..., np.newaxis]

        d2I1dC = HyperElastic.Compute_d2I1dC()
        d2I2dC = HyperElastic.Compute_d2I2dC()

        dWdI1 = 2 * I1 * (lmbda / 8 + mu / 4) - 3 * lmbda / 4 - mu / 2
        d2WdI1I1 = lmbda / 4 + mu / 2
        dWdI2 = -mu / 2

        d2W = 4 * (dWdI1 * d2I1dC + dWdI2 * d2I2dC) + 4 * (d2WdI1I1 * dI1dC @ dI1dC.T)

        return d2W
