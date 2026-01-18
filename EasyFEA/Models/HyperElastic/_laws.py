# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Hyperelastic laws."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union

# utilities
from ...FEM import MatrixType, FeArray, TensorProd
from ._state import HyperElasticState

# others
from .._utils import _IModel, ModelType, Project_vector_to_matrix
from ...Utilities import _params, _types

# ----------------------------------------------
# Hyper Elastic
# ----------------------------------------------


class _HyperElastic(_IModel, ABC):
    """HyperElastic material.\n
    `NeoHookean`, `MooneyRivlin`, `SaintVenantKirchhoff` and `HolzapfelOgden` inherit from `_HyperElas` class.
    """

    dim: float = _params.ParameterInValues([1, 2, 3])

    thickness: float = _params.PositiveScalarParameter()

    def __init__(self, dim: int, thickness: float):
        self.dim = dim
        self.thickness = thickness

    @property
    def modelType(self) -> ModelType:
        return ModelType.hyperelastic

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
    def Compute_W(self, hyperElasticState: HyperElasticState) -> FeArray:
        """Computes the quadratic energy W(u).

        Parameters
        ----------
        hyperElasticState : HyperElasticState
            Hyperelastic state containing the mesh, the discretized field, and the matrix type.

        Returns
        -------
        FeArray
            We_e_pg of shape (Ne, pg)
        """

        return None  # type: ignore [return-value]

    @abstractmethod
    def Compute_dWde(self, hyperElasticState: HyperElasticState) -> FeArray:
        """Computes the second Piola-Kirchhoff tensor Σ(u).

        Parameters
        ----------
        hyperElasticState : HyperElasticState
            Hyperelastic state containing the mesh, the discretized field, and the matrix type.

        Returns
        -------
        FeArray
            Σ_e_pg of shape (Ne, pg, d), where `d = 1, 3, 6` depending on whether the solution dimension is `1D`, `2D`, or `3D`.

        Σxx, Σyy, Σzz, sqrt(2) Σyz, sqrt(2) Σxz, sqrt(2) Σxy
        """
        return None  # type: ignore [return-value]

    @abstractmethod
    def Compute_d2Wde(self, hyperElasticState: HyperElasticState) -> FeArray:
        """Computes dΣde.

        Parameters
        ----------
        hyperElasticState : HyperElasticState
            Hyperelastic state containing the mesh, the discretized field, and the matrix type.

        Returns
        -------
        FeArray
            dΣde_e_pg of shape (Ne, pg, d, d), where `d = 1, 3, 6` depending on whether the solution dimension is `1D`, `2D`, or `3D`.
        """
        return None  # type: ignore [return-value]

    def Compute_Tangent_and_Residual(
        self, hyperElasticState: HyperElasticState
    ) -> tuple[FeArray, FeArray]:
        """Computes the tangent matrix and the residual vector.

        Parameters
        ----------
        hyperElasticState : HyperElasticState
            Hyperelastic state containing the mesh, the discretized field, and the matrix type.

        Returns
        -------
        tuple[FeArray, FeArray]
            tangent_e_pg (Ne, pg, d, d), with `d = dof_n * mesh.nPe`
        """

        # get params
        Ne, nPg, dim = hyperElasticState._GetDims()
        thickness = self.thickness if dim == 2 else 1

        # get mesh data
        mesh = hyperElasticState.mesh
        nPe = mesh.nPe
        matrixType = MatrixType.rigi
        wJ_e_pg = mesh.Get_weightedJacobian_e_pg(matrixType)
        dN_e_pg = mesh.Get_dN_e_pg(matrixType)

        # get hyper elastic matrices
        De_e_pg = hyperElasticState.Compute_De()
        nCols = De_e_pg.shape[-1]
        dWde_e_pg = self.Compute_dWde(hyperElasticState)
        d2Wde_e_pg = self.Compute_d2Wde(hyperElasticState)

        # init matrices
        grad_e_pg = FeArray.zeros(Ne, nPg, nCols, dim * nPe)
        Sig_e_pg = FeArray.zeros(Ne, nPg, nCols, nCols)
        sig_e_pg = Project_vector_to_matrix(dWde_e_pg)

        # append values in sig_e_pg and grad_e_pg
        rows = np.arange(nCols).reshape(sig_e_pg._shape)
        cols = np.arange(dim * nPe).reshape(dN_e_pg._shape)
        for i in range(dim):
            Sig_e_pg._assemble(rows[i], rows[i], value=sig_e_pg)  # type: ignore [attr-defined]
            grad_e_pg._assemble(rows[i], cols[i], value=dN_e_pg)  # type: ignore [attr-defined]

        # ------------------------------
        # Compute tangent
        # ------------------------------
        # linear part
        B_e_pg = De_e_pg @ grad_e_pg
        linearTangent_e = (wJ_e_pg * B_e_pg.T @ d2Wde_e_pg @ B_e_pg).sum(1)
        # non linear part
        nonLinearTangent_e = (wJ_e_pg * grad_e_pg.T @ Sig_e_pg @ grad_e_pg).sum(1)
        tangent_e = thickness * (linearTangent_e + nonLinearTangent_e)
        # dont use the thickness here!

        # ------------------------------
        # Compute residual
        # ------------------------------
        residual_e = thickness * (wJ_e_pg * dWde_e_pg.T @ B_e_pg).sum(1)

        # ------------------------------
        # Reorder dofs
        # ------------------------------
        # reorder xi,...,xn,yi,...yn,zi,...,zn to xi,yi,zi,...,xn,yx,zn
        reorder = np.arange(0, nPe * dim).reshape(-1, nPe).T.ravel()
        residual_e = residual_e[:, reorder]
        tangent_e = tangent_e[:, reorder][:, :, reorder]

        return tangent_e, residual_e


# ----------------------------------------------
# Neo-Hookean
# ----------------------------------------------


class NeoHookean(_HyperElastic):

    K: float = _params.PositiveScalarParameter()
    """Bulk modulus"""

    def __init__(self, dim: int, K: Union[float, _types.FloatArray], thickness=1.0):
        """Creates an Neo-Hookean material.

        Parameters
        ----------
        dim : int
            dimension (e.g 2 or 3)
        K : float|_types.FloatArray
            Bulk modulus
        thickness : float, optional
            thickness, by default 1.0
        """

        _HyperElastic.__init__(self, dim, thickness)

        self.K = K

    def Compute_W(self, hyperElasticState: HyperElasticState) -> FeArray:
        K = self.K

        I1 = hyperElasticState.Compute_I1()
        I3 = hyperElasticState.Compute_I3()

        W = K * (I1 / I3 ** (1 / 3) - 3)

        return W

    def Compute_dWde(self, hyperElasticState: HyperElasticState) -> FeArray:
        K = self.K

        I1 = hyperElasticState.Compute_I1()
        I3 = hyperElasticState.Compute_I3()

        dWdI1 = K / I3 ** (1 / 3)
        dWdI3 = -I1 * K / (3 * I3 ** (4 / 3))
        dI1dC = hyperElasticState.Compute_dI1dC()
        dI3dC = hyperElasticState.Compute_dI3dC()

        dWdI1 = K / I3 ** (1 / 3)
        dWdI3 = -I1 * K / (3 * I3 ** (4 / 3))
        dW = 2 * (dWdI1 * dI1dC + dWdI3 * dI3dC)

        return dW

    def Compute_d2Wde(self, hyperElasticState: HyperElasticState) -> FeArray:
        K = self.K

        I1 = hyperElasticState.Compute_I1()
        I3 = hyperElasticState.Compute_I3()

        dI1dC = hyperElasticState.Compute_dI1dC()
        dI3dC = hyperElasticState.Compute_dI3dC()
        d2I1dC = hyperElasticState.Compute_d2I1dC()
        d2I3dC = hyperElasticState.Compute_d2I3dC()

        dWdI1 = K / I3 ** (1 / 3)
        d2WdI1dI3 = -K / (3 * I3 ** (4 / 3))
        dWdI3 = -I1 * K / (3 * I3 ** (4 / 3))
        d2WdI3dI1 = -K / (3 * I3 ** (4 / 3))
        d2WdI3dI3 = 4 * I1 * K / (9 * I3 ** (7 / 3))

        d2W = 4 * (dWdI1 * d2I1dC + dWdI3 * d2I3dC) + 4 * (
            d2WdI1dI3 * TensorProd(dI1dC, dI3dC)
            + d2WdI3dI1 * TensorProd(dI3dC, dI1dC)
            + d2WdI3dI3 * TensorProd(dI3dC, dI3dC)
        )

        return d2W


# ----------------------------------------------
# Mooney-Rivlin
# ----------------------------------------------


class MooneyRivlin(_HyperElastic):

    K1: float = _params.PositiveScalarParameter()
    """Kappa1"""

    K2: float = _params.PositiveScalarParameter()
    """Kappa2"""

    K: float = _params.PositiveScalarParameter()
    """Bulk modulus"""

    def __init__(
        self,
        dim: int,
        K1: Union[float, _types.FloatArray],
        K2: Union[float, _types.FloatArray],
        K: Union[float, _types.FloatArray] = 0.0,
        thickness=1.0,
    ):
        """Creates an Mooney-Rivlin material.

        Parameters
        ----------
        dim : int
            dimension (e.g 2 or 3)
        K1 : float|_types.FloatArray
            Kappa1
        K2 : float|_types.FloatArray
            Kappa2 -> Neo-Hoolkean if K2=0
        K : float|_types.FloatArray, optional
            Bulk modulus, by default 0.0
        thickness : float, optional
            thickness, by default 1.0
        """

        _HyperElastic.__init__(self, dim, thickness)

        self.K1 = K1
        self.K2 = K2
        self.K = K

    def Compute_W(self, hyperElasticState: HyperElasticState) -> FeArray:
        K = self.K
        K1 = self.K1
        K2 = self.K2

        I1 = hyperElasticState.Compute_I1()
        I2 = hyperElasticState.Compute_I2()
        I3 = hyperElasticState.Compute_I3()

        W = (
            K * (np.sqrt(I3) - 1) ** 2
            + K1 * (I1 / I3 ** (1 / 3) - 3)
            + K2 * (I2 / I3 ** (2 / 3) - 3)
        )

        return W

    def Compute_dWde(self, hyperElasticState: HyperElasticState) -> FeArray:
        K = self.K
        K1 = self.K1
        K2 = self.K2

        I1 = hyperElasticState.Compute_I1()
        I2 = hyperElasticState.Compute_I2()
        I3 = hyperElasticState.Compute_I3()

        dI1dC = hyperElasticState.Compute_dI1dC()
        dI2dC = hyperElasticState.Compute_dI2dC()
        dI3dC = hyperElasticState.Compute_dI3dC()

        dWdI1 = K1 / I3 ** (1 / 3)
        dWdI2 = K2 / I3 ** (2 / 3)
        dWdI3 = (
            -I1 * K1 / (3 * I3 ** (4 / 3))
            - 2 * I2 * K2 / (3 * I3 ** (5 / 3))
            + K * (np.sqrt(I3) - 1) / np.sqrt(I3)
        )

        dW = 2 * (dWdI1 * dI1dC + dWdI2 * dI2dC + dWdI3 * dI3dC)

        return dW

    def Compute_d2Wde(self, hyperElasticState: HyperElasticState) -> FeArray:
        K = self.K
        K1 = self.K1
        K2 = self.K2

        I1 = hyperElasticState.Compute_I1()
        I2 = hyperElasticState.Compute_I2()
        I3 = hyperElasticState.Compute_I3()

        dI1dC = hyperElasticState.Compute_dI1dC()
        dI2dC = hyperElasticState.Compute_dI2dC()
        dI3dC = hyperElasticState.Compute_dI3dC()

        d2I1dC = hyperElasticState.Compute_d2I1dC()
        d2I2dC = hyperElasticState.Compute_d2I2dC()
        d2I3dC = hyperElasticState.Compute_d2I3dC()

        dWdI1 = K1 / I3 ** (1 / 3)
        d2WdI1dI3 = -K1 / (3 * I3 ** (4 / 3))
        dWdI2 = K2 / I3 ** (2 / 3)
        d2WdI2dI3 = -2 * K2 / (3 * I3 ** (5 / 3))
        dWdI3 = (
            -I1 * K1 / (3 * I3 ** (4 / 3))
            - 2 * I2 * K2 / (3 * I3 ** (5 / 3))
            + K * (np.sqrt(I3) - 1) / np.sqrt(I3)
        )
        d2WdI3dI1 = -K1 / (3 * I3 ** (4 / 3))
        d2WdI3dI2 = -2 * K2 / (3 * I3 ** (5 / 3))
        d2WdI3dI3 = (
            4 * I1 * K1 / (9 * I3 ** (7 / 3))
            + 10 * I2 * K2 / (9 * I3 ** (8 / 3))
            + K / (2 * I3)
            - K * (np.sqrt(I3) - 1) / (2 * I3 ** (3 / 2))
        )

        d2W = 4 * (dWdI1 * d2I1dC + dWdI2 * d2I2dC + dWdI3 * d2I3dC) + 4 * (
            d2WdI1dI3 * TensorProd(dI1dC, dI3dC)
            + d2WdI2dI3 * TensorProd(dI2dC, dI3dC)
            + d2WdI3dI1 * TensorProd(dI3dC, dI1dC)
            + d2WdI3dI2 * TensorProd(dI3dC, dI2dC)
            + d2WdI3dI3 * TensorProd(dI3dC, dI3dC)
        )

        return d2W


# ----------------------------------------------
# Saint-Venant-Kirchhoff
# ----------------------------------------------


class SaintVenantKirchhoff(_HyperElastic):

    lmbda: float = _params.ScalarParameter()
    """Lame's first parameter"""

    mu: float = _params.PositiveScalarParameter()
    """Shear modulus"""

    K: float = _params.PositiveScalarParameter()
    """Bulk modulus"""

    def __init__(
        self,
        dim: int,
        lmbda: Union[float, _types.FloatArray],
        mu: Union[float, _types.FloatArray],
        K: Union[float, _types.FloatArray] = 0.0,
        thickness=1.0,
    ):
        """Creates Saint-Venant-Kirchhoff material.

        Parameters
        ----------
        dim : int
            dimension (e.g 2 or 3)
        lmbda : float|_types.FloatArray
            Lame's first parameter
        mu : float|_types.FloatArray
            Shear modulus
        K : float|_types.FloatArray, optional
            Bulk modulus, by default 0.0
        """

        _HyperElastic.__init__(self, dim, thickness)

        self.lmbda = lmbda
        self.mu = mu
        self.K = K

    def Compute_W(self, hyperElasticState: HyperElasticState) -> FeArray:
        lmbda = self.lmbda
        mu = self.mu
        K = self.K

        I1 = hyperElasticState.Compute_I1()
        I2 = hyperElasticState.Compute_I2()
        I3 = hyperElasticState.Compute_I3()

        W = (
            I1**2 * (lmbda / 8 + mu / 4)
            - I1 * (3 * lmbda / 4 + mu / 2)
            - I2 * mu / 2
            + 0.5 * K * (I3 - 1) ** 2
            + 9 * lmbda / 8
            + 3 * mu / 4
        )

        return W

    def Compute_dWde(self, hyperElasticState: HyperElasticState) -> FeArray:
        lmbda = self.lmbda
        mu = self.mu
        K = self.K

        I1 = hyperElasticState.Compute_I1()
        I3 = hyperElasticState.Compute_I3()

        dI1dC = hyperElasticState.Compute_dI1dC()
        dI2dC = hyperElasticState.Compute_dI2dC()
        dI3dC = hyperElasticState.Compute_dI3dC()

        dWdI1 = 2 * I1 * (lmbda / 8 + mu / 4) - 3 * lmbda / 4 - mu / 2
        dWdI2 = -mu / 2
        dWdI3 = 0.5 * K * (2 * I3 - 2)

        dW = 2 * (dWdI1 * dI1dC + dWdI2 * dI2dC + dWdI3 * dI3dC)

        return dW

    def Compute_d2Wde(self, hyperElasticState: HyperElasticState) -> FeArray:
        lmbda = self.lmbda
        mu = self.mu
        K = self.K

        I1 = hyperElasticState.Compute_I1()
        I3 = hyperElasticState.Compute_I3()

        dI1dC = hyperElasticState.Compute_dI1dC()
        dI3dC = hyperElasticState.Compute_dI3dC()

        d2I1dC = hyperElasticState.Compute_d2I1dC()
        d2I2dC = hyperElasticState.Compute_d2I2dC()
        d2I3dC = hyperElasticState.Compute_d2I3dC()

        dWdI1 = 2 * I1 * (lmbda / 8 + mu / 4) - 3 * lmbda / 4 - mu / 2
        dWdI2 = -mu / 2
        dWdI3 = 0.5 * K * (2 * I3 - 2)

        d2WdI1dI1 = lmbda / 4 + mu / 2
        d2WdI3dI3 = 1.0 * K

        d2W = 4 * (dWdI1 * d2I1dC + dWdI2 * d2I2dC + dWdI3 * d2I3dC) + 4 * (
            d2WdI1dI1 * TensorProd(dI1dC, dI1dC) + d2WdI3dI3 * TensorProd(dI3dC, dI3dC)
        )

        return d2W


# ----------------------------------------------
# Holzapfel-Ogden
# ----------------------------------------------


class HolzapfelOgden(_HyperElastic):

    C0: float = _params.PositiveScalarParameter()
    C1: float = _params.PositiveScalarParameter()
    C2: float = _params.PositiveScalarParameter()
    C3: float = _params.PositiveScalarParameter()
    C4: float = _params.PositiveScalarParameter()
    C5: float = _params.PositiveScalarParameter()
    C6: float = _params.PositiveScalarParameter()
    C7: float = _params.PositiveScalarParameter()

    K: float = _params.PositiveScalarParameter()
    """Bulk modulus"""

    Mu1: float = _params.PositiveScalarParameter()
    Mu2: float = _params.PositiveScalarParameter()

    T1 = _params.VectorParameter()
    """direction(s) 1, used for the invariants I4 and I8"""
    T2 = _params.VectorParameter()
    """direction(s) 2, used for the invariants I6 and I8"""

    __ks: float = _params.PositiveScalarParameter()
    """A positive constant used in the incompressibility penalty term, as proposed in http://dx.doi.org/10.1016/0045-7825(94)90051-5."""

    def __init__(
        self,
        dim: int,
        C0: float,
        C1: float,
        C2: float,
        C3: float,
        C4: float,
        C5: float,
        C6: float,
        C7: float,
        K: float,
        Mu1: float,
        Mu2: float,
        T1: _types.FloatArray,
        T2: _types.FloatArray,
        ks: float = 100,
        thickness=1.0,
    ):
        """Creates Holzapfel-Ogden material.

        Parameters
        ----------
        dim : int
            dimension (e.g 2 or 3)
        C0 : float
            C0
        C1 : float
            C1
        C2 : float
            C2
        C3 : float
            C3
        C4 : float
            C4
        C5 : float
            C5
        C6 : float
            C6
        C7 : float
            C7
        K : float
            bulk modulus
        Mu1 : float
            Mu1
        Mu2 : float
            Mu2
        T1 : _type.FloatArray
            direction(s) 1, used for the invariants I4 and I8
        T2 : _type.FloatArray
            direction(s) 2, used for the invariants I6 and I8
        Mu2 : float
            Mu2
        ks : float, optional
            A positive constant used in the incompressibility penalty term.
        thickness : float, optional
            thickness, by default 1.0
        """

        _HyperElastic.__init__(self, dim, thickness)

        self.C0 = C0
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        self.C6 = C6
        self.C7 = C7

        self.K = K
        self.Mu1 = Mu1
        self.Mu2 = Mu2

        self.T1 = T1
        self.T2 = T2

        self.__ks = ks

    def Compute_W(self, hyperElasticState: HyperElasticState) -> FeArray:
        C0 = self.C0
        C1 = self.C1
        C2 = self.C2
        C3 = self.C3
        C4 = self.C4
        C5 = self.C5
        C6 = self.C6
        C7 = self.C7
        K = self.K
        Mu1 = self.Mu1
        Mu2 = self.Mu2
        T1 = self.T1
        T2 = self.T2
        ks = self.__ks

        I1 = hyperElasticState.Compute_I1()
        I2 = hyperElasticState.Compute_I2()
        I3 = hyperElasticState.Compute_I3()
        I4 = hyperElasticState.Compute_I4(T1)
        I6 = hyperElasticState.Compute_I6(T2)
        I8 = hyperElasticState.Compute_I8(T1, T2)

        W = (
            C0 * (np.exp(C1 * (I1 / I3 ** (1 / 3) - 3)) - 1)
            + C2 * (np.exp(C3 * (I4 - 1) ** 2) - 1) / (1 + np.exp(-ks * (I4 - 1)))
            + C4 * (np.exp(C5 * (I6 - 1) ** 2) - 1) / (1 + np.exp(-ks * (I6 - 1)))
            + C6 * (np.exp(C7 * I8**2) - 1)
            + K * (I3 - 2 * np.log(np.sqrt(I3)) - 1) / 4
            + Mu1 * (I1 / I3 ** (1 / 3) - 3)
            + Mu2 * (I2 / I3 ** (2 / 3) - 3)
        )

        return W

    def Compute_dWde(self, hyperElasticState: HyperElasticState) -> FeArray:
        C0 = self.C0
        C1 = self.C1
        C2 = self.C2
        C3 = self.C3
        C4 = self.C4
        C5 = self.C5
        C6 = self.C6
        C7 = self.C7
        K = self.K
        Mu1 = self.Mu1
        Mu2 = self.Mu2
        T1 = self.T1
        T2 = self.T2
        ks = self.__ks

        I1 = hyperElasticState.Compute_I1()
        I2 = hyperElasticState.Compute_I2()
        I3 = hyperElasticState.Compute_I3()
        I4 = hyperElasticState.Compute_I4(T1)
        I6 = hyperElasticState.Compute_I6(T2)
        I8 = hyperElasticState.Compute_I8(T1, T2)

        dI1dC = hyperElasticState.Compute_dI1dC()
        dI2dC = hyperElasticState.Compute_dI2dC()
        dI3dC = hyperElasticState.Compute_dI3dC()
        dI4dC = hyperElasticState.Compute_dI4dC(T1)
        dI6dC = hyperElasticState.Compute_dI6dC(T2)
        dI8dC = hyperElasticState.Compute_dI8dC(T1, T2)

        # see: examples/HyperElastic/HyperElasticLaws.py
        # TODO Optimize
        # fmt: off
        dWdI1 = C0*C1*np.exp(C1*(I1/I3**(1/3) - 3))/I3**(1/3) + Mu1/I3**(1/3)
        dWdI2 = Mu2/I3**(2/3)
        dWdI3 = -C0*C1*I1*np.exp(C1*(I1/I3**(1/3) - 3))/(3*I3**(4/3)) - I1*Mu1/(3*I3**(4/3)) - 2*I2*Mu2/(3*I3**(5/3)) + K*(1 - 1/I3)/4
        dWdI4 = C2*C3*(2*I4 - 2)*np.exp(C3*(I4 - 1)**2)/(1 + np.exp(-ks*(I4 - 1))) + C2*ks*(np.exp(C3*(I4 - 1)**2) - 1)*np.exp(-ks*(I4 - 1))/(1 + np.exp(-ks*(I4 - 1)))**2
        dWdI6 = C4*C5*(2*I6 - 2)*np.exp(C5*(I6 - 1)**2)/(1 + np.exp(-ks*(I6 - 1))) + C4*ks*(np.exp(C5*(I6 - 1)**2) - 1)*np.exp(-ks*(I6 - 1))/(1 + np.exp(-ks*(I6 - 1)))**2
        dWdI8 = 2*C6*C7*I8*np.exp(C7*I8**2)
        # fmt: on

        dW = 2 * (
            dWdI1 * dI1dC
            + dWdI2 * dI2dC
            + dWdI3 * dI3dC
            + dWdI4 * dI4dC
            + dWdI6 * dI6dC
            + dWdI8 * dI8dC
        )

        return dW

    def Compute_d2Wde(self, hyperElasticState: HyperElasticState) -> FeArray:
        C0 = self.C0
        C1 = self.C1
        C2 = self.C2
        C3 = self.C3
        C4 = self.C4
        C5 = self.C5
        C6 = self.C6
        C7 = self.C7
        K = self.K
        Mu1 = self.Mu1
        Mu2 = self.Mu2
        T1 = self.T1
        T2 = self.T2
        ks = self.__ks

        I1 = hyperElasticState.Compute_I1()
        I2 = hyperElasticState.Compute_I2()
        I3 = hyperElasticState.Compute_I3()
        I4 = hyperElasticState.Compute_I4(T1)
        I6 = hyperElasticState.Compute_I6(T2)
        I8 = hyperElasticState.Compute_I8(T1, T2)

        dI1dC = hyperElasticState.Compute_dI1dC()
        dI2dC = hyperElasticState.Compute_dI2dC()
        dI3dC = hyperElasticState.Compute_dI3dC()
        dI4dC = hyperElasticState.Compute_dI4dC(T1)
        dI6dC = hyperElasticState.Compute_dI6dC(T2)
        dI8dC = hyperElasticState.Compute_dI8dC(T1, T2)

        d2I1dC = hyperElasticState.Compute_d2I1dC()
        d2I2dC = hyperElasticState.Compute_d2I2dC()
        d2I3dC = hyperElasticState.Compute_d2I3dC()
        d2I4dC = hyperElasticState.Compute_d2I4dC()
        d2I6dC = hyperElasticState.Compute_d2I6dC()
        d2I8dC = hyperElasticState.Compute_d2I8dC()

        # see: examples/HyperElastic/HyperElasticLaws.py
        # TODO Optimize
        # fmt: off
        dWdI1 = C0*C1*np.exp(C1*(I1/I3**(1/3) - 3))/I3**(1/3) + Mu1/I3**(1/3)
        d2WdI1dI1 = C0*C1**2*np.exp(C1*(I1/I3**(1/3) - 3))/I3**(2/3)
        d2WdI1dI3 = -C0*C1**2*I1*np.exp(C1*(I1/I3**(1/3) - 3))/(3*I3**(5/3)) - C0*C1*np.exp(C1*(I1/I3**(1/3) - 3))/(3*I3**(4/3)) - Mu1/(3*I3**(4/3))
        dWdI2 = Mu2/I3**(2/3)
        d2WdI2dI3 = -2*Mu2/(3*I3**(5/3))
        dWdI3 = -C0*C1*I1*np.exp(C1*(I1/I3**(1/3) - 3))/(3*I3**(4/3)) - I1*Mu1/(3*I3**(4/3)) - 2*I2*Mu2/(3*I3**(5/3)) + K*(1 - 1/I3)/4
        d2WdI3dI1 = -C0*C1**2*I1*np.exp(C1*(I1/I3**(1/3) - 3))/(3*I3**(5/3)) - C0*C1*np.exp(C1*(I1/I3**(1/3) - 3))/(3*I3**(4/3)) - Mu1/(3*I3**(4/3))
        d2WdI3dI2 = -2*Mu2/(3*I3**(5/3))
        d2WdI3dI3 = C0*C1**2*I1**2*np.exp(C1*(I1/I3**(1/3) - 3))/(9*I3**(8/3)) + 4*C0*C1*I1*np.exp(C1*(I1/I3**(1/3) - 3))/(9*I3**(7/3)) + 4*I1*Mu1/(9*I3**(7/3)) + 10*I2*Mu2/(9*I3**(8/3)) + K/(4*I3**2)
        dWdI4 = C2*C3*(2*I4 - 2)*np.exp(C3*(I4 - 1)**2)/(1 + np.exp(-ks*(I4 - 1))) + C2*ks*(np.exp(C3*(I4 - 1)**2) - 1)*np.exp(-ks*(I4 - 1))/(1 + np.exp(-ks*(I4 - 1)))**2
        d2WdI4dI4 = C2*C3**2*(2*I4 - 2)**2*np.exp(C3*(I4 - 1)**2)/(1 + np.exp(-ks*(I4 - 1))) + 2*C2*C3*ks*(2*I4 - 2)*np.exp(C3*(I4 - 1)**2)*np.exp(-ks*(I4 - 1))/(1 + np.exp(-ks*(I4 - 1)))**2 + 2*C2*C3*np.exp(C3*(I4 - 1)**2)/(1 + np.exp(-ks*(I4 - 1))) - C2*ks**2*(np.exp(C3*(I4 - 1)**2) - 1)*np.exp(-ks*(I4 - 1))/(1 + np.exp(-ks*(I4 - 1)))**2 + 2*C2*ks**2*(np.exp(C3*(I4 - 1)**2) - 1)*np.exp(-2*ks*(I4 - 1))/(1 + np.exp(-ks*(I4 - 1)))**3
        dWdI6 = C4*C5*(2*I6 - 2)*np.exp(C5*(I6 - 1)**2)/(1 + np.exp(-ks*(I6 - 1))) + C4*ks*(np.exp(C5*(I6 - 1)**2) - 1)*np.exp(-ks*(I6 - 1))/(1 + np.exp(-ks*(I6 - 1)))**2
        d2WdI6dI6 = C4*C5**2*(2*I6 - 2)**2*np.exp(C5*(I6 - 1)**2)/(1 + np.exp(-ks*(I6 - 1))) + 2*C4*C5*ks*(2*I6 - 2)*np.exp(C5*(I6 - 1)**2)*np.exp(-ks*(I6 - 1))/(1 + np.exp(-ks*(I6 - 1)))**2 + 2*C4*C5*np.exp(C5*(I6 - 1)**2)/(1 + np.exp(-ks*(I6 - 1))) - C4*ks**2*(np.exp(C5*(I6 - 1)**2) - 1)*np.exp(-ks*(I6 - 1))/(1 + np.exp(-ks*(I6 - 1)))**2 + 2*C4*ks**2*(np.exp(C5*(I6 - 1)**2) - 1)*np.exp(-2*ks*(I6 - 1))/(1 + np.exp(-ks*(I6 - 1)))**3
        dWdI8 = 2*C6*C7*I8*np.exp(C7*I8**2)
        d2WdI8dI8 = 4*C6*C7**2*I8**2*np.exp(C7*I8**2) + 2*C6*C7*np.exp(C7*I8**2)
        # fmt: on

        d2W = 4 * (
            dWdI1 * d2I1dC
            + dWdI2 * d2I2dC
            + dWdI3 * d2I3dC
            + dWdI4 * d2I4dC
            + dWdI6 * d2I6dC
            + dWdI8 * d2I8dC
        ) + 4 * (
            d2WdI1dI1 * TensorProd(dI1dC, dI1dC)
            + d2WdI1dI3 * TensorProd(dI1dC, dI3dC)
            + d2WdI2dI3 * TensorProd(dI2dC, dI3dC)
            + d2WdI3dI1 * TensorProd(dI3dC, dI1dC)
            + d2WdI3dI2 * TensorProd(dI3dC, dI2dC)
            + d2WdI3dI3 * TensorProd(dI3dC, dI3dC)
            + d2WdI4dI4 * TensorProd(dI4dC, dI4dC)
            + d2WdI6dI6 * TensorProd(dI6dC, dI6dC)
            + d2WdI8dI8 * TensorProd(dI8dC, dI8dC)
        )

        return d2W
