# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Linearized elastic laws."""

from abc import ABC, abstractmethod
from typing import Union, Optional
from scipy.linalg import sqrtm

# utilities
import numpy as np

# others
from ..geoms import AsCoords, Normalize
from ._utils import (
    _IModel,
    ModelType,
    Heterogeneous_Array,
    KelvinMandel_Matrix,
    Project_Kelvin,
    Get_Pmat,
    Apply_Pmat,
)
from ..utilities import _params, _types
from ..fem._linalg import TensorProd

# ----------------------------------------------
# Elasticity
# ----------------------------------------------


class _Elas(_IModel, ABC):
    """Linearized Elasticity material.\n
    ElasIsot, ElasIsotTrans and ElasAnisot inherit from _Elas class.
    """

    def __init__(self, dim: int, thickness: float, planeStress: bool):
        assert dim in [2, 3], "Must be dimension 2 or 3"
        self.__dim = dim

        # must set the private value here !
        self.__planeStress = planeStress if dim == 2 else False

        if dim == 2:
            assert thickness > 0, "Must be greater than 0"
            self.__thickness = thickness

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
        """the model uses plane stress simplification"""
        return self.__planeStress

    @planeStress.setter
    def planeStress(self, value: bool) -> None:
        assert isinstance(value, bool)
        if self.__planeStress != value:
            self.Need_Update()
            self.__planeStress = value

    @property
    def simplification(self) -> str:
        """simplification used for the model"""
        if self.__dim == 2:
            return "Plane Stress" if self.planeStress else "Plane Strain"
        else:
            return "3D"

    @abstractmethod
    def _Update(self) -> None:
        """Updates the constitutives laws by updating the C stiffness and S compliance matrices. in Kelvin Mandel notation"""
        pass

    # Model
    @staticmethod
    def Available_Laws():
        laws = [ElasIsot, ElasIsotTrans, ElasAnisot]
        return laws

    @property
    def coef(self) -> float:
        """kelvin mandel coef -> sqrt(2)"""
        return np.sqrt(2)

    @property
    def C(self) -> _types.FloatArray:
        """Stifness matrix in Kelvin Mandel notation such that:\n
        In 2D: C -> C: Epsilon = Sigma [Sxx, Syy, sqrt(2)*Sxy]\n
        In 3D: C -> C: Epsilon = Sigma [Sxx, Syy, Szz, sqrt(2)*Syz, sqrt(2)*Sxz, sqrt(2)*Sxy].\n
        (Lame's law)
        """
        if self.needUpdate:
            self._Update()
            self.Need_Update(False)
        return self.__C.copy()

    @C.setter
    def C(self, array: _types.FloatArray):
        assert isinstance(array, np.ndarray), "must be an array"
        shape = (3, 3) if self.dim == 2 else (6, 6)
        assert (
            array.shape[-2:] == shape
        ), f"With dim = {self.dim} array must be a {shape} matrix"
        self.__C = array
        self.__sqrt_C = None  # dont remove

    @property
    def isHeterogeneous(self) -> bool:
        return len(self.C.shape) > 2

    @property
    def S(self) -> _types.FloatArray:
        """Compliance matrix in Kelvin Mandel notation such that:\n
        In 2D: S -> S : Sigma = Epsilon [Exx, Eyy, sqrt(2)*Exy]\n
        In 3D: S -> S: Sigma = Epsilon [Exx, Eyy, Ezz, sqrt(2)*Eyz, sqrt(2)*Exz, sqrt(2)*Exy].\n
        (Hooke's law)
        """
        if self.needUpdate:
            self._Update()
            self.Need_Update(False)
        return self.__S.copy()

    @S.setter
    def S(self, array: _types.FloatArray):
        assert isinstance(array, np.ndarray), "must be an array"
        shape = (3, 3) if self.dim == 2 else (6, 6)
        assert (
            array.shape[-2:] == shape
        ), f"With dim = {self.dim} array must be a {shape} matrix"
        self.__S = array
        self.__sqrt_S = None  # dont remove

    @abstractmethod
    def Walpole_Decomposition(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        """Walpole's decomposition in Kelvin Mandel notation such that:\n
        C = sum(ci * Ei).\n
        returns ci, Ei"""
        return np.array([]), np.array([])

    def Get_sqrt_C_S(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        """Returns the Matrix square root of C and S."""

        C = self.C

        try:
            self.__sqrt_C is None
            self.__sqrt_S is None
        except AttributeError:
            # init
            self.__sqrt_C: Optional[_types.FloatArray] = None  # type: ignore [no-redef]
            self.__sqrt_S: Optional[_types.FloatArray] = None  # type: ignore [no-redef]

        if self.__sqrt_C is None:
            if self.isHeterogeneous:
                shape = C.shape

                assert (
                    len(shape) == 3
                ), "This function is not currently implemented for heterogeneous matrices where material properties are defined on Gauss points."

                uniq_C, inverse = np.unique(C, return_inverse=True, axis=0)

                sqrtC = np.zeros_like(C, dtype=float)
                sqrtS = np.zeros_like(C, dtype=float)

                for i, C in enumerate(uniq_C):
                    elems = np.where(inverse == i)[0]

                    sqrtmC = sqrtm(C)
                    sqrtC[elems] = sqrtmC

                    sqrtmS = np.linalg.inv(sqrtmC)
                    sqrtS[elems] = sqrtmS

            else:
                sqrtC = sqrtm(C)

                sqrtS = np.linalg.inv(sqrtC)  # faster than sqrtm(self.S)
                # sqrtS = sqrtm(self.S)

                # # give the same results !!!
                # test = np.linalg.norm(sqrtS - sqrtm(self.S))/np.linalg.norm(sqrtS)
                # assert test < 1e-12

            self.__sqrt_C = sqrtC  # type: ignore [assignment]
            self.__sqrt_S = sqrtS  # type: ignore [assignment]

        else:
            sqrtC = self.__sqrt_C.copy()
            sqrtS = self.__sqrt_S.copy()

        return sqrtC, sqrtS


# ----------------------------------------------
# Isotropic
# ----------------------------------------------


class ElasIsot(_Elas):
    """Isotropic Linearized Elastic material."""

    def __str__(self) -> str:
        text = f"{type(self).__name__}:"
        text += f"\nE = {self.E:.2e}, v = {self.v}"
        if self.dim == 2:
            text += f"\nplaneStress = {self.planeStress}"
            text += f"\nthickness = {self.thickness:.2e}"
        return text

    def __init__(self, dim: int, E=210000.0, v=0.3, planeStress=True, thickness=1.0):
        """Creates an Isotropic Linearized Elastic material.

        Parameters
        ----------
        dim : int
            dimension (e.g 2 or 3)
        E : float|np.ndarray, optional
            Young's modulus
        v : float|np.ndarray, optional
            Poisson's ratio ]-1;0.5]
        planeStress : bool, optional
            uses plane stress assumption, by default True
        thickness : float, optional
            thickness, by default 1.0
        """
        _Elas.__init__(self, dim, thickness, planeStress)

        self.E = E
        # TODO Add descriptor with Need_Update() function ?
        self.v = v

    def _Update(self) -> None:
        C, S = self._Behavior(self.dim)
        self.C = C
        self.S = S

    @property
    def E(self) -> Union[float, _types.FloatArray]:
        """Young's modulus"""
        return self.__E

    @E.setter
    def E(self, value):
        _params.CheckIsPositive(value)
        self.Need_Update()
        self.__E = value

    @property
    def v(self) -> Union[float, _types.FloatArray]:
        """Poisson's ratio"""
        return self.__v

    @v.setter
    def v(self, value: float):
        _params.CheckIsInIntervalcc(value, -1, 0.5)
        self.Need_Update()
        self.__v = value

    def get_lambda(self):
        E = self.E
        v = self.v

        lmbda = E * v / ((1 + v) * (1 - 2 * v))

        if self.dim == 2 and self.planeStress:
            lmbda = E * v / (1 - v**2)

        return lmbda

    def get_mu(self):
        """Shear coefficient"""

        E = self.E
        v = self.v

        mu = E / (2 * (1 + v))

        return mu

    def get_bulk(self):
        """Bulk modulus"""

        mu = self.get_mu()
        lmbda = self.get_lambda()

        bulk = lmbda + 2 * mu / self.dim

        return bulk

    def _Behavior(self, dim: Optional[int] = None):
        """Updates the constitutives laws by updating the C stiffness and S compliance matrices in Kelvin Mandel notation.\n

        In 2D:
        -----

        C -> C : Epsilon = Sigma [Sxx Syy sqrt(2)*Sxy]\n
        S -> S : Sigma = Epsilon [Exx Eyy sqrt(2)*Exy]

        In 3D:
        -----

        C -> C : Epsilon = Sigma [Sxx Syy Szz sqrt(2)*Syz sqrt(2)*Sxz sqrt(2)*Sxy]\n
        S -> S : Sigma = Epsilon [Exx Eyy Ezz sqrt(2)*Eyz sqrt(2)*Exz sqrt(2)*Exy]

        """

        if dim is None:
            dim = self.dim
        else:
            assert dim in [2, 3]

        E = self.E
        v = self.v

        mu = self.get_mu()
        lmbda = self.get_lambda()

        dtype = object if True in [isinstance(p, np.ndarray) for p in [E, v]] else float

        if dim == 2:
            # Caution: lambda changes according to 2D simplification.

            cVoigt = np.array(
                [[lmbda + 2 * mu, lmbda, 0], [lmbda, lmbda + 2 * mu, 0], [0, 0, mu]],
                dtype=dtype,
            )

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
            cVoigt = np.array(
                [
                    [lmbda + 2 * mu, lmbda, lmbda, 0, 0, 0],
                    [lmbda, lmbda + 2 * mu, lmbda, 0, 0, 0],
                    [lmbda, lmbda, lmbda + 2 * mu, 0, 0, 0],
                    [0, 0, 0, mu, 0, 0],
                    [0, 0, 0, 0, mu, 0],
                    [0, 0, 0, 0, 0, mu],
                ],
                dtype=dtype,
            )

        cVoigt = Heterogeneous_Array(cVoigt)

        c = KelvinMandel_Matrix(dim, cVoigt)

        s = np.linalg.inv(c)

        return c, s

    def Walpole_Decomposition(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        c1 = self.get_bulk()
        c2 = self.get_mu()

        Ivect = np.array([1, 1, 1, 0, 0, 0])
        Isym = np.eye(6)

        E1 = 1 / 3 * TensorProd(Ivect, Ivect)
        E2 = Isym - E1

        if not self.isHeterogeneous:
            C = self.C
            # only test if the material is heterogeneous
            test_C = np.linalg.norm((3 * c1 * E1 + 2 * c2 * E2) - C) / np.linalg.norm(C)
            assert test_C <= 1e-12

        ci = np.array([c1, c2])
        Ei = np.array([3 * E1, 2 * E2])

        return ci, Ei


# ----------------------------------------------
# Transversely isotropic
# ----------------------------------------------


class ElasIsotTrans(_Elas):
    """Transversely Isotropic Linearized Elastic material."""

    def __str__(self) -> str:
        text = f"{type(self).__name__}:"
        text += f"\nEl = {self.El:.2e}, Et = {self.Et:.2e}, Gl = {self.Gl:.2e}"
        text += f"\nvl = {self.vl}, vt = {self.vt}"
        text += f"\naxis_l = {np.array_str(self.axis_l, precision=3)}"
        text += f"\naxis_t = {np.array_str(self.axis_t, precision=3)}"
        if self.dim == 2:
            text += f"\nplaneStress = {self.planeStress}"
            text += f"\nthickness = {self.thickness:.2e}"
        return text

    def __init__(
        self,
        dim: int,
        El: float,
        Et: float,
        Gl: float,
        vl: float,
        vt: float,
        axis_l: _types.Coords = (1, 0, 0),
        axis_t: _types.Coords = (0, 1, 0),
        planeStress: bool = True,
        thickness: float = 1.0,
    ):
        """Creates and Transversely Isotropic Linearized Elastic material.\n
        More details Torquato 2002 13.3.2 (iii) http://link.springer.com/10.1007/978-1-4757-6355-3

        Parameters
        ----------
        dim : int
            Dimension of 2D or 3D simulation
        El : float
            Longitudinal Young's modulus
        Et : float
            Transverse Young's modulus (T, R) plane
        Gl : float
            Longitudinal shear modulus
        vl : float
            Longitudinal Poisson ratio
        vt : float
            Transverse Poisson ratio (T, R) plane
        axis_l : _types.Coords, optional
            Longitudinal axis, by default np.array([1,0,0])
        axis_t : _types.Coords, optional
            Transverse axis, by default np.array([0,1,0])
        planeStress : bool, optional
            uses plane stress assumption, by default True
        thickness : float, optional
            thickness, by default 1.0
        """
        _Elas.__init__(self, dim, thickness, planeStress)

        self.El = El
        self.Et = Et
        self.Gl = Gl
        self.vl = vl
        self.vt = vt

        axis_l = AsCoords(axis_l)
        axis_t = AsCoords(axis_t)
        assert axis_l.size == 3 and len(axis_l.shape) == 1, "axis_l must be a 3D vector"
        assert axis_t.size == 3 and len(axis_t.shape) == 1, "axis_t must be a 3D vector"
        assert axis_l @ axis_t <= 1e-12, "axis1 and axis2 must be perpendicular"
        self.__axis_l = Normalize(axis_l)
        self.__axis_t = Normalize(axis_t)

    @property
    def Gt(self) -> Union[float, _types.FloatArray]:
        """Transverse shear modulus."""

        Et = self.Et
        vt = self.vt

        Gt = Et / (2 * (1 + vt))

        return Gt

    @property
    def El(self) -> Union[float, _types.FloatArray]:
        """Longitudinal Young's modulus."""
        return self.__El

    @El.setter
    def El(self, value: Union[float, _types.FloatArray]):
        _params.CheckIsPositive(value)
        self.Need_Update()
        self.__El = value

    @property
    def Et(self) -> Union[float, _types.FloatArray]:
        """Transverse Young's modulus."""
        return self.__Et

    @Et.setter
    def Et(self, value: Union[float, _types.FloatArray]):
        _params.CheckIsPositive(value)
        self.Need_Update()
        self.__Et = value

    @property
    def Gl(self) -> Union[float, _types.FloatArray]:
        """Longitudinal shear modulus."""
        return self.__Gl

    @Gl.setter
    def Gl(self, value: Union[float, _types.FloatArray]):
        _params.CheckIsPositive(value)
        self.Need_Update()
        self.__Gl = value

    @property
    def vl(self) -> Union[float, _types.FloatArray]:
        """Longitudinal Poisson's ratio."""
        return self.__vl

    @vl.setter
    def vl(self, value: Union[float, _types.FloatArray]):
        # -1<vl<0.5
        # Torquato 328
        _params.CheckIsInIntervalcc(value, -1, 0.5)
        self.Need_Update()
        self.__vl = value

    @property
    def vt(self) -> Union[float, _types.FloatArray]:
        """Transverse Poisson ratio"""
        return self.__vt

    @vt.setter
    def vt(self, value: Union[float, _types.FloatArray]):
        # -1<vt<1
        # Torquato 328
        _params.CheckIsInIntervalcc(value, -1, 1)
        self.Need_Update()
        self.__vt = value

    @property
    def kt(self) -> Union[float, _types.FloatArray]:
        # Torquato 2002 13.3.2 (iii)
        El = self.El
        Et = self.Et
        vtt = self.vt
        vtl = self.vl
        kt = El * Et / ((2 * (1 - vtt) * El) - (4 * vtl**2 * Et))

        return kt

    @property
    def axis_l(self) -> _types.FloatArray:
        """Longitudinal axis"""
        return self.__axis_l.copy()

    @property
    def axis_t(self) -> _types.FloatArray:
        """Transversal axis"""
        return self.__axis_t.copy()

    @property
    def _useSameAxis(self) -> bool:
        testAxis_l = np.linalg.norm(self.axis_l - np.array([1, 0, 0])) <= 1e-12
        testAxis_t = np.linalg.norm(self.axis_t - np.array([0, 1, 0])) <= 1e-12
        if testAxis_l and testAxis_t:
            return True
        else:
            return False

    def _Update(self) -> None:
        C, S = self._Behavior(self.dim)
        self.C = C
        self.S = S

    def _Behavior(
        self, dim: Optional[int] = None, P: Optional[_types.FloatArray] = None
    ):
        """Updates the constitutives laws by updating the C stiffness and S compliance matrices in Kelvin Mandel notation.\n

        In 2D:
        -----

        C -> C : Epsilon = Sigma [Sxx Syy sqrt(2)*Sxy]\n
        S -> S : Sigma = Epsilon [Exx Eyy sqrt(2)*Exy]

        In 3D:
        -----

        C -> C : Epsilon = Sigma [Sxx Syy Szz sqrt(2)*Syz sqrt(2)*Sxz sqrt(2)*Sxy]\n
        S -> S : Sigma = Epsilon [Exx Eyy Ezz sqrt(2)*Eyz sqrt(2)*Exz sqrt(2)*Exy]

        """

        if dim is None:
            dim = self.dim

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

        # Kelvin-Mandel compliance and stiffness matrices in the material's coordinate system.
        # L = (1, 0, 0)
        # T = (0, 1, 0)
        # R = (0, 0, 1)
        # [11, 22, 33, sqrt(2)*23, sqrt(2)*13, sqrt(2)*12]

        material_sM = np.array(
            [
                [1 / El, -vl / El, -vl / El, 0, 0, 0],
                [-vl / El, 1 / Et, -vt / Et, 0, 0, 0],
                [-vl / El, -vt / Et, 1 / Et, 0, 0, 0],
                [0, 0, 0, 1 / (2 * Gt), 0, 0],
                [0, 0, 0, 0, 1 / (2 * Gl), 0],
                [0, 0, 0, 0, 0, 1 / (2 * Gl)],
            ],
            dtype=dtype,
        )

        material_sM = Heterogeneous_Array(material_sM)

        material_cM = np.array(
            [
                [El + 4 * vl**2 * kt, 2 * kt * vl, 2 * kt * vl, 0, 0, 0],
                [2 * kt * vl, kt + Gt, kt - Gt, 0, 0, 0],
                [2 * kt * vl, kt - Gt, kt + Gt, 0, 0, 0],
                [0, 0, 0, 2 * Gt, 0, 0],
                [0, 0, 0, 0, 2 * Gl, 0],
                [0, 0, 0, 0, 0, 2 * Gl],
            ],
            dtype=dtype,
        )

        material_cM = Heterogeneous_Array(material_cM)

        if len(material_cM.shape) == 2:
            # checks that S = C^-1
            diff_S = np.linalg.norm(
                material_sM - np.linalg.inv(material_cM), axis=(-2, -1)
            ) / np.linalg.norm(material_sM, axis=(-2, -1))
            assert np.max(diff_S) < 1e-12
            # checks that C = S^-1
            diff_C = np.linalg.norm(
                material_cM - np.linalg.inv(material_sM), axis=(-2, -1)
            ) / np.linalg.norm(material_cM, axis=(-2, -1))
            assert np.max(diff_C) < 1e-12

        # Perform a basis transformation from the material's (L,T,R) coordinate system
        # to the (x,y,z) coordinate system to orient the material in space.
        global_sM = Apply_Pmat(P, material_sM)
        global_cM = Apply_Pmat(P, material_cM)

        if useSameAxis:
            # check that if the axes does not change, the same constitutive law is obtained
            test_diff_c = np.linalg.norm(
                global_cM - material_cM, axis=(-2, -1)
            ) / np.linalg.norm(material_cM, axis=(-2, -1))
            assert np.max(test_diff_c) < 1e-12

            test_diff_s = np.linalg.norm(
                global_sM - material_sM, axis=(-2, -1)
            ) / np.linalg.norm(material_sM, axis=(-2, -1))
            assert np.max(test_diff_s) < 1e-12

        c = global_cM
        s = global_sM

        if dim == 2:
            x = np.array([0, 1, 5])

            shape = c.shape

            if self.planeStress:
                if len(shape) == 2:
                    s = global_sM[x, :][:, x]
                elif len(shape) == 3:
                    s = global_sM[:, x, :][:, :, x]
                elif len(shape) == 4:
                    s = global_sM[:, :, x, :][:, :, :, x]

                c = np.linalg.inv(s)

            else:
                if len(shape) == 2:
                    c = global_cM[x, :][:, x]
                elif len(shape) == 3:
                    c = global_cM[:, x, :][:, :, x]
                elif len(shape) == 4:
                    c = global_cM[:, :, x, :][:, :, :, x]

                s = np.linalg.inv(c)

        return c, s

    def Walpole_Decomposition(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        El = self.El
        Gl = self.Gl
        vl = self.vl
        kt = self.kt
        Gt = self.Gt

        c1 = El + 4 * vl**2 * kt
        c2 = 2 * kt
        c3 = 2 * np.sqrt(2) * kt * vl
        c4 = 2 * Gt
        c5 = 2 * Gl

        n = self.axis_l
        p = TensorProd(n, n)
        q = np.eye(3) - p

        E1 = Project_Kelvin(TensorProd(p, p))
        E2 = Project_Kelvin(1 / 2 * TensorProd(q, q))
        E3 = Project_Kelvin(1 / np.sqrt(2) * (TensorProd(p, q) + TensorProd(q, p)))
        E4 = Project_Kelvin(TensorProd(q, q, True) - 1 / 2 * TensorProd(q, q))
        I = Project_Kelvin(TensorProd(np.eye(3), np.eye(3), True))
        E5 = I - E1 - E2 - E4

        if not self.isHeterogeneous:
            P = Get_Pmat(self.axis_l, self.axis_t)
            C, S = self._Behavior(3, P)
            diff_C = C - (c1 * E1 + c2 * E2 + c3 * E3 + c4 * E4 + c5 * E5)
            test_C = np.linalg.norm(diff_C, axis=(-2, -1)) / np.linalg.norm(
                C, axis=(-2, -1)
            )
            assert test_C < 1e-12

        ci = np.array([c1, c2, c3, c4, c5])
        Ei = np.array([E1, E2, E3, E4, E5])

        return ci, Ei


# ----------------------------------------------
# Anisotropic
# ----------------------------------------------


class ElasAnisot(_Elas):
    """Anisotropic Linearized Elastic material."""

    def __str__(self) -> str:
        text = f"\n{type(self).__name__}):"
        text += f"\n{self.C}"
        text += f"\naxis1 = {np.array_str(self.__axis1, precision=3)}"
        text += f"\naxis2 = {np.array_str(self.__axis2, precision=3)}"
        if self.dim == 2:
            text += f"\nplaneStress = {self.planeStress}"
            text += f"\nthickness = {self.thickness:.2e}"
        return text

    def __init__(
        self,
        dim: int,
        C: _types.FloatArray,
        useVoigtNotation: bool,
        axis1: _types.Coords = (1, 0, 0),
        axis2: _types.Coords = (0, 1, 0),
        thickness=1.0,
    ):
        """Creates an Anisotropic Linearized Elastic class.

        Parameters
        ----------
        dim : int
            dimension
        C : _types.FloatArray
            stiffness matrix in anisotropy basis
        useVoigtNotation : bool
            behavior law uses voigt notation
        axis1 : _types.Coords, optional
            axis1 vector, by default (1,0,0)
        axis2 : _types.Coords
            axis2 vector, by default (0,1,0)
        thickness: float, optional
            material thickness, by default 1.0

        Returns
        -------
        ElasAnisot
            Anisotropic behavior law
        """
        # here planeStress is set to False because we just know the C matrix
        _Elas.__init__(self, dim, thickness, False)

        axis1 = AsCoords(axis1)
        axis2 = AsCoords(axis2)
        assert axis1.size == 3 and len(axis1.shape) == 1, "axis1 must be a 3D vector"
        assert axis2.size == 3 and len(axis2.shape) == 1, "axis2 must be a 3D vector"
        assert axis1 @ axis2 <= 1e-12, "axis1 and axis2 must be perpendicular"
        self.__axis1 = Normalize(axis1)
        self.__axis2 = Normalize(axis2)

        self.Set_C(C, useVoigtNotation)

    def _Update(self) -> None:
        # doesn't do anything here, because we use Set_C to update the laws.
        return super()._Update()

    def Set_C(self, C: _types.FloatArray, useVoigtNotation=True, update_S=True):
        """Updates the constitutives laws by updating the C stiffness and S compliance matrices in Kelvin Mandel notation.\n

        Parameters
        ----------
        C : _types.FloatArray
           Stifness matrix (Lamé's law)
        useVoigtNotation : bool, optional
            uses Kevin Mandel's notation, by default True
        update_S : bool, optional
            updates the compliance matrix (Hooke's law), by default True
        """

        self.Need_Update()

        C_mandelP = self._Behavior(C, useVoigtNotation)
        self.C = C_mandelP

        if update_S:
            S_mandelP = np.linalg.inv(C_mandelP)
            self.S = S_mandelP

    def _Behavior(
        self, C: _types.FloatArray, useVoigtNotation: bool
    ) -> _types.FloatArray:
        shape = C.shape
        assert (shape[-2], shape[-1]) in [
            (3, 3),
            (6, 6),
        ], "C must be a (3,3) or (6,6) matrix"
        dim = 3 if C.shape[-1] == 6 else 2
        if len(C.shape) == 2:
            Ct = C.T
        elif len(C.shape) == 3:
            Ct = np.transpose(C, (0, 2, 1))
        elif len(C.shape) == 4:
            Ct = np.transpose(C, (0, 1, 3, 2))
        else:
            raise ValueError(
                "This matrix must be of dimensions (dim, dim), (Ne, dim, dim) or (Ne, nPg, dim, dim)."
            )

        testSym = np.linalg.norm(Ct - C, axis=(-2, -1)) / np.linalg.norm(
            C, axis=(-2, -1)
        )
        assert np.max(testSym) <= 1e-12, "The matrix is not symmetrical."

        if useVoigtNotation:
            C_mandel = KelvinMandel_Matrix(dim, C)
        else:
            C_mandel = C.copy()

        # sets to 3D
        idx = np.array([0, 1, 5])
        if dim == 2:
            if len(shape) == 2:
                C_mandel_global = np.zeros((6, 6))
                for i, I in enumerate(idx):
                    for j, J in enumerate(idx):
                        C_mandel_global[I, J] = C_mandel[i, j]
            if len(shape) == 3:
                C_mandel_global = np.zeros((shape[0], 6, 6))
                for i, I in enumerate(idx):
                    for j, J in enumerate(idx):
                        C_mandel_global[:, I, J] = C_mandel[:, i, j]
            elif len(shape) == 4:
                C_mandel_global = np.zeros((shape[0], shape[1], 6, 6))
                for i, I in enumerate(idx):
                    for j, J in enumerate(idx):
                        C_mandel_global[:, :, I, J] = C_mandel[:, :, i, j]
        else:
            C_mandel_global = C

        P = Get_Pmat(self.__axis1, self.__axis2)

        C_mandelP_global = Apply_Pmat(P, C_mandel_global)

        if self.dim == 2:
            if len(shape) == 2:
                C_mandelP = C_mandelP_global[idx, :][:, idx]
            if len(shape) == 3:
                C_mandelP = C_mandelP_global[:, idx, :][:, :, idx]
            elif len(shape) == 4:
                C_mandelP = C_mandelP_global[:, :, idx, :][:, :, :, idx]

        else:
            C_mandelP = C_mandelP_global

        return C_mandelP

    @property
    def axis1(self) -> _types.FloatArray:
        """axis1 vector"""
        return self.__axis1.copy()

    @property
    def axis2(self) -> _types.FloatArray:
        """axis2 vector"""
        return self.__axis2.copy()

    def Walpole_Decomposition(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        return super().Walpole_Decomposition()
