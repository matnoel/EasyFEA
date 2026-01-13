# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Elastic laws."""

from abc import ABC, abstractmethod
from typing import Union, Optional
from scipy.linalg import sqrtm

# utilities
import numpy as np

# others
from ...Geoms import AsCoords, Normalize
from .._utils import (
    _IModel,
    ModelType,
    Heterogeneous_Array,
    KelvinMandel_Matrix,
    Project_Kelvin,
    Get_Pmat,
    Apply_Pmat,
)
from ...Utilities import _params, _types
from ...FEM._linalg import TensorProd

# ----------------------------------------------
# Elasticity
# ----------------------------------------------


class _Elastic(_IModel, ABC):
    """Linearized Elasticity material.\n
    ElasIsot, ElasIsotTrans and ElasAnisot inherit from _Elas class.
    """

    def __init__(self, dim: int, thickness: float, planeStress: bool):
        self.dim = dim
        self.planeStress = planeStress
        self.thickness = thickness

    @property
    def modelType(self) -> ModelType:
        return ModelType.elastic

    dim: int = _params.ParameterInValues([2, 3])

    thickness: float = _params.PositiveScalarParameter()

    planeStress: bool = _params.BoolParameter()
    """the model uses plane stress simplification"""

    @property
    def simplification(self) -> str:
        """simplification used for the model"""
        if self.dim == 2:
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
        laws = [Isotropic, TransverselyIsotropic, Anisotropic]
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

    # 23 Cannot be a descriptor due to conflict with `__sqrt_C`.
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

    # 23 Cannot be a descriptor due to conflict with `__sqrt_S`.
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

    def _Apply_basis_transformation(
        self,
        dim: int,
        material_cM: _types.FloatArray,
        material_sM: _types.FloatArray,
        axis_1: _types.FloatArray,
        axis_2: _types.FloatArray,
    ) -> tuple[_types.FloatArray, _types.FloatArray]:
        """Performs a basis transformation from the material's (1,2,3) coordinate system to the (x,y,z) coordinate system to orient the material in space.

        Parameters
        ----------
        dim : int
            dimension
        material_cM : _types.FloatArray
            stiffness matrix
        material_sM : _types.FloatArray
            compliance matrix
        axis_1 : _types.FloatArray
            Axis 1
        axis_2 : _types.FloatArray
            Axis 2

        Returns
        -------
        tuple[_types.FloatArray, _types.FloatArray]
            global_cM, global_sM
        """

        P = Get_Pmat(axis_1=axis_1, axis_2=axis_2, useMandel=True)

        global_sM = Apply_Pmat(P, material_sM, toGlobal=True)
        global_cM = Apply_Pmat(P, material_cM, toGlobal=True)

        testAxis_1 = np.linalg.norm(axis_1 - np.array([1, 0, 0])) <= 1e-12
        testAxis_2 = np.linalg.norm(axis_2 - np.array([0, 1, 0])) <= 1e-12
        if testAxis_1 and testAxis_2:
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


# ----------------------------------------------
# Isotropic
# ----------------------------------------------


class Isotropic(_Elastic):
    """Isotropic Linearized Elastic material."""

    E: float = _params.PositiveParameter()
    """Young's modulus"""

    v: float = _params.IntervalccParameter(inf=-1, sup=0.5)
    """Poisson's ratio (-1<v<0.5)"""

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
        _Elastic.__init__(self, dim, thickness, planeStress)

        self.E = E
        self.v = v

    def _Update(self) -> None:
        C, S = self._Behavior(self.dim)
        self.C = C
        self.S = S

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
        ------

        C -> C : Epsilon = Sigma [Sxx Syy sqrt(2)*Sxy]\n
        S -> S : Sigma = Epsilon [Exx Eyy sqrt(2)*Exy]

        In 3D:
        ------

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

        ci = np.array([c1, c2])
        Ei = np.array([3 * E1, 2 * E2])

        if not self.isHeterogeneous:
            C, S = self._Behavior(3)
            diff_C = C - np.sum([c * E for c, E in zip(ci, Ei)], 0)
            test_C = np.linalg.norm(diff_C, axis=(-2, -1)) / np.linalg.norm(
                C, axis=(-2, -1)
            )
            assert test_C < 1e-12

        return ci, Ei


# ----------------------------------------------
# Transversely isotropic
# ----------------------------------------------


class TransverselyIsotropic(_Elastic):
    """Transversely Isotropic Linearized Elastic material."""

    El: float = _params.PositiveParameter()
    """Longitudinal Young's modulus."""

    Et: float = _params.PositiveParameter()
    """Transverse Young's modulus."""

    Gl: float = _params.PositiveParameter()
    """Longitudinal shear modulus."""

    vl: float = _params.IntervalccParameter(inf=-1, sup=0.5)
    """Longitudinal Poisson's ratio (-1<vl<0.5)."""

    vt: float = _params.IntervalccParameter(inf=-1, sup=1)
    """Transverse Poisson ratio (-1<vt<1)"""

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
        """Creates an Transversely Isotropic Linearized Elastic material.\n
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
        _Elastic.__init__(self, dim, thickness, planeStress)

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

    def _Update(self) -> None:
        C, S = self._Behavior(self.dim)
        self.C = C
        self.S = S

    def _Behavior(self, dim: Optional[int] = None):
        """Updates the constitutives laws by updating the C stiffness and S compliance matrices in Kelvin Mandel notation.\n

        In 2D:
        ------

        C -> C : Epsilon = Sigma [Sxx Syy sqrt(2)*Sxy]\n
        S -> S : Sigma = Epsilon [Exx Eyy sqrt(2)*Exy]

        In 3D:
        ------

        C -> C : Epsilon = Sigma [Sxx Syy Szz sqrt(2)*Syz sqrt(2)*Sxz sqrt(2)*Sxy]\n
        S -> S : Sigma = Epsilon [Exx Eyy Ezz sqrt(2)*Eyz sqrt(2)*Exz sqrt(2)*Exy]

        """

        if dim is None:
            dim = self.dim

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

        return self._Apply_basis_transformation(
            dim=dim,
            material_cM=material_cM,
            material_sM=material_sM,
            axis_1=self.axis_l,
            axis_2=self.axis_t,
        )

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

        ci = np.array([c1, c2, c3, c4, c5])
        Ei = np.array([E1, E2, E3, E4, E5])

        if not self.isHeterogeneous:
            C, S = self._Behavior(3)
            diff_C = C - np.sum([c * E for c, E in zip(ci, Ei)], 0)
            test_C = np.linalg.norm(diff_C, axis=(-2, -1)) / np.linalg.norm(
                C, axis=(-2, -1)
            )
            assert test_C < 1e-12

        return ci, Ei


# ----------------------------------------------
# Orthotropic
# ----------------------------------------------


class Orthotropic(_Elastic):
    """Orthotropic Linearized Elastic material."""

    E1: float = _params.PositiveParameter()
    """Young's modulus along axis_1."""

    E2: float = _params.PositiveParameter()
    """Young's modulus along axis_2."""

    E3: float = _params.PositiveParameter()
    """Young's modulus along axis_3."""

    G23: float = _params.PositiveParameter()
    """Shear modulus in the 2-3 plane."""

    G13: float = _params.PositiveParameter()
    """Shear modulus in the 1-3 plane."""

    G12: float = _params.PositiveParameter()
    """Shear modulus in the 1-2 plane."""

    v23: float = _params.IntervalccParameter(inf=-1, sup=0.5)
    """Poisson's ratio for transverse strain along the axis_3 when stressed along the axis_2."""

    v13: float = _params.IntervalccParameter(inf=-1, sup=0.5)
    """Poisson's ratio for transverse strain along the axis_3 when stressed along the axis_1."""

    v12: float = _params.IntervalccParameter(inf=-1, sup=0.5)
    """Poisson's ratio for transverse strain along the axis_2 when stressed along the axis_1."""

    def __str__(self) -> str:
        text = f"{type(self).__name__}:"
        text += f"\nE1 = {self.E1:.2e}"
        text += f"\nE2 = {self.E2:.2e}"
        text += f"\nE3 = {self.E3:.2e}"
        text += f"\nG23 = {self.G23:.2e}"
        text += f"\nG13 = {self.G13:.2e}"
        text += f"\nG12 = {self.G12:.2e}"
        text += f"\nv23 = {self.v23:.2e}"
        text += f"\nv13 = {self.v13:.2e}"
        text += f"\nv12 = {self.v12:.2e}"
        text += f"\naxis_1 = {np.array_str(self.axis_1, precision=3)}"
        text += f"\naxis_2 = {np.array_str(self.axis_1, precision=3)}"
        if self.dim == 2:
            text += f"\nplaneStress = {self.planeStress}"
            text += f"\nthickness = {self.thickness:.2e}"
        return text

    def __init__(
        self,
        dim: int,
        E1: float,
        E2: float,
        E3: float,
        G23: float,
        G13: float,
        G12: float,
        v23: float,
        v13: float,
        v12: float,
        axis_1: _types.Coords = (1, 0, 0),
        axis_2: _types.Coords = (0, 1, 0),
        planeStress: bool = True,
        thickness: float = 1.0,
    ):
        """Creates Orthotropic Linearized Elastic material.\n
        More details https://www.lusas.com/user_area/faqs/orthotropic.html#:~:text=The%20inverse%20of%20the%20compliance,of%20both%20matrices%20are%20positive

        Parameters
        ----------
        dim : int
            Dimension of 2D or 3D simulation
        E1 : float
            Young's modulus along axis_1.
        E2 : float
            Young's modulus along axis_2.
        E3 : float
            Young's modulus along axis_3.
        G23 : float
            Shear modulus in the 2-3 plane.
        G13 : float
            Shear modulus in the 1-3 plane.
        G12 : float
            Shear modulus in the 1-2 plane.
        v23 : float
            Poisson's ratio for transverse strain along the axis_3 when stressed along the axis_2.
        v13 : float
            Poisson's ratio for transverse strain along the axis_3 when stressed along the axis_1.
        v12 : float
            Poisson's ratio for transverse strain along the axis_2 when stressed along the axis_1.
        axis_1 : _types.Coords, optional
            Axis 1, by default np.array([1,0,0])
        axis_t : _types.Coords, optional
            Axis 2, by default np.array([0,1,0])
        planeStress : bool, optional
            uses plane stress assumption, by default True
        thickness : float, optional
            thickness, by default 1.0
        """
        _Elastic.__init__(self, dim, thickness, planeStress)

        self.E1 = E1
        self.E2 = E2
        self.E3 = E3
        self.G23 = G23
        self.G13 = G13
        self.G12 = G12
        self.v23 = v23
        self.v13 = v13
        self.v12 = v12

        axis_1 = AsCoords(axis_1)
        axis_2 = AsCoords(axis_2)
        assert axis_1.size == 3 and len(axis_1.shape) == 1, "axis_1 must be a 3D vector"
        assert axis_2.size == 3 and len(axis_2.shape) == 1, "axis_2 must be a 3D vector"
        assert axis_1 @ axis_2 <= 1e-12, "axis1 and axis2 must be perpendicular"
        self.__axis_1 = Normalize(axis_1)
        self.__axis_2 = Normalize(axis_2)

    @property
    def axis_1(self) -> _types.FloatArray:
        """Axis 1"""
        return self.__axis_1.copy()

    @property
    def axis_2(self) -> _types.FloatArray:
        """Axis 2"""
        return self.__axis_2.copy()

    def __get_params(self) -> list[Union[float, _types.FloatArray]]:
        """Returns E1, E2, E3, G23, G13, G12, v23, v13, v12"""
        E1 = self.E1
        E2 = self.E2
        E3 = self.E3
        G23 = self.G23
        G13 = self.G13
        G12 = self.G12
        v23 = self.v23
        v13 = self.v13
        v12 = self.v12
        return [E1, E2, E3, G23, G13, G12, v23, v13, v12]

    def __get_cij_denominator(self) -> Union[float, _types.FloatArray]:
        """Returns c11, c22, c33, c23, c13, c12 denominator"""
        E1, E2, E3, _, _, _, v23, v13, v12 = self.__get_params()
        return (
            -E1 * E2
            + E1 * E3 * v23**2
            + E2**2 * v12**2
            + 2 * E2 * E3 * v12 * v13 * v23
            + E2 * E3 * v13**2
        )

    @property
    def _c11(self) -> Union[float, _types.FloatArray]:
        E1, E2, E3, _, _, _, v23, _, _ = self.__get_params()
        return E1**2 * (-E2 + E3 * v23**2) / self.__get_cij_denominator()

    @property
    def _c22(self) -> Union[float, _types.FloatArray]:
        E1, E2, E3, _, _, _, _, v13, _ = self.__get_params()
        return E2**2 * (-E1 + E3 * v13**2) / self.__get_cij_denominator()

    @property
    def _c33(self) -> Union[float, _types.FloatArray]:
        E1, E2, E3, _, _, _, _, _, v12 = self.__get_params()
        return E2 * E3 * (-E1 + E2 * v12**2) / self.__get_cij_denominator()

    @property
    def _c44(self) -> Union[float, _types.FloatArray]:
        return 2 * self.G23

    @property
    def _c55(self) -> Union[float, _types.FloatArray]:
        return 2 * self.G13

    @property
    def _c66(self) -> Union[float, _types.FloatArray]:
        return 2 * self.G12

    @property
    def _c23(self) -> Union[float, _types.FloatArray]:
        E1, E2, E3, _, _, _, v23, v13, v12 = self.__get_params()
        return -E2 * E3 * (E1 * v23 + E2 * v12 * v13) / self.__get_cij_denominator()

    @property
    def _c13(self) -> Union[float, _types.FloatArray]:
        E1, E2, E3, _, _, _, v23, v13, v12 = self.__get_params()
        return -E1 * E2 * E3 * (v12 * v23 + v13) / self.__get_cij_denominator()

    @property
    def _c12(self) -> Union[float, _types.FloatArray]:
        E1, E2, E3, _, _, _, v23, v13, v12 = self.__get_params()
        return -E1 * E2 * (E2 * v12 + E3 * v13 * v23) / self.__get_cij_denominator()

    def _Update(self) -> None:
        C, S = self._Behavior(self.dim)
        self.C = C
        self.S = S

    def _Behavior(self, dim: Optional[int] = None):
        """Updates the constitutives laws by updating the C stiffness and S compliance matrices in Kelvin Mandel notation.\n

        In 2D:
        ------

            C -> C : Epsilon = Sigma [Sxx Syy sqrt(2)*Sxy]\n
            S -> S : Sigma = Epsilon [Exx Eyy sqrt(2)*Exy]

        In 3D:
        ------

            C -> C : Epsilon = Sigma [Sxx Syy Szz sqrt(2)*Syz sqrt(2)*Sxz sqrt(2)*Sxy]\n
            S -> S : Sigma = Epsilon [Exx Eyy Ezz sqrt(2)*Eyz sqrt(2)*Exz sqrt(2)*Exy]

        """

        if dim is None:
            dim = self.dim

        E1, E2, E3, G23, G13, G12, v23, v13, v12 = self.__get_params()

        sum = E1 + E2 + E3 + G23 + G13 + G12 + v23 + v13 + v12
        dtype = object if isinstance(sum, np.ndarray) else float

        # Kelvin-Mandel compliance and stiffness matrices in the material's coordinate system.
        # axis_1 = (1, 0, 0)
        # axis_2 = (0, 1, 0)
        # axis_3 = (0, 0, 1)
        # [11, 22, 33, sqrt(2)*23, sqrt(2)*13, sqrt(2)*12]
        material_sM = np.array(
            [
                [1 / E1, -v12 / E1, -v13 / E1, 0, 0, 0],
                [-v12 / E1, 1 / E2, -v23 / E2, 0, 0, 0],
                [-v13 / E1, -v23 / E2, 1 / E3, 0, 0, 0],
                [0, 0, 0, 1 / (2 * G23), 0, 0],
                [0, 0, 0, 0, 1 / (2 * G13), 0],
                [0, 0, 0, 0, 0, 1 / (2 * G12)],
            ],
            dtype=dtype,
        )

        # tests on S values
        s11, s22, s33 = [material_sM[d, d] for d in range(3)]
        s23, s13, s12 = material_sM[1, 2], material_sM[0, 2], material_sM[0, 1]
        assert np.all(np.abs(s23) < np.sqrt(s22 * s33)), "|s23| < sqrt(s22 * s33)"
        assert np.all(np.abs(s13) < np.sqrt(s11 * s33)), "|s13| < sqrt(s11 * s33)"
        assert np.all(np.abs(s12) < np.sqrt(s11 * s22)), "|s12| < sqrt(s11 * s22)"
        assert np.all(np.abs(v23) < np.sqrt(E2 / E3)), "|v23| < sqrt(E2 / E3)"
        assert np.all(np.abs(v13) < np.sqrt(E1 / E3)), "|v13| < sqrt(E1 / E3)"
        assert np.all(np.abs(v12) < np.sqrt(E1 / E2)), "|v12| < sqrt(E1 / E2)"

        material_sM = Heterogeneous_Array(material_sM)

        material_cM = np.array(
            [
                [self._c11, self._c12, self._c13, 0, 0, 0],
                [self._c12, self._c22, self._c23, 0, 0, 0],
                [self._c13, self._c23, self._c33, 0, 0, 0],
                [0, 0, 0, self._c44, 0, 0],
                [0, 0, 0, 0, self._c55, 0],
                [0, 0, 0, 0, 0, self._c66],
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

        return self._Apply_basis_transformation(
            dim=dim,
            material_cM=material_cM,
            material_sM=material_sM,
            axis_1=self.axis_1,
            axis_2=self.axis_2,
        )

    def Walpole_Decomposition(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        # see section 3.6: https://doi.org/10.1007/s10659-012-9396-z

        a = self.axis_1
        b = self.axis_2
        c = Normalize(np.cross(a, b))

        def tensor_prods(*args: np.ndarray):
            assert len(args) == 4
            tensor_prod = np.einsum("i,j,k,l->ijkl", *args)
            return tensor_prod

        E11 = Project_Kelvin(tensor_prods(a, a, a, a))
        E22 = Project_Kelvin(tensor_prods(b, b, b, b))
        E33 = Project_Kelvin(tensor_prods(c, c, c, c))

        def vec_sym_tensor_prod(v1: np.ndarray, v2: np.ndarray):
            # (ai bj + bi aj )(akb + bka )/2
            p1 = np.einsum("i,j->ij", v1, v2) + np.einsum("i,j->ij", v2, v1)
            p2 = np.einsum("k,l->kl", v1, v2) + np.einsum("k,l->kl", v2, v1)
            return np.einsum("ij,kl->ijkl", p1, p2) / 2

        E44 = Project_Kelvin(vec_sym_tensor_prod(b, c))  # 23
        E55 = Project_Kelvin(vec_sym_tensor_prod(a, c))  # 13
        E66 = Project_Kelvin(vec_sym_tensor_prod(a, b))  # 12

        E23 = Project_Kelvin(tensor_prods(b, b, c, c) + tensor_prods(c, c, b, b))
        E13 = Project_Kelvin(tensor_prods(a, a, c, c) + tensor_prods(c, c, a, a))
        E12 = Project_Kelvin(tensor_prods(a, a, b, b) + tensor_prods(b, b, a, a))

        ci = np.array(
            [
                self._c11,
                self._c22,
                self._c33,
                self._c44,
                self._c55,
                self._c66,
                self._c23,
                self._c13,
                self._c12,
            ]
        )
        Ei = np.array([E11, E22, E33, E44, E55, E66, E23, E13, E12])

        if not self.isHeterogeneous:
            C, S = self._Behavior(3)
            diff_C = C - np.sum([c * E for c, E in zip(ci, Ei)], 0)
            test_C = np.linalg.norm(diff_C, axis=(-2, -1)) / np.linalg.norm(
                C, axis=(-2, -1)
            )
            assert test_C < 1e-12

        return ci, Ei


# ----------------------------------------------
# Anisotropic
# ----------------------------------------------


class Anisotropic(_Elastic):
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
        _Elastic.__init__(self, dim, thickness, False)

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
