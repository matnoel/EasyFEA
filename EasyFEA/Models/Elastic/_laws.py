# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Elastic laws."""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING

# utilities
import numpy as np

# others
from ...Geoms import AsCoords
from .._utils import _IModel, _Format_parameter, Heterogeneous_Array
from ...FEM import _kelvin_mandel as kelvin_mandel
from ...Utilities import _params, _types
from ...FEM._linalg import TensorProd, FeArray

if TYPE_CHECKING:
    from ...FEM import Kinematics

# ----------------------------------------------
# Elasticity
# ----------------------------------------------


class _Elastic(_IModel, ABC):
    """Linearized elastic material: a law gives its material-frame Kelvin–Mandel C, the base rotates it, applies the 2D hypothesis, inverts and caches."""

    def __init__(
        self,
        dim: int,
        thickness: float,
        planeStress: bool,
    ):
        self.dim = dim
        self.planeStress = planeStress
        self.thickness = thickness

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
    def _Material_C(self) -> _types.FloatArray:
        """Kelvin–Mandel C in the material frame, (…, 6, 6); Anisotropic may give a 2D-only (…, 3, 3)."""

    # Model
    @staticmethod
    def Available_Laws():
        laws = [Isotropic, TransverselyIsotropic, Orthotropic, Anisotropic]
        return laws

    @cached_property
    def __C_S(self) -> tuple:
        """(C3, C, S), C3 None for a 2D-only C."""
        materialC = np.asarray(self._Material_C(), dtype=float)
        C3: _types.FloatArray | None = None
        if materialC.shape[-2:] == (3, 3):
            if self.dim == 3 or self.planeStress:
                raise ValueError(
                    "A (3, 3) C is 2D only, with no plane stress: give a (6, 6) C."
                )
            C = materialC
            S = np.linalg.inv(C)
        else:
            P = kelvin_mandel.Get_Pmat(*self._Axes())
            C3 = kelvin_mandel.Apply_Pmat(P, materialC)
            if self.dim == 3:
                C = C3
                S = np.linalg.inv(C)
            elif self.planeStress:
                S = kelvin_mandel.Reduce(np.linalg.inv(C3), 2)
                C = np.linalg.inv(S)
            else:
                C = kelvin_mandel.Reduce(C3, 2)
                S = np.linalg.inv(C)
        return C3, C, S

    @property
    def C(self) -> _types.FloatArray:
        """Stiffness in Kelvin–Mandel notation, model dimension, global frame: ``σ = C : ε``."""
        return self.__C_S[1].copy()

    @property
    def S(self) -> _types.FloatArray:
        """Compliance in Kelvin–Mandel notation, model dimension, global frame: ``ε = S : σ``."""
        return self.__C_S[2].copy()

    def _Get_C_3D(self) -> _types.FloatArray:
        """3D stiffness in Kelvin–Mandel notation, global frame, whatever ``dim``."""
        C3 = self.__C_S[0]
        if C3 is None:
            raise ValueError("A (3, 3) C is 2D only: it has no 3D stiffness.")
        return C3.copy()

    @property
    def isHeterogeneous(self) -> bool:
        return self.__C_S[1].ndim > 2

    def Compute_Sigma(self, kinematics: "Kinematics") -> FeArray.FeArrayALike:
        """Stress ``σ = C : ε`` in Kelvin-Mandel form, shape (Ne, nPg, 3 or 6)."""
        Epsilon_e_pg = kinematics.Compute_Epsilon()
        Ne, nPg, nS = Epsilon_e_pg.shape
        C_e_pg = FeArray.broadcast(self.C, Ne, nPg, tensor_shape=(nS, nS))
        return C_e_pg @ Epsilon_e_pg

    def Compute_Psi(self, kinematics: "Kinematics") -> FeArray.FeArrayALike:
        """Elastic energy density ``ψ = 1/2 σ : ε``, shape (Ne, nPg)."""
        Epsilon_e_pg = kinematics.Compute_Epsilon()
        Sigma_e_pg = self.Compute_Sigma(kinematics)
        return 0.5 * Sigma_e_pg @ Epsilon_e_pg

    @abstractmethod
    def Walpole_Decomposition(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        """Walpole's decomposition in Kelvin Mandel notation such that:\n
        C = sum(ci * Ei).\n
        returns ci, Ei"""
        return np.array([]), np.array([])

    @cached_property
    def __sqrt_C_S(self) -> tuple:
        # C is symmetric positive definite: eigh gives the principal square root, over every leading axis at once
        lam, Q = np.linalg.eigh(self.__C_S[1])
        assert lam.min() > 0, "C must be positive definite"
        sqrt_lam = np.sqrt(lam)[..., np.newaxis, :]
        Qt = np.swapaxes(Q, -2, -1)
        return (Q * sqrt_lam) @ Qt, (Q / sqrt_lam) @ Qt

    def Get_sqrt_C_S(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        """Returns the matrix square root of C and S, for a C of any shape (..., d, d)."""
        sqrtC, sqrtS = self.__sqrt_C_S
        return sqrtC.copy(), sqrtS.copy()

    def _Axes(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        """The 2 first axes of the material frame, (…, 3) each."""
        return np.array([1.0, 0, 0]), np.array([0, 1.0, 0])

    def _Frame_fields(self) -> list[FeArray.FeArrayALike]:
        """The 3 unit frame axes: (3,) each, or (Ne, nPg, 3) FeArrays, (Ne, 3) held at one point."""
        axis_1, axis_2 = kelvin_mandel.Normalise_axes(*self._Axes())
        axes = [axis_1, axis_2, np.cross(axis_1, axis_2)]
        if axes[0].ndim == 1:
            return axes
        return [FeArray.asfearray(a if a.ndim == 3 else a[:, np.newaxis]) for a in axes]

    def _Walpole(
        self, ci: list, Ei: list, check=True
    ) -> tuple[_types.FloatArray, _types.FloatArray]:
        """(k, …) moduli and (k, …, 6, 6) tensors at the frame's points; asserts ``Σ cᵢ Eᵢ`` is the 3D C when ``check`` and the moduli are uniform."""
        lead = np.shape(self._Axes()[0])[:-1]
        ci_ = np.stack(np.broadcast_arrays(*ci))
        Ei_ = np.stack([np.asarray(E).reshape(*lead, 6, 6) for E in Ei])
        if check and ci_.ndim == 1:
            C = self._Get_C_3D()
            diff_C = C - np.tensordot(ci_, Ei_, axes=1)
            test_C = np.linalg.norm(diff_C, axis=(-2, -1)) / np.linalg.norm(
                C, axis=(-2, -1)
            )
            assert np.max(test_C) < 1e-12
        return ci_, Ei_


# ----------------------------------------------
# Isotropic
# ----------------------------------------------


class Isotropic(_Elastic):
    """Isotropic Linearized Elastic material."""

    E: float = _params.PositiveParameter()
    """Young's modulus"""

    v: float = _params.IntervalooParameter(inf=-1, sup=0.5)
    """Poisson's ratio (-1<v<0.5)"""

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

    def get_lambda(self, dim: int | None = None):
        """First Lamé coefficient in ``dim`` (the model's by default), reduced under plane stress."""
        E = self.E
        v = self.v

        lmbda = E * v / ((1 + v) * (1 - 2 * v))

        if (self.dim if dim is None else dim) == 2 and self.planeStress:
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

    def _Material_C(self) -> _types.FloatArray:
        lmbda = self.get_lambda(3)
        mu = self.get_mu()
        return Heterogeneous_Array(
            [
                [lmbda + 2 * mu, lmbda, lmbda, 0, 0, 0],
                [lmbda, lmbda + 2 * mu, lmbda, 0, 0, 0],
                [lmbda, lmbda, lmbda + 2 * mu, 0, 0, 0],
                [0, 0, 0, 2 * mu, 0, 0],
                [0, 0, 0, 0, 2 * mu, 0],
                [0, 0, 0, 0, 0, 2 * mu],
            ]
        )

    def Walpole_Decomposition(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        c1 = self.get_bulk()
        c2 = self.get_mu()

        Ivect = np.array([1, 1, 1, 0, 0, 0])
        Isym = np.eye(6)

        E1 = 1 / 3 * TensorProd(Ivect, Ivect)
        E2 = Isym - E1

        # under 2D plane stress c1 is the reduced bulk, not the 3D one
        return self._Walpole(
            [c1, c2], [3 * E1, 2 * E2], not (self.dim == 2 and self.planeStress)
        )


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

    vl: float = _params.IntervalooParameter(inf=-1, sup=0.5)
    """Longitudinal Poisson's ratio (-1<vl<0.5)."""

    vt: float = _params.IntervalooParameter(inf=-1, sup=1)
    """Transverse Poisson ratio (-1<vt<1)"""

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
        _Elastic.__init__(
            self,
            dim,
            thickness,
            planeStress,
        )
        self.axis_l = AsCoords(axis_l)
        self.axis_t = AsCoords(axis_t)

        self.El = El
        self.Et = Et
        self.Gl = Gl
        self.vl = vl
        self.vt = vt

    @property
    def Gt(self) -> float | _types.FloatArray:
        """Transverse shear modulus."""

        Et = self.Et
        vt = self.vt

        Gt = Et / (2 * (1 + vt))

        return Gt

    @property
    def kt(self) -> float | _types.FloatArray:
        # Torquato 2002 13.3.2 (iii)
        El = self.El
        Et = self.Et
        vtt = self.vt
        vtl = self.vl
        kt = El * Et / ((2 * (1 - vtt) * El) - (4 * vtl**2 * Et))

        return kt

    axis_l: _types.FloatArray = _params.VectorParameter()
    """Longitudinal axis, (…, 3)."""

    axis_t: _types.FloatArray = _params.VectorParameter()
    """Transverse axis, (…, 3)."""

    def _Axes(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        return self.axis_l, self.axis_t

    def _Material_C(self) -> _types.FloatArray:
        # axes (l, t, r) = (1, 2, 3)
        El = self.El
        vl = self.vl
        Gl = self.Gl
        Gt = self.Gt
        kt = self.kt
        return Heterogeneous_Array(
            [
                [El + 4 * vl**2 * kt, 2 * kt * vl, 2 * kt * vl, 0, 0, 0],
                [2 * kt * vl, kt + Gt, kt - Gt, 0, 0, 0],
                [2 * kt * vl, kt - Gt, kt + Gt, 0, 0, 0],
                [0, 0, 0, 2 * Gt, 0, 0],
                [0, 0, 0, 0, 2 * Gl, 0],
                [0, 0, 0, 0, 0, 2 * Gl],
            ]
        )

    def Walpole_Decomposition(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        El = self.El
        Gl = self.Gl
        vl = self.vl
        kt = self.kt
        Gt = self.Gt

        c1 = El + 4 * vl**2 * kt
        c2 = 2 * kt
        c3 = 2 * kelvin_mandel.R2 * kt * vl
        c4 = 2 * Gt
        c5 = 2 * Gl

        n = self._Frame_fields()[0]
        p = TensorProd(n, n)
        q = np.eye(3) - p

        E1 = kelvin_mandel.Tensor_to_Kelvin(TensorProd(p, p))
        E2 = kelvin_mandel.Tensor_to_Kelvin(1 / 2 * TensorProd(q, q))
        E3 = kelvin_mandel.Tensor_to_Kelvin(
            1 / kelvin_mandel.R2 * (TensorProd(p, q) + TensorProd(q, p))
        )
        E4 = kelvin_mandel.Tensor_to_Kelvin(
            TensorProd(q, q, True) - 1 / 2 * TensorProd(q, q)
        )
        E5 = np.eye(6) - E1 - E2 - E4

        return self._Walpole([c1, c2, c3, c4, c5], [E1, E2, E3, E4, E5])


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

    v23: float = _params.IntervalooParameter(inf=-1, sup=0.5)
    """Poisson's ratio for transverse strain along the axis_3 when stressed along the axis_2."""

    v13: float = _params.IntervalooParameter(inf=-1, sup=0.5)
    """Poisson's ratio for transverse strain along the axis_3 when stressed along the axis_1."""

    v12: float = _params.IntervalooParameter(inf=-1, sup=0.5)
    """Poisson's ratio for transverse strain along the axis_2 when stressed along the axis_1."""

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
        _Elastic.__init__(
            self,
            dim,
            thickness,
            planeStress,
        )
        self.axis_1 = AsCoords(axis_1)
        self.axis_2 = AsCoords(axis_2)

        self.E1 = E1
        self.E2 = E2
        self.E3 = E3
        self.G23 = G23
        self.G13 = G13
        self.G12 = G12
        self.v23 = v23
        self.v13 = v13
        self.v12 = v12

    axis_1: _types.FloatArray = _params.VectorParameter()
    """Axis 1, (…, 3)."""

    axis_2: _types.FloatArray = _params.VectorParameter()
    """Axis 2, (…, 3)."""

    def _Axes(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        return self.axis_1, self.axis_2

    def __get_params(self) -> list[float | _types.FloatArray]:
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

    def __get_cij_denominator(self) -> float | _types.FloatArray:
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
    def _c11(self) -> float | _types.FloatArray:
        E1, E2, E3, _, _, _, v23, _, _ = self.__get_params()
        return E1**2 * (-E2 + E3 * v23**2) / self.__get_cij_denominator()

    @property
    def _c22(self) -> float | _types.FloatArray:
        E1, E2, E3, _, _, _, _, v13, _ = self.__get_params()
        return E2**2 * (-E1 + E3 * v13**2) / self.__get_cij_denominator()

    @property
    def _c33(self) -> float | _types.FloatArray:
        E1, E2, E3, _, _, _, _, _, v12 = self.__get_params()
        return E2 * E3 * (-E1 + E2 * v12**2) / self.__get_cij_denominator()

    @property
    def _c44(self) -> float | _types.FloatArray:
        return 2 * self.G23

    @property
    def _c55(self) -> float | _types.FloatArray:
        return 2 * self.G13

    @property
    def _c66(self) -> float | _types.FloatArray:
        return 2 * self.G12

    @property
    def _c23(self) -> float | _types.FloatArray:
        E1, E2, E3, _, _, _, v23, v13, v12 = self.__get_params()
        return -E2 * E3 * (E1 * v23 + E2 * v12 * v13) / self.__get_cij_denominator()

    @property
    def _c13(self) -> float | _types.FloatArray:
        E1, E2, E3, _, _, _, v23, v13, v12 = self.__get_params()
        return -E1 * E2 * E3 * (v12 * v23 + v13) / self.__get_cij_denominator()

    @property
    def _c12(self) -> float | _types.FloatArray:
        E1, E2, E3, _, _, _, v23, v13, v12 = self.__get_params()
        return -E1 * E2 * (E2 * v12 + E3 * v13 * v23) / self.__get_cij_denominator()

    def _Material_C(self) -> _types.FloatArray:
        E1, E2, E3, _, _, _, v23, v13, v12 = self.__get_params()

        bounds = {
            "|v23| < sqrt(E2 / E3)": np.abs(v23) < np.sqrt(E2 / E3),
            "|v13| < sqrt(E1 / E3)": np.abs(v13) < np.sqrt(E1 / E3),
            "|v12| < sqrt(E1 / E2)": np.abs(v12) < np.sqrt(E1 / E2),
        }
        for bound, holds in bounds.items():
            if not np.all(holds):
                raise ValueError(f"Orthotropic moduli must satisfy {bound}.")

        # axes (1, 2, 3)
        return Heterogeneous_Array(
            [
                [self._c11, self._c12, self._c13, 0, 0, 0],
                [self._c12, self._c22, self._c23, 0, 0, 0],
                [self._c13, self._c23, self._c33, 0, 0, 0],
                [0, 0, 0, self._c44, 0, 0],
                [0, 0, 0, 0, self._c55, 0],
                [0, 0, 0, 0, 0, self._c66],
            ]
        )

    def Walpole_Decomposition(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        # see section 3.6: https://doi.org/10.1007/s10659-012-9396-z

        a, b, c = self._Frame_fields()

        def tensor_prods(v1, v2, v3, v4):
            return TensorProd(TensorProd(v1, v2), TensorProd(v3, v4))

        def vec_sym_tensor_prod(v1, v2):
            # (ai bj + bi aj)(ak bl + bk al) / 2
            p = TensorProd(v1, v2) + TensorProd(v2, v1)
            return TensorProd(p, p) / 2

        E11 = kelvin_mandel.Tensor_to_Kelvin(tensor_prods(a, a, a, a))
        E22 = kelvin_mandel.Tensor_to_Kelvin(tensor_prods(b, b, b, b))
        E33 = kelvin_mandel.Tensor_to_Kelvin(tensor_prods(c, c, c, c))

        E44 = kelvin_mandel.Tensor_to_Kelvin(vec_sym_tensor_prod(b, c))  # 23
        E55 = kelvin_mandel.Tensor_to_Kelvin(vec_sym_tensor_prod(a, c))  # 13
        E66 = kelvin_mandel.Tensor_to_Kelvin(vec_sym_tensor_prod(a, b))  # 12

        E23 = kelvin_mandel.Tensor_to_Kelvin(
            tensor_prods(b, b, c, c) + tensor_prods(c, c, b, b)
        )
        E13 = kelvin_mandel.Tensor_to_Kelvin(
            tensor_prods(a, a, c, c) + tensor_prods(c, c, a, a)
        )
        E12 = kelvin_mandel.Tensor_to_Kelvin(
            tensor_prods(a, a, b, b) + tensor_prods(b, b, a, a)
        )

        ci = [
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
        Ei = [E11, E22, E33, E44, E55, E66, E23, E13, E12]
        return self._Walpole(ci, Ei)


# ----------------------------------------------
# Anisotropic
# ----------------------------------------------


class Anisotropic(_Elastic):
    """Anisotropic Linearized Elastic material, its C given in the global frame."""

    def __str__(self) -> str:
        text = super().__str__()
        text += f"\nC = {_Format_parameter(self.C)}"
        return text

    def __init__(
        self,
        dim: int,
        C: _types.FloatArray,
        useVoigtNotation: bool,
        thickness=1.0,
    ):
        """Creates an Anisotropic Linearized Elastic material.

        Parameters
        ----------
        dim : int
            dimension
        C : _types.FloatArray
            stiffness in the global frame, (…, 6, 6), or (…, 3, 3) in 2D only; rotate it first with ``Get_Pmat``/``Apply_Pmat``
        useVoigtNotation : bool
            C is in Voigt notation, else Kelvin–Mandel
        thickness: float, optional
            material thickness, by default 1.0
        """
        # plane strain; a (6, 6) C may switch to plane stress afterwards
        _Elastic.__init__(self, dim, thickness, False)
        self.Set_C(C, useVoigtNotation)

    def Set_C(self, C: _types.FloatArray, useVoigtNotation=True):
        """Sets the stiffness C in the global frame, in Voigt or Kelvin–Mandel notation."""
        C = np.asarray(C, dtype=float)
        if C.shape[-2:] not in ((3, 3), (6, 6)) or C.ndim > 4:
            raise ValueError(
                "C must be a (3, 3) or (6, 6), (Ne, …) or (Ne, nPg, …) matrix."
            )
        if C.shape[-1] == 3 and (self.dim == 3 or self.planeStress):
            raise ValueError(
                "A (3, 3) C is 2D only, with no plane stress: give a (6, 6) C."
            )
        if np.abs(C - np.swapaxes(C, -2, -1)).max() > 1e-12 * np.abs(C).max():
            raise ValueError("C must be symmetric.")
        self.__C = kelvin_mandel.From_Voigt(C) if useVoigtNotation else C.copy()
        self.Need_Update()

    def _Material_C(self) -> _types.FloatArray:
        return self.__C

    def Walpole_Decomposition(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        raise NotImplementedError(
            "A general anisotropic C has no Walpole decomposition."
        )
