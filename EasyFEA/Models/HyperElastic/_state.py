# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Hyper elastic module used to compute matrices."""

import numpy as np

from ...FEM import Mesh, MatrixType
from ...FEM._linalg import FeArray, Transpose, Det, Norm
from ...Utilities import _types, _params
from ...Utilities._cache import cache_computed_values

# ------------------------------------------------------------------------------
# Functions for matrices
# ------------------------------------------------------------------------------


class HyperElasticState:
    """Hyperelastic state."""

    @staticmethod
    def _CheckFormat(mesh: Mesh, u: _types.FloatArray, matrixType: MatrixType) -> None:
        assert isinstance(mesh, Mesh), "mesh must be an Mesh object"
        assert (
            isinstance(u, np.ndarray) and u.size % mesh.Nn == 0
        ), "wrong displacement field dimension"
        dim = u.size // mesh.Nn
        assert dim in [1, 2, 3], "wrong displacement field dimension"
        assert (
            matrixType in MatrixType.Get_types()
        ), f"matrixType must be in {MatrixType.Get_types()}"

    def __init__(self, mesh: Mesh, u: _types.FloatArray, matrixType: MatrixType):

        self._CheckFormat(mesh, u, matrixType)

        self.__mesh = mesh
        self.__u = u
        self.__matrixType = matrixType

    @property
    def mesh(self):
        return self.__mesh

    def _GetDims(
        self,
    ) -> tuple[int, int, int]:
        """return Ne, nPg, dim"""
        Ne = self.__mesh.Ne
        dim = self.__u.size // self.__mesh.Nn
        nPg = self.__mesh.Get_jacobian_e_pg(self.__matrixType).shape[1]
        return (Ne, nPg, dim)

    @cache_computed_values
    def Compute_F(self) -> FeArray.FeArrayALike:
        """Computes the deformation gradient F(u) = I + grad(u)

        Returns
        -------
        FeArray
            F(u) of shape (Ne, pg, 3, 3)

        dim = 1
        -------

        1+dxux 0 0\n
        0 1 0\n
        0 0 1

        dim = 2
        -------

        1+dxux dyux 0\n
        dxuy 1+dyuy 0\n
        0 0 1

        dim = 3
        -------

        1+dxux dyux dzux\n
        dxuy 1+dyuy dzuy\n
        dxuz dyuz 1+dzuz
        """

        grad_e_pg = self.__mesh.Get_Gradient_e_pg(self.__u, self.__matrixType)

        F_e_pg = np.eye(3) + grad_e_pg

        return F_e_pg

    @cache_computed_values
    def Compute_J(self) -> FeArray.FeArrayALike:
        """Computes the deformation gradient J = det(F)

        Returns
        -------
        FeArray
            J_e_pg of shape (Ne, pg)
        """

        F_e_pg = self.Compute_F()

        J_e_pg = Det(F_e_pg)

        return J_e_pg

    @cache_computed_values
    def Compute_C(self) -> FeArray.FeArrayALike:
        """Computes the right Cauchy-Green tensor  C(u) = F(u)'.F(u)

        Returns
        -------
        FeArray
            C_e_pg of shape (Ne, pg, 3, 3)

        dim = 1
        -------

        cxx 0 0\n
        0 1 0\n
        0 0 1

        dim = 2
        -------

        cxx cxy 0\n
        cyx cyy 0\n
        0 0 1

        dim = 3
        -------

        cxx cxy cxz\n
        cyx cyy cyz\n
        czx czy czz
        """

        F_e_pg = self.Compute_F()

        C_e_pg = Transpose(F_e_pg) @ F_e_pg

        return C_e_pg

    @cache_computed_values
    def _Compute_C(self) -> list[FeArray.FeArrayALike]:
        """Computes the right Cauchy-Green tensor components C(u) = F(u)'.F(u) \n

        returns [cxx, cxy, cxz, cyx, cyy, cyz, czx, czy, czz]"""

        C_e_pg = self.Compute_C()
        vectC_e_pg = np.reshape(C_e_pg, (*C_e_pg.shape[:2], -1))

        cxx, cxy, cxz, cyx, cyy, cyz, czx, czy, czz = [
            vectC_e_pg[:, :, i] for i in range(9)
        ]

        return [cxx, cxy, cxz, cyx, cyy, cyz, czx, czy, czz]

    @cache_computed_values
    def Compute_GreenLagrange(self) -> FeArray.FeArrayALike:
        """Computes the Green-Lagrange deformation E = 1/2 (C - I)

        Returns
        -------
        FeArray
            E_e_pg of shape (Ne, pg, dim, dim)
        """

        C_e_pg = self.Compute_C()

        E_e_pg = 1 / 2 * (C_e_pg - np.eye(3))

        return E_e_pg

    @cache_computed_values
    def Compute_Epsilon(self) -> FeArray.FeArrayALike:
        """Computes the linearized deformation Epsilon = 1/2 (grad(u)' + grad(u))

        Returns if dim = 2
        ------------------
        FeArray
            Eps_e_pg of shape (Ne, pg, 3)

            [xx, yy, 2**(-1/2) xy]

        Returns if dim = 3
        ------------------
        FeArray
            Eps_e_pg of shape (Ne, pg, 6)

            [xx, yy, zz, 2**(-1/2) yz, 2**(-1/2) xz, 2**(-1/2) xy]
        """

        Ne, nPg, dim = self._GetDims()
        assert dim in [2, 3]

        # compute grad
        grad_e_pg = self.__mesh.Get_Gradient_e_pg(self.__u, self.__matrixType)[
            ..., :dim, :dim
        ]

        # 2d: dxux, dyux, dxuy, dyuy
        # 3d: dxux, dyux, dzu, dxuy, dyuy, dzuy, dxuz, dyuz, dzuz
        gradAsVect_e_pg = np.reshape(grad_e_pg, (Ne, nPg, -1))

        c = 2 ** (-1 / 2)

        if dim == 2:
            mat = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, c, c, 0]])  # xx  # yy  # xy
        else:
            mat = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # xx
                    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # yy
                    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # zz
                    [0, 0, 0, 0, 0, c, 0, c, 0],  # yz
                    [0, 0, c, 0, 0, 0, c, 0, 0],  # xz
                    [0, c, 0, c, 0, 0, 0, 0, 0],  # xy
                ]
            )

        mat = FeArray.asfearray(mat, True)
        Eps_e_pg = mat @ gradAsVect_e_pg

        return Eps_e_pg

    @cache_computed_values
    def Compute_De(self) -> FeArray.FeArrayALike:
        """Computes De(u) derivative green lagrange matrix.

        Returns if dim = 2
        ------------------
        FeArray.FeArrayALike
            D_e_pg of shape (Ne, pg, 3, 4)

            [1+dxux, 0, dxuy, 0] # xx \n
            [0, dyux, 0, 1+dyuy] # yy \n
            2**(-1/2) [dyux, 1+dxux, 1+dyuy, dxuy # xy

        Returns if dim = 3
        ------------------
        FeArray.FeArrayALike
            D_e_pg of shape (Ne, pg, 6, 9)

            [1+dxux, 0, 0, dxuy, 0, 0, dxuz, 0, 0] # xx \n
            [0, dyux, 0, 0, 1+dyuy, 0, 0, dyuz, 0] # yy \n
            [0, 0, dzux, 0, 0, dzuy, 0, 0, 1+dzuz] # zz \n
            2**(-1/2) [0, dzux, dyux, 0, dzuy, 1 + dyuy, 0, 1 + dzuz, dyuz] # yz \n
            2**(-1/2) [dzux, 0, 1 + dxux, dzuy, 0, dxuy, 1 + dzuz, 0, dxuz] # xz \n
            2**(-1/2) [dyux, 1+dxux, 0, 1+dyuy, dxuy, 0, dyuz, dxuz, 0] # xy
        """

        Ne, nPg, dim = self._GetDims()
        assert dim in [2, 3]

        grad_e_pg = self.__mesh.Get_Gradient_e_pg(self.__u, self.__matrixType)

        if dim == 2:
            D_e_pg = FeArray.zeros(Ne, nPg, 3, 4)
        else:
            D_e_pg = FeArray.zeros(Ne, nPg, 6, 9)

        def Add_to_D_e_pg(p: int, line: int, values: list[_types.Any], coef=1.0):
            N = 4 if dim == 2 else 9
            for column in range(N):
                D_e_pg[:, p, line, column] = values[column] * coef

        cM = 2 ** (-1 / 2)

        for p in range(nPg):
            if dim == 2:
                dxux, dyux = [grad_e_pg[:, p, 0, i] for i in range(2)]
                dxuy, dyuy = [grad_e_pg[:, p, 1, i] for i in range(2)]

                Add_to_D_e_pg(p, 0, [1 + dxux, 0, dxuy, 0])  # xx
                Add_to_D_e_pg(p, 1, [0, dyux, 0, 1 + dyuy])  # yy
                Add_to_D_e_pg(p, 2, [dyux, 1 + dxux, 1 + dyuy, dxuy], cM)  # xy

            else:
                dxux, dyux, dzux = [grad_e_pg[:, p, 0, i] for i in range(3)]
                dxuy, dyuy, dzuy = [grad_e_pg[:, p, 1, i] for i in range(3)]
                dxuz, dyuz, dzuz = [grad_e_pg[:, p, 2, i] for i in range(3)]

                Add_to_D_e_pg(p, 0, [1 + dxux, 0, 0, dxuy, 0, 0, dxuz, 0, 0])  # xx
                Add_to_D_e_pg(p, 1, [0, dyux, 0, 0, 1 + dyuy, 0, 0, dyuz, 0])  # yy
                Add_to_D_e_pg(p, 2, [0, 0, dzux, 0, 0, dzuy, 0, 0, 1 + dzuz])  # zz
                Add_to_D_e_pg(
                    p, 3, [0, dzux, dyux, 0, dzuy, 1 + dyuy, 0, 1 + dzuz, dyuz], cM
                )  # yz
                Add_to_D_e_pg(
                    p, 4, [dzux, 0, 1 + dxux, dzuy, 0, dxuy, 1 + dzuz, 0, dxuz], cM
                )  # xz
                Add_to_D_e_pg(
                    p, 5, [dyux, 1 + dxux, 0, 1 + dyuy, dxuy, 0, dyuz, dxuz, 0], cM
                )  # xy

        return D_e_pg

    # --------------------------------------------------------------------------
    # Compute invariants
    # --------------------------------------------------------------------------

    def _Slice_Vector(self, vector: FeArray.FeArrayALike):

        assert isinstance(vector, FeArray)
        assert vector._ndim == 1 and vector._shape == (6,)

        dim = self._GetDims()[2]
        if dim == 1:
            vector = vector[..., [0]]
        elif dim == 2:
            vector = vector[..., [0, 1, 5]]

        return vector

    def _Slice_Matrix(self, matrix: FeArray.FeArrayALike):

        assert isinstance(matrix, FeArray)
        assert matrix._ndim == 2 and matrix._shape == (6, 6)

        dim = self._GetDims()[2]
        if dim == 1:
            matrix = matrix[..., [0], :][..., [0]]
        elif dim == 2:
            matrix = matrix[..., [0, 1, 5], :][..., [0, 1, 5]]

        return matrix

    # -------------------------------------
    # Compute I1
    # -------------------------------------

    @cache_computed_values
    def Compute_I1(self) -> FeArray.FeArrayALike:
        """Computes I1(u)

        Returns
        -------
        FeArray
            I1_e_pg of shape (Ne, pg)
        """

        cxx, _, _, _, cyy, _, _, _, czz = self._Compute_C()

        I1_e_pg = cxx + cyy + czz

        return I1_e_pg

    def Compute_dI1dC(self) -> FeArray.FeArrayALike:
        """Computes dI1dC(u)

        Returns
        -------
        FeArray
            dI1dC of shape (d), where `d = 1, 3, 6` depending on whether the solution dimension is `1D`, `2D`, or `3D`.
        """

        dI1dC = np.array([1, 1, 1, 0, 0, 0])

        return self._Slice_Vector(FeArray.asfearray(dI1dC, True))

    def Compute_d2I1dC(self) -> FeArray.FeArrayALike:
        """Computes d2I1dC(u)

        Returns
        -------
        FeArray
            d2I1dC of shape (d, d), where `d = 1, 3, 6` depending on whether the solution dimension is `1D`, `2D`, or `3D`.
        """

        return self._Slice_Matrix(FeArray.zeros(1, 1, 6, 6))

    # -------------------------------------
    # Compute I2
    # -------------------------------------

    @cache_computed_values
    def Compute_I2(self) -> FeArray.FeArrayALike:
        """Computes I2(u)

        Returns
        -------
        FeArray
            I2_e_pg of shape (Ne, pg)
        """

        cxx, cxy, cxz, _, cyy, cyz, _, _, czz = self._Compute_C()

        I2_e_pg = cxx * cyy + cyy * czz + cxx * czz - cxy**2 - cyz**2 - cxz**2

        return I2_e_pg

    @cache_computed_values
    def Compute_dI2dC(self) -> FeArray.FeArrayALike:
        """Computes dI2dC(u)

        Returns
        -------
        FeArray
            dI2dC_e_pg of shape (Ne, pg, d), where `d = 1, 3, 6` depending on whether the solution dimension is `1D`, `2D`, or `3D`.
        """

        Ne, nPg, _ = self._GetDims()

        cxx, cxy, cxz, _, cyy, cyz, _, _, czz = self._Compute_C()

        dI2dC_e_pg = FeArray.zeros(Ne, nPg, 6, dtype=float)

        coef = -np.sqrt(2)

        dI2dC_e_pg[:, :, 0] = cyy + czz
        dI2dC_e_pg[:, :, 1] = cxx + czz
        dI2dC_e_pg[:, :, 2] = cxx + cyy
        dI2dC_e_pg[:, :, 3] = coef * cyz
        dI2dC_e_pg[:, :, 4] = coef * cxz
        dI2dC_e_pg[:, :, 5] = coef * cxy

        return self._Slice_Vector(dI2dC_e_pg)

    def Compute_d2I2dC(self) -> FeArray.FeArrayALike:
        """Computes d2I2dC(u)

        Returns
        -------
        FeArray
            d2I2dC of shape (d, d), where `d = 1, 3, 6` depending on whether the solution dimension is `1D`, `2D`, or `3D`.
        """

        d2I2dC = np.array(
            [
                [0, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, -1, 0],
                [0, 0, 0, 0, 0, -1],
            ]
        )

        return self._Slice_Matrix(FeArray.asfearray(d2I2dC, True))

    # -------------------------------------
    # Compute I3
    # -------------------------------------

    @cache_computed_values
    def Compute_I3(self) -> FeArray.FeArrayALike:
        """Computes I3(u)

        Returns
        -------
        FeArray
            I3_e_pg of shape (Ne, pg)
        """

        cxx, cxy, cxz, _, cyy, cyz, _, _, czz = self._Compute_C()

        I3_e_pg = (
            cxx * cyy * czz
            - cxx * cyz**2
            - cxy**2 * czz
            + 2 * cxy * cxz * cyz
            - cxz**2 * cyy
        )

        return I3_e_pg

    @cache_computed_values
    def Compute_dI3dC(self) -> FeArray.FeArrayALike:
        """Computes dI3dC(u)

        Returns
        -------
        FeArray
            dI3dC_e_pg of shape (Ne, pg, d), where `d = 1, 3, 6` depending on whether the solution dimension is `1D`, `2D`, or `3D`.
        """

        cxx, cxy, cxz, _, cyy, cyz, _, _, czz = self._Compute_C()

        Ne, nPg, _ = self._GetDims()

        dI3dC_e_pg = FeArray.zeros(Ne, nPg, 6)

        coef = np.sqrt(2)

        dI3dC_e_pg[:, :, 0] = cyy * czz - cyz**2
        dI3dC_e_pg[:, :, 1] = cxx * czz - cxz**2
        dI3dC_e_pg[:, :, 2] = cxx * cyy - cxy**2
        dI3dC_e_pg[:, :, 3] = coef * (-cxx * cyz + cxy * cxz)
        dI3dC_e_pg[:, :, 4] = coef * (cxy * cyz - cxz * cyy)
        dI3dC_e_pg[:, :, 5] = coef * (-cxy * czz + cxz * cyz)

        return self._Slice_Vector(dI3dC_e_pg)

    @cache_computed_values
    def Compute_d2I3dC(self) -> FeArray.FeArrayALike:
        """Computes d2I3dC(u)

        Returns
        -------
        FeArray
            d2I3dC_e_pg of shape (Ne, pg, d, d), where `d = 1, 3, 6` depending on whether the solution dimension is `1D`, `2D`, or `3D`.
        """

        cxx, cxy, cxz, _, cyy, cyz, _, _, czz = self._Compute_C()

        Ne, nPg, _ = self._GetDims()

        d2I3dC_e_pg = FeArray.zeros(Ne, nPg, 6, 6)

        d2I3dC_e_pg[:, :, 0, 1] = d2I3dC_e_pg[:, :, 1, 0] = czz
        d2I3dC_e_pg[:, :, 0, 2] = d2I3dC_e_pg[:, :, 2, 0] = cyy
        d2I3dC_e_pg[:, :, 1, 2] = d2I3dC_e_pg[:, :, 2, 1] = cxx

        c = -np.sqrt(2)
        d2I3dC_e_pg[:, :, 0, 3] = d2I3dC_e_pg[:, :, 3, 0] = c * cyz
        d2I3dC_e_pg[:, :, 1, 4] = d2I3dC_e_pg[:, :, 4, 1] = c * cxz
        d2I3dC_e_pg[:, :, 2, 5] = d2I3dC_e_pg[:, :, 5, 2] = c * cxy

        d2I3dC_e_pg[:, :, 3, 3] = -cxx
        d2I3dC_e_pg[:, :, 4, 4] = -cyy
        d2I3dC_e_pg[:, :, 5, 5] = -czz

        d2I3dC_e_pg[:, :, 3, 4] = d2I3dC_e_pg[:, :, 4, 3] = cxy
        d2I3dC_e_pg[:, :, 3, 5] = d2I3dC_e_pg[:, :, 5, 3] = cxz
        d2I3dC_e_pg[:, :, 4, 5] = d2I3dC_e_pg[:, :, 5, 4] = cyz

        return self._Slice_Matrix(d2I3dC_e_pg)

    # -------------------------------------
    # Compute Anisotropic Invariants
    # -------------------------------------

    def _Get_normalized_components(
        self, T: _types.FloatArray
    ) -> tuple[FeArray.FeArrayALike, FeArray.FeArrayALike, FeArray.FeArrayALike]:

        _params._CheckIsVector(T)
        if not isinstance(T, FeArray):
            T = FeArray.asfearray(T, True)
        T = T.astype(float)

        norm = Norm(T, axis=-1)
        T[norm != 0] /= norm

        Tx, Ty, Tz = [T[..., i] for i in range(3)]

        dim = self._GetDims()[2]
        if dim == 1:
            Ty = Tz = 0
        elif dim == 2:
            Tz = 0

        return Tx, Ty, Tz

    def _Compute_Anisotropic_Invariants(
        self, T1: _types.FloatArray, T2: _types.FloatArray
    ):

        cxx, cxy, cxz, _, cyy, cyz, _, _, czz = self._Compute_C()

        T1x, T1y, T1z = self._Get_normalized_components(T1)
        T2x, T2y, T2z = self._Get_normalized_components(T2)

        value = (
            T1x * T2x * cxx
            + T1x * T2y * cxy
            + T1x * T2z * cxz
            + T1y * T2x * cxy
            + T1y * T2y * cyy
            + T1y * T2z * cyz
            + T1z * T2x * cxz
            + T1z * T2y * cyz
            + T1z * T2z * czz
        )

        return value

    def _Compute_Anisotropic_Invariants_First_Derivatives(
        self, T1: _types.FloatArray, T2: _types.FloatArray
    ):

        T1x, T1y, T1z = self._Get_normalized_components(T1)
        T2x, T2y, T2z = self._Get_normalized_components(T2)

        Ne, nPg = T1x.shape
        firstDerivatives = FeArray.zeros(Ne, nPg, 6)

        coef = np.sqrt(2) / 2

        firstDerivatives[:, :, 0] = T1x * T2x
        firstDerivatives[:, :, 1] = T1y * T2y
        firstDerivatives[:, :, 2] = T1z * T2z
        firstDerivatives[:, :, 3] = coef * (T1y * T2z + T1z * T2y)
        firstDerivatives[:, :, 4] = coef * (T1x * T2z + T1z * T2x)
        firstDerivatives[:, :, 5] = coef * (T1x * T2y + T1y * T2x)

        return self._Slice_Vector(firstDerivatives)

    # -------------------------------------
    # Compute I4
    # -------------------------------------
    # Compute_I4, Compute_I6, and Compute_I8 are not cacheable,
    # because the given numpy arrays are not hashable.
    def Compute_I4(
        self,
        T: _types.FloatArray,
    ) -> FeArray.FeArrayALike:
        """Computes I4(u)

        Parameters
        ----------
        T : _types.FloatArray
            direction(s)

        Returns
        -------
        FeArray
            I4_e_pg of shape (Ne, pg)
        """

        return self._Compute_Anisotropic_Invariants(T, T)

    def Compute_dI4dC(self, T: _types.FloatArray) -> FeArray.FeArrayALike:
        """Computes dI4dC(u)

        Parameters
        ----------
        T : _types.FloatArray
            direction(s)

        Returns
        -------
        FeArray
            dI4dC_e_pg of shape (Ne, pg, d), where `d = 1, 3, 6` depending on whether the solution dimension is `1D`, `2D`, or `3D`.
        """

        return self._Compute_Anisotropic_Invariants_First_Derivatives(T, T)

    def Compute_d2I4dC(self) -> FeArray.FeArrayALike:
        """Computes d2I4dC(u)

        Returns
        -------
        FeArray
            d2I4dC of shape (d, d), where `d = 1, 3, 6` depending on whether the solution dimension is `1D`, `2D`, or `3D`.
        """

        return self._Slice_Matrix(FeArray.zeros(1, 1, 6, 6))

    # -------------------------------------
    # Compute I6
    # -------------------------------------
    def Compute_I6(
        self,
        T: _types.FloatArray,
    ) -> FeArray.FeArrayALike:
        """Computes I6(u)

        Parameters
        ----------
        T : _types.FloatArray
            direction(s)

        Returns
        -------
        FeArray
            I6_e_pg of shape (Ne, pg)
        """

        return self._Compute_Anisotropic_Invariants(T, T)

    def Compute_dI6dC(self, T: _types.FloatArray) -> FeArray.FeArrayALike:
        """Computes dI6dC(u)

        Parameters
        ----------
        T : _types.FloatArray
            direction(s)

        Returns
        -------
        FeArray
            dI6dC_e_pg of shape (Ne, pg, d), where `d = 1, 3, 6` depending on whether the solution dimension is `1D`, `2D`, or `3D`.
        """

        return self.Compute_dI4dC(T)

    def Compute_d2I6dC(self) -> FeArray.FeArrayALike:
        """Computes d2I6dC(u)

        Returns
        -------
        FeArray
            d2I6dC of shape (d, d), where `d = 1, 3, 6` depending on whether the solution dimension is `1D`, `2D`, or `3D`.
        """

        return self.Compute_d2I4dC()

    # -------------------------------------
    # Compute I8
    # -------------------------------------
    def Compute_I8(
        self,
        T1: _types.FloatArray,
        T2: _types.FloatArray,
    ) -> FeArray.FeArrayALike:
        """Computes I8(u)

        Parameters
        ----------
        T1 : _types.FloatArray
            direction(s) 1
        T2 : _types.FloatArray
            direction(s) 2

        Returns
        -------
        FeArray
            I8_e_pg of shape (Ne, pg)
        """

        return self._Compute_Anisotropic_Invariants(T1, T2)

    def Compute_dI8dC(
        self, T1: _types.FloatArray, T2: _types.FloatArray
    ) -> FeArray.FeArrayALike:
        """Computes dI8dC(u)

        Parameters
        ----------
        T1 : _types.FloatArray
            direction(s) 1
        T2 : _types.FloatArray
            direction(s) 2

        Returns
        -------
        FeArray
            dI8dC_e_pg of shape (Ne, pg, d), where `d = 1, 3, 6` depending on whether the solution dimension is `1D`, `2D`, or `3D`.
        """

        return self._Compute_Anisotropic_Invariants_First_Derivatives(T1, T2)

    def Compute_d2I8dC(self) -> FeArray.FeArrayALike:
        """Computes d2I8dC(u)

        Returns
        -------
        FeArray
            d2I8dC of shape (d, d), where `d = 1, 3, 6` depending on whether the solution dimension is `1D`, `2D`, or `3D`.
        """

        return self.Compute_d2I4dC()
