# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Hyper elastic module used to compute matrices."""

from typing import Union

import numpy as np

from ...FEM import MatrixType, _GroupElem
from ...FEM._linalg import FeArray, Transpose, Det, Norm
from ...Utilities import _types, _params
from ...Utilities._cache import cache_computed_values

# ------------------------------------------------------------------------------
# Functions for matrices
# ------------------------------------------------------------------------------


class HyperElasticState:
    """Hyperelastic state."""

    @staticmethod
    def _CheckFormat(groupElem: _GroupElem, u: _types.FloatArray) -> None:
        assert isinstance(groupElem, _GroupElem)
        Ncoords = groupElem.Ncoords
        errorDim = "wrong displacement field dimension"
        assert isinstance(u, np.ndarray) and u.size % Ncoords == 0, errorDim
        dim = u.size // Ncoords
        assert dim in [1, 2, 3], errorDim

    def __init__(
        self,
        groupElem: _GroupElem,
        displacement: _types.FloatArray,
        matrixType: MatrixType,
    ):
        """
        Hyperelastic state — the displacement configuration at which the material response is evaluated.

        Parameters
        ----------
        groupElem : _GroupElem
            group of elements
        displacement : _types.FloatArray
            displacement field in (xi,yi,zi,...,xn,yn,zn) format
        matrixType : MatrixType
            matrix type
        """

        self._CheckFormat(groupElem, displacement)

        self.__groupElem = groupElem
        self.__displacement = displacement
        self.__matrixType = matrixType

    @property
    def groupElem(self):
        """group of elements."""
        return self.__groupElem

    @property
    def displacement(self):
        """displacement field in (xi,yi,zi,...,xn,yn,zn) format."""
        return self.__displacement

    @property
    def matrixType(self):
        """matrix type."""
        return self.__matrixType

    @matrixType.setter
    def matrixType(self, value: Union[int, MatrixType]):
        self.__matrixType = value

    def _GetDims(
        self,
    ) -> tuple[int, int, int]:
        """return Ne, nPg, dim"""
        Ne = self.__groupElem.Ne
        dim = self.__displacement.size // self.__groupElem.Ncoords
        nPg = self.__groupElem.Get_N_pg(self.__matrixType).shape[0]
        return (Ne, nPg, dim)

    @cache_computed_values
    def Compute_F(self) -> FeArray.FeArrayALike:
        """Deformation gradient ``F(u) = I + ∇u`` at the Gauss points, padded to ``3×3``.

        Returns
        -------
        FeArray
            Layout depends on ``dim`` (unused off-diagonal terms are zero; missing diagonal terms stay at 1, so the result is always ``3×3``):

            ``dim == 1``:
            ```
            [ 1+dxux    0       0    ]
            [   0       1       0    ]
            [   0       0       1    ]
            ```

            ``dim == 2``:
            ```
            [ 1+dxux   dyux     0    ]
            [  dxuy   1+dyuy    0    ]
            [   0       0       1    ]
            ```

            ``dim == 3``:
            ```
            [ 1+dxux   dyux    dzux  ]
            [  dxuy   1+dyuy   dzuy  ]
            [  dxuz    dyuz   1+dzuz ]
            ```

            Shape: ``(Ne, nPg, 3, 3)``.
        """

        grad_e_pg = self.__groupElem.Get_Gradient_e_pg(
            self.__displacement, self.__matrixType
        )

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
        """Right Cauchy-Green tensor ``C(u) = F(u)ᵀ · F(u)`` at the Gauss points, padded to ``3×3``.

        Returns
        -------
        FeArray
            Layout depends on ``dim`` (unused off-diagonal terms are zero; missing diagonal terms stay at 1, so the result is always ``3×3``):

            ``dim == 1``:
            ```
            [ cxx   0    0  ]
            [  0    1    0  ]
            [  0    0    1  ]
            ```

            ``dim == 2``:
            ```
            [ cxx  cxy   0  ]
            [ cyx  cyy   0  ]
            [  0    0    1  ]
            ```

            ``dim == 3``:
            ```
            [ cxx  cxy  cxz ]
            [ cyx  cyy  cyz ]
            [ czx  czy  czz ]
            ```

            Shape: ``(Ne, nPg, 3, 3)``.
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
        """Green-Lagrange strain ``E = 1/2 (C - I)`` at the Gauss points, padded to ``3×3``.

        Returns
        -------
        FeArray
            Layout depends on ``dim`` (unused components are zeroed so the result is always ``3×3``):

            ``dim == 1``:
            ```
            [ Exx   0    0  ]
            [  0    0    0  ]
            [  0    0    0  ]
            ```

            ``dim == 2``:
            ```
            [ Exx  Exy   0  ]
            [ Eyx  Eyy   0  ]
            [  0    0    0  ]
            ```

            ``dim == 3``:
            ```
            [ Exx  Exy  Exz ]
            [ Eyx  Eyy  Eyz ]
            [ Ezx  Ezy  Ezz ]
            ```

            Shape: ``(Ne, nPg, 3, 3)``.
        """

        C_e_pg = self.Compute_C()

        E_e_pg = 1 / 2 * (C_e_pg - np.eye(3))

        return E_e_pg

    @cache_computed_values
    def Compute_Epsilon(self) -> FeArray.FeArrayALike:
        """Linearized strain ``ε = 1/2 (∇uᵀ + ∇u)`` in Kelvin-Mandel vector form at the Gauss points.

        Returns
        -------
        FeArray
            Layout depends on ``dim`` (off-diagonal components carry the Kelvin-Mandel ``√2`` factor):

            ``dim == 2``:
            ```
            [ εxx,  εyy,  √2·εxy ]
            ```
            Shape: ``(Ne, nPg, 3)``.

            ``dim == 3``:
            ```
            [ εxx,  εyy,  εzz,  √2·εyz,  √2·εxz,  √2·εxy ]
            ```
            Shape: ``(Ne, nPg, 6)``.
        """

        Ne, nPg, dim = self._GetDims()
        assert dim in [2, 3]

        # compute grad
        grad_e_pg = self.__groupElem.Get_Gradient_e_pg(
            self.__displacement, self.__matrixType
        )[..., :dim, :dim]

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

    def Compute_Edot_vec(self, velocity: _types.FloatArray) -> FeArray.FeArrayALike:
        """Green–Lagrange strain rate ``Ė`` in Kelvin–Mandel vector form.

        Uses the identity ``Ė = sym(Fᵀ · ∇v) = De(u) · flat(∇v)`` — the same kinematic operator :meth:`Compute_De` that maps ``flat(∇u̇)`` to the small-strain rate also maps ``flat(∇v)`` to ``Ė_vec`` evaluated at the current ``u``.

        Parameters
        ----------
        velocity : _types.FloatArray
            velocity field, same ``(xi,yi,zi,...)`` layout as the displacement.

        Returns ``(Ne, nPg, nstrain)`` — ``nstrain = 3`` in 2D, ``6`` in 3D.
        """
        Ne, nPg, dim = self._GetDims()
        De_e_pg = self.Compute_De()
        grad_v_e_pg = self.__groupElem.Get_Gradient_e_pg(velocity, self.__matrixType)[
            ..., :dim, :dim
        ]
        grad_v_flat = np.reshape(grad_v_e_pg, (Ne, nPg, -1))
        return De_e_pg @ grad_v_flat

    def __Build_De(self, G_e_pg: FeArray.FeArrayALike) -> FeArray.FeArrayALike:
        r"""Builds the operator mapping ``flat(δ∇)`` → ``sym(Gᵀ·δ∇)`` in Kelvin-Mandel vector form, from a ``(Ne, nPg, 3, 3)`` matrix field ``G``.

        Backs both :meth:`Compute_De` (``G = F = I + ∇u``, so ``De·flat(∇v) = sym(Fᵀ∇v) = Ė``) and :meth:`Compute_Deta` (``G = ∇v``, so ``Deta·flat(∇δu) = sym(∇vᵀ∇δu) = ∂Ė``): the two operators differ only in which matrix plays the role of ``G``.

        With ``c = 1/√2`` (Kelvin-Mandel scaling on shear rows) and ``Gij = G[i, j]`` the row ``r`` holds the components of ``G`` so that row ``r`` · ``flat(δ∇)`` is the ``r``-th Kelvin-Mandel component of ``sym(Gᵀ·δ∇)``. Shape: ``(Ne, nPg, 3, 4)`` in 2D, ``(Ne, nPg, 6, 9)`` in 3D.
        """
        Ne, nPg, dim = self._GetDims()
        assert dim in [2, 3]

        D_e_pg = (
            FeArray.zeros(Ne, nPg, 3, 4) if dim == 2 else FeArray.zeros(Ne, nPg, 6, 9)
        )

        def Add(p: int, line: int, values: list[_types.Any], coef=1.0):
            for column, value in enumerate(values):
                D_e_pg[:, p, line, column] = value * coef

        cM = 2 ** (-1 / 2)

        for p in range(nPg):
            if dim == 2:
                g00, g01 = G_e_pg[:, p, 0, 0], G_e_pg[:, p, 0, 1]
                g10, g11 = G_e_pg[:, p, 1, 0], G_e_pg[:, p, 1, 1]

                Add(p, 0, [g00, 0, g10, 0])  # xx
                Add(p, 1, [0, g01, 0, g11])  # yy
                Add(p, 2, [g01, g00, g11, g10], cM)  # xy

            else:
                g00, g01, g02 = (G_e_pg[:, p, 0, i] for i in range(3))
                g10, g11, g12 = (G_e_pg[:, p, 1, i] for i in range(3))
                g20, g21, g22 = (G_e_pg[:, p, 2, i] for i in range(3))

                Add(p, 0, [g00, 0, 0, g10, 0, 0, g20, 0, 0])  # xx
                Add(p, 1, [0, g01, 0, 0, g11, 0, 0, g21, 0])  # yy
                Add(p, 2, [0, 0, g02, 0, 0, g12, 0, 0, g22])  # zz
                Add(p, 3, [0, g02, g01, 0, g12, g11, 0, g22, g21], cM)  # yz
                Add(p, 4, [g02, 0, g00, g12, 0, g10, g22, 0, g20], cM)  # xz
                Add(p, 5, [g01, g00, 0, g11, g10, 0, g21, g20, 0], cM)  # xy

        return D_e_pg

    @cache_computed_values
    def Compute_De(self) -> FeArray.FeArrayALike:
        """Kinematic operator such that ``Ė_vec = De(u) · flat(∇v)`` where ``Ė_vec`` is the Green-Lagrange strain rate in **Kelvin-Mandel** vector form.

        It is ``__Build_De`` with ``G = F = I + ∇u`` the deformation gradient (so ``De · flat(∇v) = sym(Fᵀ∇v) = Ė``).

        With ``c = 1/√2`` (Kelvin-Mandel scaling on shear rows):

        2D — 3 strain components ``(Ėxx, Ėyy, √2·Ėxy)``, columns index ``flat(∇v) = [dxux, dyux, dxuy, dyuy]``:
        ```
        [ 1+dxux       0         dxuy         0      ]
        [    0       dyux         0        1+dyuy    ]
        [ c·dyux  c·(1+dxux)  c·(1+dyuy)   c·dxuy   ]
        ```

        3D — 6 strain components ``(Ėxx, Ėyy, Ėzz, √2·Ėyz, √2·Ėxz, √2·Ėxy)``, columns index ``flat(∇v) = [dxux, dyux, dzux, dxuy, dyuy, dzuy, dxuz, dyuz, dzuz]``:
        ```
        [ 1+dxux      0          0         dxuy        0          0          dxuz        0          0      ]
        [    0      dyux         0           0       1+dyuy       0            0        dyuz        0      ]
        [    0        0        dzux          0          0        dzuy          0          0       1+dzuz   ]
        [    0    c·dzux     c·dyux          0       c·dzuy   c·(1+dyuy)       0     c·(1+dzuz)  c·dyuz   ]
        [ c·dzux     0      c·(1+dxux)   c·dzuy        0       c·dxuy     c·(1+dzuz)     0       c·dxuz   ]
        [ c·dyux c·(1+dxux)    0       c·(1+dyuy)   c·dxuy       0          c·dyuz     c·dxuz        0    ]
        ```

        Shape: ``(Ne, nPg, 3, 4)`` in 2D, ``(Ne, nPg, 6, 9)`` in 3D.
        """
        return self.__Build_De(self.Compute_F())

    def Compute_Deta(self, velocity: _types.FloatArray) -> FeArray.FeArrayALike:
        r"""Configuration derivative of the Green-Lagrange strain rate: the operator ``Deta`` such that ``∂Ė_vec = Deta · flat(∇δu)`` at fixed velocity.

        ``Ė_vec = De(u) · flat(∇v) = sym(Fᵀ∇v)`` is bilinear in ``(∇u, ∇v)``, so ``∂Ė/∂(∇u) = sym(∇vᵀ∇δu)`` is ``__Build_De`` with ``G = ∇v`` the velocity gradient (the same builder as :meth:`Compute_De`, with ``∇v`` in place of ``F``). It feeds the material-like piece ``η · Bᵀ · (Deta · grad)`` of the viscous configuration tangent (``Kgeo``) returned by :func:`Operators.NonLinear.KelvinVoigtDamping`.

        Parameters
        ----------
        velocity : _types.FloatArray
            velocity field, same ``(xi,yi,zi,...)`` layout as the displacement.

        Returns ``(Ne, nPg, 3, 4)`` in 2D, ``(Ne, nPg, 6, 9)`` in 3D — same layout as :meth:`Compute_De`.
        """
        grad_v_e_pg = self.__groupElem.Get_Gradient_e_pg(velocity, self.__matrixType)
        return self.__Build_De(grad_v_e_pg)

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
        nonzero = norm != 0
        T[nonzero] /= norm[nonzero][:, None]

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
