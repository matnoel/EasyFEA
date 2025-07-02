# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Hyper elastic module used to compute matrices."""

import numpy as np

from ..fem import Mesh, MatrixType
from ..fem._linalg import FeArray, Transpose, Det, TensorProd
from ..utilities import _types
from ._utils import Project_Kelvin

# ------------------------------------------------------------------------------
# Functions for matrices
# ------------------------------------------------------------------------------


class HyperElastic:
    @staticmethod
    def __CheckFormat(mesh: Mesh, u: _types.FloatArray, matrixType: MatrixType) -> None:
        assert isinstance(mesh, Mesh), "mesh must be an Mesh object"
        assert (
            isinstance(u, np.ndarray) and u.size % mesh.Nn == 0
        ), "wrong displacement field dimension"
        dim = u.size // mesh.Nn
        assert dim in [2, 3], "wrong displacement field dimension"
        assert (
            matrixType in MatrixType.Get_types()
        ), f"matrixType must be in {MatrixType.Get_types()}"

    @staticmethod
    def __GetDims(
        mesh: Mesh, u: _types.FloatArray, matrixType: MatrixType
    ) -> tuple[int, int, int]:
        """return Ne, nPg, dim"""
        HyperElastic.__CheckFormat(mesh, u, matrixType)
        Ne = mesh.Ne
        dim = u.size // mesh.Nn
        nPg = mesh.Get_jacobian_e_pg(matrixType).shape[1]
        return (Ne, nPg, dim)

    @staticmethod
    def Compute_F(
        mesh: Mesh, u: _types.FloatArray, matrixType=MatrixType.rigi
    ) -> FeArray.FeArrayALike:
        """Computes the deformation gradient F(u) = I + grad(u)

        Parameters
        ----------
        mesh : Mesh
            mesh
        u : _types.FloatArray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

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

        HyperElastic.__CheckFormat(mesh, u, matrixType)

        grad_e_pg = mesh.Get_Gradient_e_pg(u, matrixType)

        F_e_pg = np.eye(3) + grad_e_pg

        return F_e_pg

    @staticmethod
    def Compute_J(
        mesh: Mesh, u: _types.FloatArray, matrixType=MatrixType.rigi
    ) -> FeArray.FeArrayALike:
        """Computes the deformation gradient J = det(F)

        Parameters
        ----------
        mesh : Mesh
            mesh
        u : _types.FloatArray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        FeArray
            J_e_pg of shape (Ne, pg)
        """

        F_e_pg = HyperElastic.Compute_F(mesh, u, matrixType)

        J_e_pg = Det(F_e_pg)

        return J_e_pg

    @staticmethod
    def Compute_C(
        mesh: Mesh, u: _types.FloatArray, matrixType=MatrixType.rigi
    ) -> FeArray.FeArrayALike:
        """Computes the right Cauchy-Green tensor  C(u) = F(u)'.F(u)

        Parameters
        ----------
        mesh : Mesh
            mesh
        u : _types.FloatArray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        FeArray
            C_e_pg of shape (Ne, pg, 3, 3)

        dim = 1
        -------

        cxx 0 0\n
        0 0 0\n
        0 0 0

        dim = 2
        -------

        cxx cxy 0\n
        cyx cyy 0\n
        0 0 0

        dim = 3
        -------

        cxx cxy cxz\n
        cyx cyy cyz\n
        czx czy czz
        """

        F_e_pg = HyperElastic.Compute_F(mesh, u, matrixType)

        C_e_pg = Transpose(F_e_pg) @ F_e_pg

        return C_e_pg

    @staticmethod
    def _Compute_C(
        mesh: Mesh, u: _types.FloatArray, matrixType=MatrixType.rigi
    ) -> list[FeArray.FeArrayALike]:
        """Computes the right Cauchy-Green tensor components C(u) = F(u)'.F(u) \n

        returns [cxx, cxy, cxz, cyx, cyy, cyz, czx, czy, czz]"""

        C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)
        vectC_e_pg = np.reshape(C_e_pg, (*C_e_pg.shape[:2], -1))

        cxx, cxy, cxz, cyx, cyy, cyz, czx, czy, czz = [
            vectC_e_pg[:, :, i] for i in range(9)
        ]

        return [cxx, cxy, cxz, cyx, cyy, cyz, czx, czy, czz]

    @staticmethod
    def Compute_GreenLagrange(
        mesh: Mesh, u: _types.FloatArray, matrixType=MatrixType.rigi
    ) -> FeArray.FeArrayALike:
        """Computes the Green-Lagrange deformation E = 1/2 (C - I)

        Parameters
        ----------
        mesh : Mesh
            mesh
        u : _types.FloatArray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        FeArray
            E_e_pg of shape (Ne, pg, dim, dim)
        """

        C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)

        E_e_pg = 1 / 2 * (C_e_pg - np.eye(3))

        return E_e_pg

    @staticmethod
    def Compute_Epsilon(
        mesh: Mesh, u: _types.FloatArray, matrixType=MatrixType.rigi
    ) -> FeArray.FeArrayALike:
        """Computes the linearized deformation Epsilon = 1/2 (grad(u)' + grad(u))

        Parameters
        ----------
        mesh : Mesh
            mesh
        u : _types.FloatArray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

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

        HyperElastic.__CheckFormat(mesh, u, matrixType)
        Ne, nPg, dim = HyperElastic.__GetDims(mesh, u, matrixType)
        assert dim in [2, 3]

        # compute grad
        grad_e_pg = mesh.Get_Gradient_e_pg(u, matrixType)[..., :dim, :dim]

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

    @staticmethod
    def Compute_De(
        mesh: Mesh, u: _types.FloatArray, matrixType=MatrixType.rigi
    ) -> FeArray.FeArrayALike:
        """Computes De(u)

        Parameters
        ----------
        mesh : Mesh
            mesh
        u : _types.FloatArray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

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

        HyperElastic.__CheckFormat(mesh, u, matrixType)
        Ne, nPg, dim = HyperElastic.__GetDims(mesh, u, matrixType)
        assert dim in [2, 3]

        grad_e_pg = mesh.Get_Gradient_e_pg(u, matrixType)

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

    # -------------------------------------
    # Compute I1
    # -------------------------------------
    @staticmethod
    def Compute_I1(
        mesh: Mesh, u: _types.FloatArray, matrixType=MatrixType.rigi
    ) -> FeArray.FeArrayALike:
        """Computes I1(u)

        Parameters
        ----------
        mesh : Mesh
            mesh
        u : _types.FloatArray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        FeArray
            I1_e_pg of shape (Ne, pg)
        """

        cxx, _, _, _, cyy, _, _, _, czz = HyperElastic._Compute_C(mesh, u, matrixType)

        I1_e_pg = cxx + cyy + czz

        return I1_e_pg

    @staticmethod
    def Compute_dI1dC() -> FeArray.FeArrayALike:
        """Computes dI1dC(u)

        Returns
        -------
        FeArray
            dI1dC of shape (6)
        """

        dI1dC = np.array([1, 1, 1, 0, 0, 0])

        return FeArray.asfearray(dI1dC, True)

    @staticmethod
    def Compute_d2I1dC() -> FeArray.FeArrayALike:
        """Computes d2I1dC(u)

        Returns
        -------
        FeArray
            d2I1dC of shape (6, 6)
        """

        return FeArray.zeros(1, 1, 6, 6)

    # -------------------------------------
    # Compute I2
    # -------------------------------------
    @staticmethod
    def Compute_I2(
        mesh: Mesh, u: _types.FloatArray, matrixType=MatrixType.rigi
    ) -> FeArray.FeArrayALike:
        """Computes I2(u)

        Parameters
        ----------
        mesh : Mesh
            mesh
        u : _types.FloatArray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        FeArray
            I2_e_pg of shape (Ne, pg)
        """

        cxx, cxy, cxz, _, cyy, cyz, _, _, czz = HyperElastic._Compute_C(
            mesh, u, matrixType
        )

        I2_e_pg = cxx * cyy + cyy * czz + cxx * czz - cxy**2 - cyz**2 - cxz**2

        return I2_e_pg

    @staticmethod
    def Compute_dI2dC(
        mesh: Mesh, u: _types.FloatArray, matrixType=MatrixType.rigi
    ) -> FeArray.FeArrayALike:
        """Computes dI2dC(u)

        Parameters
        ----------
        mesh : Mesh
            mesh
        u : _types.FloatArray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        FeArray
            dI2dC_e_pg of shape (Ne, pg, 3, 3)
        """

        Ne, nPg, _ = HyperElastic.__GetDims(mesh, u, matrixType)

        cxx, cxy, cxz, _, cyy, cyz, _, _, czz = HyperElastic._Compute_C(
            mesh, u, matrixType
        )

        dI2dC_e_pg = FeArray.asfearray(np.zeros((Ne, nPg, 6), dtype=float))

        coef = -np.sqrt(2)

        dI2dC_e_pg[:, :, 0] = cyy + czz
        dI2dC_e_pg[:, :, 1] = cxx + czz
        dI2dC_e_pg[:, :, 2] = cxx + cyy
        dI2dC_e_pg[:, :, 3] = coef * cyz
        dI2dC_e_pg[:, :, 4] = coef * cxz
        dI2dC_e_pg[:, :, 5] = coef * cxy

        return dI2dC_e_pg

    @staticmethod
    def Compute_d2I2dC() -> FeArray.FeArrayALike:
        """Computes d2I2dC(u)

        Returns
        -------
        FeArray
            d2I2dC of shape (6, 6)
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

        return FeArray.asfearray(d2I2dC, True)

    # -------------------------------------
    # Compute I3
    # -------------------------------------
    @staticmethod
    def Compute_I3(
        mesh: Mesh, u: _types.FloatArray, matrixType=MatrixType.rigi
    ) -> FeArray.FeArrayALike:
        """Computes I3(u)

        Parameters
        ----------
        mesh : Mesh
            mesh
        u : _types.FloatArray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        FeArray
            I3_e_pg of shape (Ne, pg)
        """

        cxx, cxy, cxz, _, cyy, cyz, _, _, czz = HyperElastic._Compute_C(
            mesh, u, matrixType
        )

        I3_e_pg = (
            cxx * cyy * czz
            - cxx * cyz**2
            - cxy**2 * czz
            + 2 * cxy * cxz * cyz
            - cxz**2 * cyy
        )

        return I3_e_pg

    @staticmethod
    def Compute_dI3dC(
        mesh: Mesh, u: _types.FloatArray, matrixType=MatrixType.rigi
    ) -> FeArray.FeArrayALike:
        """Computes dI3dC(u)

        Parameters
        ----------
        mesh : Mesh
            mesh
        u : _types.FloatArray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        FeArray
            dI3dC_e_pg of shape (Ne, pg, 3, 3)
        """

        cxx, cxy, cxz, _, cyy, cyz, _, _, czz = HyperElastic._Compute_C(
            mesh, u, matrixType
        )

        Ne, nPg, _ = HyperElastic.__GetDims(mesh, u, matrixType)

        dI3dC_e_pg = FeArray.asfearray(np.zeros((Ne, nPg, 6)))

        coef = np.sqrt(2)

        dI3dC_e_pg[:, :, 0] = cyy * czz - cyz**2
        dI3dC_e_pg[:, :, 1] = cxx * czz - cxz**2
        dI3dC_e_pg[:, :, 2] = cxx * cyy - cxy**2
        dI3dC_e_pg[:, :, 3] = coef * (-cxx * cyz + cxy * cxz)
        dI3dC_e_pg[:, :, 4] = coef * (cxy * cyz - cxz * cyy)
        dI3dC_e_pg[:, :, 5] = coef * (-cxy * czz + cxz * cyz)

        return dI3dC_e_pg

    @staticmethod
    def Compute_d2I3dC(
        mesh: Mesh, u: _types.FloatArray, matrixType=MatrixType.rigi
    ) -> FeArray.FeArrayALike:
        """Computes d2I3dC(u)

        Parameters
        ----------
        mesh : Mesh
            mesh
        u : _types.FloatArray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        FeArray
            d2I3dC_e_pg of shape (Ne, pg, 6, 6)
        """

        cxx, cxy, cxz, _, cyy, cyz, _, _, czz = HyperElastic._Compute_C(
            mesh, u, matrixType
        )

        Ne, nPg, _ = HyperElastic.__GetDims(mesh, u, matrixType)

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

        return d2I3dC_e_pg

    # -------------------------------------
    # Compute I4
    # -------------------------------------
    @staticmethod
    def Compute_I4(
        mesh: Mesh,
        u: _types.FloatArray,
        T: _types.FloatArray,
        matrixType=MatrixType.rigi,
    ) -> FeArray.FeArrayALike:
        """Computes I4(u)

        Parameters
        ----------
        mesh : Mesh
            mesh
        u : _types.FloatArray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        T : _types.FloatArray
            direction
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        FeArray
            I4_e_pg of shape (Ne, pg)
        """

        C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)

        if not isinstance(T, FeArray):
            T = FeArray.asfearray(T, True)

        assert T._type == "vector", "T must be a (..., 3) array"  # type: ignore

        I4_e_pg = T @ C_e_pg @ T

        return I4_e_pg

    @staticmethod
    def Compute_dI4dC(T: _types.FloatArray) -> FeArray.FeArrayALike:
        """Computes dI4dC(u)

        Parameters
        ----------
        mesh : Mesh
            mesh
        T : _types.FloatArray
            direction

        Returns
        -------
        FeArray
            dI4dC_e_pg of shape (Ne, pg, 6)
        """

        assert (
            isinstance(T, np.ndarray) and T.shape[-1] == 3
        ), "T must be a (..., 3) array"

        dI4dC_e_pg = Project_Kelvin(TensorProd(T, T))

        return dI4dC_e_pg

    @staticmethod
    def Compute_d2I4dC() -> FeArray.FeArrayALike:
        """Computes d2I4dC(u)

        Returns
        -------
        FeArray
            d2I4dC of shape (6, 6)
        """

        return FeArray.zeros(1, 1, 6, 6)
