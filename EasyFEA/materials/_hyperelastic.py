# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Hyper elastic module used to compute matrices."""

import numpy as np

from ..fem import Mesh, MatrixType
from ..utilities._linalg import Transpose, Trace, Det

# ------------------------------------------------------------------------------
# Functions for matrices
# ------------------------------------------------------------------------------

class HyperElastic:

    def __CheckFormat(mesh: Mesh, u: np.ndarray, matrixType: MatrixType) -> None:
        assert isinstance(mesh, Mesh), "mesh must be an Mesh object"
        assert isinstance(u, np.ndarray) and u.size % mesh.Nn == 0, "wrong displacement field dimension"
        dim = u.size // mesh.Nn
        assert dim in [2, 3], "wrong displacement field dimension"
        assert matrixType in MatrixType.Get_types(), f"matrixType must be in {MatrixType.Get_types()}"

    def __GetDims(mesh: Mesh, u: np.ndarray, matrixType:MatrixType) -> tuple[int, int, int]:
        """return Ne, nPg, dim"""
        HyperElastic.__CheckFormat(mesh, u, matrixType)
        Ne = mesh.Ne
        dim = u.size // mesh.Nn
        nPg = mesh.Get_jacobian_e_pg(matrixType).shape[1]
        return (Ne, nPg, dim)

    def Compute_F(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes the deformation gradient F = I + grad(u)"""

        HyperElastic.__CheckFormat(mesh, u, matrixType)

        grad_e_pg = mesh.Get_Gradient_e_pg(u, matrixType)
        dim = grad_e_pg.shape[2]

        F_e_pg = np.eye(dim) + grad_e_pg

        return F_e_pg
    
    def Compute_J(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes the deformation gradient J = det(F)"""

        F_e_pg = HyperElastic.Compute_F(mesh, u, matrixType)

        J_e_pg = Det(F_e_pg)

        return J_e_pg

    def Compute_C(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes the right Cauchy-Green deformation  C = F'.F"""

        F_e_pg = HyperElastic.Compute_F(mesh, u, matrixType)

        C_e_pg = Transpose(F_e_pg) @ F_e_pg

        return C_e_pg
    
    def Compute_e(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes the Green-Lagrange deformation  e = 1/2 (C - I)"""

        C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)
        dim = C_e_pg.shape[2]

        e_e_pg = 1/2 * (C_e_pg - np.eye(dim)) 

        return e_e_pg
       
    def Compute_Epsilon(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:

        HyperElastic.__CheckFormat(mesh, u, matrixType)

        # compute grad
        grad_e_pg = mesh.Get_Gradient_e_pg(u, matrixType)
        Ne, nPg, dim = grad_e_pg.shape[:3]

        assert dim in [2, 3]

        # 2d: dxux, dyux, dxuy, dyuy
        # 3d: dxux, dyux, dzu, dxuy, dyuy, dzuy, dxuz, dyuz, dzuz
        gradAsVect_e_pg = np.reshape(grad_e_pg, (Ne, nPg, -1))

        c = 2**(-1/2)

        if dim == 2:
            mat = np.array([
                [1, 0, 0, 0], # xx
                [0, 0, 0, 1], # yy
                [0, c, c, 0]  # xy
            ])
        else:
            mat = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0], # xx
                [0, 0, 0, 0, 1, 0, 0, 0, 0], # yy
                [0, 0, 0, 0, 0, 0, 0, 0, 1], # zz
                [0, 0, 0, 0, 0, c, 0, c, 0], # yz
                [0, 0, c, 0, 0, 0, c, 0, 0], # xz
                [0, c, 0, c, 0, 0, 0, 0, 0]  # xy
            ])

        Eps_e_pg = np.einsum("ij,epj->epi", mat, gradAsVect_e_pg)

        return Eps_e_pg
    
    @staticmethod
    def Compute_De(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes De(u)

        [1+dxux, 0, 0, dxuy, 0, 0, dxuz, 0, 0] # xx \n
        [0, dyux, 0, 0, 1+dyuy, 0, 0, dyuz, 0] # yy \n
        [0, 0, dzux, 0, 0, dzuy, 0, 0, 1+dzuz] # zz \n
        [dzux, 0, 1 + dxux, dzuy, 0, dxuy, 1 + dzuz, 0, dxuz] # yz \n
        [0, dzux, dyux, 0, dzuy, 1 + dyuy, 0, 1 + dzuz, dyuz] # xz \n
        [dyux, 1+dxux, 0, 1+dyuy, dxuy, 0, dyuz, dxuz, 0] # xy


        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [u1, v1, w1, . . ., uN, vN, wN]
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        np.ndarray
            D_e_pg of shape (e, pg, 6, 9)
        """

        HyperElastic.__CheckFormat(mesh, u, matrixType)

        grad_e_pg = mesh.Get_Gradient_e_pg(u)
        Ne, nPg, dim = grad_e_pg.shape[:3]

        if dim == 2:
            D_e_pg = np.zeros((Ne, nPg, 6, 9), dtype=float)
        else:
            D_e_pg = np.zeros((Ne, nPg, 3, 4), dtype=float)

        def Add_to_D_e_pg(p: int, line: int, values: list[np.ndarray], coef=1.):
            N = 4 if dim == 2 else 9            
            for column in range(N):
                D_e_pg[:,p,line,column] = values[column] * coef

        cM = 2**(-1/2)

        for p in range(nPg):
            dxux, dyux, dzux = [grad_e_pg[:, p, 0, i] for i in range(3)]
            dxuy, dyuy, dzuy = [grad_e_pg[:, p, 1, i] for i in range(3)]
            dxuz, dyuz, dzuz = [grad_e_pg[:, p, 2, i] for i in range(3)]

            Add_to_D_e_pg(p, 0, [1+dxux, 0, 0, dxuy, 0, 0, dxuz, 0, 0]) # xx
            Add_to_D_e_pg(p, 1, [0, dyux, 0, 0, 1+dyuy, 0, 0, dyuz, 0]) # yy
            Add_to_D_e_pg(p, 2, [0, 0, dzux, 0, 0, dzuy, 0, 0, 1+dzuz]) # zz
            Add_to_D_e_pg(p, 3, [dzux, 0, 1 + dxux, dzuy, 0, dxuy, 1 + dzuz, 0, dxuz], cM) # yz
            Add_to_D_e_pg(p, 4, [0, dzux, dyux, 0, dzuy, 1 + dyuy, 0, 1 + dzuz, dyuz], cM) # xz
            Add_to_D_e_pg(p, 5, [dyux, 1+dxux, 0, 1+dyuy, dxuy, 0, dyuz, dxuz, 0], cM) # xy

        return D_e_pg
    
    # --------------------------------------------------------------------------
    # Compute invariants
    # --------------------------------------------------------------------------    
    
    # -------------------------------------
    # Compute I1
    # -------------------------------------
    def Compute_I1(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:

        C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)

        I1_e_pg = Trace(C_e_pg)

        return I1_e_pg

    def Compute_dI1dC(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:

        Ne, nPg, dim = HyperElastic.__GetDims(mesh, u, matrixType)

        dI1dC_e_pg = np.zeros((Ne, nPg, dim, dim), dtype=float)

        for d in range(dim):
            dI1dC_e_pg[:,:,d,d]=1

        return dI1dC_e_pg

    # -------------------------------------
    # Compute I2
    # -------------------------------------
    def Compute_I2(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:

        C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)

        I2_e_pg = 1/2 * (Trace(C_e_pg)**2 - Trace(C_e_pg @ C_e_pg))

        return I2_e_pg
    
    def Compute_dI2dC(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:

        _, _, dim = HyperElastic.__GetDims(mesh, u, matrixType)

        I1_e_pg = HyperElastic.Compute_I1(mesh, u, matrixType)
        C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)

        dI2dC_e_pg = I1_e_pg @ np.eye(dim) - C_e_pg 

        return dI2dC_e_pg

    # -------------------------------------
    # Compute I3
    # -------------------------------------
    def Compute_I2(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:

        C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)

        I3_e_pg = Det(C_e_pg)

        return I3_e_pg
    
    def Compute_dI2dC(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:

        _, _, dim = HyperElastic.__GetDims(mesh, u, matrixType)

        I1_e_pg = HyperElastic.Compute_I1(mesh, u, matrixType)
        C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)

        dI2dC_e_pg = I1_e_pg @ np.eye(dim) - C_e_pg 

        return dI2dC_e_pg