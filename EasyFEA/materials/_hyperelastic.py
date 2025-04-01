# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Hyper elastic module used to compute matrices."""

import numpy as np

from ..fem import Mesh, MatrixType

class HyperElastic:

    @staticmethod
    def Compute_D_e_pg(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Compute De(u)

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

        grad_e_pg = mesh.Get_Gradient_e_pg(u)
        Ne, nPg = grad_e_pg.shape[:2]

        D_e_pg = np.zeros((Ne, nPg, 6, 9), dtype=float)

        def Add_to_D_e_pg(p: int, line: int, values: list[np.ndarray], coef=1.):
            assert len(values) == 9
            for column in range(9):
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
    
    def Compute_Epsilon_e_pg(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:

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