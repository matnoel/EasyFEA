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

        grad_e_pg = mesh.Get_Gradient(u)
        Ne, nPg = grad_e_pg.shape[:2]

        D_e_pg = np.zeros((Ne, nPg, 6, 9), dtype=float)

        def Add_to_D_e_pg(p: int, line: int, values: list[np.ndarray]):
            assert len(values) == 9
            for column in range(9):
                D_e_pg[:,p,line,column] = values[column]

        for p in range(nPg):
            dxux, dyux, dzux = [grad_e_pg[:, p, 0, i] for i in range(3)]
            dxuy, dyuy, dzuy = [grad_e_pg[:, p, 1, i] for i in range(3)]
            dxuz, dyuz, dzuz = [grad_e_pg[:, p, 2, i] for i in range(3)]

            Add_to_D_e_pg(p, 0, [1+dxux, 0, 0, dxuy, 0, 0, dxuz, 0, 0]) # xx
            Add_to_D_e_pg(p, 1, [0, dyux, 0, 0, 1+dyuy, 0, 0, dyuz, 0]) # yy
            Add_to_D_e_pg(p, 2, [0, 0, dzux, 0, 0, dzuy, 0, 0, 1+dzuz]) # zz
            Add_to_D_e_pg(p, 3, [dzux, 0, 1 + dxux, dzuy, 0, dxuy, 1 + dzuz, 0, dxuz]) # yz
            Add_to_D_e_pg(p, 4, [0, dzux, dyux, 0, dzuy, 1 + dyuy, 0, 1 + dzuz, dyuz]) # xz
            Add_to_D_e_pg(p, 5, [dyux, 1+dxux, 0, 1+dyuy, dxuy, 0, dyuz, dxuz, 0]) # xy

        return D_e_pg