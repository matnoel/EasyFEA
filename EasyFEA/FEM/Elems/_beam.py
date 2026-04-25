# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Beam element module."""

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from .._group_elem import _GroupElem
from ...Utilities import _types

# fem
from .._linalg import FeArray  # , Trace, Transpose, Det, Inv
from .._mesh import Mesh
from . import _seg

# utils
from .._utils import ElemType, MatrixType

if TYPE_CHECKING:
    from ...Models.Beam import BeamStructure

# --------------------------------------------
# Euler-Bernoulli elements
# --------------------------------------------


class _EulerBernoulli(_GroupElem):

    # Beams shapes functions
    # Use hermitian shape functions

    # N

    @abstractmethod
    def _Hermitian_N(self) -> _types.FloatArray:
        """Hermitian shape functions in the (ξ, η, ζ) coordinates.\n
        [phi_i psi_i . . . phi_n psi_n]\n
        (nPe*2, 1)
        """
        return None  # type: ignore [return-value]

    def Get_Hermitian_N_pg(self) -> _types.FloatArray:
        """Evaluates Hermitian shape functions in the (ξ, η, ζ) coordinates.\n
        [phi_i psi_i . . . phi_n psi_n]\n
        (nPg, nPe*2)
        """
        if self.dim != 1:
            return None  # type: ignore [return-value]

        matrixType = MatrixType.beam

        N = self._Hermitian_N()

        gauss = self.Get_gauss(matrixType)
        N_pg = _GroupElem._Eval_Functions(N, gauss.coord)

        return N_pg

    def Get_Hermitian_N_e_pg(self) -> FeArray.FeArrayALike:  # type: ignore
        """Evaluates Hermitian shape functions in (x, y, z) coordinates.\n
        [phi_i psi_i . . . phi_n psi_n]\n
        (Ne, nPg, 1, nPe*2)
        """
        if self.dim != 1:
            return None  # type: ignore [return-value]

        invF_e_pg = self.Get_invF_e_pg(MatrixType.beam)[:, :, 0, 0]
        N_pg = FeArray.asfearray(self.Get_Hermitian_N_pg()[np.newaxis])
        nPe = self.nPe

        N_e_pg = invF_e_pg * N_pg

        # multiply by the beam length on psi_i,xx functions
        l_e = self.length_e
        columns = np.arange(1, nPe * 2, 2)
        for column in columns:
            N_e_pg[:, :, 0, column] = np.einsum(
                "ep,e->ep", N_e_pg[:, :, 0, column], l_e, optimize="optimal"
            )

        return N_e_pg

    # dN

    @abstractmethod
    def _Hermitian_dN(self) -> _types.FloatArray:
        """Hermitian shape functions first derivatives in the (ξ, η, ζ) coordinates.\n
        [phi_i,ξ psi_i,ξ . . . phi_n,ξ psi_n,ξ]\n
        (nPe*2, 1)
        """
        return None  # type: ignore [return-value]

    def Get_Hermitian_dN_pg(self) -> _types.FloatArray:
        """Evaluates Hermitian shape functions first derivatives in the (ξ, η, ζ) coordinates.\n
        [phi_i,ξ psi_i,ξ . . . phi_n,ξ psi_n,ξ]\n
        (nPg, nPe*2)
        """
        if self.dim != 1:
            return None  # type: ignore [return-value]

        matrixType = MatrixType.beam

        dN = self._Hermitian_dN()

        gauss = self.Get_gauss(matrixType)
        dN_pg = _GroupElem._Eval_Functions(dN, gauss.coord)

        return dN_pg

    def Get_Hermitian_dN_e_pg(self) -> FeArray.FeArrayALike:
        """Evaluates the first-order derivatives of Hermitian shape functions in (x, y, z) coordinates.\n
        [phi_i,x psi_i,x . . . phi_n,x psi_n,x]\n
        (Ne, nPg, 1, nPe*2)
        """
        if self.dim != 1:
            return None  # type: ignore [return-value]

        invF_e_pg = self.Get_invF_e_pg(MatrixType.beam)[:, :, 0, 0]
        dN_pg = FeArray.asfearray(self.Get_Hermitian_dN_pg()[np.newaxis])

        dN_e_pg = invF_e_pg * dN_pg

        # multiply by the beam length on psi_i,xx functions
        l_e = self.length_e
        nPe = self.nPe
        columns = np.arange(1, nPe * 2, 2)
        for column in columns:
            dN_e_pg[:, :, 0, column] = np.einsum(
                "ep,e->ep", dN_e_pg[:, :, 0, column], l_e, optimize="optimal"
            )

        return dN_e_pg

    # ddN

    @abstractmethod
    def _Hermitian_ddN(self) -> _types.FloatArray:
        """Hermitian shape functions second derivatives in the (ξ, η, ζ) coordinates.\n
        [phi_i,ξ psi_i,ξ . . . phi_n,ξ psi_n,ξ]\n
        (nPe*2, 2)
        """
        return None  # type: ignore [return-value]

    def Get_Hermitian_ddN_pg(self) -> _types.FloatArray:
        """Evaluates Hermitian shape functions second derivatives in the (ξ, η, ζ) coordinates.\n
        [phi_i,ξ psi_i,ξ . . . phi_n,ξ x psi_n,ξ]\n
        (nPg, nPe*2)
        """
        if self.dim != 1:
            return None  # type: ignore [return-value]

        matrixType = MatrixType.beam

        ddN = self._Hermitian_ddN()

        gauss = self.Get_gauss(matrixType)
        ddN_pg = _GroupElem._Eval_Functions(ddN, gauss.coord)

        return ddN_pg

    def Get_Hermitian_ddN_e_pg(self) -> FeArray.FeArrayALike:
        """Evaluates the second-order derivatives of Hermitian shape functions in (x, y, z) coordinates.\n
        [phi_i,xx psi_i,xx . . . phi_n,xx psi_n,xx]\n
        (Ne, nPg, 1, nPe*2)
        """
        if self.dim != 1:
            return None  # type: ignore [return-value]

        invF_e_pg = self.Get_invF_e_pg(MatrixType.beam)[:, :, 0, 0]
        ddN_pg = FeArray.asfearray(self.Get_Hermitian_ddN_pg()[np.newaxis])
        nPe = self.nPe

        ddN_e_pg = invF_e_pg * invF_e_pg * ddN_pg

        # multiply by the beam length on psi_i,xx functions
        l_e = self.length_e
        columns = np.arange(1, nPe * 2, 2)
        for column in columns:
            ddN_e_pg[:, :, 0, column] = np.einsum(
                "ep,e->ep", ddN_e_pg[:, :, 0, column], l_e, optimize="optimal"
            )

        return ddN_e_pg

    # projection matrix

    def _Compute_P_e_pg(self, beamStructure: "BeamStructure") -> FeArray.FeArrayALike:

        # data
        Ne = self.Ne
        nPe = self.nPe
        dof_n = beamStructure.dof_n

        P = np.zeros((self.Ne, 3, 3))
        for beam in beamStructure.beams:
            elems = self.Get_Elements_Tag(beam.name)
            P[elems] = beam._Calc_P()

        P_e_pg = FeArray.zeros(Ne, 1, dof_n * nPe, dof_n * nPe)
        N = P.shape[-1]
        lines = np.repeat(range(N), N)
        columns = np.array(list(range(N)) * N)
        for n in range(dof_n * nPe // 3):
            P_e_pg[:, 0, lines + n * N, columns + n * N] = P[:, lines, columns]

        return P_e_pg

    # Euler Bernoulli problem matrices

    def Get_beam_N_e_pg(self, beamStructure: "BeamStructure") -> FeArray.FeArrayALike:
        """Euler Bernoulli beam shape functions for the mass matrix."""

        # Example in matlab :
        # https://github.com/fpled/FEMObject/blob/master/BASIC/MODEL/ELEMENTS/%40BEAM/calc_N.m

        matrixType = MatrixType.beam

        # get the beam model
        dim = beamStructure.dim
        dof_n = beamStructure.dof_n

        # get matrices to work with
        Nu_pg = self.Get_N_pg(matrixType)
        Nv_e_pg = self.Get_Hermitian_N_e_pg()
        dNv_e_pg = self.Get_Hermitian_dN_e_pg()

        # Data
        nPe = self.nPe
        Ne, nPg = dNv_e_pg.shape[:2]

        if dim == 1:
            # u = [u1, . . . , un]

            # N = [N_i, . . . , N_n]

            idx_ux = np.arange(dof_n * nPe)

            N_e_pg = np.zeros((Ne, nPg, 1, dof_n * nPe))
            N_e_pg[:, :, 0, idx_ux] = Nu_pg[:, :, 0]

        elif dim == 2:
            # u = [u1, v1, rz1, . . . , un, vn, rzn]

            # N = [N_i, 0, 0, ... , N_n, 0, 0,]
            #     [0, Phi_i, Psi_i, ... , 0, Phi_i, Psi_i]
            #     [0, dPhi_i, dPsi_i, ... , 0, dPhi_i, dPsi_i]

            idx = np.arange(dof_n * nPe, dtype=int).reshape(nPe, -1)

            idx_ux = idx[:, 0]  # [0,3] (SEG2) [0,3,6] (SEG3)
            idx_uy = np.reshape(idx[:, 1:], -1)  # [1,2,4,5] (SEG2) [1,2,4,5,7,8] (SEG3)

            N_e_pg = np.zeros((Ne, nPg, 3, dof_n * nPe))

            N_e_pg[:, :, 0, idx_ux] = Nu_pg[:, :, 0]  # traction / compression to get u
            N_e_pg[:, :, 1, idx_uy] = Nv_e_pg[:, :, 0]  # flexion z to get v
            N_e_pg[:, :, 2, idx_uy] = dNv_e_pg[:, :, 0]  # flexion z to get rz

        elif dim == 3:
            # u = [u1, v1, w1, rx1, ry1, rz1, . . . , un, vn, wn, rxn, ryn, rzn]

            # N = [N_i, 0, 0, 0, 0, 0, ... , N_n, 0, 0, 0, 0, 0]
            #     [0, Phi_i, 0, 0, 0, Psi_i, ... , 0, Phi_n, 0, 0, 0, Psi_n]
            #     [0, 0, dPhi_i, 0, -dPsi_i, 0, ... , 0, 0, dPhi_n, 0, -dPsi_n, 0]
            #     [0, 0, 0, N_i, 0, 0, ... , 0, 0, 0, N_n, 0, 0]
            #     [0, 0, -dPhi_i, 0, dPsi_i, 0, ... , 0, 0, -dPhi_n, 0, dPsi_n, 0]
            #     [0, dPhi_i, 0, 0, 0, dPsi_i, ... , 0, dPhi_i, 0, 0, 0, dPsi_n]

            idx = np.arange(dof_n * nPe, dtype=int).reshape(nPe, -1)
            idx_ux = idx[:, 0]  # [0,6] (SEG2) [0,6,12] (SEG3)
            idx_uy = np.reshape(
                idx[:, [1, 5]], -1
            )  # [1,5,7,11] (SEG2) [1,5,7,11,13,17] (SEG3)
            idx_uz = np.reshape(
                idx[:, [2, 4]], -1
            )  # [2,4,8,10] (SEG2) [2,4,8,10,14,16] (SEG3)
            idx_rx = idx[:, 3]  # [3,9] (SEG2) [3,9,15] (SEG3)
            idPsi = np.arange(1, nPe * 2, 2)  # [1,3] (SEG2) [1,3,5] (SEG3)

            Nvz_e_pg = Nv_e_pg.copy()
            Nvz_e_pg[:, :, 0, idPsi] *= -1

            dNz_e_pg = dNv_e_pg.copy()
            dNz_e_pg[:, :, 0, idPsi] *= -1

            N_e_pg = np.zeros((Ne, nPg, 6, dof_n * nPe))

            N_e_pg[:, :, 0, idx_ux] = Nu_pg[:, :, 0]
            N_e_pg[:, :, 1, idx_uy] = Nv_e_pg[:, :, 0]
            N_e_pg[:, :, 2, idx_uz] = Nvz_e_pg[:, :, 0]
            N_e_pg[:, :, 3, idx_rx] = Nu_pg[:, :, 0]
            N_e_pg[:, :, 4, idx_uz] = -dNz_e_pg[:, :, 0]  # ry = -uz'
            N_e_pg[:, :, 5, idx_uy] = dNv_e_pg[:, :, 0]  # rz = uy'

        N_e_pg = FeArray.asfearray(N_e_pg)

        if dim > 1:
            P_e_pg = self._Compute_P_e_pg(beamStructure=beamStructure)

            N_e_pg = N_e_pg @ P_e_pg

        return N_e_pg

    def Get_beam_B_e_pg(
        self, beamStructure: "BeamStructure"
    ) -> FeArray.FeArrayALike:  # type: ignore
        """Get Euler Bernoulli beam B matrix (strains from displacements)."""

        # Example in matlab :
        # https://github.com/fpled/FEMObject/blob/master/BASIC/MODEL/ELEMENTS/%40BEAM/calc_B.m

        matrixType = MatrixType.beam

        # Recovering the beam model
        dim = beamStructure.dim
        dof_n = beamStructure.dof_n

        # Recover matrices to work with
        dN_e_pg = self.Get_dN_e_pg(matrixType)
        ddNv_e_pg = self.Get_Hermitian_ddN_e_pg()

        # Data
        nPe = self.nPe
        Ne, nPg = dN_e_pg.shape[:2]

        if dim == 1:
            # u = [u1, . . . , un]

            # B = [dN_i, . . . , dN_n]

            idx_ux = np.arange(dof_n * nPe)

            B_e_pg = np.zeros((Ne, nPg, 1, dof_n * nPe), dtype=float)
            B_e_pg[:, :, 0, idx_ux] = dN_e_pg[:, :, 0]

        elif dim == 2:
            # u = [u1, v1, rz1, . . . , un, vn, rzn]

            # B = [dN_i, 0, 0, ... , dN_n, 0, 0,]
            #     [0, ddPhi_i, ddPsi_i, ... , 0, ddPhi_i, ddPsi_i]

            idx = np.arange(dof_n * nPe, dtype=int).reshape(nPe, -1)

            idx_ux = idx[:, 0]  # [0,3] (SEG2) [0,3,6] (SEG3)
            idx_uy = np.reshape(idx[:, 1:], -1)  # [1,2,4,5] (SEG2) [1,2,4,5,7,8] (SEG3)

            B_e_pg = np.zeros((Ne, nPg, 2, dof_n * nPe), dtype=float)

            B_e_pg[:, :, 0, idx_ux] = dN_e_pg[:, :, 0]  # traction / compression
            B_e_pg[:, :, 1, idx_uy] = ddNv_e_pg[:, :, 0]  # flexion along z

        elif dim == 3:
            # u = [u1, v1, w1, rx1, ry1, rz1, . . . , un, vn, wn, rxn, ryn, rzn]

            # B = [dN_i, 0, 0, 0, 0, 0, ... , dN_n, 0, 0, 0, 0, 0]
            #     [0, 0, 0, dN_i, 0, 0, ... , 0, 0, 0, dN_n, 0, 0]
            #     [0, 0, ddPhi_i, 0, -ddPsi_i, 0, ... , 0, 0, ddPhi_n, 0, -ddPsi_n, 0]
            #     [0, ddPhi_i, 0, 0, 0, ddPsi_i, ... , 0, ddPhi_i, 0, 0, 0, ddPsi_n]

            idx = np.arange(dof_n * nPe).reshape(nPe, -1)
            idx_ux = idx[:, 0]  # [0,6] (SEG2) [0,6,12] (SEG3)
            idx_uy = np.reshape(
                idx[:, [1, 5]], -1
            )  # [1,5,7,11] (SEG2) [1,5,7,11,13,17] (SEG3)
            idx_uz = np.reshape(
                idx[:, [2, 4]], -1
            )  # [2,4,8,10] (SEG2) [2,4,8,10,14,16] (SEG3)
            idx_rx = idx[:, 3]  # [3,9] (SEG2) [3,9,15] (SEG3)

            idPsi = np.arange(1, nPe * 2, 2)  # [1,3] (SEG2) [1,3,5] (SEG3)
            ddNvz_e_pg = ddNv_e_pg.copy()
            ddNvz_e_pg[:, :, 0, idPsi] *= -1  # RY = -UZ'

            B_e_pg = np.zeros((Ne, nPg, 4, dof_n * nPe), dtype=float)

            B_e_pg[:, :, 0, idx_ux] = dN_e_pg[:, :, 0]  # traction / compression
            B_e_pg[:, :, 1, idx_rx] = dN_e_pg[:, :, 0]  # torsion
            B_e_pg[:, :, 2, idx_uz] = ddNvz_e_pg[:, :, 0]  # flexion along y
            B_e_pg[:, :, 3, idx_uy] = ddNv_e_pg[:, :, 0]  # flexion along z
        else:
            raise TypeError("dim error")

        B_e_pg = FeArray.asfearray(B_e_pg)

        if dim > 1:
            Pglob_e_pg = self._Compute_P_e_pg(beamStructure=beamStructure)

            B_e_pg = B_e_pg @ Pglob_e_pg

        return B_e_pg


class _Timoshenko(_EulerBernoulli):

    def Get_beam_N_e_pg(self, beamStructure: "BeamStructure") -> FeArray.FeArrayALike:
        """Timoshenko beam shape functions for the mass matrix."""

        matrixType = MatrixType.beam
        dim = beamStructure.dim
        dof_n = beamStructure.dof_n

        # Lagrange shape functions N_i(ξ) evaluated at Gauss points
        Nu_pg = self.Get_N_pg(matrixType)[:, 0, :]  # (nPg, nPe)
        # Hermitian shape functions for transverse displacement
        Nv_e_pg = self.Get_Hermitian_N_e_pg()  # (Ne, nPg, 1, nPe*2)

        # data
        nPe = self.nPe
        Ne, nPg = Nv_e_pg.shape[:2]

        if dim == 1:
            idx_ux = np.arange(dof_n * nPe)
            N_e_pg = np.zeros((Ne, nPg, 1, dof_n * nPe))
            N_e_pg[:, :, 0, idx_ux] = Nu_pg  # u: Lagrange

        elif dim == 2:
            # u: Lagrange   v: Hermitian   rz: independent Lagrange
            idx = np.arange(dof_n * nPe, dtype=int).reshape(nPe, -1)
            idx_ux = idx[:, 0]
            idx_uy = np.reshape(idx[:, 1:], -1)  # [v,rz] DOFs (Hermitian)
            idx_rz = idx[:, 2].flatten()  # rz DOFs only

            N_e_pg = np.zeros((Ne, nPg, 3, dof_n * nPe))
            N_e_pg[:, :, 0, idx_ux] = Nu_pg  # u: Lagrange
            N_e_pg[:, :, 1, idx_uy] = Nv_e_pg[:, :, 0]  # v: Hermitian
            N_e_pg[:, :, 2, idx_rz] = Nu_pg  # rz: independent Lagrange

        elif dim == 3:
            idx = np.arange(dof_n * nPe, dtype=int).reshape(nPe, -1)
            idx_ux = idx[:, 0]
            idx_uy = np.reshape(idx[:, [1, 5]], -1)  # [v,rz] DOFs (Hermitian xy)
            idx_uz = np.reshape(idx[:, [2, 4]], -1)  # [w,ry] DOFs (Hermitian xz)
            idx_rx = idx[:, 3]
            idx_ry = idx[:, 4].flatten()
            idx_rz = idx[:, 5].flatten()

            idPsi = np.arange(1, nPe * 2, 2)
            Nvz_e_pg = Nv_e_pg.copy()
            Nvz_e_pg[:, :, 0, idPsi] *= -1  # sign-flipped for w (ry = -w' convention)

            N_e_pg = np.zeros((Ne, nPg, 6, dof_n * nPe))
            N_e_pg[:, :, 0, idx_ux] = Nu_pg  # u: Lagrange
            N_e_pg[:, :, 1, idx_uy] = Nv_e_pg[:, :, 0]  # v: Hermitian
            N_e_pg[:, :, 2, idx_uz] = Nvz_e_pg[:, :, 0]  # w: Hermitian (sign-flipped)
            N_e_pg[:, :, 3, idx_rx] = Nu_pg  # rx: independent Lagrange
            N_e_pg[:, :, 4, idx_ry] = Nu_pg  # ry: independent Lagrange
            N_e_pg[:, :, 5, idx_rz] = Nu_pg  # rz: independent Lagrange
        else:
            raise TypeError("dim error")

        N_e_pg = FeArray.asfearray(N_e_pg)

        if dim > 1:
            P_e_pg = self._Compute_P_e_pg(beamStructure=beamStructure)
            N_e_pg = N_e_pg @ P_e_pg

        return N_e_pg

    def Get_beam_B_e_pg(
        self, beamStructure: "BeamStructure"
    ) -> FeArray.FeArrayALike:  # type: ignore
        """Get Timoshenko beam B matrix (strains from displacements)."""

        matrixType = MatrixType.beam

        dim = beamStructure.dim
        dof_n = beamStructure.dof_n

        # Lagrange shape functions and derivatives (axial / torsion / rotations)
        # Nu_pg shape (nPg, nPe) — correct node-wise values at each Gauss point
        Nu_pg = self.Get_N_pg(matrixType)[:, 0, :]  # (nPg, nPe)
        dN_e_pg = self.Get_dN_e_pg(matrixType)  # (Ne, nPg, 1, nPe)
        # Hermitian first-order derivatives for transverse displacements (v, w)
        dNv_e_pg = self.Get_Hermitian_dN_e_pg()  # (Ne, nPg, 1, nPe*2)

        nPe = self.nPe
        Ne, nPg = dN_e_pg.shape[:2]

        if dim == 1:
            # u = [u1, . . . , un]
            # B = [dN_i, . . . , dN_n]
            idx_ux = np.arange(dof_n * nPe)
            B_e_pg = np.zeros((Ne, nPg, 1, dof_n * nPe), dtype=float)
            B_e_pg[:, :, 0, idx_ux] = dN_e_pg[:, :, 0]

        elif dim == 2:
            # u = [u1, v1, rz1, . . . , un, vn, rzn]
            #
            # B = [dN_i, 0,       0,          ..., dN_n, 0,       0         ]  axial
            #     [0,    0,       dN_i,        ..., 0,    0,       dN_n      ]  bending: dθ/dx
            #     [0,    dPhi_i,  dPsi_i-N_i,  ..., 0,    dPhi_n,  dPsi_n-N_n]  shear: v'−rz

            idx = np.arange(dof_n * nPe, dtype=int).reshape(nPe, -1)
            idx_ux = idx[:, 0]  # u DOFs:  [0,3]     (SEG2)
            idx_uy = np.reshape(idx[:, 1:], -1)  # [v,rz] DOFs: [1,2,4,5] (SEG2)
            idx_rz = idx[:, 2].flatten()  # rz DOFs: [2,5]     (SEG2)

            B_e_pg = np.zeros((Ne, nPg, 3, dof_n * nPe), dtype=float)
            B_e_pg[:, :, 0, idx_ux] = dN_e_pg[:, :, 0]  # axial: du/dx
            B_e_pg[:, :, 1, idx_rz] = dN_e_pg[:, :, 0]  # bending: dθ/dx (Lagrange)
            B_e_pg[:, :, 2, idx_uy] = dNv_e_pg[:, :, 0]  # shear v' (Hermitian)
            B_e_pg[:, :, 2, idx_rz] -= Nu_pg  # shear: subtract rz (Lagrange)

        elif dim == 3:
            # u = [u1, v1, w1, rx1, ry1, rz1, . . . , un, vn, wn, rxn, ryn, rzn]
            #
            # B has 6 rows: [axial, torsion, flex-y, flex-z, shear-y, shear-z]
            #   flex-y: kappa_y = -dRy/dx  (ry = -w' in EB => d²w/dx² = -dRy/dx)
            #   flex-z: kappa_z =  dRz/dx  (rz =  v' in EB => d²v/dx² =  dRz/dx)
            #   gamma_y = v' - rz   (Hermitian dNv  at [v,rz] DOFs minus Lagrange N at rz)
            #   gamma_z = w' + ry   (Hermitian dNvz at [w,ry] DOFs plus  Lagrange N at ry)

            idx = np.arange(dof_n * nPe).reshape(nPe, -1)
            idx_ux = idx[:, 0]  # u  DOFs: [0,6]       (SEG2)
            idx_uy = np.reshape(idx[:, [1, 5]], -1)  # [v,rz] DOFs: [1,5,7,11] (SEG2)
            idx_uz = np.reshape(idx[:, [2, 4]], -1)  # [w,ry] DOFs: [2,4,8,10] (SEG2)
            idx_rx = idx[:, 3]  # rx DOFs: [3,9]       (SEG2)
            idx_ry = idx[:, 4].flatten()  # ry DOFs: [4,10]      (SEG2)
            idx_rz = idx[:, 5].flatten()  # rz DOFs: [5,11]      (SEG2)

            # sign-flipped Hermitian derivative for w' (ry = -w' convention)
            idPsi = np.arange(1, nPe * 2, 2)
            dNvz_e_pg = dNv_e_pg.copy()
            dNvz_e_pg[:, :, 0, idPsi] *= -1

            B_e_pg = np.zeros((Ne, nPg, 6, dof_n * nPe), dtype=float)
            B_e_pg[:, :, 0, idx_ux] = dN_e_pg[:, :, 0]  # axial
            B_e_pg[:, :, 1, idx_rx] = dN_e_pg[:, :, 0]  # torsion
            B_e_pg[:, :, 2, idx_ry] = -dN_e_pg[:, :, 0]  # flex-y: -dRy/dx (Lagrange)
            B_e_pg[:, :, 3, idx_rz] = dN_e_pg[:, :, 0]  # flex-z:  dRz/dx (Lagrange)
            # shear gamma_y = v' - rz
            B_e_pg[:, :, 4, idx_uy] = dNv_e_pg[:, :, 0]  # v' (Hermitian)
            B_e_pg[:, :, 4, idx_rz] -= Nu_pg  # subtract rz (Lagrange)
            # shear gamma_z = w' + ry  (ry = -w' in EB => gamma_z = 0 in EB)
            B_e_pg[:, :, 5, idx_uz] = dNvz_e_pg[:, :, 0]  # w' (Hermitian, sign-flipped)
            B_e_pg[:, :, 5, idx_ry] += Nu_pg  # add ry (Lagrange)
        else:
            raise TypeError("dim error")

        B_e_pg = FeArray.asfearray(B_e_pg)

        if dim > 1:
            Pglob_e_pg = self._Compute_P_e_pg(beamStructure=beamStructure)
            B_e_pg = B_e_pg @ Pglob_e_pg

        return B_e_pg


# --------------------------------------------
# Euler-Bernoulli elements
# --------------------------------------------


class EULER_BERNOULLI2(_EulerBernoulli, _seg.SEG2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _Hermitian_N(self) -> _types.AnyArray:
        N1 = lambda r: (r - 1) ** 2 * (r + 2) / 4
        N2 = lambda r: (r - 1) ** 2 * (r + 1) / 8
        N3 = lambda r: -(r - 2) * (r + 1) ** 2 / 4
        N4 = lambda r: (r - 1) * (r + 1) ** 2 / 8

        N = np.array([N1, N2, N3, N4]).reshape(-1, 1)

        return N

    def _Hermitian_dN(self) -> _types.AnyArray:
        dN1 = [lambda r: (r - 1) ** 2 / 4 + (r + 2) * (2 * r - 2) / 4]
        dN2 = [lambda r: (r - 1) ** 2 / 8 + (r + 1) * (2 * r - 2) / 8]
        dN3 = [lambda r: -(r - 2) * (2 * r + 2) / 4 - (r + 1) ** 2 / 4]
        dN4 = [lambda r: (r - 1) * (2 * r + 2) / 8 + (r + 1) ** 2 / 8]

        dN = np.array([dN1, dN2, dN3, dN4])

        return dN

    def _Hermitian_ddN(self) -> _types.AnyArray:
        ddN1 = [lambda r: 3 * r / 2]
        ddN2 = [lambda r: 3 * r / 4 - 1 / 4]
        ddN3 = [lambda r: -3 * r / 2]
        ddN4 = [lambda r: 3 * r / 4 + 1 / 4]

        ddN = np.array([ddN1, ddN2, ddN3, ddN4])

        return ddN


class EULER_BERNOULLI3(_EulerBernoulli, _seg.SEG3):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _Hermitian_N(self) -> _types.AnyArray:
        N1 = lambda r: r**2 * (r - 1) ** 2 * (3 * r + 4) / 4
        N2 = lambda r: r**2 * (r - 1) ** 2 * (r + 1) / 8
        N3 = lambda r: -(r**2) * (r + 1) ** 2 * (3 * r - 4) / 4
        N4 = lambda r: r**2 * (r - 1) * (r + 1) ** 2 / 8
        N5 = lambda r: (r - 1) ** 2 * (r + 1) ** 2
        N6 = lambda r: r * (r - 1) ** 2 * (r + 1) ** 2 / 2

        N = np.array([N1, N2, N3, N4, N5, N6]).reshape(-1, 1)

        return N

    def _Hermitian_dN(self) -> _types.AnyArray:
        dN1 = [
            lambda r: 3 * r**2 * (r - 1) ** 2 / 4
            + r**2 * (2 * r - 2) * (3 * r + 4) / 4
            + r * (r - 1) ** 2 * (3 * r + 4) / 2
        ]
        dN2 = [
            lambda r: r**2 * (r - 1) ** 2 / 8
            + r**2 * (r + 1) * (2 * r - 2) / 8
            + r * (r - 1) ** 2 * (r + 1) / 4
        ]
        dN3 = [
            lambda r: -3 * r**2 * (r + 1) ** 2 / 4
            - r**2 * (2 * r + 2) * (3 * r - 4) / 4
            - r * (r + 1) ** 2 * (3 * r - 4) / 2
        ]
        dN4 = [
            lambda r: r**2 * (r - 1) * (2 * r + 2) / 8
            + r**2 * (r + 1) ** 2 / 8
            + r * (r - 1) * (r + 1) ** 2 / 4
        ]
        dN5 = [lambda r: (r - 1) ** 2 * (2 * r + 2) + (r + 1) ** 2 * (2 * r - 2)]
        dN6 = [
            lambda r: r * (r - 1) ** 2 * (2 * r + 2) / 2
            + r * (r + 1) ** 2 * (2 * r - 2) / 2
            + (r - 1) ** 2 * (r + 1) ** 2 / 2
        ]

        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6])

        return dN

    def _Hermitian_ddN(self) -> _types.AnyArray:
        ddN1 = [
            lambda r: 3 * r**2 * (2 * r - 2) / 2
            + r**2 * (3 * r + 4) / 2
            + 3 * r * (r - 1) ** 2
            + r * (2 * r - 2) * (3 * r + 4)
            + (r - 1) ** 2 * (3 * r + 4) / 2
        ]
        ddN2 = [
            lambda r: r**2 * (r + 1) / 4
            + r**2 * (2 * r - 2) / 4
            + r * (r - 1) ** 2 / 2
            + r * (r + 1) * (2 * r - 2) / 2
            + (r - 1) ** 2 * (r + 1) / 4
        ]
        ddN3 = [
            lambda r: -3 * r**2 * (2 * r + 2) / 2
            - r**2 * (3 * r - 4) / 2
            - 3 * r * (r + 1) ** 2
            - r * (2 * r + 2) * (3 * r - 4)
            - (r + 1) ** 2 * (3 * r - 4) / 2
        ]
        ddN4 = [
            lambda r: r**2 * (r - 1) / 4
            + r**2 * (2 * r + 2) / 4
            + r * (r - 1) * (2 * r + 2) / 2
            + r * (r + 1) ** 2 / 2
            + (r - 1) * (r + 1) ** 2 / 4
        ]
        ddN5 = [
            lambda r: 2 * (r - 1) ** 2
            + 2 * (r + 1) ** 2
            + 2 * (2 * r - 2) * (2 * r + 2)
        ]
        ddN6 = [
            lambda r: r * (r - 1) ** 2
            + r * (r + 1) ** 2
            + r * (2 * r - 2) * (2 * r + 2)
            + (r - 1) ** 2 * (2 * r + 2)
            + (r + 1) ** 2 * (2 * r - 2)
        ]

        ddN = np.array([ddN1, ddN2, ddN3, ddN4, ddN5, ddN6])

        return ddN


class EULER_BERNOULLI4(_EulerBernoulli, _seg.SEG4):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _Hermitian_N(self) -> _types.AnyArray:
        N1 = (
            lambda r: 891 * r**7 / 512
            - 729 * r**6 / 512
            - 275976562500001 * r**5 / 100000000000000
            + 1215 * r**4 / 512
            + 137207031250001 * r**3 / 250000000000000
            - 237304687500001 * r**2 / 500000000000000
            - 292968750000009 * r / 10000000000000000
            + 63476562500001 / 2500000000000000
        )
        N2 = (
            lambda r: 81 * r**7 / 512
            - 81 * r**6 / 512
            - 99 * r**5 / 512
            + 99 * r**4 / 512
            + 185546875000001 * r**3 / 5000000000000000
            - 185546875000001 * r**2 / 5000000000000000
            - r / 512
            + 1 / 512
        )
        N3 = (
            lambda r: -891 * r**7 / 512
            - 729 * r**6 / 512
            + 275976562500001 * r**5 / 100000000000000
            + 1215 * r**4 / 512
            - 274414062500003 * r**3 / 500000000000000
            - 243 * r**2 / 512
            + 292968750000013 * r / 10000000000000000
            + 126953125000001 / 5000000000000000
        )
        N4 = (
            lambda r: 81 * r**7 / 512
            + 81 * r**6 / 512
            - 99 * r**5 / 512
            - 99 * r**4 / 512
            + 92773437500001 * r**3 / 2500000000000000
            + 371093749999999 * r**2 / 10000000000000000
            - r / 512
            - 1 / 512
        )
        N5 = (
            lambda r: 2187 * r**7 / 512
            + 729 * r**6 / 512
            - 5589 * r**5 / 512
            - 237304687499999 * r**4 / 100000000000000
            + 901757812500001 * r**3 / 100000000000000
            + 118652343749999 * r**2 / 250000000000000
            - 1215 * r / 512
            + 243 / 512
        )
        N6 = (
            lambda r: 729 * r**7 / 512
            - 243 * r**6 / 512
            - 1539 * r**5 / 512
            + 513 * r**4 / 512
            + 891 * r**3 / 512
            - 290039062500001 * r**2 / 500000000000000
            - 81 * r / 512
            + 131835937500001 / 2500000000000000
        )
        N7 = (
            lambda r: -2187 * r**7 / 512
            + 729 * r**6 / 512
            + 5589 * r**5 / 512
            - 237304687500001 * r**4 / 100000000000000
            - 901757812500001 * r**3 / 100000000000000
            + 118652343750001 * r**2 / 250000000000000
            + 1215 * r / 512
            + 243 / 512
        )
        N8 = (
            lambda r: 729 * r**7 / 512
            + 237304687499999 * r**6 / 500000000000000
            - 300585937500001 * r**5 / 100000000000000
            - 513 * r**4 / 512
            + 891 * r**3 / 512
            + 290039062499999 * r**2 / 500000000000000
            - 81 * r / 512
            - 131835937499999 / 2500000000000000
        )

        N = np.array([N1, N2, N3, N4, N5, N6, N7, N8]).reshape(-1, 1)

        return N

    def _Hermitian_dN(self) -> _types.AnyArray:
        dN1 = [
            lambda r: 6237 * r**6 / 512
            - 2187 * r**5 / 256
            - 275976562500001 * r**4 / 20000000000000
            + 1215 * r**3 / 128
            + 411621093750003 * r**2 / 250000000000000
            - 237304687500001 * r / 250000000000000
            - 292968750000009 / 10000000000000000
        ]
        dN2 = [
            lambda r: 567 * r**6 / 512
            - 243 * r**5 / 256
            - 495 * r**4 / 512
            + 99 * r**3 / 128
            + 556640625000003 * r**2 / 5000000000000000
            - 185546875000001 * r / 2500000000000000
            - 1 / 512
        ]
        dN3 = [
            lambda r: -6237 * r**6 / 512
            - 2187 * r**5 / 256
            + 275976562500001 * r**4 / 20000000000000
            + 1215 * r**3 / 128
            - 823242187500009 * r**2 / 500000000000000
            - 243 * r / 256
            + 292968750000013 / 10000000000000000
        ]
        dN4 = [
            lambda r: 567 * r**6 / 512
            + 243 * r**5 / 256
            - 495 * r**4 / 512
            - 99 * r**3 / 128
            + 278320312500003 * r**2 / 2500000000000000
            + 371093749999999 * r / 5000000000000000
            - 1 / 512
        ]
        dN5 = [
            lambda r: 15309 * r**6 / 512
            + 2187 * r**5 / 256
            - 27945 * r**4 / 512
            - 237304687499999 * r**3 / 25000000000000
            + 2705273437500003 * r**2 / 100000000000000
            + 118652343749999 * r / 125000000000000
            - 1215 / 512
        ]
        dN6 = [
            lambda r: 5103 * r**6 / 512
            - 729 * r**5 / 256
            - 7695 * r**4 / 512
            + 513 * r**3 / 128
            + 2673 * r**2 / 512
            - 290039062500001 * r / 250000000000000
            - 81 / 512
        ]
        dN7 = [
            lambda r: -15309 * r**6 / 512
            + 2187 * r**5 / 256
            + 27945 * r**4 / 512
            - 237304687500001 * r**3 / 25000000000000
            - 2705273437500003 * r**2 / 100000000000000
            + 118652343750001 * r / 125000000000000
            + 1215 / 512
        ]
        dN8 = [
            lambda r: 5103 * r**6 / 512
            + 711914062499997 * r**5 / 250000000000000
            - 300585937500001 * r**4 / 20000000000000
            - 513 * r**3 / 128
            + 2673 * r**2 / 512
            + 290039062499999 * r / 250000000000000
            - 81 / 512
        ]

        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6, dN7, dN8])

        return dN

    def _Hermitian_ddN(self) -> _types.AnyArray:
        ddN1 = [
            lambda r: 18711 * r**5 / 256
            - 10935 * r**4 / 256
            - 275976562500001 * r**3 / 5000000000000
            + 3645 * r**2 / 128
            + 411621093750003 * r / 125000000000000
            - 237304687500001 / 250000000000000
        ]
        ddN2 = [
            lambda r: 1701 * r**5 / 256
            - 1215 * r**4 / 256
            - 495 * r**3 / 128
            + 297 * r**2 / 128
            + 556640625000003 * r / 2500000000000000
            - 185546875000001 / 2500000000000000
        ]
        ddN3 = [
            lambda r: -18711 * r**5 / 256
            - 10935 * r**4 / 256
            + 275976562500001 * r**3 / 5000000000000
            + 3645 * r**2 / 128
            - 823242187500009 * r / 250000000000000
            - 243 / 256
        ]
        ddN4 = [
            lambda r: 1701 * r**5 / 256
            + 1215 * r**4 / 256
            - 495 * r**3 / 128
            - 297 * r**2 / 128
            + 278320312500003 * r / 1250000000000000
            + 371093749999999 / 5000000000000000
        ]
        ddN5 = [
            lambda r: 45927 * r**5 / 256
            + 10935 * r**4 / 256
            - 27945 * r**3 / 128
            - 711914062499997 * r**2 / 25000000000000
            + 2705273437500003 * r / 50000000000000
            + 118652343749999 / 125000000000000
        ]
        ddN6 = [
            lambda r: 15309 * r**5 / 256
            - 3645 * r**4 / 256
            - 7695 * r**3 / 128
            + 1539 * r**2 / 128
            + 2673 * r / 256
            - 290039062500001 / 250000000000000
        ]
        ddN7 = [
            lambda r: -45927 * r**5 / 256
            + 10935 * r**4 / 256
            + 27945 * r**3 / 128
            - 711914062500003 * r**2 / 25000000000000
            - 2705273437500003 * r / 50000000000000
            + 118652343750001 / 125000000000000
        ]
        ddN8 = [
            lambda r: 15309 * r**5 / 256
            + 711914062499997 * r**4 / 50000000000000
            - 300585937500001 * r**3 / 5000000000000
            - 1539 * r**2 / 128
            + 2673 * r / 256
            + 290039062499999 / 250000000000000
        ]

        ddN = np.array([ddN1, ddN2, ddN3, ddN4, ddN5, ddN6, ddN7, ddN8])

        return ddN


class EULER_BERNOULLI5(_EulerBernoulli, _seg.SEG5):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _Hermitian_N(self) -> _types.AnyArray:
        N1 = (
            lambda r: 100 * r**9 / 27
            - 162962962962963 * r**8 / 50000000000000
            - 58 * r**7 / 9
            + 52 * r**6 / 9
            + 91 * r**5 / 36
            - 41 * r**4 / 18
            - 143518518518519 * r**3 / 500000000000000
            + 7 * r**2 / 27
        )
        N2 = (
            lambda r: 2 * r**9 / 9
            - 2 * r**8 / 9
            - r**7 / 3
            + r**6 / 3
            + r**5 / 8
            - r**4 / 8
            - r**3 / 72
            + r**2 / 72
        )
        N3 = (
            lambda r: -100 * r**9 / 27
            - 162962962962963 * r**8 / 50000000000000
            + 58 * r**7 / 9
            + 52 * r**6 / 9
            - 91 * r**5 / 36
            - 41 * r**4 / 18
            + 143518518518519 * r**3 / 500000000000000
            + 7 * r**2 / 27
        )
        N4 = (
            lambda r: 2 * r**9 / 9
            + 2 * r**8 / 9
            - r**7 / 3
            - r**6 / 3
            + r**5 / 8
            + r**4 / 8
            - r**3 / 72
            - r**2 / 72
        )
        N5 = (
            lambda r: 640 * r**9 / 27
            - 128 * r**8 / 27
            - 544 * r**7 / 9
            + 128 * r**6 / 9
            + 448 * r**5 / 9
            - 128 * r**4 / 9
            - 352 * r**3 / 27
            + 128 * r**2 / 27
        )
        N6 = (
            lambda r: 32 * r**9 / 9
            - 16 * r**8 / 9
            - 8 * r**7
            + 4 * r**6
            + 16 * r**5 / 3
            - 8 * r**4 / 3
            - 8 * r**3 / 9
            + 4 * r**2 / 9
        )
        N7 = lambda r: 16 * r**8 - 40 * r**6 + 33 * r**4 - 10 * r**2 + 1
        N8 = lambda r: 8 * r**9 - 20 * r**7 + 33 * r**5 / 2 - 5 * r**3 + r / 2
        N9 = (
            lambda r: -640 * r**9 / 27
            - 128 * r**8 / 27
            + 544 * r**7 / 9
            + 128 * r**6 / 9
            - 448 * r**5 / 9
            - 128 * r**4 / 9
            + 352 * r**3 / 27
            + 128 * r**2 / 27
        )
        N10 = (
            lambda r: 32 * r**9 / 9
            + 16 * r**8 / 9
            - 8 * r**7
            - 4 * r**6
            + 16 * r**5 / 3
            + 8 * r**4 / 3
            - 8 * r**3 / 9
            - 4 * r**2 / 9
        )

        N = np.array([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10]).reshape(-1, 1)

        return N

    def _Hermitian_dN(self) -> _types.AnyArray:
        dN1 = [
            lambda r: 100 * r**8 / 3
            - 162962962962963 * r**7 / 6250000000000
            - 406 * r**6 / 9
            + 104 * r**5 / 3
            + 455 * r**4 / 36
            - 82 * r**3 / 9
            - 430555555555557 * r**2 / 500000000000000
            + 14 * r / 27
        ]
        dN2 = [
            lambda r: 2 * r**8
            - 16 * r**7 / 9
            - 7 * r**6 / 3
            + 2 * r**5
            + 5 * r**4 / 8
            - r**3 / 2
            - r**2 / 24
            + r / 36
        ]
        dN3 = [
            lambda r: -100 * r**8 / 3
            - 162962962962963 * r**7 / 6250000000000
            + 406 * r**6 / 9
            + 104 * r**5 / 3
            - 455 * r**4 / 36
            - 82 * r**3 / 9
            + 430555555555557 * r**2 / 500000000000000
            + 14 * r / 27
        ]
        dN4 = [
            lambda r: 2 * r**8
            + 16 * r**7 / 9
            - 7 * r**6 / 3
            - 2 * r**5
            + 5 * r**4 / 8
            + r**3 / 2
            - r**2 / 24
            - r / 36
        ]
        dN5 = [
            lambda r: 640 * r**8 / 3
            - 1024 * r**7 / 27
            - 3808 * r**6 / 9
            + 256 * r**5 / 3
            + 2240 * r**4 / 9
            - 512 * r**3 / 9
            - 352 * r**2 / 9
            + 256 * r / 27
        ]
        dN6 = [
            lambda r: 32 * r**8
            - 128 * r**7 / 9
            - 56 * r**6
            + 24 * r**5
            + 80 * r**4 / 3
            - 32 * r**3 / 3
            - 8 * r**2 / 3
            + 8 * r / 9
        ]
        dN7 = [lambda r: 128 * r**7 - 240 * r**5 + 132 * r**3 - 20 * r]
        dN8 = [lambda r: 72 * r**8 - 140 * r**6 + 165 * r**4 / 2 - 15 * r**2 + 1 / 2]
        dN9 = [
            lambda r: -640 * r**8 / 3
            - 1024 * r**7 / 27
            + 3808 * r**6 / 9
            + 256 * r**5 / 3
            - 2240 * r**4 / 9
            - 512 * r**3 / 9
            + 352 * r**2 / 9
            + 256 * r / 27
        ]
        dN10 = [
            lambda r: 32 * r**8
            + 128 * r**7 / 9
            - 56 * r**6
            - 24 * r**5
            + 80 * r**4 / 3
            + 32 * r**3 / 3
            - 8 * r**2 / 3
            - 8 * r / 9
        ]

        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6, dN7, dN8, dN9, dN10])

        return dN

    def _Hermitian_ddN(self) -> _types.AnyArray:
        ddN1 = [
            lambda r: 800 * r**7 / 3
            - 1140740740740741 * r**6 / 6250000000000
            - 812 * r**5 / 3
            + 520 * r**4 / 3
            + 455 * r**3 / 9
            - 82 * r**2 / 3
            - 430555555555557 * r / 250000000000000
            + 14 / 27
        ]
        ddN2 = [
            lambda r: 16 * r**7
            - 112 * r**6 / 9
            - 14 * r**5
            + 10 * r**4
            + 5 * r**3 / 2
            - 3 * r**2 / 2
            - r / 12
            + 1 / 36
        ]
        ddN3 = [
            lambda r: -800 * r**7 / 3
            - 1140740740740741 * r**6 / 6250000000000
            + 812 * r**5 / 3
            + 520 * r**4 / 3
            - 455 * r**3 / 9
            - 82 * r**2 / 3
            + 430555555555557 * r / 250000000000000
            + 14 / 27
        ]
        ddN4 = [
            lambda r: 16 * r**7
            + 112 * r**6 / 9
            - 14 * r**5
            - 10 * r**4
            + 5 * r**3 / 2
            + 3 * r**2 / 2
            - r / 12
            - 1 / 36
        ]
        ddN5 = [
            lambda r: 5120 * r**7 / 3
            - 7168 * r**6 / 27
            - 7616 * r**5 / 3
            + 1280 * r**4 / 3
            + 8960 * r**3 / 9
            - 512 * r**2 / 3
            - 704 * r / 9
            + 256 / 27
        ]
        ddN6 = [
            lambda r: 256 * r**7
            - 896 * r**6 / 9
            - 336 * r**5
            + 120 * r**4
            + 320 * r**3 / 3
            - 32 * r**2
            - 16 * r / 3
            + 8 / 9
        ]
        ddN7 = [lambda r: 896 * r**6 - 1200 * r**4 + 396 * r**2 - 20]
        ddN8 = [lambda r: 576 * r**7 - 840 * r**5 + 330 * r**3 - 30 * r]
        ddN9 = [
            lambda r: -5120 * r**7 / 3
            - 7168 * r**6 / 27
            + 7616 * r**5 / 3
            + 1280 * r**4 / 3
            - 8960 * r**3 / 9
            - 512 * r**2 / 3
            + 704 * r / 9
            + 256 / 27
        ]
        ddN10 = [
            lambda r: 256 * r**7
            + 896 * r**6 / 9
            - 336 * r**5
            - 120 * r**4
            + 320 * r**3 / 3
            + 32 * r**2
            - 16 * r / 3
            - 8 / 9
        ]

        ddN = np.array([ddN1, ddN2, ddN3, ddN4, ddN5, ddN6, ddN7, ddN8, ddN9, ddN10])

        return ddN


# --------------------------------------------
# Timoshenko elements
# --------------------------------------------


class TIMOSHENKO2(_Timoshenko, EULER_BERNOULLI2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TIMOSHENKO3(_Timoshenko, EULER_BERNOULLI3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TIMOSHENKO4(_Timoshenko, EULER_BERNOULLI4):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TIMOSHENKO5(_Timoshenko, EULER_BERNOULLI5):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# --------------------------------------------
# Beam element factories
# --------------------------------------------


BEAM_CLASS_MAP = {
    ElemType.SEG2: EULER_BERNOULLI2,
    ElemType.SEG3: EULER_BERNOULLI3,
    ElemType.SEG4: EULER_BERNOULLI4,
    ElemType.SEG5: EULER_BERNOULLI5,
}

_TIMO_CLASS_MAP = {
    ElemType.SEG2: TIMOSHENKO2,
    ElemType.SEG3: TIMOSHENKO3,
    ElemType.SEG4: TIMOSHENKO4,
    ElemType.SEG5: TIMOSHENKO5,
}


def __Construct_beam_mesh(mesh: Mesh, class_map: dict) -> Mesh:
    """Replace 1D SEG elements with beam elements from *class_map*."""

    coordinates = mesh.coord
    newDict_groupElem: dict[ElemType, _GroupElem] = {}

    for elemType, groupElem in mesh.dict_groupElem.items():

        if groupElem.dim != 1:
            newDict_groupElem[elemType] = groupElem
            continue

        try:
            BeamClass = class_map[elemType]
        except KeyError:
            raise NotImplementedError(f"Beam not implemented for {elemType}")

        newGroup: _GroupElem = BeamClass(
            gmshId=groupElem.gmshId,
            connect=groupElem.connect,
            coordinates=coordinates,
        )

        for nodeTag in groupElem.nodeTags:
            newGroup.Set_Tag(groupElem.Get_Nodes_Tag(nodeTag), nodeTag)

        newDict_groupElem[elemType] = newGroup

    return Mesh(newDict_groupElem)


def _Construct_Euler_Bernoulli_mesh(mesh: Mesh) -> Mesh:
    """Update SEG elements with Euler-Bernoulli beam elements."""
    return __Construct_beam_mesh(mesh, BEAM_CLASS_MAP)


def _Construct_Timoshenko_mesh(mesh: Mesh) -> Mesh:
    """Update SEG elements with Timoshenko beam elements."""
    return __Construct_beam_mesh(mesh, _TIMO_CLASS_MAP)
