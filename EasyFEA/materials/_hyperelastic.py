# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Hyper elastic module used to compute matrices."""

import numpy as np

from ..fem import Mesh, MatrixType
from ..utilities._linalg import Transpose, Trace, Det, Inv, TensorProd

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
        """Computes the deformation gradient F(u) = I + grad(u)
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi
        
        Returns
        -------
        np.ndarray
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
    
    def Compute_J(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes the deformation gradient J = det(F)
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        np.ndarray
            J_e_pg of shape (Ne, pg)
        """

        F_e_pg = HyperElastic.Compute_F(mesh, u, matrixType)

        J_e_pg = Det(F_e_pg)

        return J_e_pg

    def Compute_C(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes the right Cauchy-Green tensor  C(u) = F(u)'.F(u)
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        np.ndarray
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
    
    def Compute_e(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes the Green-Lagrange deformation  e = 1/2 (C - I)
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        np.ndarray
            e_e_pg of shape (Ne, pg, dim, dim)
        """

        C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)

        e_e_pg = 1/2 * (C_e_pg - np.eye(3)) 

        return e_e_pg
       
    def Compute_Epsilon(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes the linearized deformation Epsilon = 1/2 (grad(u)' + grad(u))
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns if dim = 2
        ------------------
        np.ndarray
            Eps_e_pg of shape (Ne, pg, 3)

            [xx, yy, 2**(-1/2) xy]

        Returns if dim = 3
        ------------------
        np.ndarray
            Eps_e_pg of shape (Ne, pg, 6)

            [xx, yy, zz, 2**(-1/2) yz, 2**(-1/2) xz, 2**(-1/2) xy]
        """

        HyperElastic.__CheckFormat(mesh, u, matrixType)
        Ne, nPg, dim = HyperElastic.__GetDims(mesh, u, matrixType)
        assert dim in [2, 3]

        # compute grad
        grad_e_pg = mesh.Get_Gradient_e_pg(u, matrixType)[Ellipsis,:dim, :dim]

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

        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns if dim = 2
        ------------------
        np.ndarray
            D_e_pg of shape (Ne, pg, 3, 4)

            [1+dxux, 0, dxuy, 0] # xx \n
            [0, dyux, 0, 1+dyuy] # yy \n
            2**(-1/2) [dyux, 1+dxux, 1+dyuy, dxuy # xy

        Returns if dim = 3
        ------------------
        np.ndarray
            D_e_pg of shape (Ne, pg, 6, 9)

            [1+dxux, 0, 0, dxuy, 0, 0, dxuz, 0, 0] # xx \n
            [0, dyux, 0, 0, 1+dyuy, 0, 0, dyuz, 0] # yy \n
            [0, 0, dzux, 0, 0, dzuy, 0, 0, 1+dzuz] # zz \n
            2**(-1/2) [dzux, 0, 1 + dxux, dzuy, 0, dxuy, 1 + dzuz, 0, dxuz] # yz \n
            2**(-1/2) [0, dzux, dyux, 0, dzuy, 1 + dyuy, 0, 1 + dzuz, dyuz] # xz \n
            2**(-1/2) [dyux, 1+dxux, 0, 1+dyuy, dxuy, 0, dyuz, dxuz, 0] # xy
        """

        HyperElastic.__CheckFormat(mesh, u, matrixType)
        Ne, nPg, dim = HyperElastic.__GetDims(mesh, u, matrixType)
        assert dim in [2, 3]

        grad_e_pg = mesh.Get_Gradient_e_pg(u)

        if dim == 2:
            D_e_pg = np.zeros((Ne, nPg, 3, 4), dtype=float)
        else:
            D_e_pg = np.zeros((Ne, nPg, 6, 9), dtype=float)

        def Add_to_D_e_pg(p: int, line: int, values: list[np.ndarray], coef=1.):
            N = 4 if dim == 2 else 9            
            for column in range(N):
                D_e_pg[:,p,line,column] = values[column] * coef

        cM = 2**(-1/2)

        for p in range(nPg):

            if dim == 2:
                dxux, dyux = [grad_e_pg[:, p, 0, i] for i in range(2)]
                dxuy, dyuy = [grad_e_pg[:, p, 1, i] for i in range(2)]

                Add_to_D_e_pg(p, 0, [1+dxux, 0, dxuy, 0]) # xx
                Add_to_D_e_pg(p, 1, [0, dyux, 0, 1+dyuy]) # yy
                Add_to_D_e_pg(p, 2, [dyux, 1+dxux, 1+dyuy, dxuy], cM) # xy
            
            else:

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
        """Computes I1(u)
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        np.ndarray
            I1_e_pg of shape (Ne, pg)         
        """

        C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)

        I1_e_pg = Trace(C_e_pg)

        return I1_e_pg

    def Compute_dI1dC(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes dI1dC(u)
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        np.ndarray
            dI1dC_e_pg of shape (Ne, pg, 3, 3)
        """

        Ne, nPg, dim = HyperElastic.__GetDims(mesh, u, matrixType)

        dI1dC_e_pg = np.zeros((Ne, nPg, 3, 3), dtype=float)

        for d in range(dim):
            dI1dC_e_pg[:,:,d,d] = 1

        return dI1dC_e_pg

    # -------------------------------------
    # Compute I2
    # -------------------------------------
    def Compute_I2(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes I2(u)
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        np.ndarray
            I2_e_pg of shape (Ne, pg)
        """

        C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)

        I2_e_pg = 1/2 * (Trace(C_e_pg)**2 - Trace(C_e_pg @ C_e_pg))

        return I2_e_pg
    
    def Compute_dI2dC(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes dI2dC(u)
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        np.ndarray
            dI2dC_e_pg of shape (Ne, pg, 3, 3)
        """

        _, _, dim = HyperElastic.__GetDims(mesh, u, matrixType)

        I1_e_pg = HyperElastic.Compute_I1(mesh, u, matrixType)
        C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)

        dI2dC_e_pg = I1_e_pg @ np.eye(dim) - C_e_pg 

        return dI2dC_e_pg

    # -------------------------------------
    # Compute I3
    # -------------------------------------
    def Compute_I3(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes I3(u)
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        np.ndarray
            I3_e_pg of shape (Ne, pg)
        """

        C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)

        I3_e_pg = Det(C_e_pg)

        return I3_e_pg
    
    def Compute_dI3dC(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes dI3dC(u)
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        np.ndarray
            dI3dC_e_pg of shape (Ne, pg, 3, 3)
        """

        I3_e_pg = HyperElastic.Compute_I3(mesh, u, matrixType)
        C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)

        dI3dC_e_pg = I3_e_pg @ Inv(C_e_pg)

        return dI3dC_e_pg

    # -------------------------------------
    # Compute I4
    # -------------------------------------
    def Compute_I4(mesh: Mesh, u: np.ndarray, T: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes I4(u)
            
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        np.ndarray
            I4_e_pg of shape (Ne, pg)
        """

        C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)

        assert isinstance(T, np.ndarray) and T.shape[-1] == 3, "T must be a (..., 3) array"

        I4_e_pg = T @ (C_e_pg @ T)

        return I4_e_pg
    
    def Compute_dI4dC(T: np.ndarray) -> np.ndarray:
        """Computes dI4dC(u)
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        np.ndarray
            dI4dC_e_pg of shape (Ne, pg, 3, 3)
        """

        assert isinstance(T, np.ndarray) and T.shape[-1] == 3, "T must be a (..., 3) array"

        dI4dC_e_pg = TensorProd(T, T)

        return dI4dC_e_pg

    # -------------------------------------
    # Compute J1
    # -------------------------------------
    def Compute_J1(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes J1(u)
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        np.ndarray
            J1_e_pg of shape (Ne, pg)
        """

        I1_e_pg = HyperElastic.Compute_I1(mesh, u, matrixType)
        I3_e_pg = HyperElastic.Compute_I3(mesh, u, matrixType)

        J1_e_pg = I1_e_pg * I3_e_pg**(-1/3)

        return J1_e_pg
    
    def Compute_dJ1dC(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes dJ1dC(u)
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        np.ndarray
            dJ1dC_e_pg of shape (Ne, pg, 3, 3)
        """

        I1_e_pg = HyperElastic.Compute_I1(mesh, u, matrixType)
        I3_e_pg = HyperElastic.Compute_I3(mesh, u, matrixType)
        C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)        
        I = np.eye(3)

        dJ1dC_e_pg = I3_e_pg**(-1/3) * (I - 1/3 * I1_e_pg * Inv(C_e_pg))

        return dJ1dC_e_pg

    # -------------------------------------
    # Compute J2
    # -------------------------------------
    def Compute_J2(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes J2(u)
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        np.ndarray
            J2_e_pg of shape (Ne, pg)
        """

        I2_e_pg = HyperElastic.Compute_I2(mesh, u, matrixType)
        I3_e_pg = HyperElastic.Compute_I3(mesh, u, matrixType)

        J2_e_pg = I2_e_pg * I3_e_pg**(-2/3)

        return J2_e_pg
    
    def Compute_dJ2dC(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes dJ2dC(u)
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        np.ndarray
            dJ2dC_e_pg of shape (Ne, pg, 3, 3)
        """

        I1_e_pg = HyperElastic.Compute_I1(mesh, u, matrixType)
        I2_e_pg = HyperElastic.Compute_I2(mesh, u, matrixType)
        I3_e_pg = HyperElastic.Compute_I3(mesh, u, matrixType)
        C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)
        I = np.eye(3)

        dJ2dC_e_pg = I3_e_pg**(-2/3) * (I1_e_pg @ I - C_e_pg - 2/3 * I2_e_pg * Inv(C_e_pg))

        return dJ2dC_e_pg

    # -------------------------------------
    # Compute J3 = J
    # -------------------------------------
    def Compute_J3(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes J3(u)
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        np.ndarray
            J3_e_pg of shape (Ne, pg)
        """

        J_e_pg = HyperElastic.Compute_J(mesh, u, matrixType) #  J3 = I3**(1/2) = J

        return J_e_pg
    
    def Compute_dJ3dC(mesh: Mesh, u: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes dJ2dC(u)
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        np.ndarray
            dJ2dC_e_pg of shape (Ne, pg, 3, 3)
        """

        J_e_pg = HyperElastic.Compute_J(mesh, u, matrixType) #  J3 = I3**(1/2) = J
        C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)        

        dJ3dC_e_pg = 1/2 * J_e_pg**(1/2) * Inv(C_e_pg)

        return dJ3dC_e_pg

    # -------------------------------------
    # Compute J4
    # -------------------------------------
    def Compute_J4(mesh: Mesh, u: np.ndarray, T: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes J4(u)
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        np.ndarray
            J4_e_pg of shape (Ne, pg)
        """

        I3_e_pg = HyperElastic.Compute_I3(mesh, u, matrixType)
        I4_e_pg = HyperElastic.Compute_I4(mesh, u, T, matrixType)

        J4_e_pg = I4_e_pg * I3_e_pg**(-1/3)

        return J4_e_pg
    
    def Compute_dJ4dC(mesh: Mesh, u: np.ndarray, T: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Computes dJ4dC(u)
        
        Parameters
        ----------
        mesh : Mesh
            mesh
        u : np.ndarray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        np.ndarray
            dJ4dC_e_pg of shape (Ne, pg, 3, 3)
        """

        I3_e_pg = HyperElastic.Compute_I3(mesh, u, matrixType)
        I4_e_pg = HyperElastic.Compute_I4(mesh, u, T, matrixType)
        C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)        

        dJ4dC_e_pg = I3_e_pg**(-1/3) * (TensorProd(T, T) - 1/3 * I4_e_pg * Inv(C_e_pg))

        return dJ4dC_e_pg