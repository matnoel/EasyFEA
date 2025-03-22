# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Numba functions to speed up calculations."""

import numpy as np
from numba import njit, prange, jit

__USE_CACHE = True
__USE_PARALLEL = True
__USE_FASTMATH = False

@njit(cache=__USE_CACHE, parallel=__USE_PARALLEL, fastmath=__USE_FASTMATH)
def Get_Anisot_C(Cp_e_pg: np.ndarray, mat: np.ndarray, Cm_e_pg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    # WARNING: There may be a memory problem if mat is heterogeneous (epij). That's why mat is not heterogeneous.
    
    if __USE_PARALLEL:
        range = prange
    else:
        range = np.arange

    Ne = Cp_e_pg.shape[0]
    nPg = Cp_e_pg.shape[1]
    dimMat = mat.shape[0]

    Cpp_e_pg = np.zeros((Ne, nPg, dimMat, dimMat))
    Cpm_e_pg = np.zeros_like(Cpp_e_pg)
    Cmp_e_pg = np.zeros_like(Cpp_e_pg)
    Cmm_e_pg = np.zeros_like(Cpp_e_pg)

    for e in range(Cp_e_pg.shape[0]):
        for p in range(Cp_e_pg.shape[1]):
            for i in range(dimMat):                
                for j in range(dimMat):
                    for l in range(dimMat):
                        for k in range(dimMat):

                            Cpp_e_pg[e,p,i,j] += Cp_e_pg[e,p,k,i] * mat[k,l] * Cp_e_pg[e,p,l,j]
                            Cpm_e_pg[e,p,i,j] += Cp_e_pg[e,p,k,i] * mat[k,l] * Cm_e_pg[e,p,l,j]
                            Cmp_e_pg[e,p,i,j] += Cm_e_pg[e,p,k,i] * mat[k,l] * Cp_e_pg[e,p,l,j]
                            Cmm_e_pg[e,p,i,j] += Cm_e_pg[e,p,k,i] * mat[k,l] * Cm_e_pg[e,p,l,j]
    
    return Cpp_e_pg, Cpm_e_pg, Cmp_e_pg, Cmm_e_pg

@njit(cache=__USE_CACHE, parallel=__USE_PARALLEL, fastmath=__USE_FASTMATH)
def Get_G12_G13_G23(M1: np.ndarray, M2: np.ndarray, M3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if __USE_PARALLEL:
        range = prange
    else:
        range = np.arange

    matriceI = np.eye(3)

    Ne = M1.shape[0]
    nPg = M1.shape[1]

    coef = np.sqrt(2)

    G12_ijkl = np.zeros((Ne, nPg, 3, 3, 3, 3))
    G13_ijkl = np.zeros_like(G12_ijkl)
    G23_ijkl = np.zeros_like(G12_ijkl)

    for e in range(Ne):
        for p in range(nPg):
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):

                            g12 = (M1[e,p,i,k] * M2[e,p,j,l]) + (M1[e,p,i,l] * M2[e,p,j,k]) + (M2[e,p,i,k] * M1[e,p,j,l]) + (M2[e,p,i,l] * M1[e,p,j,k])

                            g13 = (M1[e,p,i,k] * M3[e,p,j,l]) + (M1[e,p,i,l] * M3[e,p,j,k]) + (M3[e,p,i,k] * M1[e,p,j,l]) + (M3[e,p,i,l] * M1[e,p,j,k])

                            g23 = (M2[e,p,i,k] * M3[e,p,j,l]) + (M2[e,p,i,l] * M3[e,p,j,k]) + (M3[e,p,i,k] * M2[e,p,j,l]) + (M3[e,p,i,l] * M2[e,p,j,k])

                            G12_ijkl[e,p,i,j,k,l] = g12
                            G13_ijkl[e,p,i,j,k,l] = g13
                            G23_ijkl[e,p,i,j,k,l] = g23

    G12_ij = np.zeros((Ne, nPg, 6, 6))
    G13_ij = np.zeros_like(G12_ij)
    G23_ij = np.zeros_like(G12_ij)

    listI = np.array([0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0])
    listJ = np.array([0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1])
    listK = np.array([0,1,2,1,0,0,0,1,2,1,0,0,0,1,2,1,0,0,0,1,2,1,0,0,0,1,2,1,0,0,0,1,2,1,0,0])
    listL = np.array([0,1,2,2,2,1,0,1,2,2,2,1,0,1,2,2,2,1,0,1,2,2,2,1,0,1,2,2,2,1,0,1,2,2,2,1])

    colonnes = np.array([0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5])
    lignes = np.array([0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5])    

    for c in range(36):
        G12_ij[:,:,lignes[c], colonnes[c]] = G12_ijkl[:,:,listI[c],listJ[c],listK[c],listL[c]]
        G13_ij[:,:,lignes[c], colonnes[c]] = G13_ijkl[:,:,listI[c],listJ[c],listK[c],listL[c]]
        G23_ij[:,:,lignes[c], colonnes[c]] = G23_ijkl[:,:,listI[c],listJ[c],listK[c],listL[c]]    

    l03 = np.array([0,0,0,1,1,1,2,2,2])
    l36 = np.array([3,3,3,4,4,4,5,5,5])
    c03 = np.array([0,1,2,0,1,2,0,1,2])
    c36 = np.array([3,4,5,3,4,5,3,4,5])

    coef = np.sqrt(2)

    for c in range(9):

        G12_ij[:,:,l03[c],c36[c]] *= coef
        G12_ij[:,:,l36[c],c03[c]] *= coef
        G12_ij[:,:,l36[c],c36[c]] *= 2

        G13_ij[:,:,l03[c],c36[c]] *= coef
        G13_ij[:,:,l36[c],c03[c]] *= coef
        G13_ij[:,:,l36[c],c36[c]] *= 2

        G23_ij[:,:,l03[c],c36[c]] *= coef
        G23_ij[:,:,l36[c],c03[c]] *= coef
        G23_ij[:,:,l36[c],c36[c]] *= 2

    return G12_ij, G13_ij, G23_ij

@njit(cache=__USE_CACHE, parallel=__USE_PARALLEL, fastmath=__USE_FASTMATH)
def Get_projP_projM_2D(BetaP: np.ndarray, gammap: np.ndarray, BetaM: np.ndarray, gammam: np.ndarray,m1: np.ndarray, m2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    if __USE_PARALLEL:
        range = prange
    else:
        range = np.arange

    matriceI = np.eye(3)

    Ne = BetaP.shape[0]
    nPg = BetaP.shape[1]
    dimI = matriceI.shape[0]
    dimJ = matriceI.shape[1]

    projP = np.zeros((Ne, nPg, dimI, dimJ))
    projM = np.zeros((Ne, nPg, dimI, dimJ))

    for e in range(Ne):
        for p in range(nPg):
            for i in range(dimI):
                for j in range(dimJ):
                    m1xm1 = m1[e,p,i] * m1[e,p,j]
                    m2xm2 = m2[e,p,i] * m2[e,p,j]

                    projP[e,p,i,j] = BetaP[e,p] * matriceI[i,j] + gammap[e,p,0] * m1xm1 + gammap[e,p,1] * m2xm2
                    
                    projM[e,p,i,j] = BetaM[e,p] * matriceI[i,j] + gammam[e,p,0] * m1xm1 + gammam[e,p,1] * m2xm2

    return projP, projM

@njit(cache=__USE_CACHE, parallel=__USE_PARALLEL, fastmath=__USE_FASTMATH)
def Get_projP_projM_3D(dvalp: np.ndarray, dvalm: np.ndarray, thetap: np.ndarray, thetam: np.ndarray, list_mi: list[np.ndarray], list_Gab: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:

    # ../FEMOBJECT/BASIC/MODEL/MATERIALS/@ELAS_ISOT/calc_proj_Miehe.m

    if __USE_PARALLEL:
        range = prange
    else:
        range = np.arange

    m1, m2, m3 = list_mi[0], list_mi[1], list_mi[2]

    G12_ij, G13_ij, G23_ij = list_Gab[0], list_Gab[1], list_Gab[2]

    Ne = dvalp.shape[0]
    nPg = dvalp.shape[1]    

    projP = np.zeros((Ne, nPg, 6, 6))
    projM = np.zeros((Ne, nPg, 6, 6))

    for e in range(Ne):
        for p in range(nPg):
            for i in range(6):
                for j in range(6):
                    m1xm1 = m1[e,p,i] * m1[e,p,j]
                    m2xm2 = m2[e,p,i] * m2[e,p,j]
                    m3xm3 = m3[e,p,i] * m3[e,p,j]                    

                    projP[e,p,i,j] = dvalp[e,p,0]*m1xm1 + dvalp[e,p,1]*m2xm2 + dvalp[e,p,2]*m3xm3 + thetap[e,p,0]*G12_ij[e,p,i,j] + thetap[e,p,1]*G13_ij[e,p,i,j] + thetap[e,p,2]*G23_ij[e,p,i,j]
                    
                    projM[e,p,i,j] = dvalm[e,p,0]*m1xm1 + dvalm[e,p,1]*m2xm2 + dvalm[e,p,2]*m3xm3 + thetam[e,p,0]*G12_ij[e,p,i,j] + thetam[e,p,1]*G13_ij[e,p,i,j] + thetam[e,p,2]*G23_ij[e,p,i,j]

    return projP, projM

@njit(cache=__USE_CACHE, parallel=__USE_PARALLEL, fastmath=__USE_FASTMATH)
def Get_Cp_Cm_Stress(c: np.ndarray, sP_e_pg: np.ndarray, sM_e_pg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    if __USE_PARALLEL:
        range = prange
    else:
        range = np.arange

    Ne = sP_e_pg.shape[0]
    nPg = sP_e_pg.shape[1]
    dim = c.shape[0]

    cP_e_pg = np.zeros((Ne, nPg, dim, dim))
    cM_e_pg = np.zeros((Ne, nPg, dim, dim))

    for e in range(Ne):
        for p in range(nPg):
            for i in range(dim):
                for j in range(dim):
                    for k in range(dim):
                        for l in range(dim):
                            cP_e_pg[e,p,i,l] += c[j,i] * sP_e_pg[e,p,j,k] * c[k,l]
                            cM_e_pg[e,p,i,l] += c[j,i] * sM_e_pg[e,p,j,k] * c[k,l]

    return cP_e_pg, cM_e_pg