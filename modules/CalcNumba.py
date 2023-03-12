import numba
from numba import njit, prange, jit
import numpy as np

useCache = True
useParallel = True
useFastmath = True

# Calcul de splits

@njit(cache=useCache, parallel=useParallel, fastmath=useFastmath)
def Get_Anisot_C(aP_e_pg: np.ndarray, b: np.ndarray, aM_e_pg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Permet de calculer les 4 matrices de comportement Cpp, Cpm, Cmp et Cmm\n
    Permet d'executer cette opération :\n
    projP_e_pg.T x C x projM_e_pg  (epki, kl, eplj) -> (epij) \n

    ou cette opération :\n
    cP_e_pg.T x S x cM_e_pg  (epki, kl, eplj) -> (epij)

    Parameters
    ----------
    aP_e_pg : np.ndarray
        projP_e_pg ou cP_e_pg (epij)
    b : np.ndarray
        _description_
    aM_e_pg : np.ndarray
        projM_e_pg ou cM_e_pg (epij)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Cpp_e_pg, Cpm_e_pg, Cmp_e_pg, Cmm_e_pg
    """
    if useParallel:
        range = prange
    else:
        range = np.arange

    Ne = aP_e_pg.shape[0]
    nPg = aP_e_pg.shape[1]
    dimc = b.shape[0]

    Cpp_e_pg = np.zeros((Ne, nPg, dimc, dimc))
    Cpm_e_pg = np.zeros_like(Cpp_e_pg)
    Cmp_e_pg = np.zeros_like(Cpp_e_pg)
    Cmm_e_pg = np.zeros_like(Cpp_e_pg)

    for e in range(aP_e_pg.shape[0]):
        for p in range(aP_e_pg.shape[1]):
            for i in range(b.shape[0]):                
                for j in range(b.shape[0]):
                    for l in range(b.shape[0]):
                        for k in range(b.shape[0]):

                            Cpp_e_pg[e,p,i,j] += aP_e_pg[e,p,k,i] * b[k,l] * aP_e_pg[e,p,l,j]
                            Cpm_e_pg[e,p,i,j] += aP_e_pg[e,p,k,i] * b[k,l] * aM_e_pg[e,p,l,j]
                            Cmp_e_pg[e,p,i,j] += aM_e_pg[e,p,k,i] * b[k,l] * aP_e_pg[e,p,l,j]
                            Cmm_e_pg[e,p,i,j] += aM_e_pg[e,p,k,i] * b[k,l] * aM_e_pg[e,p,l,j]
    
    return Cpp_e_pg, Cpm_e_pg, Cmp_e_pg, Cmm_e_pg


# Pas tres efficace
@njit(cache=useCache, parallel=useParallel, fastmath=useFastmath)
def Split_Amor(Rp_e_pg: np.ndarray, Rm_e_pg: np.ndarray,
partieDeviateur: np.ndarray, IxI: np.ndarray, bulk) -> tuple[np.ndarray, np.ndarray]:
    if useParallel:
        range = prange
    else:
        range = np.arange

    Ne = Rp_e_pg.shape[0]
    pg = Rp_e_pg.shape[1]
    dim = IxI.shape[0]

    cP_e_pg = np.zeros((Ne, pg, dim, dim))
    cM_e_pg = np.zeros((Ne, pg, dim, dim))

    for e in range(Ne):
        for p in range(pg):
            cP_e_pg[e,p] = bulk*(Rp_e_pg[e,p] * IxI) + partieDeviateur
            cM_e_pg[e,p] = bulk*(Rm_e_pg[e,p] * IxI) + partieDeviateur

    return cP_e_pg, cM_e_pg

@njit(cache=useCache, parallel=useParallel, fastmath=useFastmath)
def Get_G12_G13_G23(M1: np.ndarray, M2: np.ndarray, M3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if useParallel:
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

    return G12_ijkl, G13_ijkl, G23_ijkl

@njit(cache=useCache, parallel=useParallel, fastmath=useFastmath)
def Get_projP_projM_2D(BetaP: np.ndarray, gammap: np.ndarray, BetaM: np.ndarray, gammam: np.ndarray,m1: np.ndarray, m2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    if useParallel:
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

@njit(cache=useCache, parallel=useParallel, fastmath=useFastmath)
def Get_projP_projM_3D(dvalp: np.ndarray, dvalm: np.ndarray, thetap: np.ndarray, thetam: np.ndarray, list_mi: list[np.ndarray], list_Gab: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:

    if useParallel:
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

@njit(cache=useCache, parallel=useParallel, fastmath=useFastmath)
def Get_Cp_Cm_Stress(c: np.ndarray, sP_e_pg: np.ndarray, sM_e_pg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    if useParallel:
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
    