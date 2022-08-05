import numba
from numba import njit, prange
import numpy as np

@njit(cache=False, parallel=True)
def Get_Anisot_C(projP_e_pg: np.ndarray, c: np.ndarray, projM_e_pg: np.ndarray):

    Ne = projP_e_pg.shape[0]
    nPg = projP_e_pg.shape[1]
    dimc = c.shape[0]

    Cpp_e_pg = np.zeros((Ne, nPg, dimc, dimc))
    Cpm_e_pg = np.zeros_like(Cpp_e_pg)
    Cmp_e_pg = np.zeros_like(Cpp_e_pg)
    Cmm_e_pg = np.zeros_like(Cpp_e_pg)

    for e in prange(projP_e_pg.shape[0]):
        for p in prange(projP_e_pg.shape[1]):
            for i in prange(c.shape[0]):                
                for j in prange(c.shape[0]):
                    for l in prange(c.shape[0]):
                        for k in prange(c.shape[0]):

                            Cpp_e_pg[e,p,i,j] += projP_e_pg[e,p,k,i] * c[k,l] * projP_e_pg[e,p,l,j]
                            Cpm_e_pg[e,p,i,j] += projP_e_pg[e,p,k,i] * c[k,l] * projM_e_pg[e,p,l,j]
                            Cmp_e_pg[e,p,i,j] += projM_e_pg[e,p,k,i] * c[k,l] * projP_e_pg[e,p,l,j]
                            Cmm_e_pg[e,p,i,j] += projM_e_pg[e,p,k,i] * c[k,l] * projM_e_pg[e,p,l,j]
    
    return Cpp_e_pg, Cpm_e_pg, Cmp_e_pg, Cmm_e_pg

@njit(cache=False, parallel=True)
def ep_ij_to_epij(ep: np.ndarray, ij: np.ndarray):
    result = np.zeros((ep.shape[0], ep.shape[1], ij.shape[0], ij.shape[1]))
    for e in prange(ep.shape[0]):
        for p in prange(ep.shape[1]):
            result[e,p] = ep[e,p] * ij
    return result

@njit(cache=False, parallel=True)
def Split_Amor(Rp_e_pg: np.ndarray, Rm_e_pg: np.ndarray,
partieDeviateur: np.ndarray, IxI: np.ndarray, bulk):
    Ne = Rp_e_pg.shape[0]
    pg = Rp_e_pg.shape[1]
    dim = IxI.shape[0]

    cP_e_pg = np.zeros((Ne, pg, dim, dim))
    cM_e_pg = np.zeros((Ne, pg, dim, dim))

    for e in prange(Ne):
        for p in prange(pg):
            cP_e_pg[e,p] = bulk*(Rp_e_pg[e,p] * IxI) + partieDeviateur
            cM_e_pg[e,p] = bulk*(Rm_e_pg[e,p] * IxI) + partieDeviateur

    return cP_e_pg, cM_e_pg
