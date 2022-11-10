import numba
from numba import njit, prange
import numpy as np

useCache = True
useParallel = True
useFastmath = True



def CompilNumba(verbosity=True):
    """Fonction qui compile toutes les fonctions disponibles"""

    import TicTac

    tic = TicTac.Tic()

    indices2 = np.ones((1,1))
    indices3 = np.ones((1,1,1))
    indices4 = np.ones((1,1,1,1))

    # ep_ij_to_epij(indices2, indices2)
    # epij_ej_to_epi(indices4, indices2)
    # ij_epj_to_epi(indices2, indices3)
    # ep_epi_to_epi(indices2, indices3)
    # ep_epij_to_epij(indices2, indices4)
    # epij_epjk_epkl_to_eil(indices4, indices4, indices4)
    # epij_jk_epkl_to_eil(indices4, indices2, indices4)
    # Split_Amor(indices2, indices2, indices2, indices2, 1)
    # Get_Anisot_C(indices4, indices2, indices4)
    Calc_psi_e_pg(indices3, indices3, indices3)
    Calc_Sigma_e_pg(indices3, indices4, indices4)
    Construit_Kd_e_and_Fd_e(indices2, indices4, 1, indices4, indices2, indices4)

    print(numba.get_num_threads())

    tic.Tac("Numba", "Compilation des fonctions", verbosity)

#Calcul indiciel

@njit(cache=useCache, parallel=useParallel, fastmath=useFastmath)
def ep_ij_to_epij(ep: np.ndarray, ij: np.ndarray) -> np.ndarray:
    if useParallel:
        range = prange
    else:
        range = np.arange

    result = np.zeros((ep.shape[0], ep.shape[1], ij.shape[0], ij.shape[1]))
    for e in range(ep.shape[0]):
        for p in range(ep.shape[1]):
            result[e,p] = ep[e, p] * ij
    return result

@njit(cache=useCache, parallel=useParallel, fastmath=useFastmath)
def epij_ej_to_epi(epij: np.ndarray, ej: np.ndarray) -> np.ndarray:
    if useParallel:
        range = prange
    else:
        range = np.arange
    
    Ne = epij.shape[0]
    nPg = epij.shape[1]
    dimI = epij.shape[2]
    dimJ = epij.shape[3]

    assert Ne == ej.shape[0], "Mauvaise dimension"
    assert dimJ == ej.shape[1], "Mauvaise dimension"

    result = np.zeros((Ne, nPg, dimI))

    for e in range(Ne):
        for p in range(nPg):
            for i in range(dimI):
                for j in range(dimJ):
                    result[e, p, i] += epij[e, p, i, j] * ej[e, j]

    return result

@njit(cache=useCache, parallel=useParallel, fastmath=useFastmath)
def Calc_m1xm1_m2xm2(m1: np.ndarray, m2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if useParallel:
        range = prange
    else:
        range = np.arange
    
    Ne = m1.shape[0]
    nPg = m1.shape[1]
    dimI = m1.shape[2]
    dimJ = m1.shape[2]

    m1xm1 = np.zeros((Ne, nPg, dimI, dimI))
    m2xm2 = np.zeros((Ne, nPg, dimI, dimI))

    for e in range(Ne):
        for p in range(nPg):
            for i in range(dimI):
                for j in range(dimJ):
                    m1xm1[e, p, i, j] += m1[e, p, i] * m1[e, p, j]
                    m2xm2[e, p, i, j] += m2[e, p, i] * m2[e, p, j]

    return m1xm1, m2xm2

@njit(cache=useCache, parallel=useParallel, fastmath=useFastmath)
def ij_epj_to_epi(ij: np.ndarray, epj: np.ndarray) -> np.ndarray:
    if useParallel:
        range = prange
    else:
        range = np.arange
    
    Ne = epj.shape[0]
    nPg = epj.shape[1]
    dimI = ij.shape[0]
    dimJ = ij.shape[1]

    assert dimJ == epj.shape[2], "Mauvaise dimension"

    result = np.zeros((Ne, nPg, dimI))

    for e in range(Ne):
        for p in range(nPg):
            for i in range(dimI):
                for j in range(dimJ):
                    result[e, p, i] += ij[i, j] * epj[e, p, j]

    return result

@njit(cache=useCache, parallel=useParallel, fastmath=useFastmath)
def ep_epi_to_epi(ep: np.ndarray, epi: np.ndarray) -> np.ndarray:
    if useParallel:
        range = prange
    else:
        range = np.arange
    
    Ne = ep.shape[0]
    nPg = ep.shape[1]
    dimI = epi.shape[2]

    assert Ne == epi.shape[0], "Mauvaise dimension"
    assert nPg == epi.shape[1], "Mauvaise dimension"

    result = np.zeros((Ne, nPg, dimI))

    for e in range(Ne):
        for p in range(nPg):
            for i in range(dimI):
                result[e, p, i] += ep[e, p] * epi[e, p, i]

    return result

@njit(cache=useCache, parallel=useParallel, fastmath=useFastmath)
def ep_epij_to_epij(ep: np.ndarray, epij: np.ndarray) -> np.ndarray:
    if useParallel:
        range = prange
    else:
        range = np.arange
    
    Ne = ep.shape[0]
    nPg = ep.shape[1]
    dimI = epij.shape[2]
    dimJ = epij.shape[3]

    assert Ne == epij.shape[0], "Mauvaise dimension"
    assert nPg == epij.shape[1], "Mauvaise dimension"

    result = np.zeros((Ne, nPg, dimI, dimJ))

    for e in range(Ne):
        for p in range(nPg):
            result[e, p, :, :] += ep[e, p] * epij[e, p, :, :]
                    

    return result

@njit(cache=useCache, parallel=useParallel, fastmath=useFastmath)
def ep_epij_to_eij(ep: np.ndarray, epij: np.ndarray) -> np.ndarray:
    if useParallel:
        range = prange
    else:
        range = np.arange
    
    Ne = ep.shape[0]
    nPg = ep.shape[1]
    dimI = epij.shape[2]
    dimJ = epij.shape[3]

    assert Ne == epij.shape[0], "Mauvaise dimension"
    assert nPg == epij.shape[1], "Mauvaise dimension"

    result = np.zeros((Ne, dimI, dimJ))

    for e in range(Ne):
        for p in range(nPg):
            for i in range(dimI):
                for j in range(dimJ):
                    result[e, i, j] += ep[e, p] * epij[e, p, i, j]

    return result


@njit(cache=useCache, parallel=useParallel, fastmath=useFastmath)
def epij_epjk_epkl_to_eil(epij: np.ndarray, epjk: np.ndarray, epkl: np.ndarray) -> np.ndarray:
    if useParallel:
        range = prange
    else:
        range = np.arange
    
    Ne = epij.shape[0]
    nPg = epij.shape[1]
    dimI = epij.shape[2]
    dimJ = epij.shape[3]
    dimK = epjk.shape[3]
    dimL = epkl.shape[3]

    assert Ne == epjk.shape[0] and Ne == epkl.shape[0], "Mauvaise dimension"
    assert nPg == epjk.shape[1] and nPg == epkl.shape[1], "Mauvaise dimension"
    assert dimJ == epjk.shape[2]
    assert dimK == epkl.shape[2]

    result = np.zeros((Ne, dimI, dimL))

    for e in range(Ne):
        for p in range(nPg):
            for i in range(dimI):
                for j in range(dimJ):
                    for k in range(dimK):
                        for l in range(dimL):
                            result[e, i, l] += epij[e, p ,i, j] * epjk[e, p ,j, k] * epkl[e, p ,k, l]

    return result

@njit(cache=useCache, parallel=useParallel, fastmath=useFastmath)
def epij_jk_epkl_to_eil(epij: np.ndarray, jk: np.ndarray, epkl: np.ndarray) -> np.ndarray:
    if useParallel:
        range = prange
    else:
        range = np.arange
    
    Ne = epij.shape[0]
    nPg = epij.shape[1]
    dimI = epij.shape[2]
    dimJ = epij.shape[3]
    dimK = jk.shape[1]
    dimL = epkl.shape[3]

    assert Ne == Ne == epkl.shape[0], "Mauvaise dimension"
    assert nPg == epkl.shape[1], "Mauvaise dimension"
    assert dimJ == jk.shape[0]
    assert dimK == epkl.shape[2]

    result = np.zeros((Ne, dimI, dimL))

    for e in range(Ne):
        for p in range(nPg):
            for i in range(dimI):
                for j in range(dimJ):
                    for k in range(dimK):
                        for l in range(dimL):
                            result[e, i, l] += epij[e, p ,i, j] * jk[j, k] * epkl[e, p ,k, l]
    
    return result

# Construction des matrices

@njit(cache=useCache, parallel=useParallel, fastmath=useFastmath)
def Construit_Kd_e_and_Fd_e(r_e_pg: np.ndarray, ReactionPart_e_pg: np.ndarray,
k: float, DiffusePart_e_pg: np.ndarray,
f_e_pg: np.ndarray, SourcePart_e_pg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if useParallel:
        range = prange
    else:
        range = np.arange
    
    Ne = r_e_pg.shape[0]
    nPg = r_e_pg.shape[1]
    dimI = ReactionPart_e_pg.shape[2]
    dimJ = ReactionPart_e_pg.shape[3]
    
    assert dimI == dimJ, "Mauvaise dimension"
    assert dimI == DiffusePart_e_pg.shape[2], "Mauvaise dimension"
    assert dimJ == DiffusePart_e_pg.shape[3], "Mauvaise dimension"

    K_r_e = np.zeros((Ne, dimI, dimJ), dtype=float)
    K_K_e = np.zeros((Ne, dimI, dimJ), dtype=float)
    Fd_e = np.zeros((Ne, dimI, 1), dtype=float)

    for e in range(Ne):
        for p in range(nPg):
            for i in range(dimI):
                for j in range(dimJ):
                    K_r_e[e, i, j] += r_e_pg[e, p] * ReactionPart_e_pg[e, p, i, j]
                    K_K_e[e, i, j] += k * DiffusePart_e_pg[e, p, i, j]
                Fd_e[e, i, 0] += f_e_pg[e, p] * SourcePart_e_pg[e, p, i, 0]
    
    return K_r_e + K_K_e, Fd_e

@njit(cache=useCache, parallel=useParallel, fastmath=useFastmath)
def Calc_psi_e_pg(Epsilon_e_pg: np.ndarray, SigmaP_e_pg: np.ndarray, SigmaM_e_pg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if useParallel:
        range = prange
    else:
        range = np.arange

    Ne = Epsilon_e_pg.shape[0]
    nPg = Epsilon_e_pg.shape[1]
    dimI = Epsilon_e_pg.shape[2]

    psiP_e_pg = np.zeros((Ne, nPg))
    psiM_e_pg = np.zeros((Ne, nPg))

    for e in range(Ne):
        for p in range(nPg):
            for i in range(dimI):
                psiP_e_pg[e,p] += 1/2 * SigmaP_e_pg[e,p,i] * Epsilon_e_pg[e,p,i]
                psiM_e_pg[e,p] += 1/2 * SigmaM_e_pg[e,p,i] * Epsilon_e_pg[e,p,i]
    
    return psiP_e_pg, psiM_e_pg

@njit(cache=useCache, parallel=useParallel, fastmath=useFastmath)
def Calc_Sigma_e_pg(Epsilon_e_pg: np.ndarray, cP_e_pg: np.ndarray, cM_e_pg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if useParallel:
        range = prange
    else:
        range = np.arange

    Ne = Epsilon_e_pg.shape[0]
    nPg = Epsilon_e_pg.shape[1]
    dimI = cP_e_pg.shape[2]
    dimJ = cP_e_pg.shape[3]

    SigmaP_e_pg = np.zeros((Ne, nPg, dimI))
    SigmaM_e_pg = np.zeros((Ne, nPg, dimI))

    for e in range(Ne):
        for p in range(nPg):
            for i in range(dimI):
                for j in range(dimJ):
                    SigmaP_e_pg[e,p,i] += cP_e_pg[e,p,i,j] * Epsilon_e_pg[e,p,j]
                    SigmaM_e_pg[e,p,i] += cM_e_pg[e,p,i,j] * Epsilon_e_pg[e,p,j]
    
    return SigmaP_e_pg, SigmaM_e_pg


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
    
