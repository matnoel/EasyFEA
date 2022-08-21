from Simu import Simu
import numpy as np

def ResolutionIteration(simu: Simu, tolConv=1, maxIter=200) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Calcul l'itération d'un probleme d'endommagement de façon étagée

    Parameters
    ----------
    simu : Simu
        simulation
    tolConv : float, optional
        tolérance de convergence entre l'ancien et le nouvelle endommagement, by default 1.0
    maxIter : int, optional
        nombre d'itération maximum pour atteindre la convergence, by default 200

    Returns
    -------
    np.ndarray, np.ndarray, int
        u, d, Kglob, iterConv\n

        tel que :\n
        u : champ vectorielle de déplacement
        d : champ scalaire d'endommagement
        Kglob : matrice de rigidité en déplacement
        iterConv : iteration nécessaire pour atteindre la convergence
    """

    assert tolConv > 0 and tolConv <= 1 , "tolConv doit être compris entre 0 et 1"
    assert maxIter > 1 , "Doit être > 1"

    iterConv=0
    convergence = False
    d = simu.damage

    while not convergence:
                
        iterConv += 1
        dold = d.copy()

        # Damage
        simu.Assemblage_d()
        d = simu.Solve_d()

        # Displacement
        Kglob = simu.Assemblage_u()            
        u = simu.Solve_u()

        dincMax = np.max(np.abs(d-dold))
        convergence = dincMax <= tolConv
        # if damage.min()>1e-5:
        #     convergence=False

        if iterConv == maxIter:
            break
        
        if tolConv == 1.0:
            convergence=True
        
    return u, d, Kglob, iterConv