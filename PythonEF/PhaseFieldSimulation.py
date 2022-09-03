from TicTac import Tic
from Simu import Simu
import numpy as np
import scipy.sparse as sp
import Dossier

def ResolutionIteration(simu: Simu, tolConv=1, maxIter=200) -> tuple[np.ndarray, np.ndarray, sp.csr_matrix, int]:
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

    iterConv = 0
    convergence = False
    dn = simu.damage

    while not convergence:
                
        iterConv += 1
        # Ancien endommagement dans la procedure de la convergence
        dk = simu.damage

        # Displacement
        Kglob = simu.Assemblage_u()            
        u = simu.Solve_u()

        # Damage
        simu.Assemblage_d()
        dkp1 = simu.Solve_d()
        
        # Condition de convergence
        dincMax = np.max(np.abs(dkp1-dk))
        print(f"{iterConv} : {dincMax}\r")
        convergence = dincMax <= tolConv
    
        if iterConv == maxIter:
            break
        
        if tolConv == 1.0:
            convergence=True

    if simu.materiau.phaseFieldModel.useHistory:
        oldAndNewDamage = np.zeros((dkp1.shape[0], 2))
        oldAndNewDamage[:, 0] = dn
        oldAndNewDamage[:, 1] = dkp1
        dnp1 = np.max(oldAndNewDamage, 1)
    else:
        dnp1 = dkp1
        
    return u, dnp1, Kglob, iterConv

class listTemps:
    def init(self):
        self.listTemps = []
    @property
    def temps(self) -> float:
        return np.mean(self.listTemps)

listTmps = listTemps()
resumeIter = ""
def ResumeIteration(simu: Simu, resol: int, dep: float, d: np.ndarray, iterConv: int, temps: float, uniteDep="m", pourcentage=0, remove=False):
    min_d = d.min()
    max_d = d.max()
    resumeIter = f"{resol:4} : ud = {np.round(dep,3)} {uniteDep},  d = [{min_d:.2e}; {max_d:.2e}], {iterConv}:{np.round(temps,3)} s  "
    
    if remove:
        end='\r'
    else:
        end=''
        
    if resol in [0,1]:
        listTmps.init()

    listTmps.listTemps.append(temps)
    temps = listTmps.temps

    if pourcentage > 0:
        tempsRestant = (1/pourcentage-1)*temps*resol
        
        tempsCoef, unite = Tic.Get_temps_unite(tempsRestant)

        resumeIter = resumeIter+f"{np.round(pourcentage*100,2)} % -> {np.round(tempsCoef,1)} {unite}   "
    
    print(resumeIter, end=end)

    simu.resumeIter = resumeIter

def ResumeChargement(simu: Simu, umax: float, listInc: list, listTreshold: list, option='damage'):
    listOption = ["damage", "displacement"]
    assert option in listOption, f"option doit etre dans {listOption}"
    assert len(listInc) == len(listTreshold), "Doit etre de la meme dimension"

    if option == 'damage':
        condition = 'd'
    elif option == 'displacement':
        condition = 'dep'
    
    resumeChargement = 'Chargement :'
    resumeChargement += f'\n\tumax = {umax}'
    for inc, treshold in zip(listInc, listTreshold):
        resumeChargement += f'\n\tinc = {inc} -> {condition} < {treshold}'
    
    simu.resumeChargement = resumeChargement

def ConstruitDossier(dossierSource: str, comp: str, split: str, regu: str, simpli2D: str, tolConv: float,
useHistory: bool, test: bool, openCrack=False, v=0.0):

    nom="_".join([comp, split, regu, simpli2D])

    if openCrack: 
        nom += '_openCrack'

    if tolConv < 1:
        nom += f'_convergence{tolConv}'
    
    if not useHistory:
        nom += '_noHistory'

    if comp == "Elas_Isot" and v != 0:
        nom = f"{nom} pour v={v}"

    folder = Dossier.NewFile(dossierSource, results=True)

    if test:
        folder = Dossier.Join([folder, "Test", nom])
    else:
        folder = Dossier.Join([folder, nom])

    print('\nSimulation dans :\n'+folder)

    return folder