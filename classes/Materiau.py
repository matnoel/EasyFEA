

from Mesh import Mesh
import numpy as np

class LoiDeComportement(object):
    """Classe des lois de comportements C de (Sigma = C * Epsilon)
    (Elas_isot, ...)
    """
    def __init__(self, dim: int, C: np.ndarray, S: np.ndarray, epaisseur: float, voigtNotation: bool):
        
        self.__dim = dim
        """dimension lié a la loi de comportement"""

        if dim == 2:
            assert epaisseur > 0 , "Doit être supérieur à 0"
            self.__epaisseur = epaisseur
        
        self.__voigtNotation = voigtNotation
        """Notation de voigt True or False"""
        
        self.__C = C
        """Loi de comportement pour la loi de Lamé en voigt"""

        self.__S = S
        """Loi de comportement pour la loi de Hooke en voigt"""
    
    def __get_voigtNotation(self):
        return self.__voigtNotation
    voigtNotation = property(__get_voigtNotation)

    def __get_coef_voigtNotation(self):
        if self.__voigtNotation:
            return 2
        else:
            return np.sqrt(2)
    coef = property(__get_coef_voigtNotation)
    """Coef lié à la notation utilisé voigt=2 kelvinMandel=racine(2)"""

    def get_C(self):
        """Renvoie une copie de la loi de comportement pour la loi de Lamé"""        
        return self.__C.copy()

    def get_S(self):
        """Renvoie une copie de la loi de comportement pour la loi de Hooke"""
        return self.__S.copy()

    def __getdim(self):
        return self.__dim
    dim = property(__getdim)

    def __getepaisseur(self):
        if self.__dim == 2:
            return self.__epaisseur
        else:
            return 1.0
    epaisseur = property(__getepaisseur)

    def __get_nom(self):
        return type(self).__name__
    nom = property(__get_nom)

    @staticmethod
    def ToKelvinMandelNotation(dim: int, voigtMatrice: np.ndarray):

        r2 = np.sqrt(2)

        if dim == 2:            
            transform = np.array([  [1,1,r2],
                                    [1,1,r2],
                                    [r2, r2, 2]])
        elif dim == 3:
            transform = np.array([  [1,1,1,r2,r2,r2],
                                    [1,1,1,r2,r2,r2],
                                    [1,1,1,r2,r2,r2],
                                    [r2,r2,r2,2,2,2],
                                    [r2,r2,r2,2,2,2],
                                    [r2,r2,r2,2,2,2]])
        else:
            raise "Pas implémenté"

        mandelMatrice = voigtMatrice*transform

        return mandelMatrice

class Elas_Isot(LoiDeComportement):   

    def __init__(self, dim: int, E=210000.0, v=0.3, contraintesPlanes=True, epaisseur=1.0,
    voigtNotation=False):
        """Creer la matrice de comportement d'un matériau : Elastique isotrope

        Parameters
        ----------
        dim : int
            Dimension de la simulation 2D ou 3D
        E : float, optional
            Module d'elasticité du matériau en MPa (> 0)
        v : float, optional
            Coef de poisson ]-1;0.5]
        contraintesPlanes : bool
            Contraintes planes si dim = 2 et True, by default True        
        """       

        # Vérification des valeurs
        assert dim in [2,3], "doit être en 2 et 3"
        self.__dim = dim

        assert E > 0.0, "Le module élastique doit être > 0 !"
        self.E=E
        """Module de Young"""

        poisson = "Le coef de poisson doit être compris entre ]-1;0.5["
        assert v > -1.0 and v < 0.5, poisson
        self.v=v
        """Coef de poisson"""        

        if dim == 2:
            self.contraintesPlanes = contraintesPlanes
            """type de simplification 2D"""

        C, S = self.__Comportement_Elas_Isot(voigtNotation)

        LoiDeComportement.__init__(self, dim, C, S, epaisseur, voigtNotation)

    def get_lambda(self):

        E=self.E
        v=self.v
        
        l = v*E/((1+v)*(1-2*v))        

        if self.__dim == 2 and self.contraintesPlanes:
            l = E*v/(1-v**2)
        
        return l
    
    def get_mu(self):
        
        E=self.E
        v=self.v

        mu = E/(2*(1+v))

        return mu
    
    def get_bulk(self):

        E=self.E
        v=self.v

        mu = self.get_mu()
        l = self.get_lambda()
        
        
        bulk = l + 2*mu/self.dim        

        return bulk

    def __Comportement_Elas_Isot(self, voigtNotation: bool):
        """"Construit les matrices de comportement"""

        E=self.E
        v=self.v

        dim = self.__dim

        mu = self.get_mu()
        l = self.get_lambda()

        if dim == 2:
            if self.contraintesPlanes:
                # C = np.array([  [4*(mu+l), 2*l, 0],
                #                 [2*l, 4*(mu+l), 0],
                #                 [0, 0, 2*mu+l]]) * mu/(2*mu+l)

                cVoigt = np.array([ [1, v, 0],
                                    [v, 1, 0],
                                    [0, 0, (1-v)/2]]) * E/(1-v**2)
                
            else:
                cVoigt = np.array([ [l + 2*mu, l, 0],
                                    [l, l + 2*mu, 0],
                                    [0, 0, mu]])

                # C = np.array([  [1, v/(1-v), 0],
                #                 [v/(1-v), 1, 0],
                #                 [0, 0, (1-2*v)/(2*(1-v))]]) * E*(1-v)/((1+v)*(1-2*v))

        elif dim == 3:
            
            cVoigt = np.array([ [l+2*mu, l, l, 0, 0, 0],
                                [l, l+2*mu, l, 0, 0, 0],
                                [l, l, l+2*mu, 0, 0, 0],
                                [0, 0, 0, mu, 0, 0],
                                [0, 0, 0, 0, mu, 0],
                                [0, 0, 0, 0, 0, mu]])
        
        # To kelvin mandel's notation

        if voigtNotation:
            c = cVoigt
        else:
            c = LoiDeComportement.ToKelvinMandelNotation(dim, cVoigt)

        return c, np.linalg.inv(c)


class PhaseFieldModel:

    __splits = ["Bourdin","Amor","Miehe"]

    __regularizations = ["AT1","AT2"]

    def __get_k(self):
        Gc = self.__Gc
        l0 = self.__l0

        k = Gc * l0

        if self.__regularization == "AT1":
            k = 3/4 * k

        return k
        
    k = property(__get_k)

    def get_r_e_pg(self, PsiP_e_pg: np.ndarray):
        Gc = self.__Gc
        l0 = self.__l0

        r = 2 * PsiP_e_pg

        if self.__regularization == "AT2":
            r = r + (Gc/l0)
        
        return r

    def get_f_e_pg(self, PsiP_e_pg: np.ndarray):
        Gc = self.__Gc
        l0 = self.__l0

        f = 2 * PsiP_e_pg

        if self.__regularization == "AT1":            
            f = f - ( (3*Gc) / (8*l0) )            
            absF = np.abs(f)
            f = (f+absF)/2
        
        return f

    def get_g_e_pg(self, d_n: np.ndarray, mesh: Mesh, k_residu=1e-10):
        """Fonction de dégradation en energies / contraintes

        Args:
            d_n (np.ndarray): Endomagement localisé aux noeuds (Nn,1)
            mesh (Mesh): maillage
        """
        d_e_n = mesh.Localise_e(d_n)
        Nd_pg = np.array(mesh.N_mass_pg)

        d_e_pg = np.einsum('pij,ej->ep', Nd_pg, d_e_n, optimize=True)        

        if self.__regularization in ["AT1","AT2"]:
            g_e_pg = (1-d_e_pg)**2 + k_residu
        else:
            raise "Pas implémenté"

        assert mesh.Ne == g_e_pg.shape[0]
        assert mesh.nPg == g_e_pg.shape[1]
        
        return g_e_pg

    def __resume(self):
        print(f'\ncomportement : {self.__loiDeComportement.nom}')
        print(f'split : {self.__split}')
        print(f'regularisation : {self.__regularization}\n')
    resume = property(__resume)

    def __init__(self, loiDeComportement: LoiDeComportement,split: str, regularization: str, Gc: float, l_0: float):
        """Création d'un objet comportement Phase Field

            Parameters
            ----------
            loiDeComportement : LoiDeComportement
                Loi de comportement du matériau ["Elas_Isot"]
            split : str
                Split de la densité d'energie elastique ["Bourdin","Amor","Miehe"]
            regularization : str
                Modèle de régularisation de la fissure ["AT1","AT2"]
            Gc : float
                Taux de libération d'énergie critque [J/m^2]
            l_0 : float
                Largeur de régularisation de la fissure
        """

        assert isinstance(loiDeComportement, LoiDeComportement), "Doit être une loi de comportement"
        self.__loiDeComportement = loiDeComportement

        assert split in PhaseFieldModel.__splits, f"Doit être compris dans {PhaseFieldModel.__splits}"
        if not isinstance(loiDeComportement, Elas_Isot):
            assert not split in ["Amor, Miehe"], "Ces splits ne sont implémentés que pour Elas_Isot"
        self.__split =  split
        """Split de la densité d'energie elastique ["Bourdin","Amor","Miehe"]"""
        
        assert regularization in PhaseFieldModel.__regularizations, f"Doit être compris dans {PhaseFieldModel.__regularizations}"
        self.__regularization = regularization
        """Modèle de régularisation de la fissure ["AT1","AT2"]"""

        assert Gc > 0, "Doit être supérieur à 0" 
        self.__Gc = Gc
        """Taux de libération d'énergie critque [J/m^2]"""

        assert l_0 > 0, "Doit être supérieur à 0" 
        self.__l0 = l_0
        """Largeur de régularisation de la fissure"""
            
    def Calc_Psi_e_pg(self, Epsilon_e_pg: np.ndarray):
        """Calcul de la densité d'energie elastique\n
        Ici on va caluler PsiP_e_pg = 1/2 SigmaP_e_pg : Epsilon_e_pg et PsiM_e_pg = 1/2 SigmaM_e_pg : Epsilon_e_pg\n
        Avec SigmaP_e_pg = CP * Epsilon_e_pg et SigmaM_e_pg = CM * Epsilon_e_pg

        Args:
            Epsilon_e_pg (np.ndarray): Deformation (Ne,pg,(3 ou 6))
            g_e_pg (list, optional): Fonction d'endomagement enégétique (Ne, pg) . Defaults to [].
        """
        
        # Data
        Ne = Epsilon_e_pg.shape[0]
        nPg = Epsilon_e_pg.shape[1]        

        SigmaP_e_pg, SigmaM_e_pg = self.Calc_Sigma_e_pg(Epsilon_e_pg)

        PsiP_e_pg = 1/2 * np.einsum('epi,epi->ep', SigmaP_e_pg, Epsilon_e_pg, optimize=True).reshape((Ne, nPg))
        PsiM_e_pg = 1/2 * np.einsum('epi,epi->ep', SigmaM_e_pg, Epsilon_e_pg, optimize=True).reshape((Ne, nPg))
        
        return PsiP_e_pg, PsiM_e_pg

    def Calc_Sigma_e_pg(self, Epsilon_e_pg: np.ndarray):
        """Calcul la contrainte en fonction de la deformation et du split
        Ici on calcul SigmaP_e_pg = CP * Epsilon_e_pg et SigmaM_e_pg = CM * Epsilon_e_pg

        Parameters
        ----------
        Epsilon_e_pg : np.ndarray
            deformations stockées aux elements et Points de Gauss

        Returns
        -------
        np.ndarray
            SigmaP_e_pg, SigmaM_e_pg : les contraintes stockées aux elements et Points de Gauss
        """       

        Ne = Epsilon_e_pg.shape[0]
        nPg = Epsilon_e_pg.shape[1]
        comp = Epsilon_e_pg.shape[2]

        cP_e_pg, cM_e_pg = self.Calc_C(Epsilon_e_pg)

        SigmaP_e_pg = np.einsum('epij,epj->epi', cP_e_pg, Epsilon_e_pg, optimize=True).reshape((Ne, nPg, comp))
        SigmaM_e_pg = np.einsum('epij,epj->epi', cM_e_pg, Epsilon_e_pg, optimize=True).reshape((Ne, nPg, comp))        

        return SigmaP_e_pg, SigmaM_e_pg
    
    def Calc_C(self, Epsilon_e_pg: np.ndarray):
        """Calcul la loi de comportement en fonction du split

        Parameters
        ----------
        Epsilon_e_pg : np.ndarray
            deformations stockées aux élements et points de gauss (Pas utilisé si bourdin)

        Returns
        -------
        np.ndarray
            Revoie cP_e_pg, cM_e_pg
        """

        Ne = Epsilon_e_pg.shape[0]
        nPg = Epsilon_e_pg.shape[1]
        comp = Epsilon_e_pg.shape[2]
        
        if self.__split == "Bourdin":
            
            c = self.__loiDeComportement.get_C()
            c = c[np.newaxis, np.newaxis,:,:]
            c = np.repeat(c, Ne, axis=0)
            c = np.repeat(c, nPg, axis=1)

            cP_e_pg = c
            cM_e_pg = 0*cP_e_pg

        elif self.__split == "Amor":
            raise "Pas encore implémenté"
        elif self.__split == "Miehe":
            raise "Pas encore implémenté"

        return cP_e_pg, cM_e_pg    

class Materiau:
    
    def __get_dim(self):
        return self.comportement.dim
    dim = property(__get_dim)    

    def __init__(self, comportement: LoiDeComportement, ro=8100.0, phaseFieldModel=None):
        """Creer un materiau

        Parameters
        ----------                        
        ro : float, optional
            Masse volumique en kg.m^-3
        epaisseur : float, optional
            epaisseur du matériau si en 2D > 0 !
        """
        
        assert isinstance(comportement, LoiDeComportement)

        assert ro > 0 , "Doit être supérieur à 0"
        self.ro = ro               

        # Initialisation des variables de la classe

        self.comportement = comportement
        """Comportement du matériau"""

        if isinstance(phaseFieldModel, PhaseFieldModel):
            self.phaseFieldModel = phaseFieldModel
            """Phase field model"""
        else:
            self.phaseFieldModel = None


# TEST ==============================

import unittest
import os

class Test_Materiau(unittest.TestCase):
    def setUp(self):

        self.E = 2
        self.v = 0.2

        self.materiau_Isot_CP = Materiau(Elas_Isot(2, E=self.E, v=self.v, contraintesPlanes=True), ro=700)
        self.materiau_Isot_DP = Materiau(Elas_Isot(2, E=self.E, v=self.v, contraintesPlanes=False), ro=700)
        self.materiau_Isot = Materiau(Elas_Isot(3, E=self.E, v=self.v), ro=700)

    def test_BienCree_Isotrope(self):

        E = self.E
        v = self.v

        self.assertIsInstance(self.materiau_Isot_CP, Materiau)

        C_CP = E/(1-v**2) * np.array([  [1, v, 0],
                                        [v, 1, 0],
                                        [0, 0, (1-v)/2]])

        C_DP = E/((1+v)*(1-2*v)) * np.array([  [1-v, v, 0],
                                                [v, 1-v, 0],
                                                [0, 0, (1-2*v)/2]])

        C_3D = E/((1+v)*(1-2*v))*np.array([ [1-v, v, v, 0, 0, 0],
                                            [v, 1-v, v, 0, 0, 0],
                                            [v, v, 1-v, 0, 0, 0],
                                            [0, 0, 0, (1-2*v)/2, 0, 0],
                                            [0, 0, 0, 0, (1-2*v)/2, 0],
                                            [0, 0, 0, 0, 0, (1-2*v)/2]  ])        
        

        self.assertTrue(np.allclose(C_CP, self.materiau_Isot_CP.comportement.get_C(), 1e-8))
        self.assertTrue(np.allclose(C_DP, self.materiau_Isot_DP.comportement.get_C(), 1e-8))
        self.assertTrue(np.allclose(C_3D, self.materiau_Isot.comportement.get_C(), 1e-8))


if __name__ == '__main__':        
    try:
        os.system("cls")
        unittest.main(verbosity=2)
    except:
        print("")