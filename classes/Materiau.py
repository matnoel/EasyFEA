import numpy as np
from typing import cast

class LoiDeComportement(object):
    """Classe des lois de comportements C de (Sigma = C * Epsilon)
    (Elas_isot, ...)
    """
    def __init__(self, nom: str, dim: int, C: np.ndarray, S: np.ndarray):

        self.__nom = nom
        self.__dim = dim
        self.__C = C
        self.__S = S
    
    def get_C(self):        
        return self.__C.copy()

    def get_S(self):
        return self.__S.copy()

    def __getdim(self):
        return self.__dim
    dim = property(__getdim)

class Elas_Isot(LoiDeComportement):   

    def __init__(self, dim: int, E=210000.0, v=0.3, contraintesPlanes=True):
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
        assert E > 0.0, "Le module élastique doit être > 0 !"
        self.E=E

        poisson = "Le coef de poisson doit être compris entre ]-1;0.5["
        assert v > -1.0 and v < 0.5, poisson
        self.v=v

        self.contraintesPlanes = contraintesPlanes

        C, S = self.__Comportement_Elas_Isot(dim)

        LoiDeComportement.__init__(self,"Elas_Isot", dim, C, S)


    def __Comportement_Elas_Isot(self, dim):
        # Construction matrice de comportement

        E=self.E
        v=self.v

        mu = E/(2*(1+v))        
        l = v*E/((1+v)*(1-2*v)) #E*v/(1-v**2)

        if dim == 2:
            if self.contraintesPlanes:
                # C = np.array([  [4*(mu+l), 2*l, 0],
                #                 [2*l, 4*(mu+l), 0],
                #                 [0, 0, 2*mu+l]]) * mu/(2*mu+l)

                C = np.array([  [1, v, 0],
                                [v, 1, 0],
                                [0, 0, (1-v)/2]]) * E/(1-v**2)
                
            else:
                C = np.array([  [l + 2*mu, l, 0],
                                [l, l + 2*mu, 0],
                                [0, 0, mu]])

                # C = np.array([  [1, v/(1-v), 0],
                #                 [v/(1-v), 1, 0],
                #                 [0, 0, (1-2*v)/(2*(1-v))]]) * E*(1-v)/((1+v)*(1-2*v))

        elif dim == 3:
            
            C = np.array([  [l+2*mu, l, l, 0, 0, 0],
                            [l, l+2*mu, l, 0, 0, 0],
                            [l, l, l+2*mu, 0, 0, 0],
                            [0, 0, 0, mu, 0, 0],
                            [0, 0, 0, 0, mu, 0],
                            [0, 0, 0, 0, 0, mu]])

        return C, np.linalg.inv(C)

class Materiau:
    
    def get_dim(self):
        return self.comportement.dim
    dim = property(get_dim)

    def __init__(self, comportement: LoiDeComportement, ro=8100.0, epaisseur=1.0):
        """Creer un materiau

        Parameters
        ----------                        
        ro : float, optional
            Masse volumique en kg.m^-3
        epaisseur : float, optional
            epaisseur du matériau si en 2D > 0 !
        """
        
        assert ro > 0 , "Doit être supérieur à 0"
        self.ro = ro
        
        if comportement.dim == 2:
            assert epaisseur > 0 , "Doit être supérieur à 0"
            self.epaisseur = epaisseur

        # Initialisation des variables de la classe

        self.comportement = comportement


    

    
        


# TEST ==============================

import unittest
import os

class Test_Materiau(unittest.TestCase):
    def setUp(self):

        self.materiau_Isot_CP = Materiau(2, E=2, v=0.2, ro=700, isotrope=True, contraintesPlanes=True)
        self.materiau_Isot_DP = Materiau(2, E=2, v=0.2, ro=700, isotrope=True, contraintesPlanes=False)
        self.materiau_Isot = Materiau(3, E=2, v=0.2, ro=700, isotrope=True, contraintesPlanes=False)

    def test_BienCree_Isotrope(self):

        E = self.materiau_Isot_CP.E
        v = self.materiau_Isot_CP.v

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
        

        self.assertTrue(np.allclose(C_CP, self.materiau_Isot_CP.C, 1e-8))
        self.assertTrue(np.allclose(C_DP, self.materiau_Isot_DP.C, 1e-8))
        self.assertTrue(np.allclose(C_3D, self.materiau_Isot.C, 1e-8))


if __name__ == '__main__':        
    try:
        os.system("cls")
        unittest.main(verbosity=2)
    except:
        print("")