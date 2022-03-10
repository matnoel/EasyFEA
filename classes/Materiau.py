import numpy as np

class Materiau:
    
    def get_dim(self):
        return self.__dim

    def __init__(self, dim: int, E=210000.0, v=0.3, ro=8100, epaisseur=1, isotrope=True, contraintesPlanes=False):
        """Creer un materiau

        Parameters
        ----------
        dim : int
            Dimension de la simulation 2D ou 3D
        E : float, optional
            Module d'elasticité du matériau en MPa (> 0)
        v : float, optional
            Coef de poisson ]-1;0.5]
        ro : int, optional
            Masse volumique en kg.m^-3
        isotrope : bool
            Matériau isotrope
        contraintesPlanes : bool
            Contraintes planes si dim = 2 et True, by default True
        """
        
        # Vérification des valeurs
        assert E > 0.0, "Le module élastique doit être > 0 !"
        
        poisson = "Le coef de poisson doit être compris entre ]-1;0.5]"
        assert v > -1.0, poisson
        assert v <= 0.5, poisson

        assert ro > 0 , "Doit être supérieur à 0"

        if dim == 2:        
            assert epaisseur>0,"Doit être supérieur à 0"
            self.epaisseur = epaisseur


        # Initialisation des variables de la classe

        self.__dim = dim

        self.E = E 
        self.v = v 
        self.ro = ro        
        
        self.isotrope = isotrope
        
        self.contraintesPlanes = contraintesPlanes
        
        self.C = None

        if isotrope:
            self.__Calc_C_Isotrope(contraintesPlanes)
       

    def __Calc_C_Isotrope(self, contraintesPlanes: bool):
        # Construction matrice de comportement C (Sigma=C*Epsilon)

        E = self.E
        v = self.v

        self.mu = E/(2+2*v)

        if self.__dim == 2:

            epaisseur = self.epaisseur
            
            if contraintesPlanes:               
                self.__lambda = E*v/(1-v**2)
            else:
                self.__lambda = v*E/((1+v)*(1-2*v))

            lambd = self.__lambda
            mu = self.mu

            self.C = np.array([ [lambd + 2*mu, lambd, 0],
                                [lambd, lambd + 2*mu, 0],
                                [0, 0, mu]]) 

            # self.C = self.C * epaisseur

        elif self.__dim == 3:
            
            self.__lambda = v*E/((1+v)*(1-2*v))

            l = self.__lambda
            m = self.mu

            self.C = np.array([[l+2*m, l, l, 0, 0, 0],
                                [l, l+2*m, l, 0, 0, 0],
                                [l, l, l+2*m, 0, 0, 0],
                                [0, 0, 0, m, 0, 0],
                                [0, 0, 0, 0, m, 0],
                                [0, 0, 0, 0, 0, m]])


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