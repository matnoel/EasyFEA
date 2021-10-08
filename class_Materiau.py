import numpy as np

class Materiau:
    
    def __init__(self, dim: int, E: float, v: float, ro: float, isotrope: bool, contraintesPlanes: bool):
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
        
        # Initialisation des variables de la classe

        self.E = E
        self.v = v
        self.ro = ro
        
        self.isotrope = isotrope
        
        self.contraintesPlanes = contraintesPlanes
        
        self.C = None
        
        # Construction matrice d'élasticité C 
        if dim == 2:                       
            if contraintesPlanes:
                # isotrope + contraintes planes
                self.C =  E/(1-v**2) * np.array([   [1, v, 0],
                                                    [v, 1, 0],
                                                    [0, 0, (1-v)/2] ])  
            else:
                # isotrope + deformations planes
                self.C =  E/((1+v)*(1-2*v)) * np.array([    [1-v, v, 0],
                                                            [v, 1-v, 0],
                                                            [0, 0, (1-2*v)/2]   ])  
        elif dim == 3:
            # isotrope
            self.C = E/((1+v)*(1-2*v))*np.array([   [1-v, v, v, 0, 0, 0],
                                                    [v, 1-v, v, 0, 0, 0],
                                                    [v, v, 1-v, 0, 0, 0],
                                                    [0, 0, 0, (1-2*v)/2, 0, 0],
                                                    [0, 0, 0, 0, (1-2*v)/2, 0],
                                                    [0, 0, 0, 0, 0, (1-2*v)/2]  ])
            
# TEST ==============================

import unittest
import os

class Test_Materiau(unittest.TestCase):
    def setUp(self):
        self.materiau = Materiau(2, E=2, v=0.2, ro=700, isotrope=True, contraintesPlanes=False)

    def test_BienCree(self):
        self.assertIsInstance(self.materiau, Materiau)       
        self.assertListEqual([self.materiau.E, self.materiau.v, self.materiau.ro, self.materiau.isotrope, self.materiau.contraintesPlanes],
                             [2, 0.2, 700, True, False])
        C = 2/((1+0.2)*(1-2*0.2)) * np.array([  [1-0.2, 0.2, 0],
                                                [0.2, 1-0.2, 0],
                                                [0, 0, (1-2*0.2)/2]   ])
        
       
        self.assertTrue((C == self.materiau.C).all())        

if __name__ == '__main__':        
    try:
        os.system("cls")
        unittest.main(verbosity=2)
    except:
        print("")