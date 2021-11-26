from numpy import ndarray


class Noeud:  
    
    def __init__(self, id: int, coordo: ndarray):
        """Constructeur pour un noeuds

        Parameters
        ----------
        id : int
            Numéro de l'élement (>=0)
        coordo : float
            Coordonnées du noeud [x,y,z]
        """
        
        # Vérification
        assert isinstance(id, int),"Doit être un entier"
        assert id >= 0,"Doit être >= 0"
        
        
        # Variables utilisées        
        self.id = id
        self.coordo = coordo       
        self.elements = []
    
   
    def AddElement(self, element):
        """Ajoute l'élément à la liste d'éléments du noeud si il ne le contient pas

        Parameters
        ----------
        element : Element
            Element à ajouter
        """
        
        assert self in element.noeuds, "L'element ne possède pas le noeud"
        
        if element not in self.elements:
            self.elements.append(element)

# ====================================

import unittest
import os

class Test_Noeud(unittest.TestCase):
    def setUp(self):
        self.n = Noeud(1, [0, 1, 2])

    def test_BienCree(self):
        self.assertIsInstance(self.n, Noeud)
        self.assertListEqual(self.n.coordo,[0, 1, 2])            
        self.assertEqual(self.n.id, 1)


if __name__ == '__main__':        
    try:
        os.system("cls")
        unittest.main(verbosity=2)
    except:
        print("")