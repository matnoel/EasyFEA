from matplotlib.pyplot import connect
import numpy as np
from class_Element import Element
from class_Materiau import Materiau
from class_Noeud import Noeud

class Mesh:
    
    def get_Ne(self):
        """Renvoie le nombre d'éléments du maillage        
        """
        return int(len(self.elements))
    Ne = property(get_Ne)
    
    def get_Nn(self):
        """Renvoie le nombre d'éléments du maillage        
        """
        return int(len(self.noeuds))
    Nn = property(get_Nn)

    def get_dim(self):
        return self.__dim

    
    def __init__(self, dim: int, coordo: np.ndarray, connect: list, verbosity=False):
        """Création du maillage depuis coordo et connection

        Parameters
        ----------
        coordo : list
            Coordonnées des noeuds dim(Nn,3), by default []
        connection : list
            Matrice de connection dim(Ne,nPe), by default []
        affichageMaillage : bool, optional
            Affichage après la construction du maillage, by default False
        """
    
        # Vérfication
        assert isinstance(coordo, np.ndarray) and isinstance(coordo[0], np.ndarray),"Doit fournir une liste de ndarray de ndarray !"
        
        assert isinstance(connect, list) and isinstance(connect[0], list),"Doit fournir une liste de liste"

        self.__dim = dim

        self.__verbosity = verbosity

        # self.coordo = np.array(coordo)
        self.coordo = coordo
        self.connection = connect
        
        self.connectionPourAffichage = []
        self.new_coordo = []
        
        self.noeuds = []
        self.elements = []
                
        # Création des noeuds
        n = 0
        for c in coordo:     
            if self.__dim ==2:
                assert c[2] == 0 or c[2] == 0.0,"Pour une étude 2D tout les noeuds doivent être dans le plan x, y"
            
            # Création du noeud
            noeud = Noeud(n, c)
            self.noeuds.append(noeud)
            
            n += 1
        
        Ne = len(connect)

        # Créations des éléments
        e = 0              
        while e < Ne:
                                   
            # Construit la liste de noeuds de l'element 
            listNoeudsElement = []
            for n in connect[e]:
                listNoeudsElement.append(self.noeuds[n])
            
            # Création de l'élement
            element = Element(e, listNoeudsElement, self.__dim)
            
            # Ajoute l'element dans la liste d'élement de la simu
            self.elements.append(element)
            
            e += 1
            
# TEST ==============================

import unittest
import os

class Test_Mesh(unittest.TestCase):
    def setUp(self):
        
        coordo = []
        
        coordo.append(np.array([0, 0, 0]))
        coordo.append(np.array([1, 0, 0]))
        coordo.append(np.array([0, 1, 0]))
        
        connect = [[0, 1, 2]]
        
        self.mesh = Mesh(2, np.array(coordo), connect)

    def test_BienCree(self):
        self.assertIsInstance(self.mesh, Mesh)

if __name__ == '__main__':        
    try:
        os.system("cls")
        unittest.main(verbosity=2)
    except:
        print("")        