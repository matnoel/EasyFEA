from matplotlib.pyplot import connect
import numpy as np
from numpy.core.records import array
from scipy.sparse import coo

from class_Element import Element
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
    
    def __init__(self, coordo: np.ndarray, connection: list, dim: int, C: np.ndarray):
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

        cc = coordo[1]
        t = type(cc)
                
        # Vérfication
        assert isinstance(coordo, np.ndarray) ,"Doit fournir une liste de ndarray' !"
        assert isinstance(coordo[0], np.ndarray) ,"Doit fournir une liste de ndarray' !"
        # assert isinstance(coordo, np.ndarray) and ,"Doit fournir une liste de ndarray' !"
        assert isinstance(connection, list) and isinstance(connection[0], list),"Doit fournir une liste de liste !"

        self.coordo = np.array(coordo)
        self.connection = connection
        
        self.connectionPourAffichage = []
        self.new_coordo = []
        
        self.noeuds = []
        self.elements = []
                
        # Création des noeuds
        n = 0
        for c in coordo:            
            x = c[0]
            y = c[1]
            z = c[2]
            
            if dim ==2:
                assert z == 0 or z == 0.0,"Pour une étude 2D tout les noeuds doivent être dans le plan x, y"
            
            # Création du noeud
            noeud = Noeud(n, x, y, z)
            self.noeuds.append(noeud)
            
            n += 1
        
        Ne = len(connection)

        # Créations des éléments
        e = 0              
        while e < Ne:
                                   
            # Construit la liste de noeuds de l'element 
            listNoeudsElement = []
            for n in connection[e]:
                listNoeudsElement.append(self.noeuds[n])
            
            # Création de l'élement
            element = Element(e, listNoeudsElement, dim, C)
            
            # Ajoute l'element dans la liste d'élement de la simu
            self.elements.append(element)
            
            e += 1
            
# TEST ==============================

import unittest
import os

class Test_Mesh(unittest.TestCase):
    def setUp(self):
        
        coordo = []
        
        coordo.append(array([0, 0, 0]))
        coordo.append(array([1, 0, 0]))
        coordo.append(array([0, 1, 0]))
        
        connect = [0, 1, 2]
        
        C = 2/((1+0.2)*(1-2*0.2)) * np.array([  [1-0.2, 0.2, 0],
                                                [0.2, 1-0.2, 0],
                                                [0, 0, (1-2*0.2)/2]   ])
        
        self.mesh = Mesh(coordo, connect, 2, C)

    def test_BienCree(self):
        self.assertIsInstance(self.mesh, Mesh)

if __name__ == '__main__':        
    try:
        os.system("cls")
        unittest.main(verbosity=2)
    except:
        print("")        