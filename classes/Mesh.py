import numpy as np
from numpy.lib.twodim_base import triu_indices_from

try:
    from Element import Element
    from Materiau import Materiau
    from TicTac import TicTac
except:
    from classes.Element import Element
    from classes.Materiau import Materiau
    from classes.TicTac import TicTac


class Mesh:
    
    def get_Ne(self):
        """Renvoie le nombre d'éléments du maillage        
        """
        return int(len(self.connect))
    Ne = property(get_Ne)
    
    def get_Nn(self):
        """Renvoie le nombre d'éléments du maillage        
        """
        return int(len(self.coordo))
    Nn = property(get_Nn)

    def get_dim(self):
        return self.__dim

    def get_connectTriangle(self):
        """Transforme la matrice de connectivité pour la passer dans le trisurf en 2D
            ou construit les faces pour la 3D
            Par exemple pour un quadrangle on construit deux triangles
            pour un triangle à 6 noeuds on construit 4 triangles
            POur la 3D on construit des faces pour passer en Poly3DCollection
            """

        if len(self.__connectPourTriangle) == 0:
            
            connection = self.connect
            new_connection = []
            
            for listIdNoeuds in self.connect:
                npe = len(listIdNoeuds)
                
                if self.__dim == 2:            
                    # TRI3
                    if npe == 3:
                        self.__connectPourTriangle = connection
                        break            
                    # TRI6
                    elif npe == 6:
                        n1 = listIdNoeuds[0]
                        n2 = listIdNoeuds[1]
                        n3 = listIdNoeuds[2]
                        n4 = listIdNoeuds[3]
                        n5 = listIdNoeuds[4]
                        n6 = listIdNoeuds[5]

                        self.__connectPourTriangle.append([n1, n4, n6])
                        self.__connectPourTriangle.append([n4, n2, n5])
                        self.__connectPourTriangle.append([n6, n5, n3])
                        self.__connectPourTriangle.append([n4, n5, n6])                    
                    # QUAD4
                    elif npe == 4:
                        n1 = listIdNoeuds[0]
                        n2 = listIdNoeuds[1]
                        n3 = listIdNoeuds[2]
                        n4 = listIdNoeuds[3]                

                        self.__connectPourTriangle.append([n1, n2, n4])
                        self.__connectPourTriangle.append([n2, n3, n4])                    
                    # QUAD8
                    elif npe == 8:
                        n1 = listIdNoeuds[0]
                        n2 = listIdNoeuds[1]
                        n3 = listIdNoeuds[2]
                        n4 = listIdNoeuds[3]
                        n5 = listIdNoeuds[4]
                        n6 = listIdNoeuds[5]
                        n7 = listIdNoeuds[6]
                        n8 = listIdNoeuds[7]

                        self.__connectPourTriangle.append([n5, n6, n8])
                        self.__connectPourTriangle.append([n6, n7, n8])
                        self.__connectPourTriangle.append([n1, n5, n8])
                        self.__connectPourTriangle.append([n5, n2, n6])
                        self.__connectPourTriangle.append([n6, n3, n7])
                        self.__connectPourTriangle.append([n7, n4, n8])                    
                    
                elif self.__dim ==3:
                    pass

        return self.__connectPourTriangle
    
    def get_connectPolygon(self):
        """Construit les faces pour chaque element

        Returns
        -------
        list de list
            Renvoie une liste de face
        """
        if len(self.__connectPolygon) == 0:            
            for listIdNoeuds in self.connect:
                npe = len(listIdNoeuds)

                if self.__dim == 2:
                    # TRI3
                    if npe == 3:
                        n1 = listIdNoeuds[0]
                        n2 = listIdNoeuds[1]
                        n3 = listIdNoeuds[2]

                        self.__connectPolygon.append([n1, n2, n3, n1])
                    # TRI6
                    elif npe == 6:
                        n1 = listIdNoeuds[0]
                        n2 = listIdNoeuds[1]
                        n3 = listIdNoeuds[2]
                        n4 = listIdNoeuds[3]
                        n5 = listIdNoeuds[4]
                        n6 = listIdNoeuds[5]

                        self.__connectPolygon.append([n1, n4, n2, n5, n3, n6, n1])
                    # QUAD4
                    elif npe == 4:
                        # self.__connectPolygon = self.connect
                        # break
                        n1 = listIdNoeuds[0]
                        n2 = listIdNoeuds[1]
                        n3 = listIdNoeuds[2]
                        n4 = listIdNoeuds[3]

                        self.__connectPolygon.append([n1, n2, n3, n4, n1])
                    # QUAD8
                    elif npe == 8:
                        n1 = listIdNoeuds[0]
                        n2 = listIdNoeuds[1]
                        n3 = listIdNoeuds[2]
                        n4 = listIdNoeuds[3]
                        n5 = listIdNoeuds[4]
                        n6 = listIdNoeuds[5]
                        n7 = listIdNoeuds[6]
                        n8 = listIdNoeuds[7]

                        self.__connectPolygon.append([n1, n5, n2, n6, n3, n7, n4, n8, n1])
                elif self.__dim == 3:
                    # TETRA4
                    if npe == 4:
                        n1 = listIdNoeuds[0]
                        n2 = listIdNoeuds[1]
                        n3 = listIdNoeuds[2]
                        n4 = listIdNoeuds[3]
                                        
                        self.__connectPolygon.append([n1 ,n2, n3])
                        self.__connectPolygon.append([n1, n2, n4])
                        self.__connectPolygon.append([n1, n3, n4])
                        self.__connectPolygon.append([n2, n3, n4])        
        return self.__connectPolygon

    def __init__(self, dim: int, coordo: np.ndarray, connect: list, verbosity=True):
        """Création du maillage depuis coordo et connection

        Parameters
        ----------
        coordo : list
            Coordonnées des noeuds dim(Nn,3), by default []
        connection : list
            Matrice de connection dim(Ne,nPe), by default []
        affichageMaillage : bool, optional
            Affichage après la construction du maillage, by default True
        """
    
        # Vérfication
        assert isinstance(coordo, np.ndarray) and isinstance(coordo[0], np.ndarray),"Doit fournir une liste de ndarray de ndarray !"
        
        assert isinstance(connect, list) and isinstance(connect[0], list),"Doit fournir une liste de liste"

        tic = TicTac()

        self.__dim = dim

        self.__verbosity = verbosity

        self.coordo = np.array(coordo)
        self.connect = connect

        self.__ConstruitMatricesPourCalculEf()

        # Creation de l'élement
        # L'element permet de calculer les fonctions de formes et ses dérivées

        self.__connectPourTriangle = []
        self.__connectPolygon =[]

        
        tic.Tac("Importation du maillage", self.__verbosity)
        if verbosity:
            print("\nNe = {}, Nn = {}, nbDdl = {}".format(self.Ne,self.Nn,self.Nn*self.__dim)) 
    
    def __ConstruitMatricesPourCalculEf(self):
        
        tic = TicTac()

        verification = False

        dim = self.__dim
        connect = self.connect
        coordo = self.coordo
        listElement = range(self.Ne)
        listNoeud = range(self.Nn)

        # Construit la matrice assembly
        self.assembly_e = [[int(n * dim + d)for n in connect[e] for d in range(dim)] for e in listElement]
        
        element = Element(dim, len(connect[0]))

        listElement = list(range(self.Ne))
        nPe = element.nPe;  listnPe = list(range(nPe))
        nPg = element.nPg;  listPg = list(range(nPg))
        nodes = coordo[:,range(dim)]
        
        self.gauss = element.gauss

        self.poid_pg = self.gauss[:,-1]
        self.F_e_pg = np.array([[element.dN_pg[pg].dot(nodes[connect[e], :]) for pg in listPg] for e in listElement])
        self.invF_e_pg = np.linalg.inv(self.F_e_pg)       
        self.jacobien_e_pg = np.linalg.det(self.F_e_pg)
        self.N_rigi_pg = element.N_rigi_pg
        self.N_mass_pg = element.N_mass_pg
        self.dN_e_pg = np.array([[self.invF_e_pg[e,pg,:,:].dot(element.dN_pg[pg]) for pg in listPg] for e in listElement])        
        self.B_mass_e_pg = self.dN_e_pg

        colonnes0 = list(range(0, nPe*dim, dim))
        colonnes1 = list(range(1, nPe*dim, dim))

        if self.__dim == 2:
            self.B_rigi_e_pg = np.array([[np.zeros((3, nPe*dim))]*element.nPg]*self.Ne)
            
            dNdx = self.dN_e_pg[:,:,0,listnPe]
            dNdy = self.dN_e_pg[:,:,1,listnPe]

            self.B_rigi_e_pg[:,:,0,colonnes0] = dNdx
            self.B_rigi_e_pg[:,:,1,colonnes1] = dNdy
            self.B_rigi_e_pg[:,:,2,colonnes0] = dNdy; self.B_rigi_e_pg[:,:,2,colonnes1] = dNdx
        else:
            self.B_rigi_e_pg = np.array([[np.zeros((6, nPe*dim))]*element.nPg]*self.Ne)

            dNdx = self.dN_e_pg[:,:,0,listnPe]
            dNdy = self.dN_e_pg[:,:,1,listnPe]
            dNdz = self.dN_e_pg[:,:,2,listnPe]

            colonnes2 = list(range(2, nPe*dim, dim))

            self.B_rigi_e_pg[:,:,0,colonnes0] = dNdx
            self.B_rigi_e_pg[:,:,1,colonnes1] = dNdy
            self.B_rigi_e_pg[:,:,2,colonnes2] = dNdz
            self.B_rigi_e_pg[:,:,3,colonnes0] = dNdy; self.B_rigi_e_pg[:,:,3,colonnes1] = dNdx
            self.B_rigi_e_pg[:,:,4,colonnes1] = dNdz; self.B_rigi_e_pg[:,:,4,colonnes2] = dNdy
            self.B_rigi_e_pg[:,:,4,colonnes0] = dNdz; self.B_rigi_e_pg[:,:,5,colonnes2] = dNdx

        if verification:
            list_B_rigi_e_pg = []

            for e in listElement:
                list_B_rigi_pg = []
                for pg in listPg:
                    if dim == 2:
                        B_rigi_pg = np.zeros((3, nPe*dim))
                        colonne = 0
                        dN = self.dN_e_pg[e][pg]
                        for n in range(nPe):
                            dNdx = dN[0, n]
                            dNdy = dN[1, n]
                            
                            # B rigi
                            B_rigi_pg[0, colonne] = dNdx
                            B_rigi_pg[1, colonne+1] = dNdy
                            B_rigi_pg[2, colonne] = dNdy; B_rigi_pg[2, colonne+1] = dNdx
                            
                            colonne += 2
                        list_B_rigi_pg.append(B_rigi_pg)    
                    else:
                        B_rigi_pg = np.zeros((6, nPe*dim))
                        
                        colonne = 0
                        for n in range(nPe):
                            dNdx = dN[0, n]
                            dNdy = dN[1, n]
                            dNdz = dN[2, n]                        
                            
                            B_rigi_pg[0, colonne] = dNdx
                            B_rigi_pg[1, colonne+1] = dNdy
                            B_rigi_pg[2, colonne+2] = dNdz
                            B_rigi_pg[3, colonne] = dNdy; B_rigi_pg[3, colonne+1] = dNdx
                            B_rigi_pg[4, colonne+1] = dNdz; B_rigi_pg[4, colonne+2] = dNdy
                            B_rigi_pg[5, colonne] = dNdz; B_rigi_pg[5, colonne+2] = dNdx
                            colonne += 3
                        list_B_rigi_pg.append(B_rigi_pg)
                        
                    
                list_B_rigi_e_pg.append(list_B_rigi_pg)
            
                test = np.array(list_B_rigi_e_pg)-self.B_rigi_e_pg
                assert test.max() == 0 and test.min() == 0, "Erreur dans la construiction de B"
            
        tic.Tac("Construit les matrices EF", self.__verbosity)
      
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