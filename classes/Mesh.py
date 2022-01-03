import numpy as np

try:
    from Noeud import Noeud
    from Element import Element
    from Materiau import Materiau
    from TicTac import TicTac
except:
    from classes.Noeud import Noeud
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

        TicTac.Tic()

        self.__dim = dim

        self.__verbosity = verbosity

        self.coordo = np.array(coordo)
        # self.coordo = coordo
        self.connect = connect 
        
        listElement = range(self.Ne)

        listNoeud = range(self.Nn)

        # Construit la matrice de connection pour les noeuds
        self.connectNoeuds = [[e for e in listElement if n in self.connect[e]] for n in listNoeud]

        # Construit la matrice assembly
        self.listAssembly = [[int(n * dim + d)for n in connect[e] for d in range(dim)] for e in listElement]
        
        self.element = Element(dim, len(connect[0]))
        
        self.list_poid_pg = self.element.listPoid_pg
        # ligne -> elements et colonne -> points de gauss
        self.list_F_e_pg = [] 
        self.list_invF_e_pg = []
        self.list_jacobien_e_pg = []
        self.list_dN_e_pg = []
        self.list_B_rigi_e_pg = []
        self.list_B_mass_e_pg = []

        xi_yi = coordo[:,[0, 1]]
        for e in range(self.Ne):
            list_F_pg = []
            list_invF_pg = []
            list_invF_pg = []
            list_jacobien_pg = []
            list_dN_pg = []
            for pg in range(self.element.nPg):
                F = self.element.listdN_pg[pg].dot(xi_yi[connect[e], :])
                list_F_pg.append(F)

                invF = np.linalg.inv(F)
                list_invF_pg.append(invF)

                jacobien = np.linalg.det(F)
                list_jacobien_pg.append(jacobien)

                dN = invF.dot(self.element.listdN_pg[pg])
                list_dN_pg.append(dN)

                list_B_rigi_pg = []
                list_B_mass_pg = []

                nPe = self.element.nPe
                if dim == 2:
                    B_rigi_pg = np.zeros((3, nPe*dim))
                    B_mass_pg = np.zeros((dim, nPe))
                    colonne = 0
                    for n in range(nPe):
                        dNdx = dN[0, n]
                        dNdy = dN[1, n]
                        
                        # B rigi
                        B_rigi_pg[0, colonne] = dNdx
                        B_rigi_pg[1, colonne+1] = dNdy
                        B_rigi_pg[2, colonne] = dNdy; B_rigi_pg[2, colonne+1] = dNdx    

                        # B mass
                        B_mass_pg[0,n] = dNdx
                        B_mass_pg[1,n] = dNdy

                        colonne += 2
                    list_B_rigi_pg.append(B_rigi_pg)
                    list_B_mass_pg.append(B_mass_pg)
                else:
                    B_rigi_pg = np.zeros((6, nPe*dim))
                    B_mass_pg = np.zeros((dim, nPe))

                    colonne = 0
                    for n in range(nPe):
                        dNdx = dN[0, n]
                        dNdy = dN[1, n]
                        dNdz = dN[2, n]
                        
                        # B rigi
                        B_rigi_pg[0, colonne] = dNdx
                        B_rigi_pg[1, colonne+1] = dNdy
                        B_rigi_pg[2, colonne+2] = dNdz
                        B_rigi_pg[3, colonne] = dNdy; B_rigi_pg[3, colonne+1] = dNdx
                        B_rigi_pg[4, colonne+1] = dNdz; B_rigi_pg[4, colonne+2] = dNdy
                        B_rigi_pg[5, colonne] = dNdz; B_rigi_pg[5, colonne+2] = dNdx
                        colonne += 3

                        # B mass
                        B_mass_pg[0,n] = dNdx
                        B_mass_pg[1,n] = dNdy
                        B_mass_pg[2,n] = dNdz

                    list_B_rigi_pg.append(B_rigi_pg)
                    list_B_mass_pg.append(B_mass_pg)
                        
            self.list_F_e_pg.append(list_F_pg)
            self.list_invF_e_pg.append(list_invF_pg)
            self.list_jacobien_e_pg.append(list_jacobien_pg)
            self.list_dN_e_pg.append(list_dN_pg)
            self.list_B_rigi_e_pg.append(list_B_rigi_pg)
            self.list_B_mass_e_pg.append(list_B_mass_pg)

         

        # Creation de l'élement
        # L'element permet de calculer les fonctions de formes et ses dérivées

        self.__connectPourTriangle = []
        self.__connectPolygon =[]

        
        t = TicTac.Tac("Importation du maillage", self.__verbosity)
    
    # def ChercheNoeuds(self, CondX=[], CondY=[], CondZ=[]):
        
    #     assert self.__dim == 2 and len(CondZ) == 0, "Pas de condition suivant Z dans une étude 2D" 

    #     def Conditions(i, coordonnée: float, valeur: float):
    #         switcher ={
    #                 "=": np.isclose(coordonnée, valeur),
    #                 "<": coordonnée < valeur,
    #                 "<=": coordonnée < valeur or np.isclose(coordonnée, valeur),
    #                 ">": coordonnée > valeur,
    #                 ">=": coordonnée > valeur or np.isclose(coordonnée, valeur),
    #             }
    #         return switcher.get(i, "Invalid")

    #     noeuds = []

    #     for n in self.noeuds:
    #         n = cast(Noeud, n)
    #         conditions = [CondX, CondY, CondZ]
    #         tests = []
    #         for c in range(len(conditions)):
    #             cond = conditions[c]
    #             if(len(cond)==0):
    #                 tests.append(True)
    #             else:
    #                 tests.append(Conditions(cond[0], n.coordo[c], cond[1]))
            
    #         if not False in tests:
    #             noeuds.append(n)

    #     return noeuds

            
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