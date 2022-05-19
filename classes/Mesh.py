
from typing import cast
import numpy as np
import scipy.sparse as sp

from Geom import *
from GroupElem import GroupElem
from TicTac import TicTac

class Mesh:

    def __init__(self, dim: int, dict_groupElem: dict, verbosity=True):
        """Création du maillage depuis coordo et connection
        Le maillage est l'entité qui possède les groupes d'élements
        
        affichageMaillage : bool, optional
            Affichage après la construction du maillage, by default True
        """

        # Onrevifie que l'on contient que des GroupElem
        for item in dict_groupElem.values():
            assert isinstance(item, GroupElem)

        self.__dim = dim

        self.__dict_groupElem = dict_groupElem

        self.__verbosity = verbosity
        """le maillage peut ecrire dans la console"""
        
        if self.__verbosity:
            print(f"\nType d'elements: {self.elemType}")
            print(f"Ne = {self.Ne}, Nn = {self.Nn}, nbDdl = {self.Nn*self.__dim}")
    
    
    def get_groupElem(self, dim=None):
        if dim != None:
            return cast(GroupElem, self.__dict_groupElem[dim])
        else:
            return cast(GroupElem, self.__dict_groupElem[self.__dim])
    groupElem = cast(GroupElem, property(get_groupElem))   

    def Get_Nodes_Conditions(self, conditionX=True, conditionY=True, conditionZ=True):
        """Renvoie la liste de noeuds qui respectent les condtions

        Args:
            conditionX (bool, optional): Conditions suivant x. Defaults to True.
            conditionY (bool, optional): Conditions suivant y. Defaults to True.
            conditionZ (bool, optional): Conditions suivant z. Defaults to True.

        Exemples de contitions:
            x ou toto ça n'a pas d'importance
            condition = lambda x: x < 40 and x > 20
            condition = lambda x: x == 40
            condition = lambda x: x >= 0

        Returns:
            list(int): lite des noeuds qui respectent les conditions
        """
        return self.groupElem.Get_Nodes_Conditions(conditionX, conditionY, conditionZ)
    
    def Get_Nodes_Point(self, line: Line):
        """Renvoie la liste le noeud sur le point"""
        return self.groupElem.Get_Nodes_Point(line)

    def Get_Nodes_Line(self, line: Line):
        """Renvoie la liste de noeuds qui sont sur la ligne"""
        return self.groupElem.Get_Nodes_Line(line)

    def Get_Nodes_Domain(self, domain: Domain):
        """Renvoie la liste de noeuds qui sont dans le domaine"""
        return self.groupElem.Get_Nodes_Domain(domain)

    def Localises_sol_e(self, sol: np.ndarray):
        """sur chaque elements on récupère les valeurs de sol"""
        return self.groupElem.Localise_sol_e(sol)
        
    def __get_Ne(self, dim=None):
        if isinstance(dim,int):
            return cast(GroupElem, self.groupElem(dim)).Ne 
        else:
            return self.groupElem.Ne
    Ne = property(__get_Ne)
    """Nombre d'élements du maillage"""
    
    def __get_Nn(self, dim=None):        
        if isinstance(dim,int):
            return cast(GroupElem, self.groupElem(dim)).Nn 
        else:
            return self.groupElem.Nn
    Nn = property(__get_Nn)
    """Nombre de noeuds du maillage"""

    def __get_nPe(self, dim=None):
        if isinstance(dim,int):
            return cast(GroupElem, self.groupElem(dim)).nPe
        else:
            return self.groupElem.nPe
    nPe = property(__get_nPe)
    """noeuds par element"""

    def __get_dim(self):
        return self.__dim
    dim = property(__get_dim)
    """Dimension du maillage"""

    def __get_coordo(self, dim=None):
        if isinstance(dim, int):
            return cast(GroupElem, self.groupElem(dim)).coordo
        else:
            return self.groupElem.coordo
    coordo = cast(np.ndarray, property(__get_coordo))
    """matrice des coordonnées de noeuds (Nn,3)"""

    def __get_connect(self, dim=None):
        if isinstance(dim, int):
            return cast(GroupElem, self.groupElem(dim)).connect
        else:
            return self.groupElem.connect        
    connect = cast(np.ndarray, property(__get_connect))
    """connection des elements (Ne, nPe)"""
    
    def __get_connect_n_e(self):
        return self.groupElem.connect_n_e
    connect_n_e = cast(sp.csr_matrix, property(__get_connect_n_e))
    """matrices de 0 et 1 avec les 1 lorsque le noeud possède l'element (Nn, Ne)\n
        tel que : valeurs_n(Nn,1) = connect_n_e(Nn,Ne) * valeurs_e(Ne,1)"""

    def __get_assembly(self):
        return self.groupElem.assembly_e
    assembly_e = cast(np.ndarray, property(__get_assembly))
    """matrice d'assemblage (Ne, nPe*dim)"""

    def __get_lignesVector_e(self):
        return np.repeat(self.assembly_e, self.nPe*self.__dim).reshape((self.Ne,-1))
    lignesVector_e = cast(np.ndarray, property(__get_lignesVector_e))
    """lignes pour remplir la matrice d'assemblage en vecteur (déplacement)"""

    def __get_colonnesVector_e(self):
        return np.repeat(self.assembly_e, self.nPe*self.__dim, axis=0).reshape((self.Ne,-1))
    colonnesVector_e = cast(np.ndarray, property(__get_colonnesVector_e))
    """colonnes pour remplir la matrice d'assemblage en vecteur (déplacement)"""

    def __get_lignesScalar_e(self):
        return np.repeat(self.connect, self.nPe).reshape((self.Ne,-1))         
    lignesScalar_e = cast(np.ndarray, property(__get_lignesScalar_e))
    """lignes pour remplir la matrice d'assemblage en scalaire (endommagement, ou thermique)"""

    def __get_colonnesScalar_e(self):
        return np.repeat(self.connect, self.nPe, axis=0).reshape((self.Ne,-1))
    colonnesScalar_e = cast(np.ndarray, property(__get_colonnesScalar_e))
    """colonnes pour remplir la matrice d'assemblage en scalaire (endommagement, ou thermique)"""

    def get_nPg(self, matriceType: str):
        """nombre de point d'intégration par élement"""
        return self.groupElem.get_gauss(matriceType).nPg

    def get_poid_pg(self, matriceType: str):
        """Points d'intégration (pg, dim, poid)"""
        return self.groupElem.get_gauss(matriceType).poids

    def get_jacobien_e_pg(self, matriceType: str):
        """jacobien (e, pg)"""
        return self.groupElem.get_jacobien_e_pg(matriceType)
    
    def get_N_scalaire_pg(self, matriceType: str):
        """Fonctions de formes dans l'element isoparamétrique pour un scalaire (npg, 1, npe)
        Matrice des fonctions de forme dans element de référence (ksi, eta)\n
        [N1(ksi,eta) N2(ksi,eta) Nn(ksi,eta)] \n
        """
        return self.groupElem.get_N_pg(matriceType)

    def get_N_vecteur_pg(self, matriceType: str):
        """Fonctions de formes dans l'element de reférences pour un vecteur (npg, dim, npe*dim)
        Matrice des fonctions de forme dans element de référence (ksi, eta)\n
        [N1(ksi,eta) 0 N2(ksi,eta) 0 Nn(ksi,eta) 0 \n
        0 N1(ksi,eta) 0 N2(ksi,eta) 0 Nn(ksi,eta)]"""
        return self.groupElem.get_N_pg(matriceType, self.__dim)

    def get_B_sclaire_e_pg(self, matriceType: str):
        """Derivé des fonctions de formes dans la base réele en sclaire\n
        [dN1,x dN2,x dNn,x\n
        dN1,y dN2,y dNn,y]\n        
        """
        return self.groupElem.get_dN_e_pg(matriceType)

    def get_B_dep_e_pg(self, matriceType: str):
        """Derivé des fonctions de formes dans la base réele pour le problème de déplacement (e, pg, (3 ou 6), nPe*dim)\n
        exemple en 2D :\n
        [dN1,x 0 dN2,x 0 dNn,x 0\n
        0 dN1,y 0 dN2,y 0 dNn,y\n
        dN1,y dN1,x dN2,y dN2,x dN3,y dN3,x]
        """         
        dN_e_pg = self.get_B_sclaire_e_pg(matriceType)

        nPg = self.get_nPg(matriceType)
        nPe = self.nPe
        dim = self.__dim
        listnPe = np.arange(nPe)
        
        colonnes0 = np.arange(0, nPe*dim, dim)
        colonnes1 = np.arange(1, nPe*dim, dim)

        if self.__dim == 2:
            B_e_pg = np.array([[np.zeros((3, nPe*dim))]*nPg]*self.Ne)
            """Derivé des fonctions de formes dans la base réele en vecteur \n
            """
            
            dNdx = dN_e_pg[:,:,0,listnPe]
            dNdy = dN_e_pg[:,:,1,listnPe]

            B_e_pg[:,:,0,colonnes0] = dNdx
            B_e_pg[:,:,1,colonnes1] = dNdy
            B_e_pg[:,:,2,colonnes0] = dNdy; B_e_pg[:,:,2,colonnes1] = dNdx
        else:
            B_e_pg = np.array([[np.zeros((6, nPe*dim))]*nPg]*self.Ne)

            dNdx = dN_e_pg[:,:,0,listnPe]
            dNdy = dN_e_pg[:,:,1,listnPe]
            dNdz = dN_e_pg[:,:,2,listnPe]

            colonnes2 = np.arange(2, nPe*dim, dim)

            B_e_pg[:,:,0,colonnes0] = dNdx
            B_e_pg[:,:,1,colonnes1] = dNdy
            B_e_pg[:,:,2,colonnes2] = dNdz
            B_e_pg[:,:,3,colonnes1] = dNdz; B_e_pg[:,:,3,colonnes2] = dNdy
            B_e_pg[:,:,4,colonnes0] = dNdz; B_e_pg[:,:,4,colonnes2] = dNdx
            B_e_pg[:,:,5,colonnes0] = dNdy; B_e_pg[:,:,5,colonnes1] = dNdx

        return B_e_pg    
    
    def get_nbFaces(self):
        return self.groupElem.nbFaces
    
    def __get_elemenType(self):
        return self.groupElem.elemType
    elemType = cast(str, property(__get_elemenType))

    def get_connectTriangle(self):
        """Transforme la matrice de connectivité pour la passer dans le trisurf en 2D"""
        return self.groupElem.get_connectTriangle()
    
    def get_connect_Faces(self):
        """Récupère les faces de chaque element
        """
        return self.groupElem.get_connect_Faces()
        

# TEST ==============================

import unittest
import os

class Test_Mesh(unittest.TestCase):
    
    def setUp(self):
        
        from Interface_Gmsh import Interface_Gmsh

        self.list_Mesh2D = Interface_Gmsh.Construction2D()

    def test_ConstructionMatrices(self):
        for mesh in self.list_Mesh2D:
            self.__VerficiationConstructionMatrices(mesh)

    # Verifivation
    def __VerficiationConstructionMatrices(self, mesh: Mesh):

        dim = mesh.dim
        connect = mesh.connect
        listElement = range(mesh.Ne)
        listPg = np.arange(mesh.get_nPg("rigi"))
        nPe = connect.shape[1]

        # Verification assemblage
        assembly_e_test = np.array([[int(n * dim + d)for n in connect[e] for d in range(dim)] for e in listElement])
        testAssembly = np.testing.assert_array_almost_equal(mesh.assembly_e, assembly_e_test, verbose=False)
        self.assertIsNone(testAssembly)

        # Verification lignes_e 
        lignes_e_test = np.array([[i for i in mesh.assembly_e[e] for j in mesh.assembly_e[e]] for e in listElement])
        testLignes = np.testing.assert_array_almost_equal(lignes_e_test, mesh.lignesVector_e, verbose=False)
        self.assertIsNone(testLignes)

        # Verification lignes_e 
        colonnes_e_test = np.array([[j for i in mesh.assembly_e[e] for j in mesh.assembly_e[e]] for e in listElement])
        testColonnes = np.testing.assert_array_almost_equal(colonnes_e_test, mesh.colonnesVector_e, verbose=False)
        self.assertIsNone(testColonnes)

        list_B_rigi_e_pg = []

        for e in listElement:
            list_B_rigi_pg = []
            for pg in listPg:
                if dim == 2:
                    B_dep_pg = np.zeros((3, nPe*dim))
                    colonne = 0
                    B_sclaire_e_pg = mesh.get_B_sclaire_e_pg("rigi")
                    dN = B_sclaire_e_pg[e,pg]
                    for n in range(nPe):
                        dNdx = dN[0, n]
                        dNdy = dN[1, n]
                        
                        # B rigi
                        B_dep_pg[0, colonne] = dNdx
                        B_dep_pg[1, colonne+1] = dNdy
                        B_dep_pg[2, colonne] = dNdy; B_dep_pg[2, colonne+1] = dNdx
                        
                        colonne += 2
                    list_B_rigi_pg.append(B_dep_pg)    
                else:
                    B_dep_pg = np.zeros((6, nPe*dim))
                    
                    colonne = 0
                    for n in range(nPe):
                        dNdx = dN[0, n]
                        dNdy = dN[1, n]
                        dNdz = dN[2, n]                        
                        
                        B_dep_pg[0, colonne] = dNdx
                        B_dep_pg[1, colonne+1] = dNdy
                        B_dep_pg[2, colonne+2] = dNdz
                        B_dep_pg[3, colonne] = dNdy; B_dep_pg[3, colonne+1] = dNdx
                        B_dep_pg[4, colonne+1] = dNdz; B_dep_pg[4, colonne+2] = dNdy
                        B_dep_pg[5, colonne] = dNdz; B_dep_pg[5, colonne+2] = dNdx
                        colonne += 3
                    list_B_rigi_pg.append(B_dep_pg)
                    
                
            list_B_rigi_e_pg.append(list_B_rigi_pg)

        B_rigi_e_pg = mesh.get_B_dep_e_pg("rigi")

        testB_rigi = np.testing.assert_array_almost_equal(np.array(list_B_rigi_e_pg), B_rigi_e_pg, verbose=False)
        self.assertIsNone(testB_rigi)

            


if __name__ == '__main__':        
    try:
        os.system("cls")
        unittest.main(verbosity=2)
    except:
        print("")        