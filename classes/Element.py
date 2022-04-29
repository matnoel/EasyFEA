
from typing import cast
from TicTac import TicTac
import numpy as np

from Gauss import Gauss

class ElementIsoparametrique:

    @staticmethod
    def get_Types2D():
        """type d'elements disponibles en 2D"""
        liste2D = ["TRI3", "TRI6", "QUAD4", "QUAD8"]
        return liste2D
    
    @staticmethod
    def get_Types3D():
        """type d'elements disponibles en 3D"""
        liste3D = ["TETRA4"]
        return liste3D
    
    @staticmethod
    def get_MatriceType():
        liste = ["rigi", "masse"]
        return liste

    def __get_ElementType(self):
        """Renvoie le type de l'élément en fonction du nombre de noeuds par élement
        """        
        if self.__dim == 2:        
            switch = {
                3 : "TRI3",
                6 : "TRI6",
                4 : "QUAD4",
                8 : "QUAD8",
            }                
            return switch[self.nPe]
        if self.__dim == 3:
            switch = {
                4 : "TETRA4",                
            }

        return switch[self.nPe]
    type = cast(str, property(__get_ElementType)) 
    """type de l'élement"""

    def get_nPg(self, matriceType: str):
        """nombre de points d'intégrations"""
        assert matriceType in ElementIsoparametrique.get_MatriceType()

        if matriceType not in self.__dict_gauss.keys():
            self.__Set_Matrices_ElemIso(matriceType)
            
        return int(cast(Gauss, self.__dict_gauss[matriceType]).nPg)

    def __get_nPe(self):
        return self.__nPe
    nPe = property(__get_nPe)
    """Noeuds par elements"""

    def get_poid_pg(self, matriceType: str):
        """poids des points d'intégrations"""
        assert matriceType in ElementIsoparametrique.get_MatriceType()
        if matriceType not in self.__dict_gauss.keys():
            self.__Set_Matrices_ElemIso(matriceType)
            
        return cast(Gauss, self.__dict_gauss[matriceType]).poids

    def get_jacobien_e_pg(self, nodes_e: np.ndarray, matriceType: str):
        """jacobiens aux pts d'integration de chaque element"""
        assert matriceType in ElementIsoparametrique.get_MatriceType()
        if matriceType not in self.__dict_jacobien_e_pg.keys():
            self.__Set_DetJ(nodes_e, matriceType)
            
        return cast(np.ndarray, self.__dict_jacobien_e_pg[matriceType])

    def get_N_scalaire_pg(self, matriceType: str):
        assert matriceType in ElementIsoparametrique.get_MatriceType()
        if matriceType not in self.__dict_N_scalaire_pg.keys():
            self.__Set_Matrices_ElemIso(matriceType)
    
        return cast(np.ndarray, self.__dict_N_scalaire_pg[matriceType])
    
    def get_N_vecteur_pg(self, matriceType: str):
        assert matriceType in ElementIsoparametrique.get_MatriceType()
        if matriceType not in self.__dict_N_vecteur_pg.keys():
            self.__Set_Matrices_ElemIso(matriceType)
    
        return cast(np.ndarray, self.__dict_N_vecteur_pg[matriceType])

    def get_dN_e_pg(self, nodes_e: np.ndarray, matriceType: str):
        assert matriceType in ElementIsoparametrique.get_MatriceType()
        return self.__get_dN_e_pg(nodes_e, matriceType)

    def __init__(self, dim: int, nPe: int, verbosity=False):
        """Constructeur d'element, on construit Be et le jacobien !

        Parameters
        ----------
        dim : int
            Numéro de l'élement (>=0)
        nPe : int
            Nombre de noeud par element
        """
        assert dim in [2,3], "Dimesion compris entre 2D et 3D"
        
        # Création des variables de la classe        
        self.__dim = dim
        """dimension de l'élement [2,3]"""
        
        self.__nPe = nPe
        """noeuds par élément"""

        self.__verbosity = verbosity

        # Pour chaque type de matrice élementaire (masse, rigi), on va construire
        # leurs matrices évaluées au pts de gaus
        self.__dict_N_scalaire_pg = {}
        """Fonctions de formes scalaires pour chaque type (masse, rigi...)\n
        "type" : (pg, 1, nPe) : \n
            [Ni . . . Nn]"""        
        self.__dict_N_vecteur_pg = {}
        """Fonctions de formes scalaires pour chaque type (masse, rigi...)\n
        "type" : (pg, dim, nPe*dim) : \n
        [Ni 0 . . . Nn 0 \n
        0 Ni . . . 0 Nn]"""        
        self.__dict_dN_pg = {}
        """ Dérivées des fonctions de formes dans l'element de référence pour chaque type (masse, rigi...)\n
        "type" : (pg, dim, nPe) : \n
        [Ni,ksi . . . Nn,ksi \n
        Ni,eta . . . Nn,eta]"""        
        self.__dict_gauss = {}
        """Points de gauss pour chaque type (masse, rigi...)"""
        self.__dict_F_e_pg = {}
        """Matrice jacobienne pour chaque type (masse, rigi...)"""
        self.__dict_invF_e_pg = {}
        """Inverse Matrice jacobienne pour chaque type (masse, rigi...)"""
        self.__dict_jacobien_e_pg = {}
        """jacobien"""
        self.__dict_dN_e_pg = {}
        """Derivé des fonctions de formes dans la base réele pour chaque type (masse, rigi...)"""
    
    def __get_dN_e_pg(self, nodes_e :np.ndarray, matriceType: str):
        assert matriceType in ElementIsoparametrique.get_MatriceType()

        if matriceType not in self.__dict_dN_e_pg.keys():

            self.__Set_DetJ(nodes_e, matriceType)

            invF_e_pg = self.__dict_invF_e_pg[matriceType]
            
            dN_pg = self.__dict_dN_pg[matriceType]

            # Derivé des fonctions de formes dans la base réele
            dN_e_pg = np.array(np.einsum('epik,pkj->epij', invF_e_pg, dN_pg, optimize=True))
            self.__dict_dN_e_pg[matriceType] = dN_e_pg

        return self.__dict_dN_e_pg[matriceType]
    
    def __Set_DetJ(self, nodes_e: np.ndarray, matriceType: str):
        """Construit les matrices de changement de base si nécessaire"""
        assert matriceType in ElementIsoparametrique.get_MatriceType()

        # Construit les matrices pour le changement de base si nécessaire
        if matriceType not in self.__dict_invF_e_pg.keys():

            tic = TicTac()

            if matriceType not in self.__dict_dN_pg.keys():
                self.__Set_Matrices_ElemIso(matriceType)
            
            dN_pg = self.__dict_dN_pg[matriceType]

            # Matrice jacobienne
            F_e_pg = np.array(np.einsum('pik,ekj->epij', dN_pg, nodes_e, optimize=True))
            self.__dict_F_e_pg[matriceType] = F_e_pg
            
            # Inverse Matrice jacobienne
            invF_e_pg = np.array(np.linalg.inv(F_e_pg))
            self.__dict_invF_e_pg[matriceType] = invF_e_pg
            
            # jacobien
            jacobien_e_pg = np.array(np.linalg.det(F_e_pg))
            self.__dict_jacobien_e_pg[matriceType] = jacobien_e_pg

            tic.Tac("Matrices", f"Calcul des matrices de changement de base de type : {matriceType}", self.__verbosity)
        
    def __Set_Matrices_ElemIso(self, matriceType: str):
        """Construit les fonctions de forme et leur dérivée pour l'element de référence et le type de matrice"""
        
        # Avec : dN = [Ni,ksi . . . Nn,ksi
        #              Ni,eta . . . Nn,eta]
        # Dérivées des fonctions de formes dans l'element de référence (pg, dim, nPe), dans la base (ksi, eta ...)

        
        # N_vecteur_pg = [Ni 0 . . . Nn 0
        #                 0 Ni . . . 0 Nn]
        # Fonctions de formes vectorielles (pg, dim, nPe*dim), dans la base (ksi, eta ...)

        # N_vecteur_pg = [Ni . . . Nn]
        # Fonctions de formes scalaires (pg, 1, nPe), dans la base (ksi, eta ...)

        tic = TicTac()

        if self.__dim == 2:

            if self.type in ["TRI3", "TRI6"]:                
                N_scalaire_pg, N_vecteur_pg, dN_pg, gauss = self.__Build_Matrices_ElemIso_Triangle(matriceType)

            elif self.type in ["QUAD4", "QUAD8"]:
                N_scalaire_pg, N_vecteur_pg, dN_pg, gauss = self.__Build_Matrices_ElemIso_Quadrangle(matriceType)

        elif self.__dim == 3:

            if self.type in ["TETRA4"]:
                N_scalaire_pg, N_vecteur_pg, dN_pg, gauss = self.__Build_Matrices_ElemIso_Tetraedre(matriceType)

        # Sauvegarde les valeurs
        self.__dict_N_scalaire_pg[matriceType] = N_scalaire_pg
        self.__dict_N_vecteur_pg[matriceType] = N_vecteur_pg
        self.__dict_dN_pg[matriceType] = dN_pg
        self.__dict_gauss[matriceType] = gauss

        tic.Tac("Matrices",f"Calcul des matrices de l'element de référence de type : {matriceType}", self.__verbosity)

    def __Matrices_Fonctions_De_Forme_Iso(self, Ntild: np.ndarray, gauss: Gauss):

        # TODO A optimiser

        coord = gauss.coord

        nbtild = len(Ntild)

        dim = self.__dim

        nPg = gauss.nPg

        Nt_scalaire_pg = np.zeros((nPg, 1, nbtild))
        Nt_vecteur_pg = np.zeros((nPg, self.__dim, nbtild*dim))

        for pg in range(nPg):

            for n, Nt in enumerate(Ntild):
                
                if dim == 2:
                    ksi = coord[pg, 0]
                    eta = coord[pg, 1]
                    Nt = Nt(ksi, eta)
                else:
                    x = coord[pg, 0]
                    y = coord[pg, 1]
                    z = coord[pg, 2]
                    Nt = Nt(x, y, z)

                Nt_scalaire_pg[pg, 0, n] = Nt

                for d in range(dim):
                    Nt_vecteur_pg[pg, d, n*dim+d] = Nt

        return Nt_scalaire_pg, Nt_vecteur_pg

    def __Matrices_Dérivées_Fonctions_De_Forme_Iso(self, dNtild: np.ndarray, gauss: Gauss):
        
        # TODO A optimiser

        coord = gauss.coord

        nbtild = len(dNtild)

        dim = self.__dim

        nPg = gauss.nPg

        dNt_pg = np.zeros((nPg, dim, nbtild))

        for pg in range(nPg):

            for n, Nt in enumerate(dNtild):

                if dim == 2:
                    ksi = coord[pg, 0]
                    eta = coord[pg, 1]

                    dN_ksi = Nt[0](ksi, eta)
                    dN_eta = Nt[1](ksi, eta)

                    dNt_pg[pg, 0, n] = dN_ksi
                    dNt_pg[pg, 1, n] = dN_eta

                else:
                    x = coord[pg, 0]
                    y = coord[pg, 1]
                    z = coord[pg, 2]

                    dN_x = Nt[0](x, y, z)
                    dN_y = Nt[1](x, y, z)
                    dN_z = Nt[2](x, y, z)

                    dNt_pg[pg, 0, n] = dN_x
                    dNt_pg[pg, 1, n] = dN_y
                    dNt_pg[pg, 2, n] = dN_z                    

        return dNt_pg
        
    def __Build_Matrices_ElemIso_Triangle(self, matriceType: str):

        gauss = Gauss(self.type, matriceType)

        # TRI3
        if self.nPe == 3:

            N1t = lambda ksi,eta: 1-ksi-eta
            N2t = lambda ksi,eta: ksi
            N3t = lambda ksi,eta: eta
            
            Ntild = np.array([N1t, N2t, N3t])

            Nt_scalaire_pg, Nt_vecteur_pg = self.__Matrices_Fonctions_De_Forme_Iso(Ntild, gauss)
            
            dN1t = [lambda ksi,eta: -1, lambda ksi,eta:-1]
            dN2t = [lambda ksi,eta: 1,  lambda ksi,eta: 0]
            dN3t = [lambda ksi,eta: 0,  lambda ksi,eta: 1]

            dNtild = np.array([dN1t, dN2t, dN3t])

            dN_pg = self.__Matrices_Dérivées_Fonctions_De_Forme_Iso(dNtild, gauss)

        # TRI6  
        if self.nPe == 6:
            
            N1t = lambda ksi,eta: -(1-ksi-eta)*(1-2*(1-ksi-eta))
            N2t = lambda ksi,eta: -ksi*(1-2*ksi)
            N3t = lambda ksi,eta: -eta*(1-2*eta)
            N4t = lambda ksi,eta: 4*ksi*(1-ksi-eta)
            N5t = lambda ksi,eta: 4*ksi*eta
            N6t = lambda ksi,eta: 4*eta*(1-ksi-eta)
            
            Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t])
                
            Nt_scalaire_pg, Nt_vecteur_pg = self.__Matrices_Fonctions_De_Forme_Iso(Ntild, gauss)

            dN1t = [lambda ksi,eta: 4*ksi+4*eta-3,  lambda ksi,eta: 4*ksi+4*eta-3]
            dN2t = [lambda ksi,eta: 4*ksi-1,        lambda ksi,eta: 0]
            dN3t = [lambda ksi,eta: 0,              lambda ksi,eta: 4*eta-1]
            dN4t = [lambda ksi,eta: 4-8*ksi-4*eta,  lambda ksi,eta: -4*ksi]
            dN5t = [lambda ksi,eta: 4*eta,          lambda ksi,eta: 4*ksi]
            dN6t = [lambda ksi,eta: -4*eta,         lambda ksi,eta: 4-4*ksi-8*eta]
            
            dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t])

            dN_pg = self.__Matrices_Dérivées_Fonctions_De_Forme_Iso(dNtild, gauss)

        return Nt_scalaire_pg, Nt_vecteur_pg, dN_pg, gauss

    def __Build_Matrices_ElemIso_Quadrangle(self, matriceType: str):

        gauss = Gauss(self.type, matriceType)
        
        if self.nPe == 4:
            
            N1t = lambda ksi,eta: (1-ksi)*(1-eta)/4
            N2t = lambda ksi,eta: (1+ksi)*(1-eta)/4
            N3t = lambda ksi,eta: (1+ksi)*(1+eta)/4
            N4t = lambda ksi,eta: (1-ksi)*(1+eta)/4
            
            Ntild = np.array([N1t, N2t, N3t, N4t])

            Nt_scalaire_pg, Nt_vecteur_pg = self.__Matrices_Fonctions_De_Forme_Iso(Ntild, gauss)
            
            dN1t = [lambda ksi,eta: (eta-1)/4,  lambda ksi,eta: (ksi-1)/4]
            dN2t = [lambda ksi,eta: (1-eta)/4,  lambda ksi,eta: (-ksi-1)/4]
            dN3t = [lambda ksi,eta: (1+eta)/4,  lambda ksi,eta: (1+ksi)/4]
            dN4t = [lambda ksi,eta: (-eta-1)/4, lambda ksi,eta: (1-ksi)/4]
            
            dNtild = [dN1t, dN2t, dN3t, dN4t]
            
            dN_pg = self.__Matrices_Dérivées_Fonctions_De_Forme_Iso(dNtild, gauss)
              
        elif self.nPe ==8:
            
            N1t = lambda ksi,eta: (1-ksi)*(1-eta)*(-1-ksi-eta)/4
            N2t = lambda ksi,eta: (1+ksi)*(1-eta)*(-1+ksi-eta)/4
            N3t = lambda ksi,eta: (1+ksi)*(1+eta)*(-1+ksi+eta)/4
            N4t = lambda ksi,eta: (1-ksi)*(1+eta)*(-1-ksi+eta)/4
            N5t = lambda ksi,eta: (1-ksi**2)*(1-eta)/2
            N6t = lambda ksi,eta: (1+ksi)*(1-eta**2)/2
            N7t = lambda ksi,eta: (1-ksi**2)*(1+eta)/2
            N8t = lambda ksi,eta: (1-ksi)*(1-eta**2)/2
            
            Ntild =  np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t])

            Nt_scalaire_pg, Nt_vecteur_pg = self.__Matrices_Fonctions_De_Forme_Iso(Ntild, gauss)
                

            dN1t = [lambda ksi,eta: (1-eta)*(2*ksi+eta)/4,      lambda ksi,eta: (1-ksi)*(ksi+2*eta)/4]
            dN2t = [lambda ksi,eta: (1-eta)*(2*ksi-eta)/4,      lambda ksi,eta: -(1+ksi)*(ksi-2*eta)/4]
            dN3t = [lambda ksi,eta: (1+eta)*(2*ksi+eta)/4,      lambda ksi,eta: (1+ksi)*(ksi+2*eta)/4]
            dN4t = [lambda ksi,eta: -(1+eta)*(-2*ksi+eta)/4,    lambda ksi,eta: (1-ksi)*(-ksi+2*eta)/4]
            dN5t = [lambda ksi,eta: -ksi*(1-eta),               lambda ksi,eta: -(1-ksi**2)/2]
            dN6t = [lambda ksi,eta: (1-eta**2)/2,               lambda ksi,eta: -eta*(1+ksi)]
            dN7t = [lambda ksi,eta: -ksi*(1+eta),               lambda ksi,eta: (1-ksi**2)/2]
            dN8t = [lambda ksi,eta: -(1-eta**2)/2,              lambda ksi,eta: -eta*(1-ksi)]
                            
            dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t])

            dN_pg = self.__Matrices_Dérivées_Fonctions_De_Forme_Iso(dNtild, gauss)

        return Nt_scalaire_pg, Nt_vecteur_pg, dN_pg, gauss
            
    def __Build_Matrices_ElemIso_Tetraedre(self, matriceType: str):

        gauss = Gauss(self.type, matriceType)

        if self.nPe == 4:

            N1t = lambda x,y,z: 1-x-y-z
            N2t = lambda x,y,z: x
            N3t = lambda x,y,z: y
            N4t = lambda x,y,z: z

            Ntild = np.array([N1t, N2t, N3t, N4t])

            Nt_scalaire_pg, Nt_vecteur_pg = self.__Matrices_Fonctions_De_Forme_Iso(Ntild, gauss)
            
            # Construit dNtild
            dN1t = [lambda x,y,z: -1,   lambda x,y,z: -1,   lambda x,y,z: -1]
            dN2t = [lambda x,y,z: 1,    lambda x,y,z: 0,    lambda x,y,z: 0]
            dN3t = [lambda x,y,z: 0,    lambda x,y,z: 1,    lambda x,y,z: 0]
            dN4t = [lambda x,y,z: 0,    lambda x,y,z: 0,    lambda x,y,z: 1]

            dNtild = np.array([dN1t, dN2t, dN3t, dN4t])

            dN_pg = self.__Matrices_Dérivées_Fonctions_De_Forme_Iso(dNtild, gauss)
        
        return Nt_scalaire_pg, Nt_vecteur_pg, dN_pg, gauss

# ====================================

import unittest
import os

class Test_Element(unittest.TestCase):
    
    def setUp(self):
        self.elements = []
        for nPe in [3,4,6,8]:
            self.elements.append(ElementIsoparametrique(2,nPe))
        
        self.elements.append(ElementIsoparametrique(3,4))

    def test_BienCree(self):
        for element in self.elements:
            self.assertIsInstance(element, ElementIsoparametrique)        

if __name__ == '__main__':        
    try:
        os.system("cls")    #nettoie terminal
        unittest.main(verbosity=2)    
    except:
        print("")