
from typing import cast, List, Dict
import numpy as np
import scipy.sparse as sp

from Geom import *
from GroupElem import GroupElem
from TicTac import Tic

class Mesh:

    def __init__(self, dict_groupElem: dict, verbosity=True):
        """Création du maillage depuis coordo et connection
        Le maillage est l'entité qui possède les groupes d'élements
        
        affichageMaillage : bool, optional
            Affichage après la construction du maillage, by default True
        """

        # Onrevifie que l'on contient que des GroupElem
        list_GroupElem = []
        dim=0
        for grp in dict_groupElem.values():
            assert isinstance(grp, GroupElem)
            if grp.dim > dim:
                # Ici on garrantie que l'element type du maillage utilisé est celui a la plus grande dimension
                self.__groupElem = grp
            list_GroupElem.append(grp)

        self.__dim = self.__groupElem.dim

        self.__dict_groupElem = dict_groupElem

        self.__dict_B_dep_e_pg = {}

        self.__verbosity = verbosity
        """le maillage peut ecrire dans la console"""
        
        if self.__verbosity:
            self.Resume()            
    
    def Resume(self):
        print(f"\nTypes d'elements: {self.elemType}")
        print(f"Ne = {self.Ne}, Nn = {self.Nn}, nbDdl = {self.Nn*self.__dim}")
    
    def Get_list_groupElem(self, dim=None) -> List[GroupElem]:
        """Liste de group d'element du maillage"""
        if dim == None:
            dim = self.__dim
            
        list_groupElem = []

        for key in self.__dict_groupElem:
            grp = cast(GroupElem, self.__dict_groupElem[key])
            if grp.dim == dim: list_groupElem.append(grp)
        
        list_groupElem = cast(List[GroupElem], list_groupElem)

        return list_groupElem

    @property
    def dict_groupElem(self) -> Dict[str, GroupElem]:
        return self.__dict_groupElem

    @property
    def groupElem(self) -> GroupElem:
        """Groupe d'element du maillage

        Returns:
            GroupElem: _description_
        """
        return self.__groupElem
    @property
    def elemType(self) -> str:
        "elements utilisés pour le maillage"
        return self.groupElem.elemType
    
    @property
    def Ne(self) -> int:
        """Nombre d'élements du maillage"""
        return self.groupElem.Ne
    
    @property
    def Nn(self, dim=None) -> int:
        """Nombre de noeuds du maillage"""
        return self.groupElem.Nn
    
    @property
    def dim(self):
        """Dimension du maillage"""
        return self.__dim
    
    @property
    def nPe(self) -> int:
        """noeuds par element"""
        return self.groupElem.nPe
    
    @property
    def coordo(self) -> np.ndarray:
        """matrice des coordonnées de noeuds (Nn,3)"""
        return self.groupElem.coordo
    
    @property
    def nodes(self) -> np.ndarray:
        """numéros des noeuds du maillage"""
        return self.groupElem.nodes

    @property
    def coordoGlob(self) -> np.ndarray:
        """matrice de coordonnées globale du maillage (maillage.Nn, 3)"""
        return self.groupElem.coordoGlob

    @property
    def connect(self) -> np.ndarray:
        """connection des elements (Ne, nPe)"""
        return self.groupElem.connect
    
    @property
    def connect_n_e(self) -> sp.csr_matrix:
        """matrices de 0 et 1 avec les 1 lorsque le noeud possède l'element (Nn, Ne)\n
        tel que : valeurs_n(Nn,1) = connect_n_e(Nn,Ne) * valeurs_e(Ne,1)"""
        return self.groupElem.connect_n_e

    @property
    def assembly_e(self) -> np.ndarray:
        """matrice d'assemblage (Ne, nPe*dim)"""
        return self.groupElem.assembly_e
    
    # Affichage

    @property
    def nbFaces(self) -> int:
        return self.groupElem.nbFaces
    
    @property
    def connectTriangle(self) -> np.ndarray:
        """Transforme la matrice de connectivité pour la passer dans le trisurf en 2D"""
        return self.groupElem.get_connectTriangle()
    
    @property
    def connect_Faces(self) -> np.ndarray:
        """Récupère les faces de chaque element et renvoie un dictionnaire pour chaque elements
        """
        return self.groupElem.get_connect_Faces()

    # Assemblage des matrices 

    @property
    def lignesVector_e(self) -> np.ndarray:
        """lignes pour remplir la matrice d'assemblage en vecteur (déplacement)"""
        assembly_e = self.assembly_e
        nPe = self.nPe
        Ne = self.Ne
        return np.repeat(assembly_e, nPe*self.__dim).reshape((Ne,-1))

    @property
    def colonnesVector_e(self) -> np.ndarray:
        """colonnes pour remplir la matrice d'assemblage en vecteur (déplacement)"""
        assembly_e = self.assembly_e
        nPe = self.nPe
        Ne = self.Ne
        return np.repeat(assembly_e, nPe*self.__dim, axis=0).reshape((Ne,-1))

    @property
    def lignesScalar_e(self) -> np.ndarray:
        """lignes pour remplir la matrice d'assemblage en scalaire (endommagement, ou thermique)"""
        connect = self.connect
        nPe = self.nPe
        Ne = self.Ne
        return np.repeat(connect, nPe).reshape((Ne,-1))

    @property
    def colonnesScalar_e(self) -> np.ndarray:
        """colonnes pour remplir la matrice d'assemblage en scalaire (endommagement, ou thermique)"""
        connect = self.connect
        nPe = self.nPe
        Ne = self.Ne
        return np.repeat(connect, nPe, axis=0).reshape((Ne,-1))

    # Construction des matrices élémentaires

    @property
    def aire(self) -> float:
        if self.dim == 1: return
        aire = 0
        for group2D in self.Get_list_groupElem(2):
            aire += group2D.aire
        return aire

    @property
    def volume(self) -> float:
        if self.dim != 3: return
        volume=0
        for group3D in self.Get_list_groupElem(3):
            volume += group3D.volume
        return volume
    
    def Get_nPg(self, matriceType: str) -> np.ndarray:
        """nombre de point d'intégration par élement"""
        return self.groupElem.get_gauss(matriceType).nPg

    def Get_poid_pg(self, matriceType: str) -> np.ndarray:
        """Points d'intégration (pg, dim, poid)"""
        return self.groupElem.get_gauss(matriceType).poids

    def Get_jacobien_e_pg(self, matriceType: str) -> np.ndarray:
        """jacobien (e, pg)"""
        return self.groupElem.get_jacobien_e_pg(matriceType)
    
    def Get_N_scalaire_pg(self, matriceType: str) -> np.ndarray:
        """Fonctions de formes dans l'element isoparamétrique pour un scalaire (npg, 1, npe)\n
        Matrice des fonctions de forme dans element de référence (ksi, eta)\n
        [N1(ksi,eta) N2(ksi,eta) Nn(ksi,eta)] \n
        """
        return self.groupElem.get_N_pg(matriceType)

    def Get_N_vecteur_pg(self, matriceType: str) -> np.ndarray:
        """Fonctions de formes dans l'element de reférences pour un vecteur (npg, dim, npe*dim)\n
        Matrice des fonctions de forme dans element de référence (ksi, eta)\n
        [N1(ksi,eta) 0 N2(ksi,eta) 0 Nn(ksi,eta) 0 \n
        0 N1(ksi,eta) 0 N2(ksi,eta) 0 Nn(ksi,eta)]\n
        """
        return self.groupElem.get_N_pg(matriceType, self.__dim)

    def Get_B_sclaire_e_pg(self, matriceType: str) -> np.ndarray:
        """Derivé des fonctions de formes dans la base réele en sclaire\n
        [dN1,x dN2,x dNn,x\n
        dN1,y dN2,y dNn,y]\n        
        (epij)
        """
        return self.groupElem.get_dN_e_pg(matriceType)

    def Get_B_dep_e_pg(self, matriceType: str) -> np.ndarray:
        """Derivé des fonctions de formes dans la base réele pour le problème de déplacement (e, pg, (3 ou 6), nPe*dim)\n
        exemple en 2D :\n
        [dN1,x 0 dN2,x 0 dNn,x 0\n
        0 dN1,y 0 dN2,y 0 dNn,y\n
        dN1,y dN1,x dN2,y dN2,x dN3,y dN3,x]\n

        (epij) Dans la base de l'element et en Kelvin Mandel
        """
        return self.groupElem.get_B_dep_e_pg(matriceType)

    def Get_leftDepPart(self, matriceType: str) -> np.ndarray:
        """Renvoie la partie qui construit le therme de gauche de déplacement\n
        Ku_e = jacobien_e_pg * poid_pg * B_dep_e_pg' * c_e_pg * B_dep_e_pg\n
        
        Renvoie (epij) -> jacobien_e_pg * poid_pg * B_dep_e_pg'
        """
        return self.groupElem.get_leftDepPart(matriceType)
    
    def Get_phaseField_ReactionPart_e_pg(self, matriceType: str) -> np.ndarray:
        """Renvoie la partie qui construit le therme de reaction\n
        K_r_e_pg = jacobien_e_pg * poid_pg * r_e_pg * Nd_pg' * Nd_pg\n
        
        Renvoie (epij) -> jacobien_e_pg * poid_pg * Nd_pg' * Nd_pg
        """
        return self.groupElem.get_phaseField_ReactionPart_e_pg(matriceType)

    def Get_phaseField_DiffusePart_e_pg(self, matriceType: str) -> np.ndarray:
        """Renvoie la partie qui construit le therme de diffusion\n
        DiffusePart_e_pg = jacobien_e_pg * poid_pg * k * Bd_e_pg' * Bd_e_pg\n
        
        Renvoie -> jacobien_e_pg * poid_pg * Bd_e_pg' * Bd_e_pg
        """
        return self.groupElem.get_phaseField_DiffusePart_e_pg(matriceType)

    def Get_phaseField_SourcePart_e_pg(self, matriceType: str) -> np.ndarray:
        """Renvoie la partie qui construit le therme de source\n
        SourcePart_e_pg = jacobien_e_pg, poid_pg, f_e_pg, Nd_pg'\n
        
        Renvoie -> jacobien_e_pg, poid_pg, Nd_pg'
        """
        return self.groupElem.get_phaseField_SourcePart_e_pg(matriceType)
    
    # Récupération des noeuds

    def Get_Nodes_Conditions(self, conditionX=True, conditionY=True, conditionZ=True) -> np.ndarray:
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
    
    def Nodes_Point(self, point: Point) -> np.ndarray:
        """Renvoie la liste le noeud sur le point"""
        return self.groupElem.Get_Nodes_Point(point)

    def Nodes_Line(self, line: Line) -> np.ndarray:
        """Renvoie la liste de noeuds qui sont sur la ligne"""
        return self.groupElem.Get_Nodes_Line(line)

    def Get_Nodes_Domain(self, domain: Domain) -> np.ndarray:
        """Renvoie la liste de noeuds qui sont dans le domaine"""
        return self.groupElem.Get_Nodes_Domain(domain)
    
    def Nodes_Circle(self, circle: Circle) -> np.ndarray:
        """Renvoie la liste de noeuds qui sont dans le cercle"""
        return self.groupElem.Get_Nodes_Circle(circle)    

    def Localises_sol_e(self, sol: np.ndarray) -> np.ndarray:
        """sur chaque elements on récupère les valeurs de sol"""
        return self.groupElem.Localise_sol_e(sol)
    