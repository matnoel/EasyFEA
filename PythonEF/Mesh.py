import numpy as np
import scipy.sparse as sp

from Geom import *
from GroupElem import GroupElem, ElemType, MatriceType

class Mesh:

    def __init__(self, dict_groupElem: dict[ElemType,GroupElem], verbosity=True):
        """Création du maillage depuis coordo et connection
        Le maillage est l'entité qui possède les groupes d'élements
        """

        list_GroupElem = []
        dim=0
        for grp in dict_groupElem.values():
            if grp.dim > dim:
                # Ici on garrantie que l'element type du maillage utilisé est celui a la plus grande dimension
                dim = grp.dim
                self.__groupElem = grp
            list_GroupElem.append(grp)

        self.__dim = self.__groupElem.dim
        self.__dict_groupElem = dict_groupElem

        self.__verbosity = verbosity
        """le maillage peut ecrire dans la console"""
        
        if self.__verbosity:
            self.Resume()

    def ResetMatrices(self) -> None:
        [groupElem.InitMatrices() for groupElem in self.Get_list_groupElem()]            
    
    def Resume(self, verbosity=True):
        resume = f"\nTypes d'elements: {self.elemType}"
        resume += f"\nNe = {self.Ne}, Nn = {self.Nn}, nbDdl = {self.Nn*self.__dim}"
        if verbosity: print(resume)
        return resume
    
    def Get_list_groupElem(self, dim=None) -> list[GroupElem]:
        """Liste de group d'element du maillage"""
        if dim == None:
            dim = self.__dim
            
        list_groupElem = [grp for grp in self.__dict_groupElem.values() if grp.dim == dim]
        
        # Retourne la liste
        list_groupElem.reverse()

        return list_groupElem

    @property
    def dict_groupElem(self) -> dict[ElemType, GroupElem]:
        """dictionnaire qui contient tout les groupe d'element du maillage"""
        return self.__dict_groupElem

    @property
    def groupElem(self) -> GroupElem:
        """Groupe d'element du maillage
        """
        return self.__groupElem
    
    @property
    def elemType(self) -> ElemType:
        "Type d'element utilisé pour le maillage"
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
    def inDim(self):
        """Dimension dans lequel se trouve le maillage\n
        Un maillage 2D peut être orienté dans l'espace"""
        return self.__groupElem.inDim
    
    @property
    def nPe(self) -> int:
        """Noeuds par element"""
        return self.groupElem.nPe
    
    @property
    def coordo(self) -> np.ndarray:
        """Matrice des coordonnées de noeuds (Nn,3)"""
        return self.groupElem.coordo
    
    @property
    def nodes(self) -> np.ndarray:
        """Numéros des noeuds du maillage"""
        return self.groupElem.nodesID

    @property
    def coordoGlob(self) -> np.ndarray:
        """Matrice de coordonnées globale du maillage (maillage.Nn, 3)\n
        Contient toutes les coordonnées du maillage"""
        return self.groupElem.coordoGlob

    @property
    def connect(self) -> np.ndarray:
        """Matrice de connection des elements (Ne, nPe)"""
        return self.groupElem.connect_e
    
    def Get_connect_n_e(self) -> sp.csr_matrix:
        """Matrices creuses de 0 et 1 avec les 1 lorsque le noeud possède l'element (Nn, Ne).\nCette matrice permet de faire l'interpolation des valeurs des elements aux noeuds : 
        valeurs_n(Nn,1) = connect_n_e(Nn,Ne) * valeurs_e(Ne,1)
        """
        return self.groupElem.Get_connect_n_e()
    
    # Affichage

    @property
    def nbFaces(self) -> int:
        return self.groupElem.nbFaces
    
    @property
    def dict_connect_Triangle(self) -> dict[ElemType, np.ndarray]:
        """Transforme la matrice de connectivité pour la passer dans le trisurf en 2D"""
        return self.groupElem.Get_dict_connect_Triangle()
    
    @property
    def dict_connect_Faces(self) -> dict[ElemType, np.ndarray]:
        """Récupère les faces de chaque element et renvoie un dictionnaire pour chaque elements
        """
        return self.groupElem.Get_dict_connect_Faces()

    # Assemblage des matrices 

    @property
    def assembly_e(self) -> np.ndarray:
        """matrice d'assemblage (Ne, nPe*dim)\n
        Permet de positionner les matrices de type rigi dans la matrice globale"""
        return self.groupElem.assembly_e
    
    def Get_assemblyBeam_e(self, nbddl_n: int) -> np.ndarray:
        """matrice d'assemblage pour les poutres (Ne, nPe*nbddl_n)
        Permet de positionner les matrices de type beam dans la matrice globale"""
        return self.groupElem.Get_assemblyBeam_e(nbddl_n)

    @property
    def lignesVector_e(self) -> np.ndarray:
        """lignes pour remplir la matrice d'assemblage en vecteur (déplacement)"""
        assembly_e = self.assembly_e
        nPe = self.nPe
        Ne = self.Ne
        return np.repeat(assembly_e, nPe*self.__dim).reshape((Ne,-1))
    
    def Get_lignesVectorBeam_e(self, nbddl_n: int) -> np.ndarray:
        """lignes pour remplir la matrice d'assemblage en vecteur (poutre)"""
        assemblyBeam_e = self.Get_assemblyBeam_e(nbddl_n)
        nPe = self.nPe
        Ne = self.Ne
        return np.repeat(assemblyBeam_e, nPe*nbddl_n).reshape((Ne,-1))

    @property
    def colonnesVector_e(self) -> np.ndarray:
        """colonnes pour remplir la matrice d'assemblage en vecteur (déplacement)"""
        assembly_e = self.assembly_e
        nPe = self.nPe
        Ne = self.Ne
        return np.repeat(assembly_e, nPe*self.__dim, axis=0).reshape((Ne,-1))
    
    def Get_colonnesVectorBeam_e(self, nbddl_n: int) -> np.ndarray:
        """colonnes pour remplir la matrice d'assemblage en vecteur (poutre)"""
        assemblyBeam_e = self.Get_assemblyBeam_e(nbddl_n)
        nPe = self.nPe
        Ne = self.Ne
        return np.repeat(assemblyBeam_e, nPe*nbddl_n, axis=0).reshape((Ne,-1))

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

    # Calcul des surfaces, volumes et moments quadratique etc ...

    @property
    def aire(self) -> float:
        if self.dim in [0,1]: return
        aires = [group2D.aire for group2D in self.Get_list_groupElem(2)]
        return np.sum(aires)

    @property
    def Ix(self) -> float:
        if self.dim in [0,1]: return
        Ixs = [group2D.Ix for group2D in self.Get_list_groupElem(2)]
        return np.sum(Ixs)
    
    @property
    def Iy(self) -> float:
        if self.dim in [0,1]: return
        Iys = [group2D.Iy for group2D in self.Get_list_groupElem(2)]
        return np.sum(Iys)

    @property
    def Ixy(self) -> float:
        if self.dim in [0,1]: return
        Ixys = [group2D.Ixy for group2D in self.Get_list_groupElem(2)]
        return np.sum(Ixys)

    @property
    def J(self) -> float:
        if self.dim in [0,1]: return
        Js = [group2D.Iy + group2D.Ix for group2D in self.Get_list_groupElem(2)]
        return np.sum(Js)

    @property
    def volume(self) -> float:
        if self.dim != 3: return
        volumes = [group3D.volume for group3D in self.Get_list_groupElem(3)]
        return np.sum(volumes)

    # Construction des matrices élémentaires
    
    def Get_nPg(self, matriceType: MatriceType) -> np.ndarray:
        """nombre de point d'intégration par élement"""
        return self.groupElem.Get_gauss(matriceType).nPg

    def Get_poid_pg(self, matriceType: MatriceType) -> np.ndarray:
        """Points d'intégration (pg, dim, poid)"""
        return self.groupElem.Get_gauss(matriceType).poids

    def Get_jacobien_e_pg(self, matriceType: MatriceType) -> np.ndarray:
        """jacobien (e, pg)"""
        return self.groupElem.Get_jacobien_e_pg(matriceType)
    
    def Get_N_scalaire_pg(self, matriceType: MatriceType) -> np.ndarray:
        """Fonctions de formes dans l'element isoparamétrique pour un scalaire (npg, 1, npe)\n
        Matrice des fonctions de forme dans element de référence (ksi, eta)\n
        [N1(ksi,eta) N2(ksi,eta) Nn(ksi,eta)] \n
        """
        return self.groupElem.Get_N_pg(matriceType)

    def Get_N_vecteur_pg(self, matriceType: MatriceType) -> np.ndarray:
        """Fonctions de formes dans l'element de reférences pour un vecteur (npg, dim, npe*dim)\n
        Matrice des fonctions de forme dans element de référence (ksi, eta)\n
        [N1(ksi,eta) 0 N2(ksi,eta) 0 Nn(ksi,eta) 0 \n
        0 N1(ksi,eta) 0 N2(ksi,eta) 0 Nn(ksi,eta)]\n
        """
        return self.groupElem.Get_N_pg_rep(matriceType, self.__dim)

    def Get_dN_sclaire_e_pg(self, matriceType: MatriceType) -> np.ndarray:
        """Derivé des fonctions de formes dans la base réele en sclaire\n
        [dN1,x dN2,x dNn,x\n
        dN1,y dN2,y dNn,y]\n        
        (epij)
        """
        return self.groupElem.Get_dN_e_pg(matriceType)

    def Get_dNv_sclaire_e_pg(self, matriceType: MatriceType) -> np.ndarray:
        """Derivé des fonctions de formes de la poutre dans la base réele en sclaire\n
        [dNv1,x dNv2,x dNvn,x\n
        dNv1,y dNv2,y dNvn,y]\n        
        (epij)
        """
        return self.groupElem.Get_dNv_e_pg(matriceType)
    
    def Get_ddNv_sclaire_e_pg(self, matriceType: MatriceType) -> np.ndarray:
        """Derivé segonde des fonctions de formes de la poutre dans la base réele en sclaire\n
        [dNv1,xx dNv2,xx dNvn,xx\n
        dNv1,yy dNv2,yy dNvn,yy]\n        
        (epij)
        """
        return self.groupElem.Get_ddNv_e_pg(matriceType)

    def Get_ddN_sclaire_e_pg(self, matriceType: MatriceType) -> np.ndarray:
        """Derivé segonde des fonctions de formes dans la base réele en sclaire\n
        [dN1,xx dN2,xx dNn,xx\n
        dN1,yy dN2,yy dNn,yy]\n        
        (epij)
        """
        return self.groupElem.Get_ddN_e_pg(matriceType)

    def Get_B_dep_e_pg(self, matriceType: MatriceType) -> np.ndarray:
        """Derivé des fonctions de formes dans la base réele pour le problème de déplacement (e, pg, (3 ou 6), nPe*dim)\n
        exemple en 2D :\n
        [dN1,x 0 dN2,x 0 dNn,x 0\n
        0 dN1,y 0 dN2,y 0 dNn,y\n
        dN1,y dN1,x dN2,y dN2,x dN3,y dN3,x]\n

        (epij) Dans la base de l'element et en Kelvin Mandel
        """
        return self.groupElem.Get_B_dep_e_pg(matriceType)

    def Get_leftDepPart(self, matriceType: MatriceType) -> np.ndarray:
        """Renvoie la partie qui construit le therme de gauche de déplacement\n
        Ku_e = jacobien_e_pg * poid_pg * B_dep_e_pg' * c_e_pg * B_dep_e_pg\n
        
        Renvoie (epij) -> jacobien_e_pg * poid_pg * B_dep_e_pg'
        """
        return self.groupElem.Get_leftDepPart(matriceType)
    
    def Get_phaseField_ReactionPart_e_pg(self, matriceType: MatriceType) -> np.ndarray:
        """Renvoie la partie qui construit le therme de reaction\n
        K_r_e_pg = jacobien_e_pg * poid_pg * r_e_pg * Nd_pg' * Nd_pg\n
        
        Renvoie (epij) -> jacobien_e_pg * poid_pg * Nd_pg' * Nd_pg
        """
        return self.groupElem.Get_phaseField_ReactionPart_e_pg(matriceType)

    def Get_phaseField_DiffusePart_e_pg(self, matriceType: MatriceType) -> np.ndarray:
        """Renvoie la partie qui construit le therme de diffusion\n
        DiffusePart_e_pg = jacobien_e_pg * poid_pg * k * Bd_e_pg' * Bd_e_pg\n
        
        Renvoie (epij) -> jacobien_e_pg * poid_pg * Bd_e_pg' * Bd_e_pg
        """
        return self.groupElem.Get_phaseField_DiffusePart_e_pg(matriceType)

    def Get_phaseField_SourcePart_e_pg(self, matriceType: MatriceType) -> np.ndarray:
        """Renvoie la partie qui construit le therme de source\n
        SourcePart_e_pg = jacobien_e_pg, poid_pg, f_e_pg, Nd_pg'\n
        
        Renvoie (epij) -> jacobien_e_pg, poid_pg, Nd_pg'
        """
        return self.groupElem.Get_phaseField_SourcePart_e_pg(matriceType)
    
    # Récupération des noeuds

    def Nodes_Conditions(self, conditionX=True, conditionY=True, conditionZ=True) -> np.ndarray:
        """Renvoie la liste d'identifiant des noeuds qui respectent les condtions

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
        """Renvoie les noeuds sur le point (identifiants)"""
        return self.groupElem.Get_Nodes_Point(point)

    def Nodes_Line(self, line: Line) -> np.ndarray:
        """Renvoie les noeuds qui sont sur la ligne (identifiants)"""
        return self.groupElem.Get_Nodes_Line(line)

    def Nodes_Domain(self, domain: Domain) -> np.ndarray:
        """Renvoie les noeuds qui sont dans le domaine  (identifiants)"""
        return self.groupElem.Get_Nodes_Domain(domain)
    
    def Nodes_Circle(self, circle: Circle) -> np.ndarray:
        """Renvoie les noeuds qui sont dans le cercle  (identifiants)"""
        return self.groupElem.Get_Nodes_Circle(circle)

    def Nodes_Cylindre(self, circle: Circle, direction=[0,0,1]) -> np.ndarray:
        """Renvoie les noeuds qui sont dans le cylindre (identifiants)"""
        return self.groupElem.Get_Nodes_Cylindre(circle, direction)

    def Elements_Nodes(self, nodes: np.ndarray, exclusivement=True):
        """Renvoie les élements qui utilisent exclusivement ou non les noeuds renseignés en fonction du groupe d'element (elements de la dimension du maillage)"""
        elements = self.groupElem.Get_ElementsIndex_Nodes(nodes=nodes, exclusivement=exclusivement)
        return elements


    @staticmethod
    def __Dim_For_Tag(tag):
        if 'P' in tag:
            dim = 0
        elif 'L' in tag:
            dim = 1            
        elif 'S' in tag:
            dim = 2            
        elif 'V' in tag:
            dim = 3
        
        return dim

    def Nodes_Tag(self, tags: list[str]) -> np.ndarray:
        """Renvoie les noeuds qui utilisent le tag (identifiants)"""
        nodes = []
        [nodes.extend(grp.Get_Nodes_Tag(tag)) for tag in tags for grp in self.Get_list_groupElem(Mesh.__Dim_For_Tag(tag))]

        return np.unique(nodes)

    def Elements_Tag(self, tags: list[str]) -> np.ndarray:
        """Renvoie les elements qui utilisent le tag"""
        elements = []
        [elements.extend(grp.Get_Elements_Tag(tag)) for tag in tags for grp in self.Get_list_groupElem(Mesh.__Dim_For_Tag(tag))]

        return np.unique(elements)

    def Localises_sol_e(self, sol: np.ndarray) -> np.ndarray:
        """sur chaque elements on récupère les valeurs de sol"""
        return self.groupElem.Localise_sol_e(sol)
    