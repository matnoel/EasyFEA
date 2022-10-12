
from inspect import stack
from typing import Dict, List, cast

from Geom import *
from Gauss import Gauss
from TicTac import Tic
from matplotlib import pyplot as plt

import numpy as np
import scipy.sparse as sp

class GroupElem:
    """Classe GoupElem\n\n
    
    Un maillage utilise plusieurs groupe d'elements par exemple un maillage avec des cubes (HEXA8) utilise :
    - POINT
    - SEG2
    - QUAD4
    - HEXA8
    """

    def __init__(self, gmshId: int, connect: np.ndarray, elementsID: np.ndarray,
    coordoGlob: np.ndarray, nodesID: np.ndarray):
        """Construction d'un groupe d'element

        Parameters
        ----------
        gmshId : int
            identifiant gmsh
        connect : np.ndarray
            matrice de connectivité
        elementsID : np.ndarray
            identifiants des noeuds
        coordoGlob : np.ndarray
            matrice de coordonnée totale du maillage (contient toutes les coordonnées du maillage)
        nodesID : np.ndarray
            identifiants des noeuds
        """

        self.__gmshId = gmshId            
        
        # Elements
        self.__elementsID = elementsID
        # ici on consruit une liste permettant de donner la position de l'element dans connect en fonction de son numéro
        self.__elementsIndex = np.zeros(elementsID.max()+1, dtype=int)
        self.__elementsIndex[elementsID] = np.arange(elementsID.shape[0])
        self.__connect = connect

        # Noeuds
        self.__nodesID = nodesID
        # ici on consruit une liste permettant de donner la position du noeuds dans coordo ou connect_n_e en fonction de son numéro
        self.__nodesIndex = np.zeros(nodesID.max()+1, dtype=int)
        self.__nodesIndex[nodesID] = np.arange(nodesID.shape[0])

        self.__coordoGlob = coordoGlob
        self.__coordo = cast(np.ndarray, coordoGlob[nodesID])
        
        if self.__coordo[:,1].max()==0:
            self.__inDim = 1
        if self.__coordo[:,2].max()==0:
            self.__inDim = 2
        else:
            self.__inDim = 3
        
        self.InitMatrices()
    
    def InitMatrices(self):
        """Initialise les dictionnaires de matrices pour la constructions elements finis"""
        # Dictionnaires pour chaque types de matrices
        if self.dim > 0:
            self.__dict_dN_e_pg = {}
            self.__dict_dNv_e_pg = {}
            self.__dict_ddNv_e_pg = {}
            self.__dict_ddN_e_pg = {}
            self.__dict_F_e_pg = {}                
            self.__dict_invF_e_pg = {}                
            self.__dict_jacobien_e_pg = {}   
            self.__dict_B_dep_e_pg = {}
            self.__dict_leftDepPart = {}
            self.__dict_phaseField_ReactionPart_e_pg = {}
            self.__dict_phaseField_DiffusePart_e_pg = {}
            self.__dict_phaseField_SourcePart_e_pg = {}


    ################################################ METHODS ##################################################

    @property
    def gmshId(self) -> int:
        """Identifiant gmsh"""
        return self.__gmshId

    @property
    def elemType(self) -> str:
        """Type d'element"""
        return GroupElem.Get_ElemInFos(self.__gmshId)[0]
    @property
    def nPe(self) -> int:
        """Nombre de noeuds par element"""
        return GroupElem.Get_ElemInFos(self.__gmshId)[1]
    
    @property
    def dim(self) -> int:
        """Dimension de l'element"""
        return GroupElem.Get_ElemInFos(self.__gmshId)[2]

    @property
    def inDim(self) -> int:
        """Dimension dans lequel ce situe l'element"""
        return self.__inDim

    @property
    def Ne(self) -> int:
        """Nombre d'elements"""
        return self.__connect.shape[0]

    @property
    def nodesID(self) -> int:
        """Numéro des noeuds, attention ID n'est pas index (voir nodesIndex)\n
        Pourquoi ? -> Parce que le noeuds 10 peut être a la lignes 3 dans coordo !"""
        return self.__nodesID.copy()

    @property
    def nodesIndex(self) -> int:
        """Position du noeud dans coordo ou connect_n_e"""
        return self.__nodesIndex.copy()

    @property
    def elementsID(self) -> np.ndarray:
        """Numéro des elements, attention ID n'est pas index (voir elementsIndex)\n
        Pourquoi ? -> Parce que l'element 3 peut être a la lignes 7 dans connect !"""
        return self.__elementsID.copy()

    @property
    def elementsIndex(self) -> int:
        """Position de l'element dans connect"""
        return self.__elementsIndex.copy()

    @property
    def Nn(self) -> int:
        """Nombre de noeuds"""
        return self.__coordo.shape[0]

    @property
    def coordo(self) -> np.ndarray:
        """Cette matrice contient les coordonnées du groupe d'element (Nn, 3)"""
        return self.__coordo.copy()

    @property
    def coordoGlob(self) -> np.ndarray:
        """Cette matrice contient tout les coordonnées du maillage (maillage.Nn, 3)"""
        return self.__coordoGlob.copy()

    @property
    def nbFaces(self) -> int:
        """Nombre de faces"""
        if self.dim in [0,1]:
            return 0
        elif self.dim == 2:
            return 1
        elif self.dim == 3:                
            if self.elemType == "TETRA4":
                return 4
            elif self.elemType == "HEXA8":
                return 6
            elif self.elemType == "PRISM6":
                return 5
            else:
                raise "Element inconnue"
    
    @property
    def connect_e(self) -> np.ndarray:
        """nodesID des elements (Ne, nPe)"""
        return self.__connect.copy()

    @property
    def connect_n_e(self) -> sp.csr_matrix:
        """Matrices de 0 et 1 avec les 1 lorsque le noeud possède l'element (Nn, Ne) soit\n
        tel que : valeurs_n(Nn,1) = connect_n_e(Nn,Ne) * valeurs_e(Ne,1)\n
        - (nodesId, elementsIndex)"""
        # Ici l'objectif est de construire une matrice qui lorsque quon va la multiplier a un vecteur valeurs_e de taille ( Ne x 1 ) va donner
        # valeurs_n_e(Nn,1) = connecNoeud(Nn,Ne) valeurs_n_e(Ne,1)
        # ou connecNoeud(Nn,:) est un vecteur ligne composé de 0 et de 1 qui permetra de sommer valeurs_e[noeuds]
        # Ensuite, il suffit juste par divisier par le nombre de fois que le noeud apparait dans la ligne        
        # L'idéal serait dobtenir connectNoeud (Nn x nombre utilisation du noeud par element) rapidement        
        Ne = self.Ne
        nPe = self.nPe
        listElem = np.arange(Ne)

        lignes = self.connect_e.reshape(-1)

        Nn = int(lignes.max()+1)
        colonnes = np.repeat(listElem, nPe)

        return sp.csr_matrix((np.ones(nPe*Ne),(lignes, colonnes)),shape=(Nn,Ne))

    @property
    def assembly_e(self, dim=None) -> np.ndarray:
        """Matrice d'assemblage (Ne, nPe*dim)"""
        nPe = self.nPe
        if dim == None:
            dim = self.dim
        taille = nPe*dim

        assembly = np.zeros((self.Ne, taille), dtype=np.int64)
        connect = self.connect_e

        for d in range(dim):
            colonnes = np.arange(d, taille, dim)
            assembly[:, colonnes] = np.array(connect) * dim + d

        return assembly
    
    def assemblyBeam_e(self, nbddl_e: int) -> np.ndarray:
        """Matrice d'assemblage pour les poutres (Ne, nPe*dim)"""

        nPe = self.nPe
        taille = nbddl_e*nPe

        assembly = np.zeros((self.Ne, taille), dtype=np.int64)
        connect = self.connect_e

        for d in range(nbddl_e):
            colonnes = np.arange(d, taille, nbddl_e)
            assembly[:, colonnes] = np.array(connect) * nbddl_e + d

        return assembly

    def get_elementsIndex(self, noeuds: np.ndarray, exclusivement=True) -> np.ndarray:
        """Récupérations des élements qui utilisent exclusivement ou non les noeuds renseignés"""
        connect = self.__connect
        connect_n_e = self.connect_n_e
        
        # Nn = self.Nn
        # # Verifie si il n'y a pas de noeuds en trop
        # if self.Nn < noeuds.max():
        #     # Il faut enlever des noeuds
        #     # On enlève tout les noeuds en trop
        #     indexNoeudsSansDepassement = np.where(noeuds < self.Nn)[0]
        #     noeuds = noeuds[indexNoeudsSansDepassement]

        nodesId = noeuds

        

        lignes, colonnes, valeurs = sp.find(connect_n_e[nodesId])
        elementsIndex = np.unique(colonnes)
        # elementsIndex = self.elementsIndex[elementsID]

        if exclusivement:
            # Verifie si les elements utilisent exculisevement les noeuds dans la liste de noeuds
            # Pour chaque element, si lelement contient un noeuds n'appartenant pas à la liste de noeuds on l'enlève
            listElemIndex = [e for e in elementsIndex if not False in [n in noeuds for n in connect[e]]]        
            listElemIndex = np.array(listElemIndex)
        else:
            listElemIndex = elementsIndex

        return listElemIndex

    def get_assembly(self, dim=None) -> np.ndarray:
        """Matrice d'assemblage pour positionner les matrices locale dans le système globale"""
        self.assembly_e(dim)

    def get_gauss(self, matriceType: str) -> Gauss:
        """Renvoie les points d'intégration en fonction du type de matrice"""
        return Gauss(self.elemType, matriceType)
    
    def get_coordo_e_p(self, matriceType: str, elements: np.ndarray) -> np.ndarray:
        """Renvoie les coordonnées des points d'intégration pour chaque element"""

        N_scalaire = self.get_N_pg(matriceType)

        # récupère les coordonnées des noeuds
        coordo = self.__coordoGlob

        # coordonnées localisées sur l'elements
        if elements.size == 0:
            coordo_e =  coordo[self.__connect]
        else:
            coordo_e =  coordo[self.__connect[elements]]

        # on localise les coordonnées sur les points de gauss
        coordo_e_p = np.einsum('pij,ejn->epn', N_scalaire, coordo_e, optimize='optimal')

        return np.array(coordo_e_p)

    def get_N_pg(self, matriceType: str, repetition=1) -> np.ndarray:
        """Fonctions de formes dans la base de réference

        Args:
            matriceType (str): ["rigi","masse"]
            isScalaire (bool): type de matrice N\n

        Returns:
            np.ndarray: . Fonctions de formes vectorielles (pg, rep=2, rep=2*dim), dans la base (ksi, eta ...)\n
                            [Ni 0 . . . Nn 0 \n
                            0 Ni . . . 0 Nn]

                        . Fonctions de formes scalaires (pg, rep=1, nPe), dans la base (ksi, eta ...)\n
                            [Ni . . . Nn]
        """
        if self.dim == 0: return

        assert isinstance(repetition, int)
        assert repetition >= 1

        N_pg = self.__get_N_pg(matriceType)

        if not isinstance(N_pg, np.ndarray): return

        if repetition <= 1:
            return N_pg
        else:
            taille = N_pg.shape[2]*(repetition)
            N_vect_pg = np.zeros((N_pg.shape[0] ,repetition , taille))

            for r in range(repetition):
                N_vect_pg[:, r, np.arange(r, taille, repetition)] = N_pg[:,0,:]
            
            return N_vect_pg
    
    def get_dN_e_pg(self, matriceType: str) -> np.ndarray:
        """Derivé des fonctions de formes dans la base réele en sclaire\n
        [dN1,x dN2,x dNn,x\n
        dN1,y dN2,y dNn,y]\n        
        """
        assert matriceType in GroupElem.get_MatriceType()

        if matriceType not in self.__dict_dN_e_pg.keys():

            invF_e_pg = self.get_invF_e_pg(matriceType)

            dN_pg = self.get_dN_pg(matriceType)

            # Derivé des fonctions de formes dans la base réele
            dN_e_pg = np.array(np.einsum('epik,pkj->epij', invF_e_pg, dN_pg, optimize='optimal'))
            self.__dict_dN_e_pg[matriceType] = dN_e_pg

        return self.__dict_dN_e_pg[matriceType].copy()

    def get_dNv_e_pg(self, matriceType: str) -> np.ndarray:
        """Derivé des fonctions de formes de la poutre dans la base réele en sclaire\n
        [dNv1,x dNv2,x dNvn,x\n
        dNv1,y dNv2,y dNvn,y]\n
        """
        assert matriceType in GroupElem.get_MatriceType()

        if matriceType not in self.__dict_dNv_e_pg.keys():

            invF_e_pg = self.get_invF_e_pg(matriceType)

            dNv_pg = self.get_dNv_pg(matriceType)

            jacobien_e_pg = self.get_jacobien_e_pg(matriceType)
            Ne = jacobien_e_pg.shape[0]
            pg = self.get_gauss(matriceType)

            # On créer la dimension sur les elements
            dNv_e_pg = dNv_pg[np.newaxis, :, 0, :].repeat(Ne,  axis=0)
            # On récupère la longeur des poutres sur chaque element aux points d'intégrations
            l_e_pg = np.einsum('ep,p->ep', jacobien_e_pg, pg.poids, optimize='optimal')
            # On multiplie par la longueur les ddNv2_e_pg et ddNv4_e_pg
            dNv_e_pg[:,:,1] = np.einsum('ep,e->ep',dNv_e_pg[:,:,1],l_e_pg[:,0])
            dNv_e_pg[:,:,3] = np.einsum('ep,e->ep',dNv_e_pg[:,:,3],l_e_pg[:,1])

            # Derivé des fonctions de formes dans la base réele
            invF_e_pg = invF_e_pg.reshape((Ne, pg.nPg, 1)).repeat(dNv_e_pg.shape[-1], axis=-1)
            dNv_e_pg = invF_e_pg * dNv_e_pg
            self.__dict_dNv_e_pg[matriceType] = dNv_e_pg

        return self.__dict_dNv_e_pg[matriceType].copy()

    def get_ddNv_e_pg(self, matriceType: str) -> np.ndarray:
        """Derivé des fonctions de formes de la poutre dans la base réele en sclaire\n
        [dNv1,xx dNv2,xx dNvn,xx\n
        dNv1,yy dNv2,yy dNvn,yy]\n        
        """
        assert matriceType in GroupElem.get_MatriceType()

        if matriceType not in self.__dict_ddNv_e_pg.keys():

            invF_e_pg = self.get_invF_e_pg(matriceType)

            ddNv_pg = self.get_ddNv_pg(matriceType)

            jacobien_e_pg = self.get_jacobien_e_pg(matriceType)
            Ne = jacobien_e_pg.shape[0]
            pg = self.get_gauss(matriceType)

            # On créer la dimension sur les elements
            ddNv_e_pg = ddNv_pg[np.newaxis, :, 0, :].repeat(Ne,  axis=0)
            # On récupère la longeur des poutres sur chaque element aux points d'intégrations
            l_e_pg = np.einsum('ep,p->e', jacobien_e_pg, pg.poids, optimize='optimal')
            l_e_pg = l_e_pg.repeat(pg.nPg).reshape(Ne, pg.nPg)
            # On multiplie par la longueur les ddNv2_e_pg et ddNv4_e_pg
            ddNv_e_pg[:,:,1] = np.einsum('ep,ep->ep',ddNv_e_pg[:,:,1],l_e_pg, optimize='optimal')
            ddNv_e_pg[:,:,3] = np.einsum('ep,ep->ep',ddNv_e_pg[:,:,3],l_e_pg, optimize='optimal')
            # Derivé des fonctions de formes dans la base réele
            invF_e_pg = invF_e_pg.reshape((Ne, pg.nPg, 1)).repeat(ddNv_e_pg.shape[-1], axis=-1)
            ddNv_e_pg = invF_e_pg * invF_e_pg * ddNv_e_pg

            # ddNv_e_pg = np.array(np.einsum('epik,pkj->epij', invF_e_pg, ddNv_pg, optimize='optimal'))

            self.__dict_ddNv_e_pg[matriceType] = ddNv_e_pg

        return self.__dict_ddNv_e_pg[matriceType].copy()

    def get_ddN_e_pg(self, matriceType: str) -> np.ndarray:
        """Derivé des fonctions de formes dans la base réele en sclaire\n
        [dN1,xx dN2,xx dNn,xx\n
        dN1,yy dN2,yy dNn,yy]\n        
        """
        assert matriceType in GroupElem.get_MatriceType()

        if matriceType not in self.__dict_ddN_e_pg.keys():

            invF_e_pg = self.get_invF_e_pg(matriceType)

            ddN_pg = self.get_ddN_pg(matriceType)

            # Derivé des fonctions de formes dans la base réele
            ddN_e_pg = np.array(np.einsum('epik,pkj->epij', invF_e_pg, ddN_pg, optimize='optimal'))
            self.__dict_ddN_e_pg[matriceType] = ddN_e_pg

        return self.__dict_ddN_e_pg[matriceType].copy()

    def get_B_dep_e_pg(self, matriceType: str) -> np.ndarray:
        """Derivé des fonctions de formes dans la base réele pour le problème de déplacement (e, pg, (3 ou 6), nPe*dim)\n
        exemple en 2D :\n
        [dN1,x 0 dN2,x 0 dNn,x 0\n
        0 dN1,y 0 dN2,y 0 dNn,y\n
        dN1,y dN1,x dN2,y dN2,x dN3,y dN3,x]\n

        (epij) Dans la base de l'element et en Kelvin Mandel
        """
        assert matriceType in GroupElem.get_MatriceType()

        if matriceType not in self.__dict_B_dep_e_pg.keys():

            dN_e_pg = self.get_dN_e_pg(matriceType)

            nPg = self.get_gauss(matriceType).nPg
            nPe = self.nPe
            dim = self.dim
            listnPe = np.arange(nPe)
            
            colonnes0 = np.arange(0, nPe*dim, dim)
            colonnes1 = np.arange(1, nPe*dim, dim)

            if self.dim == 2:
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

            import Materials
            B_e_pg = Materials.LoiDeComportement.AppliqueCoefSurBrigi(dim, B_e_pg)

            self.__dict_B_dep_e_pg[matriceType] = B_e_pg
        
        return self.__dict_B_dep_e_pg[matriceType].copy()

    def get_leftDepPart(self, matriceType: str) -> np.ndarray:
        """Renvoie la partie qui construit le therme de gauche de déplacement\n
        Ku_e = jacobien_e_pg * poid_pg * B_dep_e_pg' * c_e_pg * B_dep_e_pg\n
        
        Renvoie (epij) -> jacobien_e_pg * poid_pg * B_dep_e_pg'
        """

        assert matriceType in GroupElem.get_MatriceType()

        if matriceType not in self.__dict_leftDepPart.keys():
            
            jacobien_e_pg = self.get_jacobien_e_pg(matriceType)
            poid_pg = self.get_gauss(matriceType).poids
            B_dep_e_pg = self.get_B_dep_e_pg(matriceType)

            leftDepPart = np.einsum('ep,p,epij->epji', jacobien_e_pg, poid_pg, B_dep_e_pg, optimize='optimal')

            self.__dict_leftDepPart[matriceType] = leftDepPart

        return self.__dict_leftDepPart[matriceType].copy()
            
            
    
    def get_phaseField_ReactionPart_e_pg(self, matriceType: str) -> np.ndarray:
        """Renvoie la partie qui construit le therme de reaction\n
        ReactionPart_e_pg = jacobien_e_pg * poid_pg * r_e_pg * Nd_pg' * Nd_pg\n
        
        Renvoie -> jacobien_e_pg * poid_pg * Nd_pg' * Nd_pg
        """

        assert matriceType in GroupElem.get_MatriceType()

        if matriceType not in self.__dict_phaseField_ReactionPart_e_pg.keys():

            jacobien_e_pg = self.get_jacobien_e_pg(matriceType)
            poid_pg = self.get_gauss(matriceType).poids
            Nd_pg = self.get_N_pg(matriceType, 1)

            ReactionPart_e_pg = np.einsum('ep,p,pki,pkj->epij', jacobien_e_pg, poid_pg, Nd_pg, Nd_pg, optimize='optimal')

            self.__dict_phaseField_ReactionPart_e_pg[matriceType] = ReactionPart_e_pg
        
        return self.__dict_phaseField_ReactionPart_e_pg[matriceType].copy()
    
    def get_phaseField_DiffusePart_e_pg(self, matriceType: str) -> np.ndarray:
        """Renvoie la partie qui construit le therme de diffusion\n
        DiffusePart_e_pg = jacobien_e_pg * poid_pg * k * Bd_e_pg' * Bd_e_pg\n
        
        Renvoie -> jacobien_e_pg * poid_pg * Bd_e_pg' * Bd_e_pg
        """

        assert matriceType in GroupElem.get_MatriceType()

        if matriceType not in self.__dict_phaseField_DiffusePart_e_pg.keys():

            jacobien_e_pg = self.get_jacobien_e_pg(matriceType)
            poid_pg = self.get_gauss(matriceType).poids
            Bd_e_pg = self.get_dN_e_pg(matriceType)

            DiffusePart_e_pg = np.einsum('ep,p,epki,epkj->epij', jacobien_e_pg, poid_pg, Bd_e_pg, Bd_e_pg, optimize='optimal')

            self.__dict_phaseField_DiffusePart_e_pg[matriceType] = DiffusePart_e_pg
        
        return self.__dict_phaseField_DiffusePart_e_pg[matriceType].copy()

    def get_phaseField_SourcePart_e_pg(self, matriceType: str) -> np.ndarray:
        """Renvoie la partie qui construit le therme de source\n
        SourcePart_e_pg = jacobien_e_pg, poid_pg, f_e_pg, Nd_pg'\n
        
        Renvoie -> jacobien_e_pg, poid_pg, Nd_pg'
        """

        assert matriceType in GroupElem.get_MatriceType()

        if matriceType not in self.__dict_phaseField_SourcePart_e_pg.keys():

            jacobien_e_pg = self.get_jacobien_e_pg(matriceType)
            poid_pg = self.get_gauss(matriceType).poids
            Nd_pg = self.get_N_pg(matriceType, 1)

            SourcePart_e_pg = np.einsum('ep,p,pij->epji', jacobien_e_pg, poid_pg, Nd_pg, optimize='optimal') #le ji a son importance pour la transposé

            self.__dict_phaseField_SourcePart_e_pg[matriceType] = SourcePart_e_pg
        
        return self.__dict_phaseField_SourcePart_e_pg[matriceType].copy()
    
    def __get_sysCoord(self):
        """Matrice de changement de base pour chaque element (Ne,3,3)"""

        coordo = self.coordoGlob

        if self.elemType in ["SEG2","SEG3"]:

            points1 = coordo[self.__connect[:,0]]
            points2 = coordo[self.__connect[:,1]]

        elif self.elemType in ["TRI3","TRI6"]:

            points1 = coordo[self.__connect[:,0]]
            points2 = coordo[self.__connect[:,1]]
            points3 = coordo[self.__connect[:,2]]

        elif self.elemType in ["QUAD4","QUAD8"]:

            points1 = coordo[self.__connect[:,0]]
            points2 = coordo[self.__connect[:,1]]
            points3 = coordo[self.__connect[:,3]]

        if self.dim in [0,3]:
            sysCoord_e = np.eye(3)
            sysCoord_e = sysCoord_e[np.newaxis, :].repeat(self.Ne, axis=0)
            sysCoordLocal_e = sysCoord_e
        
        elif self.dim in [1,2]:

            i = points2-points1
            # Normalise
            i = np.einsum('ei,e->ei',i, 1/np.linalg.norm(i, axis=1), optimize='optimal')

            if self.dim == 1:
                theta = np.pi/2
                rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
                j = np.einsum('ij,ej->ei',rot, i, optimize='optimal')
            else:                    
                j = points3-points1
                j = np.einsum('ei,e->ei',j, 1/np.linalg.norm(j, axis=1), optimize='optimal')
                
            k = np.cross(i, j, axis=1)
            k = np.einsum('ei,e->ei',k, 1/np.linalg.norm(k, axis=1), optimize='optimal')

            if self.inDim == 3:
                j = np.cross(k, i, axis=1)
                j = np.einsum('ei,e->ei',j, 1/np.linalg.norm(j, axis=1), optimize='optimal')


            sysCoord_e = np.zeros((self.Ne, 3, 3))

            # sysCoord_e[:,0] = i
            # sysCoord_e[:,1] = j
            # sysCoord_e[:,2] = k

            sysCoord_e[:,:,0] = i
            sysCoord_e[:,:,1] = j
            sysCoord_e[:,:,2] = k

        return sysCoord_e
        
    @property
    def sysCoord_e(self) -> np.ndarray:
        """matrice de changement de base pour chaque element (3D)\n
        [ix, jx, kx\n
        iy, jy, ky\n
        iz, jz, kz]\n
        
        tel que coordo_e . sysCoordLocal_e -> coordonneés des noeuds dans la base de l'elements"""
        return self.__get_sysCoord()

    @property
    def sysCoordLocal_e(self) -> np.ndarray:
        """matrice de changement de base pour chaque element (2D)"""
        return self.sysCoord_e[:,:,range(self.dim)]

    @property
    def aire(self) -> float:
        """Aire que représente les elements"""
        if self.dim == 1: return
        aire = np.einsum('ep,p->', self.get_jacobien_e_pg("rigi"), self.get_gauss("rigi").poids, optimize='optimal')
        return float(aire)

    @property
    def Ix(self) -> float:
        """Moment quadratique suivant x"""
        if self.dim != 2: return

        coordo_e_p = self.get_coordo_e_p("masse", self.elementsIndex)
        x = coordo_e_p[self.elementsID, :, 0]

        Ix = np.einsum('ep,p,ep->', self.get_jacobien_e_pg("masse"), self.get_gauss("masse").poids, x**2, optimize='optimal')
        return float(Ix)

    @property
    def Iy(self) -> float:
        """Moment quadratique suivant y"""
        if self.dim != 2: return

        coordo_e_p = self.get_coordo_e_p("masse", self.elementsIndex)
        y = coordo_e_p[self.elementsID, :, 1]

        Iy = np.einsum('ep,p,ep->', self.get_jacobien_e_pg("masse"), self.get_gauss("masse").poids, y**2, optimize='optimal')
        return float(Iy)

    @property
    def Ixy(self) -> float:
        """Moment quadratique suivant xy"""
        if self.dim != 2: return

        coordo_e_p = self.get_coordo_e_p("masse", self.elementsIndex)
        x = coordo_e_p[self.elementsID, :, 0]
        y = coordo_e_p[self.elementsID, :, 1]

        Ixy = np.einsum('ep,p,ep,ep->', self.get_jacobien_e_pg("masse"), self.get_gauss("masse").poids, x, y, optimize='optimal')
        return float(Ixy)

    @property
    def volume(self) -> float:
        """Volume que représente les elements"""
        if self.dim != 3: return
        volume = np.einsum('ep,p->', self.get_jacobien_e_pg("rigi"), self.get_gauss("rigi").poids, optimize='optimal')
        return float(volume)
        

    def get_F_e_pg(self, matriceType: str) -> np.ndarray:
        """Renvoie la matrice jacobienne\n
        Cette matrice décrit les variations des axes de l'element de reference a l'element reel\n
        Permet la transformation de l'element de référence à l'element réel avec invF_e_pg"""
        if self.dim == 0: return
        if matriceType not in self.__dict_F_e_pg.keys():

            coordo_n = self.coordoGlob[:]

            coordo_e = coordo_n[self.__connect]

            dim = self.dim
            if dim == 1:
                dimCheck = 2
            else:
                dimCheck = 3

            if dim == self.inDim:
                nodesBase = coordo_e.copy()               
            else:
                sysCoordLocal_e = self.sysCoordLocal_e # matrice de changement de base pour chaque element
                nodesBase = np.einsum('eij,ejk->eik', coordo_e, sysCoordLocal_e, optimize='optimal') #coordonneés des noeuds dans la base de l'elements

            nodesBaseDim = nodesBase[:,:,range(dim)]

            dN_pg = self.get_dN_pg(matriceType)

            F_e_pg = np.array(np.einsum('pik,ekj->epij', dN_pg, nodesBaseDim, optimize='optimal'))
            
            self.__dict_F_e_pg[matriceType] = F_e_pg

        return self.__dict_F_e_pg[matriceType].copy()
    
    def get_jacobien_e_pg(self, matriceType:str) -> np.ndarray:
        """Renvoie les jacobiens\n
        variation de taille (aire ou volume) entre l'element de référence et l'element réel
        """
        if self.dim == 0: return
        if matriceType not in self.__dict_jacobien_e_pg.keys():

            F_e_pg = self.get_F_e_pg(matriceType)

            if self.dim == 1:
                Ne = F_e_pg.shape[0]
                nPg = F_e_pg.shape[1]
                jacobien_e_pg = F_e_pg.reshape((Ne, nPg))

            elif self.dim == 2:
                a_e_pg = F_e_pg[:,:,0,0]
                b_e_pg = F_e_pg[:,:,0,1]
                c_e_pg = F_e_pg[:,:,1,0]
                d_e_pg = F_e_pg[:,:,1,1]
                jacobien_e_pg = (a_e_pg*d_e_pg)-(c_e_pg*b_e_pg)
            
            elif self.dim == 3:
                a11_e_pg = F_e_pg[:,:,0,0]; a12_e_pg = F_e_pg[:,:,0,1]; a13_e_pg = F_e_pg[:,:,0,2]
                a21_e_pg = F_e_pg[:,:,1,0]; a22_e_pg = F_e_pg[:,:,1,1]; a23_e_pg = F_e_pg[:,:,1,2]
                a31_e_pg = F_e_pg[:,:,2,0]; a32_e_pg = F_e_pg[:,:,2,1]; a33_e_pg = F_e_pg[:,:,2,2]

                jacobien_e_pg = a11_e_pg * ((a22_e_pg*a33_e_pg)-(a32_e_pg*a23_e_pg)) - a12_e_pg * ((a21_e_pg*a33_e_pg)-(a31_e_pg*a23_e_pg)) + a13_e_pg * ((a21_e_pg*a32_e_pg)-(a31_e_pg*a22_e_pg))

            # jacobien_e_pg = np.linalg.det(F_e_pg) - jacobien_e_pg

            self.__dict_jacobien_e_pg[matriceType] = jacobien_e_pg

        return self.__dict_jacobien_e_pg[matriceType].copy()
    
    def get_invF_e_pg(self, matriceType: str) -> np.ndarray:
        """Renvoie l'inverse de la matrice jacobienne\n
        est utlisée pour obtenir la derivée des fonctions de formes dN_e_pg dans l'element réeel\n
        dN_e_pg = invF_e_pg . dN_pg
        """
        if self.dim == 0: return
        if matriceType not in self.__dict_invF_e_pg.keys():

            F_e_pg = self.get_F_e_pg(matriceType)

            if self.dim == 1:
                invF_e_pg = 1/F_e_pg
            elif self.dim == 2:
                # A = [alpha, beta          inv(A) = 1/det * [b, -beta
                #      a    , b   ]                           -a  alpha]

                Ne = F_e_pg.shape[0]
                nPg = F_e_pg.shape[1]
                invF_e_pg = np.zeros((Ne,nPg,2,2))

                det = self.get_jacobien_e_pg(matriceType)

                alpha = F_e_pg[:,:,0,0]
                beta = F_e_pg[:,:,0,1]
                a = F_e_pg[:,:,1,0]
                b = F_e_pg[:,:,1,1]

                invF_e_pg[:,:,0,0] = b
                invF_e_pg[:,:,0,1] = -beta
                invF_e_pg[:,:,1,0] = -a
                invF_e_pg[:,:,1,1] = alpha

                invF_e_pg = np.einsum('ep,epij->epij',1/det, invF_e_pg, optimize='optimal')
            elif self.dim == 3:
                # optimisé tel que invF_e_pg = 1/det * Adj(F_e_pg)
                # https://fr.wikihow.com/calculer-l'inverse-d'une-matrice-3x3

                det = self.get_jacobien_e_pg(matriceType)

                FT_e_pg = np.einsum('epij->epji', F_e_pg, optimize='optimal')

                a00 = FT_e_pg[:,:,0,0]; a01 = FT_e_pg[:,:,0,1]; a02 = FT_e_pg[:,:,0,2]
                a10 = FT_e_pg[:,:,1,0]; a11 = FT_e_pg[:,:,1,1]; a12 = FT_e_pg[:,:,1,2]
                a20 = FT_e_pg[:,:,2,0]; a21 = FT_e_pg[:,:,2,1]; a22 = FT_e_pg[:,:,2,2]

                det00 = (a11*a22) - (a21*a12); det01 = (a10*a22) - (a20*a12); det02 = (a10*a21) - (a20*a11)
                det10 = (a01*a22) - (a21*a02); det11 = (a00*a22) - (a20*a02); det12 = (a00*a21) - (a20*a01)
                det20 = (a01*a12) - (a11*a02); det21 = (a00*a12) - (a10*a02); det22 = (a00*a11) - (a10*a01)

                invF_e_pg = np.zeros_like(F_e_pg)

                # Ne pas oublier les - ou  + !!!
                invF_e_pg[:,:,0,0] = det00/det; invF_e_pg[:,:,0,1] = -det01/det; invF_e_pg[:,:,0,2] = det02/det
                invF_e_pg[:,:,1,0] = -det10/det; invF_e_pg[:,:,1,1] = det11/det; invF_e_pg[:,:,1,2] = -det12/det
                invF_e_pg[:,:,2,0] = det20/det; invF_e_pg[:,:,2,1] = -det21/det; invF_e_pg[:,:,2,2] = det22/det

                # invF_e_pg = np.array(np.linalg.inv(F_e_pg)) - invF_e_pg

            self.__dict_invF_e_pg[matriceType] = invF_e_pg

        return self.__dict_invF_e_pg[matriceType].copy()

    def __get_N_pg(self, matriceType: str) -> np.ndarray:
        """Fonctions de formes vectorielles (pg), dans la base (ksi, eta ...)\n
        [N1, N2, . . . ,Nn]
        """
        if self.dim == 0: return

        if self.elemType == "SEG2":

            N1t = lambda x: 0.5*(1-x)
            N2t = lambda x: 0.5*(1+x)

            Ntild = np.array([N1t, N2t])
        
        elif self.elemType == "SEG3":

            N1t = lambda x: -0.5*(1-x)*x
            N2t = lambda x: 0.5*(1+x)*x
            N3t = lambda x: (1+x)*(1-x)

            Ntild = np.array([N1t, N2t, N3t])

        elif self.elemType == "TRI3":

            N1t = lambda ksi,eta: 1-ksi-eta
            N2t = lambda ksi,eta: ksi
            N3t = lambda ksi,eta: eta
            
            Ntild = np.array([N1t, N2t, N3t])

        elif self.elemType == "TRI6":

            N1t = lambda ksi,eta: -(1-ksi-eta)*(1-2*(1-ksi-eta))
            N2t = lambda ksi,eta: -ksi*(1-2*ksi)
            N3t = lambda ksi,eta: -eta*(1-2*eta)
            N4t = lambda ksi,eta: 4*ksi*(1-ksi-eta)
            N5t = lambda ksi,eta: 4*ksi*eta
            N6t = lambda ksi,eta: 4*eta*(1-ksi-eta)
            
            Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t])
        
        elif self.elemType == "QUAD4":

            N1t = lambda ksi,eta: (1-ksi)*(1-eta)/4
            N2t = lambda ksi,eta: (1+ksi)*(1-eta)/4
            N3t = lambda ksi,eta: (1+ksi)*(1+eta)/4
            N4t = lambda ksi,eta: (1-ksi)*(1+eta)/4
            
            Ntild = np.array([N1t, N2t, N3t, N4t])

        elif self.elemType == "QUAD8":

            N1t = lambda ksi,eta: (1-ksi)*(1-eta)*(-1-ksi-eta)/4
            N2t = lambda ksi,eta: (1+ksi)*(1-eta)*(-1+ksi-eta)/4
            N3t = lambda ksi,eta: (1+ksi)*(1+eta)*(-1+ksi+eta)/4
            N4t = lambda ksi,eta: (1-ksi)*(1+eta)*(-1-ksi+eta)/4
            N5t = lambda ksi,eta: (1-ksi**2)*(1-eta)/2
            N6t = lambda ksi,eta: (1+ksi)*(1-eta**2)/2
            N7t = lambda ksi,eta: (1-ksi**2)*(1+eta)/2
            N8t = lambda ksi,eta: (1-ksi)*(1-eta**2)/2
            
            Ntild =  np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t])                    

        elif self.elemType == "TETRA4":

            N1t = lambda x,y,z: 1-x-y-z
            N2t = lambda x,y,z: x
            N3t = lambda x,y,z: y
            N4t = lambda x,y,z: z

            Ntild = np.array([N1t, N2t, N3t, N4t])

        elif self.elemType == "HEXA8":

            N1t = lambda x,y,z: 1/8 * (1-x) * (1-y) * (1-z)
            N2t = lambda x,y,z: 1/8 * (1+x) * (1-y) * (1-z)
            N3t = lambda x,y,z: 1/8 * (1+x) * (1+y) * (1-z)
            N4t = lambda x,y,z: 1/8 * (1-x) * (1+y) * (1-z)
            N5t = lambda x,y,z: 1/8 * (1-x) * (1-y) * (1+z)
            N6t = lambda x,y,z: 1/8 * (1+x) * (1-y) * (1+z)
            N7t = lambda x,y,z: 1/8 * (1+x) * (1+y) * (1+z)
            N8t = lambda x,y,z: 1/8 * (1-x) * (1+y) * (1+z)

            Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t])

        elif self.elemType == "PRISM6":

            N1t = lambda x,y,z: 1/2 * y * (1-x)
            N2t = lambda x,y,z: 1/2 * z * (1-x)
            N3t = lambda x,y,z: 1/2 * (1-y-z) * (1-x)
            N4t = lambda x,y,z: 1/2 * y * (1+x)
            N5t = lambda x,y,z: 1/2 * z * (1+x)
            N6t = lambda x,y,z: 1/2 * (1-y-z) * (1+x)
            
            # Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t])
            Ntild = np.array([N3t, N1t, N2t, N6t, N4t, N5t])
        
        else:
            raise "Element inconnue"

        
        # Evalue aux points de gauss

        gauss = self.get_gauss(matriceType)            
        coord = gauss.coord
        nPg = gauss.nPg

        N_pg = np.zeros((nPg, 1, len(Ntild)))

        for pg in range(nPg):
            for n, Nt in enumerate(Ntild):                    
                if coord.shape[1] == 1:
                    N_pg[pg, 0, n] = Nt(coord[pg,0])
                elif coord.shape[1] == 2:
                    N_pg[pg, 0, n] = Nt(coord[pg,0], coord[pg,1])
                elif coord.shape[1] == 3:
                    N_pg[pg, 0, n] = Nt(coord[pg,0], coord[pg,1], coord[pg,2])

        return N_pg

    def get_dNv_pg(self, matriceType: str) -> np.ndarray:
        """Fonctions de formes dans l'element poutre en flexion (pg, dim, nPe), dans la base (ksi) \n
        [Nv_i . . . Nv_n\n
        Nrz_i . . . Nrz_n]
        """
        if self.dim == 0: return

        if self.elemType == "SEG2":

            Nv1t = lambda x: 1/4 * (1-x)**2 * (2+x)
            Nv2t = lambda x: 1/8 * (1+x) * (1-x)**2
            Nv3t = lambda x: 1/4 * (1+x)**2 * (2-x)
            Nv4t = lambda x: 1/8 * (1+x)**2 * (x-1)

            Nvtild = np.array([Nv1t, Nv2t, Nv3t, Nv4t])
        
        else:
            raise "Pas implémenté"
        
        # Evaluation aux points de gauss
        gauss = self.get_gauss(matriceType)
        coord = gauss.coord
        
        nPg = gauss.nPg

        dNv_pg = np.zeros((nPg, 1, len(Nvtild)))

        for pg in range(nPg):
            for n, Nt in enumerate(Nvtild):
                func = Nt
                dNv_pg[pg, 0, n] = func(coord[pg,0])

        return dNv_pg
    
    def get_dN_pg(self, matriceType: str) -> np.ndarray:
        """Dérivées des fonctions de formes dans l'element de référence (pg, dim, nPe), dans la base (ksi, eta ...) \n
        [Ni,ksi . . . Nn,ksi\n
        Ni,eta . . . Nn,eta]
        """
        if self.dim == 0: return

        if self.elemType == "SEG2":

            dN1t = [lambda x: -0.5]
            dN2t = [lambda x: 0.5]

            dNtild = np.array([dN1t, dN2t])
        
        elif self.elemType == "SEG3":

            dN1t = [lambda x: x-0.5]
            dN2t = [lambda x: x+0.5]
            dN3t = [lambda x: -2*x]

            dNtild = np.array([dN1t, dN2t, dN3t])

        elif self.elemType == "TRI3":

            dN1t = [lambda ksi,eta: -1, lambda ksi,eta: -1]
            dN2t = [lambda ksi,eta: 1,  lambda ksi,eta: 0]
            dN3t = [lambda ksi,eta: 0,  lambda ksi,eta: 1]

            dNtild = np.array([dN1t, dN2t, dN3t])

        elif self.elemType == "TRI6":

            dN1t = [lambda ksi,eta: 4*ksi+4*eta-3,  lambda ksi,eta: 4*ksi+4*eta-3]
            dN2t = [lambda ksi,eta: 4*ksi-1,        lambda ksi,eta: 0]
            dN3t = [lambda ksi,eta: 0,              lambda ksi,eta: 4*eta-1]
            dN4t = [lambda ksi,eta: 4-8*ksi-4*eta,  lambda ksi,eta: -4*ksi]
            dN5t = [lambda ksi,eta: 4*eta,          lambda ksi,eta: 4*ksi]
            dN6t = [lambda ksi,eta: -4*eta,         lambda ksi,eta: 4-4*ksi-8*eta]
            
            dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t])
        
        elif self.elemType == "QUAD4":
            
            dN1t = [lambda ksi,eta: (eta-1)/4,  lambda ksi,eta: (ksi-1)/4]
            dN2t = [lambda ksi,eta: (1-eta)/4,  lambda ksi,eta: (-ksi-1)/4]
            dN3t = [lambda ksi,eta: (1+eta)/4,  lambda ksi,eta: (1+ksi)/4]
            dN4t = [lambda ksi,eta: (-eta-1)/4, lambda ksi,eta: (1-ksi)/4]
            
            dNtild = [dN1t, dN2t, dN3t, dN4t]

        elif self.elemType == "QUAD8":
            
            dN1t = [lambda ksi,eta: (1-eta)*(2*ksi+eta)/4,      lambda ksi,eta: (1-ksi)*(ksi+2*eta)/4]
            dN2t = [lambda ksi,eta: (1-eta)*(2*ksi-eta)/4,      lambda ksi,eta: -(1+ksi)*(ksi-2*eta)/4]
            dN3t = [lambda ksi,eta: (1+eta)*(2*ksi+eta)/4,      lambda ksi,eta: (1+ksi)*(ksi+2*eta)/4]
            dN4t = [lambda ksi,eta: -(1+eta)*(-2*ksi+eta)/4,    lambda ksi,eta: (1-ksi)*(-ksi+2*eta)/4]
            dN5t = [lambda ksi,eta: -ksi*(1-eta),               lambda ksi,eta: -(1-ksi**2)/2]
            dN6t = [lambda ksi,eta: (1-eta**2)/2,               lambda ksi,eta: -eta*(1+ksi)]
            dN7t = [lambda ksi,eta: -ksi*(1+eta),               lambda ksi,eta: (1-ksi**2)/2]
            dN8t = [lambda ksi,eta: -(1-eta**2)/2,              lambda ksi,eta: -eta*(1-ksi)]
                            
            dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t])

        elif self.elemType == "TETRA4":
            
            dN1t = [lambda x,y,z: -1,   lambda x,y,z: -1,   lambda x,y,z: -1]
            dN2t = [lambda x,y,z: 1,    lambda x,y,z: 0,    lambda x,y,z: 0]
            dN3t = [lambda x,y,z: 0,    lambda x,y,z: 1,    lambda x,y,z: 0]
            dN4t = [lambda x,y,z: 0,    lambda x,y,z: 0,    lambda x,y,z: 1]

            dNtild = np.array([dN1t, dN2t, dN3t, dN4t])

        elif self.elemType == "HEXA8":
            
            dN1t = [lambda x,y,z: -1/8 * (1-y) * (1-z),   lambda x,y,z: -1/8 * (1-x) * (1-z),   lambda x,y,z: -1/8 * (1-x) * (1-y)]
            dN2t = [lambda x,y,z: 1/8 * (1-y) * (1-z),    lambda x,y,z: -1/8 * (1+x) * (1-z),    lambda x,y,z: -1/8 * (1+x) * (1-y)]
            dN3t = [lambda x,y,z: 1/8 * (1+y) * (1-z),    lambda x,y,z: 1/8 * (1+x) * (1-z),    lambda x,y,z: -1/8 * (1+x) * (1+y)]
            dN4t = [lambda x,y,z: -1/8 * (1+y) * (1-z),    lambda x,y,z: 1/8 * (1-x) * (1-z),    lambda x,y,z: -1/8 * (1-x) * (1+y)]
            dN5t = [lambda x,y,z: -1/8 * (1-y) * (1+z),    lambda x,y,z: -1/8 * (1-x) * (1+z),    lambda x,y,z: 1/8 * (1-x) * (1-y)]
            dN6t = [lambda x,y,z: 1/8 * (1-y) * (1+z),    lambda x,y,z: -1/8 * (1+x) * (1+z),    lambda x,y,z: 1/8 * (1+x) * (1-y)]
            dN7t = [lambda x,y,z: 1/8 * (1+y) * (1+z),    lambda x,y,z: 1/8 * (1+x) * (1+z),    lambda x,y,z: 1/8 * (1+x) * (1+y)]
            dN8t = [lambda x,y,z: -1/8 * (1+y) * (1+z),    lambda x,y,z: 1/8 * (1-x) * (1+z),    lambda x,y,z: 1/8 * (1-x) * (1+y)]

            dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t])
        
        elif self.elemType == "PRISM6":

            dN1t = [lambda x,y,z: -1/2 * y,         lambda x,y,z: 1/2 * (1-x),      lambda x,y,z: 0]
            dN2t = [lambda x,y,z: -1/2 * z,         lambda x,y,z: 0,                lambda x,y,z: 1/2 * (1-x)]
            dN3t = [lambda x,y,z: -1/2 * (1-y-z),   lambda x,y,z: -1/2 * (1-x),     lambda x,y,z: -1/2 * (1-x)]
            dN4t = [lambda x,y,z: 1/2 * y,          lambda x,y,z: 1/2 * (1+x),      lambda x,y,z: 0]
            dN5t = [lambda x,y,z: 1/2 * z,          lambda x,y,z: 0,                lambda x,y,z: 1/2 * (1+x)]
            dN6t = [lambda x,y,z: 1/2 * (1-y-z),    lambda x,y,z: -1/2 * (1+x),     lambda x,y,z: -1/2 * (1+x)]

            # dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t])
            dNtild = np.array([dN3t, dN1t, dN2t, dN6t, dN4t, dN5t])
            

        else:
            raise "Element inconnue"
            
        
        # Evaluation aux points de gauss
        gauss = self.get_gauss(matriceType)
        coord = gauss.coord

        dim = self.dim
        nPg = gauss.nPg

        dN_pg = np.zeros((nPg, dim, len(dNtild)))

        for pg in range(nPg):
            for n, Nt in enumerate(dNtild):
                for d in range(dim):
                    func = Nt[d]
                    if coord.shape[1] == 1:
                        dN_pg[pg, d, n] = func(coord[pg,0])
                    elif coord.shape[1] == 2:
                        dN_pg[pg, d, n] = func(coord[pg,0], coord[pg,1])
                    elif coord.shape[1] == 3:
                        dN_pg[pg, d, n] = func(coord[pg,0], coord[pg,1], coord[pg,2])

        return dN_pg

    def get_dNv_pg(self, matriceType: str) -> np.ndarray:
        """Dérivées des fonctions de formes dans l'element poutre en flexion (pg, dim, nPe), dans la base (ksi) \n
        [Nv_i,ksi . . . Nv_n,ksi\n
        Nrz_i,ksi . . . Nrz_n,ksi]
        """
        if self.dim == 0: return

        if self.elemType == "SEG2":

            dNv1t = lambda x: -3/4 * (1-x) * (1+x)
            dNv2t = lambda x: -1/8 * (1-x) * (1+3*x)
            dNv3t = lambda x: 3/4 * (1-x) * (1+x)
            dNv4t = lambda x: 1/8 * (1+x) * (3*x-1)

            dNvtild = np.array([dNv1t, dNv2t, dNv3t, dNv4t])
        
        else:
            raise "Pas implémenté"
        
        # Evaluation aux points de gauss
        gauss = self.get_gauss(matriceType)
        coord = gauss.coord
        
        nPg = gauss.nPg

        dNv_pg = np.zeros((nPg, 1, len(dNvtild)))

        for pg in range(nPg):
            for n, Nt in enumerate(dNvtild):
                func = Nt
                dNv_pg[pg, 0, n] = func(coord[pg,0])

        return dNv_pg

    def get_ddNv_pg(self, matriceType: str) -> np.ndarray:
        """Dérivées des fonctions de formes dans l'element poutre en flexion (pg, dim, nPe), dans la base (ksi) \n
        [Nv_i,ksi ksi . . . Nv_n,ksi ksi\n
        Nrz_i,ksi ksi . . . Nrz_n,ksi ksi]
        """
        if self.dim != 1: return

        if self.elemType in ["SEG2"]:

            ddNv1t = lambda x: 3/2 * x
            ddNv2t = lambda x: 1/4 * (3*x-1)
            ddNv3t = lambda x: -3/2 * x
            ddNv4t = lambda x: 1/4 * (3*x+1)

            ddNvtild = np.array([ddNv1t, ddNv2t, ddNv3t, ddNv4t])
        
        else:
            raise "Pas implémenté"
        
        # Evaluation aux points de gauss
        gauss = self.get_gauss(matriceType)
        coord = gauss.coord
        
        nPg = gauss.nPg

        ddNv_pg = np.zeros((nPg, 1, len(ddNvtild)))

        for pg in range(nPg):
            for n, Nt in enumerate(ddNvtild):
                func = Nt
                ddNv_pg[pg, 0, n] = func(coord[pg,0])
        
        return ddNv_pg

    def get_ddN_pg(self, matriceType: str) -> np.ndarray:
        """Dérivées segonde des fonctions de formes dans l'element de référence (pg, dim, nPe), dans la base (ksi, eta ...) \n
        [Ni,ksi ksi . . . Nn,ksi ksi\n
        Ni,eta eta . . . Nn,eta eta]
        """
        if self.dim == 0: return

        if self.elemType == "SEG2":

            ddN1t = [lambda x: 0]
            ddN2t = [lambda x: 0]

            ddNtild = np.array([ddN1t, ddN2t])
        
        elif self.elemType == "SEG3":

            ddN1t = [lambda x: 1]
            ddN2t = [lambda x: 1]
            ddN3t = [lambda x: -2]

            ddNtild = np.array([ddN1t, ddN2t, ddN3t])

        elif self.elemType == "TRI3":

            ddN1t = [lambda ksi,eta: 0, lambda ksi,eta: 0]
            ddN2t = [lambda ksi,eta: 0, lambda ksi,eta: 0]
            ddN3t = [lambda ksi,eta: 0, lambda ksi,eta: 0]

            ddNtild = np.array([ddN1t, ddN2t, ddN3t])

        elif self.elemType == "TRI6":

            ddN1t = [lambda ksi,eta: 4,  lambda ksi,eta: 4]
            ddN2t = [lambda ksi,eta: 4,  lambda ksi,eta: 0]
            ddN3t = [lambda ksi,eta: 0,  lambda ksi,eta: 4]
            ddN4t = [lambda ksi,eta: -8, lambda ksi,eta: 0]
            ddN5t = [lambda ksi,eta: 0,  lambda ksi,eta: 0]
            ddN6t = [lambda ksi,eta: 0,  lambda ksi,eta: -8]
            
            ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t, ddN5t, ddN6t])
        
        elif self.elemType == "QUAD4":
            
            ddN1t = [lambda ksi,eta: 0,  lambda ksi,eta: 0]
            ddN2t = [lambda ksi,eta: 0,  lambda ksi,eta: 0]
            ddN3t = [lambda ksi,eta: 0,  lambda ksi,eta: 0]
            ddN4t = [lambda ksi,eta: 0,  lambda ksi,eta: 0]
            
            ddNtild = [ddN1t, ddN2t, ddN3t, ddN4t]

        elif self.elemType == "QUAD8":
            
            ddN1t = [lambda ksi,eta: (1-eta)/2,  lambda ksi,eta: (1-ksi)/2]
            ddN2t = [lambda ksi,eta: (1-eta)/2,  lambda ksi,eta: (1+ksi)/2]
            ddN3t = [lambda ksi,eta: (1+eta)/2,  lambda ksi,eta: (1+ksi)/2]
            ddN4t = [lambda ksi,eta: (1+eta)/2,  lambda ksi,eta: (1-ksi)/2]
            ddN5t = [lambda ksi,eta: -1+eta,     lambda ksi,eta: 0]
            ddN6t = [lambda ksi,eta: 0,          lambda ksi,eta: -1-ksi]
            ddN7t = [lambda ksi,eta: -1-eta,     lambda ksi,eta: 0]
            ddN8t = [lambda ksi,eta: 0,          lambda ksi,eta: -1+ksi]
                            
            ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t, ddN5t, ddN6t, ddN7t, ddN8t])

        elif self.elemType == "TETRA4":
            
            ddN1t = [lambda x,y,z: 0,    lambda x,y,z: 0,    lambda x,y,z: 0]
            ddN2t = [lambda x,y,z: 0,    lambda x,y,z: 0,    lambda x,y,z: 0]
            ddN3t = [lambda x,y,z: 0,    lambda x,y,z: 0,    lambda x,y,z: 0]
            ddN4t = [lambda x,y,z: 0,    lambda x,y,z: 0,    lambda x,y,z: 0]

            ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t])

        elif self.elemType == "HEXA8":
            
            ddN1t = [lambda x,y,z: 0,    lambda x,y,z: 0,   lambda x,y,z: 0]
            ddN2t = [lambda x,y,z: 0,    lambda x,y,z: 0,   lambda x,y,z: 0]
            ddN3t = [lambda x,y,z: 0,    lambda x,y,z: 0,   lambda x,y,z: 0]
            ddN4t = [lambda x,y,z: 0,    lambda x,y,z: 0,   lambda x,y,z: 0]
            ddN5t = [lambda x,y,z: 0,    lambda x,y,z: 0,   lambda x,y,z: 0]
            ddN6t = [lambda x,y,z: 0,    lambda x,y,z: 0,   lambda x,y,z: 0]
            ddN7t = [lambda x,y,z: 0,    lambda x,y,z: 0,   lambda x,y,z: 0]
            ddN8t = [lambda x,y,z: 0,    lambda x,y,z: 0,   lambda x,y,z: 0]

            ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t, ddN5t, ddN6t, ddN7t, ddN8t])
        
        elif self.elemType == "PRISM6":

            ddN1t = [lambda x,y,z: 0,    lambda x,y,z: 0,   lambda x,y,z: 0]
            ddN2t = [lambda x,y,z: 0,    lambda x,y,z: 0,   lambda x,y,z: 0]
            ddN3t = [lambda x,y,z: 0,    lambda x,y,z: 0,   lambda x,y,z: 0]
            ddN4t = [lambda x,y,z: 0,    lambda x,y,z: 0,   lambda x,y,z: 0]
            ddN5t = [lambda x,y,z: 0,    lambda x,y,z: 0,   lambda x,y,z: 0]
            ddN6t = [lambda x,y,z: 0,    lambda x,y,z: 0,   lambda x,y,z: 0]

            # dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t])
            # Attetion il faut faire une réorganisation
            ddNtild = np.array([ddN3t, ddN1t, ddN2t, ddN6t, ddN4t, ddN5t])

        else:
            raise "Element inconnue"
            
        
        # Evaluation aux points de gauss
        gauss = self.get_gauss(matriceType)
        coord = gauss.coord

        dim = self.dim
        nPg = gauss.nPg

        dddN_pg = np.zeros((nPg, dim, len(ddNtild)))

        for pg in range(nPg):
            for n, Nt in enumerate(ddNtild):
                for d in range(dim):
                    func = Nt[d]                        
                    if coord.shape[1] == 1:
                        dddN_pg[pg, d, n] = func(coord[pg,0])
                    elif coord.shape[1] == 2:
                        dddN_pg[pg, d, n] = func(coord[pg,0], coord[pg,1])
                    elif coord.shape[1] == 3:
                        dddN_pg[pg, d, n] = func(coord[pg,0], coord[pg,1], coord[pg,2])

        return dddN_pg

    def Get_Nodes_Conditions(self, conditionX=True, conditionY=True, conditionZ=True) -> np.ndarray:
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
        verifX = isinstance(conditionX, bool)
        verifY = isinstance(conditionY, bool)
        verifZ = isinstance(conditionZ, bool)

        listNoeud = list(range(self.Nn))
        if verifX and verifY and verifZ:
            return listNoeud

        coordo = self.__coordo

        coordoX = coordo[:,0]
        coordoY = coordo[:,1]
        coordoZ = coordo[:,2]
        
        arrayVrai = np.array([True]*self.Nn)
        
        # Verification suivant X
        if verifX:
            valideConditionX = arrayVrai
        else:
            try:
                valideConditionX = conditionX(coordoX)
            except:
                valideConditionX = [conditionX(coordoX[n]) for n in listNoeud]

        # Verification suivant Y
        if verifY:
            valideConditionY = arrayVrai
        else:
            try:
                valideConditionY = conditionY(coordoY)
            except:
                valideConditionY = [conditionY(coordoY[n]) for n in listNoeud]
        
        # Verification suivant Z
        if verifZ:
            valideConditionZ = arrayVrai
        else:
            try:
                valideConditionZ = conditionZ(coordoZ)
            except:
                valideConditionZ = [conditionZ(coordoZ[n]) for n in listNoeud]
        
        conditionsTotal = valideConditionX * valideConditionY * valideConditionZ

        nodesIndex = np.where(conditionsTotal)[0]
        
        return self.__nodesID[nodesIndex].copy()
    
    def Get_Nodes_Point(self, point: Point) -> np.ndarray:
        """Renvoie l'identifiant du noeud qui est sur le point"""

        coordo = self.__coordo

        nodesIndex = np.where((coordo[:,0] == point.x) & (coordo[:,1] == point.y) & (coordo[:,2] == point.z))[0]

        return self.__nodesID[nodesIndex].copy()

    def Get_Nodes_Line(self, line: Line) -> np.ndarray:
        """Renvoie la liste d'identifiant des noeuds qui sont sur la ligne"""
        
        vectUnitaire = line.vecteurUnitaire

        coordo = self.__coordo

        vect = coordo-line.coordo[0]

        prodScalaire = np.einsum('i,ni-> n', vectUnitaire, vect, optimize='optimal')
        prodVecteur = np.cross(vect, vectUnitaire)
        norm = np.linalg.norm(prodVecteur, axis=1)

        eps = np.finfo(float).eps

        nodesIndex = np.where((norm<eps) & (prodScalaire>=-eps) & (prodScalaire<=line.length+eps))[0]

        return self.__nodesID[nodesIndex].copy()
    
    def Get_Nodes_Domain(self, domain: Domain) -> np.ndarray:
        """Renvoie la liste de noeuds qui sont dans le domaine"""

        coordo = self.__coordo

        eps = np.finfo(float).eps

        nodesIndex = np.where(  (coordo[:,0] >= domain.pt1.x-eps) & (coordo[:,0] <= domain.pt2.x+eps) &
                            (coordo[:,1] >= domain.pt1.y-eps) & (coordo[:,1] <= domain.pt2.y+eps) &
                            (coordo[:,2] >= domain.pt1.z-eps) & (coordo[:,2] <= domain.pt2.z+eps))[0]
        
        return self.__nodesID[nodesIndex].copy()

    def Get_Nodes_Circle(self, circle: Circle) -> np.ndarray:
        """Renvoie la liste de noeuds qui sont dans le cercle"""

        coordo = self.__coordo

        eps = np.finfo(float).eps

        nodesIndex = np.where(np.sqrt((coordo[:,0]-circle.center.x)**2+(coordo[:,1]-circle.center.y)**2+(coordo[:,2]-circle.center.z)**2)<=circle.diam/2+eps)

        return self.__nodesID[nodesIndex]

    def Get_Nodes_Cylindre(self, circle: Circle, direction=[0,0,1]) -> np.ndarray:
        """Renvoie la liste de noeuds qui sont dans le cylindre"""

        coordo = self.__coordo

        eps = np.finfo(float).eps
        dx, dy, dz = direction[0], direction[1], direction[2]
        # Ne fonctionne certainement pas pour le moment pour un cylindre orienté !

        if dx == 0:
            conditionX = coordo[:,0]-circle.center.x
        else:
            conditionX = np.zeros_like(coordo[:,0])

        if dy == 0:
            conditionY = coordo[:,1]-circle.center.y
        else:
            conditionY = np.zeros_like(coordo[:,1])
        
        if dz == 0:
            conditionZ = coordo[:,2]-circle.center.z
        else:
            conditionZ = np.zeros_like(coordo[:,2])

        nodesIndex = np.where(np.sqrt(conditionX**2+conditionY**2+conditionZ**2)<=circle.diam/2+eps)

        return self.__nodesID[nodesIndex]
    
    def Localise_sol_e(self, sol: np.ndarray) -> np.ndarray:
        """localise les valeurs de noeuds sur les elements"""
        tailleVecteur = self.Nn * self.dim

        if sol.shape[0] == tailleVecteur:
            sol_e = sol[self.assembly_e]
        else:
            sol_e = sol[self.__connect]
        
        return sol_e

    def get_connectTriangle(self) -> np.ndarray:
        """Transforme la matrice de connectivité pour la passer dans la fonction trisurf en 2D\n
        Par exemple pour un quadrangle on construit deux triangles
        pour un triangle à 6 noeuds on construit 4 triangles\n

        Renvoie un dictionnaire par type
        """
        assert self.dim == 2
        dict_connect_triangle = {}
        if self.elemType == "TRI3":
            dict_connect_triangle[self.elemType] = self.__connect[:,[0,1,2]]
        elif self.elemType == "TRI6":
            dict_connect_triangle[self.elemType] = np.array(self.__connect[:, [0,3,5,3,1,4,5,4,2,3,4,5]]).reshape(-1,3)
        elif self.elemType == "QUAD4":
            dict_connect_triangle[self.elemType] = np.array(self.__connect[:, [0,1,3,1,2,3]]).reshape(-1,3)
        elif self.elemType == "QUAD8":
            dict_connect_triangle[self.elemType] = np.array(self.__connect[:, [4,5,7,5,6,7,0,4,7,4,1,5,5,2,6,6,3,7]]).reshape(-1,3)
        else:
            raise "Element inconnue"

        return dict_connect_triangle

    def get_connect_Faces(self) -> dict:
        """Récupère les identifiants des noeud constuisant les faces et renvoie les faces pour chaque types d'elements

        Returns
        -------
        list de list
            Renvoie une liste de face
        """

        dict_connect_faces = {}

        nPe = self.nPe            
        if self.elemType in ["SEG2","SEG3","POINT"]:
            dict_connect_faces[self.elemType] = self.__connect.copy()
        elif self.elemType == "TRI3":
            dict_connect_faces[self.elemType] = self.__connect[:, [0,1,2,0]]
        elif self.elemType == "TRI6":
            dict_connect_faces[self.elemType] = self.__connect[:, [0,3,1,4,2,5,0]]
        elif self.elemType == "QUAD4":
            dict_connect_faces[self.elemType] = self.__connect[:, [0,1,2,3,0]]
        elif self.elemType == "QUAD8":
            dict_connect_faces[self.elemType] = self.__connect[:, [0,4,1,5,2,6,3,7,0]]
        elif self.elemType == "TETRA4":
            # Ici par elexemple on va creer 3 faces, chaque face est composé des identifiants des noeuds
            dict_connect_faces[self.elemType] = np.array(self.__connect[:, [0,1,2,0,1,3,0,2,3,1,2,3]]).reshape(self.Ne*nPe,-1)
        elif self.elemType == "HEXA8":
            # Ici par elexemple on va creer 6 faces, chaque face est composé des identifiants des noeuds                
            dict_connect_faces[self.elemType] = np.array(self.__connect[:, [0,1,2,3,0,1,5,4,0,3,7,4,6,2,3,7,6,2,1,5,6,7,4,5]]).reshape(-1,nPe)
        elif self.elemType == "PRISM6":
            # Ici il faut faire attention parce que cette element est composé de 2 triangles et 3 quadrangles
            dict_connect_faces["QUAD4"] = np.array(self.__connect[:, [0,2,5,3,0,1,4,3,1,2,5,4]]).reshape(-1,4)
            dict_connect_faces["TRI3"] = np.array(self.__connect[:, [0,1,2,3,4,5]]).reshape(-1,3)
            
        else:
            raise "Element inconnue"

        return dict_connect_faces

    ################################################ STATIC ##################################################

    @staticmethod
    def get_MatriceType() -> List[str]:
        """type de matrice disponible"""
        liste = ["rigi", "masse","beam"]
        return liste

    @staticmethod
    def get_Types1D() -> List[str]:
        """type d'elements disponibles en 1D"""
        liste1D = ["SEG2", "SEG3"]
        return liste1D

    @staticmethod
    def get_Types2D() -> List[str]:
        """type d'elements disponibles en 2D"""
        liste2D = ["TRI3", "TRI6", "QUAD4", "QUAD8"]
        return liste2D
    
    @staticmethod
    def get_Types3D() -> List[str]:
        """type d'elements disponibles en 3D"""
        liste3D = ["TETRA4", "HEXA8", "PRISM6"]
        return liste3D

    @staticmethod
    def Get_ElemInFos(gmshId: int) -> tuple:
        """Renvoie le nom le nombre de noeuds par element et la dimension de l'élement en fonction du gmshId

        Args:
            type (int): type de l'identifiant sur gmsh

        Returns:
            tuple: (type, nPe, dim)
        """
        if gmshId == 1:
            type = "SEG2"; nPe = 2; dim = 1
        elif gmshId == 2:
            type = "TRI3"; nPe = 3; dim = 2
        elif gmshId == 3:
            type = "QUAD4"; nPe = 4; dim = 2 
        elif gmshId == 4:
            type = "TETRA4"; nPe = 4; dim = 3
        elif gmshId == 5:
            type = "HEXA8"; nPe = 8; dim = 3
        elif gmshId == 6:
            type = "PRISM6"; nPe = 6; dim = 3
        elif gmshId == 7:
            type = "PYRA5"; nPe = 5; dim = 3
        elif gmshId == 8:
            type = "SEG3"; nPe = 3; dim = 1
        elif gmshId == 9:
            type = "TRI6"; nPe = 6; dim = 2
        elif gmshId == 10:
            type = "QUAD9"; nPe = 9; dim = 2
        elif gmshId == 11:
            type = "TETRA10"; nPe = 10; dim = 3
        elif gmshId == 12:
            type = "CUBE27"; nPe = 27; dim = 3
        elif gmshId == 13:
            type = "PRISM18"; nPe = 18; dim = 3
        elif gmshId == 14:
            type = "PYRA14"; nPe = 17; dim = 3
        elif gmshId == 15:
            type = "POINT"; nPe = 1; dim = 0
        elif gmshId == 16:
            type = "QUAD8"; nPe = 8; dim = 2
        elif gmshId == 18:
            type = "PRISM15"; nPe = 15; dim = 3
        elif gmshId == 19:
            type = "PYRA13"; nPe = 13; dim = 3
        else: 
            raise "Type inconnue"
            
        return type, nPe, dim
        

