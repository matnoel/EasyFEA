from enum import Enum
from typing import List, cast

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

    def __init__(self, gmshId: int, connect: np.ndarray, elementsID: np.ndarray, coordoGlob: np.ndarray, nodesID: np.ndarray):
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

        self.__elemType, self.__nPe, self.__dim, self.__ordre, self.__nbFaces = GroupElem.Get_ElemInFos(gmshId)
        
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
        
        if self.elemType in GroupElem.get_Types3D():
            self.__inDim = 3
        else:
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
        self.__dict_nodes_tags = {}
        self.__dict_elements_tags = {}
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
    def elemType(self):
        """Type d'element"""
        return self.__elemType

    @property
    def nPe(self) -> int:
        """Nombre de noeuds par element"""
        return self.__nPe
    
    @property
    def dim(self) -> int:
        """Dimension de l'element"""
        return self.__dim
    
    @property
    def ordre(self) -> int:
        """Ordre de l'element"""
        return self.__ordre

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
        return self.__nbFaces
    
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
        
        # Verifie si il n'y a pas de noeuds en trop
        # Il est possible que les noeuds renseignés n'appartiennent pas au groupe
        if self.Nn < noeuds.max():
            # On enlève tout les noeuds en trop
            indexNoeudsSansDepassement = np.where(noeuds < self.Nn)[0]
            noeuds = noeuds[indexNoeudsSansDepassement]

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
            matriceType (str): [MatriceType.rigi,MatriceType.masse]
            isScalaire (bool): type de matrice N\n

        Returns:\n
        . Fonctions de formes vectorielles (pg, rep=2, rep=2*dim), dans la base (ksi, eta ...)\n
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
            Ne = self.Ne
            nPe = self.nPe
            pg = self.get_gauss(matriceType)
            
            # On récupère la longeur des poutres sur chaque element aux points d'intégrations
            # l = np.einsum('ep,p->', jacobien_e_pg, pg.poids, optimize='optimal')
            l_e_pg = np.einsum('ep,p->e', jacobien_e_pg, pg.poids, optimize='optimal').reshape(Ne,1).repeat(invF_e_pg.shape[1], axis=1)
            
            ddNv_e_pg = np.einsum('epik,epik,pkj->epij', invF_e_pg, invF_e_pg, ddNv_pg, optimize='optimal')
            
            colonnes = np.arange(1, nPe*2, 2)
            for colonne in colonnes:
                ddNv_e_pg[:,:,0,colonne] = np.einsum('ep,ep->ep', ddNv_e_pg[:,:,0,colonne], l_e_pg, optimize='optimal')

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

        if self.elemType in [ElemType.SEG2,ElemType.SEG3,ElemType.SEG4,ElemType.SEG5]:

            points1 = coordo[self.__connect[:,0]]
            points2 = coordo[self.__connect[:,1]]

        elif self.elemType in [ElemType.TRI3,ElemType.TRI6,ElemType.TRI10,ElemType.TRI15]:

            points1 = coordo[self.__connect[:,0]]
            points2 = coordo[self.__connect[:,1]]
            points3 = coordo[self.__connect[:,2]]

        elif self.elemType in [ElemType.QUAD4,ElemType.QUAD8]:

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

                e1 = np.array([1, 0, 0])[np.newaxis, :].repeat(i.shape[0], axis=0)
                e2 = np.array([0, 1, 0])[np.newaxis, :].repeat(i.shape[0], axis=0)
                e3 = np.array([0, 0, 1])[np.newaxis, :].repeat(i.shape[0], axis=0)

                if self.inDim == 1:
                    j = e2
                    k = e3
                elif self.inDim == 2:
                    theta = np.pi/2
                    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
                    j = np.einsum('ij,ej->ei',rot, i, optimize='optimal')
                    j = np.einsum('ei,e->ei',j, 1/np.linalg.norm(j, axis=1), optimize='optimal')

                    k = np.cross(i, j, axis=1)
                    k = np.einsum('ei,e->ei',k, 1/np.linalg.norm(k, axis=1), optimize='optimal')
                elif self.inDim == 3:
                    j = np.cross(i, e1, axis=1)

                    rep2 = np.where(np.linalg.norm(j, axis=1)<1e-12)
                    rep1 = np.setdiff1d(range(i.shape[0]), rep2)

                    k1 = j.copy()
                    j1 = np.cross(k1, i, axis=1)

                    j2 = np.cross(e2, i, axis=1)
                    k2 = np.cross(i, j2, axis=1)

                    j = np.zeros_like(i)
                    j[rep1] = j1[rep1]
                    j[rep2] = j2[rep2]
                    j = np.einsum('ei,e->ei',j, 1/np.linalg.norm(j, axis=1), optimize='optimal')
                    
                    k = np.zeros_like(i)
                    k[rep1] = k1[rep1]
                    k[rep2] = k2[rep2]
                    k = np.einsum('ei,e->ei',k, 1/np.linalg.norm(k, axis=1), optimize='optimal')

            else:                    
                j = points3-points1
                j = np.einsum('ei,e->ei',j, 1/np.linalg.norm(j, axis=1), optimize='optimal')
                
                k = np.cross(i, j, axis=1)
                k = np.einsum('ei,e->ei',k, 1/np.linalg.norm(k, axis=1), optimize='optimal')


            sysCoord_e = np.zeros((self.Ne, 3, 3))
            
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
        aire = np.einsum('ep,p->', self.get_jacobien_e_pg(MatriceType.rigi), self.get_gauss(MatriceType.rigi).poids, optimize='optimal')
        return float(aire)

    @property
    def Ix(self) -> float:
        """Moment quadratique suivant x"""
        if self.dim != 2: return

        coordo_e_p = self.get_coordo_e_p(MatriceType.masse, self.elementsIndex)
        x = coordo_e_p[self.elementsID, :, 0]

        Ix = np.einsum('ep,p,ep->', self.get_jacobien_e_pg(MatriceType.masse), self.get_gauss(MatriceType.masse).poids, x**2, optimize='optimal')
        return float(Ix)

    @property
    def Iy(self) -> float:
        """Moment quadratique suivant y"""
        if self.dim != 2: return

        coordo_e_p = self.get_coordo_e_p(MatriceType.masse, self.elementsIndex)
        y = coordo_e_p[self.elementsID, :, 1]

        Iy = np.einsum('ep,p,ep->', self.get_jacobien_e_pg(MatriceType.masse), self.get_gauss(MatriceType.masse).poids, y**2, optimize='optimal')
        return float(Iy)

    @property
    def Ixy(self) -> float:
        """Moment quadratique suivant xy"""
        if self.dim != 2: return

        coordo_e_p = self.get_coordo_e_p(MatriceType.masse, self.elementsIndex)
        x = coordo_e_p[self.elementsID, :, 0]
        y = coordo_e_p[self.elementsID, :, 1]

        Ixy = np.einsum('ep,p,ep,ep->', self.get_jacobien_e_pg(MatriceType.masse), self.get_gauss(MatriceType.masse).poids, x, y, optimize='optimal')
        return float(Ixy)

    @property
    def volume(self) -> float:
        """Volume que représente les elements"""
        if self.dim != 3: return
        volume = np.einsum('ep,p->', self.get_jacobien_e_pg(MatriceType.masse), self.get_gauss(MatriceType.masse).poids, optimize='optimal')
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

        if self.elemType == ElemType.SEG2:

            N1t = lambda x: 0.5*(1-x)
            N2t = lambda x: 0.5*(1+x)

            Ntild = np.array([N1t, N2t])
        
        elif self.elemType == ElemType.SEG3:

            N1t = lambda x: -0.5*(1-x)*x
            N2t = lambda x: 0.5*(1+x)*x
            N3t = lambda x: (1+x)*(1-x)

            Ntild = np.array([N1t, N2t, N3t])

        elif self.elemType == ElemType.SEG4:

            N1t = lambda x : -0.5625*x**3 + 0.5625*x**2 + 0.0625*x + -0.0625
            N2t = lambda x : 0.5625*x**3 + 0.5625*x**2 + -0.0625*x + -0.0625
            N3t = lambda x : 1.6875*x**3 + -0.5625*x**2 + -1.6875*x + 0.5625
            N4t = lambda x : -1.6875*x**3 + -0.5625*x**2 + 1.6875*x + 0.5625

            Ntild = np.array([N1t, N2t, N3t, N4t])

        elif self.elemType == ElemType.SEG5:

            N1t = lambda x : 0.6667*x**4 + -0.6667*x**3 + -0.1667*x**2 + 0.1667*x + 0.0
            N2t = lambda x : 0.6667*x**4 + 0.6667*x**3 + -0.1667*x**2 + -0.1667*x + 0.0
            N3t = lambda x : -2.667*x**4 + 1.333*x**3 + 2.667*x**2 + -1.333*x + 0.0
            N4t = lambda x : 4.0*x**4 + 0.0*x**3 + -5.0*x**2 + 0.0*x + 1.0
            N5t = lambda x : -2.667*x**4 + -1.333*x**3 + 2.667*x**2 + 1.333*x + 0.0

            Ntild = np.array([N1t, N2t, N3t, N4t, N5t])

        elif self.elemType == ElemType.TRI3:

            N1t = lambda ksi,eta: 1-ksi-eta
            N2t = lambda ksi,eta: ksi
            N3t = lambda ksi,eta: eta
            
            Ntild = np.array([N1t, N2t, N3t])

        elif self.elemType == ElemType.TRI6:

            N1t = lambda ksi,eta: -(1-ksi-eta)*(1-2*(1-ksi-eta))
            N2t = lambda ksi,eta: -ksi*(1-2*ksi)
            N3t = lambda ksi,eta: -eta*(1-2*eta)
            N4t = lambda ksi,eta: 4*ksi*(1-ksi-eta)
            N5t = lambda ksi,eta: 4*ksi*eta
            N6t = lambda ksi,eta: 4*eta*(1-ksi-eta)
            
            Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t])

        elif self.elemType == ElemType.TRI10:

            N1t = lambda ksi, eta : -4.5*ksi**3 + -4.5*eta**3 + -13.5*ksi**2*eta + -13.5*ksi*eta**2 + 9.0*ksi**2 + 9.0*eta**2 + 18.0*ksi*eta + -5.5*ksi + -5.5*eta + 1.0
            N2t = lambda ksi, eta : 4.5*ksi**3 + 0.0*eta**3 + -1.093e-15*ksi**2*eta + -8.119e-16*ksi*eta**2 + -4.5*ksi**2 + 0.0*eta**2 + 1.124e-15*ksi*eta + 1.0*ksi + 0.0*eta + 0.0
            N3t = lambda ksi, eta : 0.0*ksi**3 + 4.5*eta**3 + -3.747e-16*ksi**2*eta + 2.998e-15*ksi*eta**2 + 0.0*ksi**2 + -4.5*eta**2 + -7.494e-16*ksi*eta + 0.0*ksi + 1.0*eta + 0.0
            N4t = lambda ksi, eta : 13.5*ksi**3 + 0.0*eta**3 + 27.0*ksi**2*eta + 13.5*ksi*eta**2 + -22.5*ksi**2 + 0.0*eta**2 + -22.5*ksi*eta + 9.0*ksi + 0.0*eta + 0.0
            N5t = lambda ksi, eta : -13.5*ksi**3 + 0.0*eta**3 + -13.5*ksi**2*eta + -4.247e-15*ksi*eta**2 + 18.0*ksi**2 + 0.0*eta**2 + 4.5*ksi*eta + -4.5*ksi + 0.0*eta + 0.0
            N6t = lambda ksi, eta : 0.0*ksi**3 + 0.0*eta**3 + 13.5*ksi**2*eta + 1.049e-14*ksi*eta**2 + 0.0*ksi**2 + 0.0*eta**2 + -4.5*ksi*eta + 0.0*ksi + 0.0*eta + 0.0
            N7t = lambda ksi, eta : 0.0*ksi**3 + 0.0*eta**3 + 0.0*ksi**2*eta + 13.5*ksi*eta**2 + 0.0*ksi**2 + 0.0*eta**2 + -4.5*ksi*eta + 0.0*ksi + 0.0*eta + 0.0
            N8t = lambda ksi, eta : 0.0*ksi**3 + -13.5*eta**3 + -1.499e-15*ksi**2*eta + -13.5*ksi*eta**2 + 0.0*ksi**2 + 18.0*eta**2 + 4.5*ksi*eta + 0.0*ksi + -4.5*eta + 0.0
            N9t = lambda ksi, eta : 0.0*ksi**3 + 13.5*eta**3 + 13.5*ksi**2*eta + 27.0*ksi*eta**2 + 0.0*ksi**2 + -22.5*eta**2 + -22.5*ksi*eta + 0.0*ksi + 9.0*eta + 0.0
            N10t = lambda ksi, eta : 0.0*ksi**3 + 0.0*eta**3 + -27.0*ksi**2*eta + -27.0*ksi*eta**2 + 0.0*ksi**2 + 0.0*eta**2 + 27.0*ksi*eta + 0.0*ksi + 0.0*eta + 0.0
            
            Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t, N9t, N10t])

        elif self.elemType == ElemType.TRI15:

            N1t = lambda ksi, eta : 10.67*ksi**4 + 42.67*ksi**3*eta + 64.0*ksi**2**eta**2 + 42.67*ksi*eta**3 + 10.67*eta**4 + -26.67*ksi**3 + -80.0*ksi**2*eta + -80.0*ksi*eta**2 + -26.67*eta**3 + 23.33*ksi**2 + 46.67*ksi*eta + 23.33*eta**2 + -8.333*ksi + -8.333*eta + 1.0
            N2t = lambda ksi, eta : 10.67*ksi**4 + -5.222e-15*ksi**3*eta + -2.665e-15*ksi**2**eta**2 + -1.85e-15*ksi*eta**3 + 0.0*eta**4 + -16.0*ksi**3 + 7.401e-15*ksi**2*eta + 4.737e-15*ksi*eta**2 + 0.0*eta**3 + 7.333*ksi**2 + -3.331e-15*ksi*eta + 0.0*eta**2 + -1.0*ksi + 0.0*eta + 0.0
            N3t = lambda ksi, eta : 0.0*ksi**4 + 6.513e-15*ksi**3*eta + 3.138e-14*ksi**2**eta**2 + 2.842e-14*ksi*eta**3 + 10.67*eta**4 + 0.0*ksi**3 + -1.342e-14*ksi**2*eta + -3.257e-14*ksi*eta**2 + -16.0*eta**3 + 0.0*ksi**2 + 6.661e-15*ksi*eta + 7.333*eta**2 + 0.0*ksi + -1.0*eta + 0.0
            N4t = lambda ksi, eta : -42.67*ksi**4 + -128.0*ksi**3*eta + -128.0*ksi**2**eta**2 + -42.67*ksi*eta**3 + 0.0*eta**4 + 96.0*ksi**3 + 192.0*ksi**2*eta + 96.0*ksi*eta**2 + 0.0*eta**3 + -69.33*ksi**2 + -69.33*ksi*eta + 0.0*eta**2 + 16.0*ksi + 0.0*eta + 0.0
            N5t = lambda ksi, eta : 64.0*ksi**4 + 128.0*ksi**3*eta + 64.0*ksi**2**eta**2 + -7.638e-14*ksi*eta**3 + 0.0*eta**4 + -128.0*ksi**3 + -144.0*ksi**2*eta + -16.0*ksi*eta**2 + 0.0*eta**3 + 76.0*ksi**2 + 28.0*ksi*eta + 0.0*eta**2 + -12.0*ksi + 0.0*eta + 0.0
            N6t = lambda ksi, eta : -42.67*ksi**4 + -42.67*ksi**3*eta + -1.54e-14*ksi**2**eta**2 + 2.22e-14*ksi*eta**3 + 0.0*eta**4 + 74.67*ksi**3 + 32.0*ksi**2*eta + -2.724e-14*ksi*eta**2 + 0.0*eta**3 + -37.33*ksi**2 + -5.333*ksi*eta + 0.0*eta**2 + 5.333*ksi + 0.0*eta + 0.0
            N7t = lambda ksi, eta : 0.0*ksi**4 + 42.67*ksi**3*eta + 4.855e-14*ksi**2**eta**2 + -2.043e-14*ksi*eta**3 + 0.0*eta**4 + 0.0*ksi**3 + -32.0*ksi**2*eta + 7.105e-15*ksi*eta**2 + 0.0*eta**3 + 0.0*ksi**2 + 5.333*ksi*eta + 0.0*eta**2 + 0.0*ksi + 0.0*eta + 0.0
            N8t = lambda ksi, eta : 0.0*ksi**4 + 0.0*ksi**3*eta + 64.0*ksi**2**eta**2 + 2.842e-14*ksi*eta**3 + 0.0*eta**4 + 0.0*ksi**3 + -16.0*ksi**2*eta + -16.0*ksi*eta**2 + 0.0*eta**3 + 0.0*ksi**2 + 4.0*ksi*eta + 0.0*eta**2 + 0.0*ksi + 0.0*eta + 0.0
            N9t = lambda ksi, eta : 0.0*ksi**4 + 1.118e-14*ksi**3*eta + 1.066e-14*ksi**2**eta**2 + 42.67*ksi*eta**3 + 0.0*eta**4 + 0.0*ksi**3 + -1.421e-14*ksi**2*eta + -32.0*ksi*eta**2 + 0.0*eta**3 + 0.0*ksi**2 + 5.333*ksi*eta + 0.0*eta**2 + 0.0*ksi + 0.0*eta + 0.0
            N10t = lambda ksi, eta : 0.0*ksi**4 + -2.053e-14*ksi**3*eta + -2.132e-13*ksi**2**eta**2 + -42.67*ksi*eta**3 + -42.67*eta**4 + 0.0*ksi**3 + 7.816e-14*ksi**2*eta + 32.0*ksi*eta**2 + 74.67*eta**3 + 0.0*ksi**2 + -5.333*ksi*eta + -37.33*eta**2 + 0.0*ksi + 5.333*eta + 0.0
            N11t = lambda ksi, eta : 0.0*ksi**4 + 8.421e-14*ksi**3*eta + 64.0*ksi**2**eta**2 + 128.0*ksi*eta**3 + 64.0*eta**4 + 0.0*ksi**3 + -16.0*ksi**2*eta + -144.0*ksi*eta**2 + -128.0*eta**3 + 0.0*ksi**2 + 28.0*ksi*eta + 76.0*eta**2 + 0.0*ksi + -12.0*eta + 0.0
            N12t = lambda ksi, eta : 0.0*ksi**4 + -42.67*ksi**3*eta + -128.0*ksi**2**eta**2 + -128.0*ksi*eta**3 + -42.67*eta**4 + 0.0*ksi**3 + 96.0*ksi**2*eta + 192.0*ksi*eta**2 + 96.0*eta**3 + 0.0*ksi**2 + -69.33*ksi*eta + -69.33*eta**2 + 0.0*ksi + 16.0*eta + 0.0
            N13t = lambda ksi, eta : 0.0*ksi**4 + 128.0*ksi**3*eta + 256.0*ksi**2**eta**2 + 128.0*ksi*eta**3 + 0.0*eta**4 + 0.0*ksi**3 + -224.0*ksi**2*eta + -224.0*ksi*eta**2 + 0.0*eta**3 + 0.0*ksi**2 + 96.0*ksi*eta + 0.0*eta**2 + 0.0*ksi + 0.0*eta + 0.0
            N14t = lambda ksi, eta : 0.0*ksi**4 + -128.0*ksi**3*eta + -128.0*ksi**2**eta**2 + 4.974e-14*ksi*eta**3 + 0.0*eta**4 + 0.0*ksi**3 + 160.0*ksi**2*eta + 32.0*ksi*eta**2 + 0.0*eta**3 + 0.0*ksi**2 + -32.0*ksi*eta + 0.0*eta**2 + 0.0*ksi + 0.0*eta + 0.0
            N15t = lambda ksi, eta : 0.0*ksi**4 + -6.737e-14*ksi**3*eta + -128.0*ksi**2**eta**2 + -128.0*ksi*eta**3 + 0.0*eta**4 + 0.0*ksi**3 + 32.0*ksi**2*eta + 160.0*ksi*eta**2 + 0.0*eta**3 + 0.0*ksi**2 + -32.0*ksi*eta + 0.0*eta**2 + 0.0*ksi + 0.0*eta + 0.0

            Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t, N9t, N10t, N11t, N12t, N13t, N14t, N15t])
        
        elif self.elemType == ElemType.QUAD4:

            N1t = lambda ksi,eta: (1-ksi)*(1-eta)/4
            N2t = lambda ksi,eta: (1+ksi)*(1-eta)/4
            N3t = lambda ksi,eta: (1+ksi)*(1+eta)/4
            N4t = lambda ksi,eta: (1-ksi)*(1+eta)/4
            
            Ntild = np.array([N1t, N2t, N3t, N4t])

        elif self.elemType == ElemType.QUAD8:

            N1t = lambda ksi,eta: (1-ksi)*(1-eta)*(-1-ksi-eta)/4
            N2t = lambda ksi,eta: (1+ksi)*(1-eta)*(-1+ksi-eta)/4
            N3t = lambda ksi,eta: (1+ksi)*(1+eta)*(-1+ksi+eta)/4
            N4t = lambda ksi,eta: (1-ksi)*(1+eta)*(-1-ksi+eta)/4
            N5t = lambda ksi,eta: (1-ksi**2)*(1-eta)/2
            N6t = lambda ksi,eta: (1+ksi)*(1-eta**2)/2
            N7t = lambda ksi,eta: (1-ksi**2)*(1+eta)/2
            N8t = lambda ksi,eta: (1-ksi)*(1-eta**2)/2
            
            Ntild =  np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t])                    

        elif self.elemType == ElemType.TETRA4:

            N1t = lambda x,y,z: 1-x-y-z
            N2t = lambda x,y,z: x
            N3t = lambda x,y,z: y
            N4t = lambda x,y,z: z

            Ntild = np.array([N1t, N2t, N3t, N4t])

        elif self.elemType == ElemType.HEXA8:

            N1t = lambda x,y,z: 1/8 * (1-x) * (1-y) * (1-z)
            N2t = lambda x,y,z: 1/8 * (1+x) * (1-y) * (1-z)
            N3t = lambda x,y,z: 1/8 * (1+x) * (1+y) * (1-z)
            N4t = lambda x,y,z: 1/8 * (1-x) * (1+y) * (1-z)
            N5t = lambda x,y,z: 1/8 * (1-x) * (1-y) * (1+z)
            N6t = lambda x,y,z: 1/8 * (1+x) * (1-y) * (1+z)
            N7t = lambda x,y,z: 1/8 * (1+x) * (1+y) * (1+z)
            N8t = lambda x,y,z: 1/8 * (1-x) * (1+y) * (1+z)

            Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t])

        elif self.elemType == ElemType.PRISM6:

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

    def get_Nv_pg(self, matriceType: str) -> np.ndarray:
        """Fonctions de formes dans l'element poutre en flexion (pg, dim, nPe), dans la base (ksi) \n
        [phi_i psi_i . . . phi_n psi_n]
        """
        if self.dim != 1: return

        if self.elemType == ElemType.SEG2:

            phi_1 = lambda x : 0.5 + -0.75*x + 0.0*x**2 + 0.25*x**3
            psi_1 = lambda x : 0.125 + -0.125*x + -0.125*x**2 + 0.125*x**3
            phi_2 = lambda x : 0.5 + 0.75*x + 0.0*x**2 + -0.25*x**3
            psi_2 = lambda x : -0.125 + -0.125*x + 0.125*x**2 + 0.125*x**3

            Nvtild = np.array([phi_1, psi_1, phi_2, psi_2])
        
        elif self.elemType == ElemType.SEG3:

            phi_1 = lambda x : 0.0 + 0.0*x + 1.0*x**2 + -1.25*x**3 + -0.5*x**4 + 0.75*x**5
            psi_1 = lambda x : 0.0 + 0.0*x + 0.125*x**2 + -0.125*x**3 + -0.125*x**4 + 0.125*x**5
            phi_2 = lambda x : 0.0 + 0.0*x + 1.0*x**2 + 1.25*x**3 + -0.5*x**4 + -0.75*x**5
            psi_2 = lambda x : 0.0 + 0.0*x + -0.125*x**2 + -0.125*x**3 + 0.125*x**4 + 0.125*x**5
            phi_3 = lambda x : 1.0 + 0.0*x + -2.0*x**2 + 0.0*x**3 + 1.0*x**4 + 0.0*x**5
            psi_3 = lambda x : 0.0 + 0.5*x + 0.0*x**2 + -1.0*x**3 + 0.0*x**4 + 0.5*x**5

            Nvtild = np.array([phi_1, psi_1, phi_2, psi_2, phi_3, psi_3])

        elif self.elemType == ElemType.SEG4:

            phi_1 = lambda x : 0.025390624999999556 + -0.029296874999997335*x + -0.4746093750000018*x**2 + 0.548828124999992*x**3 + 2.3730468750000036*x**4 + -2.7597656249999916*x**5 + -1.4238281250000018*x**6 + 1.740234374999997*x**7
            psi_1 = lambda x : 0.0019531250000000555 + -0.0019531249999997224*x + -0.03710937500000017*x**2 + 0.03710937499999917*x**3 + 0.19335937500000028*x**4 + -0.19335937499999917*x**5 + -0.15820312500000014*x**6 + 0.15820312499999972*x**7
            phi_2 = lambda x : 0.025390625 + 0.02929687499999778*x + -0.47460937499999734*x**2 + -0.5488281249999911*x**3 + 2.373046874999995*x**4 + 2.75976562499999*x**5 + -1.4238281249999976*x**6 + -1.7402343749999962*x**7
            psi_2 = lambda x : -0.001953125 + -0.0019531249999998335*x + 0.03710937499999983*x**2 + 0.03710937499999928*x**3 + -0.19335937499999967*x**4 + -0.19335937499999908*x**5 + 0.15820312499999983*x**6 + 0.15820312499999964*x**7
            phi_3 = lambda x : 0.474609375 + -2.373046874999991*x + 0.4746093749999929*x**2 + 9.017578124999972*x**3 + -2.3730468749999845*x**4 + -10.91601562499997*x**5 + 1.4238281249999922*x**6 + 4.271484374999989*x**7
            psi_3 = lambda x : 0.05273437499999978 + -0.1582031249999971*x + -0.5800781250000018*x**2 + 1.7402343749999911*x**3 + 1.001953125000004*x**4 + -3.0058593749999907*x**5 + -0.4746093750000019*x**6 + 1.4238281249999967*x**7
            phi_4 = lambda x : 0.4746093749999991 + 2.3730468749999902*x + 0.4746093750000089*x**2 + -9.017578124999972*x**3 + -2.373046875000015*x**4 + 10.916015624999972*x**5 + 1.423828125000007*x**6 + -4.27148437499999*x**7
            psi_4 = lambda x : -0.05273437500000022 + -0.15820312499999734*x + 0.5800781249999978*x**2 + 1.7402343749999911*x**3 + -1.0019531249999953*x**4 + -3.0058593749999902*x**5 + 0.47460937499999767*x**6 + 1.4238281249999964*x**7

            Nvtild = np.array([phi_1, psi_1, phi_2, psi_2, phi_3, psi_3, phi_4, psi_4])

        elif self.elemType == ElemType.SEG5:

            phi_1 = lambda x : 8.882e-16 + 8.882e-16*x + 0.2593*x**2 + -0.287*x**3 + -2.278*x**4 + 2.528*x**5 + 5.778*x**6 + -6.444*x**7 + -3.259*x**8 + 3.704*x**9
            psi_1 = lambda x : 0.0 + 0.0*x + 0.0*x**2 + 0.0*x**3 + 0.0*x**4 + 0.0*x**5 + 0.0*x**6 + 0.0*x**7 + 0.0*x**8 + 0.0*x**9
            phi_2 = lambda x : 1.332e-15 + -8.882e-16*x + 0.2593*x**2 + 0.287*x**3 + -2.278*x**4 + -2.528*x**5 + 5.778*x**6 + 6.444*x**7 + -3.259*x**8 + -3.704*x**9
            psi_2 = lambda x : 0.0 + 0.0*x + 0.0*x**2 + 0.0*x**3 + 0.0*x**4 + 0.0*x**5 + 0.0*x**6 + 0.0*x**7 + 0.0*x**8 + 0.0*x**9
            phi_3 = lambda x : -3.553e-15 + 0.0*x + 4.741*x**2 + -13.04*x**3 + -14.22*x**4 + 49.78*x**5 + 14.22*x**6 + -60.44*x**7 + -4.741*x**8 + 23.7*x**9
            psi_3 = lambda x : 0.0 + 0.0*x + 0.0*x**2 + 0.0*x**3 + 0.0*x**4 + 0.0*x**5 + 0.0*x**6 + 0.0*x**7 + 0.0*x**8 + 0.0*x**9
            phi_4 = lambda x : 1.0 + 0.0*x + -10.0*x**2 + 0.0*x**3 + 33.0*x**4 + 0.0*x**5 + -40.0*x**6 + 0.0*x**7 + 16.0*x**8 + 0.0*x**9
            psi_4 = lambda x : 0.0 + 0.0*x + 0.0*x**2 + 0.0*x**3 + 0.0*x**4 + 0.0*x**5 + 0.0*x**6 + 0.0*x**7 + 0.0*x**8 + 0.0*x**9
            phi_5 = lambda x : -3.553e-15 + 0.0*x + 4.741*x**2 + 13.04*x**3 + -14.22*x**4 + -49.78*x**5 + 14.22*x**6 + 60.44*x**7 + -4.741*x**8 + -23.7*x**9
            psi_5 = lambda x : 0.0 + 0.0*x + 0.0*x**2 + 0.0*x**3 + 0.0*x**4 + 0.0*x**5 + 0.0*x**6 + 0.0*x**7 + 0.0*x**8 + 0.0*x**9

            Nvtild = np.array([phi_1, psi_1, phi_2, psi_2, phi_3, psi_3, phi_4, psi_4, phi_5, psi_5])

        else:
            raise "Pas implémenté"
        
        # Evaluation aux points de gauss
        gauss = self.get_gauss(matriceType)
        coord = gauss.coord
        
        nPg = gauss.nPg

        Nv_pg = np.zeros((nPg, 1, len(Nvtild)))

        for pg in range(nPg):
            for n, Nt in enumerate(Nvtild):
                func = Nt
                Nv_pg[pg, 0, n] = func(coord[pg,0])

        return Nv_pg
    
    def get_dN_pg(self, matriceType: str) -> np.ndarray:
        """Dérivées des fonctions de formes dans l'element de référence (pg, dim, nPe), dans la base (ksi, eta ...) \n
        [Ni,ksi . . . Nn,ksi\n
        Ni,eta . . . Nn,eta]
        """
        if self.dim == 0: return

        if self.elemType == ElemType.SEG2:

            dN1t = [lambda x: -0.5]
            dN2t = [lambda x: 0.5]

            dNtild = np.array([dN1t, dN2t])
        
        elif self.elemType == ElemType.SEG3:

            dN1t = [lambda x: x-0.5]
            dN2t = [lambda x: x+0.5]
            dN3t = [lambda x: -2*x]

            dNtild = np.array([dN1t, dN2t, dN3t])

        elif self.elemType == ElemType.SEG4:

            dN1t = [lambda x : -1.688*x**2 + 1.125*x + 0.0625]
            dN2t = [lambda x : 1.688*x**2 + 1.125*x + -0.0625]
            dN3t = [lambda x : 5.062*x**2 + -1.125*x + -1.688]
            dN4t = [lambda x : -5.062*x**2 + -1.125*x + 1.688]

            dNtild = np.array([dN1t, dN2t, dN3t, dN4t])

        elif self.elemType == ElemType.SEG5:

            dN1t = [lambda x : 2.667*x**3 + -2.0*x**2 + -0.3333*x + 0.1667]
            dN2t = [lambda x : 2.667*x**3 + 2.0*x**2 + -0.3333*x + -0.1667]
            dN3t = [lambda x : -10.67*x**3 + 4.0*x**2 + 5.333*x + -1.333]
            dN4t = [lambda x : 16.0*x**3 + 0.0*x**2 + -10.0*x + 0.0]
            dN5t = [lambda x : -10.67*x**3 + -4.0*x**2 + 5.333*x + 1.333]

            dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t])

        elif self.elemType == ElemType.TRI3:

            dN1t = [lambda ksi,eta: -1, lambda ksi,eta: -1]
            dN2t = [lambda ksi,eta: 1,  lambda ksi,eta: 0]
            dN3t = [lambda ksi,eta: 0,  lambda ksi,eta: 1]

            dNtild = np.array([dN1t, dN2t, dN3t])

        elif self.elemType == ElemType.TRI6:

            dN1t = [lambda ksi,eta: 4*ksi+4*eta-3,  lambda ksi,eta: 4*ksi+4*eta-3]
            dN2t = [lambda ksi,eta: 4*ksi-1,        lambda ksi,eta: 0]
            dN3t = [lambda ksi,eta: 0,              lambda ksi,eta: 4*eta-1]
            dN4t = [lambda ksi,eta: 4-8*ksi-4*eta,  lambda ksi,eta: -4*ksi]
            dN5t = [lambda ksi,eta: 4*eta,          lambda ksi,eta: 4*ksi]
            dN6t = [lambda ksi,eta: -4*eta,         lambda ksi,eta: 4-4*ksi-8*eta]
            
            dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t])

        elif self.elemType == ElemType.TRI10:

            N1_ksi = lambda ksi, eta : -13.5*ksi**2 + -27.0*ksi*eta + -13.5*eta**2 + 18.0*ksi + 18.0*eta + -5.5
            N2_ksi = lambda ksi, eta : 13.5*ksi**2 + -2.186e-15*ksi*eta + -8.119e-16*eta**2 + -9.0*ksi + 1.124e-15*eta + 1.0
            N3_ksi = lambda ksi, eta : 0.0*ksi**2 + -7.494e-16*ksi*eta + 2.998e-15*eta**2 + 0.0*ksi + -7.494e-16*eta + 0.0
            N4_ksi = lambda ksi, eta : 40.5*ksi**2 + 54.0*ksi*eta + 13.5*eta**2 + -45.0*ksi + -22.5*eta + 9.0
            N5_ksi = lambda ksi, eta : -40.5*ksi**2 + -27.0*ksi*eta + -4.247e-15*eta**2 + 36.0*ksi + 4.5*eta + -4.5
            N6_ksi = lambda ksi, eta : 0.0*ksi**2 + 27.0*ksi*eta + 1.049e-14*eta**2 + 0.0*ksi + -4.5*eta + 0.0
            N7_ksi = lambda ksi, eta : 0.0*ksi**2 + 0.0*ksi*eta + 13.5*eta**2 + 0.0*ksi + -4.5*eta + 0.0
            N8_ksi = lambda ksi, eta : 0.0*ksi**2 + -2.998e-15*ksi*eta + -13.5*eta**2 + 0.0*ksi + 4.5*eta + 0.0
            N9_ksi = lambda ksi, eta : 0.0*ksi**2 + 27.0*ksi*eta + 27.0*eta**2 + 0.0*ksi + -22.5*eta + 0.0
            N10_ksi = lambda ksi, eta : 0.0*ksi**2 + -54.0*ksi*eta + -27.0*eta**2 + 0.0*ksi + 27.0*eta + 0.0

            N1_eta = lambda ksi, eta : -13.5*eta**2 + -13.5*ksi**2 + -27.0*ksi*eta + 18.0*eta + 18.0*ksi + -5.5
            N2_eta = lambda ksi, eta : 0.0*eta**2 + -1.093e-15*ksi**2 + -1.624e-15*ksi*eta + 0.0*eta + 1.124e-15*ksi + 0.0
            N3_eta = lambda ksi, eta : 13.5*eta**2 + -3.747e-16*ksi**2 + 5.995e-15*ksi*eta + -9.0*eta + -7.494e-16*ksi + 1.0
            N4_eta = lambda ksi, eta : 0.0*eta**2 + 27.0*ksi**2 + 27.0*ksi*eta + 0.0*eta + -22.5*ksi + 0.0
            N5_eta = lambda ksi, eta : 0.0*eta**2 + -13.5*ksi**2 + -8.493e-15*ksi*eta + 0.0*eta + 4.5*ksi + 0.0
            N6_eta = lambda ksi, eta : 0.0*eta**2 + 13.5*ksi**2 + 2.098e-14*ksi*eta + 0.0*eta + -4.5*ksi + 0.0
            N7_eta = lambda ksi, eta : 0.0*eta**2 + 0.0*ksi**2 + 27.0*ksi*eta + 0.0*eta + -4.5*ksi + 0.0
            N8_eta = lambda ksi, eta : -40.5*eta**2 + -1.499e-15*ksi**2 + -27.0*ksi*eta + 36.0*eta + 4.5*ksi + -4.5
            N9_eta = lambda ksi, eta : 40.5*eta**2 + 13.5*ksi**2 + 54.0*ksi*eta + -45.0*eta + -22.5*ksi + 9.0
            N10_eta = lambda ksi, eta : 0.0*eta**2 + -27.0*ksi**2 + -54.0*ksi*eta + 0.0*eta + 27.0*ksi + 0.0

            dN1t = [N1_ksi, N1_eta]
            dN2t = [N2_ksi, N2_eta]
            dN3t = [N3_ksi, N3_eta]
            dN4t = [N4_ksi, N4_eta]
            dN5t = [N5_ksi, N5_eta]
            dN6t = [N6_ksi, N6_eta]
            dN7t = [N7_ksi, N7_eta]
            dN8t = [N8_ksi, N8_eta]
            dN9t = [N9_ksi, N9_eta]
            dN10t = [N10_ksi, N10_eta]

            dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t, dN9t, dN10t])

        elif self.elemType == ElemType.TRI15:

            N1_ksi = lambda ksi, eta: 42.67*ksi**3 + 128.0*ksi**2*eta + 128.0*ksi*eta**2 + 42.67*eta**3 + -80.0*ksi**2 + -160.0*ksi*eta + -80.0*eta**2 + 46.67*ksi + 46.67*eta + -8.333
            N2_ksi = lambda ksi, eta: 42.67*ksi**3 + -1.567e-14*ksi**2*eta + -5.329e-15*ksi*eta**2 + -1.85e-15*eta**3 + -48.0*ksi**2 + 1.48e-14*ksi*eta + 4.737e-15*eta**2 + 14.67*ksi + -3.331e-15*eta + -1.0
            N3_ksi = lambda ksi, eta: 0.0*ksi**3 + 1.954e-14*ksi**2*eta + 6.276e-14*ksi*eta**2 + 2.842e-14*eta**3 + 0.0*ksi**2 + -2.683e-14*ksi*eta + -3.257e-14*eta**2 + 0.0*ksi + 6.661e-15*eta + 0.0
            N4_ksi = lambda ksi, eta: -170.7*ksi**3 + -384.0*ksi**2*eta + -256.0*ksi*eta**2 + -42.67*eta**3 + 288.0*ksi**2 + 384.0*ksi*eta + 96.0*eta**2 + -138.7*ksi + -69.33*eta + 16.0
            N5_ksi = lambda ksi, eta: 256.0*ksi**3 + 384.0*ksi**2*eta + 128.0*ksi*eta**2 + -7.638e-14*eta**3 + -384.0*ksi**2 + -288.0*ksi*eta + -16.0*eta**2 + 152.0*ksi + 28.0*eta + -12.0
            N6_ksi = lambda ksi, eta: -170.7*ksi**3 + -128.0*ksi**2*eta + -3.079e-14*ksi*eta**2 + 2.22e-14*eta**3 + 224.0*ksi**2 + 64.0*ksi*eta + -2.724e-14*eta**2 + -74.67*ksi + -5.333*eta + 5.333
            N7_ksi = lambda ksi, eta: 0.0*ksi**3 + 128.0*ksi**2*eta + 9.711e-14*ksi*eta**2 + -2.043e-14*eta**3 + 0.0*ksi**2 + -64.0*ksi*eta + 7.105e-15*eta**2 + 0.0*ksi + 5.333*eta + 0.0
            N8_ksi = lambda ksi, eta: 0.0*ksi**3 + 0.0*ksi**2*eta + 128.0*ksi*eta**2 + 2.842e-14*eta**3 + 0.0*ksi**2 + -32.0*ksi*eta + -16.0*eta**2 + 0.0*ksi + 4.0*eta + 0.0
            N9_ksi = lambda ksi, eta: 0.0*ksi**3 + 3.355e-14*ksi**2*eta + 2.132e-14*ksi*eta**2 + 42.67*eta**3 + 0.0*ksi**2 + -2.842e-14*ksi*eta + -32.0*eta**2 + 0.0*ksi + 5.333*eta + 0.0
            N10_ksi = lambda ksi, eta: 0.0*ksi**3 + -6.158e-14*ksi**2*eta + -4.263e-13*ksi*eta**2 + -42.67*eta**3 + 0.0*ksi**2 + 1.563e-13*ksi*eta + 32.0*eta**2 + 0.0*ksi + -5.333*eta + 0.0
            N11_ksi = lambda ksi, eta: 0.0*ksi**3 + 2.526e-13*ksi**2*eta + 128.0*ksi*eta**2 + 128.0*eta**3 + 0.0*ksi**2 + -32.0*ksi*eta + -144.0*eta**2 + 0.0*ksi + 28.0*eta + 0.0
            N12_ksi = lambda ksi, eta: 0.0*ksi**3 + -128.0*ksi**2*eta + -256.0*ksi*eta**2 + -128.0*eta**3 + 0.0*ksi**2 + 192.0*ksi*eta + 192.0*eta**2 + 0.0*ksi + -69.33*eta + 0.0
            N13_ksi = lambda ksi, eta: 0.0*ksi**3 + 384.0*ksi**2*eta + 512.0*ksi*eta**2 + 128.0*eta**3 + 0.0*ksi**2 + -448.0*ksi*eta + -224.0*eta**2 + 0.0*ksi + 96.0*eta + 0.0
            N14_ksi = lambda ksi, eta: 0.0*ksi**3 + -384.0*ksi**2*eta + -256.0*ksi*eta**2 + 4.974e-14*eta**3 + 0.0*ksi**2 + 320.0*ksi*eta + 32.0*eta**2 + 0.0*ksi + -32.0*eta + 0.0
            N15_ksi = lambda ksi, eta: 0.0*ksi**3 + -2.021e-13*ksi**2*eta + -256.0*ksi*eta**2 + -128.0*eta**3 + 0.0*ksi**2 + 64.0*ksi*eta + 160.0*eta**2 + 0.0*ksi + -32.0*eta + 0.0

            N1_eta = lambda ksi, eta: 42.67*ksi**3 + 128.0*ksi**2*eta + 128.0*ksi*eta**2 + 42.67*eta**3 + -80.0*ksi**2 + -160.0*ksi*eta + -80.0*eta**2 + 46.67*ksi + 46.67*eta + -8.333
            N2_eta = lambda ksi, eta: -5.222e-15*ksi**3 + -5.329e-15*ksi**2*eta + -5.551e-15*ksi*eta**2 + 0.0*eta**3 + 7.401e-15*ksi**2 + 9.474e-15*ksi*eta + 0.0*eta**2 + -3.331e-15*ksi + 0.0*eta + 0.0
            N3_eta = lambda ksi, eta: 6.513e-15*ksi**3 + 6.276e-14*ksi**2*eta + 8.527e-14*ksi*eta**2 + 42.67*eta**3 + -1.342e-14*ksi**2 + -6.513e-14*ksi*eta + -48.0*eta**2 + 6.661e-15*ksi + 14.67*eta + -1.0
            N4_eta = lambda ksi, eta: -128.0*ksi**3 + -256.0*ksi**2*eta + -128.0*ksi*eta**2 + 0.0*eta**3 + 192.0*ksi**2 + 192.0*ksi*eta + 0.0*eta**2 + -69.33*ksi + 0.0*eta + 0.0
            N5_eta = lambda ksi, eta: 128.0*ksi**3 + 128.0*ksi**2*eta + -2.292e-13*ksi*eta**2 + 0.0*eta**3 + -144.0*ksi**2 + -32.0*ksi*eta + 0.0*eta**2 + 28.0*ksi + 0.0*eta + 0.0
            N6_eta = lambda ksi, eta: -42.67*ksi**3 + -3.079e-14*ksi**2*eta + 6.661e-14*ksi*eta**2 + 0.0*eta**3 + 32.0*ksi**2 + -5.447e-14*ksi*eta + 0.0*eta**2 + -5.333*ksi + 0.0*eta + 0.0
            N7_eta = lambda ksi, eta: 42.67*ksi**3 + 9.711e-14*ksi**2*eta + -6.128e-14*ksi*eta**2 + 0.0*eta**3 + -32.0*ksi**2 + 1.421e-14*ksi*eta + 0.0*eta**2 + 5.333*ksi + 0.0*eta + 0.0
            N8_eta = lambda ksi, eta: 0.0*ksi**3 + 128.0*ksi**2*eta + 8.527e-14*ksi*eta**2 + 0.0*eta**3 + -16.0*ksi**2 + -32.0*ksi*eta + 0.0*eta**2 + 4.0*ksi + 0.0*eta + 0.0
            N9_eta = lambda ksi, eta: 1.118e-14*ksi**3 + 2.132e-14*ksi**2*eta + 128.0*ksi*eta**2 + 0.0*eta**3 + -1.421e-14*ksi**2 + -64.0*ksi*eta + 0.0*eta**2 + 5.333*ksi + 0.0*eta + 0.0
            N10_eta = lambda ksi, eta: -2.053e-14*ksi**3 + -4.263e-13*ksi**2*eta + -128.0*ksi*eta**2 + -170.7*eta**3 + 7.816e-14*ksi**2 + 64.0*ksi*eta + 224.0*eta**2 + -5.333*ksi + -74.67*eta + 5.333
            N11_eta = lambda ksi, eta: 8.421e-14*ksi**3 + 128.0*ksi**2*eta + 384.0*ksi*eta**2 + 256.0*eta**3 + -16.0*ksi**2 + -288.0*ksi*eta + -384.0*eta**2 + 28.0*ksi + 152.0*eta + -12.0
            N12_eta = lambda ksi, eta: -42.67*ksi**3 + -256.0*ksi**2*eta + -384.0*ksi*eta**2 + -170.7*eta**3 + 96.0*ksi**2 + 384.0*ksi*eta + 288.0*eta**2 + -69.33*ksi + -138.7*eta + 16.0
            N13_eta = lambda ksi, eta: 128.0*ksi**3 + 512.0*ksi**2*eta + 384.0*ksi*eta**2 + 0.0*eta**3 + -224.0*ksi**2 + -448.0*ksi*eta + 0.0*eta**2 + 96.0*ksi + 0.0*eta + 0.0
            N14_eta = lambda ksi, eta: -128.0*ksi**3 + -256.0*ksi**2*eta + 1.492e-13*ksi*eta**2 + 0.0*eta**3 + 160.0*ksi**2 + 64.0*ksi*eta + 0.0*eta**2 + -32.0*ksi + 0.0*eta + 0.0
            N15_eta = lambda ksi, eta: -6.737e-14*ksi**3 + -256.0*ksi**2*eta + -384.0*ksi*eta**2 + 0.0*eta**3 + 32.0*ksi**2 + 320.0*ksi*eta + 0.0*eta**2 + -32.0*ksi + 0.0*eta + 0.0


            dN1t = [N1_ksi, N1_eta]
            dN2t = [N2_ksi, N2_eta]
            dN3t = [N3_ksi, N3_eta]
            dN4t = [N4_ksi, N4_eta]
            dN5t = [N5_ksi, N5_eta]
            dN6t = [N6_ksi, N6_eta]
            dN7t = [N7_ksi, N7_eta]
            dN8t = [N8_ksi, N8_eta]
            dN9t = [N9_ksi, N9_eta]
            dN10t = [N10_ksi, N10_eta]
            dN11t = [N11_ksi, N11_eta]
            dN12t = [N12_ksi, N12_eta]
            dN13t = [N13_ksi, N13_eta]
            dN14t = [N14_ksi, N14_eta]
            dN15t = [N15_ksi, N15_eta]


            dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t, dN9t, dN10t, dN11t, dN12t, dN13t, dN14t, dN15t])
        
        elif self.elemType == ElemType.QUAD4:
            
            dN1t = [lambda ksi,eta: (eta-1)/4,  lambda ksi,eta: (ksi-1)/4]
            dN2t = [lambda ksi,eta: (1-eta)/4,  lambda ksi,eta: (-ksi-1)/4]
            dN3t = [lambda ksi,eta: (1+eta)/4,  lambda ksi,eta: (1+ksi)/4]
            dN4t = [lambda ksi,eta: (-eta-1)/4, lambda ksi,eta: (1-ksi)/4]
            
            dNtild = [dN1t, dN2t, dN3t, dN4t]

        elif self.elemType == ElemType.QUAD8:
            
            dN1t = [lambda ksi,eta: (1-eta)*(2*ksi+eta)/4,      lambda ksi,eta: (1-ksi)*(ksi+2*eta)/4]
            dN2t = [lambda ksi,eta: (1-eta)*(2*ksi-eta)/4,      lambda ksi,eta: -(1+ksi)*(ksi-2*eta)/4]
            dN3t = [lambda ksi,eta: (1+eta)*(2*ksi+eta)/4,      lambda ksi,eta: (1+ksi)*(ksi+2*eta)/4]
            dN4t = [lambda ksi,eta: -(1+eta)*(-2*ksi+eta)/4,    lambda ksi,eta: (1-ksi)*(-ksi+2*eta)/4]
            dN5t = [lambda ksi,eta: -ksi*(1-eta),               lambda ksi,eta: -(1-ksi**2)/2]
            dN6t = [lambda ksi,eta: (1-eta**2)/2,               lambda ksi,eta: -eta*(1+ksi)]
            dN7t = [lambda ksi,eta: -ksi*(1+eta),               lambda ksi,eta: (1-ksi**2)/2]
            dN8t = [lambda ksi,eta: -(1-eta**2)/2,              lambda ksi,eta: -eta*(1-ksi)]
                            
            dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t])

        elif self.elemType == ElemType.TETRA4:
            
            dN1t = [lambda x,y,z: -1,   lambda x,y,z: -1,   lambda x,y,z: -1]
            dN2t = [lambda x,y,z: 1,    lambda x,y,z: 0,    lambda x,y,z: 0]
            dN3t = [lambda x,y,z: 0,    lambda x,y,z: 1,    lambda x,y,z: 0]
            dN4t = [lambda x,y,z: 0,    lambda x,y,z: 0,    lambda x,y,z: 1]

            dNtild = np.array([dN1t, dN2t, dN3t, dN4t])

        elif self.elemType == ElemType.HEXA8:
            
            dN1t = [lambda x,y,z: -1/8 * (1-y) * (1-z),   lambda x,y,z: -1/8 * (1-x) * (1-z),   lambda x,y,z: -1/8 * (1-x) * (1-y)]
            dN2t = [lambda x,y,z: 1/8 * (1-y) * (1-z),    lambda x,y,z: -1/8 * (1+x) * (1-z),    lambda x,y,z: -1/8 * (1+x) * (1-y)]
            dN3t = [lambda x,y,z: 1/8 * (1+y) * (1-z),    lambda x,y,z: 1/8 * (1+x) * (1-z),    lambda x,y,z: -1/8 * (1+x) * (1+y)]
            dN4t = [lambda x,y,z: -1/8 * (1+y) * (1-z),    lambda x,y,z: 1/8 * (1-x) * (1-z),    lambda x,y,z: -1/8 * (1-x) * (1+y)]
            dN5t = [lambda x,y,z: -1/8 * (1-y) * (1+z),    lambda x,y,z: -1/8 * (1-x) * (1+z),    lambda x,y,z: 1/8 * (1-x) * (1-y)]
            dN6t = [lambda x,y,z: 1/8 * (1-y) * (1+z),    lambda x,y,z: -1/8 * (1+x) * (1+z),    lambda x,y,z: 1/8 * (1+x) * (1-y)]
            dN7t = [lambda x,y,z: 1/8 * (1+y) * (1+z),    lambda x,y,z: 1/8 * (1+x) * (1+z),    lambda x,y,z: 1/8 * (1+x) * (1+y)]
            dN8t = [lambda x,y,z: -1/8 * (1+y) * (1+z),    lambda x,y,z: 1/8 * (1-x) * (1+z),    lambda x,y,z: 1/8 * (1-x) * (1+y)]

            dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t])
        
        elif self.elemType == ElemType.PRISM6:

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
        [phi_i,x psi_i,x . . . phi_n,x psi_n,x]
        """
        if self.dim != 1: return

        if self.elemType == ElemType.SEG2:
            
            phi_1_x = lambda x : -0.75 + 0.0*x + 0.75*x**2
            psi_1_x = lambda x : -0.125 + -0.25*x + 0.375*x**2
            phi_2_x = lambda x : 0.75 + 0.0*x + -0.75*x**2
            psi_2_x = lambda x : -0.125 + 0.25*x + 0.375*x**2

            dNvtild = np.array([phi_1_x, psi_1_x, phi_2_x, psi_2_x])
        
        elif self.elemType == ElemType.SEG3:

            phi_1_x = lambda x : 0.0 + 2.0*x + -3.75*x**2 + -2.0*x**3 + 3.75*x**4
            psi_1_x = lambda x : 0.0 + 0.25*x + -0.375*x**2 + -0.5*x**3 + 0.625*x**4
            phi_2_x = lambda x : 0.0 + 2.0*x + 3.75*x**2 + -2.0*x**3 + -3.75*x**4
            psi_2_x = lambda x : 0.0 + -0.25*x + -0.375*x**2 + 0.5*x**3 + 0.625*x**4
            phi_3_x = lambda x : 0.0 + -4.0*x + 0.0*x**2 + 4.0*x**3 + 0.0*x**4
            psi_3_x = lambda x : 0.5 + 0.0*x + -3.0*x**2 + 0.0*x**3 + 2.5*x**4

            dNvtild = np.array([phi_1_x, psi_1_x, phi_2_x, psi_2_x, phi_3_x, psi_3_x])

        elif self.elemType == ElemType.SEG4:

            phi_1_x = lambda x : -0.029296874999997335 + -0.9492187500000036*x + 1.646484374999976*x**2 + 9.492187500000014*x**3 + -13.798828124999957*x**4 + -8.54296875000001*x**5 + 12.181640624999979*x**6
            psi_1_x = lambda x : -0.0019531249999997224 + -0.07421875000000033*x + 0.1113281249999975*x**2 + 0.7734375000000011*x**3 + -0.9667968749999958*x**4 + -0.9492187500000009*x**5 + 1.107421874999998*x**6
            phi_2_x = lambda x : 0.02929687499999778 + -0.9492187499999947*x + -1.6464843749999734*x**2 + 9.49218749999998*x**3 + 13.798828124999948*x**4 + -8.542968749999986*x**5 + -12.181640624999973*x**6
            psi_2_x = lambda x : -0.0019531249999998335 + 0.07421874999999967*x + 0.11132812499999784*x**2 + -0.7734374999999987*x**3 + -0.9667968749999954*x**4 + 0.949218749999999*x**5 + 1.1074218749999976*x**6
            phi_3_x = lambda x : -2.373046874999991 + 0.9492187499999858*x + 27.052734374999915*x**2 + -9.492187499999938*x**3 + -54.58007812499985*x**4 + 8.542968749999954*x**5 + 29.900390624999925*x**6
            psi_3_x = lambda x : -0.1582031249999971 + -1.1601562500000036*x + 5.220703124999973*x**2 + 4.007812500000016*x**3 + -15.029296874999954*x**4 + -2.8476562500000115*x**5 + 9.966796874999977*x**6
            phi_4_x = lambda x : 2.3730468749999902 + 0.9492187500000178*x + -27.052734374999915*x**2 + -9.49218750000006*x**3 + 54.58007812499986*x**4 + 8.542968750000043*x**5 + -29.900390624999932*x**6
            psi_4_x = lambda x : -0.15820312499999734 + 1.1601562499999956*x + 5.220703124999973*x**2 + -4.007812499999981*x**3 + -15.02929687499995*x**4 + 2.847656249999986*x**5 + 9.966796874999975*x**6

            dNvtild = np.array([phi_1_x, psi_1_x, phi_2_x, psi_2_x, phi_3_x, psi_3_x, phi_4_x, psi_4_x])

        elif self.elemType == ElemType.SEG5:

            phi_1_x = lambda x : 8.882e-16 + 0.5185*x + -0.8611*x**2 + -9.111*x**3 + 12.64*x**4 + 34.67*x**5 + -45.11*x**6 + -26.07*x**7 + 33.33*x**8
            psi_1_x = lambda x : 0.0 + 0.0*x + 0.0*x**2 + 0.0*x**3 + 0.0*x**4 + 0.0*x**5 + 0.0*x**6 + 0.0*x**7 + 0.0*x**8
            phi_2_x = lambda x : -8.882e-16 + 0.5185*x + 0.8611*x**2 + -9.111*x**3 + -12.64*x**4 + 34.67*x**5 + 45.11*x**6 + -26.07*x**7 + -33.33*x**8
            psi_2_x = lambda x : 0.0 + 0.0*x + 0.0*x**2 + 0.0*x**3 + 0.0*x**4 + 0.0*x**5 + 0.0*x**6 + 0.0*x**7 + 0.0*x**8
            phi_3_x = lambda x : 0.0 + 9.481*x + -39.11*x**2 + -56.89*x**3 + 248.9*x**4 + 85.33*x**5 + -423.1*x**6 + -37.93*x**7 + 213.3*x**8
            psi_3_x = lambda x : 0.0 + 0.0*x + 0.0*x**2 + 0.0*x**3 + 0.0*x**4 + 0.0*x**5 + 0.0*x**6 + 0.0*x**7 + 0.0*x**8
            phi_4_x = lambda x : 0.0 + -20.0*x + 0.0*x**2 + 132.0*x**3 + 0.0*x**4 + -240.0*x**5 + 0.0*x**6 + 128.0*x**7 + 0.0*x**8
            psi_4_x = lambda x : 0.0 + 0.0*x + 0.0*x**2 + 0.0*x**3 + 0.0*x**4 + 0.0*x**5 + 0.0*x**6 + 0.0*x**7 + 0.0*x**8
            phi_5_x = lambda x : 0.0 + 9.481*x + 39.11*x**2 + -56.89*x**3 + -248.9*x**4 + 85.33*x**5 + 423.1*x**6 + -37.93*x**7 + -213.3*x**8
            psi_5_x = lambda x : 0.0 + 0.0*x + 0.0*x**2 + 0.0*x**3 + 0.0*x**4 + 0.0*x**5 + 0.0*x**6 + 0.0*x**7 + 0.0*x**8

            dNvtild = np.array([phi_1_x, psi_1_x, phi_2_x, psi_2_x, phi_3_x, psi_3_x, phi_4_x, psi_4_x, phi_5_x, psi_5_x])

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

    def get_ddN_pg(self, matriceType: str) -> np.ndarray:
        """Dérivées segonde des fonctions de formes dans l'element de référence (pg, dim, nPe), dans la base (ksi, eta ...) \n
        [Ni,ksi ksi . . . Nn,ksi ksi\n
        Ni,eta eta . . . Nn,eta eta]
        """
        if self.dim == 0: return

        elif self.dim == 1 and self.ordre < 2:

            ddNtild = np.array([lambda x: 0]*self.nPe)

        elif self.dim == 2 and self.ordre < 2:

            ddNtild = np.array([lambda ksi,eta: 0, lambda ksi,eta: 0]*self.nPe)

        elif self.dim == 3 and self.ordre < 2:

            ddNtild = np.array([lambda x,y,z: 0,lambda x,y,z: 0,lambda x,y,z: 0]*self.nPe)
        
        elif self.elemType == ElemType.SEG3:

            ddN1t = [lambda x: 1]
            ddN2t = [lambda x: 1]
            ddN3t = [lambda x: -2]

            ddNtild = np.array([ddN1t, ddN2t, ddN3t])

        elif self.elemType == ElemType.SEG5:

            ddN1t = [lambda x : 8.0*x**2 + -4.0*x + -0.3333]
            ddN2t = [lambda x : 8.0*x**2 + 4.0*x + -0.3333]
            ddN3t = [lambda x : -32.0*x**2 + 8.0*x + 5.333]
            ddN4t = [lambda x : 48.0*x**2 + 0.0*x + -10.0]
            ddN5t = [lambda x : -32.0*x**2 + -8.0*x + 5.333]

            ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t, ddN5t])

        elif self.elemType == ElemType.TRI6:

            ddN1t = [lambda ksi,eta: 4,  lambda ksi,eta: 4]
            ddN2t = [lambda ksi,eta: 4,  lambda ksi,eta: 0]
            ddN3t = [lambda ksi,eta: 0,  lambda ksi,eta: 4]
            ddN4t = [lambda ksi,eta: -8, lambda ksi,eta: 0]
            ddN5t = [lambda ksi,eta: 0,  lambda ksi,eta: 0]
            ddN6t = [lambda ksi,eta: 0,  lambda ksi,eta: -8]
            
            ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t, ddN5t, ddN6t])

        elif self.elemType == ElemType.TRI10:

            N1_ksi2 = lambda ksi, eta : -27.0*ksi + -27.0*eta + 18.0
            N2_ksi2 = lambda ksi, eta : 27.0*ksi + -2.186e-15*eta + -9.0
            N3_ksi2 = lambda ksi, eta : 0.0*ksi + -7.494e-16*eta + 0.0
            N4_ksi2 = lambda ksi, eta : 81.0*ksi + 54.0*eta + -45.0
            N5_ksi2 = lambda ksi, eta : -81.0*ksi + -27.0*eta + 36.0
            N6_ksi2 = lambda ksi, eta : 0.0*ksi + 27.0*eta + 0.0
            N7_ksi2 = lambda ksi, eta : 0.0*ksi + 0.0*eta + 0.0
            N8_ksi2 = lambda ksi, eta : 0.0*ksi + -2.998e-15*eta + 0.0
            N9_ksi2 = lambda ksi, eta : 0.0*ksi + 27.0*eta + 0.0
            N10_ksi2 = lambda ksi, eta : 0.0*ksi + -54.0*eta + 0.0

            N1_eta2 = lambda ksi, eta : -27.0*eta + -27.0*ksi + 18.0
            N2_eta2 = lambda ksi, eta : 0.0*eta + -1.624e-15*ksi + 0.0
            N3_eta2 = lambda ksi, eta : 27.0*eta + 5.995e-15*ksi + -9.0
            N4_eta2 = lambda ksi, eta : 0.0*eta + 27.0*ksi + 0.0
            N5_eta2 = lambda ksi, eta : 0.0*eta + -8.493e-15*ksi + 0.0
            N6_eta2 = lambda ksi, eta : 0.0*eta + 2.098e-14*ksi + 0.0
            N7_eta2 = lambda ksi, eta : 0.0*eta + 27.0*ksi + 0.0
            N8_eta2 = lambda ksi, eta : -81.0*eta + -27.0*ksi + 36.0
            N9_eta2 = lambda ksi, eta : 81.0*eta + 54.0*ksi + -45.0
            N10_eta2 = lambda ksi, eta : 0.0*eta + -54.0*ksi + 0.0

            ddN1t = [N1_ksi2, N1_eta2]
            ddN2t = [N2_ksi2, N2_eta2]
            ddN3t = [N3_ksi2, N3_eta2]
            ddN4t = [N4_ksi2, N4_eta2]
            ddN5t = [N5_ksi2, N5_eta2]
            ddN6t = [N6_ksi2, N6_eta2]
            ddN7t = [N7_ksi2, N7_eta2]
            ddN8t = [N8_ksi2, N8_eta2]
            ddN9t = [N9_ksi2, N9_eta2]
            ddN10t = [N10_ksi2, N10_eta2]

            ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t, ddN5t, ddN6t, ddN7t, ddN8t, ddN9t, ddN10t])

        elif self.elemType == ElemType.TRI15:

            N1_ksi2 = lambda ksi, eta: 128.0*ksi**2 + 256.0*ksi*eta + 128.0*eta**2 + -160.0*ksi + -160.0*eta + 46.67
            N2_ksi2 = lambda ksi, eta: 128.0*ksi**2 + -3.133e-14*ksi*eta + -5.329e-15*eta**2 + -96.0*ksi + 1.48e-14*eta + 14.67
            N3_ksi2 = lambda ksi, eta: 0.0*ksi**2 + 3.908e-14*ksi*eta + 6.276e-14*eta**2 + 0.0*ksi + -2.683e-14*eta + 0.0
            N4_ksi2 = lambda ksi, eta: -512.0*ksi**2 + -768.0*ksi*eta + -256.0*eta**2 + 576.0*ksi + 384.0*eta + -138.7
            N5_ksi2 = lambda ksi, eta: 768.0*ksi**2 + 768.0*ksi*eta + 128.0*eta**2 + -768.0*ksi + -288.0*eta + 152.0
            N6_ksi2 = lambda ksi, eta: -512.0*ksi**2 + -256.0*ksi*eta + -3.079e-14*eta**2 + 448.0*ksi + 64.0*eta + -74.67
            N7_ksi2 = lambda ksi, eta: 0.0*ksi**2 + 256.0*ksi*eta + 9.711e-14*eta**2 + 0.0*ksi + -64.0*eta + 0.0
            N8_ksi2 = lambda ksi, eta: 0.0*ksi**2 + 0.0*ksi*eta + 128.0*eta**2 + 0.0*ksi + -32.0*eta + 0.0
            N9_ksi2 = lambda ksi, eta: 0.0*ksi**2 + 6.711e-14*ksi*eta + 2.132e-14*eta**2 + 0.0*ksi + -2.842e-14*eta + 0.0
            N10_ksi2 = lambda ksi, eta: 0.0*ksi**2 + -1.232e-13*ksi*eta + -4.263e-13*eta**2 + 0.0*ksi + 1.563e-13*eta + 0.0
            N11_ksi2 = lambda ksi, eta: 0.0*ksi**2 + 5.053e-13*ksi*eta + 128.0*eta**2 + 0.0*ksi + -32.0*eta + 0.0
            N12_ksi2 = lambda ksi, eta: 0.0*ksi**2 + -256.0*ksi*eta + -256.0*eta**2 + 0.0*ksi + 192.0*eta + 0.0
            N13_ksi2 = lambda ksi, eta: 0.0*ksi**2 + 768.0*ksi*eta + 512.0*eta**2 + 0.0*ksi + -448.0*eta + 0.0
            N14_ksi2 = lambda ksi, eta: 0.0*ksi**2 + -768.0*ksi*eta + -256.0*eta**2 + 0.0*ksi + 320.0*eta + 0.0
            N15_ksi2 = lambda ksi, eta: 0.0*ksi**2 + -4.042e-13*ksi*eta + -256.0*eta**2 + 0.0*ksi + 64.0*eta + 0.0


            N1_eta2 = lambda ksi, eta: 128.0*ksi**2 + 256.0*ksi*eta + 128.0*eta**2 + -160.0*ksi + -160.0*eta + 46.67
            N2_eta2 = lambda ksi, eta: -5.329e-15*ksi**2 + -1.11e-14*ksi*eta + 0.0*eta**2 + 9.474e-15*ksi + 0.0*eta + 0.0
            N3_eta2 = lambda ksi, eta: 6.276e-14*ksi**2 + 1.705e-13*ksi*eta + 128.0*eta**2 + -6.513e-14*ksi + -96.0*eta + 14.67
            N4_eta2 = lambda ksi, eta: -256.0*ksi**2 + -256.0*ksi*eta + 0.0*eta**2 + 192.0*ksi + 0.0*eta + 0.0
            N5_eta2 = lambda ksi, eta: 128.0*ksi**2 + -4.583e-13*ksi*eta + 0.0*eta**2 + -32.0*ksi + 0.0*eta + 0.0
            N6_eta2 = lambda ksi, eta: -3.079e-14*ksi**2 + 1.332e-13*ksi*eta + 0.0*eta**2 + -5.447e-14*ksi + 0.0*eta + 0.0
            N7_eta2 = lambda ksi, eta: 9.711e-14*ksi**2 + -1.226e-13*ksi*eta + 0.0*eta**2 + 1.421e-14*ksi + 0.0*eta + 0.0
            N8_eta2 = lambda ksi, eta: 128.0*ksi**2 + 1.705e-13*ksi*eta + 0.0*eta**2 + -32.0*ksi + 0.0*eta + 0.0
            N9_eta2 = lambda ksi, eta: 2.132e-14*ksi**2 + 256.0*ksi*eta + 0.0*eta**2 + -64.0*ksi + 0.0*eta + 0.0
            N10_eta2 = lambda ksi, eta: -4.263e-13*ksi**2 + -256.0*ksi*eta + -512.0*eta**2 + 64.0*ksi + 448.0*eta + -74.67
            N11_eta2 = lambda ksi, eta: 128.0*ksi**2 + 768.0*ksi*eta + 768.0*eta**2 + -288.0*ksi + -768.0*eta + 152.0
            N12_eta2 = lambda ksi, eta: -256.0*ksi**2 + -768.0*ksi*eta + -512.0*eta**2 + 384.0*ksi + 576.0*eta + -138.7
            N13_eta2 = lambda ksi, eta: 512.0*ksi**2 + 768.0*ksi*eta + 0.0*eta**2 + -448.0*ksi + 0.0*eta + 0.0
            N14_eta2 = lambda ksi, eta: -256.0*ksi**2 + 2.984e-13*ksi*eta + 0.0*eta**2 + 64.0*ksi + 0.0*eta + 0.0
            N15_eta2 = lambda ksi, eta: -256.0*ksi**2 + -768.0*ksi*eta + 0.0*eta**2 + 320.0*ksi + 0.0*eta + 0.0


            ddN1t = [N1_ksi2, N1_eta2]
            ddN2t = [N2_ksi2, N2_eta2]
            ddN3t = [N3_ksi2, N3_eta2]
            ddN4t = [N4_ksi2, N4_eta2]
            ddN5t = [N5_ksi2, N5_eta2]
            ddN6t = [N6_ksi2, N6_eta2]
            ddN7t = [N7_ksi2, N7_eta2]
            ddN8t = [N8_ksi2, N8_eta2]
            ddN9t = [N9_ksi2, N9_eta2]
            ddN10t = [N10_ksi2, N10_eta2]
            ddN11t = [N11_ksi2, N11_eta2]
            ddN12t = [N12_ksi2, N12_eta2]
            ddN13t = [N13_ksi2, N13_eta2]
            ddN14t = [N14_ksi2, N14_eta2]
            ddN15t = [N15_ksi2, N15_eta2]


            ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t, ddN5t, ddN6t, ddN7t, ddN8t, ddN9t, ddN10t, ddN11t, ddN12t, ddN13t, ddN14t, ddN15t])
        
        elif self.elemType == ElemType.QUAD8:
            
            ddN1t = [lambda ksi,eta: (1-eta)/2,  lambda ksi,eta: (1-ksi)/2]
            ddN2t = [lambda ksi,eta: (1-eta)/2,  lambda ksi,eta: (1+ksi)/2]
            ddN3t = [lambda ksi,eta: (1+eta)/2,  lambda ksi,eta: (1+ksi)/2]
            ddN4t = [lambda ksi,eta: (1+eta)/2,  lambda ksi,eta: (1-ksi)/2]
            ddN5t = [lambda ksi,eta: -1+eta,     lambda ksi,eta: 0]
            ddN6t = [lambda ksi,eta: 0,          lambda ksi,eta: -1-ksi]
            ddN7t = [lambda ksi,eta: -1-eta,     lambda ksi,eta: 0]
            ddN8t = [lambda ksi,eta: 0,          lambda ksi,eta: -1+ksi]
                            
            ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t, ddN5t, ddN6t, ddN7t, ddN8t])

        else:
            raise "Element inconnue"
            
        
        # Evaluation aux points de gauss
        gauss = self.get_gauss(matriceType)
        coord = gauss.coord

        dim = self.dim
        nPg = gauss.nPg

        ddN_pg = np.zeros((nPg, dim, len(ddNtild)))

        for pg in range(nPg):
            for n, Nt in enumerate(ddNtild):
                for d in range(dim):
                    func = Nt[d]                        
                    if coord.shape[1] == 1:
                        ddN_pg[pg, d, n] = func(coord[pg,0])
                    elif coord.shape[1] == 2:
                        ddN_pg[pg, d, n] = func(coord[pg,0], coord[pg,1])
                    elif coord.shape[1] == 3:
                        ddN_pg[pg, d, n] = func(coord[pg,0], coord[pg,1], coord[pg,2])

        return ddN_pg

    def get_ddNv_pg(self, matriceType: str) -> np.ndarray:
        """Dérivées 2nd des fonctions de formes dans l'element poutre en flexion (pg, dim, nPe), dans la base (ksi) \n
        [phi_i,xx psi_i,xx . . . phi_n,xx psi_n,xx]
        """
        if self.dim != 1: return

        if self.elemType == ElemType.SEG2:
            
            phi_1_xx = lambda x : 0.0 + 1.5*x
            psi_1_xx = lambda x : -0.25 + 0.75*x
            phi_2_xx = lambda x : 0.0 + -1.5*x
            psi_2_xx = lambda x : 0.25 + 0.75*x

            ddNvtild = np.array([phi_1_xx, psi_1_xx, phi_2_xx, psi_2_xx])
        
        elif self.elemType == ElemType.SEG3:

            phi_1_xx = lambda x : 2.0 + -7.5*x + -6.0*x**2 + 15.0*x**3
            psi_1_xx = lambda x : 0.25 + -0.75*x + -1.5*x**2 + 2.5*x**3
            phi_2_xx = lambda x : 2.0 + 7.5*x + -6.0*x**2 + -15.0*x**3
            psi_2_xx = lambda x : -0.25 + -0.75*x + 1.5*x**2 + 2.5*x**3
            phi_3_xx = lambda x : -4.0 + 0.0*x + 12.0*x**2 + 0.0*x**3
            psi_3_xx = lambda x : 0.0 + -6.0*x + 0.0*x**2 + 10.0*x**3

            ddNvtild = np.array([phi_1_xx, psi_1_xx, phi_2_xx, psi_2_xx, phi_3_xx, psi_3_xx])

        elif self.elemType == ElemType.SEG4:

            phi_1_xx = lambda x : -0.9492187500000036 + 3.292968749999952*x + 28.476562500000043*x**2 + -55.19531249999983*x**3 + -42.71484375000006*x**4 + 73.08984374999987*x**5
            psi_1_xx = lambda x : -0.07421875000000033 + 0.222656249999995*x + 2.3203125000000036*x**2 + -3.867187499999983*x**3 + -4.746093750000004*x**4 + 6.6445312499999885*x**5
            phi_2_xx = lambda x : -0.9492187499999947 + -3.2929687499999467*x + 28.476562499999943*x**2 + 55.195312499999794*x**3 + -42.71484374999993*x**4 + -73.08984374999984*x**5
            psi_2_xx = lambda x : 0.07421874999999967 + 0.22265624999999567*x + -2.320312499999996*x**2 + -3.867187499999982*x**3 + 4.746093749999995*x**4 + 6.644531249999985*x**5
            phi_3_xx = lambda x : 0.9492187499999858 + 54.10546874999983*x + -28.476562499999815*x**2 + -218.3203124999994*x**3 + 42.714843749999766*x**4 + 179.40234374999955*x**5
            psi_3_xx = lambda x : -1.1601562500000036 + 10.441406249999947*x + 12.023437500000048*x**2 + -60.117187499999815*x**3 + -14.238281250000057*x**4 + 59.80078124999986*x**5
            phi_4_xx = lambda x : 0.9492187500000178 + -54.10546874999983*x + -28.47656250000018*x**2 + 218.32031249999943*x**3 + 42.71484375000021*x**4 + -179.4023437499996*x**5
            psi_4_xx = lambda x : 1.1601562499999956 + 10.441406249999947*x + -12.023437499999943*x**2 + -60.1171874999998*x**3 + 14.23828124999993*x**4 + 59.80078124999985*x**5

            ddNvtild = np.array([phi_1_xx, psi_1_xx, phi_2_xx, psi_2_xx, phi_3_xx, psi_3_xx, phi_4_xx, psi_4_xx])

        elif self.elemType == ElemType.SEG5:

            phi_1_xx = lambda x : 0.5185 + -1.722*x + -27.33*x**2 + 50.56*x**3 + 173.3*x**4 + -270.7*x**5 + -182.5*x**6 + 266.7*x**7
            psi_1_xx = lambda x : 0.0 + 0.0*x + 0.0*x**2 + 0.0*x**3 + 0.0*x**4 + 0.0*x**5 + 0.0*x**6 + 0.0*x**7
            phi_2_xx = lambda x : 0.5185 + 1.722*x + -27.33*x**2 + -50.56*x**3 + 173.3*x**4 + 270.7*x**5 + -182.5*x**6 + -266.7*x**7
            psi_2_xx = lambda x : 0.0 + 0.0*x + 0.0*x**2 + 0.0*x**3 + 0.0*x**4 + 0.0*x**5 + 0.0*x**6 + 0.0*x**7
            phi_3_xx = lambda x : 9.481 + -78.22*x + -170.7*x**2 + 995.6*x**3 + 426.7*x**4 + -2.539e+03*x**5 + -265.5*x**6 + 1.707e+03*x**7
            psi_3_xx = lambda x : 0.0 + 0.0*x + 0.0*x**2 + 0.0*x**3 + 0.0*x**4 + 0.0*x**5 + 0.0*x**6 + 0.0*x**7
            phi_4_xx = lambda x : -20.0 + 0.0*x + 396.0*x**2 + 0.0*x**3 + -1.2e+03*x**4 + 0.0*x**5 + 896.0*x**6 + 0.0*x**7
            psi_4_xx = lambda x : 0.0 + 0.0*x + 0.0*x**2 + 0.0*x**3 + 0.0*x**4 + 0.0*x**5 + 0.0*x**6 + 0.0*x**7
            phi_5_xx = lambda x : 9.481 + 78.22*x + -170.7*x**2 + -995.6*x**3 + 426.7*x**4 + 2.539e+03*x**5 + -265.5*x**6 + -1.707e+03*x**7
            psi_5_xx = lambda x : 0.0 + 0.0*x + 0.0*x**2 + 0.0*x**3 + 0.0*x**4 + 0.0*x**5 + 0.0*x**6 + 0.0*x**7

            ddNvtild = np.array([phi_1_xx, psi_1_xx, phi_2_xx, psi_2_xx, phi_3_xx, psi_3_xx, phi_4_xx, psi_4_xx, phi_5_xx, psi_5_xx])          

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

    def get_dddN_pg(self, matriceType: str) -> np.ndarray:
        """Dérivées 3 des fonctions de formes dans l'element de référence (pg, dim, nPe), dans la base (ksi, eta ...) \n
        [Ni,ksi ksi ksi . . . Nn,ksi ksi ksi\n
        Ni,eta eta eta . . . Nn,eta eta eta]
        """
        if self.elemType == 0: return

        elif self.dim == 1 and self.ordre < 3:

            dddNtild = np.array([lambda x: 0]*self.nPe)

        elif self.dim == 2 and self.ordre < 3:

            dddNtild = np.array([lambda ksi,eta: 0, lambda ksi,eta: 0]*self.nPe)

        elif self.dim == 3 and self.ordre < 3:

            dddNtild = np.array([lambda x,y,z: 0,lambda x,y,z: 0,lambda x,y,z: 0]*self.nPe)

        elif self.elemType == ElemType.SEG5:

            dddN1t = [lambda x : 16.0*x + -4.0]
            dddN2t = [lambda x : 16.0*x + 4.0]
            dddN3t = [lambda x : -64.0*x + 8.0]
            dddN4t = [lambda x : 96.0*x + 0.0]
            dddN5t = [lambda x : -64.0*x + -8.0]

            dddNtild = np.array([dddN1t, dddN2t, dddN3t, dddN4t, dddN5t])

        elif self.elemType == ElemType.TRI10:

            N1_ksi3 = lambda ksi, eta : -27.0
            N2_ksi3 = lambda ksi, eta : 27.0
            N3_ksi3 = lambda ksi, eta : 0.0
            N4_ksi3 = lambda ksi, eta : 81.0
            N5_ksi3 = lambda ksi, eta : -81.0
            N6_ksi3 = lambda ksi, eta : 0.0
            N7_ksi3 = lambda ksi, eta : 0.0
            N8_ksi3 = lambda ksi, eta : 0.0
            N9_ksi3 = lambda ksi, eta : 0.0
            N10_ksi3 = lambda ksi, eta : 0.0

            N1_eta3 = lambda ksi, eta : -27.0
            N2_eta3 = lambda ksi, eta : 0.0
            N3_eta3 = lambda ksi, eta : 27.0
            N4_eta3 = lambda ksi, eta : 0.0
            N5_eta3 = lambda ksi, eta : 0.0
            N6_eta3 = lambda ksi, eta : 0.0
            N7_eta3 = lambda ksi, eta : 0.0
            N8_eta3 = lambda ksi, eta : -81.0
            N9_eta3 = lambda ksi, eta : 81.0
            N10_eta3 = lambda ksi, eta : 0.0

            dddN1t = [N1_ksi3, N1_eta3]
            dddN2t = [N2_ksi3, N2_eta3]
            dddN3t = [N3_ksi3, N3_eta3]
            dddN4t = [N4_ksi3, N4_eta3]
            dddN5t = [N5_ksi3, N5_eta3]
            dddN6t = [N6_ksi3, N6_eta3]
            dddN7t = [N7_ksi3, N7_eta3]
            dddN8t = [N8_ksi3, N8_eta3]
            dddN9t = [N9_ksi3, N9_eta3]
            dddN10t = [N10_ksi3, N10_eta3]

            dddNtild = np.array([dddN1t, dddN2t, dddN3t, dddN4t, dddN5t, dddN6t, dddN7t, dddN8t, dddN9t, dddN10t])

        elif self.elemType == ElemType.TRI15:

            N1_ksi3 = lambda ksi, eta: 256.0*ksi + 256.0*eta + -160.0
            N2_ksi3 = lambda ksi, eta: 256.0*ksi + -3.133e-14*eta + -96.0
            N3_ksi3 = lambda ksi, eta: 0.0*ksi + 3.908e-14*eta + 0.0
            N4_ksi3 = lambda ksi, eta: -1.024e+03*ksi + -768.0*eta + 576.0
            N5_ksi3 = lambda ksi, eta: 1.536e+03*ksi + 768.0*eta + -768.0
            N6_ksi3 = lambda ksi, eta: -1.024e+03*ksi + -256.0*eta + 448.0
            N7_ksi3 = lambda ksi, eta: 0.0*ksi + 256.0*eta + 0.0
            N8_ksi3 = lambda ksi, eta: 0.0*ksi + 0.0*eta + 0.0
            N9_ksi3 = lambda ksi, eta: 0.0*ksi + 6.711e-14*eta + 0.0
            N10_ksi3 = lambda ksi, eta: 0.0*ksi + -1.232e-13*eta + 0.0
            N11_ksi3 = lambda ksi, eta: 0.0*ksi + 5.053e-13*eta + 0.0
            N12_ksi3 = lambda ksi, eta: 0.0*ksi + -256.0*eta + 0.0
            N13_ksi3 = lambda ksi, eta: 0.0*ksi + 768.0*eta + 0.0
            N14_ksi3 = lambda ksi, eta: 0.0*ksi + -768.0*eta + 0.0
            N15_ksi3 = lambda ksi, eta: 0.0*ksi + -4.042e-13*eta + 0.0


            N1_eta3 = lambda ksi, eta: 256.0*ksi + 256.0*eta + -160.0
            N2_eta3 = lambda ksi, eta: -1.11e-14*ksi + 0.0*eta + 0.0
            N3_eta3 = lambda ksi, eta: 1.705e-13*ksi + 256.0*eta + -96.0
            N4_eta3 = lambda ksi, eta: -256.0*ksi + 0.0*eta + 0.0
            N5_eta3 = lambda ksi, eta: -4.583e-13*ksi + 0.0*eta + 0.0
            N6_eta3 = lambda ksi, eta: 1.332e-13*ksi + 0.0*eta + 0.0
            N7_eta3 = lambda ksi, eta: -1.226e-13*ksi + 0.0*eta + 0.0
            N8_eta3 = lambda ksi, eta: 1.705e-13*ksi + 0.0*eta + 0.0
            N9_eta3 = lambda ksi, eta: 256.0*ksi + 0.0*eta + 0.0
            N10_eta3 = lambda ksi, eta: -256.0*ksi + -1.024e+03*eta + 448.0
            N11_eta3 = lambda ksi, eta: 768.0*ksi + 1.536e+03*eta + -768.0
            N12_eta3 = lambda ksi, eta: -768.0*ksi + -1.024e+03*eta + 576.0
            N13_eta3 = lambda ksi, eta: 768.0*ksi + 0.0*eta + 0.0
            N14_eta3 = lambda ksi, eta: 2.984e-13*ksi + 0.0*eta + 0.0
            N15_eta3 = lambda ksi, eta: -768.0*ksi + 0.0*eta + 0.0


            dddN1t = [N1_ksi3, N1_eta3]
            dddN2t = [N2_ksi3, N2_eta3]
            dddN3t = [N3_ksi3, N3_eta3]
            dddN4t = [N4_ksi3, N4_eta3]
            dddN5t = [N5_ksi3, N5_eta3]
            dddN6t = [N6_ksi3, N6_eta3]
            dddN7t = [N7_ksi3, N7_eta3]
            dddN8t = [N8_ksi3, N8_eta3]
            dddN9t = [N9_ksi3, N9_eta3]
            dddN10t = [N10_ksi3, N10_eta3]
            dddN11t = [N11_ksi3, N11_eta3]
            dddN12t = [N12_ksi3, N12_eta3]
            dddN13t = [N13_ksi3, N13_eta3]
            dddN14t = [N14_ksi3, N14_eta3]
            dddN15t = [N15_ksi3, N15_eta3]


            dddNtild = np.array([dddN1t, dddN2t, dddN3t, dddN4t, dddN5t, dddN6t, dddN7t, dddN8t, dddN9t, dddN10t, dddN11t, dddN12t, dddN13t, dddN14t, dddN15t])

        else:
            raise "Element inconnue"
            
        
        # Evaluation aux points de gauss
        gauss = self.get_gauss(matriceType)
        coord = gauss.coord

        dim = self.dim
        nPg = gauss.nPg

        dddN_pg = np.zeros((nPg, dim, len(dddNtild)))

        for pg in range(nPg):
            for n, Nt in enumerate(dddNtild):
                for d in range(dim):
                    func = Nt[d]                        
                    if coord.shape[1] == 1:
                        dddN_pg[pg, d, n] = func(coord[pg,0])
                    elif coord.shape[1] == 2:
                        dddN_pg[pg, d, n] = func(coord[pg,0], coord[pg,1])
                    elif coord.shape[1] == 3:
                        dddN_pg[pg, d, n] = func(coord[pg,0], coord[pg,1], coord[pg,2])

        return dddN_pg

    def get_ddddN_pg(self, matriceType: str) -> np.ndarray:
        """Dérivées 4 des fonctions de formes dans l'element de référence (pg, dim, nPe), dans la base (ksi, eta ...) \n
        [Ni,ksi ksi ksi ksi . . . Nn,ksi ksi ksi ksi\n
        Ni,eta eta eta eta . . . Nn,eta eta eta eta]
        """
        if self.elemType == 0: return

        elif self.dim == 1 and self.ordre < 4:

            dddNtild = np.array([lambda x: 0]*self.nPe)

        elif self.dim == 2 and self.ordre < 4:

            dddNtild = np.array([lambda ksi,eta: 0, lambda ksi,eta: 0]*self.nPe)

        elif self.dim == 3 and self.ordre < 4:

            dddNtild = np.array([lambda x,y,z: 0,lambda x,y,z: 0,lambda x,y,z: 0]*self.nPe)

        elif self.elemType == ElemType.SEG5:

            ddddN1t = [lambda x : 16.0]
            ddddN2t = [lambda x : 16.0]
            ddddN3t = [lambda x : -64.0]
            ddddN4t = [lambda x : 96.0]
            ddddN5t = [lambda x : -64.0]

            ddddNtild = np.array([ddddN1t, ddddN2t, ddddN3t, ddddN4t, ddddN5t])
        
        elif self.elemType == ElemType.TRI15:

            N1_ksi4 = lambda ksi, eta: 256.0
            N2_ksi4 = lambda ksi, eta: 256.0
            N3_ksi4 = lambda ksi, eta: 0.0
            N4_ksi4 = lambda ksi, eta: -1.024e+03
            N5_ksi4 = lambda ksi, eta: 1.536e+03
            N6_ksi4 = lambda ksi, eta: -1.024e+03
            N7_ksi4 = lambda ksi, eta: 0.0
            N8_ksi4 = lambda ksi, eta: 0.0
            N9_ksi4 = lambda ksi, eta: 0.0
            N10_ksi4 = lambda ksi, eta: 0.0
            N11_ksi4 = lambda ksi, eta: 0.0
            N12_ksi4 = lambda ksi, eta: 0.0
            N13_ksi4 = lambda ksi, eta: 0.0
            N14_ksi4 = lambda ksi, eta: 0.0
            N15_ksi4 = lambda ksi, eta: 0.0


            N1_eta4 = lambda ksi, eta: 256.0
            N2_eta4 = lambda ksi, eta: 0.0
            N3_eta4 = lambda ksi, eta: 256.0
            N4_eta4 = lambda ksi, eta: 0.0
            N5_eta4 = lambda ksi, eta: 0.0
            N6_eta4 = lambda ksi, eta: 0.0
            N7_eta4 = lambda ksi, eta: 0.0
            N8_eta4 = lambda ksi, eta: 0.0
            N9_eta4 = lambda ksi, eta: 0.0
            N10_eta4 = lambda ksi, eta: -1.024e+03
            N11_eta4 = lambda ksi, eta: 1.536e+03
            N12_eta4 = lambda ksi, eta: -1.024e+03
            N13_eta4 = lambda ksi, eta: 0.0
            N14_eta4 = lambda ksi, eta: 0.0
            N15_eta4 = lambda ksi, eta: 0.0


            ddddN1t = [N1_ksi4, N1_eta4]
            ddddN2t = [N2_ksi4, N2_eta4]
            ddddN3t = [N3_ksi4, N3_eta4]
            ddddN4t = [N4_ksi4, N4_eta4]
            ddddN5t = [N5_ksi4, N5_eta4]
            ddddN6t = [N6_ksi4, N6_eta4]
            ddddN7t = [N7_ksi4, N7_eta4]
            ddddN8t = [N8_ksi4, N8_eta4]
            ddddN9t = [N9_ksi4, N9_eta4]
            ddddN10t = [N10_ksi4, N10_eta4]
            ddddN11t = [N11_ksi4, N11_eta4]
            ddddN12t = [N12_ksi4, N12_eta4]
            ddddN13t = [N13_ksi4, N13_eta4]
            ddddN14t = [N14_ksi4, N14_eta4]
            ddddN15t = [N15_ksi4, N15_eta4]


            ddddNtild = np.array([ddddN1t, ddddN2t, ddddN3t, ddddN4t, ddddN5t, ddddN6t, ddddN7t, ddddN8t, ddddN9t, ddddN10t, ddddN11t, ddddN12t, ddddN13t, ddddN14t, ddddN15t])

        else:
            raise "Element inconnue"
            
        
        # Evaluation aux points de gauss
        gauss = self.get_gauss(matriceType)
        coord = gauss.coord

        dim = self.dim
        nPg = gauss.nPg

        dddN_pg = np.zeros((nPg, dim, len(dddNtild)))

        for pg in range(nPg):
            for n, Nt in enumerate(dddNtild):
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

    def Set_Nodes_Tag(self, noeuds: np.ndarray, tag: str):
        """Ajoute un tag sur les noeuds

        Parameters
        ----------
        noeuds : np.ndarray
            liste de noeuds
        tag : str
            tag utilisé
        """
        if noeuds.size == 0: return
        self.__dict_nodes_tags[tag] = noeuds

    @property
    def nodeTags(self) -> list:
        """Renvoie les tags associés aux noeuds"""
        try:
            return list(self.__dict_nodes_tags.keys())
        except:
            return []

    def Set_Elements_Tag(self, noeuds: np.ndarray, tag: str):
        """Ajoute un tag sur les elements associés aux noeuds

        Parameters
        ----------
        noeuds : np.ndarray
            liste de noeuds
        tag : str
            tag utilisé
        """

        if noeuds.size == 0: return

        # Récupère les elements associés aux noeuds
        elements = self.get_elementsIndex(noeuds=noeuds, exclusivement=True)

        self.__dict_elements_tags[tag] = elements

    @property
    def elementTags(self) -> list:
        """Renvoie les tags associés aux elements"""
        try:
            return list(self.__dict_elements_tags.keys())
        except:
            return []

    def Get_Elements_Tag(self, tag: str):
        try:
            return self.__dict_elements_tags[tag]
        except:
            print("Groupe physique inconnue")
    
    def Get_Nodes_Tag(self, tag: str):
        try:
            return self.__dict_nodes_tags[tag]
        except:
            print("Groupe physique inconnue")
    
    def Localise_sol_e(self, sol: np.ndarray) -> np.ndarray:
        """localise les valeurs de noeuds sur les elements"""
        tailleVecteur = self.Nn * self.dim

        if sol.shape[0] == tailleVecteur:
            sol_e = sol[self.assembly_e]
        elif sol.shape[0] == self.Nn:
            sol_e = sol[self.__connect]
        else:
            return
        
        return sol_e

    def get_connectTriangle(self) -> np.ndarray:
        """Transforme la matrice de connectivité pour la passer dans la fonction trisurf en 2D\n
        Par exemple pour un quadrangle on construit deux triangles
        pour un triangle à 6 noeuds on construit 4 triangles\n

        Renvoie un dictionnaire par type
        """
        assert self.dim == 2
        dict_connect_triangle = {}
        # TODO essayer de faire aussi avec les elements genre pour SEG2 -> dict_connect_triangle[self.elemType] = self.__connect[:,[0,1,0]] ? Est ce que ça marche ?
        if self.elemType == ElemType.TRI3:
            dict_connect_triangle[self.elemType] = self.__connect[:,[0,1,2]]
        elif self.elemType == ElemType.TRI6:
            dict_connect_triangle[self.elemType] = np.array(self.__connect[:, [0,3,5,3,1,4,5,4,2,3,4,5]]).reshape(-1,3)
        elif self.elemType == ElemType.TRI10:
            dict_connect_triangle[self.elemType] = np.array(self.__connect[:, np.array([10,1,4,
                                                                                        10,4,5,
                                                                                        10,5,6,
                                                                                        10,6,7,
                                                                                        10,7,8,
                                                                                        10,8,9,
                                                                                        10,9,1,
                                                                                        2,5,6,
                                                                                        3,7,8])-1]).reshape(-1,3)
        elif self.elemType == ElemType.TRI15:
            dict_connect_triangle[self.elemType] = np.array(self.__connect[:, np.array([1,4,13,
                                                                                        4,5,14,
                                                                                        5,6,14,
                                                                                        6,7,14,
                                                                                        2,6,7,
                                                                                        4,13,14,
                                                                                        1,12,13,
                                                                                        11,12,13,
                                                                                        11,13,15,
                                                                                        13,14,15,
                                                                                        8,14,15,
                                                                                        7,8,14,
                                                                                        10,11,15,
                                                                                        8,9,15,
                                                                                        9,10,15,
                                                                                        3,9,10])-1]).reshape(-1,3)
        elif self.elemType == ElemType.QUAD4:
            dict_connect_triangle[self.elemType] = np.array(self.__connect[:, [0,1,3,1,2,3]]).reshape(-1,3)
        elif self.elemType == ElemType.QUAD8:
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
        if self.elemType in [ElemType.SEG2,ElemType.POINT]:
            dict_connect_faces[self.elemType] = self.__connect.copy()
        elif self.elemType == ElemType.SEG3:
            dict_connect_faces[self.elemType] = self.__connect[:, [0,2,1]]
        elif self.elemType == ElemType.SEG4:
            dict_connect_faces[self.elemType] = self.__connect[:, [0, 2, 3, 1]]
        elif self.elemType == ElemType.SEG5:
            dict_connect_faces[self.elemType] = self.__connect[:, [0, 2, 3, 4, 1]]
        elif self.elemType == ElemType.TRI3:
            dict_connect_faces[self.elemType] = self.__connect[:, [0,1,2,0]]
        elif self.elemType == ElemType.TRI6:
            dict_connect_faces[self.elemType] = self.__connect[:, [0,3,1,4,2,5,0]]
        elif self.elemType == ElemType.TRI10:
            dict_connect_faces[self.elemType] = self.__connect[:, [0,3,4,1,5,6,2,7,8,0]]
        elif self.elemType == ElemType.TRI15:
            dict_connect_faces[self.elemType] = self.__connect[:, [0,3,4,5,1,6,7,8,2,9,10,11,0]]
        elif self.elemType == ElemType.QUAD4:
            dict_connect_faces[self.elemType] = self.__connect[:, [0,1,2,3,0]]
        elif self.elemType == ElemType.QUAD8:
            dict_connect_faces[self.elemType] = self.__connect[:, [0,4,1,5,2,6,3,7,0]]
        elif self.elemType == ElemType.TETRA4:
            # Ici par elexemple on va creer 3 faces, chaque face est composé des identifiants des noeuds
            dict_connect_faces[self.elemType] = np.array(self.__connect[:, [0,1,2,0,1,3,0,2,3,1,2,3]]).reshape(self.Ne*nPe,-1)
        elif self.elemType == ElemType.HEXA8:
            # Ici par elexemple on va creer 6 faces, chaque face est composé des identifiants des noeuds                
            dict_connect_faces[self.elemType] = np.array(self.__connect[:, [0,1,2,3,0,1,5,4,0,3,7,4,6,2,3,7,6,2,1,5,6,7,4,5]]).reshape(-1,nPe)
        elif self.elemType == ElemType.PRISM6:
            # Ici il faut faire attention parce que cette element est composé de 2 triangles et 3 quadrangles
            dict_connect_faces[ElemType.QUAD4] = np.array(self.__connect[:, [0,2,5,3,0,1,4,3,1,2,5,4]]).reshape(-1,4)
            dict_connect_faces[ElemType.TRI3] = np.array(self.__connect[:, [0,1,2,3,4,5]]).reshape(-1,3)
            
        else:
            raise "Element inconnue"

        return dict_connect_faces

    ################################################ STATIC ##################################################

    @staticmethod
    def get_MatriceType() -> List[str]:
        """type de matrice disponible"""        
        liste = [MatriceType.rigi, MatriceType.masse, MatriceType.beam]
        return liste

    @staticmethod
    def get_Types1D() -> List[int]:
        """type d'elements disponibles en 1D"""
        # liste1D = ["SEG2", "SEG3", "SEG4", "SEG5"]
        liste1D = [ElemType.SEG2, ElemType.SEG3, ElemType.SEG4]
        return liste1D

    @staticmethod
    def get_Types2D() -> List[int]:
        """type d'elements disponibles en 2D"""
        # liste2D = ["TRI3", "TRI6", "TRI10", "TRI15", "QUAD4", "QUAD8"]
        # TODO il reste des erreurs sur TRI15 certainement points d'intégrations
        liste2D = [ElemType.TRI3, ElemType.TRI6, ElemType.TRI10, ElemType.QUAD4, ElemType.QUAD8]
        return liste2D
    
    @staticmethod
    def get_Types3D() -> List[int]:
        """type d'elements disponibles en 3D"""
        liste3D = [ElemType.TETRA4, ElemType.HEXA8, ElemType.PRISM6]
        return liste3D

    @staticmethod
    def Get_ElemInFos(gmshId: int) -> tuple:
        """Renvoie le nom le nombre de noeuds par element et la dimension de l'ordre de l'élement en fonction du gmshId

        Args:
            type (int): type de l'identifiant sur gmsh

        Returns:
            return elemType, nPe, dim, ordre, nbFaces
        """
        if gmshId == 15:
            elemType = ElemType.POINT; nPe = 1; dim = 0; ordre = 0; nbFaces = 0
        elif gmshId == 1:
            elemType = ElemType.SEG2; nPe = 2; dim = 1; ordre = 1; nbFaces = 0
            #       v
            #       ^
            #       |
            #       |
            # 0-----+-----1 --> u
        elif gmshId == 8:
            elemType = ElemType.SEG3; nPe = 3; dim = 1; ordre = 2; nbFaces = 0
            #       v
            #       ^
            #       |
            #       |
            #  0----2----1 --> u
        elif gmshId == 26:
            elemType = ElemType.SEG4; nPe = 4; dim = 1; ordre = 3; nbFaces = 0
            #        v
            #        ^
            #        |
            #        |
            #  0---2-+-3---1 --> u
        elif gmshId == 27:
            elemType = ElemType.SEG5; nPe = 5; dim = 1; ordre = 4; nbFaces = 0
            #          v
            #          ^
            #          |
            #          |
            #  0---2---3---4---1 --> u
        elif gmshId == 2:
            elemType = ElemType.TRI3; nPe = 3; dim = 2; ordre = 2; nbFaces = 1
            # v
            # ^
            # |
            # 2
            # |`\
            # |  `\
            # |    `\
            # |      `\
            # |        `\
            # 0----------1 --> u
        elif gmshId == 9:
            elemType = ElemType.TRI6; nPe = 6; dim = 2; ordre = 2; nbFaces = 1
            # v
            # ^
            # |
            # 2
            # |`\
            # |  `\
            # 5    `4
            # |      `\
            # |        `\
            # 0----3-----1 --> u
        elif gmshId == 21:
            elemType = ElemType.TRI10; nPe = 10; dim = 2; ordre = 3; nbFaces = 1
            # v
            # ^
            # |
            # 2
            # | \
            # 7   6
            # |     \
            # 8  (9)  5
            # |         \
            # 0---3---4---1
        elif gmshId == 23:
            elemType = ElemType.TRI15; nPe = 15; dim = 2; ordre = 4; nbFaces = 1
            # 
            # 2
            # | \
            # 9   8
            # |     \
            # 10 (14)  7
            # |         \
            # 11 (12) (13) 6
            # |             \
            # 0---3---4---5---1
        elif gmshId == 3:
            elemType = ElemType.QUAD4; nPe = 4; dim = 2; ordre = 1; nbFaces = 1
            #       v
            #       ^
            #       |
            # 3-----------2
            # |     |     |
            # |     |     |
            # |     +---- | --> u
            # |           |
            # |           |
            # 0-----------1
        elif gmshId == 16:
            elemType = ElemType.QUAD8; nPe = 8; dim = 2; ordre = 2; nbFaces = 1
            #       v
            #       ^
            #       |
            # 3-----6-----2
            # |     |     |
            # |     |     |
            # 7     +---- 5 --> u
            # |           |
            # |           |
            # 0-----4-----1
        elif gmshId == 10:
            elemType = ElemType.QUAD9; nPe = 9; dim = 2; ordre = 3; nbFaces = 1
            #       v
            #       ^
            #       |
            # 3-----6-----2
            # |     |     |
            # |     |     |
            # 7     8---- 5 --> u
            # |           |
            # |           |
            # 0-----4-----1
        elif gmshId == 4:
            elemType = ElemType.TETRA4; nPe = 4; dim = 3; ordre = 1; nbFaces = 4
            #                    v
            #                  .
            #                ,/
            #               /
            #            2
            #          ,/|`\
            #        ,/  |  `\
            #      ,/    '.   `\
            #    ,/       |     `\
            #  ,/         |       `\
            # 0-----------'.--------1 --> u
            #  `\.         |      ,/
            #     `\.      |    ,/
            #        `\.   '. ,/
            #           `\. |/
            #              `3
            #                 `\.
            #                    ` w
        elif gmshId == 11:
            elemType = ElemType.TETRA10; nPe = 10; dim = 3; ordre = 2; nbFaces = 4
            #                    v
            #                  .
            #                ,/
            #               /
            #            2
            #          ,/|`\
            #        ,/  |  `\
            #      ,6    '.   `5
            #    ,/       8     `\
            #  ,/         |       `\
            # 0--------4--'.--------1 --> u
            #  `\.         |      ,/
            #     `\.      |    ,9
            #        `7.   '. ,/
            #           `\. |/
            #              `3
            #                 `\.
            #                    ` w
        elif gmshId == 5:
            elemType = ElemType.HEXA8; nPe = 8; dim = 3; ordre = 1; nbFaces = 6
            #        v
            # 3----------2
            # |\     ^   |\
            # | \    |   | \
            # |  \   |   |  \
            # |   7------+---6
            # |   |  +-- |-- | -> u
            # 0---+---\--1   |
            #  \  |    \  \  |
            #   \ |     \  \ |
            #    \|      w  \|
            #     4----------5
        elif gmshId == 12:
            elemType = ElemType.HEXA27; nPe = 27; dim = 3; ordre = 2; nbFaces = 6
            #        v
            # 3----13----2
            # |\         |\
            # |15    24  | 14
            # 9  \ 20    11 \
            # |   7----19+---6
            # |22 |  26  | 23|
            # 0---+-8----1   |
            #  \ 17    25 \  18
            #  10 |  21    12|
            #    \|         \|
            #     4----16----5
        elif gmshId == 6:
            elemType = ElemType.PRISM6; nPe = 6; dim = 3; ordre = 1; nbFaces = 5
            #            w
            #            ^
            #            |
            #            3
            #          ,/|`\
            #        ,/  |  `\
            #      ,/    |    `\
            #     4------+------5
            #     |      |      |
            #     |    ,/|`\    |
            #     |  ,/  |  `\  |
            #     |,/    |    `\|
            #    ,|      |      |\
            #  ,/ |      0      | `\
            # u   |    ,/ `\    |    v
            #     |  ,/     `\  |
            #     |,/         `\|
            #     1-------------2
        elif gmshId == 18:
            elemType = ElemType.PRISM15; nPe = 15; dim = 3; ordre = 2; nbFaces = 5
            #            w
            #            ^
            #            |
            #            3
            #          ,/|`\
            #        12  |  13
            #      ,/    |    `\
            #     4------14-----5
            #     |      8      |
            #     |    ,/|`\    |
            #     |  ,/  |  `\  |
            #     |,/    |    `\|
            #    ,10      |     11
            #  ,/ |      0      | \
            # u   |    ,/ `\    |   v
            #     |  ,6     `7  |
            #     |,/         `\|
            #     1-------------2
        elif gmshId == 13:
            elemType = ElemType.PRISM18; nPe = 18; dim = 3; ordre = 2; nbFaces = 5
            #            w
            #            ^
            #            |
            #            3
            #          ,/|`\
            #        12  |  13
            #      ,/    |    `\
            #     4------14-----5
            #     |      8      |
            #     |    ,/|`\    |
            #     |  15  |  16  |
            #     |,/    |    `\|
            #    ,10-----17-----11
            #  ,/ |      0      | `\
            # u   |    ,/ `\    |    v
            #     |  ,6     `7  |
            #     |,/         `\|
            #     1------9------2
        elif gmshId == 7:
            elemType = ElemType.PYRA5; nPe = 5; dim = 3; ordre = 1; nbFaces = 5
            #                4
            #              ,/|\
            #            ,/ .'|\
            #          ,/   | | \
            #        ,/    .' | `.
            #      ,/      |  '.  \
            #    ,/       .' w |   \
            #  ,/         |  ^ |    \
            # 0----------.'--|-3    `.
            #  `\        |   |  `\    \
            #    `\     .'   +----`\ - \ -> v
            #      `\   |    `\     `\  \
            #        `\.'      `\     `\`
            #           1----------------2
            #                     `\
            #                       u
        elif gmshId == 19:
            elemType = ElemType.PYRA13; nPe = 13; dim = 3; ordre = 2; nbFaces = 5
            #                4
            #              ,/|\
            #            ,/ .'|\
            #          ,/   | | \
            #        ,/    .' | `.
            #      ,7      |  12  \
            #    ,/       .' w |   \
            #  ,/         9  ^ |    11
            # 0--------6-.'--|-3    `.
            #  `\        |   |  `\    \
            #    `5     .'   +----10 - \ -> v
            #      `\   |    `\     `\  \
            #        `\.'      `\       `\`
            #           1--------8-------2
            #                     `\
            #                       u
        elif gmshId == 14:
            elemType = ElemType.PYRA14; nPe = 14; dim = 3; ordre = 2; nbFaces = 5
            #                4
            #              ,/|\
            #            ,/ .'|\
            #          ,/   | | \
            #        ,/    .' | `.
            #      ,7      |  12  \
            #    ,/       .' w |   \
            #  ,/         9  ^ |    11
            # 0--------6-.'--|-3    `.
            #  `\        |   |  `\    \
            #    `5     .'   13---10 - \ -> v
            #      `\   |    `\     `\  \
            #        `\.'      `\     `\`
            #           1--------8-------2
            #                     `\
            #                       u
        else: 
            raise "Type inconnue"
            
        return elemType, nPe, dim, ordre, nbFaces

class ElemType(str, Enum):
    POINT = "POINT"
    SEG2 = "SEG2"
    SEG3 = "SEG3"
    SEG4 = "SEG4"
    SEG5 = "SEG5"
    TRI3 = "TRI3"
    TRI6 = "TRI6"
    TRI10 = "TRI10"
    TRI15 = "TRI15"
    QUAD4 = "QUAD4"
    QUAD8 = "QUAD8"
    QUAD9 = "QUAD9"
    TETRA4 = "TETRA4"
    TETRA10 = "TETRA10"
    HEXA8 = "HEXA8"
    HEXA27 = "HEXA27"
    PRISM6 = "PRISM6"
    PRISM15 = "PRISM15"
    PRISM18 = "PRISM18"
    PYRA5 = "PYRA5"
    PYRA13 = "PYRA13"
    PYRA14 = "PYRA14"

class MatriceType(str, Enum):
    rigi = "rigi"
    masse = "masse"
    beam = "beam"