from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
from typing import cast
from types import LambdaType

from Geom import *
from Gauss import Gauss

import numpy as np
import scipy.sparse as sparse

class ElemType(str, Enum):
    """Implemented element types"""

    POINT = "POINT"
    SEG2 = "SEG2"
    SEG3 = "SEG3"
    SEG4 = "SEG4"
    SEG5 = "SEG5"
    TRI3 = "TRI3"
    TRI6 = "TRI6"
    TRI10 = "TRI10"
    # TRI15 = "TRI15"
    QUAD4 = "QUAD4"
    QUAD8 = "QUAD8"
    # QUAD9 = "QUAD9"
    TETRA4 = "TETRA4"
    TETRA10 = "TETRA10"
    HEXA8 = "HEXA8"
    HEXA20 = "HEXA20"
    PRISM6 = "PRISM6"
    PRISM15 = "PRISM15"
    # PRISM18 = "PRISM18"
    # PYRA5 = "PYRA5"
    # PYRA13 = "PYRA13"
    # PYRA14 = "PYRA14"

class MatrixType(str, Enum):
    rigi = "rigi"
    mass = "mass"
    beam = "beam"

class GroupElem(ABC):
    """A mesh uses several groups of elements. For example, a mesh with cubes (HEXA8) uses :
    - POINT (dim=0)
    - SEG2 (dim=1)
    - QUAD4 (dim=2)
    - HEXA8 (dim=3)
    """

    ################################################ STATIC ##################################################

    @staticmethod
    def get_MatrixType() -> list[MatrixType]:
        """available matrix type"""
        liste = list(MatrixType)
        return liste

    @staticmethod
    def get_Types1D() -> list[ElemType]:
        """1D element type"""        
        liste1D = [ElemType.SEG2, ElemType.SEG3, ElemType.SEG4]
        return liste1D

    @staticmethod
    def get_Types2D() -> list[ElemType]:
        """2D element type"""        
        liste2D = [ElemType.TRI3, ElemType.TRI6, ElemType.TRI10, ElemType.QUAD4, ElemType.QUAD8]
        # TODO there are still errors on TRI15, certainly integration points
        return liste2D
    
    @staticmethod
    def get_Types3D() -> list[ElemType]:
        """3D element type"""
        liste3D = [ElemType.TETRA4, ElemType.TETRA10, ElemType.HEXA8, ElemType.HEXA20, ElemType.PRISM6, ElemType.PRISM15]
        return liste3D

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):
        """Building an element group

        Parameters
        ----------
        gmshId : int
            gmsh id
        connect : np.ndarray
            connectivity matrix        
        coordoGlob : np.ndarray
            coordinate matrix (contains all mesh coordinates)
        nodes : np.ndarray
            nodes used by element group
        """

        self.__gmshId = gmshId
        self.__elemType, self.__nPe, self.__dim, self.__ordre, self.__nbFaces, self.__nbCorners = GroupElem_Factory.Get_ElemInFos(gmshId)
        
        # Elements
        self.__elements = np.arange(connect.shape[0], dtype=int)
        self.__connect = connect

        # Noeuds
        self.__nodes = nodes
        self.__coordoGlob = coordoGlob
        self.__coordo = cast(np.ndarray, coordoGlob[nodes])
        
        if self.elemType in GroupElem.get_Types3D():
            self.__inDim = 3
        else:
            if np.abs(self.__coordo)[:,1].max()==0:
                self.__inDim = 1
            if np.abs(self.__coordo)[:,2].max()==0:
                self.__inDim = 2
            else:
                self.__inDim = 3
        
        self._InitMatrix()
    
    def _InitMatrix(self) -> None:
        """Initialize matrix dictionaries for finite element construction"""
        # Dictionaries for each matrix type
        self.__dict_nodes_tags = {}
        self.__dict_elements_tags = {}
        self.__dict_dN_e_pg = {}
        self.__dict_dNv_e_pg = {}
        self.__dict_ddNv_e_pg = {}
        self.__dict_ddN_e_pg = {}
        self.__dict_F_e_pg = {}                
        self.__dict_invF_e_pg = {}                
        self.__dict_jacobian_e_pg = {}   
        self.__dict_B_e_pg = {}
        self.__dict_leftDispPart = {}
        self.__dict_phaseField_ReactionPart_e_pg = {}
        self.__dict_DiffusePart_e_pg = {}
        self.__dict_SourcePart_e_pg = {}

    ################################################ METHODS ##################################################
    
    @property
    def elemType(self) -> ElemType:
        """Element type"""
        return self.__elemType

    @property
    def nPe(self) -> int:
        """Nodes per element"""
        return self.__nPe
    
    @property
    def dim(self) -> int:
        """Element dimension"""
        return self.__dim
    
    @property
    def ordre(self) -> int:
        """Element order"""
        return self.__ordre    

    @property
    def inDim(self) -> int:
        """Dimension in which the elements are located"""
        return self.__inDim

    @property
    def Ne(self) -> int:
        """Number of elements"""
        return self.__connect.shape[0]

    @property
    def nodes(self) -> int:
        """Nodes used by the element group. Node 'n' is on line 'n' in coordoGlob"""
        return self.__nodes.copy()

    @property
    def elements(self) -> np.ndarray:
        """Elements"""
        return self.__elements.copy()

    @property
    def Nn(self) -> int:
        """Number of nodes"""
        return self.__coordo.shape[0]

    @property
    def coordo(self) -> np.ndarray:
        """This matrix contains the element group coordinates (Nn, 3)"""
        return self.__coordo.copy()

    @property
    def coordoGlob(self) -> np.ndarray:
        """This matrix contains all the mesh coordinates (mesh.Nn, 3)"""
        return self.__coordoGlob.copy()

    @property
    def nbFaces(self) -> int:
        """Number of faces per element"""
        return self.__nbFaces
    
    @property
    def nbCorners(self) -> int:
        """Number of corners per element"""
        return self.__nbCorners
    
    @property
    def connect(self) -> np.ndarray:
        """Connectivity matrix (Ne, nPe)"""
        return self.__connect.copy()
    
    def Get_connect_n_e(self) -> sparse.csr_matrix:
        """Sparse matrix of zeros and ones with ones when the node has the element either
        such that: values_n = connect_n_e * values_e\n
        (Nn,1) = (Nn,Ne) * (Ne,1)"""
        # Here, the aim is to construct a matrix which, when multiplied by a values_e vector of size ( Ne x 1 ), will give
        # values_n_e(Nn,1) = connecNoeud(Nn,Ne) values_n_e(Ne,1)
        # where connecNoeud(Nn,:) is a row vector composed of 0 and 1, which will be used to sum values_e[nodes].
        # Then just divide by the number of times the node appears in the line        
        Ne = self.Ne
        nPe = self.nPe
        listElem = np.arange(Ne)

        lignes = self.connect.reshape(-1)

        Nn = int(lignes.max()+1)
        colonnes = np.repeat(listElem, nPe)

        return sparse.csr_matrix((np.ones(nPe*Ne),(lignes, colonnes)),shape=(Nn,Ne))

    @property
    def assembly_e(self) -> np.ndarray:
        """Assembly matrix (Ne, nPe*dim)"""

        nPe = self.nPe
        dim = self.dim        
        taille = nPe * dim

        assembly = np.zeros((self.Ne, taille), dtype=np.int64)
        connect = self.connect

        for d in range(dim):
            colonnes = np.arange(d, taille, dim)
            assembly[:, colonnes] = np.array(connect) * dim + d

        return assembly
    
    def Get_assembly_e(self, dof_n: int) -> np.ndarray:
        """Assembly matrix for specified dof_n (Ne, nPe*dof_n)

        Parameters
        ----------
        dof_n : int
            degree of freedom per node
        """

        nPe = self.nPe
        taille = dof_n*nPe

        assembly = np.zeros((self.Ne, taille), dtype=np.int64)
        connect = self.connect

        for d in range(dof_n):
            colonnes = np.arange(d, taille, dof_n)
            assembly[:, colonnes] = np.array(connect) * dof_n + d

        return assembly    

    def Get_gauss(self, matrixType: MatrixType) -> Gauss:
        """Returns integration points according to matrix type"""
        return Gauss(self.elemType, matrixType)
    
    def Get_weight_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Returns integration point weights according to matrix type"""
        return Gauss(self.elemType, matrixType).weights
    
    def Get_GaussCoordinates_e_p(self, matrixType: MatrixType, elements=np.array([])) -> np.ndarray:
        """Returns integration point coordinates for each element (Ne, p, 3) in the (x,y,z) base."""

        N_scalar = self.Get_N_pg(matrixType)

        # retrieves node coordinates
        coordo = self.__coordoGlob

        # node coordinates for each element
        if elements.size == 0:
            coordo_e = coordo[self.__connect]
        else:
            coordo_e = coordo[self.__connect[elements]]

        # localize coordinates on Gauss points
        coordo_e_p = np.einsum('pij,ejn->epn', N_scalar, coordo_e, optimize='optimal')

        return np.array(coordo_e_p)

    def Get_N_pg_rep(self, matrixType: MatrixType, repeat=1) -> np.ndarray:
        """Shape functions in the reference database

        Parameters
        ----------
        matrixType : MatrixType
            matrix type
        repeat : int, optional
            number of repetitions, by default 1
        
        Returns:
        -------
        . Vector shape functions (pg, rep=2, rep=2*dim), in base (ksi, eta ...)\n
            [Ni 0 . . . Nn 0 \n
            0 Ni . . . 0 Nn]

        . Scalar shape functions (pg, rep=1, nPe), in base (ksi, eta ...)\n
            [Ni . . . Nn] .
        """
        if self.dim == 0: return

        assert isinstance(repeat, int)
        assert repeat >= 1

        N_pg = self.Get_N_pg(matrixType)

        if not isinstance(N_pg, np.ndarray): return

        if repeat <= 1:
            return N_pg
        else:
            taille = N_pg.shape[2]*repeat
            N_vect_pg = np.zeros((N_pg.shape[0] ,repeat , taille))

            for r in range(repeat):
                N_vect_pg[:, r, np.arange(r, taille, repeat)] = N_pg[:,0,:]
            
            return N_vect_pg
    
    def Get_dN_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Derivation of shape functions in real base (epij)\n
        [dN1,x dN2,x dNn,x\n
        dN1,y dN2,y dNn,y]\n        
        """
        assert matrixType in GroupElem.get_MatrixType()

        if matrixType not in self.__dict_dN_e_pg.keys():

            invF_e_pg = self.Get_invF_e_pg(matrixType)

            dN_pg = self.Get_dN_pg(matrixType)

            # Derivation of shape functions in the real base
            dN_e_pg = np.array(np.einsum('epik,pkj->epij', invF_e_pg, dN_pg, optimize='optimal'))            
            self.__dict_dN_e_pg[matrixType] = dN_e_pg

        return self.__dict_dN_e_pg[matrixType].copy()

    def Get_dNv_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Derivation of beam shape functions in real base (epij)\n
        [dNv1,x dNv2,x dNvn,x\n
        dNv1,y dNv2,y dNvn,y]\n
        """
        assert matrixType in GroupElem.get_MatrixType()

        if matrixType not in self.__dict_dNv_e_pg.keys():

            invF_e_pg = self.Get_invF_e_pg(matrixType)

            dNv_pg = self.Get_dNv_pg(matrixType)

            jacobian_e_pg = self.Get_jacobian_e_pg(matrixType)
            Ne = jacobian_e_pg.shape[0]
            pg = self.Get_gauss(matrixType)

            # Create dimension on elements
            dNv_e_pg = dNv_pg[np.newaxis, :, 0, :].repeat(Ne,  axis=0)
            # The length of the beams on each element is retrieved at the integration points.
            l_e_pg = np.einsum('ep,p->ep', jacobian_e_pg, pg.weights, optimize='optimal')
            # Multiply the ddNv2_e_pg and ddNv4_e_pg by the length.
            dNv_e_pg[:,:,1] = np.einsum('ep,e->ep',dNv_e_pg[:,:,1],l_e_pg[:,0])
            dNv_e_pg[:,:,3] = np.einsum('ep,e->ep',dNv_e_pg[:,:,3],l_e_pg[:,1])

            # Derivation of shape functions in the real base
            invF_e_pg = invF_e_pg.reshape((Ne, pg.nPg, 1)).repeat(dNv_e_pg.shape[-1], axis=-1)
            dNv_e_pg = invF_e_pg * dNv_e_pg
            self.__dict_dNv_e_pg[matrixType] = dNv_e_pg

        return self.__dict_dNv_e_pg[matrixType].copy()

    def Get_ddNv_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Derivation (2) of beam shape functions in real base (epij)\n
        [dNv1,xx dNv2,xx dNvn,xx\n
        dNv1,yy dNv2,yy dNvn,yy]\n
        """
        assert matrixType in GroupElem.get_MatrixType()

        if matrixType not in self.__dict_ddNv_e_pg.keys():

            invF_e_pg = self.Get_invF_e_pg(matrixType)

            ddNv_pg = self.Get_ddNv_pg(matrixType)

            jacobian_e_pg = self.Get_jacobian_e_pg(matrixType)
            Ne = self.Ne
            nPe = self.nPe
            pg = self.Get_gauss(matrixType)
            
            # On récupère la longeur des poutres sur chaque element aux points d'intégrations
            # l = np.einsum('ep,p->', jacobien_e_pg, pg.poids, optimize='optimal')
            l_e_pg = np.einsum('ep,p->e', jacobian_e_pg, pg.weights, optimize='optimal').reshape(Ne,1).repeat(invF_e_pg.shape[1], axis=1)
            
            ddNv_e_pg = np.einsum('epik,epik,pkj->epij', invF_e_pg, invF_e_pg, ddNv_pg, optimize='optimal')
            
            colonnes = np.arange(1, nPe*2, 2)
            for colonne in colonnes:
                ddNv_e_pg[:,:,0,colonne] = np.einsum('ep,ep->ep', ddNv_e_pg[:,:,0,colonne], l_e_pg, optimize='optimal')

            self.__dict_ddNv_e_pg[matrixType] = ddNv_e_pg

        return self.__dict_ddNv_e_pg[matrixType].copy()

    def Get_ddN_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Derivation (2) of shape functions in real base (epij)\n
        [dN1,xx dN2,xx dNn,xx\n
        dN1,yy dN2,yy dNn,yy]\n
        """
        assert matrixType in GroupElem.get_MatrixType()

        if matrixType not in self.__dict_ddN_e_pg.keys():

            invF_e_pg = self.Get_invF_e_pg(matrixType)

            ddN_pg = self.Get_ddN_pg(matrixType)

            # Derivé des fonctions de formes dans la base réele
            ddN_e_pg = np.array(np.einsum('epik,pkj->epij', invF_e_pg, ddN_pg, optimize='optimal'))
            self.__dict_ddN_e_pg[matrixType] = ddN_e_pg

        return self.__dict_ddN_e_pg[matrixType].copy()

    def Get_B_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Derivation of shape functions in the real base for the displacement problem (e, pg, (3 or 6), nPe*dim)\n
        2D example:\n
        [dN1,x 0 dN2,x 0 dNn,x 0\n
        0 dN1,y 0 dN2,y 0 dNn,y\n
        dN1,y dN1,x dN2,y dN2,x dN3,y dN3,x]\n

        (epij) In the element base and in Kelvin Mandel
        """
        assert matrixType in GroupElem.get_MatrixType()

        if matrixType not in self.__dict_B_e_pg.keys():

            dN_e_pg = self.Get_dN_e_pg(matrixType)

            nPg = self.Get_gauss(matrixType).nPg
            nPe = self.nPe
            dim = self.dim
            listnPe = np.arange(nPe)
            
            colonnes0 = np.arange(0, nPe*dim, dim)
            colonnes1 = np.arange(1, nPe*dim, dim)

            if self.dim == 2:
                B_e_pg = np.array([[np.zeros((3, nPe*dim))]*nPg]*self.Ne)
                # Derivation of shape functions in the real vector basis
                
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
            B_e_pg = Materials.Displacement_Model.AppliqueCoefSurBrigi(dim, B_e_pg)

            self.__dict_B_e_pg[matrixType] = B_e_pg
        
        return self.__dict_B_e_pg[matrixType].copy()

    def Get_leftDispPart(self, matrixType: MatrixType) -> np.ndarray:
        """Left side of local displacement matrices\n
        Ku_e = jacobian_e_pg * weight_pg * B_e_pg' * c_e_pg * B_e_pg\n
        
        Returns (epij) -> jacobian_e_pg * weight_pg * B_e_pg'.
        """

        assert matrixType in GroupElem.get_MatrixType()

        if matrixType not in self.__dict_leftDispPart.keys():
            
            jacobian_e_pg = self.Get_jacobian_e_pg(matrixType)
            weight_pg = self.Get_gauss(matrixType).weights
            B_e_pg = self.Get_B_e_pg(matrixType)

            leftDepPart = np.einsum('ep,p,epij->epji', jacobian_e_pg, weight_pg, B_e_pg, optimize='optimal')

            self.__dict_leftDispPart[matrixType] = leftDepPart

        return self.__dict_leftDispPart[matrixType].copy()
    
    def Get_ReactionPart_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Returns the part that builds the reaction term (scalar).
        ReactionPart_e_pg = jacobian_e_pg * weight_pg * r_e_pg * N_pg' * N_pg\n
        
        Returns -> jacobian_e_pg * weight_pg * N_pg' * N_pg
        """

        assert matrixType in GroupElem.get_MatrixType()

        if matrixType not in self.__dict_phaseField_ReactionPart_e_pg.keys():

            jacobian_e_pg = self.Get_jacobian_e_pg(matrixType)
            weight_pg = self.Get_gauss(matrixType).weights
            N_pg = self.Get_N_pg_rep(matrixType, 1)

            ReactionPart_e_pg = np.einsum('ep,p,pki,pkj->epij', jacobian_e_pg, weight_pg, N_pg, N_pg, optimize='optimal')

            self.__dict_phaseField_ReactionPart_e_pg[matrixType] = ReactionPart_e_pg
        
        return self.__dict_phaseField_ReactionPart_e_pg[matrixType].copy()
    
    def Get_DiffusePart_e_pg(self, matrixType: MatrixType, A: np.ndarray) -> np.ndarray:
        """Returns the part that builds the diffusion term (scalar).
        DiffusePart_e_pg = jacobian_e_pg * weight_pg * k * dN_e_pg' * A * dN_e_pg\n
        
        Returns -> jacobian_e_pg * weight_pg * dN_e_pg' * A * dN_e_pg
        """

        assert matrixType in GroupElem.get_MatrixType()

        if matrixType not in self.__dict_DiffusePart_e_pg.keys():

            assert len(A.shape) == 2, "A must be a 2D array."

            jacobien_e_pg = self.Get_jacobian_e_pg(matrixType)
            weight_pg = self.Get_gauss(matrixType).weights
            dN_e_pg = self.Get_dN_e_pg(matrixType)

            DiffusePart_e_pg = np.einsum('ep,p,epki,kl,eplj->epij', jacobien_e_pg, weight_pg, dN_e_pg, A, dN_e_pg, optimize='optimal')

            self.__dict_DiffusePart_e_pg[matrixType] = DiffusePart_e_pg
        
        return self.__dict_DiffusePart_e_pg[matrixType].copy()

    def Get_SourcePart_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Returns the part that builds the source term (scalar).
        SourcePart_e_pg = jacobian_e_pg, weight_pg, f_e_pg, N_pg'\n
        
        Returns -> jacobian_e_pg, weight_pg, N_pg'
        """

        assert matrixType in GroupElem.get_MatrixType()

        if matrixType not in self.__dict_SourcePart_e_pg.keys():

            jacobian_e_pg = self.Get_jacobian_e_pg(matrixType)
            weight_pg = self.Get_gauss(matrixType).weights
            N_pg = self.Get_N_pg_rep(matrixType, 1)

            SourcePart_e_pg = np.einsum('ep,p,pij->epji', jacobian_e_pg, weight_pg, N_pg, optimize='optimal') #le ji a son importance pour la transposé

            self.__dict_SourcePart_e_pg[matrixType] = SourcePart_e_pg
        
        return self.__dict_SourcePart_e_pg[matrixType].copy()
    
    def __Get_sysCoord_e(self):
        """Base change matrix for elements (Ne,3,3)"""

        coordo = self.coordoGlob

        if self.dim in [0,3]:
            sysCoord_e = np.eye(3)
            sysCoord_e = sysCoord_e[np.newaxis, :].repeat(self.Ne, axis=0)
        
        elif self.dim in [1,2]:
            # 2D lines or elements

            points1 = coordo[self.__connect[:,0]]
            points2 = coordo[self.__connect[:,1]]

            i = points2-points1
            # Normalize
            i = np.einsum('ei,e->ei',i, 1/np.linalg.norm(i, axis=1), optimize='optimal')

            if self.dim == 1:
                # Segments

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
                
                if "TRI" in self.elemType:
                    points3 = coordo[self.__connect[:,2]]
                elif "QUAD" in self.elemType:
                    points3 = coordo[self.__connect[:,3]]                

                j = points3-points1
                j = np.einsum('ei,e->ei',j, 1/np.linalg.norm(j, axis=1), optimize='optimal')
                
                k = np.cross(i, j, axis=1)
                k = np.einsum('ei,e->ei',k, 1/np.linalg.norm(k, axis=1), optimize='optimal')

                j = np.cross(k,i)
                j = np.einsum('ei,e->ei',j, 1/np.linalg.norm(j, axis=1), optimize='optimal')

            sysCoord_e = np.zeros((self.Ne, 3, 3))
            
            sysCoord_e[:,:,0] = i
            sysCoord_e[:,:,1] = j
            sysCoord_e[:,:,2] = k

        return sysCoord_e
        
    @property
    def sysCoord_e(self) -> np.ndarray:
        """Base change matrix for each element (3D)\n
        [ix, jx, kx\n
        iy, jy, ky\n
        iz, jz, kz]\n
        
        such as coordo_e * sysCoordLocal_e -> coordinates of nodes in element base"""
        return self.__Get_sysCoord_e()

    @property
    def sysCoordLocal_e(self) -> np.ndarray:
        """Base change matrix for each element (dim)"""
        return self.sysCoord_e[:,:,range(self.dim)]

    @property
    def area(self) -> float:
        """Area covered by elements"""
        if self.dim == 1: return
        matrixType = MatrixType.rigi
        aire = np.einsum('ep,p->', self.Get_jacobian_e_pg(matrixType), self.Get_gauss(matrixType).weights, optimize='optimal')
        return float(aire)

    @property
    def Ix(self) -> float:
        """Quadratic moment following x"""
        if self.dim != 2: return

        matrixType = MatrixType.mass

        coordo_e_p = self.Get_GaussCoordinates_e_p(matrixType)
        x = coordo_e_p[:, :, 0]

        Ix = np.einsum('ep,p,ep->', self.Get_jacobian_e_pg(matrixType), self.Get_gauss(matrixType).weights, x**2, optimize='optimal')
        return float(Ix)

    @property
    def Iy(self) -> float:
        """Quadratic moment following y"""
        if self.dim != 2: return

        matrixType = MatrixType.mass

        coordo_e_p = self.Get_GaussCoordinates_e_p(matrixType)
        y = coordo_e_p[:, :, 1]

        Iy = np.einsum('ep,p,ep->', self.Get_jacobian_e_pg(matrixType), self.Get_gauss(matrixType).weights, y**2, optimize='optimal')
        return float(Iy)

    @property
    def Ixy(self) -> float:
        """Quadratic moment following xy"""
        if self.dim != 2: return

        matrixType = MatrixType.mass

        coordo_e_p = self.Get_GaussCoordinates_e_p(matrixType)
        x = coordo_e_p[:, :, 0]
        y = coordo_e_p[:, :, 1]

        Ixy = np.einsum('ep,p,ep,ep->', self.Get_jacobian_e_pg(matrixType), self.Get_gauss(matrixType).weights, x, y, optimize='optimal')
        return float(Ixy)

    @property
    def volume(self) -> float:
        """Volume covered by elements"""
        if self.dim != 3: return
        matrixType = MatrixType.mass
        volume = np.einsum('ep,p->', self.Get_jacobian_e_pg(matrixType), self.Get_gauss(matrixType).weights, optimize='optimal')
        return float(volume)        

    def Get_F_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Returns the Jacobian matrix
        This matrix describes the variations of the axes from the reference element to the real element.
        Transforms the reference element to the real element with invF_e_pg"""
        if self.dim == 0: return
        if matrixType not in self.__dict_F_e_pg.keys():

            coordo_n = self.coordoGlob[:]

            coordo_e = coordo_n[self.__connect]

            dim = self.dim

            if dim == self.inDim:
                nodesBase = coordo_e.copy()               
            else:
                sysCoordLocal_e = self.sysCoordLocal_e # base change matrix for each element
                nodesBase = np.einsum('eij,ejk->eik', coordo_e, sysCoordLocal_e, optimize='optimal') #node coordinates in the elements base

            nodesBaseDim = nodesBase[:,:,range(dim)]

            dN_pg = self.Get_dN_pg(matrixType)

            F_e_pg = np.array(np.einsum('pik,ekj->epij', dN_pg, nodesBaseDim, optimize='optimal'))
            
            self.__dict_F_e_pg[matrixType] = F_e_pg

        return self.__dict_F_e_pg[matrixType].copy()
    
    def Get_jacobian_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Returns the jacobians\n
        variation in size (length, area or volume) between the reference element and the real element
        """
        if self.dim == 0: return
        if matrixType not in self.__dict_jacobian_e_pg.keys():

            F_e_pg = self.Get_F_e_pg(matrixType)

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

            # test = np.linalg.det(F_e_pg) - jacobien_e_pg

            # self.__dict_jacobien_e_pg[matrixType] = jacobien_e_pg
            self.__dict_jacobian_e_pg[matrixType] = np.abs(jacobien_e_pg)

        return self.__dict_jacobian_e_pg[matrixType].copy()
    
    def Get_invF_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Returns the inverse of the Jacobian matrix
        is used to obtain the derivative of the dN_e_pg shape functions in the real element
        dN_e_pg = invF_e_pg . dN_pg
        """
        if self.dim == 0: return 
        if matrixType not in self.__dict_invF_e_pg.keys():

            F_e_pg = self.Get_F_e_pg(matrixType)

            if self.dim == 1:
                invF_e_pg = 1/F_e_pg
            elif self.dim == 2:
                # A = [alpha, beta          inv(A) = 1/det * [b, -beta
                #      a    , b   ]                           -a  alpha]

                Ne = F_e_pg.shape[0]
                nPg = F_e_pg.shape[1]
                invF_e_pg = np.zeros((Ne,nPg,2,2))

                det = self.Get_jacobian_e_pg(matrixType)

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
                # optimized such that invF_e_pg = 1/det * Adj(F_e_pg)
                # https://fr.wikihow.com/calculer-l'inverse-d'une-matrice-3x3

                det = self.Get_jacobian_e_pg(matrixType)

                FT_e_pg = np.einsum('epij->epji', F_e_pg, optimize='optimal')

                a00 = FT_e_pg[:,:,0,0]; a01 = FT_e_pg[:,:,0,1]; a02 = FT_e_pg[:,:,0,2]
                a10 = FT_e_pg[:,:,1,0]; a11 = FT_e_pg[:,:,1,1]; a12 = FT_e_pg[:,:,1,2]
                a20 = FT_e_pg[:,:,2,0]; a21 = FT_e_pg[:,:,2,1]; a22 = FT_e_pg[:,:,2,2]

                det00 = (a11*a22) - (a21*a12); det01 = (a10*a22) - (a20*a12); det02 = (a10*a21) - (a20*a11)
                det10 = (a01*a22) - (a21*a02); det11 = (a00*a22) - (a20*a02); det12 = (a00*a21) - (a20*a01)
                det20 = (a01*a12) - (a11*a02); det21 = (a00*a12) - (a10*a02); det22 = (a00*a11) - (a10*a01)

                invF_e_pg = np.zeros_like(F_e_pg)

                # Don't forget the - or + !!!
                invF_e_pg[:,:,0,0] = det00/det; invF_e_pg[:,:,0,1] = -det01/det; invF_e_pg[:,:,0,2] = det02/det
                invF_e_pg[:,:,1,0] = -det10/det; invF_e_pg[:,:,1,1] = det11/det; invF_e_pg[:,:,1,2] = -det12/det
                invF_e_pg[:,:,2,0] = det20/det; invF_e_pg[:,:,2,1] = -det21/det; invF_e_pg[:,:,2,2] = det22/det

                # test = np.array(np.linalg.inv(F_e_pg)) - invF_e_pg

            self.__dict_invF_e_pg[matrixType] = invF_e_pg

        return self.__dict_invF_e_pg[matrixType].copy()

    # Fonctions de formes

    @staticmethod
    def Evaluates_Fonctions_Gauss(fonctions: np.ndarray, gauss: Gauss) -> np.ndarray:
        """Evaluates functions at Gauss points"""
        
        coord = gauss.coord
        nPg = gauss.nPg
        dim = fonctions.shape[1]

        fonctions_pg = np.zeros((nPg, dim, fonctions.shape[0]))

        for pg in range(nPg):
            for n, fonction_dim in enumerate(fonctions):
                for d in range(dim):
                    fonction = fonction_dim[d]
                    if coord.shape[1] == 1:
                        fonctions_pg[pg, d, n] = fonction(coord[pg,0])
                    elif coord.shape[1] == 2:
                        fonctions_pg[pg, d, n] = fonction(coord[pg,0], coord[pg,1])
                    elif coord.shape[1] == 3:
                        fonctions_pg[pg, d, n] = fonction(coord[pg,0], coord[pg,1], coord[pg,2])

        return fonctions_pg

    @abstractmethod
    def _Ntild(self) -> np.ndarray:
        """Shape functions (pg), in base (ksi, eta ...)\n
        [N1, N2, . . . ,Nn]
        """
        pass

    def __Init_Functions(self, order: int) -> np.ndarray:
        """Methods for initializing functions to be evaluated at gauss points."""
        if self.dim == 1 and self.ordre < order:
            fonctions = np.array([lambda x: 0]*self.nPe)
        elif self.dim == 2 and self.ordre < order:
            fonctions = np.array([lambda ksi,eta: 0, lambda ksi,eta: 0]*self.nPe)
        elif self.dim == 3 and self.ordre < order:
            fonctions = np.array([lambda x,y,z: 0,lambda x,y,z: 0,lambda x,y,z: 0]*self.nPe)
        return fonctions

    def Get_N_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluated shape functions (pg), in the base (ksi, eta . . . )
        [N1, N2, . . . ,Nn]
        """
        if self.dim == 0: return

        Ntild = self._Ntild()
        gauss = self.Get_gauss(matrixType)
        N_pg = GroupElem.Evaluates_Fonctions_Gauss(Ntild, gauss)

        return N_pg

    @abstractmethod
    def _dNtild(self) -> np.ndarray:
        """Derivatives of shape functions in the reference element (pg, dim, nPe), in the basis (ksi, eta . . .) \n
        [Ni,ksi . . . Nn,ksi
        Ni,eta ... Nn,eta]
        """
        return self.__Init_Functions(1)
    
    def Get_dN_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluated shape functions derivatives in the reference element (pg, dim, nPe), in the basis (ksi, eta . . .) \n
        [Ni,ksi . . . Nn,ksi
        Ni,eta ... Nn,eta]
        """
        if self.dim == 0: return

        dNtild = self._dNtild()

        gauss = self.Get_gauss(matrixType)
        dN_pg = GroupElem.Evaluates_Fonctions_Gauss(dNtild, gauss)

        return dN_pg    

    @abstractmethod
    def _ddNtild(self) -> np.ndarray:
        """Derivatives of shape functions (2) in the reference element (pg, dim, nPe), in the basis (ksi, eta . . .) \n
        [Ni,ksi ksi . . . Nn,ksi ksi\n
        Ni,eta eta . . . Nn,eta eta]
        """
        return self.__Init_Functions(2)

    def Get_ddN_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluated shape functions derivatives (2) in the reference element (pg, dim, nPe), in the basis (ksi, eta . . .) \n
        [Ni,ksi ksi . . . Nn,ksi ksi\n
        Ni,eta eta . . . Nn,eta eta]
        """
        if self.dim == 0: return

        ddNtild = self._ddNtild()

        gauss = self.Get_gauss(matrixType)
        ddN_pg = GroupElem.Evaluates_Fonctions_Gauss(ddNtild, gauss)

        return ddN_pg

    @abstractmethod
    def _dddNtild(self) -> np.ndarray:
        """Derivatives of shape functions (3) in the reference element (pg, dim, nPe), in the basis (ksi, eta . . .) \n
        [Ni,ksi ksi ksi . . . Nn,ksi ksi ksi\n
        Ni,eta eta eta . . . Nn,eta eta eta]
        """
        return self.__Init_Functions(3)

    def Get_dddN_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluated shape functions derivatives (3) in the reference element (pg, dim, nPe), in the basis (ksi, eta . . .) \n
        [Ni,ksi ksi ksi . . . Nn,ksi ksi ksi\n
        Ni,eta eta eta . . . Nn,eta eta eta]
        """
        if self.elemType == 0: return

        dddNtild = self._dddNtild()

        gauss = self.Get_gauss(matrixType)
        dddN_pg = GroupElem.Evaluates_Fonctions_Gauss(dddNtild, gauss)

        return dddN_pg

    @abstractmethod
    def _ddddNtild(self) -> np.ndarray:
        """Derivatives of shape functions (4) in the reference element (pg, dim, nPe), in the basis (ksi, eta . . .) \n
        [Ni,ksi ksi ksi ksi . . . Nn,ksi ksi ksi ksi\n
        Ni,eta eta eta eta . . . Nn,eta eta eta eta]
        """
        return self.__Init_Functions(4)

    def Get_ddddN_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluated shape functions derivatives (4) in the reference element (pg, dim, nPe), in the basis (ksi, eta . . .) \n
        [Ni,ksi ksi ksi ksi . . . Nn,ksi ksi ksi ksi\n
        Ni,eta eta eta eta . . . Nn,eta eta eta eta]
        """
        if self.elemType == 0: return

        ddddNtild = self._ddddNtild()

        gauss = self.Get_gauss(matrixType)
        ddddN_pg = GroupElem.Evaluates_Fonctions_Gauss(ddddNtild, gauss)

        return ddddN_pg

    # Fonctions de formes pour les poutres

    @abstractmethod
    def _Nvtild(self) -> np.ndarray:
        """Shape functions in the bending beam element (pg, dim, nPe), in the basis (ksi) \n
        [phi_i psi_i . . . phi_n psi_n]
        """
        pass

    def Get_Nv_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Shape functions evaluated in the bending beam element (pg, dim, nPe), in the (ksi) basis \n
        [phi_i psi_i . . . phi_n psi_n]
        """
        if self.dim != 1: return

        Nvtild = self._Nvtild()

        gauss = self.Get_gauss(matrixType)
        Nv_pg = GroupElem.Evaluates_Fonctions_Gauss(Nvtild, gauss)

        return Nv_pg

    @abstractmethod
    def dNvtild(self) -> np.ndarray:
        """Derivatives of shape functions in the bending beam element (pg, dim, nPe), in the (ksi) basis \n
        [phi_i,x psi_i,x . . . phi_n,x psi_n,x]
        """
        pass

    def Get_dNv_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluated shape functions derivatives in the bending beam element (pg, dim, nPe), in the (ksi) basis \n
        [phi_i,x psi_i,x . . . phi_n,x psi_n,x]
        """
        if self.dim != 1: return

        dNvtild = self.dNvtild()

        gauss = self.Get_gauss(matrixType)
        dNv_pg = GroupElem.Evaluates_Fonctions_Gauss(dNvtild, gauss)

        return dNv_pg

    @abstractmethod
    def _ddNvtild(self) -> np.ndarray:
        """Derivatives of shape functions (2) in the bending beam element (pg, dim, nPe), in the (ksi) basis \n
        [phi_i,x psi_i,x . . . phi_n,x psi_n,x]
        """
        return 
    
    def Get_ddNv_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluated shape functions derivatives (2) in the bending beam element (pg, dim, nPe), in the (ksi) basis \n
        [phi_i,x psi_i,x . . . phi_n,x psi_n,x]
        """
        if self.dim != 1: return

        ddNvtild = self._ddNvtild()

        gauss = self.Get_gauss(matrixType)
        ddNv_pg = GroupElem.Evaluates_Fonctions_Gauss(ddNvtild, gauss)

        return ddNv_pg

    def Get_Elements_Nodes(self, nodes: np.ndarray, exclusively=True) -> np.ndarray:
        """Returns elements that exclusively or not use the specified nodes."""
        connect = self.__connect
        connect_n_e = self.Get_connect_n_e()

        if isinstance(nodes, list):
            nodes = np.array(nodes)

        # Check that there are no excess nodes
        # It is possible that the nodes entered do not belong to the group
        if connect_n_e.shape[0] < nodes.max():
            # Remove all excess nodes
            indexNoeudsSansDepassement = np.where(nodes < self.Nn)[0]
            nodes = nodes[indexNoeudsSansDepassement]
        
        lignes, colonnes, valeurs = sparse.find(connect_n_e[nodes])

        elements, counts = np.unique(colonnes, return_counts=True)
        
        if exclusively:
            # Checks if elements exclusively use nodes in the node list
            
            # retrieve nodes used by elements
            nodesElem = np.unique(connect[elements])

            # detects nodes used by elements that are not in the nodes specified
            nodesIntru = list(set(nodesElem) - set(nodes))

            # We detect the list of elements associated with unused nodes
            elemIntru = sparse.find(connect_n_e[nodesIntru])[1]
            elementsIntru = np.unique(elemIntru)

            if elementsIntru.size > 0:
                # Remove detected elements
                elements = list(set(elements) - set(elementsIntru))
                elements = np.array(elements)

        return elements

    def Get_Nodes_Conditions(self, lambdaFunction: LambdaType) -> np.ndarray:
        """Returns nodes that meet the specified conditions.

        Parameters
        ----------
        lambdaFunction : LambdaType
            Function using the x, y and z nodes coordinates and returning a boolean value.

            examples :
            \t lambda x, y, z: (x < 40) & (x > 20) & (y<10) \n
            \t lambda x, y, z: (x == 40) | (x == 50) \n
            \t lambda x, y, z: x >= 0

        Returns
        -------
        np.ndarray
            nodes that meet conditions
        """

        coordo = self.__coordo

        xn = coordo[:,0]
        yn = coordo[:,1]
        zn = coordo[:,2]        

        try:
            arrayTest = np.asarray(lambdaFunction(xn, yn, zn))
            if arrayTest.dtype == bool:
                idx = np.where(arrayTest)[0]
                return self.__nodes[idx].copy()
            else:
                print("The function must return a Boolean.")
        except TypeError:
            print("Must provide a 3-parameter function of type lambda x,y,z: ...")
    
    def Get_Nodes_Point(self, point: Point) -> np.ndarray:
        """Returns nodes on the point."""

        coordo = self.__coordo
        
        idx = np.where((coordo[:,0] == point.x) & (coordo[:,1] == point.y) & (coordo[:,2] == point.z))[0]

        if len(idx) == 0:
            # the previous condition may be too restrictive

            tolerance = 1e-12
            
            dec = 10

            decX = np.abs(coordo[:,0].min()) + dec
            decY = np.abs(coordo[:,1].min()) + dec
            decZ = np.abs(coordo[:,2].min()) + dec

            x = point.x + decX
            y = point.y + decY
            z = point.z + decZ

            coordo = coordo + [decX, decY, decZ]

            erreurX = np.abs((coordo[:,0]-x)/coordo[:,0])
            erreurY = np.abs((coordo[:,1]-y)/coordo[:,1])
            if self.inDim == 3:
                erreurZ = np.abs((coordo[:,2]-z)/coordo[:,2])
            else:
                erreurZ = 0
            
            idx = np.where((erreurX <= tolerance) & (erreurY <= tolerance) & (erreurZ <= tolerance))[0]

        return self.__nodes[idx].copy()

    def Get_Nodes_Line(self, line: Line) -> np.ndarray:
        """Returns the nodes on the line."""
        
        vectUnitaire = line.unitVector

        coordo = self.__coordo

        vect = coordo-line.coordo[0]

        prodScalaire = np.einsum('i,ni-> n', vectUnitaire, vect, optimize='optimal')
        prodVecteur = np.cross(vect, vectUnitaire)
        norm = np.linalg.norm(prodVecteur, axis=1)

        eps = 1e-12

        idx = np.where((norm<eps) & (prodScalaire>=-eps) & (prodScalaire<=line.length+eps))[0]

        return self.__nodes[idx].copy()
    
    def Get_Nodes_Domain(self, domain: Domain) -> np.ndarray:
        """Returns nodes in the domain."""

        coordo = self.__coordo

        eps = 1e-12

        idx = np.where( (coordo[:,0] >= domain.pt1.x-eps) & (coordo[:,0] <= domain.pt2.x+eps) &
                        (coordo[:,1] >= domain.pt1.y-eps) & (coordo[:,1] <= domain.pt2.y+eps) &
                        (coordo[:,2] >= domain.pt1.z-eps) & (coordo[:,2] <= domain.pt2.z+eps))[0]
        
        return self.__nodes[idx].copy()

    def Get_Nodes_Circle(self, circle: Circle) -> np.ndarray:
        """Returns the nodes in the circle."""

        coordo = self.__coordo

        eps = 1e-12

        idx = np.where(np.sqrt((coordo[:,0]-circle.center.x)**2+(coordo[:,1]-circle.center.y)**2+(coordo[:,2]-circle.center.z)**2)<=circle.diam/2+eps)

        return self.__nodes[idx]

    def Get_Nodes_Cylindre(self, circle: Circle, direction=[0,0,1]) -> np.ndarray:
        """Returns the nodes in the cylinder."""

        coordo = self.__coordo

        eps = 1e-12
        dx, dy, dz = direction[0], direction[1], direction[2]
        # Probably doesn't work for an oriented cylinder at the moment!

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

        idx = np.where(np.sqrt(conditionX**2+conditionY**2+conditionZ**2)<=circle.diam/2+eps)

        return self.__nodes[idx]

    def Set_Nodes_Tag(self, noeuds: np.ndarray, tag: str):
        """Add a tag to the nodes

        Parameters
        ----------
        nodes : np.ndarray
            list of nodes
        tag : str
            tag used
        """
        if noeuds.size == 0: return
        self.__dict_nodes_tags[tag] = noeuds

    @property
    def nodeTags(self) -> list[str]:
        """Returns node tags."""
        return list(self.__dict_nodes_tags.keys())

    def Set_Elements_Tag(self, nodes: np.ndarray, tag: str):
        """Adds a tag to elements associated with nodes

        Parameters
        ----------
        nodes : np.ndarray
            list of nodes
        tag : str
            tag used
        """

        if nodes.size == 0: return

        # Retrieves elements associated with nodes
        elements = self.Get_Elements_Nodes(nodes=nodes, exclusively=True)

        self.__dict_elements_tags[tag] = elements

    @property
    def elementTags(self) -> list[str]:
        """Returns element tags."""
        return list(self.__dict_elements_tags.keys())

    def Get_Elements_Tag(self, tag: str) -> np.ndarray:
        """Returns elements associated with the tag."""
        if tag in self.__dict_elements_tags:
            return self.__dict_elements_tags[tag]
        else:
            print(f"The {tag} tag is unknown")
            return np.array([])
    
    def Get_Nodes_Tag(self, tag: str) -> np.ndarray:
        """Returns node associated with the tag."""
        if tag in self.__dict_nodes_tags:
            return self.__dict_nodes_tags[tag]
        else:
            print(f"The {tag} tag is unknown")
            return np.array([])
    
    def Locates_sol_e(self, sol: np.ndarray) -> np.ndarray:
        """locates sol on elements"""
        tailleVecteur = self.Nn * self.dim

        if sol.shape[0] == tailleVecteur:
            sol_e = sol[self.assembly_e]
        elif sol.shape[0] == self.Nn:
            sol_e = sol[self.__connect]
        else:
            return
        
        return sol_e
    
    def Get_pointsInElem(self, coordinates: np.ndarray, elem: int) -> np.ndarray:
        """Function that returns the indexes of the coordinates contained in the element.

        Parameters
        ----------
        coordinates : np.ndarray
            coordinates
        elem : int
            element

        Returns
        -------
        np.ndarray
            indexes of coordinates contained in element
        """

        dim = self.__dim        

        tol = 1e-12

        if dim == 0:

            coordo = self.__coordo[self.__connect[elem,0]]

            idx = np.where((coordinates[:,0] == coordo[0]) & (coordinates[:,1] == coordo[1]) & (coordinates[:,2] == coordo[2]))[0]

        elif dim == 1:

            p1 = self.__connect[elem,0]
            p2 = self.__connect[elem,1]

            vect_i = self.__coordo[p2] - self.__coordo[p1]
            longueur = np.linalg.norm(vect_i)
            vect_i = vect_i / longueur # without normalized doesn't work

            vect_j_n = coordinates - self.__coordo[p1]

            cross_n = np.cross(vect_i, vect_j_n, 0, 1)
            norm_n = np.linalg.norm(cross_n, axis=1)

            dot_n = vect_j_n @ vect_i
            
            idx = np.where((norm_n <= tol) & (dot_n >= -tol) & (dot_n <= longueur+tol))[0]

            return idx
        
        elif dim == 2:
            
            coordoMesh = self.__coordo
            indexesFace = self.indexesFaces[:-1]
            nPe = len(indexesFace)
            connectMesh = self.connect[elem, indexesFace]
            coordConnect = coordoMesh[connectMesh]

            # vector calculation
            indexReord = np.append(np.arange(1, nPe), 0)
            # Vectors i for edge segments
            vect_i_b = coordoMesh[connectMesh[indexReord]] - coordoMesh[connectMesh]
            # vect_i_b = np.einsum("ni,n->ni", vect_i_b, 1/np.linalg.norm(vect_i_b, axis=1), optimize="optimal")

            # normal vector to element face
            vect_n = np.cross(vect_i_b[0], -vect_i_b[-1])

            coordinates_n_b = coordinates[:, np.newaxis].repeat(nPe, 1)

            # Construct v vectors from corners
            vectv_n_b = coordinates_n_b - coordConnect

            cross_n_b = np.cross(vect_i_b, vectv_n_b, 1, 2)

            test_n_b = cross_n_b @ vect_n >= -tol

            filtre = np.sum(test_n_b, 1)

            # Returns the index of nodes around the element that meet all conditions
            idx = np.where(filtre == nPe)[0]

            return idx
        
        elif dim == 3:
        
            indexesFaces = self.indexesFaces
            nbFaces = self.nbFaces
            coordo = self.__coordo[self.__connect[elem]]

            if isinstance(self, PRISM6):
                indexesFaces = np.array(indexesFaces)
                faces = np.array([indexesFaces[np.arange(0,4)],
                                  indexesFaces[np.arange(4,8)],
                                  indexesFaces[np.arange(8,12)],
                                  indexesFaces[np.arange(12,15)],
                                  indexesFaces[np.arange(15,18)]], dtype=object)
            elif isinstance(self, PRISM15):
                indexesFaces = np.array(indexesFaces)
                faces = np.array([indexesFaces[np.arange(0,8)],
                                  indexesFaces[np.arange(8,16)],
                                  indexesFaces[np.arange(16,24)],
                                  indexesFaces[np.arange(24,30)],
                                  indexesFaces[np.arange(30,36)]], dtype=object)
            else:
                faces = np.reshape(indexesFaces, (nbFaces,-1))

            p0_f = [f[0] for f in faces]
            p1_f = [f[1] for f in faces]
            p2_f = [f[-1] for f in faces]

            i_f = coordo[p1_f]-coordo[p0_f]
            i_f = np.einsum("ni,n->ni", i_f, 1/np.linalg.norm(i_f, axis=1), optimize="optimal")

            j_f = coordo[p2_f]-coordo[p0_f]
            j_f = np.einsum("ni,n->ni", j_f, 1/np.linalg.norm(j_f, axis=1), optimize="optimal")

            n_f = np.cross(i_f, j_f, 1, 1)
            n_f = np.einsum("ni,n->ni", n_f, 1/np.linalg.norm(n_f, axis=1), optimize="optimal")

            coordinates_n_b = coordinates[:, np.newaxis].repeat(nbFaces, 1)

            v_f = coordinates_n_b - coordo[p0_f]

            t_f = np.einsum("nfi,fi->nf", v_f, n_f, optimize="optimal") >= -tol

            filtre = np.sum(t_f, 1)

            idx = np.where(filtre == nbFaces)[0]

            return idx

    def Get_Nodes_Connect_CoordoInElemRef(self, coordinates: np.ndarray, elements=None):
        """Function to return the nodes in the elements, the connectivity and the coordinates (ksi, eta) of the points.
        return nodes, connect_e_n, coordoInElem_n"""
        
        if elements == None:
            elements = np.arange(self.Ne, dtype=int)

        assert coordinates.shape[1] == 3, "Must be of dimension (n, 3)."

        return self.__Get_Nodes_Connect_CoordoInElemRef(coordinates, elements)    

    def __Get_Nodes_Connect_CoordoInElemRef(self, coordinates_n: np.ndarray, elements_e: np.ndarray):
        """This function locates coordinates in elements.
        We return the detected coordinates, the connectivity matrix between element and coordinates and the coordinates of these nodes in the reference elements, so that we can evaluate the shape functions."""
        
        # element group data
        coordo = self.__coordo
        connect = self.__connect        
        invF_e_pg = self.Get_invF_e_pg("rigi")

        # Detects whether coordinates come from a grid
        repX = np.unique(coordinates_n[:,0], return_counts=True)[1]; stdX = np.std(repX)
        repY = np.unique(coordinates_n[:,1], return_counts=True)[1]; stdY = np.std(repY)
        repZ = np.unique(coordinates_n[:,2], return_counts=True)[1]; stdZ = np.std(repZ)        

        if coordinates_n.dtype==int and stdX == 0 and stdY == 0 and stdZ == 0:            
            useGrid = True
            # here we retrieve the number of Y and X layers
            nY = int(np.mean(repX))
            nX = int(np.mean(repY))
        else:
            useGrid = False

        def Get_coordoInZoneElem(coord: np.ndarray) -> np.ndarray:
            """Retrieve coordinates_n indexes in the coordinates zone.
            This function is used to perform a presort"""

            if useGrid:

                xe = np.arange(np.floor(coord[:,0].min()), np.ceil(coord[:,0].max()), dtype=int)
                ye = np.arange(np.floor(coord[:,1].min()),np.ceil(coord[:,1].max()), dtype=int)
                Xe, Ye = np.meshgrid(xe,ye)
                
                idx = np.ravel_multi_index(np.concatenate(([Ye.ravel()],[Xe.ravel()])), (nY, nX))
            
            else:

                idx = np.where((coordinates_n[:,0] >= np.min(coord[:,0])) &
                                (coordinates_n[:,0] <= np.max(coord[:,0])) &
                                (coordinates_n[:,1] >= np.min(coord[:,1])) &
                                (coordinates_n[:,1] <= np.max(coord[:,1])) &
                                (coordinates_n[:,2] >= np.min(coord[:,2])) &
                                (coordinates_n[:,2] <= np.max(coord[:,2])))[0]               

            return idx
        
        # connection matrix containing the nodes used by the elements
        connect_e_n = []
        # node coordinates in the element's reference base
        coordoInElem_n = np.zeros_like(coordinates_n[:,:self.inDim], dtype=float)
        # nodes identified
        nodes = []        
        
        def ResearchFunction(e: int):
            # Retrieve element node coordinates
            coordoZone = coordo[connect[e]]

            # Retrieves indexes in coordinates_n that are within the element's bounds
            idxAroundElem = Get_coordoInZoneElem(coordoZone)

            # Returns the index of nodes around the element that meet all conditions
            idxInElem = self.Get_pointsInElem(coordinates_n[idxAroundElem], e)

            # nodes that meet all conditions
            nodesInElement = idxAroundElem[idxInElem]

            # coordinates of these nodes in the element in the real base
            nodesCoordinatesInElem = coordinates_n[nodesInElement] - coordoZone[0]
            # coordinates of these nodes in the element in the element base
            nodesCoordinatesInElemRef = nodesCoordinatesInElem[:,:self.inDim] @ invF_e_pg[e,0]

            # offset if necessary
            # here introduces an offset, as the first node is not necessarily at the origin of the reference frame
            if "SEG" in self.elemType:
                dec = [-1]
            elif "TRI" in self.elemType:
                dec = [0, 0]
            elif "QUAD" in self.elemType:
                dec = [-1, -1]
            elif "TETRA" in self.elemType:
                dec = [0, 0, 0]
            elif "HEXA" in self.elemType:
                dec = [-1, -1, -1]
            elif "PRISM" in self.elemType:
                dec = [0, 0, -1]

            nodesCoordinatesInElemRef = nodesCoordinatesInElemRef + dec

            connect_e_n.append(nodesInElement)

            coordoInElem_n[nodesInElement,:] = nodesCoordinatesInElemRef

            nodes.extend(nodesInElement)

        [ResearchFunction(e) for e in elements_e]
        
        connect_e_n = np.array(connect_e_n, dtype=object)

        nodes = np.asarray(nodes)

        return nodes, connect_e_n, coordoInElem_n

    @abstractproperty
    def indexesTriangles(self) -> list[int]:
        """List of indexes to form the triangles of an element that will be used for the 2D trisurf function"""
        pass

    @property
    def indexesSegments(self) -> np.ndarray:
        """Indexes for corner formation"""
        
        if self.__dim == 1:
            return np.array([[0, 1]], dtype=int)
        elif self.__dim == 2:
            segments = np.zeros((self.nbCorners, 2), dtype=int)
            segments[:,0] = np.arange(self.nbCorners)
            segments[:,1] = np.append(np.arange(1, self.nbCorners, 1), 0)
            return segments
        elif self.__dim == 3:
            raise Exception("To be defined for 3D element groups.")
    
    def Get_dict_connect_Triangle(self) -> dict[ElemType, np.ndarray]:
        """Transform the connectivity matrix to pass it to the trisurf function in 2D.
        For example, for a quadrangle, we construct two triangles
        for a 6-node triangle, 4 triangles are constructed

        Returns a dictionary by type
        """
        assert self.dim == 2

        indexes = self.indexesTriangles

        dict_connect_triangle = {}
        dict_connect_triangle[self.elemType] = np.array(self.__connect[:, indexes]).reshape(-1,3)

        # TODO essayer de faire aussi avec les elements genre pour SEG2 -> dict_connect_triangle[self. elemType] = self.__connect[:,[0,1,0]] ? Est ce que ça marche ?

        return dict_connect_triangle

    @abstractproperty
    def indexesFaces(self) -> list[int]:
        """List of indexes to form the faces that make up the element"""
        pass    

class GroupElem_Factory:

    @staticmethod
    def Get_ElemInFos(gmshId: int) -> tuple:
        """return elemType, nPe, dim, order, nbFaces
        """
        if gmshId == 15:
            elemType = ElemType.POINT; nPe = 1; dim = 0; order = 0; nbFaces = 0; nbCorners = 0
        elif gmshId == 1:
            elemType = ElemType.SEG2; nPe = 2; dim = 1; order = 1; nbFaces = 0; nbCorners = 2
            #       v
            #       ^
            #       |
            #       |
            #  0----+----1 --> u
        elif gmshId == 8:
            elemType = ElemType.SEG3; nPe = 3; dim = 1; order = 2; nbFaces = 0; nbCorners = 2
            #       v
            #       ^
            #       |
            #       |
            #  0----2----1 --> u
        elif gmshId == 26:
            elemType = ElemType.SEG4; nPe = 4; dim = 1; order = 3; nbFaces = 0; nbCorners = 2
            #        v
            #        ^
            #        |
            #        |
            #  0---2-+-3---1 --> u
        elif gmshId == 27:
            elemType = ElemType.SEG5; nPe = 5; dim = 1; order = 4; nbFaces = 0; nbCorners = 2
            
        elif gmshId == 2:
            elemType = ElemType.TRI3; nPe = 3; dim = 2; order = 2; nbFaces = 1; nbCorners = 3
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
            elemType = ElemType.TRI6; nPe = 6; dim = 2; order = 2; nbFaces = 1; nbCorners = 3
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
            elemType = ElemType.TRI10; nPe = 10; dim = 2; order = 3; nbFaces = 1; nbCorners = 3
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
            elemType = ElemType.TRI15; nPe = 15; dim = 2; order = 4; nbFaces = 1; nbCorners = 3
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
            elemType = ElemType.QUAD4; nPe = 4; dim = 2; order = 1; nbFaces = 1; nbCorners = 4
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
            elemType = ElemType.QUAD8; nPe = 8; dim = 2; order = 2; nbFaces = 1; nbCorners = 4
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
            elemType = ElemType.QUAD9; nPe = 9; dim = 2; order = 3; nbFaces = 1; nbCorners = 4
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
            elemType = ElemType.TETRA4; nPe = 4; dim = 3; order = 1; nbFaces = 4; nbCorners = 4
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
            elemType = ElemType.TETRA10; nPe = 10; dim = 3; order = 2; nbFaces = 4; nbCorners = 4
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
            elemType = ElemType.HEXA8; nPe = 8; dim = 3; order = 1; nbFaces = 6; nbCorners = 8
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
        elif gmshId == 17:
            elemType = ElemType.HEXA20; nPe = 20; dim = 3; order = 2; nbFaces = 6; nbCorners = 8
            #        v
            # 3----13----2
            # |\     ^   |\
            # | 15   |   | 14
            # 9  \   |   11 \
            # |   7----19+---6
            # |   |  +-- |-- | -> u
            # 0---+-8-\--1   |
            #  \  17   \  \  18
            #  10 |     \  12|
            #    \|      w  \|
            #     4----16----5
        elif gmshId == 6:
            elemType = ElemType.PRISM6; nPe = 6; dim = 3; order = 1; nbFaces = 5; nbCorners = 6
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
            elemType = ElemType.PRISM15; nPe = 15; dim = 3; order = 2; nbFaces = 5; nbCorners = 6
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
            #     1------9------2
        elif gmshId == 13:
            elemType = ElemType.PRISM18; nPe = 18; dim = 3; order = 2; nbFaces = 5; nbCorners = 6
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
            elemType = ElemType.PYRA5; nPe = 5; dim = 3; order = 1; nbFaces = 5; nbCorners = 5
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
            elemType = ElemType.PYRA13; nPe = 13; dim = 3; order = 2; nbFaces = 5; nbCorners = 5
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
            elemType = ElemType.PYRA14; nPe = 14; dim = 3; order = 2; nbFaces = 5; nbCorners = 5
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
            raise Exception("Element type unknown")
            
        return elemType, nPe, dim, order, nbFaces, nbCorners
    
    @staticmethod
    def Create_GroupElem(gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray) -> GroupElem:

        params = (gmshId, connect, coordoGlob, nodes)

        elemType = GroupElem_Factory.Get_ElemInFos(gmshId)[0]
        
        if elemType == ElemType.POINT:
            return POINT(*params)
        elif elemType == ElemType.SEG2:
            return SEG2(*params)
        elif elemType == ElemType.SEG3:
            return SEG3(*params)
        elif elemType == ElemType.SEG4:
            return SEG4(*params)
        elif elemType == ElemType.SEG5:
            return SEG5(*params)
        elif elemType == ElemType.TRI3:
            return TRI3(*params)
        elif elemType == ElemType.TRI6:
            return TRI6(*params)
        elif elemType == ElemType.TRI10:
            return TRI10(*params)
        elif elemType == ElemType.QUAD4:
            return QUAD4(*params)
        elif elemType == ElemType.QUAD8:
            return QUAD8(*params)
        elif elemType == ElemType.TETRA4:
            return TETRA4(*params)
        elif elemType == ElemType.TETRA10:
            return TETRA10(*params)
        elif elemType == ElemType.HEXA8:
            return HEXA8(*params)
        elif elemType == ElemType.HEXA20:
            return HEXA20(*params)
        elif elemType == ElemType.PRISM6:
            return PRISM6(*params)
        elif elemType == ElemType.PRISM15:
                return PRISM15(*params)
        else:
            raise Exception("Element type unknown")


class POINT(GroupElem):
    
    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def indexesTriangles(self) -> list[int]:
        return super().indexesTriangles

    @property
    def indexesFaces(self) -> list[int]:
        return [0]

    def _Ntild(self) -> np.ndarray:
        pass

    def _dNtild(self) -> np.ndarray:
        pass

    def _ddNtild(self) -> np.ndarray:
        pass
    
    def _dddNtild(self) -> np.ndarray:
        pass

    def _ddddNtild(self) -> np.ndarray:
        pass

    def _Nvtild(self) -> np.ndarray:
        pass

    def dNvtild(self) -> np.ndarray:
        pass

    def _ddNvtild(self) -> np.ndarray:
        pass

class SEG2(GroupElem):    
    #       v
    #       ^
    #       |
    #       |
    #  0----+----1 --> u

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def indexesTriangles(self) -> list[int]:
        return super().indexesTriangles
    
    @property
    def indexesFaces(self) -> list[int]:
        return [0,1]

    def _Ntild(self) -> np.ndarray:

        N1t = lambda x: 0.5*(1-x)
        N2t = lambda x: 0.5*(1+x)

        Ntild = np.array([N1t, N2t]).reshape(-1, 1)

        return Ntild
    
    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda x: -0.5]
        dN2t = [lambda x: 0.5]

        dNtild = np.array([dN1t, dN2t]).reshape(-1,1)

        return dNtild

    def _ddNtild(self) -> np.ndarray:
        return super()._ddNtild()

    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()
    
    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()

    def _Nvtild(self) -> np.ndarray:

        phi_1 = lambda x : 0.5 + -0.75*x + 0.0*x**2 + 0.25*x**3
        psi_1 = lambda x : 0.125 + -0.125*x + -0.125*x**2 + 0.125*x**3
        phi_2 = lambda x : 0.5 + 0.75*x + 0.0*x**2 + -0.25*x**3
        psi_2 = lambda x : -0.125 + -0.125*x + 0.125*x**2 + 0.125*x**3

        Nvtild = np.array([phi_1, psi_1, phi_2, psi_2]).reshape(-1,1)

        return Nvtild

    def dNvtild(self) -> np.ndarray:

        phi_1_x = lambda x : -0.75 + 0.0*x + 0.75*x**2
        psi_1_x = lambda x : -0.125 + -0.25*x + 0.375*x**2
        phi_2_x = lambda x : 0.75 + 0.0*x + -0.75*x**2
        psi_2_x = lambda x : -0.125 + 0.25*x + 0.375*x**2

        dNvtild = np.array([phi_1_x, psi_1_x, phi_2_x, psi_2_x]).reshape(-1,1)

        return dNvtild

    def _ddNvtild(self) -> np.ndarray:

        phi_1_xx = lambda x : 0.0 + 1.5*x
        psi_1_xx = lambda x : -0.25 + 0.75*x
        phi_2_xx = lambda x : 0.0 + -1.5*x
        psi_2_xx = lambda x : 0.25 + 0.75*x

        ddNvtild = np.array([phi_1_xx, psi_1_xx, phi_2_xx, psi_2_xx]).reshape(-1,1)

        return ddNvtild

class SEG3(GroupElem):
    #       v
    #       ^
    #       |
    #       |
    #  0----2----1 --> u

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def indexesTriangles(self) -> list[int]:
        return super().indexesTriangles
    
    @property
    def indexesFaces(self) -> list[int]:
        return [0,2,1]

    def _Ntild(self) -> np.ndarray:

        N1t = lambda x: -0.5*(1-x)*x
        N2t = lambda x: 0.5*(1+x)*x
        N3t = lambda x: (1+x)*(1-x)

        Ntild = np.array([N1t, N2t, N3t]).reshape(-1, 1)

        return Ntild

    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda x: x-0.5]
        dN2t = [lambda x: x+0.5]
        dN3t = [lambda x: -2*x]

        dNtild = np.array([dN1t, dN2t, dN3t]).reshape(-1,1)

        return dNtild

    def _ddNtild(self) -> np.ndarray:

        ddN1t = [lambda x: 1]
        ddN2t = [lambda x: 1]
        ddN3t = [lambda x: -2]

        ddNtild = np.array([ddN1t, ddN2t, ddN3t])

        return ddNtild

    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()

    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()
        
    def _Nvtild(self) -> np.ndarray:

        phi_1 = lambda x : 0.0 + 0.0*x + 1.0*x**2 + -1.25*x**3 + -0.5*x**4 + 0.75*x**5
        psi_1 = lambda x : 0.0 + 0.0*x + 0.125*x**2 + -0.125*x**3 + -0.125*x**4 + 0.125*x**5
        phi_2 = lambda x : 0.0 + 0.0*x + 1.0*x**2 + 1.25*x**3 + -0.5*x**4 + -0.75*x**5
        psi_2 = lambda x : 0.0 + 0.0*x + -0.125*x**2 + -0.125*x**3 + 0.125*x**4 + 0.125*x**5
        phi_3 = lambda x : 1.0 + 0.0*x + -2.0*x**2 + 0.0*x**3 + 1.0*x**4 + 0.0*x**5
        psi_3 = lambda x : 0.0 + 0.5*x + 0.0*x**2 + -1.0*x**3 + 0.0*x**4 + 0.5*x**5

        Nvtild = np.array([phi_1, psi_1, phi_2, psi_2, phi_3, psi_3]).reshape(-1,1)

        return Nvtild

    def dNvtild(self) -> np.ndarray:

        phi_1_x = lambda x : 0.0 + 2.0*x + -3.75*x**2 + -2.0*x**3 + 3.75*x**4
        psi_1_x = lambda x : 0.0 + 0.25*x + -0.375*x**2 + -0.5*x**3 + 0.625*x**4
        phi_2_x = lambda x : 0.0 + 2.0*x + 3.75*x**2 + -2.0*x**3 + -3.75*x**4
        psi_2_x = lambda x : 0.0 + -0.25*x + -0.375*x**2 + 0.5*x**3 + 0.625*x**4
        phi_3_x = lambda x : 0.0 + -4.0*x + 0.0*x**2 + 4.0*x**3 + 0.0*x**4
        psi_3_x = lambda x : 0.5 + 0.0*x + -3.0*x**2 + 0.0*x**3 + 2.5*x**4

        dNvtild = np.array([phi_1_x, psi_1_x, phi_2_x, psi_2_x, phi_3_x, psi_3_x]).reshape(-1,1)

        return dNvtild

    def _ddNvtild(self) -> np.ndarray:
        
        phi_1_xx = lambda x : 2.0 + -7.5*x + -6.0*x**2 + 15.0*x**3
        psi_1_xx = lambda x : 0.25 + -0.75*x + -1.5*x**2 + 2.5*x**3
        phi_2_xx = lambda x : 2.0 + 7.5*x + -6.0*x**2 + -15.0*x**3
        psi_2_xx = lambda x : -0.25 + -0.75*x + 1.5*x**2 + 2.5*x**3
        phi_3_xx = lambda x : -4.0 + 0.0*x + 12.0*x**2 + 0.0*x**3
        psi_3_xx = lambda x : 0.0 + -6.0*x + 0.0*x**2 + 10.0*x**3

        ddNvtild = np.array([phi_1_xx, psi_1_xx, phi_2_xx, psi_2_xx, phi_3_xx, psi_3_xx]).reshape(-1,1)

        return ddNvtild

class SEG4(GroupElem):
    #        v
    #        ^
    #        |
    #        |
    #  0---2-+-3---1 --> u

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def indexesTriangles(self) -> list[int]:
        return super().indexesTriangles

    @property
    def indexesFaces(self) -> list[int]:
        return [0,2,3,1]

    def _Ntild(self) -> np.ndarray:

        N1t = lambda x : -0.5625*x**3 + 0.5625*x**2 + 0.0625*x - 0.0625
        N2t = lambda x : 0.5625*x**3 + 0.5625*x**2 - 0.0625*x - 0.0625
        N3t = lambda x : 1.6875*x**3 - 0.5625*x**2 - 1.6875*x + 0.5625
        N4t = lambda x : -1.6875*x**3 - 0.5625*x**2 + 1.6875*x + 0.5625

        Ntild = np.array([N1t, N2t, N3t, N4t]).reshape(-1, 1)

        return Ntild

    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda x : -1.6875*x**2 + 1.125*x + 0.0625]
        dN2t = [lambda x : 1.6875*x**2 + 1.125*x - 0.0625]
        dN3t = [lambda x : 5.0625*x**2 - 1.125*x - 1.6875]
        dN4t = [lambda x : -5.0625*x**2 - 1.125*x + 1.6875]

        dNtild = np.array([dN1t, dN2t, dN3t, dN4t])

        return dNtild
    
    def _ddNtild(self) -> np.ndarray:

        ddN1t = [lambda x : -3.375*x + 1.125]
        ddN2t = [lambda x : 3.375*x + 1.125]
        ddN3t = [lambda x : 10.125*x - 1.125]
        ddN4t = [lambda x : -10.125*x - 1.125]

        ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t])

        return ddNtild

    def _dddNtild(self) -> np.ndarray:
        
        dddN1t = [lambda x : -3.375]
        dddN2t = [lambda x : 3.375]
        dddN3t = [lambda x : 10.125]
        dddN4t = [lambda x : -10.125]

        dddNtild = np.array([dddN1t, dddN2t, dddN3t, dddN4t])

        return dddNtild

    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()

    def _Nvtild(self) -> np.ndarray:

        phi_1 = lambda x : 0.025390624999999556 + -0.029296874999997335*x + -0.4746093750000018*x**2 + 0.548828124999992*x**3 + 2.3730468750000036*x**4 + -2.7597656249999916*x**5 + -1.4238281250000018*x**6 + 1.740234374999997*x**7
        psi_1 = lambda x : 0.0019531250000000555 + -0.0019531249999997224*x + -0.03710937500000017*x**2 + 0.03710937499999917*x**3 + 0.19335937500000028*x**4 + -0.19335937499999917*x**5 + -0.15820312500000014*x**6 + 0.15820312499999972*x**7
        phi_2 = lambda x : 0.025390625 + 0.02929687499999778*x + -0.47460937499999734*x**2 + -0.5488281249999911*x**3 + 2.373046874999995*x**4 + 2.75976562499999*x**5 + -1.4238281249999976*x**6 + -1.7402343749999962*x**7
        psi_2 = lambda x : -0.001953125 + -0.0019531249999998335*x + 0.03710937499999983*x**2 + 0.03710937499999928*x**3 + -0.19335937499999967*x**4 + -0.19335937499999908*x**5 + 0.15820312499999983*x**6 + 0.15820312499999964*x**7
        phi_3 = lambda x : 0.474609375 + -2.373046874999991*x + 0.4746093749999929*x**2 + 9.017578124999972*x**3 + -2.3730468749999845*x**4 + -10.91601562499997*x**5 + 1.4238281249999922*x**6 + 4.271484374999989*x**7
        psi_3 = lambda x : 0.05273437499999978 + -0.1582031249999971*x + -0.5800781250000018*x**2 + 1.7402343749999911*x**3 + 1.001953125000004*x**4 + -3.0058593749999907*x**5 + -0.4746093750000019*x**6 + 1.4238281249999967*x**7
        phi_4 = lambda x : 0.4746093749999991 + 2.3730468749999902*x + 0.4746093750000089*x**2 + -9.017578124999972*x**3 + -2.373046875000015*x**4 + 10.916015624999972*x**5 + 1.423828125000007*x**6 + -4.27148437499999*x**7
        psi_4 = lambda x : -0.05273437500000022 + -0.15820312499999734*x + 0.5800781249999978*x**2 + 1.7402343749999911*x**3 + -1.0019531249999953*x**4 + -3.0058593749999902*x**5 + 0.47460937499999767*x**6 + 1.4238281249999964*x**7

        Nvtild = np.array([phi_1, psi_1, phi_2, psi_2, phi_3, psi_3, phi_4, psi_4]).reshape(-1,1)

        return Nvtild
        
    def dNvtild(self) -> np.ndarray:

        phi_1_x = lambda x : -0.029296874999997335 + -0.9492187500000036*x + 1.646484374999976*x**2 + 9.492187500000014*x**3 + -13.798828124999957*x**4 + -8.54296875000001*x**5 + 12.181640624999979*x**6
        psi_1_x = lambda x : -0.0019531249999997224 + -0.07421875000000033*x + 0.1113281249999975*x**2 + 0.7734375000000011*x**3 + -0.9667968749999958*x**4 + -0.9492187500000009*x**5 + 1.107421874999998*x**6
        phi_2_x = lambda x : 0.02929687499999778 + -0.9492187499999947*x + -1.6464843749999734*x**2 + 9.49218749999998*x**3 + 13.798828124999948*x**4 + -8.542968749999986*x**5 + -12.181640624999973*x**6
        psi_2_x = lambda x : -0.0019531249999998335 + 0.07421874999999967*x + 0.11132812499999784*x**2 + -0.7734374999999987*x**3 + -0.9667968749999954*x**4 + 0.949218749999999*x**5 + 1.1074218749999976*x**6
        phi_3_x = lambda x : -2.373046874999991 + 0.9492187499999858*x + 27.052734374999915*x**2 + -9.492187499999938*x**3 + -54.58007812499985*x**4 + 8.542968749999954*x**5 + 29.900390624999925*x**6
        psi_3_x = lambda x : -0.1582031249999971 + -1.1601562500000036*x + 5.220703124999973*x**2 + 4.007812500000016*x**3 + -15.029296874999954*x**4 + -2.8476562500000115*x**5 + 9.966796874999977*x**6
        phi_4_x = lambda x : 2.3730468749999902 + 0.9492187500000178*x + -27.052734374999915*x**2 + -9.49218750000006*x**3 + 54.58007812499986*x**4 + 8.542968750000043*x**5 + -29.900390624999932*x**6
        psi_4_x = lambda x : -0.15820312499999734 + 1.1601562499999956*x + 5.220703124999973*x**2 + -4.007812499999981*x**3 + -15.02929687499995*x**4 + 2.847656249999986*x**5 + 9.966796874999975*x**6

        dNvtild = np.array([phi_1_x, psi_1_x, phi_2_x, psi_2_x, phi_3_x, psi_3_x, phi_4_x, psi_4_x]).reshape(-1,1)

        return dNvtild    

    def _ddNvtild(self) -> np.ndarray:
        
        phi_1_xx = lambda x : -0.9492187500000036 + 3.292968749999952*x + 28.476562500000043*x**2 + -55.19531249999983*x**3 + -42.71484375000006*x**4 + 73.08984374999987*x**5
        psi_1_xx = lambda x : -0.07421875000000033 + 0.222656249999995*x + 2.3203125000000036*x**2 + -3.867187499999983*x**3 + -4.746093750000004*x**4 + 6.6445312499999885*x**5
        phi_2_xx = lambda x : -0.9492187499999947 + -3.2929687499999467*x + 28.476562499999943*x**2 + 55.195312499999794*x**3 + -42.71484374999993*x**4 + -73.08984374999984*x**5
        psi_2_xx = lambda x : 0.07421874999999967 + 0.22265624999999567*x + -2.320312499999996*x**2 + -3.867187499999982*x**3 + 4.746093749999995*x**4 + 6.644531249999985*x**5
        phi_3_xx = lambda x : 0.9492187499999858 + 54.10546874999983*x + -28.476562499999815*x**2 + -218.3203124999994*x**3 + 42.714843749999766*x**4 + 179.40234374999955*x**5
        psi_3_xx = lambda x : -1.1601562500000036 + 10.441406249999947*x + 12.023437500000048*x**2 + -60.117187499999815*x**3 + -14.238281250000057*x**4 + 59.80078124999986*x**5
        phi_4_xx = lambda x : 0.9492187500000178 + -54.10546874999983*x + -28.47656250000018*x**2 + 218.32031249999943*x**3 + 42.71484375000021*x**4 + -179.4023437499996*x**5
        psi_4_xx = lambda x : 1.1601562499999956 + 10.441406249999947*x + -12.023437499999943*x**2 + -60.1171874999998*x**3 + 14.23828124999993*x**4 + 59.80078124999985*x**5

        ddNvtild = np.array([phi_1_xx, psi_1_xx, phi_2_xx, psi_2_xx, phi_3_xx, psi_3_xx, phi_4_xx, psi_4_xx]).reshape(-1,1)

        return ddNvtild

class SEG5(GroupElem):
    #          v
    #          ^
    #          |
    #          |
    #  0---2---3---4---1 --> u

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def indexesTriangles(self) -> list[int]:
        return super().indexesTriangles

    @property
    def indexesFaces(self) -> list[int]:
        return [0,2,3,4,1]

    def _Ntild(self) -> np.ndarray:

        N1t = lambda x : 0.6667*x**4 - 0.6667*x**3 - 0.1667*x**2 + 0.1667*x + 0.0
        N2t = lambda x : 0.6667*x**4 + 0.6667*x**3 - 0.1667*x**2 - 0.1667*x + 0.0
        N3t = lambda x : -2.667*x**4 + 1.333*x**3 + 2.667*x**2 - 1.333*x + 0.0
        N4t = lambda x : 4.0*x**4 + 0.0*x**3 - 5.0*x**2 + 0.0*x + 1.0
        N5t = lambda x : -2.667*x**4 - 1.333*x**3 + 2.667*x**2 + 1.333*x + 0.0

        Ntild = np.array([N1t, N2t, N3t, N4t, N5t]).reshape(-1, 1)

        return Ntild
    
    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda x : 2.667*x**3 - 2.0*x**2 - 0.3333*x + 0.1667]
        dN2t = [lambda x : 2.667*x**3 + 2.0*x**2 - 0.3333*x - 0.1667]
        dN3t = [lambda x : -10.67*x**3 + 4.0*x**2 + 5.333*x - 1.333]
        dN4t = [lambda x : 16.0*x**3 + 0.0*x**2 - 10.0*x + 0.0]
        dN5t = [lambda x : -10.67*x**3 - 4.0*x**2 + 5.333*x + 1.333]        

        dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t])

        return dNtild    
    
    def _ddNtild(self) -> np.ndarray:

        ddN1t = [lambda x : 8.0*x**2 - 4.0*x - 0.3333]
        ddN2t = [lambda x : 8.0*x**2 + 4.0*x - 0.3333]
        ddN3t = [lambda x : -32.0*x**2 + 8.0*x + 5.333]
        ddN4t = [lambda x : 48.0*x**2 + 0.0*x - 10.0]
        ddN5t = [lambda x : -32.0*x**2 - 8.0*x + 5.333]

        ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t, ddN5t])

        return ddNtild    

    def _dddNtild(self) -> np.ndarray:

        dddN1t = [lambda x : 16.0*x - 4.0]
        dddN2t = [lambda x : 16.0*x + 4.0]
        dddN3t = [lambda x : -64.0*x + 8.0]
        dddN4t = [lambda x : 96.0*x + 0.0]
        dddN5t = [lambda x : -64.0*x - 8.0]

        dddNtild = np.array([dddN1t, dddN2t, dddN3t, dddN4t, dddN5t])

        return dddNtild

    def _ddddNtild(self) -> np.ndarray:
        
        ddddN1t = [lambda x : 16.0]
        ddddN2t = [lambda x : 16.0]
        ddddN3t = [lambda x : -64.0]
        ddddN4t = [lambda x : 96.0]
        ddddN5t = [lambda x : -64.0]
        
        ddddNtild = np.array([ddddN1t, ddddN2t, ddddN3t, ddddN4t, ddddN5t])

        return ddddNtild

    def _Nvtild(self) -> np.ndarray:
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

        Nvtild = np.array([phi_1, psi_1, phi_2, psi_2, phi_3, psi_3, phi_4, psi_4, phi_5, psi_5]).reshape(-1,1)

        return Nvtild

    def dNvtild(self) -> np.ndarray:

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

        dNvtild = np.array([phi_1_x, psi_1_x, phi_2_x, psi_2_x, phi_3_x, psi_3_x, phi_4_x, psi_4_x, phi_5_x, psi_5_x]).reshape(-1,1)

        return dNvtild    

    def _ddNvtild(self) -> np.ndarray:
        
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

        ddNvtild = np.array([phi_1_xx, psi_1_xx, phi_2_xx, psi_2_xx, phi_3_xx, psi_3_xx, phi_4_xx, psi_4_xx, phi_5_xx, psi_5_xx]).reshape(-1,1)

        return ddNvtild

class TRI3(GroupElem):
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

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def indexesTriangles(self) -> list[int]:
        return [0,1,2]

    @property
    def indexesFaces(self) -> list[int]:
        return [0,1,2,0]
    _indexesFaces = [0,1,2,0]

    def _Ntild(self) -> np.ndarray:

        N1t = lambda ksi,eta: 1-ksi-eta
        N2t = lambda ksi,eta: ksi
        N3t = lambda ksi,eta: eta
        
        Ntild = np.array([N1t, N2t, N3t]).reshape(-1,1)

        return Ntild

    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda ksi,eta: -1, lambda ksi,eta: -1]
        dN2t = [lambda ksi,eta: 1,  lambda ksi,eta: 0]
        dN3t = [lambda ksi,eta: 0,  lambda ksi,eta: 1]

        dNtild = np.array([dN1t, dN2t, dN3t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:
        return super()._ddNtild()

    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()

    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()

    def _Nvtild(self) -> np.ndarray:
        pass

    def dNvtild(self) -> np.ndarray:
        pass

    def _ddNvtild(self) -> np.ndarray:
        pass

class TRI6(GroupElem):
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

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def indexesTriangles(self) -> list[int]:
        return [0,3,5,3,1,4,5,4,2,3,4,5]

    @property
    def indexesFaces(self) -> list[int]:
        return [0,3,1,4,2,5,0]

    def _Ntild(self) -> np.ndarray:

        N1t = lambda ksi,eta: -(1-ksi-eta)*(1-2*(1-ksi-eta))
        N2t = lambda ksi,eta: -ksi*(1-2*ksi)
        N3t = lambda ksi,eta: -eta*(1-2*eta)
        N4t = lambda ksi,eta: 4*ksi*(1-ksi-eta)
        N5t = lambda ksi,eta: 4*ksi*eta
        N6t = lambda ksi,eta: 4*eta*(1-ksi-eta)
        
        Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t]).reshape(-1, 1)

        return Ntild

    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda ksi,eta: 4*ksi+4*eta-3,  lambda ksi,eta: 4*ksi+4*eta-3]
        dN2t = [lambda ksi,eta: 4*ksi-1,        lambda ksi,eta: 0]
        dN3t = [lambda ksi,eta: 0,              lambda ksi,eta: 4*eta-1]
        dN4t = [lambda ksi,eta: 4-8*ksi-4*eta,  lambda ksi,eta: -4*ksi]
        dN5t = [lambda ksi,eta: 4*eta,          lambda ksi,eta: 4*ksi]
        dN6t = [lambda ksi,eta: -4*eta,         lambda ksi,eta: 4-4*ksi-8*eta]
        
        dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:
        ddN1t = [lambda ksi,eta: 4,  lambda ksi,eta: 4]
        ddN2t = [lambda ksi,eta: 4,  lambda ksi,eta: 0]
        ddN3t = [lambda ksi,eta: 0,  lambda ksi,eta: 4]
        ddN4t = [lambda ksi,eta: -8, lambda ksi,eta: 0]
        ddN5t = [lambda ksi,eta: 0,  lambda ksi,eta: 0]
        ddN6t = [lambda ksi,eta: 0,  lambda ksi,eta: -8]
        
        ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t, ddN5t, ddN6t])

        return ddNtild

    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()

    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()

    def _Nvtild(self) -> np.ndarray:
        return super()._Nvtild()

    def dNvtild(self) -> np.ndarray:
        return super().dNvtild()

    def _ddNvtild(self) -> np.ndarray:
        return super()._ddNvtild()

class TRI10(GroupElem):
    # v
    # ^
    # |
    # 2
    # | \
    # 7   6
    # |     \
    # 8  (9)  5
    # |         \
    # 0---3---4---1 --> u

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def indexesTriangles(self) -> list[int]:
        return list(np.array([10,1,4,10,4,5,10,5,6,10,6,7,10,7,8,10,8,9,10,9,1,2,5,6,3,7,8])-1)
    
    @property
    def indexesFaces(self) -> list[int]:
        return [0,3,4,1,5,6,2,7,8,0]

    def _Ntild(self) -> np.ndarray:

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
        
        Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t, N9t, N10t]).reshape(-1, 1)

        return Ntild

    def _dNtild(self) -> np.ndarray:

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

        return dNtild

    def _ddNtild(self) -> np.ndarray:

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

        return ddNtild

    def _dddNtild(self) -> np.ndarray:
        
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

        return dddNtild

    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()

    def _Nvtild(self) -> np.ndarray:
        pass

    def dNvtild(self) -> np.ndarray:
        pass

    def _ddNvtild(self) -> np.ndarray:
        pass

class TRI15(GroupElem):
    # 
    # 2
    # | \
    # 9   8
    # |     \
    # 10 (14)  7
    # |         \
    # 11 (12) (13) 6
    # |             \
    # 0---3---4---5---1 --> u

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def indexesTriangles(self) -> list[int]:
        return list(np.array([1,4,13,4,5,14,5,6,14,6,7,14,2,6,7,4,13,14,1,12,13,11,12,13,11,13,15,13,14,15,8,14,15,7,8,14,10,11,15,8,9,15,9,10,15,3,9,10])-1)

    @property
    def indexesFaces(self) -> list[int]:
        return [0,3,4,5,1,6,7,8,2,9,10,11,0]

    def _Ntild(self) -> np.ndarray:

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

        Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t, N9t, N10t, N11t, N12t, N13t, N14t, N15t]).reshape(-1, 1)

        return Ntild

    def _dNtild(self) -> np.ndarray:

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

        return dNtild

    def _ddNtild(self) -> np.ndarray:

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

        return ddNtild

    def _dddNtild(self) -> np.ndarray:

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

        return dddNtild

    def _ddddNtild(self) -> np.ndarray:
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

        return ddddNtild

class QUAD4(GroupElem):
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

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def indexesTriangles(self) -> list[int]:
        return [0,1,3,1,2,3]

    @property
    def indexesFaces(self) -> list[int]:
        return [0,1,2,3,0]
    _indexesFaces = [0,1,2,3,0]

    def _Ntild(self) -> np.ndarray:

        N1t = lambda ksi,eta: (1-ksi)*(1-eta)/4
        N2t = lambda ksi,eta: (1+ksi)*(1-eta)/4
        N3t = lambda ksi,eta: (1+ksi)*(1+eta)/4
        N4t = lambda ksi,eta: (1-ksi)*(1+eta)/4
        
        Ntild = np.array([N1t, N2t, N3t, N4t]).reshape(-1, 1)

        return Ntild

    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda ksi,eta: (eta-1)/4,  lambda ksi,eta: (ksi-1)/4]
        dN2t = [lambda ksi,eta: (1-eta)/4,  lambda ksi,eta: (-ksi-1)/4]
        dN3t = [lambda ksi,eta: (1+eta)/4,  lambda ksi,eta: (1+ksi)/4]
        dN4t = [lambda ksi,eta: (-eta-1)/4, lambda ksi,eta: (1-ksi)/4]
        
        dNtild = np.array([dN1t, dN2t, dN3t, dN4t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:
        return super()._ddNtild()
    
    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()
    
    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()

    def _Nvtild(self) -> np.ndarray:
        pass

    def dNvtild(self) -> np.ndarray:
        pass

    def _ddNvtild(self) -> np.ndarray:
        pass

class QUAD8(GroupElem):
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

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def indexesTriangles(self) -> list[int]:
        return [4,5,7,5,6,7,0,4,7,4,1,5,5,2,6,6,3,7]

    @property
    def indexesFaces(self) -> list[int]:
        return [0,4,1,5,2,6,3,7,0]

    def _Ntild(self) -> np.ndarray:

        N1t = lambda ksi,eta: (1-ksi)*(1-eta)*(-1-ksi-eta)/4
        N2t = lambda ksi,eta: (1+ksi)*(1-eta)*(-1+ksi-eta)/4
        N3t = lambda ksi,eta: (1+ksi)*(1+eta)*(-1+ksi+eta)/4
        N4t = lambda ksi,eta: (1-ksi)*(1+eta)*(-1-ksi+eta)/4
        N5t = lambda ksi,eta: (1-ksi**2)*(1-eta)/2
        N6t = lambda ksi,eta: (1+ksi)*(1-eta**2)/2
        N7t = lambda ksi,eta: (1-ksi**2)*(1+eta)/2
        N8t = lambda ksi,eta: (1-ksi)*(1-eta**2)/2
        
        Ntild =  np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t]).reshape(-1, 1)

        return Ntild
    
    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda ksi,eta: (1-eta)*(2*ksi+eta)/4,      lambda ksi,eta: (1-ksi)*(ksi+2*eta)/4]
        dN2t = [lambda ksi,eta: (1-eta)*(2*ksi-eta)/4,      lambda ksi,eta: -(1+ksi)*(ksi-2*eta)/4]
        dN3t = [lambda ksi,eta: (1+eta)*(2*ksi+eta)/4,      lambda ksi,eta: (1+ksi)*(ksi+2*eta)/4]
        dN4t = [lambda ksi,eta: -(1+eta)*(-2*ksi+eta)/4,    lambda ksi,eta: (1-ksi)*(-ksi+2*eta)/4]
        dN5t = [lambda ksi,eta: -ksi*(1-eta),               lambda ksi,eta: -(1-ksi**2)/2]
        dN6t = [lambda ksi,eta: (1-eta**2)/2,               lambda ksi,eta: -eta*(1+ksi)]
        dN7t = [lambda ksi,eta: -ksi*(1+eta),               lambda ksi,eta: (1-ksi**2)/2]
        dN8t = [lambda ksi,eta: -(1-eta**2)/2,              lambda ksi,eta: -eta*(1-ksi)]
                        
        dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:

        ddN1t = [lambda ksi,eta: (1-eta)/2,  lambda ksi,eta: (1-ksi)/2]
        ddN2t = [lambda ksi,eta: (1-eta)/2,  lambda ksi,eta: (1+ksi)/2]
        ddN3t = [lambda ksi,eta: (1+eta)/2,  lambda ksi,eta: (1+ksi)/2]
        ddN4t = [lambda ksi,eta: (1+eta)/2,  lambda ksi,eta: (1-ksi)/2]
        ddN5t = [lambda ksi,eta: -1+eta,     lambda ksi,eta: 0]
        ddN6t = [lambda ksi,eta: 0,          lambda ksi,eta: -1-ksi]
        ddN7t = [lambda ksi,eta: -1-eta,     lambda ksi,eta: 0]
        ddN8t = [lambda ksi,eta: 0,          lambda ksi,eta: -1+ksi]
                        
        ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t, ddN5t, ddN6t, ddN7t, ddN8t])

        return ddNtild

    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()

    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()

    def _Nvtild(self) -> np.ndarray:
        return super()._Nvtild()

    def dNvtild(self) -> np.ndarray:
        return super().dNvtild()

    def _ddNvtild(self) -> np.ndarray:
        return super()._ddNvtild()

class TETRA4(GroupElem):
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

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def indexesTriangles(self) -> list[int]:
        return super().indexesTriangles

    @property
    def indexesFaces(self) -> list[int]:
        return [0,1,2,0,3,1,0,2,3,1,3,2]
    
    @property
    def indexesSegments(self) -> np.ndarray:
        return np.array([[0,1],[0,3],[3,1],[2,0],[2,3],[2,1]])

    def _Ntild(self) -> np.ndarray:

        N1t = lambda x,y,z: 1-x-y-z
        N2t = lambda x,y,z: x
        N3t = lambda x,y,z: y
        N4t = lambda x,y,z: z

        Ntild = np.array([N1t, N2t, N3t, N4t]).reshape(-1, 1)

        return Ntild
    
    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda x,y,z: -1,   lambda x,y,z: -1,   lambda x,y,z: -1]
        dN2t = [lambda x,y,z: 1,    lambda x,y,z: 0,    lambda x,y,z: 0]
        dN3t = [lambda x,y,z: 0,    lambda x,y,z: 1,    lambda x,y,z: 0]
        dN4t = [lambda x,y,z: 0,    lambda x,y,z: 0,    lambda x,y,z: 1]

        dNtild = np.array([dN1t, dN2t, dN3t, dN4t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:
        return super()._ddNtild()

    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()
    
    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()

    def _Nvtild(self) -> np.ndarray:
        pass

    def dNvtild(self) -> np.ndarray:
        pass

    def _ddNvtild(self) -> np.ndarray:
        pass

class TETRA10(GroupElem):
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

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def indexesTriangles(self) -> list[int]:
        return super().indexesTriangles

    @property
    def indexesFaces(self) -> list[int]:        
        return [0,4,1,5,2,6,0,7,3,9,1,4,0,6,2,8,3,7,1,9,3,8,2,5]
    
    @property
    def indexesSegments(self) -> np.ndarray:
        return np.array([[0,1],[0,3],[3,1],[2,0],[2,3],[2,1]])

    def _Ntild(self) -> np.ndarray:

        N1t = lambda x,y,z: 2.0*x**2 + 2.0*y**2 + 2.0*z**2 + 4.0*x*y + 4.0*x*z + 4.0*y*z + -3.0*x + -3.0*y + -3.0*z + 1.0
        N2t = lambda x,y,z: 2.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + -1.0*x + 0.0*y + 0.0*z + 0.0
        N3t = lambda x,y,z: 0.0*x**2 + 2.0*y**2 + 0.0*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + 0.0*x + -1.0*y + 0.0*z + 0.0
        N4t = lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 2.0*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + 0.0*x + 0.0*y + -1.0*z + 0.0
        N5t = lambda x,y,z: -4.0*x**2 + 0.0*y**2 + 0.0*z**2 + -4.0*x*y + -4.0*x*z + 0.0*y*z + 4.0*x + 0.0*y + 0.0*z + 0.0
        N6t = lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 4.0*x*y + 0.0*x*z + 0.0*y*z + 0.0*x + 0.0*y + 0.0*z + 0.0
        N7t = lambda x,y,z: 0.0*x**2 + -4.0*y**2 + 0.0*z**2 + -4.0*x*y + 0.0*x*z + -4.0*y*z + 0.0*x + 4.0*y + 0.0*z + 0.0
        N8t = lambda x,y,z: 0.0*x**2 + 0.0*y**2 + -4.0*z**2 + 0.0*x*y + -4.0*x*z + -4.0*y*z + 0.0*x + 0.0*y + 4.0*z + 0.0
        N9t = lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*y + 0.0*x*z + 4.0*y*z + 0.0*x + 0.0*y + 0.0*z + 0.0
        N10t = lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*y + 4.0*x*z + 0.0*y*z + 0.0*x + 0.0*y + 0.0*z + 0.0

        Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t, N9t, N10t]).reshape(-1, 1)

        return Ntild
    
    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda x,y,z: 4.0*x + 4.0*y + 4.0*z + -3.0,   lambda x,y,z: 4.0*y + 4.0*x + 4.0*z + -3.0,   lambda x,y,z: 4.0*z + 4.0*x + 4.0*y + -3.0]
        dN2t = [lambda x,y,z: 4.0*x + 0.0*y + 0.0*z + -1.0,   lambda x,y,z: 0.0*y + 0.0*x + 0.0*z + 0.0,   lambda x,y,z: 0.0*z + 0.0*x + 0.0*y + 0.0]
        dN3t = [lambda x,y,z: 0.0*x + 0.0*y + 0.0*z + 0.0,   lambda x,y,z: 4.0*y + 0.0*x + 0.0*z + -1.0,   lambda x,y,z: 0.0*z + 0.0*x + 0.0*y + 0.0]
        dN4t = [lambda x,y,z: 0.0*x + 0.0*y + 0.0*z + 0.0,   lambda x,y,z: 0.0*y + 0.0*x + 0.0*z + 0.0,   lambda x,y,z: 4.0*z + 0.0*x + 0.0*y + -1.0]
        dN5t = [lambda x,y,z: -8.0*x + -4.0*y + -4.0*z + 4.0,   lambda x,y,z: 0.0*y + -4.0*x + 0.0*z + 0.0,   lambda x,y,z: 0.0*z + -4.0*x + 0.0*y + 0.0]
        dN6t = [lambda x,y,z: 0.0*x + 4.0*y + 0.0*z + 0.0,   lambda x,y,z: 0.0*y + 4.0*x + 0.0*z + 0.0,   lambda x,y,z: 0.0*z + 0.0*x + 0.0*y + 0.0]
        dN7t = [lambda x,y,z: 0.0*x + -4.0*y + 0.0*z + 0.0,   lambda x,y,z: -8.0*y + -4.0*x + -4.0*z + 4.0,   lambda x,y,z: 0.0*z + 0.0*x + -4.0*y + 0.0]
        dN8t = [lambda x,y,z: 0.0*x + 0.0*y + -4.0*z + 0.0,   lambda x,y,z: 0.0*y + 0.0*x + -4.0*z + 0.0,   lambda x,y,z: -8.0*z + -4.0*x + -4.0*y + 4.0]
        dN9t = [lambda x,y,z: 0.0*x + 0.0*y + 0.0*z + 0.0,   lambda x,y,z: 0.0*y + 0.0*x + 4.0*z + 0.0,   lambda x,y,z: 0.0*z + 0.0*x + 4.0*y + 0.0]
        dN10t = [lambda x,y,z: 0.0*x + 0.0*y + 4.0*z + 0.0,   lambda x,y,z: 0.0*y + 0.0*x + 0.0*z + 0.0,   lambda x,y,z: 0.0*z + 4.0*x + 0.0*y + 0.0]


        dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t, dN9t, dN10t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:

        ddN1t = [lambda x,y,z: 4.0,   lambda x,y,z: 4.0,   lambda x,y,z: 4.0]
        ddN2t = [lambda x,y,z: 4.0,   lambda x,y,z: 0.0,   lambda x,y,z: 0.0]
        ddN3t = [lambda x,y,z: 0.0,   lambda x,y,z: 4.0,   lambda x,y,z: 0.0]
        ddN4t = [lambda x,y,z: 0.0,   lambda x,y,z: 0.0,   lambda x,y,z: 4.0]
        ddN5t = [lambda x,y,z: -8.0,   lambda x,y,z: 0.0,   lambda x,y,z: 0.0]
        ddN6t = [lambda x,y,z: 0.0,   lambda x,y,z: 0.0,   lambda x,y,z: 0.0]
        ddN7t = [lambda x,y,z: 0.0,   lambda x,y,z: -8.0,   lambda x,y,z: 0.0]
        ddN8t = [lambda x,y,z: 0.0,   lambda x,y,z: 0.0,   lambda x,y,z: -8.0]
        ddN9t = [lambda x,y,z: 0.0,   lambda x,y,z: 0.0,   lambda x,y,z: 0.0]
        ddN10t = [lambda x,y,z: 0.0,   lambda x,y,z: 0.0,   lambda x,y,z: 0.0]

        ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t, ddN5t, ddN6t, ddN7t, ddN8t, ddN9t, ddN10t])

        return ddNtild

    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()
    
    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()

    def _Nvtild(self) -> np.ndarray:
        pass

    def dNvtild(self) -> np.ndarray:
        pass

    def _ddNvtild(self) -> np.ndarray:
        pass

class HEXA8(GroupElem):
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

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def indexesTriangles(self) -> list[int]:
        return super().indexesTriangles

    @property
    def indexesFaces(self) -> list[int]:
        return [0,1,2,3,0,4,5,1,0,3,7,4,6,7,3,2,6,2,1,5,6,5,4,7]
    
    @property
    def indexesSegments(self) -> np.ndarray:
        return np.array([[0,1],[1,5],[5,4],[4,0],[3,2],[2,6],[6,7],[7,3],[0,3],[1,2],[5,6],[4,7]])

    def _Ntild(self) -> np.ndarray:

        N1t = lambda x,y,z: 1/8 * (1-x) * (1-y) * (1-z)
        N2t = lambda x,y,z: 1/8 * (1+x) * (1-y) * (1-z)
        N3t = lambda x,y,z: 1/8 * (1+x) * (1+y) * (1-z)
        N4t = lambda x,y,z: 1/8 * (1-x) * (1+y) * (1-z)
        N5t = lambda x,y,z: 1/8 * (1-x) * (1-y) * (1+z)
        N6t = lambda x,y,z: 1/8 * (1+x) * (1-y) * (1+z)
        N7t = lambda x,y,z: 1/8 * (1+x) * (1+y) * (1+z)
        N8t = lambda x,y,z: 1/8 * (1-x) * (1+y) * (1+z)

        Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t]).reshape(-1, 1)

        return Ntild
    
    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda x,y,z: -1/8 * (1-y) * (1-z),   lambda x,y,z: -1/8 * (1-x) * (1-z),   lambda x,y,z: -1/8 * (1-x) * (1-y)]
        dN2t = [lambda x,y,z: 1/8 * (1-y) * (1-z),    lambda x,y,z: -1/8 * (1+x) * (1-z),    lambda x,y,z: -1/8 * (1+x) * (1-y)]
        dN3t = [lambda x,y,z: 1/8 * (1+y) * (1-z),    lambda x,y,z: 1/8 * (1+x) * (1-z),    lambda x,y,z: -1/8 * (1+x) * (1+y)]
        dN4t = [lambda x,y,z: -1/8 * (1+y) * (1-z),    lambda x,y,z: 1/8 * (1-x) * (1-z),    lambda x,y,z: -1/8 * (1-x) * (1+y)]
        dN5t = [lambda x,y,z: -1/8 * (1-y) * (1+z),    lambda x,y,z: -1/8 * (1-x) * (1+z),    lambda x,y,z: 1/8 * (1-x) * (1-y)]
        dN6t = [lambda x,y,z: 1/8 * (1-y) * (1+z),    lambda x,y,z: -1/8 * (1+x) * (1+z),    lambda x,y,z: 1/8 * (1+x) * (1-y)]
        dN7t = [lambda x,y,z: 1/8 * (1+y) * (1+z),    lambda x,y,z: 1/8 * (1+x) * (1+z),    lambda x,y,z: 1/8 * (1+x) * (1+y)]
        dN8t = [lambda x,y,z: -1/8 * (1+y) * (1+z),    lambda x,y,z: 1/8 * (1-x) * (1+z),    lambda x,y,z: 1/8 * (1-x) * (1+y)]

        dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:
        return super()._ddNtild()

    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()
    
    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()

    def _Nvtild(self) -> np.ndarray:
        pass

    def dNvtild(self) -> np.ndarray:
        pass

    def _ddNvtild(self) -> np.ndarray:
        pass

class HEXA20(GroupElem):
    #        v
    # 3----13----2
    # |\     ^   |\
    # | 15   |   | 14
    # 9  \   |   11 \
    # |   7----19+---6
    # |   |  +-- |-- | -> u
    # 0---+-8-\--1   |
    #  \  17   \  \  18
    #  10 |     \  12|
    #    \|      w  \|
    #     4----16----5

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def indexesTriangles(self) -> list[int]:
        return super().indexesTriangles

    @property
    def indexesFaces(self) -> list[int]:
        return [0,8,1,11,2,13,3,9,
                0,10,4,16,5,12,1,8,
                0,9,3,15,7,17,4,10,
                6,19,7,15,3,13,2,14,
                6,14,2,11,1,12,5,18,
                6,18,5,16,4,17,7,19]
    
    @property
    def indexesSegments(self) -> np.ndarray:
        return np.array([[0,1],[1,5],[5,4],[4,0],[3,2],[2,6],[6,7],[7,3],[0,3],[1,2],[5,6],[4,7]])

    def _Ntild(self) -> np.ndarray:        

        N1t = lambda x,y,z: 0.125*x**2*y*z + 0.125*y**2*x*z + 0.125*z**2*x*y + -0.125*x**2*y + -0.125*y**2*x + -0.125*z**2*x + -0.125*x**2*z + -0.125*y**2*z + -0.125*z**2*y + 0.125*x**2 + 0.125*y**2 + 0.125*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + 0.125*x + 0.125*y + 0.125*z + -0.125*x*y*z + -0.25
        N2t = lambda x,y,z: 0.125*x**2*y*z + -0.125*y**2*x*z + -0.125*z**2*x*y + -0.125*x**2*y + 0.125*y**2*x + 0.125*z**2*x + -0.125*x**2*z + -0.125*y**2*z + -0.125*z**2*y + 0.125*x**2 + 0.125*y**2 + 0.125*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + -0.125*x + 0.125*y + 0.125*z + 0.125*x*y*z + -0.25
        N3t = lambda x,y,z: -0.125*x**2*y*z + -0.125*y**2*x*z + 0.125*z**2*x*y + 0.125*x**2*y + 0.125*y**2*x + 0.125*z**2*x + -0.125*x**2*z + -0.125*y**2*z + 0.125*z**2*y + 0.125*x**2 + 0.125*y**2 + 0.125*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + -0.125*x + -0.125*y + 0.125*z + -0.125*x*y*z + -0.25
        N4t = lambda x,y,z: -0.125*x**2*y*z + 0.125*y**2*x*z + -0.125*z**2*x*y + 0.125*x**2*y + -0.125*y**2*x + -0.125*z**2*x + -0.125*x**2*z + -0.125*y**2*z + 0.125*z**2*y + 0.125*x**2 + 0.125*y**2 + 0.125*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + 0.125*x + -0.125*y + 0.125*z + 0.125*x*y*z + -0.25
        N5t = lambda x,y,z: -0.125*x**2*y*z + -0.125*y**2*x*z + 0.125*z**2*x*y + -0.125*x**2*y + -0.125*y**2*x + -0.125*z**2*x + 0.125*x**2*z + 0.125*y**2*z + -0.125*z**2*y + 0.125*x**2 + 0.125*y**2 + 0.125*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + 0.125*x + 0.125*y + -0.125*z + 0.125*x*y*z + -0.25
        N6t = lambda x,y,z: -0.125*x**2*y*z + 0.125*y**2*x*z + -0.125*z**2*x*y + -0.125*x**2*y + 0.125*y**2*x + 0.125*z**2*x + 0.125*x**2*z + 0.125*y**2*z + -0.125*z**2*y + 0.125*x**2 + 0.125*y**2 + 0.125*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + -0.125*x + 0.125*y + -0.125*z + -0.125*x*y*z + -0.25
        N7t = lambda x,y,z: 0.125*x**2*y*z + 0.125*y**2*x*z + 0.125*z**2*x*y + 0.125*x**2*y + 0.125*y**2*x + 0.125*z**2*x + 0.125*x**2*z + 0.125*y**2*z + 0.125*z**2*y + 0.125*x**2 + 0.125*y**2 + 0.125*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + -0.125*x + -0.125*y + -0.125*z + 0.125*x*y*z + -0.25
        N8t = lambda x,y,z: 0.125*x**2*y*z + -0.125*y**2*x*z + -0.125*z**2*x*y + 0.125*x**2*y + -0.125*y**2*x + -0.125*z**2*x + 0.125*x**2*z + 0.125*y**2*z + 0.125*z**2*y + 0.125*x**2 + 0.125*y**2 + 0.125*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + 0.125*x + -0.125*y + -0.125*z + -0.125*x*y*z + -0.25
        N9t = lambda x,y,z: -0.25*x**2*y*z + 0.0*y**2*x*z + 0.0*z**2*x*y + 0.25*x**2*y + 0.0*y**2*x + 0.0*z**2*x + 0.25*x**2*z + 0.0*y**2*z + 0.0*z**2*y + -0.25*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*y + 0.0*x*z + 0.25*y*z + 0.0*x + -0.25*y + -0.25*z + 0.0*x*y*z + 0.25
        N10t = lambda x,y,z: 0.0*x**2*y*z + -0.25*y**2*x*z + 0.0*z**2*x*y + 0.0*x**2*y + 0.25*y**2*x + 0.0*z**2*x + 0.0*x**2*z + 0.25*y**2*z + 0.0*z**2*y + 0.0*x**2 + -0.25*y**2 + 0.0*z**2 + 0.0*x*y + 0.25*x*z + 0.0*y*z + -0.25*x + 0.0*y + -0.25*z + 0.0*x*y*z + 0.25
        N11t = lambda x,y,z: 0.0*x**2*y*z + 0.0*y**2*x*z + -0.25*z**2*x*y + 0.0*x**2*y + 0.0*y**2*x + 0.25*z**2*x + 0.0*x**2*z + 0.0*y**2*z + 0.25*z**2*y + 0.0*x**2 + 0.0*y**2 + -0.25*z**2 + 0.25*x*y + 0.0*x*z + 0.0*y*z + -0.25*x + -0.25*y + 0.0*z + 0.0*x*y*z + 0.25
        N12t = lambda x,y,z: 0.0*x**2*y*z + 0.25*y**2*x*z + 0.0*z**2*x*y + 0.0*x**2*y + -0.25*y**2*x + 0.0*z**2*x + 0.0*x**2*z + 0.25*y**2*z + 0.0*z**2*y + 0.0*x**2 + -0.25*y**2 + 0.0*z**2 + 0.0*x*y + -0.25*x*z + 0.0*y*z + 0.25*x + 0.0*y + -0.25*z + 0.0*x*y*z + 0.25
        N13t = lambda x,y,z: 0.0*x**2*y*z + 0.0*y**2*x*z + 0.25*z**2*x*y + 0.0*x**2*y + 0.0*y**2*x + -0.25*z**2*x + 0.0*x**2*z + 0.0*y**2*z + 0.25*z**2*y + 0.0*x**2 + 0.0*y**2 + -0.25*z**2 + -0.25*x*y + 0.0*x*z + 0.0*y*z + 0.25*x + -0.25*y + 0.0*z + 0.0*x*y*z + 0.25
        N14t = lambda x,y,z: 0.25*x**2*y*z + 0.0*y**2*x*z + 0.0*z**2*x*y + -0.25*x**2*y + 0.0*y**2*x + 0.0*z**2*x + 0.25*x**2*z + 0.0*y**2*z + 0.0*z**2*y + -0.25*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*y + 0.0*x*z + -0.25*y*z + 0.0*x + 0.25*y + -0.25*z + 0.0*x*y*z + 0.25
        N15t = lambda x,y,z: 0.0*x**2*y*z + 0.0*y**2*x*z + -0.25*z**2*x*y + 0.0*x**2*y + 0.0*y**2*x + -0.25*z**2*x + 0.0*x**2*z + 0.0*y**2*z + -0.25*z**2*y + 0.0*x**2 + 0.0*y**2 + -0.25*z**2 + 0.25*x*y + 0.0*x*z + 0.0*y*z + 0.25*x + 0.25*y + 0.0*z + 0.0*x*y*z + 0.25
        N16t = lambda x,y,z: 0.0*x**2*y*z + 0.0*y**2*x*z + 0.25*z**2*x*y + 0.0*x**2*y + 0.0*y**2*x + 0.25*z**2*x + 0.0*x**2*z + 0.0*y**2*z + -0.25*z**2*y + 0.0*x**2 + 0.0*y**2 + -0.25*z**2 + -0.25*x*y + 0.0*x*z + 0.0*y*z + -0.25*x + 0.25*y + 0.0*z + 0.0*x*y*z + 0.25
        N17t = lambda x,y,z: 0.25*x**2*y*z + 0.0*y**2*x*z + 0.0*z**2*x*y + 0.25*x**2*y + 0.0*y**2*x + 0.0*z**2*x + -0.25*x**2*z + 0.0*y**2*z + 0.0*z**2*y + -0.25*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*y + 0.0*x*z + -0.25*y*z + 0.0*x + -0.25*y + 0.25*z + 0.0*x*y*z + 0.25
        N18t = lambda x,y,z: 0.0*x**2*y*z + 0.25*y**2*x*z + 0.0*z**2*x*y + 0.0*x**2*y + 0.25*y**2*x + 0.0*z**2*x + 0.0*x**2*z + -0.25*y**2*z + 0.0*z**2*y + 0.0*x**2 + -0.25*y**2 + 0.0*z**2 + 0.0*x*y + -0.25*x*z + 0.0*y*z + -0.25*x + 0.0*y + 0.25*z + 0.0*x*y*z + 0.25
        N19t = lambda x,y,z: 0.0*x**2*y*z + -0.25*y**2*x*z + 0.0*z**2*x*y + 0.0*x**2*y + -0.25*y**2*x + 0.0*z**2*x + 0.0*x**2*z + -0.25*y**2*z + 0.0*z**2*y + 0.0*x**2 + -0.25*y**2 + 0.0*z**2 + 0.0*x*y + 0.25*x*z + 0.0*y*z + 0.25*x + 0.0*y + 0.25*z + 0.0*x*y*z + 0.25
        N20t = lambda x,y,z: -0.25*x**2*y*z + 0.0*y**2*x*z + 0.0*z**2*x*y + -0.25*x**2*y + 0.0*y**2*x + 0.0*z**2*x + -0.25*x**2*z + 0.0*y**2*z + 0.0*z**2*y + -0.25*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*y + 0.0*x*z + 0.25*y*z + 0.0*x + 0.25*y + 0.25*z + 0.0*x*y*z + 0.25

        Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t, N9t, N10t, N11t, N12t, N13t, N14t, N15t, N16t, N17t, N18t, N19t, N20t]).reshape(-1, 1)

        return Ntild
    
    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda x,y,z: 0.25*x*y*z + 0.125*y**2*z + 0.125*z**2*y + -0.25*x*y + -0.125*y**2 + -0.125*z**2 + -0.25*x*z + 0.25*x + 0.0*y + 0.0*z + 0.125 + -0.125*y*z,
            lambda x,y,z: 0.125*x**2*z + 0.25*y*x*z + 0.125*z**2*x + -0.125*x**2 + -0.25*y*x + -0.25*y*z + -0.125*z**2 + 0.25*y + 0.0*x + 0.0*z + 0.125 + -0.125*x*z,
            lambda x,y,z: 0.125*x**2*y + 0.125*y**2*x + 0.25*z*x*y + -0.25*z*x + -0.125*x**2 + -0.125*y**2 + -0.25*z*y + 0.25*z + 0.0*x + 0.0*y + 0.125 + -0.125*x*y]
        dN2t = [lambda x,y,z: 0.25*x*y*z + -0.125*y**2*z + -0.125*z**2*y + -0.25*x*y + 0.125*y**2 + 0.125*z**2 + -0.25*x*z + 0.25*x + 0.0*y + 0.0*z + -0.125 + 0.125*y*z,
            lambda x,y,z: 0.125*x**2*z + -0.25*y*x*z + -0.125*z**2*x + -0.125*x**2 + 0.25*y*x + -0.25*y*z + -0.125*z**2 + 0.25*y + 0.0*x + 0.0*z + 0.125 + 0.125*x*z,
            lambda x,y,z: 0.125*x**2*y + -0.125*y**2*x + -0.25*z*x*y + 0.25*z*x + -0.125*x**2 + -0.125*y**2 + -0.25*z*y + 0.25*z + 0.0*x + 0.0*y + 0.125 + 0.125*x*y]
        dN3t = [lambda x,y,z: -0.25*x*y*z + -0.125*y**2*z + 0.125*z**2*y + 0.25*x*y + 0.125*y**2 + 0.125*z**2 + -0.25*x*z + 0.25*x + 0.0*y + 0.0*z + -0.125 + -0.125*y*z,
            lambda x,y,z: -0.125*x**2*z + -0.25*y*x*z + 0.125*z**2*x + 0.125*x**2 + 0.25*y*x + -0.25*y*z + 0.125*z**2 + 0.25*y + 0.0*x + 0.0*z + -0.125 + -0.125*x*z,
            lambda x,y,z: -0.125*x**2*y + -0.125*y**2*x + 0.25*z*x*y + 0.25*z*x + -0.125*x**2 + -0.125*y**2 + 0.25*z*y + 0.25*z + 0.0*x + 0.0*y + 0.125 + -0.125*x*y]
        dN4t = [lambda x,y,z: -0.25*x*y*z + 0.125*y**2*z + -0.125*z**2*y + 0.25*x*y + -0.125*y**2 + -0.125*z**2 + -0.25*x*z + 0.25*x + 0.0*y + 0.0*z + 0.125 + 0.125*y*z,
            lambda x,y,z: -0.125*x**2*z + 0.25*y*x*z + -0.125*z**2*x + 0.125*x**2 + -0.25*y*x + -0.25*y*z + 0.125*z**2 + 0.25*y + 0.0*x + 0.0*z + -0.125 + 0.125*x*z,
            lambda x,y,z: -0.125*x**2*y + 0.125*y**2*x + -0.25*z*x*y + -0.25*z*x + -0.125*x**2 + -0.125*y**2 + 0.25*z*y + 0.25*z + 0.0*x + 0.0*y + 0.125 + 0.125*x*y]
        dN5t = [lambda x,y,z: -0.25*x*y*z + -0.125*y**2*z + 0.125*z**2*y + -0.25*x*y + -0.125*y**2 + -0.125*z**2 + 0.25*x*z + 0.25*x + 0.0*y + 0.0*z + 0.125 + 0.125*y*z,
            lambda x,y,z: -0.125*x**2*z + -0.25*y*x*z + 0.125*z**2*x + -0.125*x**2 + -0.25*y*x + 0.25*y*z + -0.125*z**2 + 0.25*y + 0.0*x + 0.0*z + 0.125 + 0.125*x*z,
            lambda x,y,z: -0.125*x**2*y + -0.125*y**2*x + 0.25*z*x*y + -0.25*z*x + 0.125*x**2 + 0.125*y**2 + -0.25*z*y + 0.25*z + 0.0*x + 0.0*y + -0.125 + 0.125*x*y]
        dN6t = [lambda x,y,z: -0.25*x*y*z + 0.125*y**2*z + -0.125*z**2*y + -0.25*x*y + 0.125*y**2 + 0.125*z**2 + 0.25*x*z + 0.25*x + 0.0*y + 0.0*z + -0.125 + -0.125*y*z,
            lambda x,y,z: -0.125*x**2*z + 0.25*y*x*z + -0.125*z**2*x + -0.125*x**2 + 0.25*y*x + 0.25*y*z + -0.125*z**2 + 0.25*y + 0.0*x + 0.0*z + 0.125 + -0.125*x*z,
            lambda x,y,z: -0.125*x**2*y + 0.125*y**2*x + -0.25*z*x*y + 0.25*z*x + 0.125*x**2 + 0.125*y**2 + -0.25*z*y + 0.25*z + 0.0*x + 0.0*y + -0.125 + -0.125*x*y]
        dN7t = [lambda x,y,z: 0.25*x*y*z + 0.125*y**2*z + 0.125*z**2*y + 0.25*x*y + 0.125*y**2 + 0.125*z**2 + 0.25*x*z + 0.25*x + 0.0*y + 0.0*z + -0.125 + 0.125*y*z,
            lambda x,y,z: 0.125*x**2*z + 0.25*y*x*z + 0.125*z**2*x + 0.125*x**2 + 0.25*y*x + 0.25*y*z + 0.125*z**2 + 0.25*y + 0.0*x + 0.0*z + -0.125 + 0.125*x*z,
            lambda x,y,z: 0.125*x**2*y + 0.125*y**2*x + 0.25*z*x*y + 0.25*z*x + 0.125*x**2 + 0.125*y**2 + 0.25*z*y + 0.25*z + 0.0*x + 0.0*y + -0.125 + 0.125*x*y]
        dN8t = [lambda x,y,z: 0.25*x*y*z + -0.125*y**2*z + -0.125*z**2*y + 0.25*x*y + -0.125*y**2 + -0.125*z**2 + 0.25*x*z + 0.25*x + 0.0*y + 0.0*z + 0.125 + -0.125*y*z,
            lambda x,y,z: 0.125*x**2*z + -0.25*y*x*z + -0.125*z**2*x + 0.125*x**2 + -0.25*y*x + 0.25*y*z + 0.125*z**2 + 0.25*y + 0.0*x + 0.0*z + -0.125 + -0.125*x*z,
            lambda x,y,z: 0.125*x**2*y + -0.125*y**2*x + -0.25*z*x*y + -0.25*z*x + 0.125*x**2 + 0.125*y**2 + 0.25*z*y + 0.25*z + 0.0*x + 0.0*y + -0.125 + -0.125*x*y]
        dN9t = [lambda x,y,z: -0.5*x*y*z + 0.0*y**2*z + 0.0*z**2*y + 0.5*x*y + 0.0*y**2 + 0.0*z**2 + 0.5*x*z + -0.5*x + 0.0*y + 0.0*z + 0.0 + 0.0*y*z,
            lambda x,y,z: -0.25*x**2*z + 0.0*y*x*z + 0.0*z**2*x + 0.25*x**2 + 0.0*y*x + 0.0*y*z + 0.0*z**2 + 0.0*y + 0.0*x + 0.25*z + -0.25 + 0.0*x*z,
            lambda x,y,z: -0.25*x**2*y + 0.0*y**2*x + 0.0*z*x*y + 0.0*z*x + 0.25*x**2 + 0.0*y**2 + 0.0*z*y + 0.0*z + 0.0*x + 0.25*y + -0.25 + 0.0*x*y]
        dN10t = [lambda x,y,z: 0.0*x*y*z + -0.25*y**2*z + 0.0*z**2*y + 0.0*x*y + 0.25*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*x + 0.0*y + 0.25*z + -0.25 + 0.0*y*z,
            lambda x,y,z: 0.0*x**2*z + -0.5*y*x*z + 0.0*z**2*x + 0.0*x**2 + 0.5*y*x + 0.5*y*z + 0.0*z**2 + -0.5*y + 0.0*x + 0.0*z + 0.0 + 0.0*x*z,
            lambda x,y,z: 0.0*x**2*y + -0.25*y**2*x + 0.0*z*x*y + 0.0*z*x + 0.0*x**2 + 0.25*y**2 + 0.0*z*y + 0.0*z + 0.25*x + 0.0*y + -0.25 + 0.0*x*y]
        dN11t = [lambda x,y,z: 0.0*x*y*z + 0.0*y**2*z + -0.25*z**2*y + 0.0*x*y + 0.0*y**2 + 0.25*z**2 + 0.0*x*z + 0.0*x + 0.25*y + 0.0*z + -0.25 + 0.0*y*z,
            lambda x,y,z: 0.0*x**2*z + 0.0*y*x*z + -0.25*z**2*x + 0.0*x**2 + 0.0*y*x + 0.0*y*z + 0.25*z**2 + 0.0*y + 0.25*x + 0.0*z + -0.25 + 0.0*x*z,
            lambda x,y,z: 0.0*x**2*y + 0.0*y**2*x + -0.5*z*x*y + 0.5*z*x + 0.0*x**2 + 0.0*y**2 + 0.5*z*y + -0.5*z + 0.0*x + 0.0*y + 0.0 + 0.0*x*y]
        dN12t = [lambda x,y,z: 0.0*x*y*z + 0.25*y**2*z + 0.0*z**2*y + 0.0*x*y + -0.25*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*x + 0.0*y + -0.25*z + 0.25 + 0.0*y*z,
            lambda x,y,z: 0.0*x**2*z + 0.5*y*x*z + 0.0*z**2*x + 0.0*x**2 + -0.5*y*x + 0.5*y*z + 0.0*z**2 + -0.5*y + 0.0*x + 0.0*z + 0.0 + 0.0*x*z,
            lambda x,y,z: 0.0*x**2*y + 0.25*y**2*x + 0.0*z*x*y + 0.0*z*x + 0.0*x**2 + 0.25*y**2 + 0.0*z*y + 0.0*z + -0.25*x + 0.0*y + -0.25 + 0.0*x*y]
        dN13t = [lambda x,y,z: 0.0*x*y*z + 0.0*y**2*z + 0.25*z**2*y + 0.0*x*y + 0.0*y**2 + -0.25*z**2 + 0.0*x*z + 0.0*x + -0.25*y + 0.0*z + 0.25 + 0.0*y*z,
            lambda x,y,z: 0.0*x**2*z + 0.0*y*x*z + 0.25*z**2*x + 0.0*x**2 + 0.0*y*x + 0.0*y*z + 0.25*z**2 + 0.0*y + -0.25*x + 0.0*z + -0.25 + 0.0*x*z,
            lambda x,y,z: 0.0*x**2*y + 0.0*y**2*x + 0.5*z*x*y + -0.5*z*x + 0.0*x**2 + 0.0*y**2 + 0.5*z*y + -0.5*z + 0.0*x + 0.0*y + 0.0 + 0.0*x*y]
        dN14t = [lambda x,y,z: 0.5*x*y*z + 0.0*y**2*z + 0.0*z**2*y + -0.5*x*y + 0.0*y**2 + 0.0*z**2 + 0.5*x*z + -0.5*x + 0.0*y + 0.0*z + 0.0 + 0.0*y*z,
            lambda x,y,z: 0.25*x**2*z + 0.0*y*x*z + 0.0*z**2*x + -0.25*x**2 + 0.0*y*x + 0.0*y*z + 0.0*z**2 + 0.0*y + 0.0*x + -0.25*z + 0.25 + 0.0*x*z,
            lambda x,y,z: 0.25*x**2*y + 0.0*y**2*x + 0.0*z*x*y + 0.0*z*x + 0.25*x**2 + 0.0*y**2 + 0.0*z*y + 0.0*z + 0.0*x + -0.25*y + -0.25 + 0.0*x*y]
        dN15t = [lambda x,y,z: 0.0*x*y*z + 0.0*y**2*z + -0.25*z**2*y + 0.0*x*y + 0.0*y**2 + -0.25*z**2 + 0.0*x*z + 0.0*x + 0.25*y + 0.0*z + 0.25 + 0.0*y*z,
            lambda x,y,z: 0.0*x**2*z + 0.0*y*x*z + -0.25*z**2*x + 0.0*x**2 + 0.0*y*x + 0.0*y*z + -0.25*z**2 + 0.0*y + 0.25*x + 0.0*z + 0.25 + 0.0*x*z,
            lambda x,y,z: 0.0*x**2*y + 0.0*y**2*x + -0.5*z*x*y + -0.5*z*x + 0.0*x**2 + 0.0*y**2 + -0.5*z*y + -0.5*z + 0.0*x + 0.0*y + 0.0 + 0.0*x*y]
        dN16t = [lambda x,y,z: 0.0*x*y*z + 0.0*y**2*z + 0.25*z**2*y + 0.0*x*y + 0.0*y**2 + 0.25*z**2 + 0.0*x*z + 0.0*x + -0.25*y + 0.0*z + -0.25 + 0.0*y*z,
            lambda x,y,z: 0.0*x**2*z + 0.0*y*x*z + 0.25*z**2*x + 0.0*x**2 + 0.0*y*x + 0.0*y*z + -0.25*z**2 + 0.0*y + -0.25*x + 0.0*z + 0.25 + 0.0*x*z,
            lambda x,y,z: 0.0*x**2*y + 0.0*y**2*x + 0.5*z*x*y + 0.5*z*x + 0.0*x**2 + 0.0*y**2 + -0.5*z*y + -0.5*z + 0.0*x + 0.0*y + 0.0 + 0.0*x*y]
        dN17t = [lambda x,y,z: 0.5*x*y*z + 0.0*y**2*z + 0.0*z**2*y + 0.5*x*y + 0.0*y**2 + 0.0*z**2 + -0.5*x*z + -0.5*x + 0.0*y + 0.0*z + 0.0 + 0.0*y*z,
            lambda x,y,z: 0.25*x**2*z + 0.0*y*x*z + 0.0*z**2*x + 0.25*x**2 + 0.0*y*x + 0.0*y*z + 0.0*z**2 + 0.0*y + 0.0*x + -0.25*z + -0.25 + 0.0*x*z,
            lambda x,y,z: 0.25*x**2*y + 0.0*y**2*x + 0.0*z*x*y + 0.0*z*x + -0.25*x**2 + 0.0*y**2 + 0.0*z*y + 0.0*z + 0.0*x + -0.25*y + 0.25 + 0.0*x*y]
        dN18t = [lambda x,y,z: 0.0*x*y*z + 0.25*y**2*z + 0.0*z**2*y + 0.0*x*y + 0.25*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*x + 0.0*y + -0.25*z + -0.25 + 0.0*y*z,
            lambda x,y,z: 0.0*x**2*z + 0.5*y*x*z + 0.0*z**2*x + 0.0*x**2 + 0.5*y*x + -0.5*y*z + 0.0*z**2 + -0.5*y + 0.0*x + 0.0*z + 0.0 + 0.0*x*z,
            lambda x,y,z: 0.0*x**2*y + 0.25*y**2*x + 0.0*z*x*y + 0.0*z*x + 0.0*x**2 + -0.25*y**2 + 0.0*z*y + 0.0*z + -0.25*x + 0.0*y + 0.25 + 0.0*x*y]
        dN19t = [lambda x,y,z: 0.0*x*y*z + -0.25*y**2*z + 0.0*z**2*y + 0.0*x*y + -0.25*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*x + 0.0*y + 0.25*z + 0.25 + 0.0*y*z,
            lambda x,y,z: 0.0*x**2*z + -0.5*y*x*z + 0.0*z**2*x + 0.0*x**2 + -0.5*y*x + -0.5*y*z + 0.0*z**2 + -0.5*y + 0.0*x + 0.0*z + 0.0 + 0.0*x*z,
            lambda x,y,z: 0.0*x**2*y + -0.25*y**2*x + 0.0*z*x*y + 0.0*z*x + 0.0*x**2 + -0.25*y**2 + 0.0*z*y + 0.0*z + 0.25*x + 0.0*y + 0.25 + 0.0*x*y]
        dN20t = [lambda x,y,z: -0.5*x*y*z + 0.0*y**2*z + 0.0*z**2*y + -0.5*x*y + 0.0*y**2 + 0.0*z**2 + -0.5*x*z + -0.5*x + 0.0*y + 0.0*z + 0.0 + 0.0*y*z,
            lambda x,y,z: -0.25*x**2*z + 0.0*y*x*z + 0.0*z**2*x + -0.25*x**2 + 0.0*y*x + 0.0*y*z + 0.0*z**2 + 0.0*y + 0.0*x + 0.25*z + 0.25 + 0.0*x*z,
            lambda x,y,z: -0.25*x**2*y + 0.0*y**2*x + 0.0*z*x*y + 0.0*z*x + -0.25*x**2 + 0.0*y**2 + 0.0*z*y + 0.0*z + 0.0*x + 0.25*y + 0.25 + 0.0*x*y]
        
        dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t, dN9t, dN10t, dN11t, dN12t, dN13t, dN14t, dN15t, dN16t, dN17t, dN18t, dN19t, dN20t])        

        return dNtild

    def _ddNtild(self) -> np.ndarray:

        ddN1t = [lambda x,y,z: 0.25*y*z + -0.25*y + -0.25*z + 0.25, lambda x,y,z: 0.25*x*z + -0.25*x + -0.25*z + 0.25, lambda x,y,z: 0.25*x*y + -0.25*x + -0.25*y + 0.25]
        ddN2t = [lambda x,y,z: 0.25*y*z + -0.25*y + -0.25*z + 0.25,lambda x,y,z: -0.25*x*z + 0.25*x + -0.25*z + 0.25, lambda x,y,z: -0.25*x*y + 0.25*x + -0.25*y + 0.25]
        ddN3t = [lambda x,y,z: -0.25*y*z + 0.25*y + -0.25*z + 0.25, lambda x,y,z: -0.25*x*z + 0.25*x + -0.25*z + 0.25, lambda x,y,z: 0.25*x*y + 0.25*x + 0.25*y + 0.25]
        ddN4t = [lambda x,y,z: -0.25*y*z + 0.25*y + -0.25*z + 0.25, lambda x,y,z: 0.25*x*z + -0.25*x + -0.25*z + 0.25, lambda x,y,z: -0.25*x*y + -0.25*x + 0.25*y + 0.25]
        ddN5t = [lambda x,y,z: -0.25*y*z + -0.25*y + 0.25*z + 0.25, lambda x,y,z: -0.25*x*z + -0.25*x + 0.25*z + 0.25, lambda x,y,z: 0.25*x*y + -0.25*x + -0.25*y + 0.25]
        ddN6t = [lambda x,y,z: -0.25*y*z + -0.25*y + 0.25*z + 0.25, lambda x,y,z: 0.25*x*z + 0.25*x + 0.25*z + 0.25, lambda x,y,z: -0.25*x*y + 0.25*x + -0.25*y + 0.25]
        ddN7t = [lambda x,y,z: 0.25*y*z + 0.25*y + 0.25*z + 0.25, lambda x,y,z: 0.25*x*z + 0.25*x + 0.25*z + 0.25, lambda x,y,z: 0.25*x*y + 0.25*x + 0.25*y + 0.25]
        ddN8t = [lambda x,y,z: 0.25*y*z + 0.25*y + 0.25*z + 0.25, lambda x,y,z: -0.25*x*z + -0.25*x + 0.25*z + 0.25, lambda x,y,z: -0.25*x*y + -0.25*x + 0.25*y + 0.25]
        ddN9t = [lambda x,y,z: -0.5*y*z + 0.5*y + 0.5*z + -0.5, lambda x,y,z: 0.0*x*z + 0.0*x + 0.0*z + 0.0, lambda x,y,z: 0.0*x*y + 0.0*x + 0.0*y + 0.0]
        ddN10t = [lambda x,y,z: 0.0*y*z + 0.0*y + 0.0*z + 0.0, lambda x,y,z: -0.5*x*z + 0.5*x + 0.5*z + -0.5, lambda x,y,z: 0.0*x*y + 0.0*x + 0.0*y + 0.0]
        ddN11t = [lambda x,y,z: 0.0*y*z + 0.0*y + 0.0*z + 0.0, lambda x,y,z: 0.0*x*z + 0.0*x + 0.0*z + 0.0, lambda x,y,z: -0.5*x*y + 0.5*x + 0.5*y + -0.5]
        ddN12t = [lambda x,y,z: 0.0*y*z + 0.0*y + 0.0*z + 0.0, lambda x,y,z: 0.5*x*z + -0.5*x + 0.5*z + -0.5, lambda x,y,z: 0.0*x*y + 0.0*x + 0.0*y + 0.0]
        ddN13t = [lambda x,y,z: 0.0*y*z + 0.0*y + 0.0*z + 0.0, lambda x,y,z: 0.0*x*z + 0.0*x + 0.0*z + 0.0, lambda x,y,z: 0.5*x*y + -0.5*x + 0.5*y + -0.5]
        ddN14t = [lambda x,y,z: 0.5*y*z + -0.5*y + 0.5*z + -0.5, lambda x,y,z: 0.0*x*z + 0.0*x + 0.0*z + 0.0, lambda x,y,z: 0.0*x*y + 0.0*x + 0.0*y + 0.0]
        ddN15t = [lambda x,y,z: 0.0*y*z + 0.0*y + 0.0*z + 0.0, lambda x,y,z: 0.0*x*z + 0.0*x + 0.0*z + 0.0, lambda x,y,z: -0.5*x*y + -0.5*x + -0.5*y + -0.5]
        ddN16t = [lambda x,y,z: 0.0*y*z + 0.0*y + 0.0*z + 0.0, lambda x,y,z: 0.0*x*z + 0.0*x + 0.0*z + 0.0, lambda x,y,z: 0.5*x*y + 0.5*x + -0.5*y + -0.5]
        ddN17t = [lambda x,y,z: 0.5*y*z + 0.5*y + -0.5*z + -0.5, lambda x,y,z: 0.0*x*z + 0.0*x + 0.0*z + 0.0, lambda x,y,z: 0.0*x*y + 0.0*x + 0.0*y + 0.0]
        ddN18t = [lambda x,y,z: 0.0*y*z + 0.0*y + 0.0*z + 0.0, lambda x,y,z: 0.5*x*z + 0.5*x + -0.5*z + -0.5, lambda x,y,z: 0.0*x*y + 0.0*x + 0.0*y + 0.0]
        ddN19t = [lambda x,y,z: 0.0*y*z + 0.0*y + 0.0*z + 0.0, lambda x,y,z: -0.5*x*z + -0.5*x + -0.5*z + -0.5, lambda x,y,z: 0.0*x*y + 0.0*x + 0.0*y + 0.0]
        ddN20t = [lambda x,y,z: -0.5*y*z + -0.5*y + -0.5*z + -0.5, lambda x,y,z: 0.0*x*z + 0.0*x + 0.0*z + 0.0, lambda x,y,z: 0.0*x*y + 0.0*x + 0.0*y + 0.0]

        ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t, ddN5t, ddN6t, ddN7t, ddN8t, ddN9t, ddN10t, ddN11t, ddN12t, ddN13t, ddN14t, ddN15t, ddN16t, ddN17t, ddN18t, ddN19t, ddN20t])        

        return ddNtild

    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()
    
    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()

    def _Nvtild(self) -> np.ndarray:
        pass

    def dNvtild(self) -> np.ndarray:
        pass

    def _ddNvtild(self) -> np.ndarray:
        pass

class PRISM6(GroupElem):
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

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def indexesTriangles(self) -> list[int]:
        return super().indexesTriangles

    @property
    def indexesFaces(self) -> list[int]:
        return [0,3,4,1,0,2,5,3,1,4,5,2,3,5,4,0,1,2]
    
    @property
    def indexesSegments(self) -> np.ndarray:
        return np.array([[0,1],[1,2],[2,0],[3,4],[4,5],[5,3],[0,3],[1,4],[2,5]])

    def _Ntild(self) -> np.ndarray:        

        N1t = lambda x,y,z: 0.5*x*z + 0.5*y*z + -0.5*x + -0.5*y + -0.5*z + 0.5
        N2t = lambda x,y,z: -0.5*x*z + 0.0*y*z + 0.5*x + 0.0*y + 0.0*z + 0.0
        N3t = lambda x,y,z: 0.0*x*z + -0.5*y*z + 0.0*x + 0.5*y + 0.0*z + 0.0
        N4t = lambda x,y,z: -0.5*x*z + -0.5*y*z + -0.5*x + -0.5*y + 0.5*z + 0.5
        N5t = lambda x,y,z: 0.5*x*z + 0.0*y*z + 0.5*x + 0.0*y + 0.0*z + 0.0
        N6t = lambda x,y,z: 0.0*x*z + 0.5*y*z + 0.0*x + 0.5*y + 0.0*z + 0.0        

        Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t]).reshape(-1, 1)

        return Ntild
    
    def _dNtild(self) -> np.ndarray:        

        dN1t = [lambda x,y,z: 0.5*z + -0.5, lambda x,y,z: 0.5*z + -0.5, lambda x,y,z: 0.5*x + 0.5*y + -0.5]
        dN2t = [lambda x,y,z: -0.5*z + 0.5, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: -0.5*x + 0.0*y + 0.0]
        dN3t = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: -0.5*z + 0.5, lambda x,y,z: 0.0*x + -0.5*y + 0.0]
        dN4t = [lambda x,y,z: -0.5*z + -0.5, lambda x,y,z: -0.5*z + -0.5, lambda x,y,z: -0.5*x + -0.5*y + 0.5]
        dN5t = [lambda x,y,z: 0.5*z + 0.5, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.5*x + 0.0*y + 0.0]
        dN6t = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.5*z + 0.5, lambda x,y,z: 0.0*x + 0.5*y + 0.0]

        dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t])        

        return dNtild

    def _ddNtild(self) -> np.ndarray:
        return super()._ddNtild()

    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()
    
    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()

    def _Nvtild(self) -> np.ndarray:
        pass

    def dNvtild(self) -> np.ndarray:
        pass

    def _ddNvtild(self) -> np.ndarray:
        pass

class PRISM15(GroupElem):
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
    #     1------9------2

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def indexesTriangles(self) -> list[int]:
        return super().indexesTriangles

    @property
    def indexesFaces(self) -> list[int]:
        return [0,8,3,12,4,10,1,6,
                0,7,2,11,5,13,3,8,
                1,10,4,14,5,11,2,9,
                3,13,5,14,4,12,
                0,6,1,9,2,7]
    
    @property
    def indexesSegments(self) -> np.ndarray:
        return np.array([[0,1],[1,2],[2,0],[3,4],[4,5],[5,3],[0,3],[1,4],[2,5]])

    def _Ntild(self) -> np.ndarray:
        
        N1t = lambda x,y,z: -1.0*x**2*z + 4.0*x**2*y + -1.0*y**2*z + -0.5*z**2*x + -0.5*z**2*y + -2.0*x*y*z + 1.0*x**2 + 1.0*y**2 + 0.5*z**2 + 1.5*x*z + 1.5*y*z + -1.0*x + -1.0*y + -0.5*z + 0.0
        N2t = lambda x,y,z: -1.0*x**2*z + 0.0*x**2*y + 0.0*y**2*z + 0.5*z**2*x + 0.0*z**2*y + 0.0*x*y*z + 1.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.5*x*z + 0.0*y*z + -1.0*x + 0.0*y + 0.0*z + 0.0
        N3t = lambda x,y,z: 0.0*x**2*z + 0.0*x**2*y + -1.0*y**2*z + 0.0*z**2*x + 0.5*z**2*y + 0.0*x*y*z + 0.0*x**2 + 1.0*y**2 + 0.0*z**2 + 0.0*x*z + 0.5*y*z + 0.0*x + -1.0*y + 0.0*z + 0.0
        N4t = lambda x,y,z: 1.0*x**2*z + 4.0*x**2*y + 1.0*y**2*z + -0.5*z**2*x + -0.5*z**2*y + 2.0*x*y*z + 1.0*x**2 + 1.0*y**2 + 0.5*z**2 + -1.5*x*z + -1.5*y*z + -1.0*x + -1.0*y + 0.5*z + 0.0
        N5t = lambda x,y,z: 1.0*x**2*z + 0.0*x**2*y + 0.0*y**2*z + 0.5*z**2*x + 0.0*z**2*y + 0.0*x*y*z + 1.0*x**2 + 0.0*y**2 + 0.0*z**2 + -0.5*x*z + 0.0*y*z + -1.0*x + 0.0*y + 0.0*z + 0.0
        N6t = lambda x,y,z: 0.0*x**2*z + 0.0*x**2*y + 1.0*y**2*z + 0.0*z**2*x + 0.5*z**2*y + 0.0*x*y*z + 0.0*x**2 + 1.0*y**2 + 0.0*z**2 + 0.0*x*z + -0.5*y*z + 0.0*x + -1.0*y + 0.0*z + 0.0
        N7t = lambda x,y,z: 2.0*x**2*z + -4.0*x**2*y + 0.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + 2.0*x*y*z + -2.0*x**2 + 0.0*y**2 + 0.0*z**2 + -2.0*x*z + 0.0*y*z + 2.0*x + 0.0*y + 0.0*z + 0.0
        N8t = lambda x,y,z: 0.0*x**2*z + -4.0*x**2*y + 2.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + 2.0*x*y*z + 0.0*x**2 + -2.0*y**2 + 0.0*z**2 + 0.0*x*z + -2.0*y*z + 0.0*x + 2.0*y + 0.0*z + 0.0
        N9t = lambda x,y,z: 0.0*x**2*z + 0.0*x**2*y + 0.0*y**2*z + 1.0*z**2*x + 1.0*z**2*y + 0.0*x*y*z + 0.0*x**2 + 0.0*y**2 + -1.0*z**2 + 0.0*x*z + 0.0*y*z + -1.0*x + -1.0*y + 0.0*z + 1.0
        N10t = lambda x,y,z: 0.0*x**2*z + 4.0*x**2*y + 0.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + -2.0*x*y*z + 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*y*z + 0.0*x + 0.0*y + 0.0*z + 0.0
        N11t = lambda x,y,z: 0.0*x**2*z + 0.0*x**2*y + 0.0*y**2*z + -1.0*z**2*x + 0.0*z**2*y + 0.0*x*y*z + 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*y*z + 1.0*x + 0.0*y + 0.0*z + 0.0
        N12t = lambda x,y,z: 0.0*x**2*z + 0.0*x**2*y + 0.0*y**2*z + 0.0*z**2*x + -1.0*z**2*y + 0.0*x*y*z + 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*y*z + 0.0*x + 1.0*y + 0.0*z + 0.0
        N13t = lambda x,y,z: -2.0*x**2*z + -4.0*x**2*y + 0.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + -2.0*x*y*z + -2.0*x**2 + 0.0*y**2 + 0.0*z**2 + 2.0*x*z + 0.0*y*z + 2.0*x + 0.0*y + 0.0*z + 0.0
        N14t = lambda x,y,z: 0.0*x**2*z + -4.0*x**2*y + -2.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + -2.0*x*y*z + 0.0*x**2 + -2.0*y**2 + 0.0*z**2 + 0.0*x*z + 2.0*y*z + 0.0*x + 2.0*y + 0.0*z + 0.0
        N15t = lambda x,y,z: 0.0*x**2*z + 4.0*x**2*y + 0.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + 2.0*x*y*z + 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*y*z + 0.0*x + 0.0*y + 0.0*z + 0.0

        Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t, N9t, N10t, N11t, N12t, N13t, N14t, N15t]).reshape(-1, 1)

        return Ntild
    
    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda x,y,z: -2.0*x*z + 8.0*x*y + -0.5*z**2 + -2.0*y*z + 2.0*x + 1.5*z + -1.0,
                lambda x,y,z: 4.0*x**2 + -2.0*y*z + -0.5*z**2 + -2.0*x*z + 2.0*y + 1.5*z + -1.0,
                lambda x,y,z: -1.0*x**2 + -1.0*y**2 + -1.0*z*x + -1.0*z*y + -2.0*x*y + 1.0*z + 1.5*x + 1.5*y + -0.5]
        dN2t = [lambda x,y,z: -2.0*x*z + 0.0*x*y + 0.5*z**2 + 0.0*y*z + 2.0*x + 0.5*z + -1.0,
                lambda x,y,z: 0.0*x**2 + 0.0*y*z + 0.0*z**2 + 0.0*x*z + 0.0*y + 0.0*z + 0.0,
                lambda x,y,z: -1.0*x**2 + 0.0*y**2 + 1.0*z*x + 0.0*z*y + 0.0*x*y + 0.0*z + 0.5*x + 0.0*y + 0.0]
        dN3t = [lambda x,y,z: 0.0*x*z + 0.0*x*y + 0.0*z**2 + 0.0*y*z + 0.0*x + 0.0*z + 0.0,
                lambda x,y,z: 0.0*x**2 + -2.0*y*z + 0.5*z**2 + 0.0*x*z + 2.0*y + 0.5*z + -1.0,
                lambda x,y,z: 0.0*x**2 + -1.0*y**2 + 0.0*z*x + 1.0*z*y + 0.0*x*y + 0.0*z + 0.0*x + 0.5*y + 0.0]
        dN4t = [lambda x,y,z: 2.0*x*z + 8.0*x*y + -0.5*z**2 + 2.0*y*z + 2.0*x + -1.5*z + -1.0,
                lambda x,y,z: 4.0*x**2 + 2.0*y*z + -0.5*z**2 + 2.0*x*z + 2.0*y + -1.5*z + -1.0,
                lambda x,y,z: 1.0*x**2 + 1.0*y**2 + -1.0*z*x + -1.0*z*y + 2.0*x*y + 1.0*z + -1.5*x + -1.5*y + 0.5]
        dN5t = [lambda x,y,z: 2.0*x*z + 0.0*x*y + 0.5*z**2 + 0.0*y*z + 2.0*x + -0.5*z + -1.0,
                lambda x,y,z: 0.0*x**2 + 0.0*y*z + 0.0*z**2 + 0.0*x*z + 0.0*y + 0.0*z + 0.0,
                lambda x,y,z: 1.0*x**2 + 0.0*y**2 + 1.0*z*x + 0.0*z*y + 0.0*x*y + 0.0*z + -0.5*x + 0.0*y + 0.0]
        dN6t = [lambda x,y,z: 0.0*x*z + 0.0*x*y + 0.0*z**2 + 0.0*y*z + 0.0*x + 0.0*z + 0.0,
                lambda x,y,z: 0.0*x**2 + 2.0*y*z + 0.5*z**2 + 0.0*x*z + 2.0*y + -0.5*z + -1.0,
                lambda x,y,z: 0.0*x**2 + 1.0*y**2 + 0.0*z*x + 1.0*z*y + 0.0*x*y + 0.0*z + 0.0*x + -0.5*y + 0.0]
        dN7t = [lambda x,y,z: 4.0*x*z + -8.0*x*y + 0.0*z**2 + 2.0*y*z + -4.0*x + -2.0*z + 2.0,
                lambda x,y,z: -4.0*x**2 + 0.0*y*z + 0.0*z**2 + 2.0*x*z + 0.0*y + 0.0*z + 0.0,
                lambda x,y,z: 2.0*x**2 + 0.0*y**2 + 0.0*z*x + 0.0*z*y + 2.0*x*y + 0.0*z + -2.0*x + 0.0*y + 0.0]
        dN8t = [lambda x,y,z: 0.0*x*z + -8.0*x*y + 0.0*z**2 + 2.0*y*z + 0.0*x + 0.0*z + 0.0,
                lambda x,y,z: -4.0*x**2 + 4.0*y*z + 0.0*z**2 + 2.0*x*z + -4.0*y + -2.0*z + 2.0,
                lambda x,y,z: 0.0*x**2 + 2.0*y**2 + 0.0*z*x + 0.0*z*y + 2.0*x*y + 0.0*z + 0.0*x + -2.0*y + 0.0]
        dN9t = [lambda x,y,z: 0.0*x*z + 0.0*x*y + 1.0*z**2 + 0.0*y*z + 0.0*x + 0.0*z + -1.0,
                lambda x,y,z: 0.0*x**2 + 0.0*y*z + 1.0*z**2 + 0.0*x*z + 0.0*y + 0.0*z + -1.0,
                lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 2.0*z*x + 2.0*z*y + 0.0*x*y + -2.0*z + 0.0*x + 0.0*y + 0.0]
        dN10t = [lambda x,y,z: 0.0*x*z + 8.0*x*y + 0.0*z**2 + -2.0*y*z + 0.0*x + 0.0*z + 0.0,
                lambda x,y,z: 4.0*x**2 + 0.0*y*z + 0.0*z**2 + -2.0*x*z + 0.0*y + 0.0*z + 0.0,
                lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 0.0*z*x + 0.0*z*y + -2.0*x*y + 0.0*z + 0.0*x + 0.0*y + 0.0]
        dN11t = [lambda x,y,z: 0.0*x*z + 0.0*x*y + -1.0*z**2 + 0.0*y*z + 0.0*x + 0.0*z + 1.0,
                lambda x,y,z: 0.0*x**2 + 0.0*y*z + 0.0*z**2 + 0.0*x*z + 0.0*y + 0.0*z + 0.0,
                lambda x,y,z: 0.0*x**2 + 0.0*y**2 + -2.0*z*x + 0.0*z*y + 0.0*x*y + 0.0*z + 0.0*x + 0.0*y + 0.0]
        dN12t = [lambda x,y,z: 0.0*x*z + 0.0*x*y + 0.0*z**2 + 0.0*y*z + 0.0*x + 0.0*z + 0.0,
                lambda x,y,z: 0.0*x**2 + 0.0*y*z + -1.0*z**2 + 0.0*x*z + 0.0*y + 0.0*z + 1.0,
                lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 0.0*z*x + -2.0*z*y + 0.0*x*y + 0.0*z + 0.0*x + 0.0*y + 0.0]
        dN13t = [lambda x,y,z: -4.0*x*z + -8.0*x*y + 0.0*z**2 + -2.0*y*z + -4.0*x + 2.0*z + 2.0,
                lambda x,y,z: -4.0*x**2 + 0.0*y*z + 0.0*z**2 + -2.0*x*z + 0.0*y + 0.0*z + 0.0,
                lambda x,y,z: -2.0*x**2 + 0.0*y**2 + 0.0*z*x + 0.0*z*y + -2.0*x*y + 0.0*z + 2.0*x + 0.0*y + 0.0]
        dN14t = [lambda x,y,z: 0.0*x*z + -8.0*x*y + 0.0*z**2 + -2.0*y*z + 0.0*x + 0.0*z + 0.0,
                lambda x,y,z: -4.0*x**2 + -4.0*y*z + 0.0*z**2 + -2.0*x*z + -4.0*y + 2.0*z + 2.0,
                lambda x,y,z: 0.0*x**2 + -2.0*y**2 + 0.0*z*x + 0.0*z*y + -2.0*x*y + 0.0*z + 0.0*x + 2.0*y + 0.0]
        dN15t = [lambda x,y,z: 0.0*x*z + 8.0*x*y + 0.0*z**2 + 2.0*y*z + 0.0*x + 0.0*z + 0.0,
                lambda x,y,z: 4.0*x**2 + 0.0*y*z + 0.0*z**2 + 2.0*x*z + 0.0*y + 0.0*z + 0.0,
                lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 0.0*z*x + 0.0*z*y + 2.0*x*y + 0.0*z + 0.0*x + 0.0*y + 0.0]
        
        dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t, dN9t, dN10t, dN11t, dN12t, dN13t, dN14t, dN15t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:

        ddN1t = [lambda x,y,z: -2.0*z + 8.0*y + 2.0, lambda x,y,z: -2.0*z + 2.0, lambda x,y,z: -1.0*x + -1.0*y + 1.0]
        ddN2t = [lambda x,y,z: -2.0*z + 0.0*y + 2.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 1.0*x + 0.0*y + 0.0]
        ddN3t = [lambda x,y,z: 0.0*z + 0.0*y + 0.0, lambda x,y,z: -2.0*z + 2.0, lambda x,y,z: 0.0*x + 1.0*y + 0.0]
        ddN4t = [lambda x,y,z: 2.0*z + 8.0*y + 2.0, lambda x,y,z: 2.0*z + 2.0, lambda x,y,z: -1.0*x + -1.0*y + 1.0]
        ddN5t = [lambda x,y,z: 2.0*z + 0.0*y + 2.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 1.0*x + 0.0*y + 0.0]
        ddN6t = [lambda x,y,z: 0.0*z + 0.0*y + 0.0, lambda x,y,z: 2.0*z + 2.0, lambda x,y,z: 0.0*x + 1.0*y + 0.0]
        ddN7t = [lambda x,y,z: 4.0*z + -8.0*y + -4.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]
        ddN8t = [lambda x,y,z: 0.0*z + -8.0*y + 0.0, lambda x,y,z: 4.0*z + -4.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]
        ddN9t = [lambda x,y,z: 0.0*z + 0.0*y + 0.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 2.0*x + 2.0*y + -2.0]
        ddN10t = [lambda x,y,z: 0.0*z + 8.0*y + 0.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]
        ddN11t = [lambda x,y,z: 0.0*z + 0.0*y + 0.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: -2.0*x + 0.0*y + 0.0]
        ddN12t = [lambda x,y,z: 0.0*z + 0.0*y + 0.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*x + -2.0*y + 0.0]
        ddN13t = [lambda x,y,z: -4.0*z + -8.0*y + -4.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]
        ddN14t = [lambda x,y,z: 0.0*z + -8.0*y + 0.0, lambda x,y,z: -4.0*z + -4.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]
        ddN15t = [lambda x,y,z: 0.0*z + 8.0*y + 0.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]

        ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t, ddN5t, ddN6t, ddN7t, ddN8t, ddN9t, ddN10t, ddN11t, ddN12t, ddN13t, ddN14t, ddN15t])

        return ddNtild

    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()
    
    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()

    def _Nvtild(self) -> np.ndarray:
        pass

    def dNvtild(self) -> np.ndarray:
        pass

    def _ddNvtild(self) -> np.ndarray:
        pass

        

