"""Module for creating element groups. A mesh uses several groups of elements. For example, a TRI3 mesh uses POINT, SEG2 and TRI3 elements."""

from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
from scipy.optimize import least_squares
import numpy as np
import scipy.sparse as sparse

from Geoms import *
from Gauss import Gauss

class ElemType(str, Enum):
    """Implemented element types"""

    POINT = "POINT"
    SEG2 = "SEG2"
    SEG3 = "SEG3"
    SEG4 = "SEG4"
    # SEG5 = "SEG5"
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

    def __str__(self) -> str:
        return self.name
    
    @staticmethod
    def get_1D() -> list[str]:
        """1D element types"""        
        liste1D = [ElemType.SEG2, ElemType.SEG3, ElemType.SEG4]
        return liste1D
    
    @staticmethod
    def get_2D() -> list[str]:
        """2D element types"""
        liste2D = [ElemType.TRI3, ElemType.TRI6, ElemType.TRI10, ElemType.QUAD4, ElemType.QUAD8]
        return liste2D
    
    @staticmethod
    def get_3D() -> list[str]:
        """3D element types"""
        liste3D = [ElemType.TETRA4, ElemType.TETRA10, ElemType.HEXA8, ElemType.HEXA20, ElemType.PRISM6, ElemType.PRISM15]
        return liste3D

class MatrixType(str, Enum):
    """Order used for integration over elements (determines the number of integration points)."""

    rigi = "rigi"
    """dN*dN type"""
    mass = "mass"
    """N*N type"""
    beam = "beam"
    """ddNv*ddNv type"""

    @staticmethod
    def get_types() -> list[str]:
        return [MatrixType.rigi, MatrixType.mass, MatrixType.beam]

class _GroupElem(ABC):

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

        elemType, nPe, dim, order, nbFaces, nbCorners = _GroupElem_Factory.Get_ElemInFos(gmshId)

        self.__elemType = elemType
        self.__nPe = nPe
        self.__dim = dim
        self.__order = order
        self.__nbFaces = nbFaces
        self.__nbCorners = nbCorners 
        
        # Elements
        self.__connect = connect

        # Nodes
        self.__nodes = nodes
        self.__coordoGlob = coordoGlob

        self.__dict_nodes_tags = {}
        self.__dict_elements_tags = {}        
        self._InitMatrix()
    
    def _InitMatrix(self) -> None:
        """Initialize matrix dictionaries for finite element construction"""
        # Dictionaries for each matrix type        
        self.__dict_dN_e_pg = {}
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
    def gmshId(self) -> int:
        """Gmsh Id"""
        return self.__gmshId

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
    def order(self) -> int:
        """Element order"""
        return self.__order    

    @property
    def inDim(self) -> int:
        """Dimension in which the elements are located"""
        if self.elemType in ElemType.get_3D():
            return 3
        else:
            x,y,z = np.abs(self.coordo.T)
            if np.max(y)==0 and np.max(z)==0:
                inDim = 1
            if np.max(z)==0:
                inDim = 2
            else:
                inDim = 3
            return inDim

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
        return np.arange(self.__connect.shape[0], dtype=int)

    @property
    def Nn(self) -> int:
        """Number of nodes"""
        return self.__nodes.size

    @property
    def coordo(self) -> np.ndarray:
        """This matrix contains the element group coordinates (Nn, 3)"""
        coordo: np.ndarray = self.__coordoGlob[self.__nodes]
        return coordo.copy()

    @property
    def coordoGlob(self) -> np.ndarray:
        """This matrix contains all the mesh coordinates (mesh.Nn, 3)"""
        return self.__coordoGlob.copy()
    
    @coordoGlob.setter
    def coordoGlob(self, coordo: np.ndarray) -> None:
        if coordo.shape == self.__coordoGlob.shape:
            self.__coordoGlob = coordo
            self._InitMatrix()

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

        lines = self.connect.reshape(-1)

        Nn = int(lines.max()+1)
        columns = np.repeat(listElem, nPe)

        return sparse.csr_matrix((np.ones(nPe*Ne),(lines, columns)),shape=(Nn,Ne))

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
        """Repeat shape functions in the local coordinates

        Parameters
        ----------
        matrixType : MatrixType
            matrix type
        repeat : int, optional
            number of repetitions, by default 1
        
        Returns:
        -------
        • Vector shape functions (pg, rep=2, rep=2*dim)\n
            [Ni 0 . . . Nn 0 \n
            0 Ni . . . 0 Nn]

        • Scalar shape functions (pg, rep=1, nPe)\n
            [Ni . . . Nn]
        """
        if self.dim == 0: return

        assert isinstance(repeat, int)
        assert repeat >= 1

        N_pg = self.Get_N_pg(matrixType)

        if not isinstance(N_pg, np.ndarray): return

        if repeat <= 1:
            return N_pg
        else:
            size = N_pg.shape[2]*repeat
            N_vect_pg = np.zeros((N_pg.shape[0] ,repeat , size))

            for r in range(repeat):
                N_vect_pg[:, r, np.arange(r, size, repeat)] = N_pg[:,0,:]
            
            return N_vect_pg
    
    def Get_dN_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluate shape functions first derivatives in the global coordinates.\n
        [Ni,x . . . Nn,x\n
        Ni,y ... Nn,y]\n
        (e, pg, dim, nPe)\n
        """
        assert matrixType in MatrixType.get_types()

        if matrixType not in self.__dict_dN_e_pg.keys():

            invF_e_pg = self.Get_invF_e_pg(matrixType)

            dN_pg = self.Get_dN_pg(matrixType)

            # Derivation of shape functions in the real base
            dN_e_pg: np.ndarray = np.einsum('epik,pkj->epij', invF_e_pg, dN_pg, optimize='optimal')
            self.__dict_dN_e_pg[matrixType] = dN_e_pg

        return self.__dict_dN_e_pg[matrixType].copy()
    
    def Get_ddN_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluate shape functions second derivatives in the global coordinates.\n
        [Ni,xx . . . Nn,xx\n
        Ni,yy ... Nn,yy]\n
        (e, pg, dim, nPe)\n
        """
        assert matrixType in MatrixType.get_types()

        if matrixType not in self.__dict_ddN_e_pg.keys():

            invF_e_pg = self.Get_invF_e_pg(matrixType)

            ddN_pg = self.Get_ddN_pg(matrixType)

            # Derivé des fonctions de formes dans la base réele
            ddN_e_pg = np.array(np.einsum('epik,pkj->epij', invF_e_pg, ddN_pg, optimize='optimal'))
            self.__dict_ddN_e_pg[matrixType] = ddN_e_pg

        return self.__dict_ddN_e_pg[matrixType].copy()
    
    def Get_Nv_e_pg(self) -> np.ndarray:
        """Evaluate beam shape functions in the global coordinates.\n
        [phi_i psi_i . . . phi_n psi_n]\n
        (e, pg, 1, nPe*2)
        """
        if self.dim != 1: return

        matrixType = MatrixType.beam

        invF_e_pg = self.Get_invF_e_pg(matrixType)
        Nv_pg = self.Get_Nv_pg(matrixType)

        Ne = self.Ne
        nPe = self.nPe
        jacobian_e_pg = self.Get_jacobian_e_pg(matrixType)
        pg = self.Get_gauss(matrixType)
        
        Nv_e_pg = invF_e_pg @ Nv_pg
        
        # multiply by the beam length on psi_i,xx functions            
        l_e = self.length_e
        columns = np.arange(1, nPe*2, 2)
        for column in columns:
            Nv_e_pg[:,:,0,column] = np.einsum('ep,e->ep', Nv_e_pg[:,:,0,column], l_e, optimize='optimal')    

        return Nv_e_pg

    def Get_dNv_e_pg(self) -> np.ndarray:
        """Evaluate beam shape functions first derivatives in the global coordinates.\n
        [phi_i,x psi_i,x . . . phi_n,x psi_n,x]\n
        (e, pg, 1, nPe*2)
        """
        if self.dim != 1: return

        matrixType = MatrixType.beam

        invF_e_pg = self.Get_invF_e_pg(matrixType)
        dNv_pg = self.Get_dNv_pg(matrixType)

        Ne = self.Ne
        nPe = self.nPe
        jacobian_e_pg = self.Get_jacobian_e_pg(matrixType)
        pg = self.Get_gauss(matrixType)
        
        dNv_e_pg = invF_e_pg @ dNv_pg
        
        # multiply by the beam length on psi_i,xx functions            
        l_e = self.length_e
        columns = np.arange(1, nPe*2, 2)
        for column in columns:
            dNv_e_pg[:,:,0,column] = np.einsum('ep,e->ep', dNv_e_pg[:,:,0,column], l_e, optimize='optimal')

        return dNv_e_pg

    def Get_ddNv_e_pg(self) -> np.ndarray:
        """Evaluate beam shape functions second derivatives in the global coordinates.\n
        [phi_i,xx psi_i,xx . . . phi_n,xx psi_n,xx]\n
        (e, pg, 1, nPe*2)
        """
        if self.dim != 1: return
        
        matrixType = MatrixType.beam        

        invF_e_pg = self.Get_invF_e_pg(matrixType)
        ddNv_pg = self.Get_ddNv_pg(matrixType)

        Ne = self.Ne
        nPe = self.nPe
        jacobian_e_pg = self.Get_jacobian_e_pg(matrixType)
        pg = self.Get_gauss(matrixType)
        
        ddNv_e_pg = invF_e_pg @ invF_e_pg @ ddNv_pg
        
        # multiply by the beam length on psi_i,xx functions            
        l_e = self.length_e
        columns = np.arange(1, nPe*2, 2)
        for column in columns:
            ddNv_e_pg[:,:,0,column] = np.einsum('ep,e->ep', ddNv_e_pg[:,:,0,column], l_e, optimize='optimal')

        return ddNv_e_pg

    def Get_B_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Construct the matrix used to calculate deformations from displacements.\n
        WARNING: Use Kelvin Mandel Notation\n
        [N1,x 0 N2,x 0 Nn,x 0\n
        0 N1,y 0 N2,y 0 Nn,y\n
        N1,y N1,x N2,y N2,x N3,y N3,x]\n
        (e, pg, (3 or 6), nPe*dim)        
        """
        assert matrixType in MatrixType.get_types()

        if matrixType not in self.__dict_B_e_pg.keys():

            dN_e_pg = self.Get_dN_e_pg(matrixType)

            Ne = self.Ne
            nPg = self.Get_gauss(matrixType).nPg
            nPe = self.nPe
            dim = self.dim

            cM = 1/np.sqrt(2)
            
            columnsX = np.arange(0, nPe*dim, dim)
            columnsY = np.arange(1, nPe*dim, dim)
            columnsZ = np.arange(2, nPe*dim, dim)

            if self.dim == 2:                
                B_e_pg = np.zeros((Ne, nPg, 3, nPe*dim))                
                
                dNdx = dN_e_pg[:,:,0]
                dNdy = dN_e_pg[:,:,1]

                B_e_pg[:,:,0,columnsX] = dNdx
                B_e_pg[:,:,1,columnsY] = dNdy
                B_e_pg[:,:,2,columnsX] = dNdy*cM; B_e_pg[:,:,2,columnsY] = dNdx*cM
            else:
                B_e_pg = np.zeros((Ne, nPg, 6, nPe*dim))

                dNdx = dN_e_pg[:,:,0]
                dNdy = dN_e_pg[:,:,1]
                dNdz = dN_e_pg[:,:,2]

                B_e_pg[:,:,0,columnsX] = dNdx
                B_e_pg[:,:,1,columnsY] = dNdy
                B_e_pg[:,:,2,columnsZ] = dNdz
                B_e_pg[:,:,3,columnsY] = dNdz*cM; B_e_pg[:,:,3,columnsZ] = dNdy*cM
                B_e_pg[:,:,4,columnsX] = dNdz*cM; B_e_pg[:,:,4,columnsZ] = dNdx*cM
                B_e_pg[:,:,5,columnsX] = dNdy*cM; B_e_pg[:,:,5,columnsY] = dNdx*cM

            self.__dict_B_e_pg[matrixType] = B_e_pg
        
        return self.__dict_B_e_pg[matrixType].copy()

    def Get_leftDispPart(self, matrixType: MatrixType) -> np.ndarray:
        """Left side of local displacement matrices\n
        Ku_e = jacobian_e_pg * weight_pg * B_e_pg' * c_e_pg * B_e_pg\n
        
        Returns (epij) -> jacobian_e_pg * weight_pg * B_e_pg'.
        """

        assert matrixType in MatrixType.get_types()

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

        assert matrixType in MatrixType.get_types()

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

        assert matrixType in MatrixType.get_types()

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

        assert matrixType in MatrixType.get_types()

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
                    j = np.cross([0,0,1], i)
                    k = np.cross(i, j, axis=1)

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
        
        coordo_e * sysCoord_e -> coordinates of nodes in element base.
        """
        return self.__Get_sysCoord_e()
    
    def Integrate_e(self, func=lambda x,y,z: 1) -> np.ndarray:
        """Integrates the function over elements.

        Parameters
        ----------
        func : lambda
            function that uses the x,y,z coordinates of the element's integration points\n
            Examples:\n
            lambda x,y,z: 1 -> that will just integrate the element
            lambda x,y,z: x
            lambda x,y,z: x + y\n
            lambda x,y,z: z**2

        Returns
        -------
        np.ndarray
            integrated values on elements
        """

        matrixType = MatrixType.mass

        jacobian_e_pg = self.Get_jacobian_e_pg(matrixType)
        weight_pg = self.Get_weight_pg(matrixType)
        coord_e_pg = self.Get_GaussCoordinates_e_p(matrixType)
        eval_e_pg = func(coord_e_pg[:,:,0],coord_e_pg[:,:,1],coord_e_pg[:,:,2])

        if isinstance(eval_e_pg, (float,int)):
            ind = ''
        else:
            ind = 'ep'

        values_e: np.ndarray = np.einsum(f'ep,p,{ind}->e',jacobian_e_pg, weight_pg, eval_e_pg, optimize='optimal')

        return values_e
    
    @property
    def length_e(self) -> np.ndarray:
        """Length covered by each element"""
        if self.dim != 1: return        
        length_e = self.Integrate_e(lambda x,y,z: 1)
        return length_e

    @property
    def length(self) -> float:
        """Length covered by elements"""
        if self.dim != 1: return
        return self.length_e.sum()
    
    @property
    def area_e(self) -> np.ndarray:
        """Area covered by each element"""
        if self.dim != 2: return
        area_e = self.Integrate_e(lambda x,y,z: 1)
        return area_e

    @property
    def area(self) -> float:
        """Area covered by elements"""
        if self.dim != 2: return
        return self.area_e.sum()
    
    @property
    def volume_e(self) -> np.ndarray:
        """Volume covered by each element"""
        if self.dim != 3: return        
        volume_e = self.Integrate_e(lambda x,y,z: 1)
        return volume_e
    
    @property
    def volume(self) -> float:
        """Volume covered by elements"""
        if self.dim != 3: return
        return self.volume_e.sum()
    
    @property
    def center(self) -> np.ndarray:
        """Center of mass / barycenter / inertia center"""

        matrixType = MatrixType.mass

        coordo_e_p = self.Get_GaussCoordinates_e_p(matrixType)

        jacobian_e_p = self.Get_jacobian_e_pg(matrixType)
        weight_p = self.Get_weight_pg(matrixType)

        size = np.einsum('ep,p->', jacobian_e_p, weight_p, optimize='optimal')

        center: np.ndarray = np.einsum('ep,p,epi->i', jacobian_e_p, weight_p, coordo_e_p, optimize='optimal') / size

        return center

    def Get_F_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Returns the Jacobian matrix
        This matrix describes the variations of the axes from the reference element to the real element.
        Transforms the reference element to the real element with invF_e_pg"""
        if self.dim == 0: return
        if matrixType not in self.__dict_F_e_pg.keys():

            coordo_n = self.coordoGlob[:]

            coordo_e: np.ndarray = coordo_n[self.__connect]

            nodesBase = coordo_e.copy()
            if self.dim != self.inDim:
                sysCoord_e = self.sysCoord_e # base change matrix for each element
                nodesBase = coordo_e @ sysCoord_e #node coordinates in the elements base

            nodesBaseDim = nodesBase[:,:,range(self.dim)]

            dN_pg = self.Get_dN_pg(matrixType)

            F_e_pg: np.ndarray = np.einsum('pik,ekj->epij', dN_pg, nodesBaseDim, optimize='optimal')
            
            self.__dict_F_e_pg[matrixType] = F_e_pg

        return self.__dict_F_e_pg[matrixType].copy()
    
    def Get_jacobian_e_pg(self, matrixType: MatrixType, absoluteValues=True) -> np.ndarray:
        """Returns the jacobians\n
        variation in size (length, area or volume) between the reference element and the real element
        """
        if self.dim == 0: return
        if matrixType not in self.__dict_jacobian_e_pg.keys():

            F_e_pg = self.Get_F_e_pg(matrixType)

            if self.dim == 1:
                Ne = F_e_pg.shape[0]
                nPg = F_e_pg.shape[1]
                jacobian_e_pg = F_e_pg.reshape((Ne, nPg))

            elif self.dim == 2:
                a_e_pg = F_e_pg[:,:,0,0]
                b_e_pg = F_e_pg[:,:,0,1]
                c_e_pg = F_e_pg[:,:,1,0]
                d_e_pg = F_e_pg[:,:,1,1]
                jacobian_e_pg = (a_e_pg*d_e_pg)-(c_e_pg*b_e_pg)
            
            elif self.dim == 3:
                a11_e_pg = F_e_pg[:,:,0,0]; a12_e_pg = F_e_pg[:,:,0,1]; a13_e_pg = F_e_pg[:,:,0,2]
                a21_e_pg = F_e_pg[:,:,1,0]; a22_e_pg = F_e_pg[:,:,1,1]; a23_e_pg = F_e_pg[:,:,1,2]
                a31_e_pg = F_e_pg[:,:,2,0]; a32_e_pg = F_e_pg[:,:,2,1]; a33_e_pg = F_e_pg[:,:,2,2]

                jacobian_e_pg = a11_e_pg * ((a22_e_pg*a33_e_pg)-(a32_e_pg*a23_e_pg)) - a12_e_pg * ((a21_e_pg*a33_e_pg)-(a31_e_pg*a23_e_pg)) + a13_e_pg * ((a21_e_pg*a32_e_pg)-(a31_e_pg*a22_e_pg))

            # test = np.linalg.det(F_e_pg) - jacobian_e_pg
            self.__dict_jacobian_e_pg[matrixType] = jacobian_e_pg

        jacobian_e_pg = self.__dict_jacobian_e_pg[matrixType].copy()

        if absoluteValues:
            jacobian_e_pg = np.abs(jacobian_e_pg)

        return jacobian_e_pg
    
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

                det = self.Get_jacobian_e_pg(matrixType, absoluteValues=False)

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

                det = self.Get_jacobian_e_pg(matrixType, absoluteValues=False)

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
    def _Evaluates_Functions(functions: np.ndarray, coord: np.ndarray) -> np.ndarray:
        """Evaluates functions at coordinates. Uses this function to evaluate shape functions.

        Parameters
        ----------
        functions : np.ndarray
            functions to evaluate, (nPe, nF)
        coord : np.ndarray
            coordinates where functions will be evaluated (nP, dim). Be careful dim must be consistent with function arguments

        Returns
        -------
        np.ndarray
            Evaluated functions (nP, nF, nPe)
        """
        
        nP = coord.shape[0]
        nPe = functions.shape[0]
        nF = functions.shape[1]

        evalFunctions = np.zeros((nP, nF, nPe))

        # for each points
        for p in range(nP):
            # for each functions
            for n, function_nPe in enumerate(functions):
                # for each dimension
                for f in range(nF):
                    # appy the function                     
                    evalFunctions[p, f, n] = function_nPe[f](*coord[p])
                    # * means take all the coordinates 

        return evalFunctions
    
    def __Init_Functions(self, order: int) -> np.ndarray:
        """Methods for initializing functions to be evaluated at gauss points."""
        if self.dim == 1 and self.order < order:
            fonctions = np.array([lambda x: 0]*self.nPe)
        elif self.dim == 2 and self.order < order:
            fonctions = np.array([lambda xi,eta: 0, lambda xi,eta: 0]*self.nPe)
        elif self.dim == 3 and self.order < order:
            fonctions = np.array([lambda x,y,z: 0,lambda x,y,z: 0,lambda x,y,z: 0]*self.nPe)
        return fonctions

    # Here we use legendre polynomials

    @abstractmethod
    def _Ntild(self) -> np.ndarray:
        """Shape functions in local coordinates.\n
        [N1, N2, . . . ,Nn]\n
        (nPe)
        """
        pass    

    def Get_N_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluate shape functions in local coordinates.\n
        [N1, N2, . . . ,Nn]\n
        (pg, nPe)
        """
        if self.dim == 0: return

        Ntild = self._Ntild()
        gauss = self.Get_gauss(matrixType)
        N_pg = _GroupElem._Evaluates_Functions(Ntild, gauss.coord)

        return N_pg

    @abstractmethod
    def _dNtild(self) -> np.ndarray:
        """Shape functions first derivatives in the local coordinates.\n
        [Ni,xi . . . Nn,xi\n
        Ni,eta ... Nn,eta]\n
        (dim, nPe)
        """
        return self.__Init_Functions(1)
    
    def Get_dN_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluate shape functions first derivatives in the local coordinates.\n
        [Ni,xi . . . Nn,xi\n
        Ni,eta ... Nn,eta]\n
        (pg, dim, nPe)
        """
        if self.dim == 0: return

        dNtild = self._dNtild()

        gauss = self.Get_gauss(matrixType)
        dN_pg = _GroupElem._Evaluates_Functions(dNtild, gauss.coord)

        return dN_pg    

    @abstractmethod
    def _ddNtild(self) -> np.ndarray:
        """Shape functions second derivatives in the local coordinates.\n
        [Ni,xi2 . . . Nn,xi2\n
        Ni,eta2 . . . Nn,eta eta]\n
        (dim, nPe)
        """
        return self.__Init_Functions(2)

    def Get_ddN_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluate shape functions second derivatives in the local coordinates.\n
        [Ni,xi2 . . . Nn,xi2\n
        Ni,eta2 . . . Nn,eta2]\n
        (pg, dim, nPe)
        """
        if self.dim == 0: return

        ddNtild = self._ddNtild()

        gauss = self.Get_gauss(matrixType)
        ddN_pg = _GroupElem._Evaluates_Functions(ddNtild, gauss.coord)

        return ddN_pg

    @abstractmethod
    def _dddNtild(self) -> np.ndarray:
        """Shape functions third derivatives in the local coordinates.\n
        [Ni,xi3 . . . Nn,xi3\n
        Ni,eta3 . . . Nn,eta3]\n
        (dim, nPe)
        """
        return self.__Init_Functions(3)

    def Get_dddN_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluate shape functions third derivatives in the local coordinates.\n
        [Ni,xi3 . . . Nn,xi3\n
        Ni,eta3 . . . Nn,eta3]\n
        (pg, dim, nPe)
        """
        if self.elemType == 0: return

        dddNtild = self._dddNtild()

        gauss = self.Get_gauss(matrixType)
        dddN_pg = _GroupElem._Evaluates_Functions(dddNtild, gauss.coord)

        return dddN_pg

    @abstractmethod
    def _ddddNtild(self) -> np.ndarray:
        """Shape functions fourth derivatives in the local coordinates.\n
        [Ni,xi4 . . . Nn,xi4\n
        Ni,eta4 . . . Nn,eta4]
        \n
        (dim, nPe)
        """
        return self.__Init_Functions(4)

    def Get_ddddN_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluate shape functions fourth derivatives in the local coordinates.\n
        [Ni,xi4 . . . Nn,xi4\n
        Ni,eta4 . . . Nn,eta4]
        \n
        (pg, dim, nPe)
        """
        if self.elemType == 0: return

        ddddNtild = self._ddddNtild()

        gauss = self.Get_gauss(matrixType)
        ddddN_pg = _GroupElem._Evaluates_Functions(ddddNtild, gauss.coord)

        return ddddN_pg

    # Beams shapes functions
    # Use hermitian shape functions
    
    def _Nvtild(self) -> np.ndarray:
        """Beam shape functions in the local coordinates.\n
        [phi_i psi_i . . . phi_n psi_n]\n
        (nPe*2)
        """
        pass

    def Get_Nv_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluate beam shape functions in the local coordinates.\n
        [phi_i psi_i . . . phi_n psi_n]\n        
        (pg, nPe*2)
        """
        if self.dim != 1: return

        Nvtild = self._Nvtild()

        gauss = self.Get_gauss(matrixType)
        Nv_pg = _GroupElem._Evaluates_Functions(Nvtild, gauss.coord)

        return Nv_pg
    
    def dNvtild(self) -> np.ndarray:
        """Beam shape functions first derivatives in the local coordinates.\n
        [phi_i,xi psi_i,xi . . . phi_n,xi psi_n,xi]\n
        (nPe*2)
        """
        pass

    def Get_dNv_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluate beam shape functions first derivatives in the local coordinates.\n
        [phi_i,xi psi_i,xi . . . phi_n,xi psi_n,xi]\n
        (pg, nPe*2)
        """
        if self.dim != 1: return

        dNvtild = self.dNvtild()

        gauss = self.Get_gauss(matrixType)
        dNv_pg = _GroupElem._Evaluates_Functions(dNvtild, gauss.coord)

        return dNv_pg
    
    def _ddNvtild(self) -> np.ndarray:
        """Beam shape functions second derivatives in the local coordinates.\n
        [phi_i,xi2 psi_i,xi2 . . . phi_n,xi2 psi_n,xi2]\n
        (nPe*2)
        """
        return 
    
    def Get_ddNv_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluate beam shape functions second derivatives in the local coordinates.\n
        [phi_i,xi2 psi_i,xi2 . . . phi_n,xi2 x psi_n,xi2]\n
        (pg, nPe*2)
        """
        if self.dim != 1: return

        ddNvtild = self._ddNvtild()

        gauss = self.Get_gauss(matrixType)
        ddNv_pg = _GroupElem._Evaluates_Functions(ddNvtild, gauss.coord)

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

    def Get_Nodes_Conditions(self, func) -> np.ndarray:
        """Returns nodes that meet the specified conditions.

        Parameters
        ----------
        func
            Function using the x, y and z nodes coordinates and returning boolean values.

            examples :
            \t lambda x, y, z: (x < 40) & (x > 20) & (y<10) \n
            \t lambda x, y, z: (x == 40) | (x == 50) \n
            \t lambda x, y, z: x >= 0

        Returns
        -------
        np.ndarray
            nodes that meet conditions
        """

        coordo = self.coordo

        xn = coordo[:,0]
        yn = coordo[:,1]
        zn = coordo[:,2]        

        try:
            arrayTest = np.asarray(func(xn, yn, zn))
            if arrayTest.dtype == bool:
                idx = np.where(arrayTest)[0]
                return self.__nodes[idx].copy()
            else:
                print("The function must return a Boolean.")
        except TypeError:
            print("Must provide a 3-parameter function of type lambda x,y,z: ...")
    
    def Get_Nodes_Point(self, point: Point) -> np.ndarray:
        """Returns nodes on the point."""

        coordo = self.coordo

        idx = np.where((coordo[:,0] == point.x) & (coordo[:,1] == point.y) & (coordo[:,2] == point.z))[0]        

        if len(idx) == 0:
            # the previous condition may be too restrictive
            tolerance = 1e-3
            
            # we make sure there is no coordinates = 0
            dec = 10
            decX = np.abs(coordo[:,0].min()) + dec
            decY = np.abs(coordo[:,1].min()) + dec
            decZ = np.abs(coordo[:,2].min()) + dec
            x = point.x + decX
            y = point.y + decY
            z = point.z + decZ
            coordo = coordo + [decX, decY, decZ]
            
            # get errors between coordinates
            errorX = np.abs((coordo[:,0]-x)/coordo[:,0])
            errorY = np.abs((coordo[:,1]-y)/coordo[:,1])
            errorZ = np.abs((coordo[:,2]-z)/coordo[:,2])
            
            idx = np.where((errorX <= tolerance) & (errorY <= tolerance) & (errorZ <= tolerance))[0]

        return self.__nodes[idx].copy()

    def Get_Nodes_Line(self, line: Line) -> np.ndarray:
        """Returns the nodes on the line."""
        
        vectUnitaire = line.unitVector

        coordo = self.coordo

        vect = coordo-line.coordo[0]

        prodScalaire = np.einsum('i,ni-> n', vectUnitaire, vect, optimize='optimal')
        prodVecteur = np.cross(vect, vectUnitaire)
        norm = np.linalg.norm(prodVecteur, axis=1)

        eps = 1e-12

        idx = np.where((norm<eps) & (prodScalaire>=-eps) & (prodScalaire<=line.length+eps))[0]

        return self.__nodes[idx].copy()
    
    def Get_Nodes_Domain(self, domain: Domain) -> np.ndarray:
        """Returns nodes in the domain."""

        coordo = self.coordo

        eps = 1e-12

        idx = np.where( (coordo[:,0] >= domain.pt1.x-eps) & (coordo[:,0] <= domain.pt2.x+eps) &
                        (coordo[:,1] >= domain.pt1.y-eps) & (coordo[:,1] <= domain.pt2.y+eps) &
                        (coordo[:,2] >= domain.pt1.z-eps) & (coordo[:,2] <= domain.pt2.z+eps))[0]
        
        return self.__nodes[idx].copy()

    def Get_Nodes_Circle(self, circle: Circle) -> np.ndarray:
        """Returns the nodes in the circle."""

        coordo = self.coordo

        eps = 1e-12

        idx = np.where(np.sqrt((coordo[:,0]-circle.center.x)**2+(coordo[:,1]-circle.center.y)**2+(coordo[:,2]-circle.center.z)**2)<=circle.diam/2+eps)

        return self.__nodes[idx]

    def Get_Nodes_Cylinder(self, circle: Circle, direction=[0,0,1]) -> np.ndarray:
        """Returns the nodes in the cylinder."""

        coordo = self.coordo
        rotAxis = np.cross(circle.n, direction)
        if np.linalg.norm(rotAxis) <= 1e-12:
            # n == direction
            i = (circle.pt1 - circle.center).coordo
            J = Jacobian_Matrix(i,direction)
        else:
            # n != direction
            # Change base (rotAxis, j, direction)
            R = circle.diam/2
            J = Jacobian_Matrix(rotAxis, direction)
            coordN = np.einsum('ij,nj->ni', np.linalg.inv(J), circle.coordo - circle.center.coordo)
            # Change base (rotAxis, j*cj, direction)
            cj = (R - coordN[:,1].max())/R
            J[:,1] *= cj
        
        eps = 1e-12
        coordo = np.einsum('ij,nj->ni', np.linalg.inv(J), coordo - circle.center.coordo)
        idx = np.where(np.linalg.norm(coordo[:,:2], axis=1) <= circle.diam/2+eps)[0]

        return self.__nodes[idx]
    
    # TODO Get_Nodes_Points
    # use Points.contour also give a normal
    # get all geom contour exept le last one
    # Line -> Plane equation
    # CircleArc -> Cylinder do something like Get_Nodes_Cylinder

    def Set_Nodes_Tag(self, nodes: np.ndarray, tag: str):
        """Add a tag to the nodes

        Parameters
        ----------
        nodes : np.ndarray
            list of nodes
        tag : str
            tag used
        """
        if nodes.size == 0: return
        self.__dict_nodes_tags[tag] = nodes

    @property
    def nodeTags(self) -> list[str]:
        """Returns node tags."""
        return list(self.__dict_nodes_tags.keys())
    
    @property
    def _dict_nodes_tags(self) -> dict[str, np.ndarray]:
        """Dictionary associating tags with nodes."""
        return self.__dict_nodes_tags.copy()

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
    
    @property
    def _dict_elements_tags(self) -> dict[str, np.ndarray]:
        """Dictionary associating tags with elements."""
        return self.__dict_elements_tags.copy()

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
        size = self.Nn * self.dim
        if sol.shape[0] == size:
            sol_e = sol[self.assembly_e]
        elif sol.shape[0] == self.Nn:
            sol_e = sol[self.__connect]
        else:
            raise Exception('Wrong dimension')
        
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

            coordo = self.coordo[self.__connect[elem,0]]

            idx = np.where((coordinates[:,0] == coordo[0]) & (coordinates[:,1] == coordo[1]) & (coordinates[:,2] == coordo[2]))[0]

            return idx

        elif dim == 1:

            coordo = self.coordo

            p1 = self.__connect[elem,0]
            p2 = self.__connect[elem,1]

            vect_i = coordo[p2] - coordo[p1]
            longueur = np.linalg.norm(vect_i)
            vect_i = vect_i / longueur # without normalized doesn't work

            vect_j_n = coordinates - coordo[p1]

            cross_n = np.cross(vect_i, vect_j_n, 0, 1)
            norm_n = np.linalg.norm(cross_n, axis=1)

            dot_n = vect_j_n @ vect_i
            
            idx = np.where((norm_n <= tol) & (dot_n >= -tol) & (dot_n <= longueur+tol))[0]

            return idx
        
        elif dim == 2:            
            
            coordo = self.coordo
            faces = self.faces[:-1]
            nPe = len(faces)
            connectMesh = self.connect[elem, faces]
            coordConnect = coordo[connectMesh]

            # vector calculation
            indexReord = np.append(np.arange(1, nPe), 0)
            # Vectors i for edge segments
            vect_i_b = coordo[connectMesh[indexReord]] - coordo[connectMesh]
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
        
            faces = self.faces
            nbFaces = self.nbFaces
            coordo = self.coordo[self.__connect[elem]]

            if isinstance(self, PRISM6):
                faces = np.array(faces)
                faces = np.array([faces[np.arange(0,4)],
                                  faces[np.arange(4,8)],
                                  faces[np.arange(8,12)],
                                  faces[np.arange(12,15)],
                                  faces[np.arange(15,18)]], dtype=object)
            elif isinstance(self, PRISM15):
                faces = np.array(faces)
                faces = np.array([faces[np.arange(0,8)],
                                  faces[np.arange(8,16)],
                                  faces[np.arange(16,24)],
                                  faces[np.arange(24,30)],
                                  faces[np.arange(30,36)]], dtype=object)
            else:
                faces = np.reshape(faces, (nbFaces,-1))

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

    def Get_Mapping(self, coordinates: np.ndarray, elements=None):
        """Function to return the nodes in the elements, the connectivity and the coordinates (xi, eta) of the points.
        return nodes, connect_e_n, coordoInElem_n"""
        
        if elements is None:
            elements = np.arange(self.Ne, dtype=int)

        assert coordinates.shape[1] == 3, "Must be of dimension (n, 3)."

        return self.__Get_Mapping(coordinates, elements)
    
    def __Get_coordoNear(self, coordinates_n: np.ndarray, coordElem: np.ndarray, dims: np.ndarray) -> np.ndarray:
        """Retrieves indexes in coordinates_n that are within the coordElem's bounds.

        Parameters
        ----------
        coordinates_n : np.ndarray
            coordinates to check
        coordElem : np.ndarray
            element's bounds
        dims : np.ndarray
            (nX, nY, nZ) = np.max(coordinates_n, 0) - np.min(coordinates_n, 0) + 1

        Returns
        -------
        np.ndarray
            indexes in element's bounds. 
        """

        nX, nY, nZ = dims
        
        # If all the coordinates appear the same number of times and the coordinates are of type int, we are on a grid/image.
        testShape = nX * nY - coordinates_n.shape[0] == 0
        usePixel = coordinates_n.dtype==int and testShape and nZ == 1

        if usePixel:
            # here coordinates_n are pixels

            xe = np.arange(np.floor(coordElem[:,0].min()), np.ceil(coordElem[:,0].max()), dtype=int)
            ye = np.arange(np.floor(coordElem[:,1].min()), np.ceil(coordElem[:,1].max()), dtype=int)
            Xe, Ye = np.meshgrid(xe,ye)

            grid_elements_coordinates = np.concatenate(([Ye.ravel()],[Xe.ravel()]))
            idx = np.ravel_multi_index(grid_elements_coordinates, (nY, nX))

            # if something goes wrong, check that the mesh is correctly positioned in the image 
        
        else:

            xn, yn, zn = tuple(coordinates_n.T)
            xe, ye, ze = tuple(coordElem.T)
            
            idx = np.where((xn >= np.min(xe)) & (xn <= np.max(xe)) &
                            (yn >= np.min(ye)) & (yn <= np.max(ye)) & 
                            (zn >= np.min(ze)) & (zn <= np.max(ze)))[0]

        return idx

    def __Get_Mapping(self, coordinates_n: np.ndarray, elements_e: np.ndarray):
        """This function locates coordinates in elements.
        We return the detected coordinates, the connectivity matrix between element and coordinates and the coordinates of these nodes in the reference elements, so that we can evaluate the shape functions."""
        
        # retrieves informations from element group
        dim = self.dim
        inDim = self.inDim
        coordo = self.coordo
        connect = self.connect
        sysCoord_e = self.sysCoord_e # base change matrix for each element
        matrixType = MatrixType.rigi
        invF_e_pg = self.Get_invF_e_pg(matrixType)
        nPg = invF_e_pg.shape[1]
        dN_tild = self._dNtild()
        gaussCoord_e_pg = self.Get_GaussCoordinates_e_p(matrixType)

        # calculates the ratio between jacob min and max to detect if the element is distorted 
        jacobian_e_pg = self.Get_jacobian_e_pg(matrixType, absoluteValues=False)
        diff_e = jacobian_e_pg.min(1) * 1/jacobian_e_pg.max(1)
        error_e = 1 - diff_e # a perfect element has an error max <= 1e-12
        # a distorted element has a max error greater than 0
        useIterative = np.max(error_e) > 1e-12        
        
        # connection matrix containing the nodes used by the elements
        connect_e_n = []
        # node coordinates in the element's reference base
        coordoInElem_n = np.zeros_like(coordinates_n[:,:dim], dtype=float)
        # nodes identified
        nodes = []

        # Calculates the number of times a coordinate appears
        # here dims is a 3d array used in __Get_coordoNear to check if coordinates_n comes from an image
        dims = np.max(coordinates_n, 0) - np.min(coordinates_n, 0) + 1 # faster

        def ResearchFunction(e: int):
    
            # Retrieve element node coordinates
            coordoElem: np.ndarray = coordo[connect[e]]

            # Retrieves indexes in coordinates_n that are within the element's bounds
            idxNearElem = self.__Get_coordoNear(coordinates_n, coordoElem, dims)

            # Returns the index of nodes around the element that meet all conditions
            idxInElem = self.Get_pointsInElem(coordinates_n[idxNearElem], e)

            if idxInElem.size == 0: return

            # nodes that meet all conditions (nodes in the element e)
            nodesInElement = idxNearElem[idxInElem]

            # project coordinates in the basis of the element if dim != inDim
            # its the case when a 2D mesh is in 3D space
            coordoElemBase = coordoElem.copy()
            coordinatesBase_n: np.ndarray = coordinates_n[nodesInElement].copy()
            if dim != inDim:            
                coordoElemBase = coordoElemBase @ sysCoord_e[e]
                coordinatesBase_n = coordinatesBase_n @ sysCoord_e[e]
            
            # now we want to know the coordinates of the nodes in the reference element
            xiOrigin = self.origin # origin of the reference element (xi, eta)            
            x0  = coordoElemBase[0,:dim] # orign of the real element (x,y)
            # xPs = coordinates_n[nodesInElement,:dim] # points coordinates
            xPs = coordinatesBase_n[:,:dim] # points coordinates            
            
            # points coordinates in the reference base
            if useIterative:
                
                def Eval(xi: np.ndarray, xP):
                    dN = _GroupElem._Evaluates_Functions(dN_tild, xi.reshape(1, -1))
                    F = dN[0] @ coordoElemBase[:,:dim]                    
                    J = x0 - xP + (xi - xiOrigin) @ F
                    return J

                xiP = []
                for xP in xPs:
                    res = least_squares(Eval, 0*xP, args=(xP,))
                    tes = Eval(res.x, xP)
                    xiP.append(res.x)

                xiP = np.array(xiP)
            else:
                if nPg == 1:
                    # invF_e_pg is constant in the element
                    xiP: np.ndarray = xiOrigin + (xPs - x0) @ invF_e_pg[e,0]

                else:
                    # If the element have more than 1 integration point, it is necessary to choose the closest integration points. Because invF_e_pg is not constant in the element

                    # for each node detected, we'll calculate its distance from all integration points and see where it's closest
                    dist = np.zeros((xPs.shape[0], nPg))
                    for p in range(nPg):
                        dist[:,p] = np.linalg.norm(xPs - gaussCoord_e_pg[e, p, :dim], axis=1)
                    invMin = invF_e_pg[e, np.argmin(dist, axis=1)]

                    xiP: np.ndarray = xiOrigin + np.einsum('ni,nij->nj',(xPs - x0), invMin, optimize='optimal')
            
            connect_e_n.append(nodesInElement)
            coordoInElem_n[nodesInElement,:] = xiP.copy()
            nodes.extend(nodesInElement)

        [ResearchFunction(e) for e in elements_e]
        
        connect_e_n = np.array(connect_e_n, dtype=object)

        nodes = np.asarray(nodes)

        return nodes, connect_e_n, coordoInElem_n
    
    @abstractproperty
    def origin(self) -> list[int]:
        """Reference element origin coordinates"""
        return [0]

    @abstractproperty
    def triangles(self) -> list[int]:
        """List of indexes to form the triangles of an element that will be used for the 2D trisurf function"""
        pass

    @property
    def segments(self) -> np.ndarray:
        """List of indexes used to construct segments"""
        if self.__dim == 1:
            return np.array([[0, 1]], dtype=int)
        elif self.__dim == 2:
            segments = np.zeros((self.nbCorners, 2), dtype=int)
            segments[:,0] = np.arange(self.nbCorners)
            segments[:,1] = np.append(np.arange(1, self.nbCorners, 1), 0)
            return segments
        elif self.__dim == 3:
            raise Exception("To be defined for 3D element groups.")
    
    @abstractproperty
    def faces(self) -> list[int]:
        """List of indexes to form the faces that make up the element"""
        pass    

class _GroupElem_Factory:

    @staticmethod
    def Get_ElemInFos(gmshId: int) -> tuple[ElemType, int, int, int, int, int]:
        """return elemType, nPe, dim, order, nbFaces, nbCorners\n
        associated with the gmsh id.
        """
        # could be clearer with match but only available with python 3.10
        if gmshId == 15:
            elemType = ElemType.POINT; nPe = 1; dim = 0; order = 0; nbFaces = 0; nbCorners = 0
        elif gmshId == 1:
            elemType = ElemType.SEG2; nPe = 2; dim = 1; order = 1; nbFaces = 0; nbCorners = 2
        elif gmshId == 8:
            elemType = ElemType.SEG3; nPe = 3; dim = 1; order = 2; nbFaces = 0; nbCorners = 2
        elif gmshId == 26:
            elemType = ElemType.SEG4; nPe = 4; dim = 1; order = 3; nbFaces = 0; nbCorners = 2
        # elif gmshId == 27:
        #     elemType = ElemType.SEG5; nPe = 5; dim = 1; order = 4; nbFaces = 0; nbCorners = 2
        elif gmshId == 2:
            elemType = ElemType.TRI3; nPe = 3; dim = 2; order = 2; nbFaces = 1; nbCorners = 3
        elif gmshId == 9:
            elemType = ElemType.TRI6; nPe = 6; dim = 2; order = 2; nbFaces = 1; nbCorners = 3
        elif gmshId == 21:
            elemType = ElemType.TRI10; nPe = 10; dim = 2; order = 3; nbFaces = 1; nbCorners = 3
        # elif gmshId == 23:
        #     elemType = ElemType.TRI15; nPe = 15; dim = 2; order = 4; nbFaces = 1; nbCorners = 3
        elif gmshId == 3:
            elemType = ElemType.QUAD4; nPe = 4; dim = 2; order = 1; nbFaces = 1; nbCorners = 4
        elif gmshId == 16:
            elemType = ElemType.QUAD8; nPe = 8; dim = 2; order = 2; nbFaces = 1; nbCorners = 4
        # elif gmshId == 10:
        #     elemType = ElemType.QUAD9; nPe = 9; dim = 2; order = 3; nbFaces = 1; nbCorners = 4
        elif gmshId == 4:
            elemType = ElemType.TETRA4; nPe = 4; dim = 3; order = 1; nbFaces = 4; nbCorners = 4
        elif gmshId == 11:
            elemType = ElemType.TETRA10; nPe = 10; dim = 3; order = 2; nbFaces = 4; nbCorners = 4
        elif gmshId == 5:
            elemType = ElemType.HEXA8; nPe = 8; dim = 3; order = 1; nbFaces = 6; nbCorners = 8
        elif gmshId == 17:
            elemType = ElemType.HEXA20; nPe = 20; dim = 3; order = 2; nbFaces = 6; nbCorners = 8
        elif gmshId == 6:
            elemType = ElemType.PRISM6; nPe = 6; dim = 3; order = 1; nbFaces = 5; nbCorners = 6
        elif gmshId == 18:
            elemType = ElemType.PRISM15; nPe = 15; dim = 3; order = 2; nbFaces = 5; nbCorners = 6
        # elif gmshId == 13:
        #     elemType = ElemType.PRISM18; nPe = 18; dim = 3; order = 2; nbFaces = 5; nbCorners = 6
        # elif gmshId == 7:
        #     elemType = ElemType.PYRA5; nPe = 5; dim = 3; order = 1; nbFaces = 5; nbCorners = 5
        # elif gmshId == 19:
        #     elemType = ElemType.PYRA13; nPe = 13; dim = 3; order = 2; nbFaces = 5; nbCorners = 5
        # elif gmshId == 14:
        #     elemType = ElemType.PYRA14; nPe = 14; dim = 3; order = 2; nbFaces = 5; nbCorners = 5
        else: 
            raise Exception("Element type unknown")
            
        return elemType, nPe, dim, order, nbFaces, nbCorners
    
    @staticmethod
    def Create(gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray) -> _GroupElem:
        """Create an element group
        
        Parameters
        ----------
        gmshId : int
            id gmsh
        connect : np.ndarray
            connection matrix storing nodes for each element (Ne, nPe)
        coordoGlob : np.ndarray
            node coordinates
        nodes : np.ndarray
            nodes used by the element group
        
        Returns
        -------
        GroupeElem
            the element group
        """

        params = (gmshId, connect, coordoGlob, nodes)

        elemType = _GroupElem_Factory.Get_ElemInFos(gmshId)[0]
        
        if elemType == ElemType.POINT:
            return POINT(*params)
        elif elemType == ElemType.SEG2:
            return SEG2(*params)
        elif elemType == ElemType.SEG3:
            return SEG3(*params)
        elif elemType == ElemType.SEG4:
            return SEG4(*params)
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


class POINT(_GroupElem):
    
    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def origin(self) -> list[int]:
        return super().origin

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def faces(self) -> list[int]:
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

class SEG2(_GroupElem):    
    #      v
    #      ^
    #      |
    #      |
    # 0----+----1 --> u

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def origin(self) -> list[int]:
        return [-1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles
    
    @property
    def faces(self) -> list[int]:
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

class SEG3(_GroupElem):
    #      v
    #      ^
    #      |
    #      |
    # 0----2----1 --> u

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def origin(self) -> list[int]:
        return [-1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles
    
    @property
    def faces(self) -> list[int]:
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

class SEG4(_GroupElem):
    #       v
    #       ^
    #       |
    #       |
    # 0---2-+-3---1 --> u

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def origin(self) -> list[int]:
        return [-1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def faces(self) -> list[int]:
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

class TRI3(_GroupElem):
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
    def origin(self) -> list[int]:
        return super().origin

    @property
    def triangles(self) -> list[int]:
        return [0,1,2]

    @property
    def faces(self) -> list[int]:
        return [0,1,2,0]    

    def _Ntild(self) -> np.ndarray:

        N1t = lambda xi,eta: 1-xi-eta
        N2t = lambda xi,eta: xi
        N3t = lambda xi,eta: eta
        
        Ntild = np.array([N1t, N2t, N3t]).reshape(-1,1)

        return Ntild

    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda xi,eta: -1, lambda xi,eta: -1]
        dN2t = [lambda xi,eta: 1,  lambda xi,eta: 0]
        dN3t = [lambda xi,eta: 0,  lambda xi,eta: 1]

        dNtild = np.array([dN1t, dN2t, dN3t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:
        return super()._ddNtild()

    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()

    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()


class TRI6(_GroupElem):
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
    def origin(self) -> list[int]:
        return super().origin

    @property
    def triangles(self) -> list[int]:
        return [0,3,5,3,1,4,5,4,2,3,4,5]

    @property
    def faces(self) -> list[int]:
        return [0,3,1,4,2,5,0]

    def _Ntild(self) -> np.ndarray:

        N1t = lambda xi,eta: -(1-xi-eta)*(1-2*(1-xi-eta))
        N2t = lambda xi,eta: -xi*(1-2*xi)
        N3t = lambda xi,eta: -eta*(1-2*eta)
        N4t = lambda xi,eta: 4*xi*(1-xi-eta)
        N5t = lambda xi,eta: 4*xi*eta
        N6t = lambda xi,eta: 4*eta*(1-xi-eta)
        
        Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t]).reshape(-1, 1)

        return Ntild

    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda xi,eta: 4*xi+4*eta-3,  lambda xi,eta: 4*xi+4*eta-3]
        dN2t = [lambda xi,eta: 4*xi-1,        lambda xi,eta: 0]
        dN3t = [lambda xi,eta: 0,              lambda xi,eta: 4*eta-1]
        dN4t = [lambda xi,eta: 4-8*xi-4*eta,  lambda xi,eta: -4*xi]
        dN5t = [lambda xi,eta: 4*eta,          lambda xi,eta: 4*xi]
        dN6t = [lambda xi,eta: -4*eta,         lambda xi,eta: 4-4*xi-8*eta]
        
        dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:
        ddN1t = [lambda xi,eta: 4,  lambda xi,eta: 4]
        ddN2t = [lambda xi,eta: 4,  lambda xi,eta: 0]
        ddN3t = [lambda xi,eta: 0,  lambda xi,eta: 4]
        ddN4t = [lambda xi,eta: -8, lambda xi,eta: 0]
        ddN5t = [lambda xi,eta: 0,  lambda xi,eta: 0]
        ddN6t = [lambda xi,eta: 0,  lambda xi,eta: -8]
        
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

class TRI10(_GroupElem):
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
    def origin(self) -> list[int]:
        return super().origin

    @property
    def triangles(self) -> list[int]:
        return list(np.array([10,1,4,10,4,5,10,5,6,10,6,7,10,7,8,10,8,9,10,9,1,2,5,6,3,7,8])-1)
    
    @property
    def faces(self) -> list[int]:
        return [0,3,4,1,5,6,2,7,8,0]

    def _Ntild(self) -> np.ndarray:

        N1t = lambda xi, eta : -4.5*xi**3 + -4.5*eta**3 + -13.5*xi**2*eta + -13.5*xi*eta**2 + 9.0*xi**2 + 9.0*eta**2 + 18.0*xi*eta + -5.5*xi + -5.5*eta + 1.0
        N2t = lambda xi, eta : 4.5*xi**3 + 0.0*eta**3 + -1.093e-15*xi**2*eta + -8.119e-16*xi*eta**2 + -4.5*xi**2 + 0.0*eta**2 + 1.124e-15*xi*eta + 1.0*xi + 0.0*eta + 0.0
        N3t = lambda xi, eta : 0.0*xi**3 + 4.5*eta**3 + -3.747e-16*xi**2*eta + 2.998e-15*xi*eta**2 + 0.0*xi**2 + -4.5*eta**2 + -7.494e-16*xi*eta + 0.0*xi + 1.0*eta + 0.0
        N4t = lambda xi, eta : 13.5*xi**3 + 0.0*eta**3 + 27.0*xi**2*eta + 13.5*xi*eta**2 + -22.5*xi**2 + 0.0*eta**2 + -22.5*xi*eta + 9.0*xi + 0.0*eta + 0.0
        N5t = lambda xi, eta : -13.5*xi**3 + 0.0*eta**3 + -13.5*xi**2*eta + -4.247e-15*xi*eta**2 + 18.0*xi**2 + 0.0*eta**2 + 4.5*xi*eta + -4.5*xi + 0.0*eta + 0.0
        N6t = lambda xi, eta : 0.0*xi**3 + 0.0*eta**3 + 13.5*xi**2*eta + 1.049e-14*xi*eta**2 + 0.0*xi**2 + 0.0*eta**2 + -4.5*xi*eta + 0.0*xi + 0.0*eta + 0.0
        N7t = lambda xi, eta : 0.0*xi**3 + 0.0*eta**3 + 0.0*xi**2*eta + 13.5*xi*eta**2 + 0.0*xi**2 + 0.0*eta**2 + -4.5*xi*eta + 0.0*xi + 0.0*eta + 0.0
        N8t = lambda xi, eta : 0.0*xi**3 + -13.5*eta**3 + -1.499e-15*xi**2*eta + -13.5*xi*eta**2 + 0.0*xi**2 + 18.0*eta**2 + 4.5*xi*eta + 0.0*xi + -4.5*eta + 0.0
        N9t = lambda xi, eta : 0.0*xi**3 + 13.5*eta**3 + 13.5*xi**2*eta + 27.0*xi*eta**2 + 0.0*xi**2 + -22.5*eta**2 + -22.5*xi*eta + 0.0*xi + 9.0*eta + 0.0
        N10t = lambda xi, eta : 0.0*xi**3 + 0.0*eta**3 + -27.0*xi**2*eta + -27.0*xi*eta**2 + 0.0*xi**2 + 0.0*eta**2 + 27.0*xi*eta + 0.0*xi + 0.0*eta + 0.0
        
        Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t, N9t, N10t]).reshape(-1, 1)

        return Ntild

    def _dNtild(self) -> np.ndarray:

        N1_xi = lambda xi, eta : -13.5*xi**2 + -27.0*xi*eta + -13.5*eta**2 + 18.0*xi + 18.0*eta + -5.5
        N2_xi = lambda xi, eta : 13.5*xi**2 + -2.186e-15*xi*eta + -8.119e-16*eta**2 + -9.0*xi + 1.124e-15*eta + 1.0
        N3_xi = lambda xi, eta : 0.0*xi**2 + -7.494e-16*xi*eta + 2.998e-15*eta**2 + 0.0*xi + -7.494e-16*eta + 0.0
        N4_xi = lambda xi, eta : 40.5*xi**2 + 54.0*xi*eta + 13.5*eta**2 + -45.0*xi + -22.5*eta + 9.0
        N5_xi = lambda xi, eta : -40.5*xi**2 + -27.0*xi*eta + -4.247e-15*eta**2 + 36.0*xi + 4.5*eta + -4.5
        N6_xi = lambda xi, eta : 0.0*xi**2 + 27.0*xi*eta + 1.049e-14*eta**2 + 0.0*xi + -4.5*eta + 0.0
        N7_xi = lambda xi, eta : 0.0*xi**2 + 0.0*xi*eta + 13.5*eta**2 + 0.0*xi + -4.5*eta + 0.0
        N8_xi = lambda xi, eta : 0.0*xi**2 + -2.998e-15*xi*eta + -13.5*eta**2 + 0.0*xi + 4.5*eta + 0.0
        N9_xi = lambda xi, eta : 0.0*xi**2 + 27.0*xi*eta + 27.0*eta**2 + 0.0*xi + -22.5*eta + 0.0
        N10_xi = lambda xi, eta : 0.0*xi**2 + -54.0*xi*eta + -27.0*eta**2 + 0.0*xi + 27.0*eta + 0.0

        N1_eta = lambda xi, eta : -13.5*eta**2 + -13.5*xi**2 + -27.0*xi*eta + 18.0*eta + 18.0*xi + -5.5
        N2_eta = lambda xi, eta : 0.0*eta**2 + -1.093e-15*xi**2 + -1.624e-15*xi*eta + 0.0*eta + 1.124e-15*xi + 0.0
        N3_eta = lambda xi, eta : 13.5*eta**2 + -3.747e-16*xi**2 + 5.995e-15*xi*eta + -9.0*eta + -7.494e-16*xi + 1.0
        N4_eta = lambda xi, eta : 0.0*eta**2 + 27.0*xi**2 + 27.0*xi*eta + 0.0*eta + -22.5*xi + 0.0
        N5_eta = lambda xi, eta : 0.0*eta**2 + -13.5*xi**2 + -8.493e-15*xi*eta + 0.0*eta + 4.5*xi + 0.0
        N6_eta = lambda xi, eta : 0.0*eta**2 + 13.5*xi**2 + 2.098e-14*xi*eta + 0.0*eta + -4.5*xi + 0.0
        N7_eta = lambda xi, eta : 0.0*eta**2 + 0.0*xi**2 + 27.0*xi*eta + 0.0*eta + -4.5*xi + 0.0
        N8_eta = lambda xi, eta : -40.5*eta**2 + -1.499e-15*xi**2 + -27.0*xi*eta + 36.0*eta + 4.5*xi + -4.5
        N9_eta = lambda xi, eta : 40.5*eta**2 + 13.5*xi**2 + 54.0*xi*eta + -45.0*eta + -22.5*xi + 9.0
        N10_eta = lambda xi, eta : 0.0*eta**2 + -27.0*xi**2 + -54.0*xi*eta + 0.0*eta + 27.0*xi + 0.0

        dN1t = [N1_xi, N1_eta]
        dN2t = [N2_xi, N2_eta]
        dN3t = [N3_xi, N3_eta]
        dN4t = [N4_xi, N4_eta]
        dN5t = [N5_xi, N5_eta]
        dN6t = [N6_xi, N6_eta]
        dN7t = [N7_xi, N7_eta]
        dN8t = [N8_xi, N8_eta]
        dN9t = [N9_xi, N9_eta]
        dN10t = [N10_xi, N10_eta]

        dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t, dN9t, dN10t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:

        N1_xi2 = lambda xi, eta : -27.0*xi + -27.0*eta + 18.0
        N2_xi2 = lambda xi, eta : 27.0*xi + -2.186e-15*eta + -9.0
        N3_xi2 = lambda xi, eta : 0.0*xi + -7.494e-16*eta + 0.0
        N4_xi2 = lambda xi, eta : 81.0*xi + 54.0*eta + -45.0
        N5_xi2 = lambda xi, eta : -81.0*xi + -27.0*eta + 36.0
        N6_xi2 = lambda xi, eta : 0.0*xi + 27.0*eta + 0.0
        N7_xi2 = lambda xi, eta : 0.0*xi + 0.0*eta + 0.0
        N8_xi2 = lambda xi, eta : 0.0*xi + -2.998e-15*eta + 0.0
        N9_xi2 = lambda xi, eta : 0.0*xi + 27.0*eta + 0.0
        N10_xi2 = lambda xi, eta : 0.0*xi + -54.0*eta + 0.0

        N1_eta2 = lambda xi, eta : -27.0*eta + -27.0*xi + 18.0
        N2_eta2 = lambda xi, eta : 0.0*eta + -1.624e-15*xi + 0.0
        N3_eta2 = lambda xi, eta : 27.0*eta + 5.995e-15*xi + -9.0
        N4_eta2 = lambda xi, eta : 0.0*eta + 27.0*xi + 0.0
        N5_eta2 = lambda xi, eta : 0.0*eta + -8.493e-15*xi + 0.0
        N6_eta2 = lambda xi, eta : 0.0*eta + 2.098e-14*xi + 0.0
        N7_eta2 = lambda xi, eta : 0.0*eta + 27.0*xi + 0.0
        N8_eta2 = lambda xi, eta : -81.0*eta + -27.0*xi + 36.0
        N9_eta2 = lambda xi, eta : 81.0*eta + 54.0*xi + -45.0
        N10_eta2 = lambda xi, eta : 0.0*eta + -54.0*xi + 0.0

        ddN1t = [N1_xi2, N1_eta2]
        ddN2t = [N2_xi2, N2_eta2]
        ddN3t = [N3_xi2, N3_eta2]
        ddN4t = [N4_xi2, N4_eta2]
        ddN5t = [N5_xi2, N5_eta2]
        ddN6t = [N6_xi2, N6_eta2]
        ddN7t = [N7_xi2, N7_eta2]
        ddN8t = [N8_xi2, N8_eta2]
        ddN9t = [N9_xi2, N9_eta2]
        ddN10t = [N10_xi2, N10_eta2]

        ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t, ddN5t, ddN6t, ddN7t, ddN8t, ddN9t, ddN10t])

        return ddNtild

    def _dddNtild(self) -> np.ndarray:
        
        N1_xi3 = lambda xi, eta : -27.0
        N2_xi3 = lambda xi, eta : 27.0
        N3_xi3 = lambda xi, eta : 0.0
        N4_xi3 = lambda xi, eta : 81.0
        N5_xi3 = lambda xi, eta : -81.0
        N6_xi3 = lambda xi, eta : 0.0
        N7_xi3 = lambda xi, eta : 0.0
        N8_xi3 = lambda xi, eta : 0.0
        N9_xi3 = lambda xi, eta : 0.0
        N10_xi3 = lambda xi, eta : 0.0

        N1_eta3 = lambda xi, eta : -27.0
        N2_eta3 = lambda xi, eta : 0.0
        N3_eta3 = lambda xi, eta : 27.0
        N4_eta3 = lambda xi, eta : 0.0
        N5_eta3 = lambda xi, eta : 0.0
        N6_eta3 = lambda xi, eta : 0.0
        N7_eta3 = lambda xi, eta : 0.0
        N8_eta3 = lambda xi, eta : -81.0
        N9_eta3 = lambda xi, eta : 81.0
        N10_eta3 = lambda xi, eta : 0.0

        dddN1t = [N1_xi3, N1_eta3]
        dddN2t = [N2_xi3, N2_eta3]
        dddN3t = [N3_xi3, N3_eta3]
        dddN4t = [N4_xi3, N4_eta3]
        dddN5t = [N5_xi3, N5_eta3]
        dddN6t = [N6_xi3, N6_eta3]
        dddN7t = [N7_xi3, N7_eta3]
        dddN8t = [N8_xi3, N8_eta3]
        dddN9t = [N9_xi3, N9_eta3]
        dddN10t = [N10_xi3, N10_eta3]

        dddNtild = np.array([dddN1t, dddN2t, dddN3t, dddN4t, dddN5t, dddN6t, dddN7t, dddN8t, dddN9t, dddN10t])

        return dddNtild

    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()

class QUAD4(_GroupElem):
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
    def origin(self) -> list[int]:
        return [-1, -1]

    @property
    def triangles(self) -> list[int]:
        return [0,1,3,1,2,3]

    @property
    def faces(self) -> list[int]:
        return [0,1,2,3,0]

    def _Ntild(self) -> np.ndarray:

        N1t = lambda xi,eta: (1-xi)*(1-eta)/4
        N2t = lambda xi,eta: (1+xi)*(1-eta)/4
        N3t = lambda xi,eta: (1+xi)*(1+eta)/4
        N4t = lambda xi,eta: (1-xi)*(1+eta)/4
        
        Ntild = np.array([N1t, N2t, N3t, N4t]).reshape(-1, 1)

        return Ntild

    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda xi,eta: (eta-1)/4,  lambda xi,eta: (xi-1)/4]
        dN2t = [lambda xi,eta: (1-eta)/4,  lambda xi,eta: (-xi-1)/4]
        dN3t = [lambda xi,eta: (1+eta)/4,  lambda xi,eta: (1+xi)/4]
        dN4t = [lambda xi,eta: (-eta-1)/4, lambda xi,eta: (1-xi)/4]
        
        dNtild = np.array([dN1t, dN2t, dN3t, dN4t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:
        return super()._ddNtild()
    
    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()
    
    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()

    

class QUAD8(_GroupElem):
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
    def origin(self) -> list[int]:
        return [-1, -1]

    @property
    def triangles(self) -> list[int]:
        return [4,5,7,5,6,7,0,4,7,4,1,5,5,2,6,6,3,7]

    @property
    def faces(self) -> list[int]:
        return [0,4,1,5,2,6,3,7,0]

    def _Ntild(self) -> np.ndarray:

        N1t = lambda xi,eta: (1-xi)*(1-eta)*(-1-xi-eta)/4
        N2t = lambda xi,eta: (1+xi)*(1-eta)*(-1+xi-eta)/4
        N3t = lambda xi,eta: (1+xi)*(1+eta)*(-1+xi+eta)/4
        N4t = lambda xi,eta: (1-xi)*(1+eta)*(-1-xi+eta)/4
        N5t = lambda xi,eta: (1-xi**2)*(1-eta)/2
        N6t = lambda xi,eta: (1+xi)*(1-eta**2)/2
        N7t = lambda xi,eta: (1-xi**2)*(1+eta)/2
        N8t = lambda xi,eta: (1-xi)*(1-eta**2)/2
        
        Ntild =  np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t]).reshape(-1, 1)

        return Ntild
    
    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda xi,eta: (1-eta)*(2*xi+eta)/4,      lambda xi,eta: (1-xi)*(xi+2*eta)/4]
        dN2t = [lambda xi,eta: (1-eta)*(2*xi-eta)/4,      lambda xi,eta: -(1+xi)*(xi-2*eta)/4]
        dN3t = [lambda xi,eta: (1+eta)*(2*xi+eta)/4,      lambda xi,eta: (1+xi)*(xi+2*eta)/4]
        dN4t = [lambda xi,eta: -(1+eta)*(-2*xi+eta)/4,    lambda xi,eta: (1-xi)*(-xi+2*eta)/4]
        dN5t = [lambda xi,eta: -xi*(1-eta),               lambda xi,eta: -(1-xi**2)/2]
        dN6t = [lambda xi,eta: (1-eta**2)/2,               lambda xi,eta: -eta*(1+xi)]
        dN7t = [lambda xi,eta: -xi*(1+eta),               lambda xi,eta: (1-xi**2)/2]
        dN8t = [lambda xi,eta: -(1-eta**2)/2,              lambda xi,eta: -eta*(1-xi)]
                        
        dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:

        ddN1t = [lambda xi,eta: (1-eta)/2,  lambda xi,eta: (1-xi)/2]
        ddN2t = [lambda xi,eta: (1-eta)/2,  lambda xi,eta: (1+xi)/2]
        ddN3t = [lambda xi,eta: (1+eta)/2,  lambda xi,eta: (1+xi)/2]
        ddN4t = [lambda xi,eta: (1+eta)/2,  lambda xi,eta: (1-xi)/2]
        ddN5t = [lambda xi,eta: -1+eta,     lambda xi,eta: 0]
        ddN6t = [lambda xi,eta: 0,          lambda xi,eta: -1-xi]
        ddN7t = [lambda xi,eta: -1-eta,     lambda xi,eta: 0]
        ddN8t = [lambda xi,eta: 0,          lambda xi,eta: -1+xi]
                        
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

class TETRA4(_GroupElem):
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
    def origin(self) -> list[int]:
        return super().origin

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def faces(self) -> list[int]:
        return [0,1,2,0,3,1,0,2,3,1,3,2]
    
    @property
    def segments(self) -> np.ndarray:
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

    

class TETRA10(_GroupElem):
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
    def origin(self) -> list[int]:
        return super().origin

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def faces(self) -> list[int]:        
        return [0,4,1,5,2,6,0,7,3,9,1,4,0,6,2,8,3,7,1,9,3,8,2,5]
    
    @property
    def segments(self) -> np.ndarray:
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

    

class HEXA8(_GroupElem):
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
    def origin(self) -> list[int]:
        return [-1, -1, -1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def faces(self) -> list[int]:
        return [0,1,2,3,0,4,5,1,0,3,7,4,6,7,3,2,6,2,1,5,6,5,4,7]
    
    @property
    def segments(self) -> np.ndarray:
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

    

class HEXA20(_GroupElem):
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
    def origin(self) -> list[int]:
        return [-1, -1, -1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def faces(self) -> list[int]:
        return [0,8,1,11,2,13,3,9,
                0,10,4,16,5,12,1,8,
                0,9,3,15,7,17,4,10,
                6,19,7,15,3,13,2,14,
                6,14,2,11,1,12,5,18,
                6,18,5,16,4,17,7,19]
    
    @property
    def segments(self) -> np.ndarray:
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

    

class PRISM6(_GroupElem):
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
    def origin(self) -> list[int]:
        return [0, 0, -1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def faces(self) -> list[int]:
        return [0,3,4,1,
                0,2,5,3,
                1,4,5,2,
                3,5,4,3,
                0,1,2,0]
    
    @property
    def segments(self) -> np.ndarray:
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

class PRISM15(_GroupElem):
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
    def origin(self) -> list[int]:
        return [0, 0, -1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def faces(self) -> list[int]:
        return [0,8,3,12,4,10,1,6,
                0,7,2,11,5,13,3,8,
                1,10,4,14,5,11,2,9,
                3,13,5,14,4,12,3,3,
                0,6,1,9,2,7,0,0]
    
    @property
    def segments(self) -> np.ndarray:
        return np.array([[0,1],[1,2],[2,0],[3,4],[4,5],[5,3],[0,3],[1,4],[2,5]])

    def _Ntild(self) -> np.ndarray:

        N1t = lambda x,y,z: -1.0*x**2*z + -1.0*y**2*z + -0.5*z**2*x + -0.5*z**2*y + -2.0*x*y*z + 1.0*x**2 + 1.0*y**2 + 0.5*z**2 + 1.5*x*z + 1.5*y*z + -1.0*x + -1.0*y + -0.5*z + 2.0*x*y + 0.0
        N2t = lambda x,y,z: -1.0*x**2*z + 0.0*y**2*z + 0.5*z**2*x + 0.0*z**2*y + 0.0*x*y*z + 1.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.5*x*z + 0.0*y*z + -1.0*x + 0.0*y + 0.0*z + 0.0*x*y + 0.0
        N3t = lambda x,y,z: 0.0*x**2*z + -1.0*y**2*z + 0.0*z**2*x + 0.5*z**2*y + 0.0*x*y*z + 0.0*x**2 + 1.0*y**2 + 0.0*z**2 + 0.0*x*z + 0.5*y*z + 0.0*x + -1.0*y + 0.0*z + 0.0*x*y + 0.0
        N4t = lambda x,y,z: 1.0*x**2*z + 1.0*y**2*z + -0.5*z**2*x + -0.5*z**2*y + 2.0*x*y*z + 1.0*x**2 + 1.0*y**2 + 0.5*z**2 + -1.5*x*z + -1.5*y*z + -1.0*x + -1.0*y + 0.5*z + 2.0*x*y + 0.0
        N5t = lambda x,y,z: 1.0*x**2*z + 0.0*y**2*z + 0.5*z**2*x + 0.0*z**2*y + 0.0*x*y*z + 1.0*x**2 + 0.0*y**2 + 0.0*z**2 + -0.5*x*z + 0.0*y*z + -1.0*x + 0.0*y + 0.0*z + 0.0*x*y + 0.0
        N6t = lambda x,y,z: 0.0*x**2*z + 1.0*y**2*z + 0.0*z**2*x + 0.5*z**2*y + 0.0*x*y*z + 0.0*x**2 + 1.0*y**2 + 0.0*z**2 + 0.0*x*z + -0.5*y*z + 0.0*x + -1.0*y + 0.0*z + 0.0*x*y + 0.0
        N7t = lambda x,y,z: 2.0*x**2*z + 0.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + 2.0*x*y*z + -2.0*x**2 + 0.0*y**2 + 0.0*z**2 + -2.0*x*z + 0.0*y*z + 2.0*x + 0.0*y + 0.0*z + -2.0*x*y + 0.0
        N8t = lambda x,y,z: 0.0*x**2*z + 2.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + 2.0*x*y*z + 0.0*x**2 + -2.0*y**2 + 0.0*z**2 + 0.0*x*z + -2.0*y*z + 0.0*x + 2.0*y + 0.0*z + -2.0*x*y + 0.0
        N9t = lambda x,y,z: 0.0*x**2*z + 0.0*y**2*z + 1.0*z**2*x + 1.0*z**2*y + 0.0*x*y*z + 0.0*x**2 + 0.0*y**2 + -1.0*z**2 + 0.0*x*z + 0.0*y*z + -1.0*x + -1.0*y + 0.0*z + 0.0*x*y + 1.0
        N10t = lambda x,y,z: 0.0*x**2*z + 0.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + -2.0*x*y*z + 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*y*z + 0.0*x + 0.0*y + 0.0*z + 2.0*x*y + 0.0
        N11t = lambda x,y,z: 0.0*x**2*z + 0.0*y**2*z + -1.0*z**2*x + 0.0*z**2*y + 0.0*x*y*z + 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*y*z + 1.0*x + 0.0*y + 0.0*z + 0.0*x*y + 0.0
        N12t = lambda x,y,z: 0.0*x**2*z + 0.0*y**2*z + 0.0*z**2*x + -1.0*z**2*y + 0.0*x*y*z + 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*y*z + 0.0*x + 1.0*y + 0.0*z + 0.0*x*y + 0.0
        N13t = lambda x,y,z: -2.0*x**2*z + 0.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + -2.0*x*y*z + -2.0*x**2 + 0.0*y**2 + 0.0*z**2 + 2.0*x*z + 0.0*y*z + 2.0*x + 0.0*y + 0.0*z + -2.0*x*y + 0.0
        N14t = lambda x,y,z: 0.0*x**2*z + -2.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + -2.0*x*y*z + 0.0*x**2 + -2.0*y**2 + 0.0*z**2 + 0.0*x*z + 2.0*y*z + 0.0*x + 2.0*y + 0.0*z + -2.0*x*y + 0.0
        N15t = lambda x,y,z: 0.0*x**2*z + 0.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + 2.0*x*y*z + 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*y*z + 0.0*x + 0.0*y + 0.0*z + 2.0*x*y + 0.0

        Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t, N9t, N10t, N11t, N12t, N13t, N14t, N15t]).reshape(-1, 1)

        return Ntild
    
    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda x,y,z: -2.0*x*z + -0.5*z**2 + -2.0*y*z + 2.0*x + 1.5*z + -1.0 + 2.0*y,
                lambda x,y,z: -2.0*y*z + -0.5*z**2 + -2.0*x*z + 2.0*y + 1.5*z + -1.0 + 2.0*x,
                lambda x,y,z: -1.0*x**2 + -1.0*y**2 + -1.0*z*x + -1.0*z*y + -2.0*x*y + 1.0*z + 1.5*x + 1.5*y + -0.5]
        dN2t = [lambda x,y,z: -2.0*x*z + 0.5*z**2 + 0.0*y*z + 2.0*x + 0.5*z + -1.0 + 0.0*y,
                lambda x,y,z: 0.0*y*z + 0.0*z**2 + 0.0*x*z + 0.0*y + 0.0*z + 0.0 + 0.0*x,
                lambda x,y,z: -1.0*x**2 + 0.0*y**2 + 1.0*z*x + 0.0*z*y + 0.0*x*y + 0.0*z + 0.5*x + 0.0*y + 0.0]
        dN3t = [lambda x,y,z: 0.0*x*z + 0.0*z**2 + 0.0*y*z + 0.0*x + 0.0*z + 0.0 + 0.0*y,
                lambda x,y,z: -2.0*y*z + 0.5*z**2 + 0.0*x*z + 2.0*y + 0.5*z + -1.0 + 0.0*x,
                lambda x,y,z: 0.0*x**2 + -1.0*y**2 + 0.0*z*x + 1.0*z*y + 0.0*x*y + 0.0*z + 0.0*x + 0.5*y + 0.0]
        dN4t = [lambda x,y,z: 2.0*x*z + -0.5*z**2 + 2.0*y*z + 2.0*x + -1.5*z + -1.0 + 2.0*y,
                lambda x,y,z: 2.0*y*z + -0.5*z**2 + 2.0*x*z + 2.0*y + -1.5*z + -1.0 + 2.0*x,
                lambda x,y,z: 1.0*x**2 + 1.0*y**2 + -1.0*z*x + -1.0*z*y + 2.0*x*y + 1.0*z + -1.5*x + -1.5*y + 0.5]
        dN5t = [lambda x,y,z: 2.0*x*z + 0.5*z**2 + 0.0*y*z + 2.0*x + -0.5*z + -1.0 + 0.0*y,
                lambda x,y,z: 0.0*y*z + 0.0*z**2 + 0.0*x*z + 0.0*y + 0.0*z + 0.0 + 0.0*x,
                lambda x,y,z: 1.0*x**2 + 0.0*y**2 + 1.0*z*x + 0.0*z*y + 0.0*x*y + 0.0*z + -0.5*x + 0.0*y + 0.0]
        dN6t = [lambda x,y,z: 0.0*x*z + 0.0*z**2 + 0.0*y*z + 0.0*x + 0.0*z + 0.0 + 0.0*y,
                lambda x,y,z: 2.0*y*z + 0.5*z**2 + 0.0*x*z + 2.0*y + -0.5*z + -1.0 + 0.0*x,
                lambda x,y,z: 0.0*x**2 + 1.0*y**2 + 0.0*z*x + 1.0*z*y + 0.0*x*y + 0.0*z + 0.0*x + -0.5*y + 0.0]
        dN7t = [lambda x,y,z: 4.0*x*z + 0.0*z**2 + 2.0*y*z + -4.0*x + -2.0*z + 2.0 + -2.0*y,
                lambda x,y,z: 0.0*y*z + 0.0*z**2 + 2.0*x*z + 0.0*y + 0.0*z + 0.0 + -2.0*x,
                lambda x,y,z: 2.0*x**2 + 0.0*y**2 + 0.0*z*x + 0.0*z*y + 2.0*x*y + 0.0*z + -2.0*x + 0.0*y + 0.0]
        dN8t = [lambda x,y,z: 0.0*x*z + 0.0*z**2 + 2.0*y*z + 0.0*x + 0.0*z + 0.0 + -2.0*y,
                lambda x,y,z: 4.0*y*z + 0.0*z**2 + 2.0*x*z + -4.0*y + -2.0*z + 2.0 + -2.0*x,
                lambda x,y,z: 0.0*x**2 + 2.0*y**2 + 0.0*z*x + 0.0*z*y + 2.0*x*y + 0.0*z + 0.0*x + -2.0*y + 0.0]
        dN9t = [lambda x,y,z: 0.0*x*z + 1.0*z**2 + 0.0*y*z + 0.0*x + 0.0*z + -1.0 + 0.0*y,
                lambda x,y,z: 0.0*y*z + 1.0*z**2 + 0.0*x*z + 0.0*y + 0.0*z + -1.0 + 0.0*x,
                lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 2.0*z*x + 2.0*z*y + 0.0*x*y + -2.0*z + 0.0*x + 0.0*y + 0.0]
        dN10t = [lambda x,y,z: 0.0*x*z + 0.0*z**2 + -2.0*y*z + 0.0*x + 0.0*z + 0.0 + 2.0*y,
                lambda x,y,z: 0.0*y*z + 0.0*z**2 + -2.0*x*z + 0.0*y + 0.0*z + 0.0 + 2.0*x,
                lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 0.0*z*x + 0.0*z*y + -2.0*x*y + 0.0*z + 0.0*x + 0.0*y + 0.0]
        dN11t = [lambda x,y,z: 0.0*x*z + -1.0*z**2 + 0.0*y*z + 0.0*x + 0.0*z + 1.0 + 0.0*y,
                lambda x,y,z: 0.0*y*z + 0.0*z**2 + 0.0*x*z + 0.0*y + 0.0*z + 0.0 + 0.0*x,
                lambda x,y,z: 0.0*x**2 + 0.0*y**2 + -2.0*z*x + 0.0*z*y + 0.0*x*y + 0.0*z + 0.0*x + 0.0*y + 0.0]
        dN12t = [lambda x,y,z: 0.0*x*z + 0.0*z**2 + 0.0*y*z + 0.0*x + 0.0*z + 0.0 + 0.0*y,
                lambda x,y,z: 0.0*y*z + -1.0*z**2 + 0.0*x*z + 0.0*y + 0.0*z + 1.0 + 0.0*x,
                lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 0.0*z*x + -2.0*z*y + 0.0*x*y + 0.0*z + 0.0*x + 0.0*y + 0.0]
        dN13t = [lambda x,y,z: -4.0*x*z + 0.0*z**2 + -2.0*y*z + -4.0*x + 2.0*z + 2.0 + -2.0*y,
                lambda x,y,z: 0.0*y*z + 0.0*z**2 + -2.0*x*z + 0.0*y + 0.0*z + 0.0 + -2.0*x,
                lambda x,y,z: -2.0*x**2 + 0.0*y**2 + 0.0*z*x + 0.0*z*y + -2.0*x*y + 0.0*z + 2.0*x + 0.0*y + 0.0]
        dN14t = [lambda x,y,z: 0.0*x*z + 0.0*z**2 + -2.0*y*z + 0.0*x + 0.0*z + 0.0 + -2.0*y,
                lambda x,y,z: -4.0*y*z + 0.0*z**2 + -2.0*x*z + -4.0*y + 2.0*z + 2.0 + -2.0*x,
                lambda x,y,z: 0.0*x**2 + -2.0*y**2 + 0.0*z*x + 0.0*z*y + -2.0*x*y + 0.0*z + 0.0*x + 2.0*y + 0.0]
        dN15t = [lambda x,y,z: 0.0*x*z + 0.0*z**2 + 2.0*y*z + 0.0*x + 0.0*z + 0.0 + 2.0*y,
                lambda x,y,z: 0.0*y*z + 0.0*z**2 + 2.0*x*z + 0.0*y + 0.0*z + 0.0 + 2.0*x,
                lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 0.0*z*x + 0.0*z*y + 2.0*x*y + 0.0*z + 0.0*x + 0.0*y + 0.0]
        
        dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t, dN9t, dN10t, dN11t, dN12t, dN13t, dN14t, dN15t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:

        ddN1t = [lambda x,y,z: -2.0*z + 2.0, lambda x,y,z: -2.0*z + 2.0, lambda x,y,z: -1.0*x + -1.0*y + 1.0]
        ddN2t = [lambda x,y,z: -2.0*z + 2.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 1.0*x + 0.0*y + 0.0]
        ddN3t = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: -2.0*z + 2.0, lambda x,y,z: 0.0*x + 1.0*y + 0.0]
        ddN4t = [lambda x,y,z: 2.0*z + 2.0, lambda x,y,z: 2.0*z + 2.0, lambda x,y,z: -1.0*x + -1.0*y + 1.0]
        ddN5t = [lambda x,y,z: 2.0*z + 2.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 1.0*x + 0.0*y + 0.0]
        ddN6t = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 2.0*z + 2.0, lambda x,y,z: 0.0*x + 1.0*y + 0.0]
        ddN7t = [lambda x,y,z: 4.0*z + -4.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]
        ddN8t = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 4.0*z + -4.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]
        ddN9t = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 2.0*x + 2.0*y + -2.0]
        ddN10t = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]
        ddN11t = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: -2.0*x + 0.0*y + 0.0]
        ddN12t = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*x + -2.0*y + 0.0]
        ddN13t = [lambda x,y,z: -4.0*z + -4.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]
        ddN14t = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: -4.0*z + -4.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]
        ddN15t = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]

        ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t, ddN5t, ddN6t, ddN7t, ddN8t, ddN9t, ddN10t, ddN11t, ddN12t, ddN13t, ddN14t, ddN15t])

        return ddNtild

    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()
    
    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()

    

        

