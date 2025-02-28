# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Group elem module.\n
A mesh uses several element groups.
For instance, a TRI3 mesh uses POINT, SEG2 and TRI3 elements."""

# Sections :
# Gauss
# Properties
# Isoparametric elements
# Finite element shape functions
# Finite element matrices
# Nodes & Elements
# Factory

from abc import ABC, abstractmethod

from scipy.optimize import least_squares
import numpy as np
import scipy.sparse as sparse
from typing import Callable

# fem
from ._gauss import Gauss
# utils
from ._utils import ElemType, MatrixType

# # others
from ..Geoms import Point, Domain, Line, Circle, Jacobian_Matrix

class _GroupElem(ABC):

    def __init__(self, gmshId: int, connect: np.ndarray, coordGlob: np.ndarray, nodes: np.ndarray):
        """Creates a goup of elements.

        Parameters
        ----------
        gmshId : int
            gmsh id
        connect : np.ndarray
            connectivity matrix
        coordGlob : np.ndarray
            coordinate matrix (contains all mesh coordinates)
        nodes : np.ndarray
            nodes used by element group
        """

        self.__gmshId = gmshId

        elemType, nPe, dim, order, nbFaces, nbCorners = GroupElemFactory.Get_ElemInFos(gmshId)
        # TODO construct without gmshId and auto detect

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
        self.__coordGlob = coordGlob

        # dictionnary associated with tags on elements or nodes
        self.__dict_nodes_tags: dict[str, np.ndarray] = {}
        self.__dict_elements_tags: dict[str, np.ndarray] = {}        
        self._InitMatrix()
    
    def _InitMatrix(self) -> None:
        """Initializes matrix dictionaries for finite element construction"""
        # Dictionaries for each matrix type
        self.__dict_dN_e_pg: dict[MatrixType, np.ndarray] = {}
        self.__dict_ddN_e_pg: dict[MatrixType, np.ndarray] = {}
        self.__dict_F_e_pg: dict[MatrixType, np.ndarray] = {}
        self.__dict_invF_e_pg: dict[MatrixType, np.ndarray] = {}
        self.__dict_jacobian_e_pg: dict[MatrixType, np.ndarray] = {}
        self.__dict_B_e_pg: dict[MatrixType, np.ndarray] = {}
        self.__dict_leftDispPart: dict[MatrixType, np.ndarray] = {}
        self.__dict_phaseField_ReactionPart_e_pg: dict[MatrixType, np.ndarray] = {}
        self.__dict_DiffusePart_e_pg: dict[MatrixType, np.ndarray] = {}
        self.__dict_SourcePart_e_pg: dict[MatrixType, np.ndarray] = {}

    # --------------------------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------------------------
    
    @property
    def gmshId(self) -> int:
        """gmsh Id"""
        return self.__gmshId

    @property
    def elemType(self) -> ElemType:
        """element type"""
        return self.__elemType

    @property
    def nPe(self) -> int:
        """nodes per element"""
        return self.__nPe
    
    @property
    def dim(self) -> int:
        """element dimension"""
        return self.__dim
    
    @property
    def order(self) -> int:
        """element order"""
        return self.__order    

    @property
    def inDim(self) -> int:
        """dimension in which the elements are located"""
        if self.elemType in ElemType.Get_3D():
            return 3
        else:
            x,y,z = np.abs(self.coord.T)
            if np.max(y)==0 and np.max(z)==0:
                inDim = 1
            if np.max(z)==0:
                inDim = 2
            else:
                inDim = 3
            return inDim

    @property
    def Ne(self) -> int:
        """number of elements"""
        return self.__connect.shape[0]

    @property
    def nodes(self) -> int:
        """nodes used by the element group. Node 'n' is on line 'n' in coordGlob"""
        return self.__nodes.copy()

    @property
    def elements(self) -> np.ndarray:
        """elements"""
        return np.arange(self.__connect.shape[0], dtype=int)

    @property
    def Nn(self) -> int:
        """number of nodes used by the element group"""
        return self.__nodes.size

    @property
    def coord(self) -> np.ndarray:
        """this matrix contains the element group coordinates (Nn, 3)"""
        coord: np.ndarray = self.coordGlob[self.__nodes]
        return coord

    @property
    def coordGlob(self) -> np.ndarray:
        """this matrix contains all the mesh coordinates (mesh.Nn, 3)"""
        return self.__coordGlob.copy()
    
    @coordGlob.setter
    def coordGlob(self, coord: np.ndarray) -> None:
        if coord.shape == self.__coordGlob.shape:
            self.__coordGlob = coord
            self._InitMatrix()

    @property
    def nbFaces(self) -> int:
        """number of faces per element"""
        return self.__nbFaces
    
    @property
    def nbCorners(self) -> int:
        """number of corners per element"""
        return self.__nbCorners
    
    @property
    def connect(self) -> np.ndarray:
        """connectivity matrix (Ne, nPe)"""
        return self.__connect.copy()
    
    def Get_connect_n_e(self) -> sparse.csr_matrix:
        """Sparse matrix (Nn, Ne) of zeros and ones with ones when the node has the element such that:
        values_n = connect_n_e * values_e\n
        (Nn,1) = (Nn,Ne) * (Ne,1)"""

        # Here, the aim is to construct a matrix which, when multiplied by a values_e vector of size ( Ne x 1 ), will give
        # values_n_e(Nn,1) = connecNoeud(Nn,Ne) values_n_e(Ne,1)
        # where connecNoeud(Nn,:) is a row vector composed of 0 and 1, which will be used to sum values_e[nodes].
        # Then just divide by the number of times the node appears in the line        
        
        Ne = self.Ne
        nPe = self.nPe
        elems = self.elements

        lines = self.connect.ravel()

        Nn = lines.max()+1 # Do not use either self.Nn or self.__coordGlob.shape[0].
        columns = np.repeat(elems, nPe)

        return sparse.csr_matrix((np.ones(nPe*Ne),(lines, columns)),shape=(Nn,Ne))

    @property
    def assembly_e(self) -> np.ndarray:
        """assembly matrix (Ne, nPe*dim)"""

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
        """Get the assembly matrix for the specified dof_n (Ne, nPe*dof_n)

        Parameters
        ----------
        dof_n : int
            degree of freedom per node
        """

        nPe = self.nPe
        ndof = dof_n*nPe

        assembly = np.zeros((self.Ne, ndof), dtype=np.int64)
        connect = self.connect

        for d in range(dof_n):
            columns = np.arange(d, ndof, dof_n)
            assembly[:, columns] = np.array(connect) * dof_n + d

        return assembly
    
    def _Get_sysCoord_e(self, displacementMatrix:np.ndarray=None):
        """Get the basis transformation matrix (Ne,3,3).\n
        This matrix can be used to project points with (x, y, z) coordinates into the element's (i, j, k) coordinate system."""

        coordo = self.coordGlob

        if displacementMatrix is not None:
            displacementMatrix = np.asarray(displacementMatrix, dtype=float)
            assert displacementMatrix.shape == coordo.shape, f'displacmentMatrix must be of size {coordo.shape}'
            coordo += displacementMatrix

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
        """Basis transformation matrix (Ne, 3, 3)\n
        [ix, jx, kx\n
        iy, jy, ky\n
        iz, jz, kz]\n
        
        This matrix can be used to project points with (x, y, z) coordinates into the element's (i, j, k) coordinate system.\n
        coordo_e * sysCoord_e -> coordinates in element's (i, j, k) coordinate system.
        """
        return self._Get_sysCoord_e()
    
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
        """length covered by each element"""
        if self.dim != 1: return        
        length_e = self.Integrate_e(lambda x,y,z: 1)
        return length_e

    @property
    def length(self) -> float:
        """length covered by elements"""
        if self.dim != 1: return
        return self.length_e.sum()
    
    @property
    def area_e(self) -> np.ndarray:
        """area covered by each element"""
        if self.dim != 2: return
        area_e = self.Integrate_e(lambda x,y,z: 1)
        return area_e

    @property
    def area(self) -> float:
        """area covered by elements"""
        if self.dim != 2: return
        return self.area_e.sum()
    
    @property
    def volume_e(self) -> np.ndarray:
        """volume covered by each element"""
        if self.dim != 3: return        
        volume_e = self.Integrate_e(lambda x,y,z: 1)
        return volume_e
    
    @property
    def volume(self) -> float:
        """volume covered by elements"""
        if self.dim != 3: return
        return self.volume_e.sum()
    
    @property
    def center(self) -> np.ndarray:
        """center of mass / barycenter / inertia center"""

        matrixType = MatrixType.mass

        coordo_e_p = self.Get_GaussCoordinates_e_p(matrixType)

        jacobian_e_p = self.Get_jacobian_e_pg(matrixType)
        weight_p = self.Get_weight_pg(matrixType)

        size = np.einsum('ep,p->', jacobian_e_p, weight_p, optimize='optimal')

        center: np.ndarray = np.einsum('ep,p,epi->i', jacobian_e_p, weight_p, coordo_e_p, optimize='optimal') / size

        return center
    
    @property  
    @abstractmethod
    def origin(self) -> list[int]:
        """reference element origin coordinates"""
        return [0]

    @property  
    @abstractmethod
    def triangles(self) -> list[int]:
        """list of indexes to form the triangles of an element that will be used for the 2D trisurf function"""
        pass

    @property
    def segments(self) -> np.ndarray:
        """list of indexes used to construct segments"""
        if self.__dim == 1:
            return np.array([[0, 1]], dtype=int)
        elif self.__dim == 2:
            segments = np.zeros((self.nbCorners, 2), dtype=int)
            segments[:,0] = np.arange(self.nbCorners)
            segments[:,1] = np.append(np.arange(1, self.nbCorners, 1), 0)
            return segments
        elif self.__dim == 3:
            raise Exception("To be defined for 3D element groups.")
    
    @property  
    @abstractmethod
    def faces(self) -> list[int]:
        """list of indexes to form the faces that make up the element"""
        pass

    # --------------------------------------------------------------------------------------------
    # Gauss
    # --------------------------------------------------------------------------------------------

    def Get_gauss(self, matrixType: MatrixType) -> Gauss:
        """Returns integration points according to the matrix type."""
        return Gauss(self.elemType, matrixType)
    
    def Get_weight_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Returns integration point weights according to the matrix type."""
        return Gauss(self.elemType, matrixType).weights
    
    def Get_GaussCoordinates_e_p(self, matrixType: MatrixType, elements=np.array([])) -> np.ndarray:
        """Returns integration point coordinates for each element (Ne, p, 3) in the (x,y,z) coordinates."""

        N_scalar = self.Get_N_pg(matrixType)

        # retrieve node coordinates
        coordo = self.coordGlob

        # nodes coordinates for each element
        if elements.size == 0:
            coordo_e = coordo[self.__connect]
        else:
            coordo_e = coordo[self.__connect[elements]]

        # localize coordinates on Gauss points
        coordo_e_p = np.einsum('pij,ejn->epn', N_scalar, coordo_e, optimize='optimal')

        return np.array(coordo_e_p)
    
    # --------------------------------------------------------------------------------------------
    # Isoparametric elements
    # --------------------------------------------------------------------------------------------

    def Get_F_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Returns the Jacobian matrix.\n        
        This matrix describes the transformation of the (ξ,η,ζ) axes from the reference element to the (x,y,z) coordinate system of the actual element.\n
        """
        if self.dim == 0: return
        if matrixType not in self.__dict_F_e_pg.keys():

            coordo_e: np.ndarray = self.coordGlob[self.__connect]

            nodesBase = coordo_e.copy()
            if self.dim != self.inDim:
                sysCoord_e = self.sysCoord_e # basis transformation matrix for each element
                # This matrix can be used to project points with (x, y, z) coordinates into the element's (i, j, k) coordinate system.
                nodesBase = coordo_e @ sysCoord_e # nodes coordinates in the element's (i, j, k) coordinate system.

            nodesBaseDim = nodesBase[:,:,range(self.dim)]

            dN_pg = self.Get_dN_pg(matrixType)

            F_e_pg: np.ndarray = np.einsum('pik,ekj->epij', dN_pg, nodesBaseDim, optimize='optimal')
            
            self.__dict_F_e_pg[matrixType] = F_e_pg

        return self.__dict_F_e_pg[matrixType].copy()
    
    def Get_jacobian_e_pg(self, matrixType: MatrixType, absoluteValues=True) -> np.ndarray:
        """Returns the jacobians.\n
        variation in size (length, area or volume) between the reference element and the actual element
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
        """Returns the inverse of the Jacobian matrix.\n
        Used to obtain the derivative of the dN_e_pg shape functions in the actual element
        dN_e_pg = invF_e_pg • dN_pg
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
    
    # --------------------------------------------------------------------------------------------
    # Finite element shape functions
    # --------------------------------------------------------------------------------------------

    @staticmethod
    def _Evaluates_Functions(functions: np.ndarray, coord: np.ndarray) -> np.ndarray:
        """Evaluates functions at coordinates.\n
        Use this function to evaluate shape functions.

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
        """Initializes functions to be evaluated at gauss points."""
        if self.dim == 1 and self.order < order:
            functions = np.array([lambda x: 0]*self.nPe)
        elif self.dim == 2 and self.order < order:
            functions = np.array([lambda xi,eta: 0, lambda xi,eta: 0]*self.nPe)
        elif self.dim == 3 and self.order < order:
            functions = np.array([lambda x,y,z: 0,lambda x,y,z: 0,lambda x,y,z: 0]*self.nPe)
        functions = np.reshape(functions, (self.nPe, -1))
        return functions

    # N

    @abstractmethod
    def _N(self) -> np.ndarray:
        """Shape functions in (ξ,η,ζ) coordinates.\n
        [N1, N2, . . . ,Nn]\n
        (nPe)
        """
        pass

    def Get_N_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluates shape functions in (ξ,η,ζ) coordinates.\n
        [N1, N2, . . . ,Nn]\n
        (pg, nPe)
        """
        if self.dim == 0: return

        N = self._N()
        gauss = self.Get_gauss(matrixType)
        N_pg = _GroupElem._Evaluates_Functions(N, gauss.coord)

        return N_pg

    def Get_N_pg_rep(self, matrixType: MatrixType, repeat=1) -> np.ndarray:
        """Repeats shape functions in the (ξ,η,ζ) coordinates.

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

    # dN
    
    @abstractmethod
    def _dN(self) -> np.ndarray:
        """Shape functions first derivatives in the (ξ,η,ζ) coordinates.\n
        [Ni,xi . . . Nn,xi\n
        Ni,eta ... Nn,eta]\n
        (dim, nPe)
        """
        return self.__Init_Functions(1)
    
    def Get_dN_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluates shape functions first derivatives in the (ξ,η,ζ) coordinates.\n
        [Ni,xi . . . Nn,xi\n
        Ni,eta ... Nn,eta]\n
        (pg, dim, nPe)
        """
        if self.dim == 0: return

        dN = self._dN()

        gauss = self.Get_gauss(matrixType)
        dN_pg = _GroupElem._Evaluates_Functions(dN, gauss.coord)

        return dN_pg    

    def Get_dN_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluates the first-order derivatives of shape functions in (x,y,z) coordinates.\n
        [Ni,x . . . Nn,x\n
        Ni,y ... Nn,y]\n
        (e, pg, dim, nPe)\n
        """
        assert matrixType in MatrixType.Get_types()

        if matrixType not in self.__dict_dN_e_pg.keys():

            invF_e_pg = self.Get_invF_e_pg(matrixType)

            dN_pg = self.Get_dN_pg(matrixType)

            # Derivation of shape functions in the (x,y,z) coordinates
            dN_e_pg: np.ndarray = np.einsum('epdk,pkn->epdn', invF_e_pg, dN_pg, optimize='optimal')
            self.__dict_dN_e_pg[matrixType] = dN_e_pg

        return self.__dict_dN_e_pg[matrixType].copy()
    
    # ddN
    
    @abstractmethod
    def _ddN(self) -> np.ndarray:
        """Shape functions second derivatives in the (ξ,η,ζ) coordinates.\n
        [Ni,xi2 . . . Nn,xi2\n
        Ni,eta2 . . . Nn,eta eta]\n
        (dim, nPe)
        """
        return self.__Init_Functions(2)

    def Get_ddN_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluates the second-order derivatives of shape functions in (x,y,z) coordinates.\n
        [Ni,xx . . . Nn,xx\n
        Ni,yy ... Nn,yy]\n
        (e, pg, dim, nPe)\n
        """
        assert matrixType in MatrixType.Get_types()

        if matrixType not in self.__dict_ddN_e_pg.keys():

            invF_e_pg = self.Get_invF_e_pg(matrixType)

            invF_e_pg = invF_e_pg @ invF_e_pg

            ddN_pg = self.Get_ddN_pg(matrixType)
            
            ddN_e_pg = np.array(np.einsum('epdk,pkn->epdn', invF_e_pg, ddN_pg, optimize='optimal'))
            self.__dict_ddN_e_pg[matrixType] = ddN_e_pg

        return self.__dict_ddN_e_pg[matrixType].copy()
    
    def Get_ddN_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluates shape functions second derivatives in the (ξ,η,ζ) coordinates.\n
        [Ni,xi2 . . . Nn,xi2\n
        Ni,eta2 . . . Nn,eta2]\n
        (pg, dim, nPe)
        """
        if self.dim == 0: return

        ddN = self._ddN()

        gauss = self.Get_gauss(matrixType)
        ddN_pg = _GroupElem._Evaluates_Functions(ddN, gauss.coord)

        return ddN_pg

    # dddN

    @abstractmethod
    def _dddN(self) -> np.ndarray:
        """Shape functions third derivatives in the (ξ,η,ζ) coordinates.\n
        [Ni,xi3 . . . Nn,xi3\n
        Ni,eta3 . . . Nn,eta3]\n
        (dim, nPe)
        """
        return self.__Init_Functions(3)

    def Get_dddN_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluates shape functions third derivatives in the (ξ,η,ζ) coordinates.\n
        [Ni,xi3 . . . Nn,xi3\n
        Ni,eta3 . . . Nn,eta3]\n
        (pg, dim, nPe)
        """
        if self.elemType == 0: return

        dddN = self._dddN()

        gauss = self.Get_gauss(matrixType)
        dddN_pg = _GroupElem._Evaluates_Functions(dddN, gauss.coord)

        return dddN_pg

    # ddddN
    
    @abstractmethod
    def _ddddN(self) -> np.ndarray:
        """Shape functions fourth derivatives in the (ξ,η,ζ) coordinates.\n
        [Ni,xi4 . . . Nn,xi4\n
        Ni,eta4 . . . Nn,eta4]
        \n
        (dim, nPe)
        """
        return self.__Init_Functions(4)

    def Get_ddddN_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluates shape functions fourth derivatives in the (ξ,η,ζ) coordinates.\n
        [Ni,xi4 . . . Nn,xi4\n
        Ni,eta4 . . . Nn,eta4]
        \n
        (pg, dim, nPe)
        """
        if self.elemType == 0: return

        ddddN = self._ddddN()

        gauss = self.Get_gauss(matrixType)
        ddddN_pg = _GroupElem._Evaluates_Functions(ddddN, gauss.coord)

        return ddddN_pg
        
    # Beams shapes functions
    # Use hermitian shape functions

    # N
    
    def _EulerBernoulli_N(self) -> np.ndarray:
        """Euler-Bernoulli beam shape functions in the (ξ,η,ζ) coordinates.\n
        [phi_i psi_i . . . phi_n psi_n]\n
        (nPe*2)
        """
        pass

    def Get_EulerBernoulli_N_pg(self) -> np.ndarray:
        """Evaluates Euler-Bernoulli beam shape functions in the (ξ,η,ζ) coordinates.\n
        [phi_i psi_i . . . phi_n psi_n]\n        
        (pg, nPe*2)
        """
        if self.dim != 1: return

        matrixType = MatrixType.beam

        N = self._EulerBernoulli_N()

        gauss = self.Get_gauss(matrixType)
        N_pg = _GroupElem._Evaluates_Functions(N, gauss.coord)

        return N_pg
    
    def Get_EulerBernoulli_N_e_pg(self) -> np.ndarray:
        """Evaluates Euler-Bernoulli beam shape functions in (x,y,z) coordinates.\n
        [phi_i psi_i . . . phi_n psi_n]\n
        (e, pg, 1, nPe*2)
        """
        if self.dim != 1: return

        invF_e_pg = self.Get_invF_e_pg(MatrixType.beam)
        N_pg = self.Get_EulerBernoulli_N_pg()
        nPe = self.nPe
        
        N_e_pg = invF_e_pg @ N_pg
        
        # multiply by the beam length on psi_i,xx functions
        l_e = self.length_e
        columns = np.arange(1, nPe*2, 2)
        for column in columns:
            N_e_pg[:,:,0,column] = np.einsum('ep,e->ep', N_e_pg[:,:,0,column], l_e, optimize='optimal')    

        return N_e_pg
    
    # dN
    
    def _EulerBernoulli_dN(self) -> np.ndarray:
        """Euler-Bernoulli beam shape functions first derivatives in the (ξ,η,ζ) coordinates.\n
        [phi_i,xi psi_i,xi . . . phi_n,xi psi_n,xi]\n
        (nPe*2)
        """
        pass

    def Get_EulerBernoulli_dN_pg(self) -> np.ndarray:
        """Evaluates Euler-Bernoulli beam shape functions first derivatives in the (ξ,η,ζ) coordinates.\n
        [phi_i,xi psi_i,xi . . . phi_n,xi psi_n,xi]\n
        (pg, nPe*2)
        """
        if self.dim != 1: return

        matrixType = MatrixType.beam

        dN = self._EulerBernoulli_dN()

        gauss = self.Get_gauss(matrixType)
        dN_pg = _GroupElem._Evaluates_Functions(dN, gauss.coord)

        return dN_pg
    
    def Get_EulerBernoulli_dN_e_pg(self) -> np.ndarray:
        """Evaluates the first-order derivatives of Euler-Bernoulli beam shape functions in (x,y,z) coordinates.\n
        [phi_i,x psi_i,x . . . phi_n,x psi_n,x]\n
        (e, pg, 1, nPe*2)
        """
        if self.dim != 1: return

        invF_e_pg = self.Get_invF_e_pg(MatrixType.beam)
        dN_pg = self.Get_EulerBernoulli_dN_pg()
        nPe = self.nPe
        
        dN_e_pg = invF_e_pg @ dN_pg
        
        # multiply by the beam length on psi_i,xx functions
        l_e = self.length_e
        columns = np.arange(1, nPe*2, 2)
        for column in columns:
            dN_e_pg[:,:,0,column] = np.einsum('ep,e->ep', dN_e_pg[:,:,0,column], l_e, optimize='optimal')

        return dN_e_pg
    
    # ddN
    
    def _EulerBernoulli_ddN(self) -> np.ndarray:
        """Euler-Bernoulli beam shape functions second derivatives in the (ξ,η,ζ) coordinates.\n
        [phi_i,xi2 psi_i,xi2 . . . phi_n,xi2 psi_n,xi2]\n
        (nPe*2)
        """
        return 
    
    def Get_EulerBernoulli_ddN_pg(self) -> np.ndarray:
        """Evaluates Euler-Bernoulli beam shape functions second derivatives in the (ξ,η,ζ) coordinates.\n
        [phi_i,xi2 psi_i,xi2 . . . phi_n,xi2 x psi_n,xi2]\n
        (pg, nPe*2)
        """
        if self.dim != 1: return

        matrixType = MatrixType.beam

        ddN = self._EulerBernoulli_ddN()

        gauss = self.Get_gauss(matrixType)
        ddN_pg = _GroupElem._Evaluates_Functions(ddN, gauss.coord)

        return ddN_pg
    
    def Get_EulerBernoulli_ddN_e_pg(self) -> np.ndarray:
        """Evaluates the second-order derivatives of Euler-Bernoulli beam shape functions in (x,y,z) coordinates.\n
        [phi_i,xx psi_i,xx . . . phi_n,xx psi_n,xx]\n
        (e, pg, 1, nPe*2)
        """
        if self.dim != 1: return

        invF_e_pg = self.Get_invF_e_pg(MatrixType.beam)
        ddN_pg = self.Get_EulerBernoulli_ddN_pg()
        nPe = self.nPe
        
        ddN_e_pg = invF_e_pg @ invF_e_pg @ ddN_pg
        
        # multiply by the beam length on psi_i,xx functions
        l_e = self.length_e
        columns = np.arange(1, nPe*2, 2)
        for column in columns:
            ddN_e_pg[:,:,0,column] = np.einsum('ep,e->ep', ddN_e_pg[:,:,0,column], l_e, optimize='optimal')

        return ddN_e_pg
    
    # --------------------------------------------------------------------------------------------
    # Finite element matrices
    # --------------------------------------------------------------------------------------------
    
    # Linear elastic problem

    def Get_B_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Get the matrix used to calculate deformations from displacements.\n
        WARNING: Use Kelvin Mandel Notation\n
        [N1,x 0 N2,x 0 Nn,x 0\n
        0 N1,y 0 N2,y 0 Nn,y\n
        N1,y N1,x N2,y N2,x N3,y N3,x]\n
        (e, pg, (3 or 6), nPe*dim)        
        """
        assert matrixType in MatrixType.Get_types()

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
        """Get the left side of local displacement matrices.\n
        Ku_e = jacobian_e_pg * weight_pg * B_e_pg' * c_e_pg * B_e_pg\n
        
        Returns (epij) -> jacobian_e_pg * weight_pg * B_e_pg'.
        """

        assert matrixType in MatrixType.Get_types()

        if matrixType not in self.__dict_leftDispPart.keys():
            
            jacobian_e_pg = self.Get_jacobian_e_pg(matrixType)
            weight_pg = self.Get_gauss(matrixType).weights
            B_e_pg = self.Get_B_e_pg(matrixType)

            leftDispPart = np.einsum('ep,p,epij->epji', jacobian_e_pg, weight_pg, B_e_pg, optimize='optimal')

            self.__dict_leftDispPart[matrixType] = leftDispPart

        return self.__dict_leftDispPart[matrixType].copy()

    # Euler Bernoulli problem

    def Get_EulerBernoulli_N_e_pg(self, beamStructure) -> np.ndarray:
        """Euler-Bernoulli beam shape functions."""

        from ..materials._beam import BeamStructure

        # Example in matlab :
        # https://github.com/fpled/FEMObject/blob/master/BASIC/MODEL/ELEMENTS/%40BEAM/calc_N.m

        matrixType = MatrixType.beam

        # get the beam model
        beamStructure: BeamStructure = beamStructure
        dim = beamStructure.dim
        dof_n = beamStructure.dof_n

        # Data
        jacobian_e_pg = self.Get_jacobian_e_pg(matrixType)
        nPe = self.nPe
        Ne = jacobian_e_pg.shape[0]
        nPg = jacobian_e_pg.shape[1]

        # get matrices to work with
        N_pg = self.Get_N_pg(matrixType)        
        if beamStructure.dim > 1:
            N_e_pg = self.Get_EulerBernoulli_N_e_pg()
            dN_e_pg = self.Get_EulerBernoulli_dN_e_pg()

        if dim == 1:
            # u = [u1, . . . , un]
            
            # N = [N_i, . . . , N_n]

            idx_ux = np.arange(dof_n*nPe)

            N_e_pg = np.zeros((Ne, nPg, 1, dof_n*nPe))
            N_e_pg[:,:,0, idx_ux] = N_pg[:,:,0]
                
        elif dim == 2:
            # u = [u1, v1, rz1, . . . , un, vn, rzn]
            
            # N = [N_i, 0, 0, ... , N_n, 0, 0,]
            #     [0, Phi_i, Psi_i, ... , 0, Phi_i, Psi_i]
            #     [0, dPhi_i, dPsi_i, ... , 0, dPhi_i, dPsi_i]

            idx = np.arange(dof_n*nPe).reshape(nPe,-1)
            
            idx_ux = idx[:,0] # [0,3] (SEG2) [0,3,6] (SEG3)
            idx_uy = np.reshape(idx[:,1:], -1) # [1,2,4,5] (SEG2) [1,2,4,5,7,8] (SEG3)

            N_e_pg = np.zeros((Ne, nPg, 3, dof_n*nPe))
            
            N_e_pg[:,:,0, idx_ux] = N_pg[:,:,0] # traction / compression to get u
            N_e_pg[:,:,1, idx_uy] = N_e_pg[:,:,0] # flexion z to get v
            N_e_pg[:,:,2, idx_uy] = dN_e_pg[:,:,0] # flexion z to get rz

        elif dim == 3:
            # u = [u1, v1, w1, rx1, ry1, rz1, . . . , un, vn, wn, rxn, ryn, rzn]

            # N = [N_i, 0, 0, 0, 0, 0, ... , N_n, 0, 0, 0, 0, 0]
            #     [0, Phi_i, 0, 0, 0, Psi_i, ... , 0, Phi_n, 0, 0, 0, Psi_n]
            #     [0, 0, dPhi_i, 0, -dPsi_i, 0, ... , 0, 0, dPhi_n, 0, -dPsi_n, 0]
            #     [0, 0, 0, N_i, 0, 0, ... , 0, 0, 0, N_n, 0, 0]
            #     [0, 0, -dPhi_i, 0, dPsi_i, 0, ... , 0, 0, -dPhi_n, 0, dPsi_n, 0]
            #     [0, dPhi_i, 0, 0, 0, dPsi_i, ... , 0, dPhi_i, 0, 0, 0, dPsi_n]

            idx = np.arange(dof_n*nPe).reshape(nPe,-1)
            idx_ux = idx[:,0] # [0,6] (SEG2) [0,6,12] (SEG3)
            idx_uy = np.reshape(idx[:,[1,5]], -1) # [1,5,7,11] (SEG2) [1,5,7,11,13,17] (SEG3)
            idx_uz = np.reshape(idx[:,[2,4]], -1) # [2,4,8,10] (SEG2) [2,4,8,10,14,16] (SEG3)
            idx_rx = idx[:,3] # [3,9] (SEG2) [3,9,15] (SEG3)
            idPsi = np.arange(1, nPe*2, 2) # [1,3] (SEG2) [1,3,5] (SEG3)

            Nvz_e_pg = N_e_pg.copy()
            Nvz_e_pg[:,:,0,idPsi] *= -1

            dNvz_e_pg = dN_e_pg.copy()
            dNvz_e_pg[:,:,0,idPsi] *= -1

            N_e_pg = np.zeros((Ne, nPg, 6, dof_n*nPe))
            
            N_e_pg[:,:,0, idx_ux] = N_pg[:,:,0]
            N_e_pg[:,:,1, idx_uy] = N_e_pg[:,:,0]
            N_e_pg[:,:,2, idx_uz] = Nvz_e_pg[:,:,0]
            N_e_pg[:,:,3, idx_rx] = N_pg[:,:,0]
            N_e_pg[:,:,4, idx_uz] = -dNvz_e_pg[:,:,0] # ry = -uz'
            N_e_pg[:,:,5, idx_uy] = dN_e_pg[:,:,0] # rz = uy'
        
        if dim > 1:
            # Construct the matrix used to change the matrix coordinates 
            P = np.zeros((self.Ne, 3, 3), dtype=float)
            for beam in beamStructure.beams:
                elems = self.Get_Elements_Tag(beam.name)
                P[elems] = beam._Calc_P()

            Pglob_e = np.zeros((Ne, dof_n*nPe, dof_n*nPe))            
            N = P.shape[1]
            lines = np.repeat(range(N), N)
            columns = np.array(list(range(N))*N)
            for n in range(dof_n*nPe//3):
                # apply P on the diagonal
                Pglob_e[:, lines + n*N, columns + n*N] = P[:,lines,columns]

            N_e_pg = np.einsum('epij,ejk->epik', N_e_pg, Pglob_e, optimize='optimal')

        return N_e_pg
    
    def Get_EulerBernoulli_B_e_pg(self, beamStructure) -> np.ndarray:
        """Get Euler-Bernoulli beam shape functions derivatives"""

        from ..materials._beam import BeamStructure

        # Example in matlab :
        # https://github.com/fpled/FEMObject/blob/master/BASIC/MODEL/ELEMENTS/%40BEAM/calc_B.m

        matrixType = MatrixType.beam

        # Recovering the beam model
        beamStructure: BeamStructure = beamStructure
        dim = beamStructure.dim
        dof_n = beamStructure.dof_n

        # Data
        jacobian_e_pg = self.Get_jacobian_e_pg(matrixType)
        nPe = self.nPe
        Ne = jacobian_e_pg.shape[0]
        nPg = jacobian_e_pg.shape[1]

        # Recover matrices to work with
        dN_e_pg = self.Get_dN_e_pg(matrixType)
        if beamStructure.dim > 1:
            ddNv_e_pg = self.Get_EulerBernoulli_ddN_e_pg()

        if dim == 1:
            # u = [u1, . . . , un]
            
            # B = [dN_i, . . . , dN_n]

            idx_ux = np.arange(dof_n*nPe)

            B_e_pg = np.zeros((Ne, nPg, 1, dof_n*nPe))
            B_e_pg[:,:,0, idx_ux] = dN_e_pg[:,:,0]
                
        elif dim == 2:
            # u = [u1, v1, rz1, . . . , un, vn, rzn]
            
            # B = [dN_i, 0, 0, ... , dN_n, 0, 0,]
            #     [0, ddPhi_i, ddPsi_i, ... , 0, ddPhi_i, ddPsi_i]

            idx = np.arange(dof_n*nPe).reshape(nPe,-1)
            
            idx_ux = idx[:,0] # [0,3] (SEG2) [0,3,6] (SEG3)
            idx_uy = np.reshape(idx[:,1:], -1) # [1,2,4,5] (SEG2) [1,2,4,5,7,8] (SEG3)

            B_e_pg = np.zeros((Ne, nPg, 2, dof_n*nPe))
            
            B_e_pg[:,:,0, idx_ux] = dN_e_pg[:,:,0] # traction / compression
            B_e_pg[:,:,1, idx_uy] = ddNv_e_pg[:,:,0] # flexion along z

        elif dim == 3:
            # u = [u1, v1, w1, rx1, ry1, rz1, . . . , un, vn, wn, rxn, ryn, rzn]

            # B = [dN_i, 0, 0, 0, 0, 0, ... , dN_n, 0, 0, 0, 0, 0]
            #     [0, 0, 0, dN_i, 0, 0, ... , 0, 0, 0, dN_n, 0, 0]
            #     [0, 0, ddPhi_i, 0, -ddPsi_i, 0, ... , 0, 0, ddPhi_n, 0, -ddPsi_n, 0]
            #     [0, ddPhi_i, 0, 0, 0, ddPsi_i, ... , 0, ddPhi_i, 0, 0, 0, ddPsi_n]

            idx = np.arange(dof_n*nPe).reshape(nPe,-1)
            idx_ux = idx[:,0] # [0,6] (SEG2) [0,6,12] (SEG3)
            idx_uy = np.reshape(idx[:,[1,5]], -1) # [1,5,7,11] (SEG2) [1,5,7,11,13,17] (SEG3)
            idx_uz = np.reshape(idx[:,[2,4]], -1) # [2,4,8,10] (SEG2) [2,4,8,10,14,16] (SEG3)
            idx_rx = idx[:,3] # [3,9] (SEG2) [3,9,15] (SEG3)
            
            idPsi = np.arange(1, nPe*2, 2) # [1,3] (SEG2) [1,3,5] (SEG3)
            ddNvz_e_pg = ddNv_e_pg.copy()
            ddNvz_e_pg[:,:,0,idPsi] *= -1 # RY = -UZ'

            B_e_pg = np.zeros((Ne, nPg, 4, dof_n*nPe))
            
            B_e_pg[:,:,0, idx_ux] = dN_e_pg[:,:,0] # traction / compression
            B_e_pg[:,:,1, idx_rx] = dN_e_pg[:,:,0] # torsion
            B_e_pg[:,:,2, idx_uz] = ddNvz_e_pg[:,:,0] # flexion along y
            B_e_pg[:,:,3, idx_uy] = ddNv_e_pg[:,:,0] # flexion along z        

        if dim > 1:
            # Construct the matrix used to change the matrix coordinates 
            P = np.zeros((self.Ne, 3, 3), dtype=float)
            for beam in beamStructure.beams:
                elems = self.Get_Elements_Tag(beam.name)
                P[elems] = beam._Calc_P()

            Pglob_e = np.zeros((Ne, dof_n*nPe, dof_n*nPe))            
            N = P.shape[1]
            lines = np.repeat(range(N), N)
            columns = np.array(list(range(N))*N)
            for n in range(dof_n*nPe//3):
                # apply P on the diagonal
                Pglob_e[:, lines + n*N, columns + n*N] = P[:,lines,columns]

            B_e_pg = np.einsum('epij,ejk->epik', B_e_pg, Pglob_e, optimize='optimal')

        return B_e_pg
    
    # reaction diffusion problem

    def Get_ReactionPart_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Get the part that builds the reaction term (scalar).\n
        ReactionPart_e_pg = r_e_pg * jacobian_e_pg * weight_pg * N_pg' * N_pg\n
        
        Returns -> jacobian_e_pg * weight_pg * N_pg' * N_pg
        """

        assert matrixType in MatrixType.Get_types()

        if matrixType not in self.__dict_phaseField_ReactionPart_e_pg.keys():

            jacobian_e_pg = self.Get_jacobian_e_pg(matrixType)
            weight_pg = self.Get_gauss(matrixType).weights
            N_pg = self.Get_N_pg_rep(matrixType, 1)

            ReactionPart_e_pg = np.einsum('ep,p,pki,pkj->epij', jacobian_e_pg, weight_pg, N_pg, N_pg, optimize='optimal')

            self.__dict_phaseField_ReactionPart_e_pg[matrixType] = ReactionPart_e_pg
        
        return self.__dict_phaseField_ReactionPart_e_pg[matrixType].copy()
    
    def Get_DiffusePart_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Get the part that builds the diffusion term (scalar).\n
        DiffusePart_e_pg = k_e_pg * jacobian_e_pg * weight_pg * dN_e_pg' * A * dN_e_pg\n
        
        Returns -> jacobian_e_pg * weight_pg * dN_e_pg'
        """

        assert matrixType in MatrixType.Get_types()

        if matrixType not in self.__dict_DiffusePart_e_pg.keys():

            jacobian_e_pg = self.Get_jacobian_e_pg(matrixType)
            weight_pg = self.Get_gauss(matrixType).weights
            dN_e_pg = self.Get_dN_e_pg(matrixType)

            DiffusePart_e_pg = np.einsum('ep,p,epij->epji', jacobian_e_pg, weight_pg, dN_e_pg, optimize='optimal')

            self.__dict_DiffusePart_e_pg[matrixType] = DiffusePart_e_pg
        
        return self.__dict_DiffusePart_e_pg[matrixType].copy()

    def Get_SourcePart_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Get the part that builds the source term (scalar).\n
        SourcePart_e_pg = f_e_pg * jacobian_e_pg, weight_pg, N_pg'\n
        
        Returns -> jacobian_e_pg, weight_pg, N_pg'
        """

        assert matrixType in MatrixType.Get_types()

        if matrixType not in self.__dict_SourcePart_e_pg.keys():

            jacobian_e_pg = self.Get_jacobian_e_pg(matrixType)
            weight_pg = self.Get_gauss(matrixType).weights
            N_pg = self.Get_N_pg_rep(matrixType, 1)

            SourcePart_e_pg = np.einsum('ep,p,pij->epji', jacobian_e_pg, weight_pg, N_pg, optimize='optimal') # the ji is important for the transposition

            self.__dict_SourcePart_e_pg[matrixType] = SourcePart_e_pg
        
        return self.__dict_SourcePart_e_pg[matrixType].copy()

    # --------------------------------------------------------------------------------------------
    # Nodes & Elements
    # --------------------------------------------------------------------------------------------

    def Get_Elements_Nodes(self, nodes: np.ndarray, exclusively=True) -> np.ndarray:
        """Returns elements that exclusively or not use the specified nodes."""
        connect = self.__connect
        connect_n_e = self.Get_connect_n_e()

        if isinstance(nodes, list):
            nodes = np.array(nodes)

        # Check that there are no excess nodes
        # It is possible that the nodes entered do not belong to the group
        if connect_n_e.shape[0] <= nodes.max():
            # Remove all excess nodes
            availableNodes = np.where(nodes < self.Nn)[0]
            nodes = nodes[availableNodes]
        
        __, columns, __ = sparse.find(connect_n_e[nodes])

        elements =  list(set(columns))
        
        if exclusively:
            # Check whether elements use only nodes in the node list
            
            # get nodes used by elements
            nodesElem = set(connect[elements].ravel())

            # detect nodes used by elements that are not in the nodes specified
            nodesIntru = list(nodesElem - set(nodes))

            # We detect the list of elements associated with unused nodes
            cols = sparse.find(connect_n_e[nodesIntru])[1]
            elementsIntru = list(set(cols))

            if len(elementsIntru) > 0:
                # Remove detected elements
                elements = list(set(elements) - set(elementsIntru))

        return np.asarray(elements, dtype=int)

    def Get_Nodes_Conditions(self, func: Callable) -> np.ndarray:
        """Returns nodes that meet the specified conditions.

        Parameters
        ----------
        func
            Function using x, y and z nodes coordinates and returning boolean values.

            examples :\n
            \t lambda x, y, z: (x < 40) & (x > 20) & (y<10) \n
            \t lambda x, y, z: (x == 40) | (x == 50) \n
            \t lambda x, y, z: x >= 0

        Returns
        -------
        np.ndarray
            nodes that meet conditions
        """
        
        xn, yn, zn = self.coord.T

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

        assert isinstance(point, Point)

        xn, yn, zn = self.coord.T

        idx = np.where((xn == point.x) & (yn == point.y) & (zn == point.z))[0]        

        if len(idx) == 0:
            # the previous condition may be too restrictive
            tolerance = 1e-3
            
            # we make sure there is no coordinates = 0
            dec = 10
            decX = np.abs(xn.min()) + dec
            decY = np.abs(yn.min()) + dec
            decZ = np.abs(zn.min()) + dec
            x = point.x + decX
            y = point.y + decY
            z = point.z + decZ            
            
            # get errors between coordinates
            errorX = np.abs((xn-x)/xn)
            errorY = np.abs((yn-y)/yn)
            errorZ = np.abs((zn-z)/zn)
            
            idx = np.where((errorX <= tolerance) & (errorY <= tolerance) & (errorZ <= tolerance))[0]

        return self.__nodes[idx].copy()

    def Get_Nodes_Line(self, line: Line) -> np.ndarray:
        """Returns nodes on the line."""

        assert isinstance(line, Line)
        
        unitVector = line.unitVector

        vect = self.coord-line.coord[0]

        scalarProd = np.einsum('i,ni-> n', unitVector, vect, optimize='optimal')
        crossProd = np.cross(vect, unitVector)
        norm = np.linalg.norm(crossProd, axis=1)

        eps = 1e-12

        idx = np.where((norm<eps) & (scalarProd>=-eps) & (scalarProd<=line.length+eps))[0]

        return self.__nodes[idx].copy()
    
    def Get_Nodes_Domain(self, domain: Domain) -> np.ndarray:
        """Returns nodes in the domain."""

        assert isinstance(domain, Domain)

        xn, yn, zn = self.coord.T

        eps = 1e-12

        idx = np.where( (xn >= domain.pt1.x-eps) & (xn <= domain.pt2.x+eps) &
                        (yn >= domain.pt1.y-eps) & (yn <= domain.pt2.y+eps) &
                        (zn >= domain.pt1.z-eps) & (zn <= domain.pt2.z+eps))[0]
        
        return self.__nodes[idx].copy()

    def Get_Nodes_Circle(self, circle: Circle, onlyOnEdge=False) -> np.ndarray:
        """Returns nodes in the circle."""

        assert isinstance(circle, Circle)

        eps = 1e-12

        vals = np.linalg.norm(self.coord - circle.center.coord, axis=1)

        if onlyOnEdge:
            idx = np.where((vals <= circle.diam/2+eps) & (vals >= circle.diam/2-eps))
        else:
            idx = np.where(vals <= circle.diam/2+eps)    

        return self.__nodes[idx]

    def Get_Nodes_Cylinder(self, circle: Circle, direction=[0,0,1], onlyOnEdge=False) -> np.ndarray:
        """Returns nodes in the cylinder."""

        assert isinstance(circle, Circle)
        
        rotAxis = np.cross(circle.n, direction)
        if np.linalg.norm(rotAxis) <= 1e-12:
            # n == direction
            i = (circle.pt1 - circle.center).coord
            J = Jacobian_Matrix(i,direction)
        else:
            # n != direction
            # (rotAxis, j, direction)
            R = circle.diam/2
            J = Jacobian_Matrix(rotAxis, direction)
            coordN = np.einsum('ij,nj->ni', np.linalg.inv(J), circle.coord - circle.center.coord)
            # (rotAxis, j*cj, direction)
            cj = (R - coordN[:,1].max())/R
            J[:,1] *= cj
        
        eps = 1e-12
        coord = np.einsum('ij,nj->ni', np.linalg.inv(J), self.coord - circle.center.coord)

        vals = np.linalg.norm(coord[:,:2], axis=1)
        if onlyOnEdge:
            idx = np.where((vals <= circle.diam/2+eps) & (vals >= circle.diam/2-eps))
        else:
            idx = np.where(vals <= circle.diam/2+eps)    

        return self.__nodes[idx]
    
    # TODO Get_Nodes_Points
    # use Points.contour also give a normal
    # get all geom contour exept le last one
    # Line -> Plane equation
    # CircleArc -> Cylinder do something like Get_Nodes_Cylinder

    def _Set_Nodes_Tag(self, nodes: np.ndarray, tag: str):
        """Adds a tag to the nodes.

        Parameters
        ----------
        nodes : np.ndarray
            list of nodes
        tag : str
            tag used
        """
        if nodes.size == 0: return
        assert isinstance(tag, str), 'tag must be a string'

        if np.min(nodes) < 0 or np.max(nodes) >= self.__coordGlob.shape[0]:
            raise Exception(f"nodes must be within the range [0, {self.__coordGlob.shape[0]-1}].")

        self.__dict_nodes_tags[tag] = nodes

    @property
    def nodeTags(self) -> list[str]:
        """Returns node tags."""
        return list(self.__dict_nodes_tags.keys())
    
    @property
    def _dict_nodes_tags(self) -> dict[str, np.ndarray]:
        """Dictionary associating tags with nodes."""
        return self.__dict_nodes_tags.copy()

    def _Set_Elements_Tag(self, nodes: np.ndarray, tag: str):
        """Adds a tag to elements associated with nodes

        Parameters
        ----------
        nodes : np.ndarray
            list of nodes
        tag : str
            tag used
        """

        if nodes.size == 0: return
        assert isinstance(tag, str), 'tag must be a string'

        # Retrieve elements associated with nodes
        elements = self.Get_Elements_Nodes(nodes=nodes, exclusively=True)
        if elements.size == 0: return

        if np.min(elements) < 0 or np.max(elements) >= self.Ne:
            raise Exception(f"elements must be within the range [0, {self.Ne-1}].")

        self.__dict_elements_tags[tag] = elements

    @property
    def elementTags(self) -> list[str]:
        """returns element tags."""
        return list(self.__dict_elements_tags.keys())
    
    @property
    def _dict_elements_tags(self) -> dict[str, np.ndarray]:
        """dictionary associating tags with elements."""
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
        """Locates sol on elements"""
        size = self.Nn * self.dim
        if sol.shape[0] == size:
            sol_e = sol[self.assembly_e]
        elif sol.shape[0] == self.Nn:
            sol_e = sol[self.__connect]
        else:
            raise Exception('Wrong dimension')
        
        return sol_e
    
    def Get_pointsInElem(self, coordinates_n: np.ndarray, elem: int) -> np.ndarray:
        """Returns the indexes of the coordinates contained in the element.

        Parameters
        ----------
        coordinates_n : np.ndarray
            coordinates (n, 3)
        elem : int
            element

        Returns
        -------
        np.ndarray
            indexes of coordinates contained in element
        """

        if coordinates_n.size == 0:
            return np.array([])

        dim = self.__dim

        tol = 1e-12
        # tol = 1e-6

        if dim == 0:

            coord = self.coord[self.__connect[elem,0]]

            idx = np.where((coordinates_n[:,0] == coord[0]) & (coordinates_n[:,1] == coord[1]) & (coordinates_n[:,2] == coord[2]))[0]

            return idx

        elif dim == 1:

            coord = self.coord

            p1 = self.__connect[elem,0]
            p2 = self.__connect[elem,1]

            # vector between the points of the segment
            vect = coord[p2] - coord[p1]
            length = np.linalg.norm(vect)
            vect = vect / length

            # vector starting from the first point of the element
            p_n = coordinates_n - coord[p1]

            cross_n = np.cross(e_i, p_n, axisa=0, axisb=1)
            norm_n = np.linalg.norm(cross_n, axis=1)

            dot_n = p_n @ vect
            
            idx = np.where((norm_n <= tol) & (dot_n >= -tol) & (dot_n <= length+tol))[0]

            return idx
        
        elif dim == 2:
            # coordinates_n (n,3)
            # points n
            # corners i [1, nPe]
            
            coord = self.coord
            faces = self.faces[:-1]
            nPe = len(faces)
            connect_e = self.connect[elem, faces]
            corners_i = coord[connect_e]
            
            # Vectors e_i for edge segments (nPe, 3)
            indexReord = np.append(np.arange(1, nPe), 0)
            e_i = coord[connect_e[indexReord]] - corners_i
            e_i = np.einsum("id,i->id", e_i, 1/np.linalg.norm(e_i, axis=1), optimize="optimal")

            # normal vector to element face
            n_i = np.cross(e_i[0], -e_i[-1])

            # (n, i, 3)
            coordinates_n_i = coordinates_n[:, np.newaxis].repeat(nPe, 1)

            # Construct p vectors from corners
            p_n_i = coordinates_n_i - corners_i
            
            cross_n_i = np.cross(e_i, p_n_i, axisa=1, axisb=2)
            test_n_i = cross_n_i @ n_i >= -tol
            # Return the index of nodes around the element that meet all conditions
            test_n = np.sum(test_n_i, 1)
            idx = np.where(test_n == nPe)[0]

            return idx
        
        elif dim == 3:
        
            faces = self.faces
            nbFaces = self.nbFaces
            coord = self.coord[self.__connect[elem]]

            if self.elemType is ElemType.PRISM6:
                faces = np.array(faces)
                faces = np.array([faces[np.arange(0,4)],
                                  faces[np.arange(4,8)],
                                  faces[np.arange(8,12)],
                                  faces[np.arange(12,15)],
                                  faces[np.arange(15,18)]], dtype=object)
            elif self.elemType is ElemType.PRISM15:
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

            i_f = coord[p1_f]-coord[p0_f]
            i_f = np.einsum("ni,n->ni", i_f, 1/np.linalg.norm(i_f, axis=1), optimize="optimal")

            j_f = coord[p2_f]-coord[p0_f]
            j_f = np.einsum("ni,n->ni", j_f, 1/np.linalg.norm(j_f, axis=1), optimize="optimal")

            n_f = np.cross(i_f, j_f, 1, 1)
            n_f = np.einsum("ni,n->ni", n_f, 1/np.linalg.norm(n_f, axis=1), optimize="optimal")

            coordinates_n_i = coordinates_n[:, np.newaxis].repeat(nbFaces, 1)

            v_f = coordinates_n_i - coord[p0_f]

            t_f = np.einsum("nfi,fi->nf", v_f, n_f, optimize="optimal") >= -tol

            filtre = np.sum(t_f, 1)

            idx = np.where(filtre == nbFaces)[0]

            return idx

    def Get_Mapping(self, coordinates_n: np.ndarray, elements_e=None, needCoordinates=True):
        """Locates coordinates within elements.

        Returns
        -------

            - detectedNodes : The nodes that have been identified within the detected elements with shape=(Nn,).
            - detectedElements_e : The elements in which the nodes have been detected with shape=(Ne,).
            - connect_e_n : The connectivity matrix that includes the nodes identified in each element with shape=(Ne, ?).
                The "?" indicates that the array may have varying dimensions along axis=1.
            - coordInElem_n : The coordinates of the identified nodes, expressed in the reference element's (ξ,η,ζ) coordinate system.
                This is applicable only if needCoordinates is set to True.
        """
        
        if elements_e is None:
            elements_e = np.arange(self.Ne, dtype=int)

        assert coordinates_n.shape[1] == 3, "Must be of dimension (n, 3)."

        return self.__Get_Mapping(coordinates_n, elements_e, needCoordinates)

    def __Get_Mapping(self, coordinates_n: np.ndarray, elements_e: np.ndarray, needCoordinates=True):
        """Locates coordinates within elements.

        Returns
        -------

            - detectedNodes : The nodes that have been identified within the detected elements with shape=(Nn,).
            - detectedElements_e : The elements in which the nodes have been detected with shape=(Ne,).
            - connect_e_n : The connectivity matrix that includes the nodes identified in each element with shape=(Ne, ?).
                The "?" indicates that the array may have varying dimensions along axis=1.
            - coordInElem_n : The coordinates of the identified nodes, expressed in the reference element's (ξ,η,ζ) coordinate system.
                This is applicable only if needCoordinates is set to True.
        """
        
        dim = self.dim
        connect = self.connect
        coord = self.coord
        
        # Initialize lists
        detectedNodes: list[int] = []
        # Elements where nodes have been identified
        detectedElements_e: list[int] = []
        # connectivity matrix containing the nodes used by the elements
        connect_e_n: list[list[int]] = []

        # Calculate the number of times a coordinate appears
        dims = np.max(coordinates_n, 0) - np.min(coordinates_n, 0) + 1
        # here dims is a 3d array used in __Get_coordoNear to check if coordinates_n comes from an image/grid
        # If the coordinates come from an image/grid, the _Get_coordoNear function will be faster.

        if needCoordinates:
            # Here we want to know the coordinates of the nodes in
            # the reference element's (ξ,η) coordinate system.
            coordInElem_n = np.zeros_like(coordinates_n[:,:dim], dtype=float)

            # get coordinates in the reference element
            # get groupElem datas
            inDim = self.inDim
            sysCoord_e = self.sysCoord_e # basis transformation matrix for each element
            # This matrix can be used to project points with (x, y, z) coordinates into the element's (i, j, k) coordinate system.
            matrixType = MatrixType.mass
            jacobian_e_pg = self.Get_jacobian_e_pg(matrixType, absoluteValues=False)
            invF_e_pg = self.Get_invF_e_pg(matrixType)
            dN_tild = self._dN()
            xiOrigin = self.origin # origin of the reference element (ξ0,η0)
            
            # Check whether iterative resolution is required
            # calculate the ratio between jacob max and min to detect if the element is distorted
            diff_e = jacobian_e_pg.max(1) * 1/jacobian_e_pg.min(1)
            error_e = np.abs(1 - diff_e) # a perfect element has an error max <= 1e-12
            # A distorted element exhibits a maximum error greater than zero.
            useIterative_e = error_e > 1e-12
        else:
            coordInElem_n = None

        def ResearchFunction(e: int):
    
            # get element's node coordinates (x, y, z)
            coordElem: np.ndarray = coord[connect[e]]

            # Retrieve indexes in coordinates_n that are within the element's bounds
            idxNearElem = self.__Get_coordoNear(coordinates_n, coordElem, dims)

            # Return the index of idxNearElem that satisfies all the specified conditions.
            idxInElem = self.Get_pointsInElem(coordinates_n[idxNearElem], e)

            if idxInElem.size == 0:
                # here no nodes have been detected in the element
                return
                                
            # Nodes contained within element e.
            nodesInElement = idxNearElem[idxInElem]

            # Save de detected nodes elements and connectivity matrix
            detectedNodes.extend(nodesInElement)
            connect_e_n.append(nodesInElement)
            detectedElements_e.append(e)

            if needCoordinates:
                # Inverse mapping is required here,
                # i.e., to determine the position in the reference element (ξ, η) from the physical coordinates (x, y).
                # This is particularly relevant for elements with multiple integration points (e.g., QUAD4, TRI6, TRI10, ..),
                # i.e., all elements that can be distorted and have a Jacobian ratio different from 1.

                # Project (x, y, z) coordinates into the element's (i, j, k) coordinate system if dim != inDim.
                # its the case when a 2D mesh is in 3D space
                coordElemBase = coordElem.copy()
                coordinatesBase_n: np.ndarray = coordinates_n[nodesInElement].copy()
                if dim != inDim:
                    coordElemBase = coordElemBase @ sysCoord_e[e]
                    coordinatesBase_n = coordinatesBase_n @ sysCoord_e[e]
                
                # Origin of the element in (x, y, z) coordinates.
                x0  = coordElemBase[0,:dim]
                # Coordinates of the n points in (x, y, z).
                xP_n = coordinatesBase_n[:,:dim]

                if not useIterative_e[e]:
                    # The fastest method, available only for undistorted meshes.
                    xiP: np.ndarray = xiOrigin + (xP_n - x0) @ invF_e_pg[e,0]
                         
                else:
                    # This is the most time-consuming method.
                    # We need to construct the Jacobian matrices here.
                    def Eval(xi: np.ndarray, xP):
                        dN = _GroupElem._Evaluates_Functions(dN_tild, xi.reshape(1, -1))
                        F = dN[0] @ coordElemBase[:,:dim] # jacobian matrix [J]                  
                        J = x0 + (xi - xiOrigin) @ F - xP # cost function
                        return J

                    xiP = []
                    for xP in xP_n:
                        res = least_squares(Eval, 0*xP, args=(xP,))
                        xiP.append(res.x)

                    xiP = np.array(xiP)

                # xiP are the n coordinates of the n points in (ξ,η,ζ).
                coordInElem_n[nodesInElement,:] = xiP.copy()

        [ResearchFunction(e) for e in elements_e]

        assert len(detectedElements_e) == len(connect_e_n), "The number of detected elements must match the number of lines in connect_e_n."

        ar_detectedNodes = np.asarray(detectedNodes,dtype=int)
        ar_detectedElements_e = np.asarray(detectedElements_e,dtype=int)
        ar_connect_e_n = np.asarray(connect_e_n,dtype=object)

        return ar_detectedNodes, ar_detectedElements_e, ar_connect_e_n, coordInElem_n
    
    def __Get_coordoNear(self, coordinates_n: np.ndarray, coordElem: np.ndarray, dims: np.ndarray) -> np.ndarray:
        """Get indexes in coordinates_n that are within the coordElem's bounds.

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
        coordinatesInImage = coordinates_n.dtype==int and testShape and nZ == 1

        if coordinatesInImage:
            # here coordinates_n are pixels

            xe = np.arange(np.floor(coordElem[:,0].min()), np.ceil(coordElem[:,0].max()), dtype=int)
            ye = np.arange(np.floor(coordElem[:,1].min()), np.ceil(coordElem[:,1].max()), dtype=int)
            Xe, Ye = np.meshgrid(xe,ye)

            grid_elements_coordinates = np.concatenate(([Ye.ravel()],[Xe.ravel()]))
            idx = np.ravel_multi_index(grid_elements_coordinates, (nY, nX))
            # if something goes wrong, check that the mesh is correctly positioned in the image
            
        else:

            xn, yn, zn = coordinates_n.T
            xe, ye, ze = coordElem.T
            
            idx = np.where((xn >= np.min(xe)) & (xn <= np.max(xe)) &
                            (yn >= np.min(ye)) & (yn <= np.max(ye)) & 
                            (zn >= np.min(ze)) & (zn <= np.max(ze)))[0]

        return idx

# --------------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------------

# elems
from .elems._point import POINT
from .elems._seg import SEG2, SEG3, SEG4, SEG5
from .elems._tri import TRI3, TRI6, TRI10, TRI15
from .elems._quad import QUAD4, QUAD8, QUAD9
from .elems._tetra import TETRA4, TETRA10
from .elems._hexa import HEXA8, HEXA20, HEXA27
from .elems._prism import PRISM6, PRISM15, PRISM18

class GroupElemFactory:

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
        elif gmshId == 27:
            elemType = ElemType.SEG5; nPe = 5; dim = 1; order = 4; nbFaces = 0; nbCorners = 2
        elif gmshId == 2:
            elemType = ElemType.TRI3; nPe = 3; dim = 2; order = 1; nbFaces = 1; nbCorners = 3
        elif gmshId == 9:
            elemType = ElemType.TRI6; nPe = 6; dim = 2; order = 2; nbFaces = 1; nbCorners = 3
        elif gmshId == 21:
            elemType = ElemType.TRI10; nPe = 10; dim = 2; order = 3; nbFaces = 1; nbCorners = 3
        elif gmshId == 23:
            elemType = ElemType.TRI15; nPe = 15; dim = 2; order = 4; nbFaces = 1; nbCorners = 3
        elif gmshId == 3:
            elemType = ElemType.QUAD4; nPe = 4; dim = 2; order = 1; nbFaces = 1; nbCorners = 4
        elif gmshId == 16:
            elemType = ElemType.QUAD8; nPe = 8; dim = 2; order = 2; nbFaces = 1; nbCorners = 4
        elif gmshId == 10:
            elemType = ElemType.QUAD9; nPe = 9; dim = 2; order = 2; nbFaces = 1; nbCorners = 4
        elif gmshId == 4:
            elemType = ElemType.TETRA4; nPe = 4; dim = 3; order = 1; nbFaces = 4; nbCorners = 4
        elif gmshId == 11:
            elemType = ElemType.TETRA10; nPe = 10; dim = 3; order = 2; nbFaces = 4; nbCorners = 4
        elif gmshId == 5:
            elemType = ElemType.HEXA8; nPe = 8; dim = 3; order = 1; nbFaces = 6; nbCorners = 8
        elif gmshId == 17:
            elemType = ElemType.HEXA20; nPe = 20; dim = 3; order = 2; nbFaces = 6; nbCorners = 8
        elif gmshId == 12:
            elemType = ElemType.HEXA27; nPe = 27; dim = 3; order = 2; nbFaces = 6; nbCorners = 8
        elif gmshId == 6:
            elemType = ElemType.PRISM6; nPe = 6; dim = 3; order = 1; nbFaces = 5; nbCorners = 6
        elif gmshId == 18:
            elemType = ElemType.PRISM15; nPe = 15; dim = 3; order = 2; nbFaces = 5; nbCorners = 6
        elif gmshId == 13:
            elemType = ElemType.PRISM18; nPe = 18; dim = 3; order = 2; nbFaces = 5; nbCorners = 6
        # elif gmshId == 7:
        #     elemType = ElemType.PYRA5; nPe = 5; dim = 3; order = 1; nbFaces = 5; nbCorners = 5
        # elif gmshId == 19:
        #     elemType = ElemType.PYRA13; nPe = 13; dim = 3; order = 2; nbFaces = 5; nbCorners = 5
        # elif gmshId == 14:
        #     elemType = ElemType.PYRA14; nPe = 14; dim = 3; order = 2; nbFaces = 5; nbCorners = 5
        else: 
            raise Exception("Element type unknown.")
            
        return elemType, nPe, dim, order, nbFaces, nbCorners
    
    @staticmethod
    def Create(gmshId: int, connect: np.ndarray, coordGlob: np.ndarray, nodes: np.ndarray) -> _GroupElem:
        """Creates an element group
        
        Parameters
        ----------
        gmshId : int
            id gmsh
        connect : np.ndarray
            connection matrix storing nodes for each element (Ne, nPe)
        coordGlob : np.ndarray
            nodes coordinates
        nodes : np.ndarray
            nodes used by the element group
        
        Returns
        -------
        GroupeElem
            the element group
        """

        params = (gmshId, connect, coordGlob, nodes)

        elemType = GroupElemFactory.Get_ElemInFos(gmshId)[0]
        
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
        elif elemType == ElemType.TRI15:
            return TRI15(*params)
        elif elemType == ElemType.QUAD4:
            return QUAD4(*params)
        elif elemType == ElemType.QUAD8:
            return QUAD8(*params)
        elif elemType == ElemType.QUAD9:
                return QUAD9(*params)
        elif elemType == ElemType.TETRA4:
            return TETRA4(*params)
        elif elemType == ElemType.TETRA10:
            return TETRA10(*params)
        elif elemType == ElemType.HEXA8:
            return HEXA8(*params)
        elif elemType == ElemType.HEXA20:
            return HEXA20(*params)
        elif elemType == ElemType.HEXA27:
            return HEXA27(*params)
        elif elemType == ElemType.PRISM6:
            return PRISM6(*params)
        elif elemType == ElemType.PRISM15:
            return PRISM15(*params)
        elif elemType == ElemType.PRISM18:
            return PRISM18(*params)
        else:
            raise Exception("Element type unknown.")