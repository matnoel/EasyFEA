import numpy as np
import scipy.sparse as sp
from types import LambdaType

from Geom import *
from GroupElem import GroupElem, ElemType, MatrixType
import TicTac

class Mesh:

    def __init__(self, dict_groupElem: dict[ElemType,GroupElem], verbosity=True):
        """Setup the mesh

        Parameters
        ----------
        dict_groupElem : dict[ElemType,GroupElem]
            element group dictionary
        verbosity : bool, optional
            can write in terminal, by default True
        """

        list_GroupElem = []
        dim=0
        for grp in dict_groupElem.values():
            if grp.dim > dim:
                # Here we guarantee that the mesh element used is the one with the largest dimension
                dim = grp.dim
                self.__groupElem = grp
            list_GroupElem.append(grp)

        self.__dim = self.__groupElem.dim
        self.__dict_groupElem = dict_groupElem

        self.__verbosity = verbosity
        """The mesh can write to the console"""
        
        if self.__verbosity:
            print(self)

    def _ResetMatrix(self) -> None:
        [groupElem._InitMatrix() for groupElem in self.Get_list_groupElem()]            
    
    def __str__(self) -> str:
        text = f"\nElement type : {self.elemType}"
        text += f"\nNe = {self.Ne}, Nn = {self.Nn}, dof = {self.Nn*self.__dim}"
        return text
    
    def Get_list_groupElem(self, dim=None) -> list[GroupElem]:
        """Mesh element group list"""
        if dim == None:
            dim = self.__dim
            
        list_groupElem = [grp for grp in self.__dict_groupElem.values() if grp.dim == dim]
        list_groupElem.reverse() # reverse the list

        return list_groupElem

    @property
    def dict_groupElem(self) -> dict[ElemType, GroupElem]:
        """dictionary containing all the element groups in the mesh"""
        return self.__dict_groupElem

    @property
    def groupElem(self) -> GroupElem:
        """Main mesh element group"""
        return self.__groupElem
    
    @property
    def elemType(self) -> ElemType:
        """Element type used for meshing"""
        return self.groupElem.elemType
    
    @property
    def Ne(self) -> int:
        """Number of elements in the mesh"""
        return self.groupElem.Ne
    
    @property
    def Nn(self, dim=None) -> int:
        """Number of nodes in the mesh"""
        return self.groupElem.Nn
    
    @property
    def dim(self):
        """Mesh dimension"""
        return self.__dim

    @property
    def inDim(self):
        """Dimension in which the mesh is located
        A 2D mesh can be oriented in space"""
        return self.__groupElem.inDim
    
    @property
    def nPe(self) -> int:
        """Nodes per element"""
        return self.groupElem.nPe
    
    @property
    def coordo(self) -> np.ndarray:
        """Node coordinates matrix (Nn,3) for the main groupElem"""
        return self.groupElem.coordo
    
    @property
    def nodes(self) -> np.ndarray:
        """Mesh nodes"""
        return self.groupElem.nodes

    @property
    def coordoGlob(self) -> np.ndarray:
        """Global mesh coordinate matrix (mesh.Nn, 3)\n
        Contains all mesh coordinates"""
        return self.groupElem.coordoGlob

    @property
    def connect(self) -> np.ndarray:
        """Connectivity matrix (Ne, nPe)"""
        return self.groupElem.connect
    
    def Get_connect_n_e(self) -> sp.csr_matrix:
        """Sparse matrix of zeros and ones with ones when the node has the element either
        such that: values_n = connect_n_e * values_e

        (Nn,1) = (Nn,Ne) * (Ne,1)
        """
        return self.groupElem.Get_connect_n_e()
    
    # Affichage
    
    @property
    def dict_connect_Triangle(self) -> dict[ElemType, np.ndarray]:
        """Transform the connectivity matrix to pass it to the trisurf function in 2D.
        For example, for a quadrangle, we construct two triangles
        for a 6-node triangle, 4 triangles are constructed

        Returns a dictionary by type"""
        return self.groupElem.Get_dict_connect_Triangle()
    
    def Get_dict_connect_Faces(self) -> dict[ElemType, np.ndarray]:        
        """Retrieves face-building nodes and returns faces for each element type.
        """

        dict_connect_faces = {}
        
        for elemType, groupElem in self.dict_groupElem.items():

            indexesFaces = groupElem.indexesFaces

            if self.__groupElem.elemType == ElemType.PRISM6 and elemType == ElemType.TRI3:
                indexesFaces.append(indexesFaces[0])
            elif self.__groupElem.elemType == ElemType.PRISM15 and elemType == ElemType.TRI6:
                indexesFaces.extend([indexesFaces[0]]*2)

            dict_connect_faces[elemType] = groupElem.connect[:, indexesFaces]
            
        return dict_connect_faces

    # Assemblage des matrices 

    @property
    def assembly_e(self) -> np.ndarray:
        """assembly matrix (Ne, nPe*dim)\n
        Allows rigi matrix to be positioned in the global matrix"""
        return self.groupElem.assembly_e
    
    def Get_assembly_e(self, dof_n: int) -> np.ndarray:
        """Assembly matrix for specified dof_n (Ne, nPe*dof_n)
        Used to position matrices in the global matrix"""
        return self.groupElem.Get_assembly_e(dof_n)

    @property
    def linesVector_e(self) -> np.ndarray:
        """lines to fill the assembly matrix in vector (displacement)"""
        return self.Get_linesVector_e(self.__dim)
    
    def Get_linesVector_e(self, dof_n: int) -> np.ndarray:
        """lines to fill the assembly matrix in vector"""
        assembly_e = self.Get_assembly_e(dof_n)
        nPe = self.nPe
        Ne = self.Ne
        return np.repeat(assembly_e, nPe*dof_n).reshape((Ne,-1))

    @property
    def columnsVector_e(self) -> np.ndarray:
        """columns to fill the assembly matrix in vector (displacement)"""
        return self.Get_columnsVector_e(self.__dim)
    
    def Get_columnsVector_e(self, dof_n: int) -> np.ndarray:
        """columns to fill the vector assembly matrix"""
        assembly_e = self.Get_assembly_e(dof_n)
        nPe = self.nPe
        Ne = self.Ne
        return np.repeat(assembly_e, nPe*dof_n, axis=0).reshape((Ne,-1))

    @property
    def linesScalar_e(self) -> np.ndarray:
        """lines to fill the assembly matrix in scalar form (damage or thermal)"""
        connect = self.connect
        nPe = self.nPe
        Ne = self.Ne
        return np.repeat(connect, nPe).reshape((Ne,-1))

    @property
    def columnsScalar_e(self) -> np.ndarray:
        """columns to fill the assembly matrix in scalar form (damage or thermal)"""
        connect = self.connect
        nPe = self.nPe
        Ne = self.Ne
        return np.repeat(connect, nPe, axis=0).reshape((Ne,-1))    

    # Calculation of areas, volumes and quadratic moments etc ...
    @property
    def area(self) -> float:
        if self.dim in [0,1]: return
        areas = [group2D.area for group2D in self.Get_list_groupElem(2)]
        return np.sum(areas)

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
    
    def Get_meshSize_e(self) -> np.ndarray:
        """Returns the mesh size for each element."""

        # recovery of the physical group and coordinates
        groupElem = self.groupElem
        coordo = groupElem.coordo

        # indexes to access segments of each element
        indexesSegments = groupElem.indexesSegments
        segments_e = groupElem.connect[:, indexesSegments]

        # Calculates the length of each segment (s) of the mesh elements (e).
        h_e_s = np.linalg.norm(coordo[segments_e[:,:,1]] - coordo[segments_e[:,:,0]], axis=2)
        # average segment size per element
        h_e = np.mean(h_e_s, axis=1)
        
        return h_e

    # Construction of elementary matrices
    
    def Get_nPg(self, matrixType: MatrixType) -> np.ndarray:
        """number of integration points"""
        return self.groupElem.Get_gauss(matrixType).nPg

    def Get_weight_pg(self, matrixType: MatrixType) -> np.ndarray:
        """integration point weights"""
        return self.groupElem.Get_gauss(matrixType).weights

    def Get_jacobian_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Returns the jacobians\n
        variation in size (length, area or volume) between the reference element and the real element
        """
        return self.groupElem.Get_jacobian_e_pg(matrixType)
    
    def Get_N_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluated shape functions (pg), in the base (ksi, eta . . . )
        [N1, N2, . . . ,Nn]
        """
        return self.groupElem.Get_N_pg(matrixType)

    def Get_N_vector_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Shape functions in reference element for vector (npg, dim, npe*dim)\n
        Matrix of shape functions in reference element (ksi, eta)\n
        [N1(ksi,eta) 0 N2(ksi,eta) 0 Nn(ksi,eta) 0 \n
        0 N1(ksi,eta) 0 N2(ksi,eta) 0 Nn(ksi,eta)]\n
        """
        return self.groupElem.Get_N_pg_rep(matrixType, self.__dim)

    def Get_dN_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Derivation of shape functions in real base (epij)\n
        [dN1,x dN2,x dNn,x\n
        dN1,y dN2,y dNn,y]\n        
        """
        return self.groupElem.Get_dN_e_pg(matrixType)

    def Get_dNv_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Derivation of beam shape functions in real base (epij)\n
        [dNv1,x dNv2,x dNvn,x\n
        dNv1,y dNv2,y dNvn,y]\n
        """
        return self.groupElem.Get_dNv_e_pg(matrixType)
    
    def Get_ddNv_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Derivation (2) of beam shape functions in real base (epij)\n
        [dNv1,xx dNv2,xx dNvn,xx\n
        dNv1,yy dNv2,yy dNvn,yy]\n
        """
        return self.groupElem.Get_ddNv_e_pg(matrixType)

    def Get_ddN_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Derivation (2) of shape functions in real base (epij)\n
        [dN1,xx dN2,xx dNn,xx\n
        dN1,yy dN2,yy dNn,yy]\n
        """
        return self.groupElem.Get_ddN_e_pg(matrixType)

    def Get_B_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Derivation of shape functions in the real base for the displacement problem (e, pg, (3 or 6), nPe*dim)\n
        2D example:\n
        [dN1,x 0 dN2,x 0 dNn,x 0\n
        0 dN1,y 0 dN2,y 0 dNn,y\n
        dN1,y dN1,x dN2,y dN2,x dN3,y dN3,x]\n

        (epij) In the element base and in Kelvin Mandel
        """
        return self.groupElem.Get_B_e_pg(matrixType)

    def Get_leftDispPart(self, matrixType: MatrixType) -> np.ndarray:
        """Left side of local displacement matrices\n
        Ku_e = jacobian_e_pg * weight_pg * B_e_pg' * c_e_pg * B_e_pg\n
        
        Returns (epij) -> jacobian_e_pg * weight_pg * B_e_pg'.
        """
        return self.groupElem.Get_leftDispPart(matrixType)
    
    def Get_ReactionPart_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Returns the part that builds the reaction term (scalar).
        ReactionPart_e_pg = jacobian_e_pg * weight_pg * r_e_pg * N_pg' * N_pg\n
        
        Returns -> jacobian_e_pg * weight_pg * N_pg' * N_pg
        """
        return self.groupElem.Get_ReactionPart_e_pg(matrixType)

    def Get_DiffusePart_e_pg(self, matrixType: MatrixType, A: np.ndarray) -> np.ndarray:
        """Returns the part that builds the diffusion term (scalar).
        DiffusePart_e_pg = jacobian_e_pg * weight_pg * k * dN_e_pg' * A * dN_e_pg\n
        
        Returns -> jacobian_e_pg * weight_pg * dN_e_pg' * A * dN_e_pg
        """
        return self.groupElem.Get_DiffusePart_e_pg(matrixType, A)

    def Get_SourcePart_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Returns the part that builds the source term (scalar).
        SourcePart_e_pg = jacobian_e_pg, weight_pg, f_e_pg, N_pg'\n
        
        Returns -> jacobian_e_pg, weight_pg, N_pg'
        """
        return self.groupElem.Get_SourcePart_e_pg(matrixType)
    
    # Node recovery

    def Nodes_Conditions(self, lambdaFunction: LambdaType) -> np.ndarray:
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
            nodes that meet the specified conditions.
        """
        return self.groupElem.Get_Nodes_Conditions(lambdaFunction)
    
    def Nodes_Point(self, point: Point) -> np.ndarray:
        """Returns nodes on the point."""
        return self.groupElem.Get_Nodes_Point(point)

    def Nodes_Line(self, line: Line) -> np.ndarray:
        """Returns the nodes on the line."""
        return self.groupElem.Get_Nodes_Line(line)

    def Nodes_Domain(self, domain: Domain) -> np.ndarray:
        """Returns nodes in the domain."""
        return self.groupElem.Get_Nodes_Domain(domain)
    
    def Nodes_Circle(self, circle: Circle) -> np.ndarray:
        """Returns the nodes in the circle."""
        return self.groupElem.Get_Nodes_Circle(circle)

    def Nodes_Cylindre(self, circle: Circle, direction=[0,0,1]) -> np.ndarray:
        """Returns the nodes in the cylinder."""
        return self.groupElem.Get_Nodes_Cylindre(circle, direction)

    def Elements_Nodes(self, nodes: np.ndarray, exclusively=True):
        """Returns elements that exclusively or not use the specified nodes."""
        elements = self.groupElem.Get_Elements_Nodes(nodes=nodes, exclusively=exclusively)
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
        elif "beam" in tag:
            dim = 1
        
        return dim

    def Nodes_Tags(self, tags: list[str]) -> np.ndarray:
        """Returns node associated with the tag."""
        nodes = []
        [nodes.extend(grp.Get_Nodes_Tag(tag)) for tag in tags for grp in self.Get_list_groupElem(Mesh.__Dim_For_Tag(tag))]

        return np.unique(nodes)

    def Elements_Tags(self, tags: list[str]) -> np.ndarray:
        """Returns elements associated with the tag."""
        elements = []
        [elements.extend(grp.Get_Elements_Tag(tag)) for tag in tags for grp in self.Get_list_groupElem(Mesh.__Dim_For_Tag(tag))]

        return np.unique(elements)

    def Locates_sol_e(self, sol: np.ndarray) -> np.ndarray:
        """locates sol on elements"""
        return self.groupElem.Locates_sol_e(sol)
    
def Calc_New_meshSize_n(mesh: Mesh, erreur_e: np.ndarray, coef=1/2) -> np.ndarray:
    """Returns the scalar field (at nodes) to be used to refine the mesh.
    
    meshSize = (coef - 1) * err / max(err) + 1

    Parameters
    ----------
    mesh : Mesh
        support mesh
    error_e : np.ndarray
        error evaluated on elements
    coef : float, optional
        mesh size division ratio, by default 1/2

    Returns
    -------
    np.ndarray
        meshSize_n, new mesh size at nodes (Nn)
    """

    assert mesh.Ne == erreur_e.size, "erreur_e must be an array of dim Ne"

    h_e = mesh.Get_meshSize_e()
    
    meshSize_e = (coef-1)/erreur_e.max() * erreur_e + 1

    import Simulations
    meshSize_n = Simulations._Simu.Results_NodeInterpolation(mesh, meshSize_e * h_e)

    return meshSize_n
    
def Calc_projector(oldMesh: Mesh, newMesh: Mesh) -> sp.csr_matrix:
    """Builds the matrix used to project the solution from the old mesh to the new mesh.
    newU = proj * oldU\n
    (newNn) = (newNn x oldNn) (oldNn) 

    Parameters
    ----------
    oldMesh : Mesh
        old mesh 
    newMesh : Mesh
        new mesh

    Returns
    -------
    sp.csr_matrix
        dimensional projection matrix (newMesh.Nn, oldMesh.Nn)
    """

    assert oldMesh.dim == newMesh.dim, "Mesh dimensions must be the same."
    dim = oldMesh.dim

    tic = TicTac.Tic()

    # recovery of nodes detected in old mesh elements
    # connnectivity of these nodes in the elements
    # position of nodes in reference element
    nodes, connect_e_n, coordo_n = oldMesh.groupElem.Get_Nodes_Connect_CoordoInElemRef(newMesh.coordo)

    tic.Tac("Mesh", "Mapping between meshes", False)

    # Evaluation of shape functions
    Ntild = oldMesh.groupElem._Ntild()        
    nPe = oldMesh.groupElem.nPe
    phi_n_nPe = np.zeros((coordo_n.shape[0], nPe))
    for n in range(nPe):
        if dim == 1:
            phi_n_nPe[:,n] = Ntild[n,0](coordo_n[:,0])
        elif dim == 2:
            phi_n_nPe[:,n] = Ntild[n,0](coordo_n[:,0], coordo_n[:,1])
        elif dim == 3:
            phi_n_nPe[:,n] = Ntild[n,0](coordo_n[:,0], coordo_n[:,1], coordo_n[:,2])
    
    # Here we detect whether nodes appear more than once
    counts = np.unique(nodes, return_counts=True)[1]
    idxSup1 = np.where(counts > 1)[0]
    if idxSup1.size > 0:
        # if nodes are used several times, divide the shape function values by the number of appearances. At the end, do like an average
        phi_n_nPe[idxSup1] = np.einsum("ni,n->ni", phi_n_nPe[idxSup1], 1/counts[idxSup1], optimize="optimal")

    # Projector construction
    connect_e = oldMesh.connect
    lignes = []
    colonnes = []
    valeurs = []
    nodesElem = []
    def FuncExtend_Proj(e: int, nodes: np.ndarray):
        nodesElem.extend(nodes)
        valeurs.extend(phi_n_nPe[nodes].reshape(-1))
        lignes.extend(np.repeat(nodes, nPe))
        colonnes.extend(np.asarray(list(connect_e[e]) * nodes.size))

    [FuncExtend_Proj(e, nodes) for e, nodes in enumerate(connect_e_n)]
    
    proj = sp.csr_matrix((valeurs, (lignes, colonnes)), (newMesh.Nn, oldMesh.Nn), dtype=float)

    tic.Tac("Mesh", "Projector construction", False)

    return proj.tocsr()