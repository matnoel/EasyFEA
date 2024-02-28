"""Mesh module. Allows you to manipulate different element groups. These element groups are used to construct finite element matrix."""

import numpy as np
import scipy.sparse as sp
import copy

from Geoms import *
from GroupElems import _GroupElem, ElemType, MatrixType
import TicTac

from Observers import Observable

class Mesh(Observable):
    """A mesh uses several groups of elements. For example, a mesh with cubes (HEXA8) uses :
    - POINT (dim=0)
    - SEG2 (dim=1)
    - QUAD4 (dim=2)
    - HEXA8 (dim=3)
    """

    def __init__(self, dict_groupElem: dict[ElemType, _GroupElem], verbosity=False):
        """Setup the mesh.

        Parameters
        ----------
        dict_groupElem : dict[ElemType, _GroupElem]
            element group dictionary
        verbosity : bool, optional
            can write in terminal, by default True
        """

        list_GroupElem = []        
        dim = 0
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
        
        Nn = self.coordoGlob.shape[0]
        usedNodes = set(self.connect.ravel())
        nodes = set(range(Nn))
        orphanNodes = list(nodes - usedNodes)
        self.__orphanNodes: list[int] = orphanNodes
        if len(orphanNodes) > 0 and verbosity:
            from Display import myPrintError
            myPrintError("WARNING: Orphan nodes have been detected (stored in mesh.orphanNodes).")

    def _ResetMatrix(self) -> None:
        """Reset matrix for each groupElem"""
        [groupElem._InitMatrix() for groupElem in self.Get_list_groupElem()]

    def __str__(self) -> str:
        """Return a string representation of the mesh."""
        text = f"\nElement type : {self.elemType}"
        text += f"\nNe = {self.Ne}, Nn = {self.Nn}, dof = {self.Nn * self.__dim}"
        return text

    def Get_list_groupElem(self, dim=None) -> list[_GroupElem]:
        """Get the list of mesh element groups.

        Parameters
        ----------
        dim : int, optional
            The dimension of elements to retrieve, by default None (uses the main mesh dimension).

        Returns
        -------
        list[_GroupElem]
            A list of _GroupElem objects with the specified dimension.
        """
        if dim is None:
            dim = self.__dim

        list_groupElem = [grp for grp in self.__dict_groupElem.values() if grp.dim == dim]
        list_groupElem.reverse()  # reverse the list

        return list_groupElem
    
    @property
    def orphanNodes(self) -> list[int]:
        """Nodes not connected to the main mesh element group"""
        return self.__orphanNodes

    @property
    def dict_groupElem(self) -> dict[ElemType, _GroupElem]:
        """dictionary containing all the element groups in the mesh"""
        return self.__dict_groupElem

    @property
    def groupElem(self) -> _GroupElem:
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
        """Dimension in which the mesh is located.\n
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
    
    def copy(self):
        newMesh = copy.deepcopy(self)
        return newMesh

    def translate(self, dx: float=0.0, dy: float=0.0, dz: float=0.0) -> None:
        """Translate the mesh coordinates."""
        oldCoord = self.coordoGlob
        newCoord = oldCoord + np.array([dx, dy, dz])
        for grp in self.dict_groupElem.values():
            grp.coordoGlob = newCoord
        self._notify('The mesh has been modified')

    
    def rotate(self, theta: float, center: tuple=(0,0,0), direction: tuple=(0,0,1)) -> None:        
        """Rotate the mesh coordinates around an axis.

        Parameters
        ----------        
        theta : float
            rotation angle [deg] 
        center : tuple, optional
            rotation center, by default (0,0,0)
        direction : tuple, optional
            rotation direction, by default (0,0,1)
        """

        oldCoord = self.coordo
        newCoord = Rotate_coordo(oldCoord, theta, center, direction)
        for grp in self.dict_groupElem.values():
            grp.coordoGlob = newCoord
        self._notify('The mesh has been modified')

    def symmetry(self, point=(0,0,0), n=(1,0,0)) -> None:
        """Symmetrise the mesh coordinates with a plane.

        Parameters
        ----------
        point : tuple, optional
            a point belonging to the plane, by default (0,0,0)
        n : tuple, optional
            normal to the plane, by default (1,0,0)
        """

        oldCoord = self.coordo
        newCoord = Symmetry_coordo(oldCoord, point, n)
        for grp in self.dict_groupElem.values():
            grp.coordoGlob = newCoord
        self._notify('The mesh has been modified')

    @property
    def nodes(self) -> np.ndarray:
        """Mesh nodes"""
        return self.groupElem.nodes

    @property
    def coordoGlob(self) -> np.ndarray:
        """Global mesh coordinate matrix (mesh.Nn, 3)\n
        Contains all mesh coordinates"""
        return self.groupElem.coordoGlob
    
    @coordoGlob.setter
    def coordoGlob(self, coordo: np.ndarray) -> None:
        if coordo.shape == self.coordoGlob.shape:
            for grp in self.dict_groupElem.values():
                grp.coordoGlob = coordo

    @property
    def connect(self) -> np.ndarray:
        """Connectivity matrix (Ne, nPe)"""
        return self.groupElem.connect
    
    @property
    def verbosity(self) -> bool:
        """The mesh can write to the console"""
        return self.__verbosity

    def Get_connect_n_e(self) -> sp.csr_matrix:
        """Sparse matrix of zeros and ones with ones when the node has the element either
        such that: values_n = connect_n_e * values_e

        (Nn,1) = (Nn,Ne) * (Ne,1)
        """
        return self.groupElem.Get_connect_n_e()

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
        linesVector_e = np.repeat(assembly_e, nPe * dof_n).reshape((Ne, -1))
        return linesVector_e

    @property
    def columnsVector_e(self) -> np.ndarray:
        """columns to fill the assembly matrix in vector (displacement)"""
        return self.Get_columnsVector_e(self.__dim)

    def Get_columnsVector_e(self, dof_n: int) -> np.ndarray:
        """columns to fill the vector assembly matrix"""
        assembly_e = self.Get_assembly_e(dof_n)
        nPe = self.nPe
        Ne = self.Ne
        columnsVector_e = np.repeat(assembly_e, nPe * dof_n, axis=0).reshape((Ne, -1))
        return columnsVector_e

    @property
    def linesScalar_e(self) -> np.ndarray:
        """lines to fill the assembly matrix in scalar form (damage or thermal)"""
        connect = self.connect
        nPe = self.nPe
        Ne = self.Ne
        return np.repeat(connect, nPe).reshape((Ne, -1))

    @property
    def columnsScalar_e(self) -> np.ndarray:
        """columns to fill the assembly matrix in scalar form (damage or thermal)"""
        connect = self.connect
        nPe = self.nPe
        Ne = self.Ne
        return np.repeat(connect, nPe, axis=0).reshape((Ne, -1))

    @property
    def length(self) -> float:
        """Calculate the total length of the mesh."""
        if self.dim < 1:
            return None
        lengths = [group1D.length for group1D in self.Get_list_groupElem(1)]
        return np.sum(lengths)
    
    @property
    def area(self) -> float:
        """Calculate the total area of the mesh."""
        if self.dim < 2:
            return None
        areas = [group2D.area for group2D in self.Get_list_groupElem(2)]
        return np.sum(areas)

    @property
    def volume(self) -> float:
        """Calculate the total volume of the mesh."""
        if self.dim != 3:
            return None
        volumes = [group3D.volume for group3D in self.Get_list_groupElem(3)]
        return np.sum(volumes)
    
    @property
    def center(self) -> np.ndarray:
        """Center of mass / barycenter / inertia center"""
        return self.groupElem.center

    def Get_meshSize(self, doMean=True) -> np.ndarray:
        """Returns the mesh size for each element or for each element and each segment.\n
        return meshSize_e if doMean else meshSize_e_s"""
        # recovery of the physical group and coordinates
        groupElem = self.groupElem
        coordo = groupElem.coordo

        # indexes to access segments of each element
        segments = groupElem.segments
        segments_e = groupElem.connect[:, segments]

        # Calculates the length of each segment (s) of the mesh elements (e).
        h_e_s = np.linalg.norm(coordo[segments_e[:, :, 1]] - coordo[segments_e[:, :, 0]], axis=2)

        if doMean:
            # average segment size per element        
            return np.mean(h_e_s, axis=1)
        else:
            return h_e_s
        
    def Get_normals(self, nodes: np.ndarray=None) -> np.ndarray:
        """Get normal vectors and the nodes belonging to the edge of the mesh.\n
        return normals, nodes."""

        if nodes is None:
            nodes = self.nodes

        assert nodes.max() <= self.Nn

        dim = self.dim
        idx = 2 if dim == 3 else 1 # normal vectors position in sysCoord_e

        list_normals = []
        list_nodes: list[int] = [] # used nodes in nodes

        # for each elements on the boundary
        for groupElem in self.Get_list_groupElem(dim-1):

            elements = groupElem.Get_Elements_Nodes(nodes, True)

            if elements.size == 0: continue

            elementsNodes = np.ravel(groupElem.connect[elements])

            usedNodes = np.asarray(list(set(elementsNodes)), dtype=int)

            if usedNodes.size == 0: continue

            # get the normal vectors for elements
            n_e = groupElem.sysCoord_e[elements, :, idx]

            # here we want to get the normal vector on the nodes
            # need to get the nodes connectivity
            connect_n_e = groupElem.Get_connect_n_e()[usedNodes, :].tocsc()[:, elements].tocsr()
            # get the number of elements per nodes
            sum = np.ravel(connect_n_e.sum(1))            
            # get the normal vector on normal
            normal_n = np.einsum('ni,n->ni',connect_n_e @ n_e, 1/sum, optimize='optimal')

            # append the values on each direction and add nodes
            list_normals.append(normal_n)
            list_nodes.extend(usedNodes)

        nodes = np.asarray(list_nodes, dtype=int)
        normals = np.concatenate(list_normals, 0, dtype=float)

        return normals, nodes

    # Construction of elementary matrices

    def Get_nPg(self, matrixType: MatrixType) -> np.ndarray:
        """number of integration points"""
        return self.groupElem.Get_gauss(matrixType).nPg

    def Get_weight_pg(self, matrixType: MatrixType) -> np.ndarray:
        """integration point weights"""
        return self.groupElem.Get_gauss(matrixType).weights

    def Get_jacobian_e_pg(self, matrixType: MatrixType, absoluteValues=True) -> np.ndarray:
        """Returns the jacobians\n
        variation in size (length, area or volume) between the reference element and the real element
        """
        return self.groupElem.Get_jacobian_e_pg(matrixType, absoluteValues)

    def Get_N_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluate shape functions in local coordinates.\n
        [N1, N2, . . . ,Nn]\n
        (pg, nPe)
        """
        return self.groupElem.Get_N_pg(matrixType)

    def Get_N_vector_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Matrix of shape functions in local coordinates\n
        [N1 0 N2 0 Nn 0 \n
        0 N1 0 N2 0 Nn]\n
        (pg, dim, npe*dim)
        """
        return self.groupElem.Get_N_pg_rep(matrixType, self.__dim)

    def Get_dN_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluate shape functions first derivatives in the global coordinates.\n
        [Ni,x . . . Nn,x\n
        Ni,y ... Nn,y]\n
        (e, pg, dim, nPe)\n
        """
        return self.groupElem.Get_dN_e_pg(matrixType)

    def Get_ddN_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Evaluate shape functions second derivatives in the global coordinates.\n
        [Ni,xx . . . Nn,xx\n
        Ni,yy ... Nn,yy]\n
        (e, pg, dim, nPe)\n
        """
        return self.groupElem.Get_ddN_e_pg(matrixType)

    def Get_B_e_pg(self, matrixType: MatrixType) -> np.ndarray:
        """Construct the matrix used to calculate deformations from displacements.\n
        WARNING: Use Kelvin Mandel Notation\n
        [N1,x 0 N2,x 0 Nn,x 0\n
        0 N1,y 0 N2,y 0 Nn,y\n
        N1,y N1,x N2,y N2,x N3,y N3,x]\n
        (e, pg, (3 or 6), nPe*dim)        
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

    # Nodes recovery

    def Nodes_Conditions(self, func) -> np.ndarray:
        """Returns nodes that meet the specified conditions.

        Parameters
        ----------
        func : function 
            Function using the x, y and z nodes coordinates and returning boolean values.

            examples :
            \t lambda x, y, z: (x < 40) & (x > 20) & (y<10) \n
            \t lambda x, y, z: (x == 40) | (x == 50) \n
            \t lambda x, y, z: x >= 0

        Returns
        -------
        np.ndarray
            nodes that meet the specified conditions.
        """
        return self.groupElem.Get_Nodes_Conditions(func)

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

    def Nodes_Cylinder(self, circle: Circle, direction=[0, 0, 1]) -> np.ndarray:
        """Returns the nodes in the cylinder."""
        return self.groupElem.Get_Nodes_Cylinder(circle, direction)

    def Elements_Nodes(self, nodes: np.ndarray, exclusively=True):
        """Returns elements that exclusively or not use the specified nodes."""
        elements = self.groupElem.Get_Elements_Nodes(nodes=nodes, exclusively=exclusively)
        return elements

    def Nodes_Tags(self, tags: list[str]) -> np.ndarray:
        """Returns node associated with the tag."""
        nodes = []

        if isinstance(tags, str):
            tags = [tags]

        # get dictionnary linking tags to nodes
        dict_nodes = {}
        [dict_nodes.update(grp._dict_nodes_tags) for grp in self.dict_groupElem.values()]
        # add nodes belonging to the tags

        if len(dict_nodes) == 0:
            from Display import myPrintError
            myPrintError("There is no tags available in the mesh, so don't forget to use the '_Set_PhysicalGroups()' function before meshing your geometry with '_Meshing()' in the gmsh interface 'Gmsh_Interface'.")
            return np.asarray([])

        [nodes.extend(dict_nodes[tag]) for tag in tags]
        # make sure that that the list is unique
        nodes = np.asarray(list(set(nodes)), dtype=int)

        return nodes

    def Elements_Tags(self, tags: list[str]) -> np.ndarray:
        """Returns elements associated with the tag."""
        elements = []

        if isinstance(tags, str):
            tags = [tags]

        # get dictionnary linking tags to elements
        dict_elements = {}
        [dict_elements.update(grp._dict_elements_tags) for grp in self.dict_groupElem.values()]

        if len(dict_elements) == 0:
            from Display import myPrintError
            myPrintError("There is no tags available in the mesh, so don't forget to use the '_Set_PhysicalGroups()' function before meshing your geometry with '_Meshing()' in the gmsh interface 'Gmsh_Interface'.")
            return np.asarray([])

        # add elements belonging to the tags
        [elements.extend(dict_elements[tag]) for tag in tags]
        # make sure that that the list is unique
        elements = np.asarray(list(set(elements)), dtype=int)

        return elements

    def Locates_sol_e(self, sol: np.ndarray) -> np.ndarray:
        """Locates solution on elements."""
        return self.groupElem.Locates_sol_e(sol)
    
    def Get_Node_Values(self, result_e: np.ndarray) -> np.ndarray:
        """Get node values from element values.\n
        The value of a node is calculated by averaging the values of the surrounding elements.

        Parameters
        ----------
        mesh : Mesh
            mesh
        result_e : np.ndarray
            element values (Ne, i)

        Returns
        -------
        np.ndarray
            nodes values (Nn, i)
        """

        assert self.Ne == result_e.shape[0], "Must be of size (Ne,i)"

        tic = TicTac.Tic()

        Ne = self.Ne
        Nn = self.Nn

        if len(result_e.shape) == 1:
            # In this case it is a 1d vector
            # we need to reshape as
            result_e = result_e.reshape(Ne,1)
            isDim1 = True
        else:
            isDim1 = False
        
        nCols = result_e.shape[1]

        result_n = np.zeros((Nn, nCols), dtype=float)

        # connectivity of the nodes
        connect_n_e = self.Get_connect_n_e()
        # get elements per ndoes
        elements_n = np.reshape(np.sum(connect_n_e, axis=1), (Nn, 1))

        for c in range(nCols):
            values_e = result_e[:, c].reshape(Ne,1)
            values_n = (connect_n_e @ values_e) * 1/elements_n
            result_n[:,c] = values_n.ravel()

        tic.Tac("PostProcessing","Element to nodes values", False)

        if isDim1:
            return result_n.ravel()
        else:
            return result_n
        
    def Get_Paired_Nodes(self, corners: np.ndarray, plot=False) -> np.ndarray:
        """Get the paired nodes used to construct periodic boundary conditions.

        Parameters
        ----------
        corners : np.ndarray
            Either nodes or nodes coordinates.

        plot : bool, optional
            Set whether to plot the link between nodes; defaults to False.

        Returns
        -------
        np.ndarray
            Paired nodes, a 2-column matrix storing the paired nodes (n, 2).
        """

        corners = np.asarray(corners)

        if corners.ndim == 1:
            # corners are nodes
            # corners become the corners coordinates
            corners: np.ndarray = self.coordoGlob[corners]
        

        nCorners = len(corners) # number of corners
        nEdges = nCorners//2 # number of edges

        nodes1: list[int] = []
        nodes2: list[int] = []
        nNodes: list[int] = []

        coordo = self.coordo

        for c, corner in enumerate(corners):

            # here corner and next_corner are coordinates
            
            if c+1 == nCorners:
                next_corner = corners[0]                
            else:
                next_corner = corners[c+1]

            line = next_corner - corner # construct line between 2 corners
            lineLength = np.linalg.norm(line) # length of the line
            vect = Normalize_vect(line) # normalized vector between the edge corners
            vect_i = coordo - corner # vector coordinates from the first corner of the edge
            scalarProduct = np.einsum('ni,i', vect_i, vect, optimize="optimal")
            crossProduct = np.cross(vect_i, vect)
            norm = np.linalg.norm(crossProduct, axis=1)

            eps=1e-12
            nodes = np.where((norm<eps) & (scalarProduct>=-eps) & (scalarProduct<=lineLength+eps))[0]
            # norm<eps : must be on the line formed by corner and next corner
            # scalarProduct>=-eps : points must belong to the line
            # scalarProduct<=lineLength+eps : points must belong to the line

            # sort the nodes along the lines and take
            # remove the first and the last nodes with [1:-1]
            nodes: np.ndarray = nodes[np.argsort(scalarProduct[nodes])][1:-1]

            if c+1 > nEdges:
                # revert the nodes order
                nodes = nodes[::-1]
                nodes2.extend(nodes)
            else:
                nodes1.extend(nodes)
                if nodes.size > 0:
                    nNodes.append(nodes.size)

        assert len(nodes1) != 0 and len(nodes2) != 0, "No nodes detected"

        assert len(nodes1) == len(nodes2), 'Edges must contain the same number of nodes.'

        paired_nodes = np.array([nodes1, nodes2]).T

        if plot:
            import Display            
            inDim = self.inDim

            if inDim == 3:
                from mpl_toolkits.mplot3d.art3d import Line3DCollection
            else:
                from matplotlib.collections import LineCollection

            ax = Display.Plot_Mesh(self, alpha=0, title='Periodic boundary conditions')

            # nEdges = np.min([len(nNodes)//2, nEdges])

            start = 0
            
            for edge in range(len(nNodes)):

                start += 0 if edge == 0 else nNodes[edge-1]

                edge_node = paired_nodes[start:start+nNodes[edge]]

                lines = coordo[edge_node, :inDim]
                if inDim == 3:
                    pc = ax.scatter(lines[:,:,0], lines[:,:,1], lines[:,:,2], label=f'edges{edge}')
                    ax.add_collection3d(Line3DCollection(lines, edgecolor=pc.get_edgecolor()))
                else:
                    pc = ax.scatter(lines[:,:,0], lines[:,:,1], label=f'edges{edge}')
                    ax.add_collection(LineCollection(lines, edgecolor=pc.get_edgecolor()))
                
            ax.legend()

        return paired_nodes

def Calc_New_meshSize_n(mesh: Mesh, error_e: np.ndarray, coef=1 / 2) -> np.ndarray:
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

    assert mesh.Ne == error_e.size, "error_e must be an array of dim Ne"

    h_e = mesh.Get_meshSize()

    meshSize_e = (coef - 1) / error_e.max() * error_e + 1
    
    meshSize_n = mesh.Get_Node_Values(meshSize_e * h_e)

    return meshSize_n

def Calc_projector(oldMesh: Mesh, newMesh: Mesh) -> sp.csr_matrix:
    """Builds the matrix used to project the solution from the old mesh to the new mesh.
    newU = proj * oldU\n
    (newNn) = (newNn x oldNn) (oldNn) 
    (newNn) = (newNn x oldNn) (oldNn)
    Parameters
    ----------
    oldMesh : Mesh
        old mesh 
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
    # connectivity of these nodes in the elements for the new mesh
    # position of nodes in reference element
    nodes, connect_e_n, coordo_n = oldMesh.groupElem.Get_Mapping(newMesh.coordo)

    tic.Tac("Mesh", "Mapping between meshes", False)

    # Evaluation of shape functions
    Ntild = oldMesh.groupElem._Ntild()        
    nPe = oldMesh.groupElem.nPe
    phi_n_nPe = np.zeros((coordo_n.shape[0], nPe)) # functions evaluated at identified coordinates

    for n in range(nPe):
        phi_n_nPe[:,n] = Ntild[n,0](*coordo_n.T)
        # *coordo_n.T give a list for every direction *(ksis, etas, ..)    
    
    # Here we check that the evaluated shape functions give values between [0, 1].
    valMax = phi_n_nPe.max()
    valMin = phi_n_nPe.min()
    # assert valMin > -1e-12 and valMax <= 1+1e-12, "the coordinates of the nodes in the reference elements have been incorrectly evaluated"

    nodesExact = np.where((phi_n_nPe >= 1-1e-12) & (phi_n_nPe <= 1+1e-12))[0]

    if valMin < -1e-12 or valMax >= 1+1e-12:
        phi_n_nPe[phi_n_nPe>1] = 1
        # phi_n_nPe[phi_n_nPe<0] = 0
        print("ERROR in PROJ")

    # Here we detect whether nodes appear more than once
    # can't be change for the moment
    counts = np.unique(nodes, return_counts=True)[1]
    nodesSup1 = np.where(counts > 1)[0]
    if nodesSup1.size > 0:
        # detect if notdes are used severral times
        # divide the shape function values by the number of appearances.
        # Its like doing an average on shapes functions
        phi_n_nPe[nodesSup1] = np.einsum("ni,n->ni", phi_n_nPe[nodesSup1], 1/counts[nodesSup1], optimize="optimal")

    # Projector construction
    connect_e = oldMesh.connect
    lines = []
    columns = []
    values = []
    nodesElem = []
    def FuncExtend_Proj(e: int, nodes: np.ndarray):
        nodesElem.extend(nodes)
        values.extend(phi_n_nPe[nodes].ravel())
        lines.extend(np.repeat(nodes, nPe))
        columns.extend(np.asarray(list(connect_e[e]) * nodes.size))

    [FuncExtend_Proj(e, nodes) for e, nodes in enumerate(connect_e_n)]

    proj = sp.csr_matrix((values, (lines, columns)), (newMesh.Nn, oldMesh.Nn), dtype=float)

    proj = proj.tolil()

    # get back the corners to link nodes
    newCorners = newMesh.Get_list_groupElem(0)[0].nodes
    oldCorners = oldMesh.Get_list_groupElem(0)[0].nodes
    # link nodes
    for newNode, oldNode in zip(newCorners, oldCorners):
        proj[newNode,:] = 0
        proj[newNode, oldNode] = 1

    nodesExact = list(set(nodesExact) - set(newCorners))

    tt = phi_n_nPe[nodesExact]
    for node in nodesExact:
        oldNode = oldMesh.Nodes_Point(Point(*newMesh.coordo[node]))
        if oldNode.size == 0: continue
        proj[node,:] = 0
        proj[node, oldNode] = 1

    # import Display
    # ax = Display.Plot_Mesh(oldMesh)
    # ax.scatter(*newMesh.coordo[nodesSup1,:dim].T,label='sup1')
    # ax.scatter(*newMesh.coordo[newCorners,:dim].T,label='corners')
    # ax.scatter(*newMesh.coordo[nodesExact,:dim].T,label='exact')
    # ax.legend()

    tic.Tac("Mesh", "Projector construction", False)

    return proj.tocsr()