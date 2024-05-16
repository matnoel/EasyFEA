# Copyright (C) 2021-2024 Université Gustave Eiffel. All rights reserved.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the mesh class.
This class allows you to manipulate different groups of elements.
These element groups are used to construct finite element matrices."""

import numpy as np
import scipy.sparse as sp
import copy
from typing import Callable

# utilities
from ..utilities import Display
from ..utilities import Tic
from ..utilities._observers import Observable
# fem
from ._utils import ElemType, MatrixType
from ._group_elems import _GroupElem
# others
from ..Geoms import *

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
        
        Nn = self.coordGlob.shape[0]
        usedNodes = set(self.connect.ravel())
        nodes = set(range(Nn))
        orphanNodes = list(nodes - usedNodes)
        self.__orphanNodes: list[int] = orphanNodes
        if len(orphanNodes) > 0 and verbosity:            
            Display.MyPrintError("WARNING: Orphan nodes have been detected (stored in mesh.orphanNodes).")

    def _ResetMatrix(self) -> None:
        """Reset matrix for each groupElem"""
        [groupElem._InitMatrix() for groupElem in self.Get_list_groupElem()]

    def __str__(self) -> str:
        """Return a string representation of the mesh."""
        text = f"\nElement type: {self.elemType}"
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
    def coord(self) -> np.ndarray:
        """Node coordinates matrix (Nn,3) for the main groupElem"""
        return self.groupElem.coord
    
    def copy(self):
        newMesh = copy.deepcopy(self)
        return newMesh

    def translate(self, dx: float=0.0, dy: float=0.0, dz: float=0.0) -> None:
        """Translate the mesh coordinates."""
        oldCoord = self.coordGlob
        newCoord = oldCoord + np.array([dx, dy, dz])
        for grp in self.dict_groupElem.values():
            grp.coordGlob = newCoord
        self._Notify('The mesh has been modified')

    
    def Rotate(self, theta: float, center: tuple=(0,0,0), direction: tuple=(0,0,1)) -> None:        
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

        oldCoord = self.coord
        newCoord = Rotate_coord(oldCoord, theta, center, direction)
        for grp in self.dict_groupElem.values():
            grp.coordGlob = newCoord
        self._Notify('The mesh has been modified')

    def Symmetry(self, point=(0,0,0), n=(1,0,0)) -> None:
        """Symmetrise the mesh coordinates with a plane.

        Parameters
        ----------
        point : tuple, optional
            a point belonging to the plane, by default (0,0,0)
        n : tuple, optional
            normal to the plane, by default (1,0,0)
        """

        oldCoord = self.coord
        newCoord = Symmetry_coord(oldCoord, point, n)
        for grp in self.dict_groupElem.values():
            grp.coordGlob = newCoord
        self._Notify('The mesh has been modified')

    @property
    def nodes(self) -> np.ndarray:
        """Mesh nodes"""
        return self.groupElem.nodes

    @property
    def coordGlob(self) -> np.ndarray:
        """Global mesh coordinate matrix (mesh.Nn, 3)\n
        Contains all mesh coordinates"""
        return self.groupElem.coordGlob
    
    @coordGlob.setter
    def coordGlob(self, coordo: np.ndarray) -> None:
        if coordo.shape == self.coordGlob.shape:
            for grp in self.dict_groupElem.values():
                grp.coordGlob = coordo

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
        
    def Get_normals(self, nodes: np.ndarray=None, displacementMatrix:np.ndarray=None) -> np.ndarray:
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
            n_e = groupElem._Get_sysCoord_e(displacementMatrix)[elements, :, idx]

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

    def Nodes_Circle(self, circle: Circle, onlyOnCircle=False) -> np.ndarray:
        """Returns the nodes in the circle."""
        return self.groupElem.Get_Nodes_Circle(circle, onlyOnCircle)

    def Nodes_Cylinder(self, circle: Circle, direction=[0, 0, 1], onlyOnEdge=False) -> np.ndarray:
        """Returns the nodes in the cylinder."""
        return self.groupElem.Get_Nodes_Cylinder(circle, direction, onlyOnEdge)

    def Elements_Nodes(self, nodes: np.ndarray, exclusively=True, neighborLayer:int=1):
        """Returns elements that exclusively or not use the specified nodes."""
        # elements = self.groupElem.Get_Elements_Nodes(nodes=nodes, exclusively=exclusively)
        
        for i in range(neighborLayer):
            elements = self.groupElem.Get_Elements_Nodes(nodes=nodes, exclusively=exclusively)
            nodes = list(set(np.ravel(self.connect[elements])))

            if neighborLayer > 1 and elements.size == self.Ne:                
                Display.MyPrint("All the neighbors have been found.")
                break

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
            Display.MyPrintError("There is no tags available in the mesh, so don't forget to use the '_Set_PhysicalGroups()' function before meshing your geometry with '_Meshing()' in the gmsh interface 'Gmsh_Interface'.")
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
            Display.MyPrintError("There is no tags available in the mesh, so don't forget to use the '_Set_PhysicalGroups()' function before meshing your geometry with '_Meshing()' in the gmsh interface 'Gmsh_Interface'.")
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

        tic = Tic()

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
            corners: np.ndarray = self.coordGlob[corners]
        

        nCorners = len(corners) # number of corners
        nEdges = nCorners//2 # number of edges

        nodes1: list[int] = []
        nodes2: list[int] = []
        nNodes: list[int] = []

        coordo = self.coord

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

    def Get_meshSize(self, doMean=True) -> np.ndarray:
        """Returns the mesh size for each element or for each element and each segment.\n
        return meshSize_e if doMean else meshSize_e_s"""
        # recovery of the physical group and coordinates
        groupElem = self.groupElem
        coordo = groupElem.coord

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
        
    def Get_Quality(self, criteria: str ='aspect', nodeValues=False) -> np.ndarray:
        """Calculates mesh quality [0, 1] (bad, good).

        Parameters
        ----------
        criteria : str, optional
            criterion used, by default 'aspect'\n
            - "aspect": hMin / hMax, ratio between minimum and maximum element length\n
            - "angular": angleMin / angleMax, ratio between the minimum and maximum angle of an element\n
            - "gamma": 2 rci/rcc, ratio between the radius of the inscribed circle and the circumscribed circle multiplied by 2. Useful for triangular elements.\n
            - "jacobian": jMax / jMin, ratio between the maximum jacobian and the minimum jacobian. Useful for higher-order elements.

        nodeValues : bool, optional
            Calculates values on nodes, by default False

        Returns
        -------
        np.ndarray
            Mesh quality between 0 and 1.
        """

        groupElem = self.groupElem
        coordo = groupElem.coordGlob
        connect = groupElem.connect

        # length of each segments
        h_e_s = self.Get_meshSize(False)

        # perimeter
        p_e = np.sum(h_e_s, -1)
        
        # area
        area_e = groupElem.area_e
        
        if groupElem.dim == 2:
            # calculate the angle in each corners of 2d elements
            angle_e_s = np.zeros((groupElem.Ne, groupElem.nbCorners), float)

            for c in range(groupElem.nbCorners):
                
                next = c+1 if c+1 < groupElem.nbCorners else 0
                prev = -1 if c == 0 else c-1
                
                p0_e = coordo[connect[:, c]]
                p1_e = coordo[connect[:, next]]
                p2_e = coordo[connect[:, prev]]

                angle_e = AngleBetween_a_b(p1_e-p0_e, p2_e-p0_e)

                angle_e_s[:,c] = np.abs(angle_e)

        if criteria == 'gamma':
            # only available for triangular elements

            if groupElem.elemType not in [ElemType.TRI3, ElemType.TRI6, ElemType.TRI10]:
                Display.MyPrintError("The gamma criterion is only available for triangular elements.")
                return None

            # inscribed circle
            # https://fr.wikipedia.org/wiki/Cercles_inscrit_et_exinscrits_d%27un_triangle
            rci_e = 2*area_e/p_e
            # circumscribed circle
            # https://fr.wikipedia.org/wiki/Cercle_circonscrit_%C3%A0_un_triangle
            rcc_e = p_e/2/np.sum(np.sin(angle_e_s), 1)

            values_e = rci_e/rcc_e * 2

        elif criteria == "aspect":
            # hMin / hMax
            values_e = np.min(h_e_s, 1) / np.max(h_e_s, 1)

        elif criteria == "angular":
            # only available for 2d elements
            if groupElem.dim != 2:
                Display.MyPrintError("The angular criterion is only available for 2D elements.")
                return None

            # min(angle) / max(angle)
            values_e = np.min(angle_e_s, 1) / np.max(angle_e_s, 1)

        elif criteria == "jacobian":
            # jMin / jMax
            jacobian_e_pg = groupElem.Get_jacobian_e_pg(MatrixType.mass)
            values_e = np.max(jacobian_e_pg, 1) / np.min(jacobian_e_pg, 1)

        else:
            Display.MyPrintError(f"The criterion ({criteria}) is not implemented")

        if nodeValues:
            return self.Get_Node_Values(values_e)
        else:
            return np.asarray(values_e)

    def Get_New_meshSize_n(self, error_e: np.ndarray, coef=1/2) -> np.ndarray:
        """Returns the scalar field (at nodes) to be used to refine the mesh.

        meshSize = (coef - 1) * err / max(err) + 1

        Parameters
        ----------
        error_e : np.ndarray
            error evaluated on elements
        coef : float, optional
            mesh size division ratio, by default 1/2

        Returns
        -------
        np.ndarray
            meshSize_n, new mesh size at nodes (Nn)
        """

        assert self.Ne == error_e.size, "error_e must be an array of dim Ne"

        h_e = self.Get_meshSize()

        meshSize_e = (coef - 1) / error_e.max() * error_e + 1
        
        meshSize_n = self.Get_Node_Values(meshSize_e * h_e)

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

    tic = Tic()
    
    detectedNodes, detectedElements_e, connect_e_n, coordo_n = oldMesh.groupElem.Get_Mapping(newMesh.coord)
    # detectedNodes (size(connect_e_n)) are the nodes detected in detectedElements_e
    # detectedElements_e (e) are the elements for which we have detected the nodes
    # connect_e_n (e, ?) is the connectivity matrix containing the nodes detected in each element
    # coordInElem_n (coordinates.shape[0]) are the coordinates of the nodes detected in the base of the reference element.

    tic.Tac("Mesh", "Mapping between meshes", False)

    # Evaluation of shape functions
    Ntild = oldMesh.groupElem._Ntild()        
    nPe = oldMesh.groupElem.nPe
    phi_n_nPe = np.zeros((coordo_n.shape[0], nPe)) # functions evaluated at identified coordinates
    for n in range(nPe):
        # *coordo_n.T give a list for every direction *(xis, etas, ..)
        phi_n_nPe[:,n] = Ntild[n,0](*coordo_n.T)

    # Check that the sum of the shape functions is 1  
    testSum1 = (np.sum(phi_n_nPe) - phi_n_nPe.size)/phi_n_nPe.size <= 1e-12
    assert testSum1

    # Here we detect whether nodes appear more than once
    #   I don't know how to modify this function because np.unique can take a long time.
    counts = np.unique(detectedNodes, return_counts=True)[1]
    nodesSup1 = np.where(counts > 1)[0]
    # nodesSup1 are nodes that have been detected several times.
    if nodesSup1.size > 0:
        # divide the shape function values by the number of appearances.
        # Its like doing an average on shapes functions
        phi_n_nPe[nodesSup1] = np.einsum("ni,n->ni", phi_n_nPe[nodesSup1], 1/counts[nodesSup1], optimize="optimal")

    # Building the projector
    # This projector is a hollow matrix of dimension (newMesh.Nn, oldMesh.Nn)
    connect_e = oldMesh.connect
    lines: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    def FuncExtend_Proj(element: int, nodes: np.ndarray):
        values.extend(np.ravel(phi_n_nPe[nodes]))
        lines.extend(np.repeat(nodes, nPe))
        columns.extend(np.asarray(list(connect_e[element]) * nodes.size))

    [FuncExtend_Proj(element, connect) for element, connect in zip(detectedElements_e, connect_e_n)]

    proj = sp.csr_matrix((values, (lines, columns)), (newMesh.Nn, oldMesh.Nn), dtype=float)

    # Here we'll impose the exact values of overlapping nodes (which have the same coordinate) on the nodes.
    proj = proj.tolil()
    
    # get back the corners to link nodes
    # here we assume that the points are identical
    newCorners = newMesh.Get_list_groupElem(0)[0].nodes
    oldCorners = oldMesh.Get_list_groupElem(0)[0].nodes
    # link nodes
    for newNode, oldNode in zip(newCorners, oldCorners):
        proj[newNode,:] = 0
        proj[newNode, oldNode] = 1

    nodesExact = np.where((phi_n_nPe >= 1-1e-12) & (phi_n_nPe <= 1+1e-12))[0]
    # nodesExact nodes exact are nodes for which a shape function has detected 1.
    #   These are the nodes detected in an element corner.

    nodesExact = list(set(nodesExact) - set(newCorners))
    for node in nodesExact:
        oldNode = oldMesh.Nodes_Point(Point(*newMesh.coord[node]))
        if oldNode.size == 0: continue
        proj[node,:] = 0
        proj[node, oldNode] = 1

    # from EasyFEA import Display
    # dim = oldMesh.dim
    # ax = Display.Plot_Mesh(oldMesh)
    # ax.scatter(*newMesh.coordo[nodesSup1,:dim].T,label='sup1')
    # ax.scatter(*newMesh.coordo[newCorners,:dim].T,label='corners')
    # ax.scatter(*newMesh.coordo[nodesExact,:dim].T,label='exact')
    # ax.legend()

    tic.Tac("Mesh", "Projector construction", False)

    return proj.tocsr()

def Mesh_Optim(DoMesh: Callable[[str], Mesh], folder: str, criteria:str='aspect', quality=.8, ratio: float=0.7, iterMax=20, coef:float=1/2) -> tuple[Mesh, float]:
    """Optimize the mesh using the given criterion.

    Parameters
    ----------
    DoMesh : Callable[[str], Mesh]
        Function that constructs the mesh and takes a .pos file as argument for mesh optimization. The function must return a Mesh.
    folder : str
        Folder in which .pos files are created and then deleted.
    criteria : str, optional
        criterion used, by default 'aspect'\n
        - "aspect": hMin / hMax, ratio between minimum and maximum element length\n
        - "angular": angleMin / angleMax, ratio between the minimum and maximum angle of an element\n
        - "gamma": 2 rci/rcc, ratio between the radius of the inscribed circle and the circumscribed circle multiplied by 2. Useful for triangular elements.\n
        - "jacobian": jMax / jMin, ratio between the maximum jacobian and the minimum jacobian. Useful for higher-order elements.
    quality : float, optional
        Target quality, by default .8
    ratio : float, optional
        The target ratio of mesh elements that must respect the specified quality, by default 0.7 (must be in [0,1])
    iterMax : int, optional
        Maximum number of iterations, by default 20
    coef : float, optional
        mesh size division ratio, by default 1/2

    Returns
    -------
    tuple[Mesh, float]
        optimized mesh size and ratio
    """
    
    from EasyFEA import Folder, Mesher

    targetRatio = ratio
    assert targetRatio > 0 and targetRatio <= 1, "targetRatio must be in ]0, 1]"

    i = -1
    ratio = 0
    optimGeom = None
    # max=1
    while ratio <= targetRatio and i <= iterMax:

        i += 1

        mesh = DoMesh(optimGeom)

        if i > 0:
            # remove previous .pos file
            Folder.os.remove(optimGeom)        
        
        # mesh quality calculation
        qual_e = mesh.Get_Quality(criteria, False)

        # the element ratio that respects quality
        ratio = np.where(qual_e >= quality)[0].size / mesh.Ne

        if ratio == 1:
            return mesh, ratio

        print(f'ratio = {ratio*100:.3f} %')
        
        # # assigns max quality for elements that exceed quality
        # qual_e[qual_e >= quality] = quality
        
        # calculates the relative error between element quality and desired quality
        error_e = np.abs(qual_e-quality)/quality

        # calculates the new mesh size for the associated error
        meshSize_n = mesh.Get_New_meshSize_n(error_e, coef)

        # builds the .pos file that will be used to refine the mesh
        optimGeom = Mesher().Create_posFile(mesh.coord, meshSize_n, folder, f"pos{i}")

    if Folder.Exists(optimGeom):
        # remove last .pos file
        Folder.os.remove(optimGeom)

    return mesh, ratio