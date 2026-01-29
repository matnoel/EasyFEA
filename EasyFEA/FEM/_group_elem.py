# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

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
from scipy import sparse, spatial
import numpy as np
from typing import Callable, Optional, TYPE_CHECKING

# fem
from ._gauss import Gauss
from ._linalg import FeArray, Trace, Transpose, Det, Inv

# utils
from ._utils import ElemType, MatrixType

# # others
from ..Geoms import Point, Domain, Line, Circle
from ..Geoms._utils import AsPoint
from ..Geoms import Jacobian_Matrix, Normalize

from ..Utilities import _types, _params
from ..Utilities._cache import cache_computed_values, clear_cached_computed_values

if TYPE_CHECKING:
    from ..Models.Beam import BeamStructure


class _GroupElem(ABC):
    """The `_GroupElem` base class, from which all element types inherit."""

    def __init__(
        self, gmshId: int, connect: _types.IntArray, coordGlob: _types.FloatArray
    ):
        """Creates a goup of elements.

        Parameters
        ----------
        gmshId : int
            gmsh id
        connect : _types.IntArray
            connectivity matrix
        coordGlob : _types.FloatArray
            coordinate matrix (contains all mesh coordinates)
        """

        self.__gmshId = gmshId

        elemType, nPe, dim, order, Nvertex, Nedge, Nface, Nvolume = (
            GroupElemFactory.Get_ElemInFos(gmshId)
        )

        self.__elemType = elemType
        self.__nPe = nPe
        self.__dim = dim
        self.__order = order
        self.__Nvertex = Nvertex
        self.__Nedge = Nedge
        self.__Nface = Nface
        self.__Nvolume = Nvolume

        # Elements
        if connect.size != 0:  # connect can be empty
            assert (
                connect.ndim == 2 and connect.shape[1] == nPe
            ), "connect must be a (Ne, nPe) array."
        self.__connect = connect
        self.__connect_n_e: sparse.csr_matrix = None

        # Ensure coordGlob is a (Nn, 3) array
        if coordGlob.size != 0:  # coordGlob can be empty
            error = "Must be a (Nn, 3) array."
            assert coordGlob.ndim == 2 and coordGlob.shape[1] == 3, error
        self.__coordGlob = coordGlob

        # Nodes
        nodes = np.asarray(list(set(connect.ravel())), dtype=int)
        Ncoords = coordGlob.shape[0]
        if nodes.size != 0:
            error = f"Nodes {nodes[nodes > Ncoords]} has not corresponding entry in the coordGlob array."
            assert nodes.max() + 1 <= Ncoords, error
        self.__nodes = nodes

        # Set the paritionned data.
        self._Set_partitionned_data(self.elements, self.nodes)

        # dictionnary associated with tags on elements or nodes
        self.__dict_nodes_tags: dict[str, _types.IntArray] = {}
        self.__dict_elements_tags: dict[str, _types.IntArray] = {}

    def _InitMatrix(self) -> None:
        """Initializes matrix dictionaries for finite element construction"""
        # Dictionaries for each matrix type
        clear_cached_computed_values(self)

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
    def topology(self) -> str:
        """element topology"""
        return "".join(char for char in self.__elemType if not char.isdigit())

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
            x, y, z = np.abs(self.coord.T)
            if np.max(y) == 0 and np.max(z) == 0:
                inDim = 1
            if np.max(z) == 0:
                inDim = 2
            else:
                inDim = 3
            return inDim

    @property
    def Ne(self) -> int:
        """number of elements"""
        return self.__connect.shape[0]

    @property
    def nodes(self) -> _types.IntArray:
        """nodes used by the element group. Node 'n' is on line 'n' in coordGlob"""
        return self.__nodes.copy()

    @property
    def elements(self) -> _types.IntArray:
        """elements"""
        return np.arange(self.__connect.shape[0], dtype=int)

    def _Set_partitionned_data(
        self,
        elementsGlob: _types.IntArray,
        nodes: _types.IntArray,
        rank: int = 0,
    ) -> None:
        """Sets the paritionned data used in mpi.

        Parameters
        ----------
        elementsGlob : _types.IntArray
            the positions of elements in the global mesh.
        nodes : _types.IntArray
            the (non-ghost) nodes.\n
        rank : int, optional
            mpi rank, by default 0

        Remark
        ------
        Ghost nodes will be computed using the given (non-ghost) nodes array.
        """

        # set elements glob
        assert elementsGlob.size == self.__connect.shape[0], "Must be a (Ne,) array."
        elementsGlob = np.asarray(elementsGlob, dtype=int)
        # get (non-ghost) nodes and ghost nodes
        nodes = np.asarray(nodes, dtype=int)
        ghostNodes = np.asarray(list(set(self.nodes) - set(nodes)), dtype=int)

        self.__partitionned_data = (rank, elementsGlob, nodes, ghostNodes)

    def _Get_partitionned_data(
        self,
    ) -> tuple[_types.IntArray, _types.IntArray, _types.IntArray]:
        """Returns the paritionned data used in mpi.\n
        (rank, elementsGlob, nodes, ghostNodes)"""
        return self.__partitionned_data

    @property
    def Nn(self) -> int:
        """number of nodes used by the element group"""
        return self.__nodes.size

    @property
    def coord(self) -> _types.FloatArray:
        """this matrix contains the element group coordinates (Nn, 3)"""
        coord: _types.FloatArray = self.coordGlob[self.__nodes]
        return coord

    @property
    def coordGlob(self) -> _types.FloatArray:
        """this matrix contains all the mesh coordinates (mesh.Nn, 3)"""
        return self.__coordGlob.copy()

    @coordGlob.setter
    def coordGlob(self, coord: _types.FloatArray) -> None:
        if coord.shape == self.__coordGlob.shape:
            self.__coordGlob = coord
            self._InitMatrix()

    @property
    def Nvertex(self) -> int:
        """number of vertex nodes per element"""
        return self.__Nvertex

    @property
    def Nedge(self) -> int:
        """number of edge nodes per element"""
        return self.__Nedge

    @property
    def Nface(self) -> int:
        """number of face nodes per element"""
        return self.__Nface

    @property
    def Nvolume(self) -> int:
        """number of volume nodes per element"""
        return self.__Nvolume

    @property
    def connect(self) -> _types.IntArray:
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

        if self.__connect_n_e is None:
            Ne = self.Ne
            nPe = self.nPe
            elems = self.elements

            lines = self.connect.ravel()

            Nn = (
                lines.max() + 1
            )  # Do not use either self.Nn or self.__coordGlob.shape[0].
            columns = np.repeat(elems, nPe)

            self.__connect_n_e = sparse.csr_matrix(
                (np.ones(nPe * Ne), (lines, columns)), shape=(Nn, Ne)
            )

        return self.__connect_n_e.copy()

    def Get_assembly_e(self, dof_n: int) -> _types.IntArray:
        """Get the assembly matrix for the specified dof_n (Ne, nPe*dof_n)

        Parameters
        ----------
        dof_n : int
            degree of freedom per node
        """

        return _GroupElem._Get_assembly_e(self.connect, dof_n)

    @staticmethod
    def _Get_assembly_e(connect: _types.IntArray, dof_n: int) -> _types.IntArray:
        """Get the assembly matrix for the specified dof_n (Ne, nPe*dof_n)

        Parameters
        ----------
        connect : _types.IntArray
            connectivity matrix (Ne, nPe)
        dof_n : int
            degree of freedom per node
        """

        assert connect.ndim, "connect must be an (Ne, nPe) array of int"

        Ne, nPe = connect.shape
        ndof = dof_n * nPe

        assembly = np.zeros((Ne, ndof), dtype=np.int64)

        for d in range(dof_n):
            columns = np.arange(d, ndof, dof_n)
            assembly[:, columns] = np.array(connect) * dof_n + d

        return assembly

    def Get_rows_e(self, dof_n: int) -> _types.IntArray:
        """Returns the row indices used to assemble local matrices into the global matrix."""
        assembly_e = self.Get_assembly_e(dof_n)
        nPe = self.nPe
        Ne = self.Ne
        rowsVector_e = np.repeat(assembly_e, nPe * dof_n).reshape((Ne, -1))
        return rowsVector_e

    def Get_columns_e(self, dof_n: int) -> _types.IntArray:
        """Returns the column indices used to assemble local matrices into the global matrix."""
        assembly_e = self.Get_assembly_e(dof_n)
        nPe = self.nPe
        Ne = self.Ne
        columnsVector_e = np.repeat(assembly_e, nPe * dof_n, axis=0).reshape((Ne, -1))
        return columnsVector_e

    def _Get_sysCoord_e(self, displacementMatrix: Optional[_types.AnyArray] = None):
        """Get the basis transformation matrix (Ne, 3, 3).\n
        [ix, jx, kx\n
        iy, jy, ky\n
        iz, jz, kz]\n

        This matrix can be used to project points with (x, y, z) coordinates into the element's (i, j, k) coordinate system.\n
        coordo_e * sysCoord_e -> coordinates in element's (i, j, k) coordinate system.
        """

        coordo = self.coordGlob
        connect = self.connect

        if displacementMatrix is not None:
            displacementMatrix = np.asarray(displacementMatrix, dtype=float)
            assert (
                displacementMatrix.shape == coordo.shape
            ), f"displacmentMatrix must be of size {coordo.shape}"
            coordo += displacementMatrix

        if self.dim in [0, 3]:
            sysCoord_e = np.eye(3)
            sysCoord_e = sysCoord_e[np.newaxis, :].repeat(self.Ne, axis=0)

        elif self.dim in [1, 2]:
            # 2D lines or elements

            if self.dim == 2:
                # Ensure outward-facing normals for all mesh faces.
                connect = connect[:, self.faces]

            points1 = coordo[connect[:, 0]]
            points2 = coordo[connect[:, 1]]

            i = Normalize(points2 - points1)

            if self.dim == 1:
                # Segments

                e1 = np.array([1, 0, 0])[np.newaxis, :].repeat(i.shape[0], axis=0)
                e2 = np.array([0, 1, 0])[np.newaxis, :].repeat(i.shape[0], axis=0)
                e3 = np.array([0, 0, 1])[np.newaxis, :].repeat(i.shape[0], axis=0)

                if self.inDim == 1:
                    j = e2
                    k = e3

                elif self.inDim == 2:
                    j = np.cross((0, 0, 1), i)
                    k = np.cross(i, j, axis=1)

                elif self.inDim == 3:
                    j = np.cross(i, e1, axis=1)

                    rep2 = np.where(np.linalg.norm(j, axis=1) < 1e-12)
                    rep1 = np.setdiff1d(range(i.shape[0]), rep2)

                    k1 = j.copy()
                    j1 = np.cross(k1, i, axis=1)

                    j2 = np.cross(e2, i, axis=1)
                    k2 = np.cross(i, j2, axis=1)

                    j = np.zeros_like(i)
                    j[rep1] = j1[rep1]
                    j[rep2] = j2[rep2]
                    j = Normalize(j)

                    k = np.zeros_like(i, dtype=float)
                    k[rep1] = k1[rep1]
                    k[rep2] = k2[rep2]
                    k = Normalize(k)

            else:
                if "TRI" in self.elemType:
                    points3 = coordo[connect[:, 2]]
                elif "QUAD" in self.elemType:
                    points3 = coordo[connect[:, 3]]
                else:
                    raise TypeError("unknown type")

                j = Normalize(points3 - points1)
                k = Normalize(np.cross(i, j, axis=1))
                j = Normalize(np.cross(k, i, axis=1))

            sysCoord_e = np.zeros((self.Ne, 3, 3), dtype=float)

            sysCoord_e[:, :, 0] = i
            sysCoord_e[:, :, 1] = j
            sysCoord_e[:, :, 2] = k

        return sysCoord_e

    def Get_normals_e_pg(
        self,
        matrixType: MatrixType,
        displacementMatrix: Optional[_types.FloatArray] = None,
    ) -> FeArray:
        """Returns the normals for each elements and gauss points (Ne, nPg, 3)."""

        dim = self.dim
        assert dim in [1, 2], "You can't compute normals for 0D or 3D elements."

        # get coords as a (Ne, nPe, 3) array

        coordGlob = self.coordGlob
        if displacementMatrix is not None:
            coordGlob += displacementMatrix

        coords_e = coordGlob[self.connect]

        # get the first derivatives of the shape functions as a (nPg, dim, nPe) array
        dN_pg = self.Get_dN_pg(matrixType)
        dNdr_pg = dN_pg[:, 0]

        dxdr_e_pg = np.einsum("pn,end->epd", dNdr_pg, coords_e, optimize="optimal")

        if dim == 1:
            normals_e_pg = np.cross((0, 0, 1), dxdr_e_pg)
        else:
            dNds_pg = dN_pg[:, 1]
            dxds_e_pg = np.einsum("pn,end->epd", dNds_pg, coords_e, optimize="optimal")
            normals_e_pg = np.cross(dxdr_e_pg, dxds_e_pg)

        normals_e_pg = np.einsum(
            "epi,ep->epi",
            normals_e_pg,
            1 / np.linalg.norm(normals_e_pg, axis=-1),
            optimize="optimal",
        )

        return FeArray.asfearray(normals_e_pg)

    def Integrate_e(
        self, func=lambda x, y, z: 1, matrixType=MatrixType.mass
    ) -> _types.FloatArray:
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
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.mass

        Returns
        -------
        _types.FloatArray
            integrated values on elements
        """

        wJ_e_pg = self.Get_weightedJacobian_e_pg(matrixType)
        coord_e_pg = self.Get_GaussCoordinates_e_pg(matrixType)
        eval_e_pg: FeArray.FeArrayALike = func(
            coord_e_pg[:, :, 0], coord_e_pg[:, :, 1], coord_e_pg[:, :, 2]
        )

        eval_e_pg = FeArray.asfearray(eval_e_pg)

        values_e = (wJ_e_pg * eval_e_pg).sum(1)

        return values_e

    @property
    def length_e(self) -> _types.FloatArray:
        """length covered by each element"""
        if self.dim != 1:
            return None  # type: ignore [return-value]
        length_e = self.Integrate_e(lambda x, y, z: 1, MatrixType.rigi)
        return length_e

    @property
    def length(self) -> float:
        """length covered by elements"""
        if self.dim != 1:
            return None  # type: ignore [return-value]
        return self.length_e.sum()

    @property
    def area_e(self) -> _types.FloatArray:
        """area covered by each element"""
        if self.dim != 2:
            return None  # type: ignore [return-value]
        area_e = self.Integrate_e(lambda x, y, z: 1, MatrixType.rigi)
        return area_e

    @property
    def area(self) -> float:
        """area covered by elements"""
        if self.dim != 2:
            return None  # type: ignore [return-value]
        return self.area_e.sum()

    @property
    def volume_e(self) -> _types.FloatArray:
        """volume covered by each element"""
        if self.dim != 3:
            return None  # type: ignore [return-value]
        volume_e = self.Integrate_e(lambda x, y, z: 1, MatrixType.rigi)
        return volume_e

    @property
    def volume(self) -> float:
        """volume covered by elements"""
        if self.dim != 3:
            return None  # type: ignore [return-value]
        return self.volume_e.sum()

    @property
    def center(self) -> _types.FloatArray:
        """center of mass / barycenter / inertia center"""

        matrixType = MatrixType.mass

        coordo_e_p = self.Get_GaussCoordinates_e_pg(matrixType)

        wJ_e_pg = self.Get_weightedJacobian_e_pg(matrixType)

        size = wJ_e_pg.sum(axis=(0, 1))

        center = (wJ_e_pg * coordo_e_p / size).sum(axis=(0, 1))

        return center

    @property
    @abstractmethod
    def origin(self) -> list[int]:
        """reference element origin coordinates"""
        return [0]

    @property
    @abstractmethod
    def triangles(self) -> list[int]:
        """list of index used to form the triangles of an element that will be used for the 2D trisurf function"""
        pass

    @property
    def segments(self) -> _types.IntArray:  # type: ignore [return]
        """array of indices used to construct segments (for display purposes)."""
        nPe = 2 + self.order - 1
        if self.__dim == 1:
            segments = np.zeros((1, nPe), dtype=int)
            segments[0, 0] = 0
            segments[0, -1] = 1
            if nPe > 2:
                vertices_on_seg = np.arange(
                    segments.max() + 1, segments.max() + 1 + self.order - 1
                )
                segments[0, 1 : nPe - 1] = vertices_on_seg

            return segments
        elif self.__dim == 2:
            segments = np.zeros((self.Nvertex, nPe), dtype=int)
            segments[:, 0] = np.arange(self.Nvertex)
            segments[:, -1] = np.append(np.arange(1, self.Nvertex, 1), 0)

            if nPe > 2:
                for i in range(self.Nvertex):
                    vertices_on_seg = np.arange(
                        segments.max() + 1, segments.max() + 1 + self.order - 1
                    )
                    segments[i, 1 : nPe - 1] = vertices_on_seg

            return segments
        elif self.__dim == 3:
            raise Exception("Needs to be defined for 3D element groups.")

    @property
    def edges(self) -> _types.IntArray:
        """array of indices used to form the element edges (for FEM purposes)."""
        segments = self.segments
        idx = np.arange(segments.shape[1])
        order = [0, -1] + idx[1:-1].tolist()
        edges = segments[:, order]
        return edges

    @property
    @abstractmethod
    def surfaces(self) -> _types.IntArray:
        """array of indices used to form the contour of the surfaces that make up the element (for display purposes).

        WARNING
        -------
        When adding new 3D elements, ensure that the resulting surface normals point inward the element.
        """
        pass

    @property
    @abstractmethod
    def faces(self) -> _types.IntArray:
        """array of indices used to form the element faces (for FEM purposes)."""
        pass

    @abstractmethod
    def Get_Local_Coords(self) -> _types.FloatArray:
        """Get local ξ, η, ζ coordinates as a (nPe, dim) numpy array"""
        pass

    # --------------------------------------------------------------------------------------------
    # Gauss
    # --------------------------------------------------------------------------------------------

    def Get_gauss(self, matrixType: MatrixType) -> Gauss:
        """Returns integration points according to the matrix type."""
        return Gauss(self.elemType, matrixType)

    def Get_weight_pg(self, matrixType: MatrixType) -> _types.FloatArray:
        """Returns integration point weights according to the matrix type."""
        return Gauss(self.elemType, matrixType).weights

    def Get_GaussCoordinates_e_pg(
        self, matrixType: MatrixType, elements=np.array([])
    ) -> FeArray.FeArrayALike:
        """Returns integration point coordinates for each element (Ne, nPg, 3) in the (x, y, z) coordinates."""

        N_pg = self.Get_N_pg(matrixType)

        # retrieve node coordinates
        coord = self.coordGlob

        # nodes coordinates for each element
        if elements.size == 0:
            coord_e = coord[self.__connect]
        else:
            coord_e = coord[self.__connect[elements]]

        # localize coordinates on Gauss points
        coordo_e_p = np.einsum("pin,end->epd", N_pg, coord_e, optimize="optimal")

        return FeArray.asfearray(coordo_e_p)

    # --------------------------------------------------------------------------------------------
    # Isoparametric elements
    # --------------------------------------------------------------------------------------------

    @cache_computed_values
    def Get_F_e_pg(self, matrixType: MatrixType) -> FeArray.FeArrayALike:
        """Returns the transposed Jacobian matrix.\n
        This matrix describes the transformation of the (ξ, η, ζ) axes from the reference element to the (x, y, z) coordinate system of the actual element.\n
        """
        if self.dim == 0:
            return None  # type: ignore [return-value]

        coordo_e = self.coordGlob[self.__connect]
        # Node coordinates in the (X, Y, Z) coordinate system of each element

        rebased_coord_e = coordo_e.copy()
        if self.dim != self.inDim:
            P_e = self._Get_sysCoord_e()  # transformation matrix for each element
            # matrix used to project element's points with (x, y, z) coordinates
            # into the (X, Y, Z) coordinate system.

            # check whether P_e is orthogonal
            isOrth_e = Trace(Transpose(P_e) @ P_e) == 3

            # (x, y, z) = (X, Y, Z) * P_e  <==>  aj = bi Pij
            rebased_coord_e[isOrth_e] = coordo_e[isOrth_e] @ P_e[isOrth_e]

            # (x, y, z) = (X, Y, Z) * P_e^(-T)  <==>  aj = bi inv(P)ji
            rebased_coord_e[~isOrth_e] = coordo_e[~isOrth_e] @ Transpose(
                Inv(P_e[~isOrth_e])
            )

        rebased_coord_e = rebased_coord_e[:, :, : self.dim]
        # (Ne, nPe, dim)

        dN_pg = FeArray.asfearray(self.Get_dN_pg(matrixType)[np.newaxis])
        rebased_coord_e = FeArray.asfearray(rebased_coord_e[:, np.newaxis])

        F_e_pg = dN_pg @ rebased_coord_e

        return F_e_pg

    @cache_computed_values
    def Get_jacobian_e_pg(
        self, matrixType: MatrixType, absoluteValues=True
    ) -> FeArray.FeArrayALike:
        """Returns the jacobians.\n
        variation in size (length, area or volume) between the reference element and the actual element
        """
        if self.dim == 0:
            return None  # type: ignore [return-value]

        F_e_pg = self.Get_F_e_pg(matrixType)

        jacobian_e_pg = FeArray.asfearray(Det(F_e_pg))

        if absoluteValues:
            jacobian_e_pg = np.abs(jacobian_e_pg)

        return jacobian_e_pg

    def Get_weightedJacobian_e_pg(self, matrixType: MatrixType) -> FeArray.FeArrayALike:
        """Returns the jacobian_e_pg * weight_pg."""
        if self.dim == 0:
            return None  # type: ignore [return-value]

        jacobian_e_pg = self.Get_jacobian_e_pg(matrixType)
        weight_pg = self.Get_weight_pg(matrixType)

        wJ_e_pg = np.asarray(jacobian_e_pg) * weight_pg

        return FeArray.asfearray(wJ_e_pg)

    @cache_computed_values
    def Get_invF_e_pg(self, matrixType: MatrixType) -> FeArray.FeArrayALike:
        """Returns the inverse of the transposed Jacobian matrix.\n
        Used to obtain the derivative of the dN_e_pg shape functions in the actual element
        dN_e_pg = invF_e_pg • dN_pg
        """
        if self.dim == 0:
            return None  # type: ignore [return-value]

        F_e_pg = self.Get_F_e_pg(matrixType)

        invF_e_pg = FeArray.asfearray(Inv(F_e_pg))

        return invF_e_pg

    # --------------------------------------------------------------------------------------------
    # Finite element shape functions
    # --------------------------------------------------------------------------------------------

    @staticmethod
    def _Eval_Functions(
        functions: _types.FloatArray, gaussPoints: _types.FloatArray
    ) -> _types.FloatArray:
        """Evaluates functions at coordinates.\n
        Use this function to evaluate shape functions.

        Parameters
        ----------
        functions : _types.FloatArray
            functions to evaluate, (nPe, dim)
        gaussPoints : _types.FloatArray
            gauss coordinates where functions will be evaluated (nPg, dim).\n
            Be careful dim must be consistent with function arguments

        Returns
        -------
        _types.FloatArray
            Evaluated functions (nPg, dim, nPe)
        """

        nPg = gaussPoints.shape[0]
        nPe = functions.shape[0]
        nF = functions.shape[1]

        evalFunctions = np.zeros((nPg, nF, nPe))

        # for each points
        for p in range(nPg):
            # for each functions
            for n, function_nPe in enumerate(functions):
                # for each dimension
                for f in range(nF):
                    # appy the function
                    evalFunctions[p, f, n] = function_nPe[f](*gaussPoints[p])  # type: ignore [index]
                    # * means take all the coordinates

        return evalFunctions

    def _Init_Functions(self, order: int) -> _types.FloatArray:
        """Initializes functions to be evaluated at Gauss points."""
        if self.dim == 1 and self.order < order:
            functions = np.array([lambda x: 0] * self.nPe)
        elif self.dim == 2 and self.order < order:
            functions = np.array([lambda xi, η: 0, lambda xi, η: 0] * self.nPe)
        elif self.dim == 3 and self.order < order:
            functions = np.array(
                [lambda x, y, z: 0, lambda x, y, z: 0, lambda x, y, z: 0] * self.nPe
            )
        else:
            raise TypeError("unknwown dim")
        functions = np.reshape(functions, (self.nPe, -1))
        return functions

    # N

    @abstractmethod
    def _N(self) -> _types.FloatArray:
        """Shape functions in (ξ, η, ζ) coordinates.\n
        N1 \n
        ⋮  \n
        Nn \n
        (nPe, 1)
        """
        pass

    def Get_N_pg(self, matrixType: MatrixType) -> _types.FloatArray:
        """Evaluates shape functions in (ξ, η, ζ) coordinates.\n
        [N1, . . . , Nn]\n
        (nPg, 1, nPe)
        """
        if self.dim == 0:
            return None  # type: ignore [return-value]

        N = self._N()
        gauss = self.Get_gauss(matrixType)
        N_pg = _GroupElem._Eval_Functions(N, gauss.coord)

        return N_pg

    def Get_N_pg_rep(self, matrixType: MatrixType, repeat=1) -> _types.FloatArray:
        """Repeats shape functions in the (ξ, η, ζ) coordinates.

        Parameters
        ----------
        matrixType : MatrixType
            matrix type
        repeat : int, optional
            number of repetitions, by default 1

        Returns:
        -------
        • Vector shape functions (nPg, rep=2, rep=2*dim)\n
            [Ni 0 . . . Nn 0 \n
            0 Ni . . . 0 Nn]

        • Scalar shape functions (nPg, rep=1, nPe)\n
            [Ni . . . Nn]
        """
        if self.dim == 0:
            return None  # type: ignore [return-value]

        assert isinstance(repeat, int)
        assert repeat >= 1

        N_pg = self.Get_N_pg(matrixType)

        if not isinstance(N_pg, np.ndarray):
            return None  # type: ignore [return-value]

        if repeat <= 1:
            return N_pg
        else:
            size = N_pg.shape[2] * repeat
            N_vect_pg = np.zeros((N_pg.shape[0], repeat, size))

            for r in range(repeat):
                N_vect_pg[:, r, np.arange(r, size, repeat)] = N_pg[:, 0, :]

            return N_vect_pg

    # dN

    @abstractmethod
    def _dN(self) -> _types.FloatArray:
        """Shape functions first derivatives in the (ξ, η, ζ) coordinates.\n
        Ni,ξ  Ni,η  Ni,ζ \n
        \t \t \t \t \t ⋮ \n
        Nn,ξ  Nn,η  Nn,ζ \n
        (nPe, dim)
        """
        return self._Init_Functions(1)

    def Get_dN_pg(self, matrixType: MatrixType) -> _types.FloatArray:
        """Evaluates shape functions first derivatives in the (ξ, η, ζ) coordinates.\n
        Ni,ξ . . . Nn,ξ\n
        Ni,η . . . Nn,η\n
        Ni,ζ . . . Nn,ζ\n
        (nPg, dim, nPe)
        """
        if self.dim == 0:
            return None  # type: ignore [return-value]

        dN = self._dN()

        gauss = self.Get_gauss(matrixType)
        dN_pg = _GroupElem._Eval_Functions(dN, gauss.coord)

        return dN_pg

    @cache_computed_values
    def Get_dN_e_pg(self, matrixType: MatrixType) -> FeArray.FeArrayALike:
        """Evaluates the first-order derivatives of shape functions in (x, y, z) coordinates.\n
        [Ni,x . . . Nn,x\n
        Ni,y . . . Nn,y\n
        Ni,z . . . Nn,z]\n
        (Ne, nPg, dim, nPe)\n
        """
        assert matrixType in MatrixType.Get_types()

        invF_e_pg = self.Get_invF_e_pg(matrixType)

        dN_pg = FeArray.asfearray(self.Get_dN_pg(matrixType)[np.newaxis])

        # Derivation of shape functions in the (x, y, z) coordinates
        dN_e_pg = invF_e_pg @ dN_pg

        return dN_e_pg

    # ddN

    @abstractmethod
    def _ddN(self) -> _types.FloatArray:
        """Shape functions second derivatives in the (ξ, η, ζ) coordinates.\n
        Ni,ξ2  Ni,η2  Ni,ζ2 \n
        \t \t \t \t \t ⋮ \n
        Nn,ξ2  Nn,η2  Nn,ζ2 \n
        (nPe, dim)
        """
        return self._Init_Functions(2)

    @cache_computed_values
    def Get_ddN_e_pg(self, matrixType: MatrixType) -> FeArray.FeArrayALike:
        """Evaluates the second-order derivatives of shape functions in (x, y, z) coordinates.\n
        [Ni,x2 . . . Nn,x2\n
        Ni,y2 . . . Nn,y2\n
        Ni,z2 . . . Nn,z2]\n
        (Ne, nPg, dim, nPe)\n
        """
        assert matrixType in MatrixType.Get_types()

        invF_e_pg = self.Get_invF_e_pg(matrixType)

        ddN_pg = FeArray.asfearray(self.Get_ddN_pg(matrixType)[np.newaxis])

        ddN_e_pg = invF_e_pg @ invF_e_pg @ ddN_pg

        return ddN_e_pg

    def Get_ddN_pg(self, matrixType: MatrixType) -> _types.FloatArray:
        """Evaluates shape functions second derivatives in the (ξ, η, ζ) coordinates.\n
        [Ni,ξ2 . . . Nn,ξ2\n
        Ni,η2 . . . Nn,η2\n
        Ni,ζ2 . . . Nn,ζ2]\n
        (nPg, dim, nPe)
        """
        if self.dim == 0:
            return None  # type: ignore [return-value]

        ddN = self._ddN()

        gauss = self.Get_gauss(matrixType)
        ddN_pg = _GroupElem._Eval_Functions(ddN, gauss.coord)

        return ddN_pg

    # dddN

    @abstractmethod
    def _dddN(self) -> _types.FloatArray:
        """Shape functions third derivatives in the (ξ, η, ζ) coordinates.\n
        Ni,ξ3  Ni,η3  Ni,ζ3 \n
        \t \t \t \t \t ⋮ \n
        Nn,ξ3  Nn,η3  Nn,ζ3 \n
        (nPe, dim)
        """
        return self._Init_Functions(3)

    def Get_dddN_pg(self, matrixType: MatrixType) -> _types.FloatArray:
        """Evaluates shape functions third derivatives in the (ξ, η, ζ) coordinates.\n
        [Ni,ξ3 . . . Nn,ξ3\n
        Ni,η3 . . . Nn,η3\n
        Ni,ζ3 . . . Nn,ζ3]\n
        (nPg, dim, nPe)
        """
        if self.elemType == 0:
            return None  # type: ignore [return-value]

        dddN = self._dddN()

        gauss = self.Get_gauss(matrixType)
        dddN_pg = _GroupElem._Eval_Functions(dddN, gauss.coord)

        return dddN_pg

    # ddddN

    @abstractmethod
    def _ddddN(self) -> _types.FloatArray:
        """Shape functions fourth derivatives in the (ξ, η, ζ) coordinates.\n
        Ni,ξ4  Ni,η4  Ni,ζ4 \n
        \t \t \t \t \t ⋮ \n
        Nn,ξ4  Nn,η4  Nn,ζ4 \n
        (nPe, dim)
        """
        return self._Init_Functions(4)

    def Get_ddddN_pg(self, matrixType: MatrixType) -> _types.FloatArray:
        """Evaluates shape functions fourth derivatives in the (ξ, η, ζ) coordinates.\n
        [Ni,ξ4 . . . Nn,ξ4\n
        Ni,η4 . . . Nn,η4\n
        Ni,ζ4 . . . Nn,ζ4]\n
        (pg, dim, nPe)
        """
        if self.elemType == 0:
            return None  # type: ignore [return-value]

        ddddN = self._ddddN()

        gauss = self.Get_gauss(matrixType)
        ddddN_pg = _GroupElem._Eval_Functions(ddddN, gauss.coord)

        return ddddN_pg

    # Beams shapes functions
    # Use hermitian shape functions

    # N

    def _EulerBernoulli_N(self) -> _types.FloatArray:
        """Euler-Bernoulli beam shape functions in the (ξ, η, ζ) coordinates.\n
        [phi_i psi_i . . . phi_n psi_n]\n
        (nPe*2, 1)
        """
        return None  # type: ignore [return-value]

    def Get_EulerBernoulli_N_pg(self) -> _types.FloatArray:
        """Evaluates Euler-Bernoulli beam shape functions in the (ξ, η, ζ) coordinates.\n
        [phi_i psi_i . . . phi_n psi_n]\n
        (nPg, nPe*2)
        """
        if self.dim != 1:
            return None  # type: ignore [return-value]

        matrixType = MatrixType.beam

        N = self._EulerBernoulli_N()

        gauss = self.Get_gauss(matrixType)
        N_pg = _GroupElem._Eval_Functions(N, gauss.coord)

        return N_pg

    def Get_EulerBernoulli_N_e_pg(self) -> FeArray.FeArrayALike:  # type: ignore
        """Evaluates Euler-Bernoulli beam shape functions in (x, y, z) coordinates.\n
        [phi_i psi_i . . . phi_n psi_n]\n
        (Ne, nPg, 1, nPe*2)
        """
        if self.dim != 1:
            return None  # type: ignore [return-value]

        invF_e_pg = self.Get_invF_e_pg(MatrixType.beam)[:, :, 0, 0]
        N_pg = FeArray.asfearray(self.Get_EulerBernoulli_N_pg()[np.newaxis])
        nPe = self.nPe

        N_e_pg = invF_e_pg * N_pg

        # multiply by the beam length on psi_i,xx functions
        l_e = self.length_e
        columns = np.arange(1, nPe * 2, 2)
        for column in columns:
            N_e_pg[:, :, 0, column] = np.einsum(
                "ep,e->ep", N_e_pg[:, :, 0, column], l_e, optimize="optimal"
            )

        return N_e_pg

    # dN

    def _EulerBernoulli_dN(self) -> _types.FloatArray:
        """Euler-Bernoulli beam shape functions first derivatives in the (ξ, η, ζ) coordinates.\n
        [phi_i,ξ psi_i,ξ . . . phi_n,ξ psi_n,ξ]\n
        (nPe*2, 1)
        """
        return None  # type: ignore [return-value]

    def Get_EulerBernoulli_dN_pg(self) -> _types.FloatArray:
        """Evaluates Euler-Bernoulli beam shape functions first derivatives in the (ξ, η, ζ) coordinates.\n
        [phi_i,ξ psi_i,ξ . . . phi_n,ξ psi_n,ξ]\n
        (nPg, nPe*2)
        """
        if self.dim != 1:
            return None  # type: ignore [return-value]

        matrixType = MatrixType.beam

        dN = self._EulerBernoulli_dN()

        gauss = self.Get_gauss(matrixType)
        dN_pg = _GroupElem._Eval_Functions(dN, gauss.coord)

        return dN_pg

    def Get_EulerBernoulli_dN_e_pg(self) -> FeArray.FeArrayALike:
        """Evaluates the first-order derivatives of Euler-Bernoulli beam shape functions in (x, y, z) coordinates.\n
        [phi_i,x psi_i,x . . . phi_n,x psi_n,x]\n
        (Ne, nPg, 1, nPe*2)
        """
        if self.dim != 1:
            return None  # type: ignore [return-value]

        invF_e_pg = self.Get_invF_e_pg(MatrixType.beam)[:, :, 0, 0]
        dN_pg = FeArray.asfearray(self.Get_EulerBernoulli_dN_pg()[np.newaxis])

        dN_e_pg = invF_e_pg * dN_pg

        # multiply by the beam length on psi_i,xx functions
        l_e = self.length_e
        nPe = self.nPe
        columns = np.arange(1, nPe * 2, 2)
        for column in columns:
            dN_e_pg[:, :, 0, column] = np.einsum(
                "ep,e->ep", dN_e_pg[:, :, 0, column], l_e, optimize="optimal"
            )

        return dN_e_pg

    # ddN

    def _EulerBernoulli_ddN(self) -> _types.FloatArray:
        """Euler-Bernoulli beam shape functions second derivatives in the (ξ, η, ζ) coordinates.\n
        [phi_i,ξ psi_i,ξ . . . phi_n,ξ psi_n,ξ]\n
        (nPe*2, 2)
        """
        return None  # type: ignore [return-value]

    def Get_EulerBernoulli_ddN_pg(self) -> _types.FloatArray:
        """Evaluates Euler-Bernoulli beam shape functions second derivatives in the (ξ, η, ζ) coordinates.\n
        [phi_i,ξ psi_i,ξ . . . phi_n,ξ x psi_n,ξ]\n
        (nPg, nPe*2)
        """
        if self.dim != 1:
            return None  # type: ignore [return-value]

        matrixType = MatrixType.beam

        ddN = self._EulerBernoulli_ddN()

        gauss = self.Get_gauss(matrixType)
        ddN_pg = _GroupElem._Eval_Functions(ddN, gauss.coord)

        return ddN_pg

    def Get_EulerBernoulli_ddN_e_pg(self) -> FeArray.FeArrayALike:
        """Evaluates the second-order derivatives of Euler-Bernoulli beam shape functions in (x, y, z) coordinates.\n
        [phi_i,xx psi_i,xx . . . phi_n,xx psi_n,xx]\n
        (Ne, nPg, 1, nPe*2)
        """
        if self.dim != 1:
            return None  # type: ignore [return-value]

        invF_e_pg = self.Get_invF_e_pg(MatrixType.beam)[:, :, 0, 0]
        ddN_pg = FeArray.asfearray(self.Get_EulerBernoulli_ddN_pg()[np.newaxis])
        nPe = self.nPe

        ddN_e_pg = invF_e_pg * invF_e_pg * ddN_pg

        # multiply by the beam length on psi_i,xx functions
        l_e = self.length_e
        columns = np.arange(1, nPe * 2, 2)
        for column in columns:
            ddN_e_pg[:, :, 0, column] = np.einsum(
                "ep,e->ep", ddN_e_pg[:, :, 0, column], l_e, optimize="optimal"
            )

        return ddN_e_pg

    # --------------------------------------------------------------------------------------------
    # Finite element matrices
    # --------------------------------------------------------------------------------------------

    # Linear elastic problem

    @cache_computed_values
    def Get_B_e_pg(self, matrixType: MatrixType) -> FeArray.FeArrayALike:
        """Get the matrix used to calculate deformations from displacements.\n

        WARNING
        -------
        Use Kelvin Mandel Notation\n
        [N1,x 0 . . . Nn,x 0\n
        0 N1,y . . . 0 Nn,y\n
        N1,y N1,x . . . N3,y N3,x]\n
        (Ne, nPg, (3 or 6), nPe*dim)
        """
        assert matrixType in MatrixType.Get_types()

        dN_e_pg = self.Get_dN_e_pg(matrixType)

        Ne = self.Ne
        nPg = self.Get_gauss(matrixType).nPg
        nPe = self.nPe
        dim = self.dim

        cM = 1 / np.sqrt(2)

        columnsX = np.arange(0, nPe * dim, dim)
        columnsY = np.arange(1, nPe * dim, dim)
        columnsZ = np.arange(2, nPe * dim, dim)

        if self.dim == 2:
            B_e_pg = np.zeros((Ne, nPg, 3, nPe * dim))

            dNdx = dN_e_pg[:, :, 0]
            dNdy = dN_e_pg[:, :, 1]

            B_e_pg[:, :, 0, columnsX] = dNdx
            B_e_pg[:, :, 1, columnsY] = dNdy
            B_e_pg[:, :, 2, columnsX] = dNdy * cM
            B_e_pg[:, :, 2, columnsY] = dNdx * cM
        else:
            B_e_pg = np.zeros((Ne, nPg, 6, nPe * dim))

            dNdx = dN_e_pg[:, :, 0]
            dNdy = dN_e_pg[:, :, 1]
            dNdz = dN_e_pg[:, :, 2]

            B_e_pg[:, :, 0, columnsX] = dNdx
            B_e_pg[:, :, 1, columnsY] = dNdy
            B_e_pg[:, :, 2, columnsZ] = dNdz
            B_e_pg[:, :, 3, columnsY] = dNdz * cM
            B_e_pg[:, :, 3, columnsZ] = dNdy * cM
            B_e_pg[:, :, 4, columnsX] = dNdz * cM
            B_e_pg[:, :, 4, columnsZ] = dNdx * cM
            B_e_pg[:, :, 5, columnsX] = dNdy * cM
            B_e_pg[:, :, 5, columnsY] = dNdx * cM

        return FeArray.asfearray(B_e_pg)

    @cache_computed_values
    def Get_leftDispPart(self, matrixType: MatrixType) -> FeArray.FeArrayALike:
        """Get the left side of local displacement matrices.\n
        Ku_e = jacobian_e_pg * weight_pg * B_e_pg' @ c_e_pg @ B_e_pg\n

        Returns (epij) -> jacobian_e_pg * weight_pg * B_e_pg'
        """

        assert matrixType in MatrixType.Get_types()

        wJ_e_pg = self.Get_weightedJacobian_e_pg(matrixType)
        B_e_pg = self.Get_B_e_pg(matrixType)

        leftDispPart = wJ_e_pg * B_e_pg.T

        return leftDispPart

    # Euler Bernoulli problem

    def Get_EulerBernoulli_N_e_pg_for_beam(
        self, beamStructure: "BeamStructure"
    ) -> FeArray.FeArrayALike:
        """Euler-Bernoulli beam shape functions."""

        # Example in matlab :
        # https://github.com/fpled/FEMObject/blob/master/BASIC/MODEL/ELEMENTS/%40BEAM/calc_N.m

        matrixType = MatrixType.beam

        # get the beam model
        dim = beamStructure.dim
        dof_n = beamStructure.dof_n

        # Data
        jacobian_e_pg = self.Get_jacobian_e_pg(matrixType)
        nPe = self.nPe
        Ne = jacobian_e_pg.shape[0]
        nPg = jacobian_e_pg.shape[1]

        # get matrices to work with
        N_pg = self.Get_N_pg(matrixType)
        N_e_pg = self.Get_EulerBernoulli_N_e_pg()
        dN_e_pg = self.Get_EulerBernoulli_dN_e_pg()

        if dim == 1:
            # u = [u1, . . . , un]

            # N = [N_i, . . . , N_n]

            idx_ux = np.arange(dof_n * nPe)

            N_e_pg = np.zeros((Ne, nPg, 1, dof_n * nPe))
            N_e_pg[:, :, 0, idx_ux] = N_pg[:, :, 0]

        elif dim == 2:
            # u = [u1, v1, rz1, . . . , un, vn, rzn]

            # N = [N_i, 0, 0, ... , N_n, 0, 0,]
            #     [0, Phi_i, Psi_i, ... , 0, Phi_i, Psi_i]
            #     [0, dPhi_i, dPsi_i, ... , 0, dPhi_i, dPsi_i]

            idx = np.arange(dof_n * nPe, dtype=int).reshape(nPe, -1)

            idx_ux = idx[:, 0]  # [0,3] (SEG2) [0,3,6] (SEG3)
            idx_uy = np.reshape(idx[:, 1:], -1)  # [1,2,4,5] (SEG2) [1,2,4,5,7,8] (SEG3)

            N_e_pg = np.zeros((Ne, nPg, 3, dof_n * nPe))

            N_e_pg[:, :, 0, idx_ux] = N_pg[:, :, 0]  # traction / compression to get u
            N_e_pg[:, :, 1, idx_uy] = N_e_pg[:, :, 0]  # flexion z to get v
            N_e_pg[:, :, 2, idx_uy] = dN_e_pg[:, :, 0]  # flexion z to get rz

        elif dim == 3:
            # u = [u1, v1, w1, rx1, ry1, rz1, . . . , un, vn, wn, rxn, ryn, rzn]

            # N = [N_i, 0, 0, 0, 0, 0, ... , N_n, 0, 0, 0, 0, 0]
            #     [0, Phi_i, 0, 0, 0, Psi_i, ... , 0, Phi_n, 0, 0, 0, Psi_n]
            #     [0, 0, dPhi_i, 0, -dPsi_i, 0, ... , 0, 0, dPhi_n, 0, -dPsi_n, 0]
            #     [0, 0, 0, N_i, 0, 0, ... , 0, 0, 0, N_n, 0, 0]
            #     [0, 0, -dPhi_i, 0, dPsi_i, 0, ... , 0, 0, -dPhi_n, 0, dPsi_n, 0]
            #     [0, dPhi_i, 0, 0, 0, dPsi_i, ... , 0, dPhi_i, 0, 0, 0, dPsi_n]

            idx = np.arange(dof_n * nPe, dtype=int).reshape(nPe, -1)
            idx_ux = idx[:, 0]  # [0,6] (SEG2) [0,6,12] (SEG3)
            idx_uy = np.reshape(
                idx[:, [1, 5]], -1
            )  # [1,5,7,11] (SEG2) [1,5,7,11,13,17] (SEG3)
            idx_uz = np.reshape(
                idx[:, [2, 4]], -1
            )  # [2,4,8,10] (SEG2) [2,4,8,10,14,16] (SEG3)
            idx_rx = idx[:, 3]  # [3,9] (SEG2) [3,9,15] (SEG3)
            idPsi = np.arange(1, nPe * 2, 2)  # [1,3] (SEG2) [1,3,5] (SEG3)

            Nvz_e_pg = N_e_pg.copy()
            Nvz_e_pg[:, :, 0, idPsi] *= -1

            dNvz_e_pg = dN_e_pg.copy()
            dNvz_e_pg[:, :, 0, idPsi] *= -1

            N_e_pg = np.zeros((Ne, nPg, 6, dof_n * nPe))

            N_e_pg[:, :, 0, idx_ux] = N_pg[:, :, 0]
            N_e_pg[:, :, 1, idx_uy] = N_e_pg[:, :, 0]
            N_e_pg[:, :, 2, idx_uz] = Nvz_e_pg[:, :, 0]
            N_e_pg[:, :, 3, idx_rx] = N_pg[:, :, 0]
            N_e_pg[:, :, 4, idx_uz] = -dNvz_e_pg[:, :, 0]  # ry = -uz'
            N_e_pg[:, :, 5, idx_uy] = dN_e_pg[:, :, 0]  # rz = uy'

        N_e_pg = FeArray.asfearray(N_e_pg)

        if dim > 1:
            # Construct the matrix used to change the matrix coordinates
            P = np.zeros((self.Ne, 3, 3), dtype=float)
            for beam in beamStructure.beams:
                elems = self.Get_Elements_Tag(beam.name)
                P[elems] = beam._Calc_P()

            Pglob_e_pg = FeArray.zeros(Ne, 1, dof_n * nPe, dof_n * nPe)
            N = P.shape[1]
            lines = np.repeat(range(N), N)
            columns = np.array(list(range(N)) * N)
            for n in range(dof_n * nPe // 3):
                # apply P on the diagonal
                Pglob_e_pg[:, lines + n * N, columns + n * N] = P[:, lines, columns]

            N_e_pg = N_e_pg @ Pglob_e_pg

        return N_e_pg

    def Get_EulerBernoulli_B_e_pg(
        self, beamStructure: "BeamStructure"
    ) -> FeArray.FeArrayALike:  # type: ignore
        """Get Euler-Bernoulli beam shape functions derivatives"""

        # Example in matlab :
        # https://github.com/fpled/FEMObject/blob/master/BASIC/MODEL/ELEMENTS/%40BEAM/calc_B.m

        matrixType = MatrixType.beam

        # Recovering the beam model
        dim = beamStructure.dim
        dof_n = beamStructure.dof_n

        # Data
        jacobian_e_pg = self.Get_jacobian_e_pg(matrixType)
        nPe = self.nPe
        Ne = jacobian_e_pg.shape[0]
        nPg = jacobian_e_pg.shape[1]

        # Recover matrices to work with
        dN_e_pg = self.Get_dN_e_pg(matrixType)
        ddNv_e_pg = self.Get_EulerBernoulli_ddN_e_pg()

        if dim == 1:
            # u = [u1, . . . , un]

            # B = [dN_i, . . . , dN_n]

            idx_ux = np.arange(dof_n * nPe)

            B_e_pg = np.zeros((Ne, nPg, 1, dof_n * nPe), dtype=float)
            B_e_pg[:, :, 0, idx_ux] = dN_e_pg[:, :, 0]

        elif dim == 2:
            # u = [u1, v1, rz1, . . . , un, vn, rzn]

            # B = [dN_i, 0, 0, ... , dN_n, 0, 0,]
            #     [0, ddPhi_i, ddPsi_i, ... , 0, ddPhi_i, ddPsi_i]

            idx = np.arange(dof_n * nPe, dtype=int).reshape(nPe, -1)

            idx_ux = idx[:, 0]  # [0,3] (SEG2) [0,3,6] (SEG3)
            idx_uy = np.reshape(idx[:, 1:], -1)  # [1,2,4,5] (SEG2) [1,2,4,5,7,8] (SEG3)

            B_e_pg = np.zeros((Ne, nPg, 2, dof_n * nPe), dtype=float)

            B_e_pg[:, :, 0, idx_ux] = dN_e_pg[:, :, 0]  # traction / compression
            B_e_pg[:, :, 1, idx_uy] = ddNv_e_pg[:, :, 0]  # flexion along z

        elif dim == 3:
            # u = [u1, v1, w1, rx1, ry1, rz1, . . . , un, vn, wn, rxn, ryn, rzn]

            # B = [dN_i, 0, 0, 0, 0, 0, ... , dN_n, 0, 0, 0, 0, 0]
            #     [0, 0, 0, dN_i, 0, 0, ... , 0, 0, 0, dN_n, 0, 0]
            #     [0, 0, ddPhi_i, 0, -ddPsi_i, 0, ... , 0, 0, ddPhi_n, 0, -ddPsi_n, 0]
            #     [0, ddPhi_i, 0, 0, 0, ddPsi_i, ... , 0, ddPhi_i, 0, 0, 0, ddPsi_n]

            idx = np.arange(dof_n * nPe).reshape(nPe, -1)
            idx_ux = idx[:, 0]  # [0,6] (SEG2) [0,6,12] (SEG3)
            idx_uy = np.reshape(
                idx[:, [1, 5]], -1
            )  # [1,5,7,11] (SEG2) [1,5,7,11,13,17] (SEG3)
            idx_uz = np.reshape(
                idx[:, [2, 4]], -1
            )  # [2,4,8,10] (SEG2) [2,4,8,10,14,16] (SEG3)
            idx_rx = idx[:, 3]  # [3,9] (SEG2) [3,9,15] (SEG3)

            idPsi = np.arange(1, nPe * 2, 2)  # [1,3] (SEG2) [1,3,5] (SEG3)
            ddNvz_e_pg = ddNv_e_pg.copy()
            ddNvz_e_pg[:, :, 0, idPsi] *= -1  # RY = -UZ'

            B_e_pg = np.zeros((Ne, nPg, 4, dof_n * nPe), dtype=float)

            B_e_pg[:, :, 0, idx_ux] = dN_e_pg[:, :, 0]  # traction / compression
            B_e_pg[:, :, 1, idx_rx] = dN_e_pg[:, :, 0]  # torsion
            B_e_pg[:, :, 2, idx_uz] = ddNvz_e_pg[:, :, 0]  # flexion along y
            B_e_pg[:, :, 3, idx_uy] = ddNv_e_pg[:, :, 0]  # flexion along z
        else:
            raise TypeError("dim error")

        B_e_pg = FeArray.asfearray(B_e_pg)

        if dim > 1:
            # Construct the matrix used to change the matrix coordinates
            P = np.zeros((self.Ne, 3, 3))
            for beam in beamStructure.beams:
                elems = self.Get_Elements_Tag(beam.name)
                P[elems] = beam._Calc_P()

            Pglob_e = FeArray.zeros(Ne, 1, dof_n * nPe, dof_n * nPe)
            N = P.shape[-1]
            lines = np.repeat(range(N), N)
            columns = np.array(list(range(N)) * N)
            for n in range(dof_n * nPe // 3):
                # apply P on the diagonal
                Pglob_e[:, 0, lines + n * N, columns + n * N] = P[:, lines, columns]

            B_e_pg = B_e_pg @ Pglob_e

        return B_e_pg

    # reaction diffusion problem

    @cache_computed_values
    def Get_ReactionPart_e_pg(self, matrixType: MatrixType) -> FeArray.FeArrayALike:
        """Get the part that builds the reaction term (scalar).\n
        ReactionPart_e_pg = r_e_pg * jacobian_e_pg * weight_pg * N_pg' @ N_pg\n

        Returns (epij) -> jacobian_e_pg * weight_pg * N_pg' @ N_pg
        """

        assert matrixType in MatrixType.Get_types()

        weightedJacobian = self.Get_weightedJacobian_e_pg(matrixType)
        N_pg = FeArray.asfearray(self.Get_N_pg_rep(matrixType, 1)[np.newaxis])

        ReactionPart_e_pg = weightedJacobian * N_pg.T @ N_pg

        return ReactionPart_e_pg

    @cache_computed_values
    def Get_DiffusePart_e_pg(self, matrixType: MatrixType) -> FeArray.FeArrayALike:
        """Get the part that builds the diffusion term (scalar).\n
        DiffusePart_e_pg = k_e_pg * jacobian_e_pg * weight_pg * dN_e_pg' @ A @ dN_e_pg\n

        Returns (epij) -> jacobian_e_pg * weight_pg * dN_e_pg'
        """

        assert matrixType in MatrixType.Get_types()

        wJ_e_pg = self.Get_weightedJacobian_e_pg(matrixType)
        dN_e_pg = self.Get_dN_e_pg(matrixType)

        DiffusePart_e_pg = wJ_e_pg * dN_e_pg.T

        return DiffusePart_e_pg

    @cache_computed_values
    def Get_SourcePart_e_pg(self, matrixType: MatrixType) -> FeArray.FeArrayALike:
        """Get the part that builds the source term (scalar).\n
        SourcePart_e_pg = f_e_pg * jacobian_e_pg * weight_pg * N_pg'\n

        Returns (epij) -> jacobian_e_pg * weight_pg * N_pg'
        """

        assert matrixType in MatrixType.Get_types()

        wJ_e_pg = self.Get_weightedJacobian_e_pg(matrixType)
        N_pg = FeArray.asfearray(self.Get_N_pg_rep(matrixType, 1)[np.newaxis])

        SourcePart_e_pg = wJ_e_pg * N_pg.T

        return SourcePart_e_pg

    def Get_Gradient_e_pg(
        self, u: _types.FloatArray, matrixType=MatrixType.rigi
    ) -> FeArray.FeArrayALike:
        """Returns the gradient of the discretized displacement field u as a matrix

        Parameters
        ----------
        u : _types.FloatArray
            discretized displacement field [ux1, uy1, uz1, . . ., uxN, uyN, uzN] of size Nn * dim
        matrixType : MatrixType, optional
            matrix type, by default MatrixType.rigi

        Returns
        -------
        FeArray
            grad(u) of shape (Ne, nPg, 3, 3)

        dim = 1
        -------

        dxux 0 0\n
        0 0 0\n
        0 0 0

        dim = 2
        -------

        dxux dyux 0\n
        dxuy dyuy 0\n
        0 0 0

        dim = 3
        -------

        dxux dyux dzux\n
        dxuy dyuy dzuy\n
        dxuz dyuz dzuz
        """

        Nn = self.coordGlob.shape[0]
        assert isinstance(u, np.ndarray) and u.size % Nn == 0
        dof_n = u.size // Nn
        assert dof_n in [1, 2, 3]

        # properties
        Ne = self.Ne
        nPe = self.nPe
        dim = self.dim

        dN_e_pg = self.Get_dN_e_pg(matrixType)
        nPg = dN_e_pg.shape[1]
        # Shape functions (Ne, nPg, nPe)
        dxN_e_pg = dN_e_pg[:, :, 0, :]
        if dim > 1:
            dyN_e_pg = dN_e_pg[:, :, 1, :]
        if dim > 2:
            dzN_e_pg = dN_e_pg[:, :, 2, :]

        # u for each elements as (Ne, nPe*dim) array
        u_e = self.Locates_sol_e(u, dof_n)
        # u for each elements reshaped as (Ne, nPe, dof_n) array
        u_e_n = np.reshape(u_e, (Ne, nPe, dof_n))

        grad_e_pg = FeArray.zeros(Ne, nPg, 3, 3)
        for p in range(nPg):
            grad_e_pg[:, p, :dim, 0] = np.einsum(
                "en,end->ed", dxN_e_pg[:, p], u_e_n[..., :dim]
            )
            if dim > 1:
                grad_e_pg[:, p, :dim, 1] = np.einsum(
                    "en,end->ed",
                    dyN_e_pg[:, p],
                    u_e_n[..., :dim],  # type: ignore
                )
            if dim > 2:
                grad_e_pg[:, p, :dim, 2] = np.einsum(
                    "en,end->ed",
                    dzN_e_pg[:, p],
                    u_e_n[..., :dim],  # type: ignore
                )

        return grad_e_pg

    # --------------------------------------------------------------------------------------------
    # Nodes & Elements
    # --------------------------------------------------------------------------------------------

    def Get_Elements_Nodes(
        self, nodes: _types.IntArray, exclusively=True
    ) -> _types.IntArray:
        """Returns elements that exclusively or not use the specified nodes."""
        connect = self.__connect
        connect_n_e = self.Get_connect_n_e()

        if isinstance(nodes, list):
            nodes = np.array(nodes)

        # Check that there are no excess nodes
        # It is possible that the nodes entered do not belong to the group
        if connect_n_e.shape[0] <= nodes.max():  # type: ignore
            # Remove all excess nodes
            nodes = nodes[nodes < self.Nn]

        columns = connect_n_e[nodes].nonzero()[1]

        elements = list(set(columns))

        if exclusively:
            # Check whether elements use only nodes in the node list

            # get nodes used by elements
            nodesElem = set(connect[elements].ravel())

            # detect nodes used by elements that are not in the nodes specified
            nodesIntru = list(nodesElem - set(nodes))

            # We detect the list of elements associated with unused nodes
            cols = connect_n_e[nodesIntru].nonzero()[1]
            elementsIntru = list(set(cols))

            if len(elementsIntru) > 0:
                # Remove detected elements
                elements = list(set(elements) - set(elementsIntru))

        return np.asarray(elements, dtype=int)

    def Get_Nodes_Conditions(self, func: Callable) -> _types.IntArray:  # type: ignore
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
        _types.IntArray
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

            return None  # type: ignore [return-value]

    def Get_Nodes_Point(self, point: Point.PointALike) -> _types.IntArray:
        """Returns nodes on the point."""

        point = AsPoint(point)

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
            errorX = np.abs((xn - x) / xn)
            errorY = np.abs((yn - y) / yn)
            errorZ = np.abs((zn - z) / zn)

            idx = np.where(
                (errorX <= tolerance) & (errorY <= tolerance) & (errorZ <= tolerance)
            )[0]

        return self.__nodes[idx].copy()

    def Get_Nodes_Line(self, line: "Line") -> _types.IntArray:
        """Returns nodes on the line."""

        assert isinstance(line, Line)

        unitVector = line.unitVector

        vect = self.coord - line.coord[0]

        scalarProd = np.einsum("i,ni-> n", unitVector, vect, optimize="optimal")
        crossProd = np.cross(vect, unitVector)
        norm = np.linalg.norm(crossProd, axis=1)

        eps = 1e-12

        idx = np.where(
            (norm < eps) & (scalarProd >= -eps) & (scalarProd <= line.length + eps)
        )[0]

        return self.__nodes[idx].copy()

    def Get_Nodes_Domain(self, domain: "Domain") -> _types.IntArray:
        """Returns nodes in the domain."""

        assert isinstance(domain, Line)

        xn, yn, zn = self.coord.T

        eps = 1e-12

        idx = np.where(
            (xn >= domain.pt1.x - eps)
            & (xn <= domain.pt2.x + eps)
            & (yn >= domain.pt1.y - eps)
            & (yn <= domain.pt2.y + eps)
            & (zn >= domain.pt1.z - eps)
            & (zn <= domain.pt2.z + eps)
        )[0]

        return self.__nodes[idx].copy()

    def Get_Nodes_Circle(self, circle: "Circle", onlyOnEdge=False) -> _types.IntArray:
        """Returns nodes in the circle."""

        assert isinstance(circle, Circle)

        eps = 1e-12

        vals = np.linalg.norm(self.coord - circle.center.coord, axis=1)

        if onlyOnEdge:
            idx = np.where(
                (vals <= circle.diam / 2 + eps) & (vals >= circle.diam / 2 - eps)
            )
        else:
            idx = np.where(vals <= circle.diam / 2 + eps)

        return self.__nodes[idx]

    def Get_Nodes_Cylinder(
        self, circle: "Circle", direction=[0, 0, 1], onlyOnEdge=False
    ) -> _types.IntArray:
        """Returns nodes in the cylinder."""

        assert isinstance(circle, Circle)

        rotAxis = np.cross(np.asarray(circle.n), direction)
        if np.linalg.norm(rotAxis) <= 1e-12:
            # n == direction
            i = (circle.pt1 - circle.center).coord
            J = Jacobian_Matrix(i, direction)
        else:
            # n != direction
            # (rotAxis, j, direction)
            R = circle.diam / 2
            J = Jacobian_Matrix(rotAxis, direction)
            coordN = np.einsum(
                "ij,nj->ni", np.linalg.inv(J), circle.coord - circle.center.coord
            )
            # (rotAxis, j*cj, direction)
            cj = (R - coordN[:, 1].max()) / R
            J[:, 1] *= cj

        eps = 1e-12
        coord = np.einsum(
            "ij,nj->ni", np.linalg.inv(J), self.coord - circle.center.coord
        )

        vals = np.linalg.norm(coord[:, :2], axis=1)
        if onlyOnEdge:
            idx = np.where(
                (vals <= circle.diam / 2 + eps) & (vals >= circle.diam / 2 - eps)
            )
        else:
            idx = np.where(vals <= circle.diam / 2 + eps)

        return self.__nodes[idx]

    # TODO Get_Nodes_Points
    # use Points.contour also give a normal
    # get all geom contour exept le last one
    # "Line" -> Plane equation
    # CircleArc -> Cylinder do something like Get_Nodes_Cylinder

    def Set_Tag(self, nodes: _types.IntArray, tag: str):
        """Set a tag on the nodes and elements belonging to the group of elements."""
        assert isinstance(tag, str), "tag must be a string"
        self.__Set_Nodes_Tag(nodes, tag)
        # The elements used by the nodes are automatically defined using the function
        # Get_Elements_Nodes(nodes, exclusively=True) in the following function.
        self.__Set_Elements_Tag(nodes, tag)

    def __Set_Nodes_Tag(self, nodes: _types.IntArray, tag: str):
        """Adds a tag to the nodes.

        Parameters
        ----------
        nodes : _types.IntArray
            list of nodes
        tag : str
            tag used
        """
        if nodes.size == 0:
            return
        assert isinstance(tag, str), "tag must be a string"

        if np.min(nodes) < 0 or np.max(nodes) >= self.__coordGlob.shape[0]:
            raise Exception(
                f"nodes must be within the range [0, {self.__coordGlob.shape[0] - 1}]."
            )

        self.__dict_nodes_tags[tag] = nodes

    @property
    def nodeTags(self) -> list[str]:
        """Returns node tags."""
        return list(self.__dict_nodes_tags.keys())

    @property
    def _dict_nodes_tags(self) -> dict[str, _types.IntArray]:
        """Dictionary associating tags with nodes."""
        return self.__dict_nodes_tags.copy()

    def __Set_Elements_Tag(self, nodes: _types.IntArray, tag: str):
        """Adds a tag to elements associated with nodes.

        Parameters
        ----------
        nodes : _types.IntArray
            list of nodes
        tag : str
            tag used
        """

        if nodes.size == 0:
            return
        assert isinstance(tag, str), "tag must be a string"

        # Retrieve elements associated with nodes
        elements = self.Get_Elements_Nodes(nodes=nodes, exclusively=True)
        if elements.size == 0:
            return

        if np.min(elements) < 0 or np.max(elements) >= self.Ne:
            raise Exception(f"elements must be within the range [0, {self.Ne - 1}].")

        self.__dict_elements_tags[tag] = elements

    @property
    def elementTags(self) -> list[str]:
        """returns element tags."""
        return list(self.__dict_elements_tags.keys())

    @property
    def _dict_elements_tags(self) -> dict[str, _types.IntArray]:
        """dictionary associating tags with elements."""
        return self.__dict_elements_tags.copy()

    def Get_Elements_Tag(self, tag: str) -> _types.IntArray:
        """Returns elements associated with the tag."""
        if tag in self.__dict_elements_tags:
            return self.__dict_elements_tags[tag]
        else:
            print(f"The {tag} tag is unknown")
            return np.array([])

    def Get_Nodes_Tag(self, tag: str) -> _types.IntArray:
        """Returns node associated with the tag."""
        if tag in self.__dict_nodes_tags:
            return self.__dict_nodes_tags[tag]
        else:
            print(f"The {tag} tag is unknown")
            return np.array([])

    def Locates_sol_e(
        self, sol: _types.FloatArray, dof_n: Optional[int] = None, asFeArray=False
    ) -> FeArray.FeArrayALike:
        """Locates sol on elements"""

        Nn = self.coordGlob.shape[0]

        if isinstance(dof_n, (int, float)):
            sol_e = sol[self.Get_assembly_e(dof_n)]
        elif sol.shape[0] == Nn * self.dim:
            sol_e = sol[self.Get_assembly_e(self.dim)]
        elif sol.shape[0] == Nn:
            sol_e = sol[self.__connect]
        else:
            raise Exception("Wrong dimension")

        if asFeArray:
            return FeArray.asfearray(sol_e[:, np.newaxis])
        else:
            return sol_e

    def Get_pointsInElem(
        self, coordinates_n: _types.FloatArray, elem: int
    ) -> _types.IntArray:
        """Returns the indexes of the coordinates contained in the element.

        Parameters
        ----------
        coordinates_n : _types.FloatArray
            coordinates (n, 3)
        elem : int
            element

        Returns
        -------
        _types.IntArray
            indexes of coordinates contained in element
        """

        if coordinates_n.size == 0:
            return np.array([])

        dim = self.__dim

        tol = 1e-12
        # tol = 1e-6

        if dim == 0:
            coord = self.coord[self.__connect[elem, 0]]

            idx = np.where(
                (coordinates_n[:, 0] == coord[0])
                & (coordinates_n[:, 1] == coord[1])
                & (coordinates_n[:, 2] == coord[2])
            )[0]

            return idx

        elif dim == 1:
            coord = self.coord

            p1 = self.__connect[elem, 0]
            p2 = self.__connect[elem, 1]

            # vector between the points of the segment
            vect = coord[p2] - coord[p1]
            length = np.linalg.norm(vect)
            vect = vect / length

            # vector starting from the first point of the element
            p_n = coordinates_n - coord[p1]

            cross_n = np.cross(vect, p_n, axisa=0, axisb=1)
            norm_n = np.linalg.norm(cross_n, axis=1)

            dot_n = p_n @ vect

            idx = np.where((norm_n <= tol) & (dot_n >= -tol) & (dot_n <= length + tol))[
                0
            ]

            return idx

        elif dim == 2:
            # coordinates_n (n,3)
            # points n
            # corners i [1, nPe]

            coord = self.coord
            surfaces = self.surfaces.ravel().tolist()[:-1]
            nPe = len(surfaces)
            connect_e = self.connect[elem, surfaces]
            corners_i = coord[connect_e]

            # Vectors e_i for edge segments (nPe, 3)
            indexReord = np.append(np.arange(1, nPe), 0)
            e_i = coord[connect_e[indexReord]] - corners_i
            e_i = np.einsum(
                "id,i->id", e_i, 1 / np.linalg.norm(e_i, axis=1), optimize="optimal"
            )

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
            surfaces = self.surfaces
            coord = self.coord[self.__connect[elem]]

            if self.elemType.startswith("PRISM"):
                surfaces = np.array(  # type: ignore [type-var]
                    [
                        surfaces[0, :],  # type: ignore [call-overload]
                        surfaces[1, :],  # type: ignore [call-overload]
                        surfaces[2, :],  # type: ignore [call-overload]
                        surfaces[3, :-1],  # type: ignore [call-overload]
                        surfaces[4, :-1],  # type: ignore [call-overload]
                    ],
                    dtype=object,
                )
            Nface = surfaces.shape[0]  # type: ignore [attr-defined]

            p0_f = [surface[0] for surface in surfaces]
            p1_f = [surface[1] for surface in surfaces]
            p2_f = [surface[-1] for surface in surfaces]

            i_f = Normalize(coord[p1_f] - coord[p0_f])

            j_f = Normalize(coord[p2_f] - coord[p0_f])

            n_f = Normalize(np.cross(i_f, j_f, 1, 1))

            coordinates_n_i = coordinates_n[:, np.newaxis].repeat(Nface, 1)

            v_f = coordinates_n_i - coord[p0_f]

            t_f = np.einsum("nfi,fi->nf", v_f, n_f, optimize="optimal") <= tol

            filtre = np.sum(t_f, 1)

            idx = np.where(filtre == Nface)[0]

            return idx

        else:
            raise ValueError("unknown dimensio")

    def _Get_nearby_nodes(self, coordinates_n: _types.FloatArray) -> _types.IntArray:
        """Get nearby nodes.

        Parameters
        ----------
        coordinates_n : _types.FloatArray
            coordinates (n, 3) array

        Returns
        -------
        _types.IntArray
            nearby nodes
        """

        _params._CheckIsVector(coordinates_n)

        #  Build a KDTree once for all nodes
        tree = spatial.KDTree(self.coord)

        # Find the index of the closest node for each coordinate
        _, closest_node_indices = tree.query(coordinates_n, k=1)

        # Retrieve the closest nodes
        closest_nodes = self.nodes[closest_node_indices]

        return closest_nodes

    def _Get_nearby_elements(self, coordinates_n: _types.FloatArray) -> _types.IntArray:
        """Get nearby elements.

        Parameters
        ----------
        coordinates_n : _types.FloatArray
            coordinates (n, 3) array

        Returns
        -------
        _types.IntArray
            nearby elements
        """
        # Retrieve the closest nodes
        closest_nodes = self._Get_nearby_nodes(coordinates_n)

        # Retrieve the elements associated with these nodes
        all_elements = self.Get_Elements_Nodes(closest_nodes, exclusively=False)
        unique_elements = np.unique(all_elements)

        return unique_elements

    def Get_Mapping(
        self,
        coordinates_n: _types.FloatArray,
        elements_e: Optional[_types.IntArray] = None,
        needCoordinates=False,
    ) -> tuple[
        _types.IntArray,
        _types.IntArray,
        _types.IntArray,
        Optional[_types.FloatArray],
    ]:
        """Locates coordinates within elements.

        Returns
        -------

            - detectedNodes : The nodes that have been identified within the detected elements with shape=(Nn,).
            - detectedElements_e : The elements in which the nodes have been detected with shape=(Ne,).
            - connect_e_n : The connectivity matrix that includes the nodes identified in each element with shape=(Ne, ?).
                The "?" indicates that the array may have varying dimensions along axis=1.
            - coordInElem_n : The coordinates of the identified nodes, expressed in the reference element's (ξ, η, ζ) coordinate system.
                This is applicable only if needCoordinates is set to True.
        """

        _params._CheckIsVector(coordinates_n)

        if elements_e is None:
            elements_e = self._Get_nearby_elements(coordinates_n)

        return self._Get_Mapping(coordinates_n, elements_e, needCoordinates)

    def _Get_Mapping(
        self,
        coordinates_n: _types.FloatArray,
        elements_e: _types.IntArray,
        needCoordinates=False,
    ) -> tuple[
        _types.IntArray,
        _types.IntArray,
        _types.IntArray,
        Optional[_types.FloatArray],
    ]:
        """Locates coordinates within elements.

        Returns
        -------

            - detectedNodes : The nodes that have been identified within the detected elements with shape=(Nn,).
            - detectedElements_e : The elements in which the nodes have been detected with shape=(Ne,).
            - connect_e_n : The connectivity matrix that includes the nodes identified in each element with shape=(Ne, ?).
                The "?" indicates that the array may have varying dimensions along axis=1.
            - coordInElem_n : The coordinates of the identified nodes, expressed in the reference element's (ξ, η, ζ) coordinate system.
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
            coordInElem_n = np.ones_like(coordinates_n[:, :dim], dtype=float) * np.inf
            # Use `np.inf` here to ensure that all coordinates are detected.

            # get coordinates in the reference element
            # get groupElem datas
            inDim = self.inDim
            # basis transformation matrix for each element
            sysCoord_e = self._Get_sysCoord_e()
            # This matrix can be used to project points with (x, y, z) coordinates into the element's (i, j, k) coordinate system.
            matrixType = MatrixType.mass
            jacobian_e_pg = self.Get_jacobian_e_pg(matrixType, absoluteValues=False)
            invF_e_pg = self.Get_invF_e_pg(matrixType)
            dN_tild = self._dN()
            xiOrigin = self.origin  # origin of the reference element (ξ0,η0)

            # Check whether iterative resolution is required
            # calculate the ratio between jacob max and min to detect if the element is distorted
            diff_e = jacobian_e_pg.max(1) / jacobian_e_pg.min(1)
            error_e = np.abs(1 - diff_e)  # a perfect element has an error max <= 1e-12
            # A distorted element exhibits a maximum error greater than zero.
            useIterative_e = error_e > 1e-12
        else:
            coordInElem_n = None

        def Research(e: int):
            # get element's node coordinates (x, y, z)
            coordElem = coord[connect[e]]

            # Retrieve indexes in coordinates_n that are within the element's bounds
            idxNearElem = self._Get_coord_Near(coordinates_n, coordElem, dims)

            # Return the index of idxNearElem that satisfies all the specified conditions.
            idxInElem = self.Get_pointsInElem(coordinates_n[idxNearElem], e)

            if idxInElem.size == 0:
                # here no nodes have been detected in the element
                return

            # Nodes contained within element e.
            nodesInElement = idxNearElem[idxInElem]

            # Save de detected nodes elements and connectivity matrix
            detectedNodes.extend(nodesInElement)
            connect_e_n.append(nodesInElement.tolist())
            detectedElements_e.append(e)

            if needCoordinates:
                # Inverse mapping is required here,
                # i.e., to determine the position in the reference element (ξ, η) from the physical coordinates (x, y).
                # This is particularly relevant for elements with multiple integration points (e.g., QUAD4, TRI6, TRI10, ..),
                # i.e., all elements that can be distorted and have a Jacobian ratio different from 1.

                # Project (x, y, z) coordinates into the element's (i, j, k) coordinate system if dim != inDim.
                # its the case when a 2D mesh is in 3D space
                coordElemBase = coordElem.copy()
                coordinatesBase_n = coordinates_n[nodesInElement].copy()
                if dim != inDim:
                    coordElemBase = coordElemBase @ sysCoord_e[e]
                    coordinatesBase_n = coordinatesBase_n @ sysCoord_e[e]

                # Origin of the element in (x, y, z) coordinates.
                x0 = coordElemBase[0, :dim]
                # Coordinates of the n points in (x, y, z).
                xP_n = coordinatesBase_n[:, :dim]

                if not useIterative_e[e]:
                    # The fastest method, available only for undistorted meshes.
                    xiP = xiOrigin + (xP_n - x0) @ invF_e_pg[e, 0]

                else:
                    # This is the most time-consuming method.
                    # We need to construct the Jacobian matrices here.
                    def Eval(xi: _types.FloatArray, xP: _types.FloatArray):
                        dN = _GroupElem._Eval_Functions(dN_tild, xi.reshape(1, -1))
                        F = dN[0] @ coordElemBase[:, :dim]  # jacobian matrix [J]
                        J = x0 + (xi - xiOrigin) @ F - xP  # cost function
                        return J

                    xiP = []
                    for xP in xP_n:
                        res = least_squares(Eval, 0 * xP, args=(xP,))
                        xiP.append(res.x)

                # xiP are the n coordinates of the n points in (ξ, η, ζ).
                coordInElem_n[nodesInElement, :] = np.asarray(xiP)  # type: ignore

        [Research(e) for e in elements_e]

        assert len(detectedElements_e) == len(
            connect_e_n
        ), "The number of detected elements must match the number of lines in connect_e_n."

        ar_detectedNodes = np.asarray(detectedNodes, dtype=int)
        ar_detectedElements_e = np.asarray(detectedElements_e, dtype=int)
        ar_connect_e_n = np.asarray(connect_e_n, dtype=object)

        if needCoordinates:
            # make sure each coordinates get detected
            mask = coordInElem_n == np.inf
            if np.any(mask):
                idx = np.unique(np.where(coordInElem_n == np.inf)[0])
                error = f"No elements were detected at the given coordinates {coordinates_n[idx]}."
                raise ValueError(error)

        return ar_detectedNodes, ar_detectedElements_e, ar_connect_e_n, coordInElem_n

    def _Get_coord_Near(
        self,
        coordinates_n: _types.FloatArray,
        coordElem: _types.FloatArray,
        dims: _types.FloatArray,
    ) -> _types.IntArray:
        """Get indexes in coordinates_n that are within the coordElem's bounds.

        Parameters
        ----------
        coordinates_n : _types.FloatArray
            coordinates to check
        coordElem : _types.FloatArray
            element's bounds
        dims : _types.FloatArray
            (nX, nY, nZ) = np.max(coordinates_n, 0) - np.min(coordinates_n, 0) + 1

        Returns
        -------
        _types.IntArray
            indexes in element's bounds.
        """

        nX, nY, nZ = dims

        # If all the coordinates appear the same number of times and the coordinates are of type int, we are on a grid/image.
        testShape = nX * nY - coordinates_n.shape[0] == 0
        coordinatesInImage = coordinates_n.dtype == int and testShape and nZ == 1

        if coordinatesInImage:
            # here coordinates_n are pixels

            xe = np.arange(
                np.floor(coordElem[:, 0].min()),
                np.ceil(coordElem[:, 0].max()),
                dtype=int,
            )
            ye = np.arange(
                np.floor(coordElem[:, 1].min()),
                np.ceil(coordElem[:, 1].max()),
                dtype=int,
            )
            Xe, Ye = np.meshgrid(xe, ye)

            grid_elements_coordinates = np.concatenate(([Ye.ravel()], [Xe.ravel()]))
            idx = np.ravel_multi_index(grid_elements_coordinates, (nY, nX))  # type: ignore
            # if something goes wrong, check that the mesh is correctly positioned in the image

        else:
            xn, yn, zn = coordinates_n.T
            xe, ye, ze = coordElem.T
            tol = 1e-12

            idx = np.where(
                (xn >= np.min(xe) - tol)
                & (xn <= np.max(xe) + tol)
                & (yn >= np.min(ye) - tol)
                & (yn <= np.max(ye) + tol)
                & (zn >= np.min(ze) - tol)
                & (zn <= np.max(ze) + tol)
            )[0].astype(np.uint64)

        return idx


# --------------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------------

# elems
# fmt: off
# import must be done here to avoid circular imports
from .Elems import (  # noqa: E402
    POINT,
    SEG2, SEG3, SEG4, SEG5,
    TRI3, TRI6, TRI10, TRI15,
    QUAD4, QUAD8, QUAD9,
    TETRA4, TETRA10,
    HEXA8, HEXA20, HEXA27,
    PRISM6, PRISM15, PRISM18
)
# fmt: on


class GroupElemFactory:
    DICT_GMSH_DATA: dict[int, tuple[ElemType, int, int, int, int, int, int, int]] = {
        #  key: ElemType, nPe, dim, order,  Nvertex, Nedge, Nface, Nvolume
        # fmt: off
        15: (ElemType.POINT, 1, 0, 0,       0, 0, 0, 0),
        1: (ElemType.SEG2, 2, 1, 1,         2, 0, 0, 0),
        8: (ElemType.SEG3, 3, 1, 2,         2, 1, 0, 0),
        26: (ElemType.SEG4, 4, 1, 3,        2, 2, 0, 0),
        27: (ElemType.SEG5, 5, 1, 4,        2, 3, 0, 0),
        2: (ElemType.TRI3, 3, 2, 1,         3, 0, 0, 0),
        9: (ElemType.TRI6, 6, 2, 2,         3, 3, 0, 0),
        21: (ElemType.TRI10, 10, 2, 3,      3, 6, 1, 0),
        23: (ElemType.TRI15, 15, 2, 4,      3, 9, 3, 0),
        3: (ElemType.QUAD4, 4, 2, 1,        4, 0, 0, 0),
        16: (ElemType.QUAD8, 8, 2, 2,       4, 4, 0, 0),
        10: (ElemType.QUAD9, 9, 2, 2,       4, 4, 1, 0),
        4: (ElemType.TETRA4, 4, 3, 1,       4, 0, 0, 0),
        11: (ElemType.TETRA10, 10, 3, 2,    4, 6, 0, 0),
        5: (ElemType.HEXA8, 8, 3, 1,        8, 0, 0, 0),
        17: (ElemType.HEXA20, 20, 3, 2,     8, 12, 0, 0),
        12: (ElemType.HEXA27, 27, 3, 2,     8, 12, 6, 1),
        6: (ElemType.PRISM6, 6, 3, 1,       6, 0, 0, 0),
        18: (ElemType.PRISM15, 15, 3, 2,    6, 9, 0, 0),
        13: (ElemType.PRISM18, 18, 3, 2,    6, 9, 3, 0),
        # 7: (ElemType.PYRA5, 5, 3, 1,        5, 0, 0, 0),
        # 19: (ElemType.PYRA13, 13, 3, 2,     5, 8, 0, 0),
        # 14: (ElemType.PYRA14, 14, 3, 2,     5, 8, 1, 0),
        # fmt: on
    }
    """gmshId: (ElemType, nPe, dim, order, Nvertex, Nedge, Nface, Nvolume)"""

    DICT_ELEMTYPE: dict[ElemType, tuple[int, int, int, int, int, int, int, int]] = {
        values[0]: (key, *values[1:]) for key, values in DICT_GMSH_DATA.items()
    }
    """ElemType: (gmshId, nPe, dim, order, Nvertex, Nedge, Nface, Nvolume)"""

    @staticmethod
    def Get_ElemInFos(
        gmshId: int,
    ) -> tuple[ElemType, int, int, int, int, int, int, int]:
        """return elemType, nPe, dim, order, Nvertex, Nedge, Nface, Nvolume\n
        associated with the gmsh id.
        """

        if gmshId not in GroupElemFactory.DICT_GMSH_DATA:
            raise KeyError("gmshId is unknown.")

        return GroupElemFactory.DICT_GMSH_DATA[gmshId]

    @staticmethod
    def _Create(
        gmshId: int, connect: _types.IntArray, coordGlob: _types.FloatArray
    ) -> _GroupElem:
        """Creates an element group.

        Parameters
        ----------
        gmshId : int
            id gmsh
        connect : _types.IntArray
            connection matrix storing nodes for each element (Ne, nPe)
        coordGlob : _types.FloatArray
            nodes coordinates

        Returns
        -------
        GroupeElem
            the element group
        """

        params = (gmshId, connect, coordGlob)

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
            raise KeyError("Element type unknown.")

    @staticmethod
    def Create(
        elemType: ElemType, connect: _types.IntArray, coordGlob: _types.FloatArray
    ) -> _GroupElem:
        """Creates an element group

        Parameters
        ----------
        elemType : ElemType
            element type
        connect : _types.IntArray
            connection matrix storing nodes for each element (Ne, nPe)
        coordGlob : _types.FloatArray
            nodes coordinates

        Returns
        -------
        GroupeElem
            the element group
        """

        if elemType not in GroupElemFactory.DICT_ELEMTYPE:
            raise KeyError("Element type unknown.")

        gmshId = GroupElemFactory.DICT_ELEMTYPE[elemType][0]

        return GroupElemFactory._Create(gmshId, connect, coordGlob)

    @staticmethod
    def _Get_2d_element_types(elemType: ElemType) -> list[ElemType]:
        """Returns 2d element types associated with the elementType.

        Parameters
        ----------
        elemType : ElemType
            element type

        Returns
        -------
        list[ElemType]
            the element types
        """

        assert elemType in ElemType.Get_3D(), "eleme type must be 3d element"

        if elemType is ElemType.TETRA4:
            return [ElemType.TRI3]
        elif elemType is ElemType.TETRA10:
            return [ElemType.TRI6]
        elif elemType is ElemType.HEXA8:
            return [ElemType.QUAD4]
        elif elemType is ElemType.HEXA20:
            return [ElemType.QUAD8]
        elif elemType is ElemType.HEXA27:
            return [ElemType.QUAD9]
        elif elemType is ElemType.PRISM6:
            return [ElemType.TRI3, ElemType.QUAD4]
        elif elemType is ElemType.PRISM15:
            return [ElemType.TRI6, ElemType.QUAD8]
        elif elemType is ElemType.PRISM18:
            return [ElemType.TRI6, ElemType.QUAD9]
        else:
            raise KeyError("Element type unknown.")
