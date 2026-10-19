# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
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
import itertools
from scipy import sparse, spatial
import numpy as np
from typing import Callable

# fem
from ._gauss import Gauss
from ._fearray import FeArray, Trace, Transpose, Det, Inv, Normalize

# utils
from ._utils import ElemType, MatrixType

# # others
from ..Geoms import Point, Domain, Line, Circle
from ..Geoms._utils import AsPoint
from ..Geoms import Jacobian_Matrix

from ..Utilities import _types, _params
from ..Utilities._cache import cache_computed_values, clear_cached_computed_values


class _GroupElem(ABC):
    """The `_GroupElem` base class, from which all element types inherit."""

    def __init__(
        self,
        gmshId: int,
        connect: _types.IntArray,
        coordinates: _types.FloatArray,
    ):
        """Creates a goup of elements.

        Parameters
        ----------
        gmshId : int
            gmsh id
        connect : _types.IntArray
            connectivity matrix
        coordinates : _types.FloatArray
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

        # Ensure coordinates is a (Ncoords, 3) array
        if coordinates.size != 0:  # coordinates can be empty
            error = "Must be a (Ncoords, 3) array."
            assert coordinates.ndim == 2 and coordinates.shape[1] == 3, error

        # Nodes
        nodes = np.unique(connect.ravel()).astype(int)
        Ncoords = coordinates.shape[0]
        if nodes.size != 0:
            error = f"Nodes {nodes[nodes > Ncoords]} has not corresponding entry in the coordinates array."
            assert nodes.max() + 1 <= Ncoords, error
        self.__nodes = nodes
        self.__Ncoords = Ncoords
        self.__coord = coordinates[nodes]

        # Map global nodes to local nodes
        # Global nodes correspond to the node positions in global coordinates, and local nodes correspond to the local positions in self.coord.
        self._global_to_local_nodes = np.empty(Ncoords, dtype=int)
        self._global_to_local_nodes[nodes] = np.arange(nodes.size, dtype=int)

        # Set the paritionned data.
        self._Set_partitioned_data(self.elements, self.nodes)

        # rank partition: set on rank 0 after Gather(), None otherwise
        self.__rankPartition: _types.IntArray = None

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
        coord = self.coord
        if coord.size == 0:
            # empty group (e.g. an MPI rank owning no element of this type):
            # the embedding cannot be read from the coordinates, fall back to
            # the topological dimension (a lower bound that does not disturb the
            # mesh-wide max over groups).
            return self.dim
        # embedding dimension deduced from the coordinates: 3D if any z, else
        # 2D if any y, else 1D (elements lying on the x-axis).
        _, y, z = np.abs(coord.T)
        if np.max(z) > 0:
            return 3
        elif np.max(y) > 0:
            return 2
        else:
            return 1

    @property
    def Ne(self) -> int:
        """number of elements"""
        return self.__connect.shape[0]

    @property
    def nodes(self) -> _types.IntArray:
        """nodes used by the element group. Node 'n' is on line 'n' in global coordinates"""
        return self.__nodes.copy()

    @property
    def elements(self) -> _types.IntArray:
        """elements"""
        return np.arange(self.__connect.shape[0], dtype=int)

    def _Set_partitioned_data(
        self,
        elements: _types.IntArray,
        nodes: _types.IntArray,
        rank: int = 0,
        ghostElements: _types.IntArray = [],
    ) -> None:
        """Sets the partitioned data used in mpi.

        Parameters
        ----------
        elements : _types.IntArray
            the positions of elements in the global mesh.
        nodes : _types.IntArray
            the (non-ghost) nodes.\n
        rank : int, optional
            mpi rank, by default 0
        ghostElements : _types.IntArray, optional
            the positions of ghost elements in the global mesh, by default [].

        Remark
        ------
        Ghost nodes will be computed using the given (non-ghost) nodes array.

        The four index arrays are stored **sorted**, whatever order they arrive in: `Mesher` builds them
        from python sets, so they come in hash order, while `connect` is in global-index order. Every
        consumer pairs the two through `np.searchsorted`, which requires sorted input and returns
        out-of-range indices rather than complaining when it does not get it.
        """

        Ne = self.__connect.shape[0]
        # set elements (may be smaller than Ne when ghost elements are present)
        elements = np.sort(np.asarray(elements, dtype=int))
        assert elements.size <= Ne
        ghostElements = np.sort(np.asarray(ghostElements, dtype=int))
        assert ghostElements.size <= Ne
        # get nodes and ghost nodes
        nodes = np.sort(np.asarray(nodes, dtype=int))
        ghostNodes = np.sort(np.asarray(list(set(self.nodes) - set(nodes)), dtype=int))

        self.__partitionned_data = (rank, elements, ghostElements, nodes, ghostNodes)

    def _Get_partitioned_data(
        self,
    ) -> tuple[int, _types.IntArray, _types.IntArray, _types.IntArray, _types.IntArray]:
        """Returns the partitioned data used in mpi.\n
        (rank, elements, ghostElements, nodes, ghostNodes)"""
        return self.__partitionned_data

    @property
    def _globalElements(self) -> _types.IntArray:
        """Global element index of each row of `connect`.

        `arange(Ne)` in serial. Under MPI the rows are this rank's owned and ghost elements, sorted by
        global index. Use it to index an array stored for the whole mesh (a fibre field, a per-element
        contractility) with this rank's elements — the element counterpart of `_global_to_local_nodes`.

        Not to be confused with `elements`, which is `arange(Ne)` in both cases because tags, `vol_e`
        and every other per-element array are indexed by local row.

        Always `Ne` long, so it indexes a whole-mesh array into exactly this group's rows. That holds
        at the edges too: an empty group gives an empty `int` array rather than a float one, and a rank
        holding only the boundary layer of a type (no owned element, ghosts only) gives the ghosts —
        `connect` is built from the same `unique(owned ∪ ghosts)`.
        """
        _, elements, ghostElements, _, _ = self.__partitionned_data

        # both arrays are stored sorted, so the merge is the only work left
        if ghostElements.size == 0:
            return elements

        return np.unique(np.concatenate([elements, ghostElements]))

    @property
    def _rankPartition(self) -> _types.IntArray | None:
        """Array of shape (Ne,) where rankPartition[e] is the MPI rank that owned
        element e. Set on rank 0 only after Gather(). None on all other ranks."""
        return self.__rankPartition

    @_rankPartition.setter
    def _rankPartition(self, value: _types.IntArray) -> None:
        assert isinstance(value, np.ndarray)
        self.__rankPartition = value

    @property
    def Nn(self) -> int:
        """number of nodes used by the element group"""
        return self.__nodes.size

    @property
    def Ncoords(self) -> int:
        """number of coordinates in the mesh"""
        return self.__Ncoords

    @property
    def coord(self) -> _types.FloatArray:
        """this matrix contains the element group (local) coordinates (Nn, 3)"""
        return self.__coord.copy()

    @coord.setter
    def coord(self, coord: _types.FloatArray) -> None:
        shape = (self.Ncoords, 3)
        assert coord.shape == shape, f"coord must be a {shape} array."
        self.__coord = coord[self.nodes]
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

            rows = self.connect.ravel()
            columns = np.repeat(elems, nPe)

            self.__connect_n_e = sparse.csr_matrix(
                (np.ones(nPe * Ne), (rows, columns)), shape=(self.Ncoords, Ne)
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
        # each (ndof, ndof) local matrix flattens to ndof*ndof entries (ndof = nPe*dof_n);
        # the explicit width keeps reshape valid for empty groups (Ne == 0, MPI).
        ndof2 = (nPe * dof_n) ** 2
        rowsVector_e = np.repeat(assembly_e, nPe * dof_n).reshape((Ne, ndof2))
        return rowsVector_e

    def Get_columns_e(self, dof_n: int) -> _types.IntArray:
        """Returns the column indices used to assemble local matrices into the global matrix."""
        assembly_e = self.Get_assembly_e(dof_n)
        nPe = self.nPe
        Ne = self.Ne
        ndof2 = (nPe * dof_n) ** 2
        columnsVector_e = np.repeat(assembly_e, nPe * dof_n, axis=0).reshape(
            (Ne, ndof2)
        )
        return columnsVector_e

    def _Get_sysCoord_e(self, displacementMatrix: _types.AnyArray | None = None):
        """Get the basis transformation matrix (Ne, 3, 3).\n
        [ix, jx, kx\n
        iy, jy, ky\n
        iz, jz, kz]\n

        This matrix can be used to project points with (x, y, z) coordinates into the element's (i, j, k) coordinate system.\n
        coord_e * sysCoord_e -> coordinates in element's (i, j, k) coordinate system.
        """

        coord = self.coord
        connect = self._global_to_local_nodes[self.connect]
        if displacementMatrix is not None:
            displacementMatrix = np.asarray(displacementMatrix, dtype=float)
            shape = (self.Ncoords, 3)
            assert (
                displacementMatrix.shape == shape
            ), f"displacmentMatrix must be of shape {shape}"
            coord += displacementMatrix[self.nodes]

        if self.dim in [0, 3]:
            sysCoord_e = np.eye(3)
            sysCoord_e = sysCoord_e[np.newaxis, :].repeat(self.Ne, axis=0)

        elif self.dim in [1, 2]:
            # 2D lines or elements

            if self.dim == 2:
                # Ensure outward-facing normals for all mesh faces.
                connect = connect[:, self.faces]

            points1 = coord[connect[:, 0]]
            points2 = coord[connect[:, 1]]

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
                    points3 = coord[connect[:, 2]]
                elif "QUAD" in self.elemType:
                    points3 = coord[connect[:, 3]]
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
        displacementMatrix: _types.FloatArray | None = None,
        normalize: bool = True,
    ) -> FeArray:
        """Returns the normals for each elements and gauss points (Ne, nPg, 3).

        With ``normalize=False`` the vectors are returned as computed, i.e. as the cross product of the surface tangents: their norm is then the surface jacobian, so ``weight_pg * normals`` is the area-weighted normal ``wJ_e_pg * normals_e_pg``. On the configuration selected by ``displacementMatrix`` this is the only route to a deformed surface measure, since ``Get_weightedJacobian_e_pg`` reaches the coordinates through the cached ``Get_jacobian_e_pg`` / ``_Get_J_e_pg`` and only ever knows the reference configuration.
        """

        dim = self.dim
        assert dim in [1, 2], "You can't compute normals for 0D or 3D elements."

        # get coords as a (Ne, nPe, 3) array
        coord = self.coord
        connect = self._global_to_local_nodes[self.connect]
        if displacementMatrix is not None:
            displacementMatrix = np.asarray(displacementMatrix, dtype=float)
            shape = (self.Ncoords, 3)
            assert (
                displacementMatrix.shape == shape
            ), f"displacmentMatrix must be of shape {shape}"
            coord += displacementMatrix[self.nodes]
        coord_e = coord[connect]

        # get the first derivatives of the shape functions as a (nPg, dim, nPe) array
        dN_pg = self.Get_dN_pg(matrixType)
        dNdr_pg = dN_pg[:, 0]

        dxdr_e_pg = np.einsum("pn,end->epd", dNdr_pg, coord_e, optimize="optimal")

        if dim == 1:
            normals_e_pg = np.cross((0, 0, 1), dxdr_e_pg)
        else:
            dNds_pg = dN_pg[:, 1]
            dxds_e_pg = np.einsum("pn,end->epd", dNds_pg, coord_e, optimize="optimal")
            normals_e_pg = np.cross(dxdr_e_pg, dxds_e_pg)

        if normalize:
            normals_e_pg = Normalize(normals_e_pg)

        return FeArray.asfearray(normals_e_pg)

    def _Get_gap_and_normal(
        self,
        x_e_pg: FeArray,
        elements: _types.IntArray | None = None,
        coord=None,
        matrixType: MatrixType = MatrixType.mass,
    ) -> tuple[FeArray, FeArray]:
        """Outward normal and signed gap of integration points relative to this (1D or 2D) surface group.

        For each query point, projects it onto the nearest surface sample (integration point) of this group and returns the outward unit normal there and the signed gap ``(x - closest)·n`` — positive on the outward (free) side, negative under penetration. This is the geometric kernel for contact against this surface (e.g. a rigid obstacle surface). Advanced API.

        The projection is nearest-sample: the closest point is the nearest integration point of the surface, not the exact closest point on the element. Refine ``matrixType`` for denser sampling.

        Parameters
        ----------
        x_e_pg : FeArray
            query points as a ``(Ne, nPg, 3)`` FeArray — i.e. element-/Gauss-point-structured coordinates (typically the deformed contact-surface Gauss coordinates ``x = X + u``).
        elements : _types.IntArray, optional
            restrict this surface to these element indices (e.g. a contact patch), by default all.
        coord : Coords, optional
            interior point used to orient the normals outward (``Get_normals_e_pg`` is sign-ambiguous); by default the group centroid, which is correct for convex surfaces. Pass the obstacle *volume* centroid for a robust orientation.
        matrixType : MatrixType, optional
            integration scheme giving the surface samples, by default MatrixType.mass.

        Returns
        -------
        gap_e_pg : FeArray
            signed gap, shape ``(Ne, nPg)``; negative under penetration.
        normal_e_pg : FeArray
            outward unit normal at the nearest surface sample, shape ``(Ne, nPg, 3)``.
        """
        assert self.dim in [1, 2], "only defined for 1D/2D surface groups."
        assert (
            isinstance(x_e_pg, FeArray) and x_e_pg.shape[-1] == 3
        ), "coordinates must be a (Ne, nPg, 3) FeArray."

        if elements is None:
            elements = np.arange(self.Ne)
        else:
            elements = np.asarray(elements, dtype=int).ravel()

        # surface samples and their normals
        samples_e_pg = self.Get_GaussCoordinates_e_pg(matrixType)[elements]
        normals_e_pg = self.Get_normals_e_pg(matrixType)[elements]

        # orient the normals outward, i.e. away from the reference point
        if coord is None:
            coord = self.center
        else:
            _params._CheckIsVector(coord)
        outward = samples_e_pg.mean(axis=1) - coord  # (Ne, 3)
        sign_e = np.sign(np.einsum("ei,ei->e", normals_e_pg.mean(axis=1), outward))
        sign_e[sign_e == 0] = 1.0
        normals_e_pg = normals_e_pg * FeArray(sign_e[:, None])

        samples = samples_e_pg.reshape(-1, 3)  # (Ne*nPg, 3)
        normals = normals_e_pg.reshape(-1, 3)

        # nearest surface sample for each query point, then signed gap along its normal
        x_e_pg = np.asarray(x_e_pg)
        nearest_e_pg = spatial.cKDTree(samples).query(x_e_pg)[1]  # (Ne, nPg)
        closest_e_pg = samples[nearest_e_pg]  # (Ne, nPg, 3)
        normal_e_pg = normals[nearest_e_pg]  # (Ne, nPg, 3)
        gap_e_pg = np.einsum("epi,epi->ep", x_e_pg - closest_e_pg, normal_e_pg)

        # return as FeArrays sharing the same (Ne, nPg) element/Gauss structure
        return FeArray.asfearray(gap_e_pg), FeArray.asfearray(normal_e_pg)

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

        eval_e_pg = FeArray.broadcast(eval_e_pg, *wJ_e_pg.shape, tensor_shape=())

        values_e = (wJ_e_pg * eval_e_pg).integrate()  # type: ignore [attr-defined]

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
        return float(self.length_e.sum())

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
        return float(self.area_e.sum())

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
        return float(self.volume_e.sum())

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
        When adding new 3D elements, ensure that the resulting surface normals point outward from the element.
        """

    @property
    @abstractmethod
    def faces(self) -> _types.IntArray:
        """array of indices used to form the element faces (for FEM purposes)."""

    @abstractmethod
    def Get_Local_Coords(self) -> _types.FloatArray:
        """Get local ξ, η, ζ coordinates as a (nPe, dim) numpy array"""

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
        self,
        matrixType: MatrixType,
        elements=np.array([]),
        displacementMatrix: _types.FloatArray | None = None,
    ) -> FeArray.FeArrayALike:
        """Returns integration point coordinates for each element (Ne, nPg, 3) in the (x, y, z) coordinates.

        With ``displacementMatrix`` the coordinates are evaluated on the deformed configuration, as in :meth:`Get_normals_e_pg`. This is the supported way to reach a deformed configuration: mutating ``mesh.coord`` in place does not propagate here, because each group holds its own local copy of the coordinates (see the ``coord`` setter).
        """

        N_pg = self.Get_N_pg(matrixType)

        # retrieve node coordinates
        coord = self.coord
        connect = self._global_to_local_nodes[self.connect]

        if displacementMatrix is not None:
            displacementMatrix = np.asarray(displacementMatrix, dtype=float)
            shape = (self.Ncoords, 3)
            assert (
                displacementMatrix.shape == shape
            ), f"displacmentMatrix must be of shape {shape}"
            coord += displacementMatrix[self.nodes]

        # nodes coordinates for each element
        if elements.size == 0:
            coord_e = coord[connect]
        else:
            coord_e = coord[connect[elements]]

        # localize coordinates on Gauss points
        coordo_e_p = np.einsum("pin,end->epd", N_pg, coord_e, optimize="optimal")

        return FeArray.asfearray(coordo_e_p)

    def Interpolate_e_pg(
        self, sol: _types.FloatArray, matrixType=MatrixType.rigi
    ) -> FeArray.FeArrayALike:
        """Nodal ``sol`` at the Gauss points, ``(Ne, nPg)``, or ``(Ne, nPg, dof_n)`` with ``dof_n > 1`` values per node."""
        N_pg = self.Get_N_pg(matrixType)[:, 0]  # (nPg, nPe)
        sol_e = sol.reshape(self.__Ncoords, -1)[self.connect]  # (Ne, nPe, dof_n)
        sol_e_pg = N_pg @ sol_e
        return FeArray.asfearray(sol_e_pg[..., 0] if sol_e.shape[-1] == 1 else sol_e_pg)

    # --------------------------------------------------------------------------------------------
    # Isoparametric elements
    # --------------------------------------------------------------------------------------------

    @cache_computed_values
    def _Get_J_e_pg(self, matrixType: MatrixType) -> FeArray.FeArrayALike:
        """Jacobian matrix, one row per reference axis: ``J[k, d] = ∂x_d/∂ξ_k``, in the element frame for a line or face in space."""
        if self.dim == 0:
            return None  # type: ignore [return-value]

        connect = self._global_to_local_nodes[self.connect]
        coord_e = self.coord[connect]
        # Node coordinates in the (X, Y, Z) coordinate system of each element

        rebased_coord_e = coord_e.copy()
        if self.dim != self.inDim:
            P_e = self._Get_sysCoord_e()  # transformation matrix for each element
            # matrix used to project element's points with (x, y, z) coordinates
            # into the (X, Y, Z) coordinate system.

            # check whether P_e is orthogonal
            isOrth_e = Trace(Transpose(P_e) @ P_e) == 3

            # (x, y, z) = (X, Y, Z) * P_e  <==>  aj = bi Pij
            rebased_coord_e[isOrth_e] = coord_e[isOrth_e] @ P_e[isOrth_e]

            # (x, y, z) = (X, Y, Z) * P_e^(-T)  <==>  aj = bi inv(P)ji
            rebased_coord_e[~isOrth_e] = coord_e[~isOrth_e] @ Transpose(
                Inv(P_e[~isOrth_e])
            )

        rebased_coord_e = rebased_coord_e[:, :, : self.dim]
        # (Ne, nPe, dim)

        dN_pg = self.Get_dN_pg(matrixType)

        # (nPg, dim, nPe) @ (Ne, 1, nPe, dim) -> (Ne, nPg, dim, dim)
        J_e_pg = FeArray.asfearray(dN_pg @ rebased_coord_e[:, None])

        return J_e_pg

    @cache_computed_values
    def Get_jacobian_e_pg(
        self, matrixType: MatrixType, absoluteValues=True
    ) -> FeArray.FeArrayALike:
        """Jacobian ``det J``: actual over reference length, area or volume."""
        if self.dim == 0:
            return None  # type: ignore [return-value]

        J_e_pg = self._Get_J_e_pg(matrixType)

        jacobian_e_pg = Det(J_e_pg)

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
    def _Get_invJ_e_pg(self, matrixType: MatrixType) -> FeArray.FeArrayALike:
        """Inverse jacobian matrix: ``dN_e_pg = invJ_e_pg @ dN_pg``."""
        if self.dim == 0:
            return None  # type: ignore [return-value]

        J_e_pg = self._Get_J_e_pg(matrixType)

        invJ_e_pg = Inv(J_e_pg)

        return invJ_e_pg

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

        # each function at every point at once, constants broadcast
        for n, function_nPe in enumerate(functions):
            for f in range(nF):
                evalFunctions[:, f, n] = function_nPe[f](*gaussPoints.T)  # type: ignore [index]

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
        """Shape functions in (ξ, η, ζ) reference coordinates.

        ```
        [ N₁ ]
        [ ⋮  ]
        [ Nₙ ]
        ```
        Shape: ``(nPe, 1)``.
        """

    def Get_N_pg(self, matrixType: MatrixType) -> _types.FloatArray:
        """Shape functions evaluated at Gauss points in (ξ, η, ζ).

        ```
        [ N₁  ⋯  Nₙ ]
        ```
        Shape: ``(nPg, 1, nPe)``.
        """
        if self.dim == 0:
            return None  # type: ignore [return-value]

        N = self._N()
        gauss = self.Get_gauss(matrixType)
        N_pg = _GroupElem._Eval_Functions(N, gauss.coord)

        return N_pg

    def Get_N_pg_rep(self, matrixType: MatrixType, repeat=1) -> _types.FloatArray:
        """Shape functions arranged in a block-diagonal matrix of width ``repeat``.

        ``repeat=1`` — scalar fields (thermal, phase-field):
        ```
        [ N₁  ⋯  Nₙ ]
        ```

        ``repeat=2`` — 2D vector fields (elastic):
        ```
        [ N₁  0   ⋯  Nₙ  0  ]
        [ 0   N₁  ⋯  0   Nₙ ]
        ```

        ``repeat=3`` — 3D vector fields: analogous ``3 × 3·nPe`` block-diagonal
        pattern.

        Shape: ``(nPg, repeat, repeat·nPe)``.
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
        """First derivatives of the shape functions in (ξ, η, ζ).

        ```
        [ N₁,ξ  N₁,η  N₁,ζ ]
        [  ⋮     ⋮     ⋮   ]
        [ Nₙ,ξ  Nₙ,η  Nₙ,ζ ]
        ```
        Shape: ``(nPe, dim)``.
        """
        return self._Init_Functions(1)

    def Get_dN_pg(self, matrixType: MatrixType) -> _types.FloatArray:
        """First derivatives of the shape functions, evaluated at Gauss points in (ξ, η, ζ).

        ```
        [ N₁,ξ  ⋯  Nₙ,ξ ]
        [ N₁,η  ⋯  Nₙ,η ]
        [ N₁,ζ  ⋯  Nₙ,ζ ]
        ```
        Shape: ``(nPg, dim, nPe)``.
        """
        if self.dim == 0:
            return None  # type: ignore [return-value]

        dN = self._dN()

        gauss = self.Get_gauss(matrixType)
        dN_pg = _GroupElem._Eval_Functions(dN, gauss.coord)

        return dN_pg

    @cache_computed_values
    def Get_dN_e_pg(self, matrixType: MatrixType) -> FeArray.FeArrayALike:
        """First derivatives of the shape functions in physical (x, y, z) coordinates.

        ```
        [ N₁,x  ⋯  Nₙ,x ]
        [ N₁,y  ⋯  Nₙ,y ]
        [ N₁,z  ⋯  Nₙ,z ]
        ```

        Obtained from :meth:`Get_dN_pg` via the inverse Jacobian.

        Shape: ``(Ne, nPg, dim, nPe)``.
        """

        invJ_e_pg = self._Get_invJ_e_pg(matrixType)

        dN_pg = self.Get_dN_pg(matrixType)
        # Derivation of shape functions in the (x, y, z) coordinates
        dN_e_pg = invJ_e_pg @ FeArray.from_pg(dN_pg, invJ_e_pg.shape[0])

        return dN_e_pg

    # ddN

    @abstractmethod
    def _ddN(self) -> _types.FloatArray:
        """Second derivatives of the shape functions in (ξ, η, ζ).

        ```
        [ N₁,ξξ  N₁,ηη  N₁,ζζ ]
        [  ⋮      ⋮      ⋮    ]
        [ Nₙ,ξξ  Nₙ,ηη  Nₙ,ζζ ]
        ```
        Shape: ``(nPe, dim)``.
        """
        return self._Init_Functions(2)

    @cache_computed_values
    def Get_ddN_e_pg(self, matrixType: MatrixType) -> FeArray.FeArrayALike:
        """Second derivatives of the shape functions in physical (x, y, z) coordinates.

        ```
        [ N₁,xx  ⋯  Nₙ,xx ]
        [ N₁,yy  ⋯  Nₙ,yy ]
        [ N₁,zz  ⋯  Nₙ,zz ]
        ```

        Obtained from :meth:`Get_ddN_pg` via the (squared) inverse Jacobian.

        Shape: ``(Ne, nPg, dim, nPe)``.
        """

        invJ_e_pg = self._Get_invJ_e_pg(matrixType)

        ddN_pg = self.Get_ddN_pg(matrixType)
        ddN_e_pg = invJ_e_pg @ invJ_e_pg @ FeArray.from_pg(ddN_pg, invJ_e_pg.shape[0])

        return ddN_e_pg

    def Get_ddN_pg(self, matrixType: MatrixType) -> _types.FloatArray:
        """Second derivatives of the shape functions, evaluated at Gauss points in (ξ, η, ζ).

        ```
        [ N₁,ξξ  ⋯  Nₙ,ξξ ]
        [ N₁,ηη  ⋯  Nₙ,ηη ]
        [ N₁,ζζ  ⋯  Nₙ,ζζ ]
        ```
        Shape: ``(nPg, dim, nPe)``.
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
        """Third derivatives of the shape functions in (ξ, η, ζ).

        ```
        [ N₁,ξξξ  N₁,ηηη  N₁,ζζζ ]
        [  ⋮       ⋮       ⋮     ]
        [ Nₙ,ξξξ  Nₙ,ηηη  Nₙ,ζζζ ]
        ```
        Shape: ``(nPe, dim)``.
        """
        return self._Init_Functions(3)

    def Get_dddN_pg(self, matrixType: MatrixType) -> _types.FloatArray:
        """Third derivatives of the shape functions, evaluated at Gauss points in (ξ, η, ζ).

        ```
        [ N₁,ξξξ  ⋯  Nₙ,ξξξ ]
        [ N₁,ηηη  ⋯  Nₙ,ηηη ]
        [ N₁,ζζζ  ⋯  Nₙ,ζζζ ]
        ```
        Shape: ``(nPg, dim, nPe)``.
        """
        if self.dim == 0:
            return None  # type: ignore [return-value]

        dddN = self._dddN()

        gauss = self.Get_gauss(matrixType)
        dddN_pg = _GroupElem._Eval_Functions(dddN, gauss.coord)

        return dddN_pg

    # ddddN

    @abstractmethod
    def _ddddN(self) -> _types.FloatArray:
        """Fourth derivatives of the shape functions in (ξ, η, ζ).

        ```
        [ N₁,ξξξξ  N₁,ηηηη  N₁,ζζζζ ]
        [  ⋮        ⋮        ⋮      ]
        [ Nₙ,ξξξξ  Nₙ,ηηηη  Nₙ,ζζζζ ]
        ```
        Shape: ``(nPe, dim)``.
        """
        return self._Init_Functions(4)

    def Get_ddddN_pg(self, matrixType: MatrixType) -> _types.FloatArray:
        """Fourth derivatives of the shape functions, evaluated at Gauss points in (ξ, η, ζ).

        ```
        [ N₁,ξξξξ  ⋯  Nₙ,ξξξξ ]
        [ N₁,ηηηη  ⋯  Nₙ,ηηηη ]
        [ N₁,ζζζζ  ⋯  Nₙ,ζζζζ ]
        ```
        Shape: ``(nPg, dim, nPe)``.
        """
        if self.dim == 0:
            return None  # type: ignore [return-value]

        ddddN = self._ddddN()

        gauss = self.Get_gauss(matrixType)
        ddddN_pg = _GroupElem._Eval_Functions(ddddN, gauss.coord)

        return ddddN_pg

    # --------------------------------------------------------------------------------------------
    # Finite element matrices
    # --------------------------------------------------------------------------------------------

    # Linear elastic problem

    @cache_computed_values
    def Get_B_e_pg(self, matrixType: MatrixType) -> FeArray.FeArrayALike:
        """Strain-displacement operator ``ε = B · u`` in **Kelvin–Mandel** notation.

        With ``c = 1/√2`` (the Kelvin–Mandel scaling on shear terms) and DOF
        columns ordered ``(uₓ₁, u_y₁, uₓ₂, u_y₂, …, uₓₙ, u_yₙ)`` in 2D
        (extend with ``u_z`` in 3D):

        2D — 3 strain components ``(ε_xx, ε_yy, √2·ε_xy)``:
        ```
        [ N₁,x    0     N₂,x    0    ⋯  Nₙ,x    0   ]
        [  0     N₁,y    0     N₂,y  ⋯   0     Nₙ,y ]
        [ c·N₁,y c·N₁,x c·N₂,y c·N₂,x ⋯ c·Nₙ,y c·Nₙ,x ]
        ```

        3D — 6 strain components ``(ε_xx, ε_yy, ε_zz, √2·ε_yz, √2·ε_xz, √2·ε_xy)``:
        ```
        [ N₁,x    0      0     ⋯  Nₙ,x    0      0    ]
        [  0     N₁,y    0     ⋯   0     Nₙ,y    0    ]
        [  0      0     N₁,z   ⋯   0      0     Nₙ,z  ]
        [  0     c·N₁,z c·N₁,y ⋯   0     c·Nₙ,z c·Nₙ,y]
        [ c·N₁,z  0     c·N₁,x ⋯  c·Nₙ,z  0     c·Nₙ,x]
        [ c·N₁,y c·N₁,x  0     ⋯  c·Nₙ,y c·Nₙ,x  0    ]
        ```

        The ``c`` factor makes ``B`` orthonormal so ``εᵀ·σ`` equals the inner
        product ``ε:σ`` directly — no factor-of-2 correction like Voigt needs.

        Shape: ``(Ne, nPg, 3, nPe·dim)`` in 2D, ``(Ne, nPg, 6, nPe·dim)`` in 3D.
        """

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
    def Get_leftDispPart_e_pg(self, matrixType: MatrixType) -> FeArray.FeArrayALike:
        """Left half of the elastic stiffness — precomputed ``wJ · Bᵀ``.

        ```
        leftDispPart_e_pg = wJ · Bᵀ
        K_e              = (leftDispPart_e_pg · C · B).sum(over Gauss points)
        ```

        Pulled out as a cached factor so the constitutive matrix ``C`` is the only operand left when assembling.

        Shape: ``(Ne, nPg, nPe·dim, nstrain)``.
        """

        wJ_e_pg = self.Get_weightedJacobian_e_pg(matrixType)
        B_e_pg = self.Get_B_e_pg(matrixType)

        leftDispPart = wJ_e_pg * B_e_pg.T

        return leftDispPart

    # reaction diffusion problem

    @cache_computed_values
    def Get_ReactionPart_e_pg(
        self, matrixType: MatrixType, dof_n: int = 1
    ) -> FeArray.FeArrayALike:
        """Building block of the reaction/mass term — precomputed ``wJ · Nᵀ · N``.

        ``dof_n=1`` (default) — scalar fields (thermal, phase-field).
        ``dof_n=dim``         — vector fields (elastic dynamics); uses the
        block-diagonal shape functions from :meth:`Get_N_pg_rep`.

        ```
        reactionPart_e_pg = wJ · Nᵀ · N
        M_e               = (coef · reactionPart_e_pg).sum(over Gauss points)
        ```

        Shape: ``(Ne, nPg, nPe·dof_n, nPe·dof_n)``.
        """

        weightedJacobian = self.Get_weightedJacobian_e_pg(matrixType)
        N_pg = self.Get_N_pg_rep(matrixType, dof_n)
        N_e_pg = FeArray.from_pg(N_pg, weightedJacobian.shape[0])

        ReactionPart_e_pg = weightedJacobian * N_e_pg.T @ N_e_pg

        return ReactionPart_e_pg

    @cache_computed_values
    def Get_DiffusePart_e_pg(self, matrixType: MatrixType) -> FeArray.FeArrayALike:
        """Building block of the scalar diffusion term — precomputed ``wJ · dNᵀ``.

        ```
        diffusePart_e_pg = wJ · dNᵀ
        D1_e      = (diffusePart_e_pg · A · dN).sum(over Gauss points)
        D2_e      = (diffusePart_e_pg · dN).sum(over Gauss points)
        ```

        Shape: ``(Ne, nPg, nPe, dim)``.
        """

        wJ_e_pg = self.Get_weightedJacobian_e_pg(matrixType)
        dN_e_pg = self.Get_dN_e_pg(matrixType)

        DiffusePart_e_pg = wJ_e_pg * dN_e_pg.T

        return DiffusePart_e_pg

    @cache_computed_values
    def Get_SourcePart_e_pg(
        self, matrixType: MatrixType, dof_n: int = 1
    ) -> FeArray.FeArrayALike:
        """Building block of the source/body-force term — precomputed ``wJ · Nᵀ``.

        ``dof_n=1`` (default) — scalar fields (thermal heat source, phase-field).
        ``dof_n=dim``         — vector body forces (elastic, hyperelastic);
        uses the block-diagonal shape functions from :meth:`Get_N_pg_rep`.

        ```
        sourcePart_e_pg = wJ · Nᵀ
        F_e             = (f · sourcePart_e_pg).sum(over Gauss points)
        ```

        Same pattern as :meth:`Get_ReactionPart_e_pg` but for the linear form ``∫ f · v dΩ``. Each ``dof_n`` value is cached independently.

        Shape: ``(Ne, nPg, nPe·dof_n, dof_n)``.
        """

        wJ_e_pg = self.Get_weightedJacobian_e_pg(matrixType)
        N_pg = self.Get_N_pg_rep(matrixType, dof_n)
        N_e_pg = FeArray.from_pg(N_pg, wJ_e_pg.shape[0])

        SourcePart_e_pg = wJ_e_pg * N_e_pg.T

        return SourcePart_e_pg

    def Get_Gradient_e_pg(
        self, u: _types.FloatArray, matrixType=MatrixType.rigi
    ) -> FeArray.FeArrayALike:
        """Displacement gradient ``∇u`` at the Gauss points, padded to ``3×3``.

        Parameters
        ----------
        u
            Flattened nodal field ``[ux₁, uy₁, uz₁, …, uxₙ, uyₙ, uzₙ]`` of
            size ``Nn · dim``. ``dim`` is inferred from ``u.size``.
        matrixType
            Integration scheme. Defaults to :attr:`MatrixType.rigi`.

        Returns
        -------
        FeArray
            Gradient field. Layout depends on ``dim`` (unused components are
            zeroed so the result is always ``3×3``):

            ``dim == 1``:
            ```
            [ ux,x   0    0  ]
            [  0     0    0  ]
            [  0     0    0  ]
            ```

            ``dim == 2``:
            ```
            [ ux,x  ux,y   0  ]
            [ uy,x  uy,y   0  ]
            [  0     0     0  ]
            ```

            ``dim == 3``:
            ```
            [ ux,x  ux,y  ux,z ]
            [ uy,x  uy,y  uy,z ]
            [ uz,x  uz,y  uz,z ]
            ```

            Shape: ``(Ne, nPg, 3, 3)``.
        """

        Ncoords = self.__Ncoords
        assert isinstance(u, np.ndarray) and u.size % Ncoords == 0
        dof_n = u.size // Ncoords
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
            nodes = nodes[nodes < connect_n_e.shape[0]]

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
            # the previous condition may be too restrictive: allow round-off relative to the group size
            extent = np.ptp(self.coord, axis=0).max()
            tolerance = 1e-6 * (extent if extent > 0 else 1.0)
            distances = np.linalg.norm(self.coord - point.coord, axis=1)
            idx = np.where(distances <= tolerance)[0]

        return self.__nodes[idx].copy()

    def Get_Nodes_Line(self, line: "Line") -> _types.IntArray:
        """Returns nodes on the line."""

        assert isinstance(line, Line)

        idx = np.where(line.Contains(self.coord, 1e-12))[0]
        return self.__nodes[idx].copy()

    def Get_Nodes_Domain(self, domain: "Domain") -> _types.IntArray:
        """Returns nodes in the domain."""

        assert isinstance(domain, Domain)

        idx = np.where(domain.Encloses(self.coord, 1e-12))[0]
        return self.__nodes[idx].copy()

    def Get_Nodes_Circle(self, circle: "Circle", onlyOnEdge=False) -> _types.IntArray:
        """Returns nodes in the circle.

        Both selections are taken in the circle's own plane; use `Get_Nodes_Cylinder` to select through a thickness.
        """

        assert isinstance(circle, Circle)

        test = circle.Contains if onlyOnEdge else circle.Encloses

        idx = np.where(test(self.coord, 1e-12))[0]
        return self.__nodes[idx].copy()

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

    def Set_Tag(
        self,
        nodes: _types.IntArray,
        tag: str,
        elements: _types.IntArray | None = None,
    ):
        """Set a tag on the nodes and elements belonging to the group of elements.

        Parameters
        ----------
        nodes : _types.IntArray
            list of nodes
        tag : str
            tag used
        elements : _types.IntArray, optional
            elements to tag, by default those using only `nodes`, which also catches neighbours whose nodes all lie on the tag's border
        """
        assert isinstance(tag, str), "tag must be a string"
        if nodes.size == 0:
            return
        self.__Set_Nodes_Tag(nodes, tag)
        if elements is None:
            elements = self.Get_Elements_Nodes(nodes=nodes, exclusively=True)
        self.__Set_Elements_Tag(elements, tag)

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

        if np.min(nodes) < 0 or np.max(nodes) >= self.__Ncoords:
            raise Exception(
                f"nodes must be within the range [0, {self.__Ncoords - 1}]."
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

    def __Set_Elements_Tag(self, elements: _types.IntArray, tag: str):
        """Adds a tag to elements.

        Parameters
        ----------
        elements : _types.IntArray
            list of elements
        tag : str
            tag used
        """

        elements = np.asarray(elements)
        if elements.size == 0:
            return
        assert isinstance(tag, str), "tag must be a string"

        if elements.ndim != 1 or not np.issubdtype(elements.dtype, np.integer):
            raise Exception("elements must be a 1D array of integers.")

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
        if tag not in self.__dict_elements_tags:
            raise KeyError(f"The {tag} tag is unknown.")
        return self.__dict_elements_tags[tag]

    def Get_Nodes_Tag(self, tag: str) -> _types.IntArray:
        """Returns node associated with the tag."""
        if tag not in self.__dict_nodes_tags:
            raise KeyError(f"The {tag} tag is unknown.")
        return self.__dict_nodes_tags[tag]

    def Locates_sol_e(
        self, sol: _types.FloatArray, dof_n: int | None = None
    ) -> _types.FloatArray:
        """Locates sol on elements, ``(Ne, ...)``."""

        Ncoords = self.__Ncoords

        if isinstance(dof_n, (int, float)):
            sol_e = sol[self.Get_assembly_e(dof_n)]
        elif sol.shape[0] == Ncoords * self.dim:
            sol_e = sol[self.Get_assembly_e(self.dim)]
        elif sol.shape[0] == Ncoords:
            sol_e = sol[self.connect]
        else:
            raise Exception("Wrong dimension")

        return sol_e

    def Get_pointsInElem(
        self, coordinates_n: _types.NumberArray, elem: int
    ) -> _types.IntArray:
        """Indexes of the coordinates (n, 3) contained in the element."""
        coordinates_n = np.asarray(coordinates_n, dtype=float).reshape(-1, 3)
        elements_n = np.full(coordinates_n.shape[0], elem)
        refDistance_n = self.__Get_refDistance(coordinates_n, elements_n)[0]
        return np.flatnonzero(refDistance_n >= -1e-10)

    def Get_Mapping(
        self,
        coordinates_n: _types.NumberArray,
        elements_e: _types.IntArray | None = None,
        needCoordinates=False,
    ) -> tuple[
        _types.IntArray,
        _types.IntArray,
        _types.IntArray,
        _types.FloatArray | None,
    ]:
        """Locates coordinates within elements.

        Returns
        -------

            - detectedNodes : The nodes that have been identified within the detected elements with shape=(Nn,).
            - detectedElements_e : The elements in which the nodes have been detected with shape=(Ne,).
            - connect_e_n : The connectivity matrix that includes the nodes identified in each element with shape=(Ne, ?).
                The "?" indicates that the array may have varying dimensions along axis=1.
            - coordInElem_n : The coordinates of the identified nodes, expressed in the reference element's (ξ, η, ζ) coordinate system.
                This is applicable only if needCoordinates is set to True. A node in several elements takes its ξ in the most inside one.
        """

        _params._CheckIsVector(coordinates_n)

        # (node, element) pairs, _p = per pair
        nodes_p, elements_p, xi_p, refDistance_p = self._Get_all_containing_elements(
            coordinates_n, elements_e
        )

        # group the pairs by element: each element gets its nodes
        order_p = np.lexsort((nodes_p, elements_p))
        detectedElements_e, start_e = np.unique(elements_p[order_p], return_index=True)
        connect_e_n = np.empty(detectedElements_e.size, dtype=object)  # (Ne, ?)
        for e, nodes in enumerate(np.split(nodes_p[order_p], start_e[1:])):
            connect_e_n[e] = nodes

        if needCoordinates:
            # ξ in the most inside element: sort by node, then by decreasing refDistance
            order_p = np.lexsort((-refDistance_p, nodes_p))
            best_n = order_p[np.unique(nodes_p[order_p], return_index=True)[1]]
            coordInElem_n = np.full((len(coordinates_n), self.dim), np.inf)
            coordInElem_n[nodes_p[best_n]] = xi_p[best_n]
        else:
            coordInElem_n = None

        return np.unique(nodes_p), detectedElements_e, connect_e_n, coordInElem_n

    def _Get_containing_elements(
        self,
        coordinates_n: _types.NumberArray,
        elements_e: _types.IntArray | None = None,
        keep: (
            Callable[[_types.IntArray, _types.IntArray], _types.BoolArray] | None
        ) = None,
    ) -> tuple[_types.IntArray, _types.FloatArray]:
        """Element (-1 if none) and ξ (inf if none) of each node, in its most inside element; keep(nodes_p, elements_p) filters the (node, element) pairs."""

        # (node, element) pairs, _p = per pair
        nodes_p, elements_p, xi_p, refDistance_p = self._Get_all_containing_elements(
            coordinates_n, elements_e
        )
        if keep is not None:
            keep_p = keep(nodes_p, elements_p)
            nodes_p, elements_p = nodes_p[keep_p], elements_p[keep_p]
            xi_p, refDistance_p = xi_p[keep_p], refDistance_p[keep_p]

        # sort by node, then by decreasing refDistance: the first pair of each node is the best
        order_p = np.lexsort((-refDistance_p, nodes_p))
        best_n = order_p[np.unique(nodes_p[order_p], return_index=True)[1]]
        Nn = len(coordinates_n)
        element_n = np.full(Nn, -1)  # (Nn,)
        xi_n = np.full((Nn, self.dim), np.inf)  # (Nn, dim)
        element_n[nodes_p[best_n]] = elements_p[best_n]
        xi_n[nodes_p[best_n]] = xi_p[best_n]
        return element_n, xi_n

    def _Get_all_containing_elements(
        self,
        coordinates_n: _types.NumberArray,
        elements_e: _types.IntArray | None = None,
        chunk: int = 200_000,
    ) -> tuple[_types.IntArray, _types.IntArray, _types.FloatArray, _types.FloatArray]:
        """Every (node, element, ξ, refDistance) with the node inside the element; candidates from a ball query on element bounding spheres."""

        coordinates_n = np.asarray(coordinates_n, dtype=float)  # (Nn, 3)
        elements_e = self.elements if elements_e is None else np.asarray(elements_e)
        Nn = coordinates_n.shape[0]
        empty = (
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
            np.empty((0, self.dim)),
            np.empty(0),
        )
        if elements_e.size == 0 or Nn == 0:
            return empty

        # bounding sphere of each element
        connect = self._global_to_local_nodes[self.connect]
        coordElem_e = self.coord[connect[elements_e]]  # (Ne, nPe, 3)
        center_e = coordElem_e.mean(1)  # (Ne, 3)
        radius_e = np.linalg.norm(coordElem_e - center_e[:, None], axis=2).max(1)
        radius_e *= 1 + 1e-8  # (Ne,)
        tree = spatial.cKDTree(coordinates_n)

        # elements by chunks of about `chunk` (node, element) pairs, to bound the memory
        count_e = tree.query_ball_point(
            center_e, radius_e, workers=-1, return_length=True
        )
        chunk_e = np.cumsum(count_e) // chunk
        starts = np.flatnonzero(np.diff(chunk_e)) + 1

        # _p = per (node, element) pair
        containing = [empty]
        for iElem in np.split(np.arange(elements_e.size), starts):
            # nodes in each element's own sphere, one list per element
            near_e = tree.query_ball_point(center_e[iElem], radius_e[iElem], workers=-1)
            # flattened into pairs
            elements_p = np.repeat(elements_e[iElem], count_e[iElem])
            nodes_p = np.fromiter(
                itertools.chain.from_iterable(near_e), int, count_e[iElem].sum()
            )
            # node inside the element
            refDistance_p, xi_p = self.__Get_refDistance(
                coordinates_n[nodes_p], elements_p
            )
            inside_p = refDistance_p >= -1e-10
            containing.append(
                (
                    nodes_p[inside_p],
                    elements_p[inside_p],
                    xi_p[inside_p],
                    refDistance_p[inside_p],
                )
            )

        return tuple(np.concatenate(c) for c in zip(*containing))  # type: ignore [return-value]

    def __Get_refDistance(
        self, coordinates_n: _types.FloatArray, elements_n: _types.IntArray
    ) -> tuple[_types.FloatArray, _types.FloatArray]:
        """(refDistance, ξ) of each node in its element: how far ξ is inside the reference element, > 0 inside, 0 on a side, < 0 outside; -inf off a 0D, 1D or 2D element."""

        topology = self.topology
        if topology not in ("POINT", "SEG", "QUAD", "HEXA", "TRI", "TETRA", "PRISM"):
            raise ValueError(f"refDistance not implemented for {topology}.")

        xi_n = self._Get_xi(coordinates_n, elements_n)  # (Nn, dim)

        with np.errstate(invalid="ignore"):
            if topology == "POINT":
                refDistance_n = np.zeros(xi_n.shape[0])
            elif topology in ("TRI", "TETRA"):
                # ξ_i ≥ 0 and Σ ξ_i ≤ 1
                refDistance_n = np.minimum(xi_n.min(1), 1 - xi_n.sum(1))
            elif topology == "PRISM":
                # triangle in (ξ, η), [-1, 1] in ζ
                tri_n = xi_n[:, :2]
                refDistance_n = np.minimum.reduce(
                    [tri_n.min(1), 1 - tri_n.sum(1), 1 - np.abs(xi_n[:, 2])]
                )
            else:  # SEG, QUAD, HEXA: [-1, 1]^dim
                refDistance_n = 1 - np.abs(xi_n).max(1)
        refDistance_n[~np.isfinite(refDistance_n)] = -np.inf  # (Nn,)

        # a surface (line, point) in a higher dim space: ξ only gives the projection, check the distance
        if self.dim < 3 and coordinates_n.shape[0] > 0:
            connect = self._global_to_local_nodes[self.connect]
            coordElem_n = self.coord[connect[elements_n]]  # (Nn, nPe, 3)
            with np.errstate(all="ignore"):
                if self.dim == 0:
                    x_n = coordElem_n[:, 0]
                else:
                    N_n = self._Eval_Functions(self._N(), xi_n)[:, 0]  # (Nn, nPe)
                    x_n = np.einsum("np,npd->nd", N_n, coordElem_n)  # x(ξ) (Nn, 3)
                distance_n = np.linalg.norm(x_n - coordinates_n, axis=1)
                size_n = np.ptp(coordElem_n, 1).max(1)
                refDistance_n[~(distance_n <= 1e-10 * size_n)] = -np.inf

        return refDistance_n, xi_n

    def _Get_xi(
        self, coordinates_n: _types.NumberArray, elements_n: _types.IntArray
    ) -> _types.FloatArray:
        """(ξ, η, ζ) of each node in its element by inverse mapping, outside the reference element if the node is outside, inf if Newton fails."""

        dim = self.dim
        coordinates_n = np.asarray(coordinates_n, dtype=float)  # (Nn, 3)
        Nn = coordinates_n.shape[0]
        if dim == 0 or Nn == 0:
            return np.zeros((Nn, dim))

        # _u = per element queried, elements_u[inverse_n] = elements_n
        elements_u, inverse_n = np.unique(elements_n, return_inverse=True)
        connect = self._global_to_local_nodes[self.connect]
        coordElem_u = self.coord[connect[elements_u]]  # (Nu, nPe, 3)
        x_n = coordinates_n
        if dim != self.inDim:
            # element's (i, j, k) coordinate system (2D mesh in 3D space)
            sysCoord_u = self._Get_sysCoord_e()[elements_u]  # (Nu, 3, 3)
            coordElem_u = coordElem_u @ sysCoord_u
            x_n = np.einsum("nd,ndj->nj", x_n, sysCoord_u[inverse_n])
        coordElem_u = coordElem_u[..., :dim]  # (Nu, nPe, dim)
        x_n = x_n[:, :dim]  # (Nn, dim)

        # affine element: x = ξ J + b, i.e. [ξ_i, 1] [J; b] = x_i at every node
        # L = [ξ_i, 1] (nPe, dim+1)
        L = np.column_stack([self.Get_Local_Coords(), np.ones(self.nPe)])
        invL = np.linalg.pinv(L)  # (dim+1, nPe)
        # part of the node coordinates that no affine map gives: (I - L L⁺) x_i
        nonAffine_u = (np.eye(self.nPe) - L @ invL) @ coordElem_u  # (Nu, nPe, dim)
        size_u = np.ptp(coordElem_u, axis=1).max(1)  # (Nu,)
        affine_u = np.abs(nonAffine_u).max((1, 2)) <= 1e-12 * size_u
        affine_n = affine_u[inverse_n]
        iterative_n = ~affine_n

        # affine: [J; b] = L⁺ x_i, then ξ = (x - b) J⁻¹
        Jb_u = invL @ coordElem_u  # (Nu, dim+1, dim)
        J_u, b_u = Jb_u[:, :dim], Jb_u[:, dim]  # (Nu, dim, dim), (Nu, dim)
        invJ_u = np.full_like(J_u, np.nan)
        invJ_u[affine_u] = np.linalg.inv(J_u[affine_u])
        xi_n = np.empty((Nn, dim))
        xi_n[affine_n] = np.einsum(
            "nd,ndk->nk",
            x_n[affine_n] - b_u[inverse_n[affine_n]],
            invJ_u[inverse_n[affine_n]],
        )
        if not iterative_n.any():
            return xi_n

        # iterative: Newton on x(ξ) = Σ N_i(ξ) x_i = x, from the reference element's center
        # _i = per iterative node, inf where it does not converge
        coordElem_i = coordElem_u[inverse_n[iterative_n]]  # (Ni, nPe, dim)
        x_i = x_n[iterative_n]  # (Ni, dim)
        N, dN = self._N(), self._dN()
        Ni = x_i.shape[0]
        xi_i = np.tile(self.Get_Local_Coords().mean(0), (Ni, 1))  # (Ni, dim)
        active_i = np.ones(Ni, dtype=bool)
        with np.errstate(all="ignore"):
            for _ in range(20):
                # _a = per active node
                nodes_a = np.flatnonzero(active_i)
                if nodes_a.size == 0:
                    break
                xi_a = xi_i[nodes_a]
                coordElem_a = coordElem_i[nodes_a]  # (Na, nPe, dim)
                N_a = self._Eval_Functions(N, xi_a)[:, 0]  # (Na, nPe)
                dN_a = self._Eval_Functions(dN, xi_a)  # (Na, dim, nPe)
                residual_a = np.einsum("np,npd->nd", N_a, coordElem_a) - x_i[nodes_a]
                # Jᵀ[d, k] = ∂x_d/∂ξ_k (Na, dim, dim)
                Jt_a = np.einsum("nkp,npd->ndk", dN_a, coordElem_a)
                regular_a = np.isfinite(Jt_a).all((1, 2)) & (
                    np.abs(np.linalg.det(Jt_a)) > 0
                )
                # Jᵀ δ = -r
                delta_a = np.full_like(xi_a, np.inf)
                delta_a[regular_a] = np.linalg.solve(
                    Jt_a[regular_a], -residual_a[regular_a, :, None]
                )[..., 0]
                xi_i[nodes_a] = xi_a + delta_a
                # stops once converged or diverged
                active_i[nodes_a] = ~(np.abs(delta_a).max(1) <= 1e-12)
                active_i[nodes_a[~np.isfinite(delta_a).all(1)]] = False
        xi_i[active_i | ~np.isfinite(xi_i).all(1)] = np.inf
        xi_n[iterative_n] = xi_i

        return xi_n


# --------------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------------

# import must be done here to avoid circular imports
from . import Elems  # noqa: E402


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
            raise KeyError(
                f"gmshId {gmshId} is not supported yet, but it can be added to DICT_GMSH_DATA."
            )

        return GroupElemFactory.DICT_GMSH_DATA[gmshId]

    GROUP_CLASS_MAP: dict[ElemType, type[_GroupElem]] = {
        ElemType.POINT: Elems.POINT,
        ElemType.SEG2: Elems.SEG2,
        ElemType.SEG3: Elems.SEG3,
        ElemType.SEG4: Elems.SEG4,
        ElemType.SEG5: Elems.SEG5,
        ElemType.TRI3: Elems.TRI3,
        ElemType.TRI6: Elems.TRI6,
        ElemType.TRI10: Elems.TRI10,
        ElemType.TRI15: Elems.TRI15,
        ElemType.QUAD4: Elems.QUAD4,
        ElemType.QUAD8: Elems.QUAD8,
        ElemType.QUAD9: Elems.QUAD9,
        ElemType.TETRA4: Elems.TETRA4,
        ElemType.TETRA10: Elems.TETRA10,
        ElemType.HEXA8: Elems.HEXA8,
        ElemType.HEXA20: Elems.HEXA20,
        ElemType.HEXA27: Elems.HEXA27,
        ElemType.PRISM6: Elems.PRISM6,
        ElemType.PRISM15: Elems.PRISM15,
        ElemType.PRISM18: Elems.PRISM18,
    }

    @staticmethod
    def _Create(
        gmshId: int, connect: _types.IntArray, coordinates: _types.FloatArray
    ) -> _GroupElem:
        """Creates an element group.

        Parameters
        ----------
        gmshId : int
            id gmsh
        connect : _types.IntArray
            connection matrix storing nodes for each element (Ne, nPe)
        coordinates : _types.FloatArray
            coordinate matrix (contains all mesh coordinates)

        Returns
        -------
        GroupeElem
            the element group
        """

        elemType = GroupElemFactory.Get_ElemInFos(gmshId)[0]

        try:
            GroupElem = GroupElemFactory.GROUP_CLASS_MAP[elemType]
        except KeyError:
            raise NotImplementedError(f"Beam not implemented for {elemType}")

        return GroupElem(gmshId, connect, coordinates)

    @staticmethod
    def Create(
        elemType: ElemType, connect: _types.IntArray, coordinates: _types.FloatArray
    ) -> _GroupElem:
        """Creates an element group

        Parameters
        ----------
        elemType : ElemType
            element type
        connect : _types.IntArray
            connection matrix storing nodes for each element (Ne, nPe)
        coordinates : _types.FloatArray
            coordinate matrix (contains all mesh coordinates)

        Returns
        -------
        GroupeElem
            the element group
        """

        if elemType not in GroupElemFactory.DICT_ELEMTYPE:
            raise KeyError("Element type unknown.")

        gmshId = GroupElemFactory.DICT_ELEMTYPE[elemType][0]

        return GroupElemFactory._Create(gmshId, connect, coordinates)

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
