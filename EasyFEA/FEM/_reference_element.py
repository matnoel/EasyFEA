# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Reference element module.\n
The mesh-free element of one `ElemType`: shape functions in (ξ, η, ζ), local coordinates, entity topology and integration rules.
"""

from abc import ABC, abstractmethod

import numpy as np

from ._gauss import Gauss
from ._utils import ElemType, MatrixType
from ..Utilities import _types
from ..Utilities._cache import cache_computed_values


class _ReferenceElement(ABC):
    """The `_ReferenceElement` base class, from which every element type inherits; one instance per `ElemType`, from `GroupElemFactory._Get_reference_element`."""

    _reducedOrders: dict[MatrixType, int] = {}
    """integration orders below the matrix type's rule, kept from the former rule table"""

    def __init__(
        self,
        elemType: ElemType,
        nPe: int,
        dim: int,
        order: int,
        Nvertex: int,
        Nedge: int,
        Nface: int,
        Nvolume: int,
    ):
        self.__elemType = elemType
        self.__nPe = nPe
        self.__dim = dim
        self.__order = order
        self.__Nvertex = Nvertex
        self.__Nedge = Nedge
        self.__Nface = Nface
        self.__Nvolume = Nvolume

    def __reduce__(self):
        """Pickles as the shared instance of its elemType, not a copy."""
        from ._group_elem import GroupElemFactory

        return GroupElemFactory._Get_reference_element, (self.__elemType,)

    # --------------------------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------------------------

    @property
    def elemType(self) -> ElemType:
        """element type"""
        return self.__elemType

    @property
    def topology(self) -> str:
        """element topology"""
        return self.__elemType.topology

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

    def Get_refDistance(self, xi_n: _types.FloatArray) -> _types.FloatArray:
        """How far each ξ (Nn, dim) is inside the reference element: > 0 inside, 0 on a side, < 0 outside, nan if ξ is not finite."""

        topology = self.topology
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
            elif topology in ("SEG", "QUAD", "HEXA"):  # [-1, 1]^dim
                refDistance_n = 1 - np.abs(xi_n).max(1)
            else:
                raise ValueError(f"refDistance not implemented for {topology}.")

        return refDistance_n

    # --------------------------------------------------------------------------------------------
    # Gauss
    # --------------------------------------------------------------------------------------------

    def _Get_order(self, matrixType: MatrixType) -> int:
        """Integration order of matrixType on this element."""
        if matrixType in self._reducedOrders:
            return self._reducedOrders[matrixType]
        if isinstance(matrixType, str):
            matrixType = MatrixType(matrixType)
        return matrixType._Get_order(self.order, self.topology)

    def Get_gauss(self, matrixType: MatrixType) -> Gauss:
        """Returns integration points according to the matrix type."""
        return Gauss(self.elemType, matrixType)

    def Get_weight_pg(self, matrixType: MatrixType) -> _types.FloatArray:
        """Returns integration point weights according to the matrix type."""
        return Gauss(self.elemType, matrixType).weights

    # --------------------------------------------------------------------------------------------
    # Shape functions
    # --------------------------------------------------------------------------------------------

    @staticmethod
    def _Eval_Functions(
        functions: _types.FloatArray, gaussPoints: _types.FloatArray
    ) -> _types.FloatArray:
        """Evaluates ``functions`` (nPe, dim) at ``gaussPoints`` (nPg, dim), giving (nPg, dim, nPe)."""

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

    def __Tabulate(
        self, functions: _types.FloatArray, matrixType: MatrixType
    ) -> _types.FloatArray:
        """functions at the Gauss points of matrixType (nPg, dim, nPe), None on a point."""
        if self.dim == 0:
            return None  # type: ignore [return-value]
        return _ReferenceElement._Eval_Functions(
            functions, self.Get_gauss(matrixType).coord
        )

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

    @cache_computed_values
    def Get_N_pg(self, matrixType: MatrixType) -> _types.FloatArray:
        """`_N` at the Gauss points (nPg, 1, nPe), read-only."""
        return self.__Tabulate(self._N(), matrixType)

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

    @cache_computed_values
    def Get_dN_pg(self, matrixType: MatrixType) -> _types.FloatArray:
        """`_dN` at the Gauss points (nPg, dim, nPe), read-only."""
        return self.__Tabulate(self._dN(), matrixType)

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
    def Get_ddN_pg(self, matrixType: MatrixType) -> _types.FloatArray:
        """`_ddN` at the Gauss points (nPg, dim, nPe), read-only."""
        return self.__Tabulate(self._ddN(), matrixType)

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

    @cache_computed_values
    def Get_dddN_pg(self, matrixType: MatrixType) -> _types.FloatArray:
        """`_dddN` at the Gauss points (nPg, dim, nPe), read-only."""
        return self.__Tabulate(self._dddN(), matrixType)

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

    @cache_computed_values
    def Get_ddddN_pg(self, matrixType: MatrixType) -> _types.FloatArray:
        """`_ddddN` at the Gauss points (nPg, dim, nPe), read-only."""
        return self.__Tabulate(self._ddddN(), matrixType)
