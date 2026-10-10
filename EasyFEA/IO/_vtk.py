# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""VTK cell types and node orderings. EasyFEA's local node order is gmsh's."""

from enum import Enum

from ..FEM._utils import ElemType


class VTKCellType(int, Enum):
    # https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
    # Linear cells
    EMPTY_CELL = 0
    VERTEX = 1
    POLY_VERTEX = 2
    LINE = 3
    POLY_LINE = 4
    TRIANGLE = 5
    TRIANGLE_STRIP = 6
    POLYGON = 7
    PIXEL = 8
    QUAD = 9
    TETRA = 10
    VOXEL = 11
    HEXAHEDRON = 12
    WEDGE = 13
    PYRAMID = 14
    PENTAGONAL_PRISM = 15
    HEXAGONAL_PRISM = 16
    # Quadratic, isoparametric cells
    QUADRATIC_EDGE = 21
    QUADRATIC_TRIANGLE = 22
    QUADRATIC_QUAD = 23
    QUADRATIC_POLYGON = 36
    QUADRATIC_TETRA = 24
    QUADRATIC_HEXAHEDRON = 25
    QUADRATIC_WEDGE = 26
    QUADRATIC_PYRAMID = 27
    BIQUADRATIC_QUAD = 28
    TRIQUADRATIC_HEXAHEDRON = 29
    TRIQUADRATIC_PYRAMID = 37
    QUADRATIC_LINEAR_QUAD = 30
    QUADRATIC_LINEAR_WEDGE = 31
    BIQUADRATIC_QUADRATIC_WEDGE = 32
    BIQUADRATIC_QUADRATIC_HEXAHEDRON = 33
    BIQUADRATIC_TRIANGLE = 34
    # Cubic, isoparametric cell
    CUBIC_LINE = 35
    # Special class of cells formed by convex group of points
    CONVEX_POINT_SET = 41
    # Polyhedron cell (consisting of polygonal faces)
    POLYHEDRON = 42
    # Higher order cells in parametric form
    PARAMETRIC_CURVE = 51
    PARAMETRIC_SURFACE = 52
    PARAMETRIC_TRI_SURFACE = 53
    PARAMETRIC_QUAD_SURFACE = 54
    PARAMETRIC_TETRA_REGION = 55
    PARAMETRIC_HEX_REGION = 56
    # Higher order cells
    HIGHER_ORDER_EDGE = 60
    HIGHER_ORDER_TRIANGLE = 61
    HIGHER_ORDER_QUAD = 62
    HIGHER_ORDER_POLYGON = 63
    HIGHER_ORDER_TETRAHEDRON = 64
    HIGHER_ORDER_WEDGE = 65
    HIGHER_ORDER_PYRAMID = 66
    HIGHER_ORDER_HEXAHEDRON = 67
    # Arbitrary order Lagrange elements (formulated separated from generic higher order cells)
    LAGRANGE_CURVE = 68
    LAGRANGE_TRIANGLE = 69
    LAGRANGE_QUADRILATERAL = 70
    LAGRANGE_TETRAHEDRON = 71
    LAGRANGE_HEXAHEDRON = 72
    LAGRANGE_WEDGE = 73
    LAGRANGE_PYRAMID = 74
    # Arbitrary order Bezier elements (formulated separated from generic higher order cells)
    BEZIER_CURVE = 75
    BEZIER_TRIANGLE = 76
    BEZIER_QUADRILATERAL = 77
    BEZIER_TETRAHEDRON = 78
    BEZIER_HEXAHEDRON = 79
    BEZIER_WEDGE = 80
    BEZIER_PYRAMID = 81
    NUMBER_OF_CELL_TYPES = 82


DICT_ELEMTYPE_TO_VTK: dict[ElemType, VTKCellType] = {
    # (to Pyvista, to Paraview)
    # see https://dev.pyvista.org/api/utilities/_autosummary/pyvista.celltype#pyvista.CellType
    ElemType.POINT: VTKCellType.VERTEX,
    ElemType.SEG2: VTKCellType.LINE,
    ElemType.SEG3: VTKCellType.QUADRATIC_EDGE,
    ElemType.SEG4: VTKCellType.CUBIC_LINE,
    ElemType.SEG5: VTKCellType.HIGHER_ORDER_EDGE,
    ElemType.TRI3: VTKCellType.TRIANGLE,
    ElemType.TRI6: VTKCellType.QUADRATIC_TRIANGLE,
    ElemType.TRI10: VTKCellType.LAGRANGE_TRIANGLE,
    ElemType.TRI15: VTKCellType.LAGRANGE_TRIANGLE,
    ElemType.QUAD4: VTKCellType.QUAD,
    ElemType.QUAD8: VTKCellType.QUADRATIC_QUAD,
    ElemType.QUAD9: VTKCellType.BIQUADRATIC_QUAD,
    ElemType.TETRA4: VTKCellType.TETRA,
    ElemType.TETRA10: VTKCellType.QUADRATIC_TETRA,
    ElemType.HEXA8: VTKCellType.HEXAHEDRON,
    ElemType.HEXA20: VTKCellType.QUADRATIC_HEXAHEDRON,
    ElemType.HEXA27: VTKCellType.TRIQUADRATIC_HEXAHEDRON,
    ElemType.PRISM6: VTKCellType.WEDGE,
    ElemType.PRISM15: VTKCellType.QUADRATIC_WEDGE,
    ElemType.PRISM18: VTKCellType.BIQUADRATIC_QUADRATIC_WEDGE,
}
"""ElemType: CellType"""

DICT_VTK_TO_ELEMTYPE: dict[VTKCellType, ElemType] = {
    cellType: elemType for elemType, cellType in DICT_ELEMTYPE_TO_VTK.items()
}
"""CellType: ElemType"""

# reorganize the connectivity order
# because some elements in gmsh don't have the same numbering order as in vtk
# pyvista -> https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.UnstructuredGrid.celltypes.html
# vtk -> https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
# https://dev.pyvista.org/api/utilities/_autosummary/pyvista.celltype
# you can search for vtk elements on the internet
DICT_EASYFEA_TO_VTK_INDEXES: dict[ElemType, list[int]] = {
    # https://dev.pyvista.org/api/examples/_autosummary/pyvista.examples.cells.quadratichexahedron#pyvista.examples.cells.QuadraticHexahedron
    # fmt: off
    ElemType.HEXA20: [
        0, 1, 2, 3, 4, 5, 6, 7,  # vertices
        8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15 # edges
    ],    
    # https://dev.pyvista.org/api/examples/_autosummary/pyvista.examples.cells.triquadratichexahedron#pyvista.examples.cells.TriQuadraticHexahedron    
    ElemType.HEXA27: [
        0, 1, 2, 3, 4, 5, 6, 7,  # vertices
        8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15,  # edges
        22, 23, 21, 24, 20, 25,  # faces
        26  # volumes
    ],
    ElemType.PRISM15: [
        0, 1, 2, 3, 4, 5, # vertices
        6, 9, 7, 12, 14, 13, 8, 10, 11 # edges
    ],
    ElemType.PRISM18: [
        0, 1, 2, 3, 4, 5, # vertices
        6, 9, 7, 12, 14, 13, 8, 10, 11, # edges
        15, 17, 16 # faces
    ],
    # nodes 8 and 9 are switch
    ElemType.TETRA10: [
        0, 1, 2, 3, # vertices
        4, 5, 6, 7, 9, 8 # faces
    ],
    # fmt: on
}
"""ElemType: list[int]"""

DICT_VTK_TO_EASYFEA_INDEXES: dict[VTKCellType, list[int]] = {
    DICT_ELEMTYPE_TO_VTK[elemType]: [indexes.index(i) for i in range(len(indexes))]
    for elemType, indexes in DICT_EASYFEA_TO_VTK_INDEXES.items()
}
"""CellType: list[int]"""
