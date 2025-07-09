# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from ._boundary_conditions import BoundaryCondition, LagrangeCondition
from ._gauss import Gauss
from ._gmsh_interface import Mesher, gmsh
from ._forms import BiLinearForm, LinearForm
from ._group_elems import _GroupElem, GroupElemFactory
from ._mesh import Mesh, Calc_projector, Mesh_Optim
from ._utils import ElemType, MatrixType
from ._linalg import FeArray, Transpose, Trace, Det, Inv, TensorProd, Norm

# must be after the import of FeArray, _GroupElem, MatrixType
from ._field import (
    Field,
    Sym_Grad,
)
