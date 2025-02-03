# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from ._utils import ElemType, MatrixType
from ._mesh import Mesh, Mesh_Optim, Calc_projector
from ._boundary_conditions import BoundaryCondition, LagrangeCondition
from ._gmsh_interface import Mesher, gmsh
from ._group_elems import _GroupElem, GroupElemFactory