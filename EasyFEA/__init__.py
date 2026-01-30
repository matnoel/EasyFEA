# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

BUILDING_GALLERY = False

# utilities
from .Utilities import Display, Folder, Paraview, PyVista, Vizir, MeshIO, Tic

# fem
from .FEM import Mesher, ElemType, Mesh, MatrixType

# simulations
from .Simulations.Solvers import SolverType

# version
from .__about__ import __version__
