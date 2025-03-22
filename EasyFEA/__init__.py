# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------
# utilities
# ----------------------------------------------
from .utilities import Display, Folder, MeshIO, Numba, Paraview, PyVista, Tic

# ----------------------------------------------
# geometry
# ----------------------------------------------
from . import Geoms

# ----------------------------------------------
# fem
# ----------------------------------------------
from .fem import Mesher, ElemType, Mesh, gmsh, MatrixType

# ----------------------------------------------
# materials
# ----------------------------------------------
from . import Materials

# ----------------------------------------------
# simulations
# ----------------------------------------------
from . import Simulations

# ----------------------------------------------
# version
# ----------------------------------------------
from .__about__ import __version__