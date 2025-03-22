# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------
# utilities
# ----------------------------------------------
from .utilities import (Display, Folder, Numba, Paraview, PyVista, Tic,
                        meshio_Interface)

# ----------------------------------------------
# geometry
# ----------------------------------------------
from . import Geoms

# ----------------------------------------------
# fem
# ----------------------------------------------
from .fem import (Mesher, gmsh,
                  Mesh, MatrixType, ElemType)

# ----------------------------------------------
# materials
# ----------------------------------------------
from . import Materials

# ----------------------------------------------
# simulations
# ----------------------------------------------
from . import Simulations