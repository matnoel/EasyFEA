# Copyright (C) 2021-2024 Université Gustave Eiffel. All rights reserved.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.txt for more information.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------
# utilities
# ----------------------------------------------
from .utilities import (Display, Folder, Tic,
                        Numba_Interface,
                        Paraview_Interface,
                        PyVista_Interface)

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