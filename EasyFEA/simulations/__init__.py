# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from ._simu import _Simu, _Init_obj, _Get_values, Load_Simu

# from ._thermal import ThermalSimu
from ._utils import Save_pickle, Load_pickle
from .Solvers import AlgoType, ResolType, SolverType, Solve_simu
