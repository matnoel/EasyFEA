# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import importlib
from typing import TYPE_CHECKING

from .Utilities import Terminal, Folder, Tic
from .FEM import Mesher, ElemType, Mesh, MatrixType
from . import Models, Simulations, IO
from .Simulations.Solvers import SolverType, AlgoType
from .__about__ import __version__

if TYPE_CHECKING:
    from .Viz import Matplotlib, PyVista

_LAZY_VIEWERS = ("Matplotlib", "PyVista")  # pyplot 188 ms, pyvista 151 ms


def __getattr__(name: str):
    """Viewers imported on first access (PEP 562)."""
    if name in _LAZY_VIEWERS:
        return importlib.import_module(f".Viz.{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_VIEWERS))
