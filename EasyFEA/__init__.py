# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import importlib
from typing import TYPE_CHECKING

# utilities
from .Utilities import Terminal, Folder, Tic, _LAZY_FRONT_ENDS

# fem
from .FEM import Mesher, ElemType, Mesh, MatrixType

# version
from .__about__ import __version__

if TYPE_CHECKING:
    from .Utilities import Matplotlib, Paraview, PyVista, Vizir, MeshIO, GLTF, USD
    from . import Models, Simulations
    from .Simulations.Solvers import SolverType, AlgoType

_LAZY_MODULES = {
    **{name: f".Utilities.{name}" for name in _LAZY_FRONT_ENDS},
    "Models": ".Models",
    "Simulations": ".Simulations",
}
_LAZY_ATTRIBUTES = {
    "SolverType": ".Simulations.Solvers",
    "AlgoType": ".Simulations.Solvers",
}


def __getattr__(name: str):
    """Front-ends, Models, Simulations and solver enums imported on first access (PEP 562)."""
    if name in _LAZY_MODULES:
        return importlib.import_module(_LAZY_MODULES[name], __name__)
    if name in _LAZY_ATTRIBUTES:
        return getattr(importlib.import_module(_LAZY_ATTRIBUTES[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_MODULES) | set(_LAZY_ATTRIBUTES))
