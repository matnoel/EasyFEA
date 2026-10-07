# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import importlib
from typing import TYPE_CHECKING

from ._observers import Observable, _IObserver
from ._params import (
    _CheckIsPositive,
    _CheckIsNegative,
    _CheckIsInIntervalcc,
    _CheckIsInIntervaloo,
)
from ._tic import Tic
from ._types import Number, Numbers, FloatArray, IntArray, AnyArray, Coords
from . import Terminal
from . import Folder

if TYPE_CHECKING:
    from . import Matplotlib, MeshIO, Paraview, PyVista, Vizir, GLTF, USD

_LAZY_FRONT_ENDS = (
    "Matplotlib",
    "MeshIO",
    "Paraview",
    "PyVista",
    "Vizir",
    "GLTF",
    "USD",
)


def __getattr__(name: str):
    """Front-ends imported on first access (PEP 562)."""
    if name in _LAZY_FRONT_ENDS:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_FRONT_ENDS))
