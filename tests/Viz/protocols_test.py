# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Parameter names of the protocols, which mypy does not check: positional names, and keywords a `**kwargs` would swallow."""

import inspect

import pytest

from EasyFEA import IO, Matplotlib, PyVista
from EasyFEA.IO._protocols import MeshSaver, MeshLoader, SimuSaver
from EasyFEA.Viz._protocols import Viewer

CLAIMS = [
    (Viewer, Matplotlib),
    (Viewer, PyVista),
    (MeshSaver, IO.Gmsh),
    (MeshSaver, IO.Medit),
    (MeshSaver, IO.Ensight),
    (MeshSaver, IO.USD),
    (MeshSaver, IO.GLTF),
    (MeshLoader, IO.Gmsh),
    (MeshLoader, IO.Medit),
    (MeshLoader, IO.Ensight),
    (SimuSaver, IO.Gmsh),
    (SimuSaver, IO.Paraview),
    (SimuSaver, IO.Vizir),
    (SimuSaver, IO.USD),
    (SimuSaver, IO.GLTF),
]

VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def _named(func) -> list[tuple[str, inspect._ParameterKind]]:
    params = inspect.signature(func).parameters.values()
    return [(p.name, p.kind) for p in params if p.kind not in VARIADIC]


@pytest.mark.parametrize(
    "protocol, module", CLAIMS, ids=[f"{p.__name__}-{m.__name__}" for p, m in CLAIMS]
)
def test_parameters_are_spelled_as_the_protocol(protocol, module):
    for member in [name for name in vars(protocol) if not name.startswith("_")]:
        expected = _named(getattr(protocol, member))[1:]  # without self
        got = dict(_named(getattr(module, member)))
        positional = [
            n for n, k in _named(getattr(module, member)) if k != k.KEYWORD_ONLY
        ]
        for i, (name, kind) in enumerate(expected):
            assert name in got, f"{module.__name__}.{member} lacks '{name}'"
            if kind != kind.KEYWORD_ONLY:
                assert (
                    positional[i] == name
                ), f"{module.__name__}.{member}: '{name}' at {i}"
