# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import inspect

VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def _named(func) -> list[tuple[str, inspect._ParameterKind]]:
    params = inspect.signature(func).parameters.values()
    return [(p.name, p.kind) for p in params if p.kind not in VARIADIC]


def assert_spelled_as(protocolMethod, func) -> None:
    """`func` takes every parameter of `protocolMethod`, positional ones at the same place; mypy checks neither positional names nor keywords a `**kwargs` would swallow."""
    expected = _named(protocolMethod)[1:]  # without self
    got = dict(_named(func))
    positional = [n for n, k in _named(func) if k != k.KEYWORD_ONLY]
    for i, (name, kind) in enumerate(expected):
        assert name in got, f"{func.__qualname__} lacks '{name}'"
        if kind != kind.KEYWORD_ONLY:
            assert positional[i] == name, f"{func.__qualname__}: '{name}' at {i}"
