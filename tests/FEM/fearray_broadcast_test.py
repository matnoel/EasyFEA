# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Every value `FeArray.broadcast` can meet, at regular and coinciding sizes: each cell is right or raises."""

import numpy as np
import pytest

from EasyFEA.FEM._linalg import FeArray

pytestmark = pytest.mark.xfail(strict=True, reason="#64 strict broadcast")

# (value, tensor shape, accepted, meaning): value "" is a float, "fe:..." a FeArray;
# meaning "full" (constant or already per point), "e" per element, "pg" per point (the hole), None ill-formed
S, V, M = "", "n", "n, n"
CASES = [
    ("", S, True, "full"),
    ("Ne", S, True, "e"),
    ("Ne, nPg", S, True, "full"),
    ("fe:Ne, nPg", S, True, "full"),
    ("nPg", S, False, "pg"),
    ("Ne, 1", S, False, "full"),
    ("1, nPg", S, False, "full"),
    ("Ne, nPg + 2", S, False, None),
    ("Ne + 1", S, False, None),
    ("n", S, False, None),
    ("Ne, nPg, n", S, False, None),
    ("fe:Ne, 1", S, False, "full"),
    ("fe:Ne, nPg, n", S, False, None),
    ("n", V, True, "full"),
    ("Ne, n", V, True, "e"),
    ("Ne, nPg, n", V, True, "full"),
    ("fe:Ne, nPg, n", V, True, "full"),
    ("", V, False, None),
    ("Ne, nPg", V, False, None),
    ("nPg, n", V, False, "pg"),
    ("n + 1", V, False, None),
    ("n, n", M, True, "full"),
    ("Ne, n, n", M, True, "e"),
    ("Ne, nPg, n, n", M, True, "full"),
    ("fe:Ne, nPg, n, n", M, True, "full"),
    ("Ne", M, False, None),
    ("n", M, False, None),
    ("nPg, n, n", M, False, "pg"),
    ("Ne, 1, n, n", M, False, "full"),
    ("Ne, n, n + 1", M, False, None),
    ("fe:1, 1, n, n", M, False, "full"),
]

SIZES = {
    "regular": (5, 4, 3),
    "Ne==nPg": (4, 4, 3),
    "nPg==n": (5, 3, 3),
    "Ne==nPg==n": (3, 3, 3),
    "Ne==1": (1, 4, 3),
    "nPg==1": (5, 1, 3),
}


def _shape(spec: str, Ne: int, nPg: int, n: int) -> tuple:
    return eval(f"({spec},)", {}, dict(Ne=Ne, nPg=nPg, n=n)) if spec else ()


@pytest.mark.parametrize("sizes", SIZES.values(), ids=SIZES.keys())
@pytest.mark.parametrize(
    "spec, tensor, accepted, meaning", CASES, ids=[f"{c[0]}|{c[1]}" for c in CASES]
)
def test_broadcast(spec, tensor, accepted, meaning, sizes):
    """A rejected value may become valid where sizes coincide; it must then be right."""
    Ne, nPg, n = sizes
    if meaning == "pg" and Ne == nPg:
        pytest.skip("(nPg, ...) at Ne == nPg reads per element: the accepted hole")
    shape = _shape(spec.removeprefix("fe:"), Ne, nPg, n)
    # distinct values, so a misread axis cannot pass for the right one
    value = np.arange(1.0, np.prod(shape) + 1).reshape(shape) if shape else 2.0
    if spec.startswith("fe:"):
        value = FeArray.asfearray(value)
    t = _shape(tensor, Ne, nPg, n)

    try:
        res = FeArray.broadcast(value, Ne, nPg, tensor_shape=t)
    except ValueError:
        assert not accepted, f"{spec} raised at {sizes}"
        return
    assert accepted or sizes != SIZES["regular"], f"{spec} accepted"

    assert isinstance(res, FeArray) or (isinstance(res, float) and shape == t == ())
    assert np.shape(res) in [(), (Ne, nPg, *t)]
    if meaning is not None:
        v = np.asarray(value)
        v = v[:, None] if meaning == "e" else v[None] if meaning == "pg" else v
        expected = np.broadcast_to(v, (Ne, nPg, *t))
        assert np.array_equal(np.broadcast_to(np.asarray(res), (Ne, nPg, *t)), expected)


def test_constants_are_read_only_views():
    C = np.eye(3)
    C_e_pg = FeArray.broadcast(C, 5, 4, tensor_shape=(3, 3))
    assert not C_e_pg.flags.writeable
    assert np.shares_memory(C_e_pg, C)


def test_tensor_shape_is_mandatory():
    with pytest.raises(TypeError):
        FeArray.broadcast(np.ones(5), 5, 4)  # type: ignore [call-arg]
