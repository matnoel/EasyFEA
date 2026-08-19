# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""jax plumbing for constitutive kernels: float64, the ``(Ne, nPg)`` lift, the Kelvin boundary.

Optional (``pip install easyfea[jax]``). Importing this module imports jax, so import it inside a function on the eager path.
"""

from typing import Callable, Union

import numpy as np

from ..FEM._linalg import FeArray
from ..Utilities._requires import Create_requires_decorator

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass

requires_jax = Create_requires_decorator("jax")

_SQ2 = 1.0 / np.sqrt(2.0)

_KELVIN_BASIS = np.array(
    [
        [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
        [[0, 0, 0], [0, 0, _SQ2], [0, _SQ2, 0]],
        [[0, 0, _SQ2], [0, 0, 0], [_SQ2, 0, 0]],
        [[0, _SQ2, 0], [_SQ2, 0, 0], [0, 0, 0]],
    ]
)
"""The six Kelvin basis tensors, ordered ``[xx, yy, zz, yz, xz, xy]``."""


@requires_jax
def Enable_x64() -> None:
    """Switches jax to float64. Call it before building any kernel: the default is float32."""
    jax.config.update("jax_enable_x64", True)


@requires_jax
def Kelvin_to_tensor(vec):
    """Kelvin ``(6,)`` vector → symmetric ``(3, 3)`` tensor, at one point.

    Differentiating through this gives a ``(6, 6)`` hessian already in Kelvin notation, not a ``(3, 3, 3, 3)`` to project.
    """
    return jnp.einsum("I,Iij->ij", vec, _KELVIN_BASIS)


@requires_jax
def Vmap_e_pg(kernel: Callable, in_axes: Union[int, tuple] = 0) -> Callable:
    """Lifts a one-point kernel to ``(Ne, nPg, ...)`` fields, returning a :class:`.FeArray`.

    Parameters
    ----------
    kernel : Callable
        function of one or more single-point arrays
    in_axes : int | tuple, optional
        as :func:`jax.vmap` reads it, applied to the element and the Gauss-point axis
    """
    mapped = jax.jit(jax.vmap(jax.vmap(kernel, in_axes=in_axes), in_axes=in_axes))

    def field(*args) -> FeArray.FeArrayALike:
        out = mapped(*(jnp.asarray(arg) for arg in args))
        return FeArray.asfearray(np.asarray(out))

    return field
