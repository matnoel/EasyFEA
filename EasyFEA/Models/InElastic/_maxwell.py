# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import TYPE_CHECKING, NamedTuple, Sequence

import numpy as np

from ..Elastic._laws import _Elastic
from ...Utilities import _params
from ._behavior import ZERO_TENSOR, _Behavior

if TYPE_CHECKING:
    from jax import Array


class Maxwell(_Behavior):
    r"""Generalized Maxwell viscoelasticity: branches carrying :math:`g_i \Crm`, each relaxing with :math:`\tau_i`, beside the spring :math:`(1 - \sum_i g_i)\Crm`.

    Backward Euler in closed form: :math:`\Eps^v_i = (\Eps^v_{i,n} + k_i \Eps) / (1 + k_i)` with :math:`k_i = \dt/\tau_i`, and :math:`\Sig = \Crm : (\Eps - \sum_i g_i \Eps^v_i)`.
    """

    class Internals(NamedTuple):
        eps_v: "Array" = ZERO_TENSOR[None]
        """one branch strain per row, (n_branches, 6)"""

    g: np.ndarray = _params.StrictlyPositiveParameter()
    tau: np.ndarray = _params.StrictlyPositiveParameter()

    def __init__(
        self,
        elastic: _Elastic,
        g: float | Sequence[float],
        tau: float | Sequence[float],
    ):
        """One ``g`` and one ``tau`` per branch."""
        g_arr = np.atleast_1d(np.asarray(g, dtype=float))
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
        assert g_arr.shape == tau_arr.shape, "one g and one tau per branch"
        assert (
            g_arr.sum() < 1.0
        ), "the branch stiffness fractions must sum to less than 1"
        super().__init__(elastic)
        self.g = g_arr
        self.tau = tau_arr

    def Virgin_internals(self) -> "Maxwell.Internals":
        return Maxwell.Internals(eps_v=np.zeros((self.g.size, 6)))  # type: ignore[arg-type]

    def Update(
        self,
        eps: "Array",
        z: "Maxwell.Internals",
        dt: float,
        **external,
    ) -> tuple["Array", "Maxwell.Internals"]:
        k = (dt / self.tau)[:, None]
        new = Maxwell.Internals(eps_v=(z.eps_v + k * eps) / (1 + k))
        return self.Stress(eps, new), new

    def Stress(
        self,
        eps: "Array",
        z: "Maxwell.Internals",
        **external,
    ) -> "Array":
        return self.C @ (eps - self.g @ z.eps_v)
