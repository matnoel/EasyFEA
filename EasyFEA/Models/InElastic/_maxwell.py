# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import TYPE_CHECKING, NamedTuple, Sequence

import numpy as np

from ..Elastic._laws import _Elastic
from .Contract import ZERO_TENSOR, _Behavior

if TYPE_CHECKING:
    from jax import Array


class Maxwell(_Behavior):
    r"""Generalized Maxwell viscoelasticity: branches carrying :math:`g_i \Crm`, each relaxing with :math:`\tau_i`, beside the spring :math:`(1 - \sum_i g_i)\Crm`.

    Backward Euler in closed form: :math:`\Eps^v_i = (\Eps^v_{i,n} + k_i \Eps) / (1 + k_i)` with :math:`k_i = \dt/\tau_i`, and :math:`\Sig = \Crm : (\Eps - \sum_i g_i \Eps^v_i)`.
    """

    class State(NamedTuple):
        eps_v: "Array" = ZERO_TENSOR[None]
        """one branch strain per row, (n_branches, 6)"""

    def __init__(
        self,
        elastic: _Elastic,
        g: float | Sequence[float],
        tau: float | Sequence[float],
        dim: int = 3,
        planeStress: bool = False,
        thickness: float = 1.0,
    ):
        """One ``g`` and one ``tau`` per branch; ``elastic`` must be 3D."""
        g_arr = np.atleast_1d(np.asarray(g, dtype=float))
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
        assert isinstance(elastic, _Elastic), "elastic must be an elastic model"
        assert elastic.dim == 3, "the elastic model must be 3D"
        assert g_arr.shape == tau_arr.shape, "one g and one tau per branch"
        assert np.all(g_arr > 0), "every branch g must be > 0"
        assert np.all(tau_arr > 0), "every branch tau must be > 0"
        assert (
            g_arr.sum() < 1.0
        ), "the branch stiffness fractions must sum to less than 1"
        super().__init__(dim, planeStress, thickness)

        # numpy, so that building a behavior needs no jax
        self.C: Array = np.asarray(elastic.C, dtype=float)  # type: ignore[assignment]
        self.g = g_arr
        self.tau = tau_arr

    def Virgin_state(self) -> "Maxwell.State":
        return Maxwell.State(eps_v=np.zeros((self.g.size, 6)))  # type: ignore[arg-type]

    def Update(
        self,
        eps: "Array",
        z: State,
        dt: float,
        **external,
    ) -> tuple["Array", State]:
        k = (dt / self.tau)[:, None]
        eps_v = (z.eps_v + k * eps) / (1 + k)
        return self.C @ (eps - self.g @ eps_v), Maxwell.State(eps_v=eps_v)
