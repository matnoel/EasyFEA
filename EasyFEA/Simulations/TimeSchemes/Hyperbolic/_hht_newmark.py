# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from .._utils import _Interval
from ._hyperbolic import _Hyperbolic
from ._newmark import Newmark


class HHTNewmark(_Hyperbolic):
    r"""Hilber–Hughes–Taylor–Newmark method (Doyen, Ern & Piperno, SIAM J. Sci. Comput. 33(1), 2011): only the displacement is shifted, :math:`\urm_t = (1-\alpha)\urm^{n+1} + \alpha\urm^n`, v and a stay at n+1. :math:`\beta = (1+\alpha)^2/4`, :math:`\gamma = 1/2+\alpha` are derived from :math:`\alpha \in [0, 1/3]` (Doyen's own α, negated), the pair their stability proof requires; 0 is Newmark."""

    alpha: float = _Interval(
        0.0,
        1 / 3,
        closedSup=True,
        message="HHTNewmark requires alpha in [0, 1/3], the range of Doyen et al. 2011's stability and accuracy proof",
    )

    def __init__(self, dt: float, alpha: float = 1 / 6):
        super().__init__(dt)
        self.alpha = alpha

    @property
    def beta(self) -> float:
        return 1 / 4 * (1 + self.alpha) ** 2

    @property
    def gamma(self) -> float:
        return 1 / 2 + self.alpha

    def Evaluate(self, x, u_n, v_n, a_n):
        u_np1, v_np1, a_np1 = Newmark.Evaluate_np1(self, x, u_n, v_n, a_n)  # type: ignore [arg-type]
        return (1 - self.alpha) * u_np1 + self.alpha * u_n, v_np1, a_np1

    def Evaluate_np1(self, x, u_n, v_n, a_n):
        return Newmark.Evaluate_np1(self, x, u_n, v_n, a_n)  # type: ignore [arg-type]
