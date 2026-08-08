# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""Viscoelastic relaxation, as a generalized Maxwell chain.

Each branch is a spring in series with a dashpot, in parallel with the equilibrium spring:

.. math::
    \psi = \tfrac12 (1 - \textstyle\sum_i g_i)\, \Eps^e : \Crm : \Eps^e
         + \sum_i \tfrac12 g_i\, (\Eps^e - \Eps^v_i) : \Crm : (\Eps^e - \Eps^v_i)

so the stress is :math:`\Sig = \Crm : \Eps^e - \sum_i g_i \Crm : \Eps^v_i`, and each branch
relaxes towards the current strain with its own time constant,
:math:`\dot{\Eps^v_i} = (\Eps^e - \Eps^v_i)/\tau_i`.

The instantaneous (glassy) response is the full ``C``; after all branches have relaxed only
:math:`(1 - \sum g_i)\Crm` is left, so the fractions must sum to less than one.

Unlike plasticity, this evolution is linear — it needs no yield surface, and the local Newton
converges in a single iteration.
"""

from typing import NamedTuple


class Maxwell(NamedTuple):
    r"""One Maxwell branch.

    Parameters
    ----------
    g : float
        fraction of ``C`` carried by this branch, in ``(0, 1)``
    tau : float
        relaxation time
    """

    g: float
    tau: float
