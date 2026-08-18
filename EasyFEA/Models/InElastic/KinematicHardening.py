# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""Kinematic hardening — stored energy that moves the centre of the yield surface.

Isotropic hardening (see :mod:`.IsotropicHardening`) grows the surface; kinematic hardening translates
it by a back-stress :math:`X`, so the surface is read at :math:`\Sig - X`:

.. math::
    \psi_{kin} = \tfrac13 C\, \bm{\alpha} : \bm{\alpha}
    \qquad
    X = \dpartial{\psi_{kin}}{\bm{\alpha}} = \tfrac23 C \bm{\alpha}

That is the Bauschinger effect: yielding in tension lowers the stress at which the material
yields in compression. It is why cyclic plasticity needs this and not isotropic hardening alone.

The internal variable is a **tensor**, which is why this is a separate piece rather than an
option on :class:`~EasyFEA.Models.InElastic.IsotropicHardening.Hardening`.
"""

from typing import Callable, NamedTuple

from ...FEM._linalg import FeArray


class KinematicHardening(NamedTuple):
    r"""Back-stress and how it saturates.

    - ``psi(alpha) -> (Ne, nPg)`` — the stored energy.
    - ``X(alpha) -> (Ne, nPg, 6)`` — the back-stress, :math:`\dpartial{\psi}{\bm{\alpha}}`.
    - ``modulus`` — :math:`\dpartial{X}{\bm{\alpha}}`, constant because ``X`` is linear in ``alpha``.
    - ``recall`` — the Armstrong-Frederick recall coefficient; ``0`` is linear (Prager)
      hardening, which never saturates.
    """

    psi: Callable[[FeArray.FeArrayALike], FeArray.FeArrayALike]
    X: Callable[[FeArray.FeArrayALike], FeArray.FeArrayALike]
    modulus: float
    recall: float


def ArmstrongFrederick(C: float, gamma: float = 0.0) -> KinematicHardening:
    r"""Armstrong-Frederick back-stress, :math:`\dot{\bm{\alpha}} = \dot\gamma\,(N - \gamma\bm{\alpha})`.

    The recall term makes ``X`` saturate at ``2C/(3·gamma)``, which is what bounds a stabilised
    cycle. Without it the back-stress grows without limit.

    Parameters
    ----------
    C : float
        kinematic hardening modulus
    gamma : float, optional
        recall coefficient, by default 0.0 (linear Prager hardening)
    """
    assert C > 0 and gamma >= 0, "need C > 0 and gamma >= 0"
    modulus = 2.0 / 3.0 * C

    return KinematicHardening(
        lambda alpha: (C / 3.0) * alpha.dot(alpha),
        lambda alpha: modulus * alpha,
        modulus,
        gamma,
    )


def Prager(C: float) -> KinematicHardening:
    r"""Linear kinematic hardening, :math:`X = \tfrac23 C \Eps^p` — ArmstrongFrederick with no recall."""
    return ArmstrongFrederick(C, 0.0)


def Chaboche(*components: tuple[float, float]) -> tuple[KinematicHardening, ...]:
    r"""Chaboche superposition: :math:`X = \sum_i X_i`, each an Armstrong-Frederick back-stress.

    One Armstrong-Frederick term is a single exponential, so it cannot follow both the sharp knee
    just after yield and the long, nearly linear tail of a measured hysteresis loop. A fitted
    cyclic law therefore carries two or three components with very different
    :math:`(C_i, \gamma_i)`. A component with :math:`\gamma_i = 0` contributes a purely linear
    term, and is what keeps ratcheting under control — a single saturating term over-predicts it
    badly.

    Parameters
    ----------
    *components : tuple[float, float]
        one ``(C, gamma)`` pair per back-stress

    Examples
    --------
    Three components, fast to slow::

        Chaboche((60000.0, 500.0), (20000.0, 100.0), (2000.0, 0.0))
    """
    assert components, "Chaboche needs at least one (C, gamma) component"
    return tuple(ArmstrongFrederick(C, gamma) for C, gamma in components)
