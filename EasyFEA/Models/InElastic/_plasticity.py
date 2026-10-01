# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import TYPE_CHECKING, Any, NamedTuple, Sequence

import numpy as np

from ..Elastic._laws import _Elastic
from ...FEM._linalg import FeArray
from ._behavior import ZERO_SCALAR, ZERO_TENSOR, _Behavior, Newton
from .IsotropicHardening import Hardening, Perfect
from .Yield import Surface

if TYPE_CHECKING:
    from jax import Array


def _Add(z, dz):
    """The internal state after the increment ``dz``."""
    import jax

    return jax.tree_util.tree_map(lambda a, b: a + b, z, dz)


class _Plastic(_Behavior):
    """Elasticity, a surface and an isotropic hardening: what Plasticity and Chaboche share."""

    def __init__(
        self,
        elastic: _Elastic,
        surface: Surface,
        hardening: Hardening = Perfect(),
    ):
        super().__init__(elastic)
        self.surface = surface
        self.hardening = hardening

    def Stress(
        self,
        eps: "Array",
        z: Any,
        **external,
    ) -> "Array":
        """``z`` carries ``eps_p``."""
        return self.C @ (eps - z.eps_p)

    def _Flows(self, f_trial: "Array") -> "Array":
        """A point on the surface, as every plastic point is at the start of a step, flows, so that it gets the loading tangent whatever the roundoff in f."""
        return f_trial > -self._tol * self.surface.sigma_y


class Plasticity(_Plastic):
    r"""Associated plasticity: any elasticity, any surface :math:`f(\Sig, R)`, any isotropic hardening :math:`R(p)`; backward Euler, returned onto the surface by :func:`Newton`."""

    class Internals(NamedTuple):
        eps_p: "Array" = ZERO_TENSOR
        """plastic strain"""
        p: "Array" = ZERO_SCALAR
        """accumulated plastic strain"""

    def _Overstress(self, f: "Array", dp: "Array", dt: float) -> "Array":
        """What the return drives to zero."""
        return f

    def _First_guess(
        self,
        trial: "Array",
        f_trial: "Array",
        z: "Plasticity.Internals",
        dt: float,
        flows: "Array",
    ) -> "Plasticity.Internals":
        return Plasticity.Internals()

    def Update(
        self,
        eps: "Array",
        z: "Plasticity.Internals",
        dt: float,
        **external,
    ) -> tuple["Array", "Plasticity.Internals"]:
        import jax

        f, R = self.surface, self.hardening
        trial = self.Stress(eps, z)
        f_trial = f(trial, R(z.p))
        flows = self._Flows(f_trial)

        def Residual(dz: "Plasticity.Internals") -> "Plasticity.Internals":
            new = _Add(z, dz)
            sig = self.Stress(eps, new)
            R_new = R(new.p)
            N = jax.grad(f)(sig, R_new)
            return Plasticity.Internals(
                eps_p=dz.eps_p - dz.p * N,
                p=self._Overstress(f(sig, R_new), dz.p, dt) / f.sigma_y,
            )

        # the unknown is the increment of the internal state; where nothing flows it stays zero
        dz = Newton(
            Residual,
            self._First_guess(trial, f_trial, z, dt, flows),
            flows,
            tol=self._tol,
            maxIter=self._maxIter,
        )
        new = _Add(z, dz)
        return self.Stress(eps, new), new


class Norton(Plasticity):
    r"""Plasticity with a Norton flow rate :math:`\dot p = A \langle f/\sigma_0 \rangle^n`: past the surface ``f`` no longer vanishes, it drives the flow."""

    def __init__(
        self,
        elastic: _Elastic,
        surface: Surface,
        hardening: Hardening = Perfect(),
        A: float = 1.0,
        n: float = 1.0,
        sigma_0: float = 1.0,
    ):
        """Large ``A`` approaches rate-independent plasticity."""
        assert A > 0 and n > 0 and sigma_0 > 0, "need A > 0, n > 0, sigma_0 > 0"
        super().__init__(
            elastic,
            surface,
            hardening,
        )
        self.A = A
        self.n = n
        self.sigma_0 = sigma_0

    def Integrate(
        self,
        eps_e_pg: FeArray.FeArrayALike,
        z_e_pg: dict[str, FeArray] | None = None,
        dt: float = 0.0,
        **external,
    ) -> tuple[FeArray, FeArray, dict[str, FeArray]]:
        assert dt > 0.0, (
            "a rate-dependent behavior needs a positive time increment; "
            "set `simu.dt` or pass `dt=` to Integrate"
        )
        return super().Integrate(
            eps_e_pg,
            z_e_pg,
            dt,
            **external,
        )

    def Creep(self, dp: "Array", dt: float) -> "Array":
        """The overstress ``f`` sustaining the rate ``dp / dt``."""
        import jax.numpy as jnp

        rate = jnp.maximum(dp, 1e-300) / dt
        return self.sigma_0 * (rate / self.A) ** (1 / self.n)

    def _Flows(self, f_trial: "Array") -> "Array":
        """Strict, unlike :class:`Plasticity`: Creep has an infinite slope at dp = 0."""
        return f_trial > 0

    def _Overstress(self, f: "Array", dp: "Array", dt: float) -> "Array":
        return f - self.Creep(dp, dt)

    def _First_guess(
        self,
        trial: "Array",
        f_trial: "Array",
        z: "Plasticity.Internals",
        dt: float,
        flows: "Array",
    ) -> "Plasticity.Internals":
        """One explicit step, capped by the rate-independent return."""
        import jax
        import jax.numpy as jnp

        f, R = self.surface, self.hardening
        N = jax.grad(f)(trial, R(z.p))
        overstress = jnp.maximum(f_trial, 0.0)
        explicit = dt * self.A * (overstress / self.sigma_0) ** self.n
        capped = overstress / (N @ self.C @ N + jax.grad(R)(z.p))
        dp = jnp.where(flows, jnp.minimum(explicit, capped), 0.0)
        return Plasticity.Internals(eps_p=dp * N, p=dp)


class Chaboche(_Plastic):
    r"""Plasticity with Armstrong-Frederick back-stresses :math:`X = \sum_i \tfrac23 C_i \bm{\alpha}_i`, the surface read at :math:`\Sig - X`; :math:`\gamma_i = 0` is linear (Prager) hardening."""

    class Internals(NamedTuple):
        eps_p: "Array" = ZERO_TENSOR
        """plastic strain"""
        p: "Array" = ZERO_SCALAR
        """accumulated plastic strain"""
        alpha: "Array" = ZERO_TENSOR[None]
        """one kinematic variable per row, (n_components, 6)"""

    def __init__(
        self,
        elastic: _Elastic,
        surface: Surface,
        C_X: float | Sequence[float],
        gamma: float | Sequence[float],
        hardening: Hardening = Perfect(),
    ):
        """One ``C_X`` and one ``gamma`` per back-stress."""
        C_arr = np.atleast_1d(np.asarray(C_X, dtype=float))
        gamma_arr = np.atleast_1d(np.asarray(gamma, dtype=float))
        assert C_arr.shape == gamma_arr.shape, "one C_X and one gamma per back-stress"
        assert np.all(C_arr > 0) and np.all(gamma_arr >= 0), "need C_X > 0, gamma >= 0"
        super().__init__(
            elastic,
            surface,
            hardening,
        )
        self.C_X = C_arr
        self.gamma = gamma_arr

    def Virgin_internals(self) -> "Chaboche.Internals":
        return Chaboche.Internals(alpha=np.zeros((self.C_X.size, 6)))  # type: ignore[arg-type]

    def Shifted_stress(
        self,
        eps: "Array",
        z: "Chaboche.Internals",
    ) -> "Array":
        """The stress the surface reads, :math:`\\Sig - X`."""
        return self.Stress(eps, z) - 2 / 3 * self.C_X @ z.alpha

    def Update(
        self,
        eps: "Array",
        z: "Chaboche.Internals",
        dt: float,
        **external,
    ) -> tuple["Array", "Chaboche.Internals"]:
        import jax

        f, R = self.surface, self.hardening
        flows = self._Flows(f(self.Shifted_stress(eps, z), R(z.p)))

        def Residual(dz: "Chaboche.Internals") -> "Chaboche.Internals":
            new = _Add(z, dz)
            xi = self.Shifted_stress(eps, new)
            R_new = R(new.p)
            N = jax.grad(f)(xi, R_new)
            return Chaboche.Internals(
                eps_p=dz.eps_p - dz.p * N,
                p=f(xi, R_new) / f.sigma_y,
                alpha=dz.alpha - dz.p * (N - self.gamma[:, None] * new.alpha),
            )

        dz = Newton(
            Residual,
            self.Virgin_internals(),
            flows,
            tol=self._tol,
            maxIter=self._maxIter,
        )
        new = _Add(z, dz)
        return self.Stress(eps, new), new
