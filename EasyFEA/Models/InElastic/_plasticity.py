# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Protocol, Sequence

import numpy as np

from ..Elastic._laws import _Elastic
from .Contract import (
    ZERO_SCALAR,
    ZERO_TENSOR,
    _Behavior,
    Newton,
    Trace,
    Von_Mises_stress,
)

if TYPE_CHECKING:
    from jax import Array

# ----------------------------------------------
# Yield surfaces f(sig, R): negative is elastic
# ----------------------------------------------


class Surface(Protocol):
    """A yield function ``f(sig, R)`` on a (6,) Kelvin stress, with the stress that scales it."""

    sigma_y: float

    def __call__(self, sig: "Array", R: "Array") -> "Array": ...


@dataclass(frozen=True)
class VonMises:
    r""":math:`f = \sigma_{eq} - \sigma_y - R`."""

    sigma_y: float

    def __post_init__(self):
        assert self.sigma_y > 0, "sigma_y must be > 0"

    def __call__(self, sig: "Array", R: "Array") -> "Array":
        return Von_Mises_stress(sig) - self.sigma_y - R


@dataclass(frozen=True)
class Hill:
    r""":math:`f = \sqrt{\Sig : \Prm : \Sig} - \sigma_y - R`, Hill 1948; the defaults are von Mises."""

    sigma_y: float
    F: float = 0.5
    G: float = 0.5
    H: float = 0.5
    L: float = 1.5
    M: float = 1.5
    N: float = 1.5

    def __post_init__(self):
        assert self.sigma_y > 0, "sigma_y must be > 0"

    @property
    def P(self) -> np.ndarray:
        """(6, 6) Kelvin form; the sqrt(2) on the shear entries turns Hill's 2L syz^2 into L syz_kelvin^2."""
        F, G, H = self.F, self.G, self.H
        P = np.diag([0.0, 0.0, 0.0, self.L, self.M, self.N])
        P[:3, :3] = [[G + H, -H, -G], [-H, F + H, -F], [-G, -F, F + G]]
        return P

    def __call__(self, sig: "Array", R: "Array") -> "Array":
        import jax.numpy as jnp

        # finite at sig = 0, so that it stays differentiable
        return jnp.sqrt(sig @ self.P @ sig + 1e-300) - self.sigma_y - R


@dataclass(frozen=True)
class DruckerPrager:
    r""":math:`f = \sigma_{eq} + \eta\,\tr\Sig - \sigma_y - R`; associated, so the flow is dilatant."""

    sigma_y: float
    eta: float

    def __post_init__(self):
        assert self.sigma_y > 0, "sigma_y must be > 0"

    def __call__(self, sig: "Array", R: "Array") -> "Array":
        return Von_Mises_stress(sig) + self.eta * Trace(sig) - self.sigma_y - R


# ----------------------------------------------
# Isotropic hardening R(p), with R(0) = 0
# ----------------------------------------------


@dataclass(frozen=True)
class Perfect:
    """:math:`R = 0`."""

    def __call__(self, p: "Array") -> "Array":
        return 0.0 * p


@dataclass(frozen=True)
class Linear:
    """:math:`R = H p`."""

    H: float

    def __post_init__(self):
        assert self.H >= 0, "H must be >= 0"

    def __call__(self, p: "Array") -> "Array":
        return self.H * p


@dataclass(frozen=True)
class Voce:
    """:math:`R = Q (1 - e^{-b p})`, saturating at ``Q``."""

    Q: float
    b: float

    def __post_init__(self):
        assert self.Q >= 0 and self.b > 0, "Q must be >= 0 and b > 0"

    def __call__(self, p: "Array") -> "Array":
        import jax.numpy as jnp

        return self.Q * (1 - jnp.exp(-self.b * p))


@dataclass(frozen=True)
class Swift:
    r""":math:`R = K(\varepsilon_0 + p)^n - K\varepsilon_0^n`; ``eps0`` keeps the slope finite at the origin."""

    K: float
    n: float
    eps0: float = 1e-4

    def __post_init__(self):
        assert (
            self.K > 0 and 0 < self.n < 1 and self.eps0 > 0
        ), "need K > 0, 0 < n < 1, eps0 > 0"

    def __call__(self, p: "Array") -> "Array":
        return self.K * ((self.eps0 + p) ** self.n - self.eps0**self.n)


# ----------------------------------------------
# Behaviors
# ----------------------------------------------


class Plasticity(_Behavior):
    r"""Associated plasticity: any elasticity, any surface :math:`f(\Sig, R)`, any isotropic hardening :math:`R(p)`; backward Euler, returned onto the surface by :func:`Newton`."""

    class State(NamedTuple):
        eps_p: "Array" = ZERO_TENSOR
        """plastic strain"""
        p: "Array" = ZERO_SCALAR
        """accumulated plastic strain"""

    def __init__(
        self,
        elastic: _Elastic,
        surface: Surface,
        hardening=Perfect(),
        dim: int = 3,
        planeStress: bool = False,
        thickness: float = 1.0,
    ):
        """``elastic`` must be 3D."""
        assert isinstance(elastic, _Elastic), "elastic must be an elastic model"
        assert elastic.dim == 3, "the elastic model must be 3D"
        super().__init__(dim, planeStress, thickness)

        # numpy, so that building a behavior needs no jax
        self.C: Array = np.asarray(elastic.C, dtype=float)  # type: ignore[assignment]
        self.surface = surface
        self.hardening = hardening

    def Stress(
        self,
        eps: "Array",
        eps_p: "Array",
    ) -> "Array":
        return self.C @ (eps - eps_p)

    def Update(
        self,
        eps: "Array",
        z: State,
        dt: float,
        **external,
    ) -> tuple["Array", State]:
        import jax

        f, R = self.surface, self.hardening
        # a point on the surface, as every plastic point is at the start of a step, flows, so that it gets the loading tangent whatever the roundoff in f
        flows = f(self.Stress(eps, z.eps_p), R(z.p)) > -self._tol * f.sigma_y

        def Residual(dz: Plasticity.State) -> Plasticity.State:
            sig = self.Stress(eps, z.eps_p + dz.eps_p)
            R_new = R(z.p + dz.p)
            N = jax.grad(f)(sig, R_new)
            return Plasticity.State(
                eps_p=dz.eps_p - dz.p * N,
                p=f(sig, R_new) / f.sigma_y,
            )

        # the unknown is the increment of the state; where nothing flows it stays zero
        dz = Newton(
            Residual,
            Plasticity.State(),
            flows,
            tol=self._tol,
            maxIter=self._maxIter,
        )
        new = Plasticity.State(
            eps_p=z.eps_p + dz.eps_p,
            p=z.p + dz.p,
        )
        return self.Stress(eps, new.eps_p), new


class Norton(Plasticity):
    r"""Plasticity with a Norton flow rate :math:`\dot p = A \langle f/\sigma_0 \rangle^n`: past the surface ``f`` no longer vanishes, it drives the flow."""

    def __init__(
        self,
        elastic: _Elastic,
        surface: Surface,
        hardening=Perfect(),
        A: float = 1.0,
        n: float = 1.0,
        sigma_0: float = 1.0,
        dim: int = 3,
        planeStress: bool = False,
        thickness: float = 1.0,
    ):
        """Large ``A`` approaches rate-independent plasticity."""
        assert A > 0 and n > 0 and sigma_0 > 0, "need A > 0, n > 0, sigma_0 > 0"
        super().__init__(
            elastic,
            surface,
            hardening,
            dim,
            planeStress,
            thickness,
        )
        self.A = A
        self.n = n
        self.sigma_0 = sigma_0

    def Integrate(
        self,
        eps_e_pg,
        z_e_pg=None,
        dt=0.0,
        **external,
    ):
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

    def Update(
        self,
        eps: "Array",
        z: Plasticity.State,
        dt: float,
        **external,
    ) -> tuple["Array", Plasticity.State]:
        import jax
        import jax.numpy as jnp

        f, R = self.surface, self.hardening
        trial = self.Stress(eps, z.eps_p)
        f_trial = f(trial, R(z.p))
        # strict: Creep has an infinite slope at dp = 0
        flows = f_trial > 0

        def Residual(dz: Plasticity.State) -> Plasticity.State:
            sig = self.Stress(eps, z.eps_p + dz.eps_p)
            R_new = R(z.p + dz.p)
            N = jax.grad(f)(sig, R_new)
            over = f(sig, R_new) - self.Creep(dz.p, dt)
            return Plasticity.State(
                eps_p=dz.eps_p - dz.p * N,
                p=over / f.sigma_y,
            )

        # one explicit step as the first guess, capped by the rate-independent return
        N = jax.grad(f)(trial, R(z.p))
        overstress = jnp.maximum(f_trial, 0.0)
        explicit = dt * self.A * (overstress / self.sigma_0) ** self.n
        capped = overstress / (N @ self.C @ N + jax.grad(R)(z.p))
        dp = jnp.where(flows, jnp.minimum(explicit, capped), 0.0)

        dz = Newton(
            Residual,
            Plasticity.State(eps_p=dp * N, p=dp),
            flows,
            tol=self._tol,
            maxIter=self._maxIter,
        )
        new = Plasticity.State(
            eps_p=z.eps_p + dz.eps_p,
            p=z.p + dz.p,
        )
        return self.Stress(eps, new.eps_p), new


class Chaboche(_Behavior):
    r"""Plasticity with Armstrong-Frederick back-stresses :math:`X = \sum_i \tfrac23 C_i \bm{\alpha}_i`, the surface read at :math:`\Sig - X`; :math:`\gamma_i = 0` is linear (Prager) hardening."""

    class State(NamedTuple):
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
        hardening=Perfect(),
        dim: int = 3,
        planeStress: bool = False,
        thickness: float = 1.0,
    ):
        """One ``C_X`` and one ``gamma`` per back-stress; ``elastic`` must be 3D."""
        C_arr = np.atleast_1d(np.asarray(C_X, dtype=float))
        gamma_arr = np.atleast_1d(np.asarray(gamma, dtype=float))
        assert isinstance(elastic, _Elastic), "elastic must be an elastic model"
        assert elastic.dim == 3, "the elastic model must be 3D"
        assert C_arr.shape == gamma_arr.shape, "one C_X and one gamma per back-stress"
        assert np.all(C_arr > 0) and np.all(gamma_arr >= 0), "need C_X > 0, gamma >= 0"
        super().__init__(dim, planeStress, thickness)

        self.C: Array = np.asarray(elastic.C, dtype=float)  # type: ignore[assignment]
        self.surface = surface
        self.hardening = hardening
        self.C_X = C_arr
        self.gamma = gamma_arr

    def Virgin_state(self) -> "Chaboche.State":
        return Chaboche.State(alpha=np.zeros((self.C_X.size, 6)))  # type: ignore[arg-type]

    def Shifted_stress(
        self,
        eps: "Array",
        eps_p: "Array",
        alpha: "Array",
    ) -> "Array":
        """The stress the surface reads, :math:`\\Sig - X`."""
        return self.C @ (eps - eps_p) - 2 / 3 * self.C_X @ alpha

    def Update(
        self,
        eps: "Array",
        z: State,
        dt: float,
        **external,
    ) -> tuple["Array", State]:
        import jax

        f, R = self.surface, self.hardening
        xi = self.Shifted_stress(eps, z.eps_p, z.alpha)
        # on the surface flows, as in Plasticity
        flows = f(xi, R(z.p)) > -self._tol * f.sigma_y

        def Residual(dz: Chaboche.State) -> Chaboche.State:
            alpha = z.alpha + dz.alpha
            xi = self.Shifted_stress(eps, z.eps_p + dz.eps_p, alpha)
            R_new = R(z.p + dz.p)
            N = jax.grad(f)(xi, R_new)
            return Chaboche.State(
                eps_p=dz.eps_p - dz.p * N,
                p=f(xi, R_new) / f.sigma_y,
                alpha=dz.alpha - dz.p * (N - self.gamma[:, None] * alpha),
            )

        dz = Newton(
            Residual,
            Chaboche.State(alpha=np.zeros((self.C_X.size, 6))),  # type: ignore[arg-type]
            flows,
            tol=self._tol,
            maxIter=self._maxIter,
        )
        new = Chaboche.State(
            eps_p=z.eps_p + dz.eps_p, p=z.p + dz.p, alpha=z.alpha + dz.alpha
        )
        return self.C @ (eps - new.eps_p), new
