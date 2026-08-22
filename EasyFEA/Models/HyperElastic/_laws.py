# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Hyperelastic laws."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Union

# utilities
from ...FEM import FeArray, TensorProd, Normalize
from ._state import HyperElasticState

# others
from .._utils import _IModel, Project_matrix_to_vector
from ...Utilities import _params, _types

# ----------------------------------------------
# Hyper Elastic
# ----------------------------------------------


class _HyperElastic(_IModel, ABC):
    """HyperElastic material.\n
    `NeoHookean`, `MooneyRivlin`, `SaintVenantKirchhoff` and `HolzapfelOgden` inherit from `_HyperElas` class.
    """

    dim: float = _params.ParameterInValues([1, 2, 3])

    thickness: float = _params.PositiveScalarParameter()

    eta: float = _params.PositiveScalarParameter()
    """Kelvin–Voigt viscosity. ``0`` (default) → purely elastic
    When ``> 0`` and a velocity field is available, the viscous contribution is delivered via the damping matrix :func:`Operators.NonLinear.KelvinVoigtDamping`.
    The simulation handles both the residual contribution (``b -= C @ v_t``) and the linear tangent piece (``coefC · C`` in the global assembly) — same uniform pattern as Rayleigh damping in :class:`Elastic`."""

    active_stress: Union[float, _types.FloatArray] = _params.ScalarOrFieldParameter()
    """Active stress magnitude. ``0`` (default) → inactive.
    When non-zero and the direction tensor has been registered via :meth:`Set_active_stress_vec`, the PK2 contribution ``active_stress · (T̂ ⊗ T̂)`` is delivered by :func:`Operators.NonLinear.ActiveStressTensor` — **not** by :meth:`Compute_dWde`, which stays the derivative of :meth:`Compute_W` (see :meth:`Compute_active_stress`).
    Typical cardiac use: precompute the fiber direction tensor once with :meth:`Set_active_stress_vec`, then update only this scalar between :meth:`Solve` calls — ``material.active_stress = float(tau_values[i])``."""

    def __init__(self, dim: int, thickness: float):
        self.dim = dim
        self.thickness = thickness
        self.eta = 0.0
        self.active_stress = 0.0
        self.__TxT = None

    def Set_active_stress_vec(self, T) -> None:
        r"""Registers the **direction** of the active PK2 contribution.

        Normalises ``T`` per Gauss point and precomputes the Kelvin-Mandel vector form of ``T̂ ⊗ T̂``, which :meth:`Compute_active_stress` then scales by :attr:`active_stress`. Only that scalar is updated between steps; the direction is constant across a run, so call this **once** during setup.

        Note: with a time scheme, set :attr:`active_stress` at the scheme's effective time (midpoint ``t + dt/2``, HHT ``t + (1−α)·dt``, Newmark/Euler ``t + dt``), not the endpoint ``t``.

        Parameters
        ----------
        T
            Direction tensor at every Gauss point, shape ``(Ne, nPg, 3)``. Need not be unit-norm — divided by ``|T|`` per Gauss point internally.
        """
        # Per-Gauss-point unit normalisation (axis=-1 is the per-vector length).
        T_hat = Normalize(T)
        # The fiber pattern doesn't move with time, so this is precomputed
        # once and only `active_stress` is updated each step.
        self.__TxT = Project_matrix_to_vector(TensorProd(T_hat, T_hat))  # (Ne, nPg, 6)

    def Compute_active_stress(self, hyperElasticState: HyperElasticState) -> FeArray:
        r"""Active PK2 contribution ``τ · (T̂ ⊗ T̂)`` in Kelvin-Mandel vector form, shape ``(Ne, pg, d)`` with ``d = 1, 3, 6`` for a `1D`, `2D` or `3D` solution — same layout as :meth:`Compute_dWde`.

        Strain-independent, hence **not** derivable from :meth:`Compute_W`: it is a non-conservative stress, delivered by its own operator :func:`Operators.NonLinear.ActiveStressTensor` exactly as Kelvin–Voigt viscosity is delivered by :func:`Operators.NonLinear.KelvinVoigtDamping`. Folding it into :meth:`Compute_dWde` would break ``Compute_dWde == ∂(Compute_W)/∂e`` and silently corrupt every energy-based algorithm.

        The direction is registered in 3D once and for all, so the state — which knows the solution dimension — supplies the slice (in `2D` the out-of-plane fiber component drops out, as it does for the elastic stress under plane strain).
        """
        assert (
            self.__TxT is not None
        ), "active_stress is set but its direction is not — call Set_active_stress_vec(T) first."

        magnitude = self.active_stress

        if isinstance(magnitude, np.ndarray):
            if magnitude.ndim == 1:
                # one value per element, repeated over that element's integration points
                magnitude = magnitude[:, None]
            Ne, nPg = self.__TxT.shape[:2]
            assert magnitude.shape[0] == Ne and magnitude.shape[1] in (1, nPg), (
                f"active_stress is {magnitude.shape}, expected a scalar, ({Ne},) or ({Ne}, {nPg}) "
                "— the shapes registered by Set_active_stress_vec."
            )
            # a rank-0 field, so FeArray._align pads it to (Ne, nPg, 1) against __TxT's (Ne, nPg, 6)
            magnitude = FeArray.asfearray(magnitude)

        return hyperElasticState._Slice_Vector(magnitude * self.__TxT)

    @property
    def coef(self) -> float:
        """kelvin mandel coef -> sqrt(2)"""
        return np.sqrt(2)

    # Model
    @staticmethod
    def Available_Laws():
        laws = [NeoHookean, MooneyRivlin, SaintVenantKirchhoff]
        return laws

    @property
    def isHeterogeneous(self) -> bool:
        return False

    @abstractmethod
    def Compute_W(self, hyperElasticState: HyperElasticState) -> FeArray:
        """Computes the quadratic energy W(u).

        Parameters
        ----------
        hyperElasticState : HyperElasticState
            Hyperelastic state containing the mesh, the discretized field, and the matrix type.

        Returns
        -------
        FeArray
            We_e_pg of shape (Ne, pg)
        """

        return None  # type: ignore [return-value]

    @abstractmethod
    def Compute_dWde(self, hyperElasticState: HyperElasticState) -> FeArray:
        """Computes the second Piola-Kirchhoff tensor ``Σ = ∂W/∂e``.

        **Invariant**: this is exactly the derivative of :meth:`Compute_W`. Stress
        contributions that are not derivatives of ``W`` — Kelvin-Voigt viscosity
        ``η·ė``, active stress ``τ·(T̂⊗T̂)`` — must never be added here; they have
        their own operators (:func:`Operators.NonLinear.KelvinVoigtDamping`,
        :func:`Operators.NonLinear.ActiveStressTensor`). Breaking the invariant
        silently corrupts every energy-based algorithm, e.g. the discrete gradient
        of :func:`Operators.NonLinear.GonzalezStressTensor`.

        Returns
        -------
        FeArray
            Σ_e_pg of shape (Ne, pg, d), where `d = 1, 3, 6` depending on whether the solution dimension is `1D`, `2D`, or `3D`.

        Σxx, Σyy, Σzz, sqrt(2) Σyz, sqrt(2) Σxz, sqrt(2) Σxy
        """
        return None  # type: ignore [return-value]

    @abstractmethod
    def Compute_d2Wde(self, hyperElasticState: HyperElasticState) -> FeArray:
        """Computes the consistent tangent ``∂²W/∂e² = ∂Σ/∂e``.

        Same invariant as :meth:`Compute_dWde`: strictly the second derivative of
        :meth:`Compute_W`. The non-conservative contributions add nothing here
        anyway — both ``η·ė`` and ``τ·(T̂⊗T̂)`` are independent of ``e`` — but they
        do feed the *geometric* tangent, which their own operators assemble.

        Returns
        -------
        FeArray
            dΣde_e_pg of shape (Ne, pg, d, d), where `d = 1, 3, 6` depending on whether the solution dimension is `1D`, `2D`, or `3D`.
        """
        return None  # type: ignore [return-value]


# ----------------------------------------------
# Neo-Hookean
# ----------------------------------------------


class NeoHookean(_HyperElastic):

    K: float = _params.PositiveScalarParameter()
    """Bulk modulus"""

    def __init__(self, dim: int, K: Union[float, _types.FloatArray], thickness=1.0):
        """Creates an Neo-Hookean material.

        Parameters
        ----------
        dim : int
            dimension (e.g 2 or 3)
        K : float|_types.FloatArray
            Bulk modulus
        thickness : float, optional
            thickness, by default 1.0
        """

        _HyperElastic.__init__(self, dim, thickness)

        self.K = K

    def Compute_W(self, hyperElasticState: HyperElasticState) -> FeArray:
        K = self.K

        I1 = hyperElasticState.Compute_I1()
        I3 = hyperElasticState.Compute_I3()

        W = K * (I1 / I3 ** (1 / 3) - 3)

        return W

    def Compute_dWde(self, hyperElasticState: HyperElasticState) -> FeArray:
        K = self.K

        I1 = hyperElasticState.Compute_I1()
        I3 = hyperElasticState.Compute_I3()

        dWdI1 = K / I3 ** (1 / 3)
        dWdI3 = -I1 * K / (3 * I3 ** (4 / 3))
        dI1dC = hyperElasticState.Compute_dI1dC()
        dI3dC = hyperElasticState.Compute_dI3dC()

        dWdI1 = K / I3 ** (1 / 3)
        dWdI3 = -I1 * K / (3 * I3 ** (4 / 3))
        dW = 2 * (dWdI1 * dI1dC + dWdI3 * dI3dC)

        return dW

    def Compute_d2Wde(self, hyperElasticState: HyperElasticState) -> FeArray:
        K = self.K

        I1 = hyperElasticState.Compute_I1()
        I3 = hyperElasticState.Compute_I3()

        dI1dC = hyperElasticState.Compute_dI1dC()
        dI3dC = hyperElasticState.Compute_dI3dC()
        d2I1dC = hyperElasticState.Compute_d2I1dC()
        d2I3dC = hyperElasticState.Compute_d2I3dC()

        dWdI1 = K / I3 ** (1 / 3)
        d2WdI1dI3 = -K / (3 * I3 ** (4 / 3))
        dWdI3 = -I1 * K / (3 * I3 ** (4 / 3))
        d2WdI3dI1 = -K / (3 * I3 ** (4 / 3))
        d2WdI3dI3 = 4 * I1 * K / (9 * I3 ** (7 / 3))

        d2W = 4 * (dWdI1 * d2I1dC + dWdI3 * d2I3dC) + 4 * (
            d2WdI1dI3 * TensorProd(dI1dC, dI3dC)
            + d2WdI3dI1 * TensorProd(dI3dC, dI1dC)
            + d2WdI3dI3 * TensorProd(dI3dC, dI3dC)
        )

        return d2W


# ----------------------------------------------
# Mooney-Rivlin
# ----------------------------------------------


class MooneyRivlin(_HyperElastic):

    K1: float = _params.PositiveScalarParameter()
    """Kappa1"""

    K2: float = _params.PositiveScalarParameter()
    """Kappa2"""

    K: float = _params.PositiveScalarParameter()
    """Bulk modulus"""

    def __init__(
        self,
        dim: int,
        K1: Union[float, _types.FloatArray],
        K2: Union[float, _types.FloatArray],
        K: Union[float, _types.FloatArray] = 0.0,
        thickness=1.0,
    ):
        """Creates an Mooney-Rivlin material.

        Parameters
        ----------
        dim : int
            dimension (e.g 2 or 3)
        K1 : float|_types.FloatArray
            Kappa1
        K2 : float|_types.FloatArray
            Kappa2 -> Neo-Hoolkean if K2=0
        K : float|_types.FloatArray, optional
            Bulk modulus, by default 0.0
        thickness : float, optional
            thickness, by default 1.0
        """

        _HyperElastic.__init__(self, dim, thickness)

        self.K1 = K1
        self.K2 = K2
        self.K = K

    def Compute_W(self, hyperElasticState: HyperElasticState) -> FeArray:
        K = self.K
        K1 = self.K1
        K2 = self.K2

        I1 = hyperElasticState.Compute_I1()
        I2 = hyperElasticState.Compute_I2()
        I3 = hyperElasticState.Compute_I3()

        W = (
            K * (np.sqrt(I3) - 1) ** 2
            + K1 * (I1 / I3 ** (1 / 3) - 3)
            + K2 * (I2 / I3 ** (2 / 3) - 3)
        )

        return W

    def Compute_dWde(self, hyperElasticState: HyperElasticState) -> FeArray:
        K = self.K
        K1 = self.K1
        K2 = self.K2

        I1 = hyperElasticState.Compute_I1()
        I2 = hyperElasticState.Compute_I2()
        I3 = hyperElasticState.Compute_I3()

        dI1dC = hyperElasticState.Compute_dI1dC()
        dI2dC = hyperElasticState.Compute_dI2dC()
        dI3dC = hyperElasticState.Compute_dI3dC()

        dWdI1 = K1 / I3 ** (1 / 3)
        dWdI2 = K2 / I3 ** (2 / 3)
        dWdI3 = (
            -I1 * K1 / (3 * I3 ** (4 / 3))
            - 2 * I2 * K2 / (3 * I3 ** (5 / 3))
            + K * (np.sqrt(I3) - 1) / np.sqrt(I3)
        )

        dW = 2 * (dWdI1 * dI1dC + dWdI2 * dI2dC + dWdI3 * dI3dC)

        return dW

    def Compute_d2Wde(self, hyperElasticState: HyperElasticState) -> FeArray:
        K = self.K
        K1 = self.K1
        K2 = self.K2

        I1 = hyperElasticState.Compute_I1()
        I2 = hyperElasticState.Compute_I2()
        I3 = hyperElasticState.Compute_I3()

        dI1dC = hyperElasticState.Compute_dI1dC()
        dI2dC = hyperElasticState.Compute_dI2dC()
        dI3dC = hyperElasticState.Compute_dI3dC()

        d2I1dC = hyperElasticState.Compute_d2I1dC()
        d2I2dC = hyperElasticState.Compute_d2I2dC()
        d2I3dC = hyperElasticState.Compute_d2I3dC()

        dWdI1 = K1 / I3 ** (1 / 3)
        d2WdI1dI3 = -K1 / (3 * I3 ** (4 / 3))
        dWdI2 = K2 / I3 ** (2 / 3)
        d2WdI2dI3 = -2 * K2 / (3 * I3 ** (5 / 3))
        dWdI3 = (
            -I1 * K1 / (3 * I3 ** (4 / 3))
            - 2 * I2 * K2 / (3 * I3 ** (5 / 3))
            + K * (np.sqrt(I3) - 1) / np.sqrt(I3)
        )
        d2WdI3dI1 = -K1 / (3 * I3 ** (4 / 3))
        d2WdI3dI2 = -2 * K2 / (3 * I3 ** (5 / 3))
        d2WdI3dI3 = (
            4 * I1 * K1 / (9 * I3 ** (7 / 3))
            + 10 * I2 * K2 / (9 * I3 ** (8 / 3))
            + K / (2 * I3)
            - K * (np.sqrt(I3) - 1) / (2 * I3 ** (3 / 2))
        )

        d2W = 4 * (dWdI1 * d2I1dC + dWdI2 * d2I2dC + dWdI3 * d2I3dC) + 4 * (
            d2WdI1dI3 * TensorProd(dI1dC, dI3dC)
            + d2WdI2dI3 * TensorProd(dI2dC, dI3dC)
            + d2WdI3dI1 * TensorProd(dI3dC, dI1dC)
            + d2WdI3dI2 * TensorProd(dI3dC, dI2dC)
            + d2WdI3dI3 * TensorProd(dI3dC, dI3dC)
        )

        return d2W


# ----------------------------------------------
# Ciarlet-Geymonat
# ----------------------------------------------


class CiarletGeymonat(_HyperElastic):

    K1: float = _params.PositiveScalarParameter()
    """Kappa1"""

    K2: float = _params.PositiveScalarParameter()
    """Kappa2"""

    K: float = _params.PositiveScalarParameter()
    """Bulk modulus"""

    def __init__(
        self,
        dim: int,
        K1: Union[float, _types.FloatArray],
        K2: Union[float, _types.FloatArray],
        K: Union[float, _types.FloatArray] = 0.0,
        thickness=1.0,
    ):
        """Creates an Ciarlet-Geymonat material.

        Parameters
        ----------
        dim : int
            dimension (e.g 2 or 3)
        K1 : float|_types.FloatArray
            Kappa1
        K2 : float|_types.FloatArray
            Kappa2 -> Neo-Hoolkean if K2=0
        K : float|_types.FloatArray, optional
            Bulk modulus, by default 0.0
        thickness : float, optional
            thickness, by default 1.0
        """

        _HyperElastic.__init__(self, dim, thickness)

        self.K1 = K1
        self.K2 = K2
        self.K = K

    def Compute_W(self, hyperElasticState: HyperElasticState) -> FeArray:
        K = self.K
        K1 = self.K1
        K2 = self.K2

        I1 = hyperElasticState.Compute_I1()
        I2 = hyperElasticState.Compute_I2()
        I3 = hyperElasticState.Compute_I3()

        W = (
            K * (np.sqrt(I3) - np.log(np.sqrt(I3)) - 1)
            + K1 * (I1 / I3 ** (1 / 3) - 3)
            + K2 * (I2 / I3 ** (2 / 3) - 3)
        )

        return W

    def Compute_dWde(self, hyperElasticState: HyperElasticState) -> FeArray:
        K = self.K
        K1 = self.K1
        K2 = self.K2

        I1 = hyperElasticState.Compute_I1()
        I2 = hyperElasticState.Compute_I2()
        I3 = hyperElasticState.Compute_I3()

        dI1dC = hyperElasticState.Compute_dI1dC()
        dI2dC = hyperElasticState.Compute_dI2dC()
        dI3dC = hyperElasticState.Compute_dI3dC()

        dWdI1 = K1 / I3 ** (1 / 3)
        dWdI2 = K2 / I3 ** (2 / 3)
        dWdI3 = (
            -I1 * K1 / (3 * I3 ** (4 / 3))
            - 2 * I2 * K2 / (3 * I3 ** (5 / 3))
            + K * (-1 / (2 * I3) + 1 / (2 * np.sqrt(I3)))
        )

        dW = 2 * (dWdI1 * dI1dC + dWdI2 * dI2dC + dWdI3 * dI3dC)

        return dW

    def Compute_d2Wde(self, hyperElasticState: HyperElasticState) -> FeArray:
        K = self.K
        K1 = self.K1
        K2 = self.K2

        I1 = hyperElasticState.Compute_I1()
        I2 = hyperElasticState.Compute_I2()
        I3 = hyperElasticState.Compute_I3()

        dI1dC = hyperElasticState.Compute_dI1dC()
        dI2dC = hyperElasticState.Compute_dI2dC()
        dI3dC = hyperElasticState.Compute_dI3dC()

        d2I1dC = hyperElasticState.Compute_d2I1dC()
        d2I2dC = hyperElasticState.Compute_d2I2dC()
        d2I3dC = hyperElasticState.Compute_d2I3dC()

        dWdI1 = K1 / I3 ** (1 / 3)
        d2WdI1dI3 = -K1 / (3 * I3 ** (4 / 3))
        dWdI2 = K2 / I3 ** (2 / 3)
        d2WdI2dI3 = -2 * K2 / (3 * I3 ** (5 / 3))
        dWdI3 = (
            -I1 * K1 / (3 * I3 ** (4 / 3))
            - 2 * I2 * K2 / (3 * I3 ** (5 / 3))
            + K * (-1 / (2 * I3) + 1 / (2 * np.sqrt(I3)))
        )
        d2WdI3dI1 = -K1 / (3 * I3 ** (4 / 3))
        d2WdI3dI2 = -2 * K2 / (3 * I3 ** (5 / 3))
        d2WdI3dI3 = (
            4 * I1 * K1 / (9 * I3 ** (7 / 3))
            + 10 * I2 * K2 / (9 * I3 ** (8 / 3))
            + K * (1 / (2 * I3**2) - 1 / (4 * I3 ** (3 / 2)))
        )

        d2W = 4 * (dWdI1 * d2I1dC + dWdI2 * d2I2dC + dWdI3 * d2I3dC) + 4 * (
            d2WdI1dI3 * TensorProd(dI1dC, dI3dC)
            + d2WdI2dI3 * TensorProd(dI2dC, dI3dC)
            + d2WdI3dI1 * TensorProd(dI3dC, dI1dC)
            + d2WdI3dI2 * TensorProd(dI3dC, dI2dC)
            + d2WdI3dI3 * TensorProd(dI3dC, dI3dC)
        )

        return d2W


# ----------------------------------------------
# Saint-Venant-Kirchhoff
# ----------------------------------------------


class SaintVenantKirchhoff(_HyperElastic):

    lmbda: float = _params.ScalarParameter()
    """Lame's first parameter"""

    mu: float = _params.PositiveScalarParameter()
    """Shear modulus"""

    K: float = _params.PositiveScalarParameter()
    """Bulk modulus"""

    def __init__(
        self,
        dim: int,
        lmbda: Union[float, _types.FloatArray],
        mu: Union[float, _types.FloatArray],
        K: Union[float, _types.FloatArray] = 0.0,
        thickness=1.0,
    ):
        """Creates Saint-Venant-Kirchhoff material.

        Parameters
        ----------
        dim : int
            dimension (e.g 2 or 3)
        lmbda : float|_types.FloatArray
            Lame's first parameter
        mu : float|_types.FloatArray
            Shear modulus
        K : float|_types.FloatArray, optional
            Bulk modulus, by default 0.0
        """

        _HyperElastic.__init__(self, dim, thickness)

        self.lmbda = lmbda
        self.mu = mu
        self.K = K

    def Compute_W(self, hyperElasticState: HyperElasticState) -> FeArray:
        lmbda = self.lmbda
        mu = self.mu
        K = self.K

        I1 = hyperElasticState.Compute_I1()
        I2 = hyperElasticState.Compute_I2()
        I3 = hyperElasticState.Compute_I3()

        W = (
            I1**2 * (lmbda / 8 + mu / 4)
            - I1 * (3 * lmbda / 4 + mu / 2)
            - I2 * mu / 2
            + 0.5 * K * (I3 - 1) ** 2
            + 9 * lmbda / 8
            + 3 * mu / 4
        )

        return W

    def Compute_dWde(self, hyperElasticState: HyperElasticState) -> FeArray:
        lmbda = self.lmbda
        mu = self.mu
        K = self.K

        I1 = hyperElasticState.Compute_I1()
        I3 = hyperElasticState.Compute_I3()

        dI1dC = hyperElasticState.Compute_dI1dC()
        dI2dC = hyperElasticState.Compute_dI2dC()
        dI3dC = hyperElasticState.Compute_dI3dC()

        dWdI1 = 2 * I1 * (lmbda / 8 + mu / 4) - 3 * lmbda / 4 - mu / 2
        dWdI2 = -mu / 2
        dWdI3 = 0.5 * K * (2 * I3 - 2)

        dW = 2 * (dWdI1 * dI1dC + dWdI2 * dI2dC + dWdI3 * dI3dC)

        return dW

    def Compute_d2Wde(self, hyperElasticState: HyperElasticState) -> FeArray:
        lmbda = self.lmbda
        mu = self.mu
        K = self.K

        I1 = hyperElasticState.Compute_I1()
        I3 = hyperElasticState.Compute_I3()

        dI1dC = hyperElasticState.Compute_dI1dC()
        dI3dC = hyperElasticState.Compute_dI3dC()

        d2I1dC = hyperElasticState.Compute_d2I1dC()
        d2I2dC = hyperElasticState.Compute_d2I2dC()
        d2I3dC = hyperElasticState.Compute_d2I3dC()

        dWdI1 = 2 * I1 * (lmbda / 8 + mu / 4) - 3 * lmbda / 4 - mu / 2
        dWdI2 = -mu / 2
        dWdI3 = 0.5 * K * (2 * I3 - 2)

        d2WdI1dI1 = lmbda / 4 + mu / 2
        d2WdI3dI3 = 1.0 * K

        d2W = 4 * (dWdI1 * d2I1dC + dWdI2 * d2I2dC + dWdI3 * d2I3dC) + 4 * (
            d2WdI1dI1 * TensorProd(dI1dC, dI1dC) + d2WdI3dI3 * TensorProd(dI3dC, dI3dC)
        )

        return d2W


# ----------------------------------------------
# Holzapfel-Ogden
# ----------------------------------------------


class HolzapfelOgden(_HyperElastic):

    C0: float = _params.PositiveScalarParameter()
    C1: float = _params.PositiveScalarParameter()
    C2: float = _params.PositiveScalarParameter()
    C3: float = _params.PositiveScalarParameter()
    C4: float = _params.PositiveScalarParameter()
    C5: float = _params.PositiveScalarParameter()
    C6: float = _params.PositiveScalarParameter()
    C7: float = _params.PositiveScalarParameter()

    K: float = _params.PositiveScalarParameter()
    """Bulk modulus"""

    Mu1: float = _params.PositiveScalarParameter()
    Mu2: float = _params.PositiveScalarParameter()

    T1 = _params.VectorParameter()
    """direction(s) 1, used for the invariants I4 and I8"""
    T2 = _params.VectorParameter()
    """direction(s) 2, used for the invariants I6 and I8"""

    __ks: float = _params.PositiveScalarParameter()
    """A positive constant used in the incompressibility penalty term, as proposed in http://dx.doi.org/10.1016/0045-7825(94)90051-5."""

    def __init__(
        self,
        dim: int,
        C0: float,
        C1: float,
        C2: float,
        C3: float,
        C4: float,
        C5: float,
        C6: float,
        C7: float,
        K: float,
        Mu1: float,
        Mu2: float,
        T1: _types.FloatArray,
        T2: _types.FloatArray,
        ks: float = 100,
        thickness=1.0,
    ):
        """Creates Holzapfel-Ogden material.

        Parameters
        ----------
        dim : int
            dimension (e.g 2 or 3)
        C0 : float
            C0
        C1 : float
            C1
        C2 : float
            C2
        C3 : float
            C3
        C4 : float
            C4
        C5 : float
            C5
        C6 : float
            C6
        C7 : float
            C7
        K : float
            bulk modulus
        Mu1 : float
            Mu1
        Mu2 : float
            Mu2
        T1 : _type.FloatArray
            direction(s) 1, used for the invariants I4 and I8
        T2 : _type.FloatArray
            direction(s) 2, used for the invariants I6 and I8
        Mu2 : float
            Mu2
        ks : float, optional
            A positive constant used in the incompressibility penalty term.
        thickness : float, optional
            thickness, by default 1.0
        """

        _HyperElastic.__init__(self, dim, thickness)

        self.C0 = C0
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        self.C6 = C6
        self.C7 = C7

        self.K = K
        self.Mu1 = Mu1
        self.Mu2 = Mu2

        self.T1 = Normalize(T1)
        self.T2 = Normalize(T2)

        self.__ks = ks

    def Compute_W(self, hyperElasticState: HyperElasticState) -> FeArray:
        C0 = self.C0
        C1 = self.C1
        C2 = self.C2
        C3 = self.C3
        C4 = self.C4
        C5 = self.C5
        C6 = self.C6
        C7 = self.C7
        K = self.K
        Mu1 = self.Mu1
        Mu2 = self.Mu2
        T1 = self.T1
        T2 = self.T2
        ks = self.__ks

        I1 = hyperElasticState.Compute_I1()
        I2 = hyperElasticState.Compute_I2()
        I3 = hyperElasticState.Compute_I3()
        I4 = hyperElasticState.Compute_I4(T1)
        I6 = hyperElasticState.Compute_I6(T2)
        I8 = hyperElasticState.Compute_I8(T1, T2)

        W = (
            C0 * (np.exp(C1 * (I1 / I3 ** (1 / 3) - 3)) - 1)
            + C2 * (np.exp(C3 * (I4 - 1) ** 2) - 1) / (1 + np.exp(-ks * (I4 - 1)))
            + C4 * (np.exp(C5 * (I6 - 1) ** 2) - 1) / (1 + np.exp(-ks * (I6 - 1)))
            + C6 * (np.exp(C7 * I8**2) - 1)
            + K * (I3 - 2 * np.log(np.sqrt(I3)) - 1) / 4
            + Mu1 * (I1 / I3 ** (1 / 3) - 3)
            + Mu2 * (I2 / I3 ** (2 / 3) - 3)
        )

        return W

    def Compute_dWde(self, hyperElasticState: HyperElasticState) -> FeArray:
        C0 = self.C0
        C1 = self.C1
        C2 = self.C2
        C3 = self.C3
        C4 = self.C4
        C5 = self.C5
        C6 = self.C6
        C7 = self.C7
        K = self.K
        Mu1 = self.Mu1
        Mu2 = self.Mu2
        T1 = self.T1
        T2 = self.T2
        ks = self.__ks

        I1 = hyperElasticState.Compute_I1()
        I2 = hyperElasticState.Compute_I2()
        I3 = hyperElasticState.Compute_I3()
        I4 = hyperElasticState.Compute_I4(T1)
        I6 = hyperElasticState.Compute_I6(T2)
        I8 = hyperElasticState.Compute_I8(T1, T2)

        dI1dC = hyperElasticState.Compute_dI1dC()
        dI2dC = hyperElasticState.Compute_dI2dC()
        dI3dC = hyperElasticState.Compute_dI3dC()
        dI4dC = hyperElasticState.Compute_dI4dC(T1)
        dI6dC = hyperElasticState.Compute_dI6dC(T2)
        dI8dC = hyperElasticState.Compute_dI8dC(T1, T2)

        # see: examples/HyperElastic/HyperElasticLaws.py
        # Common subexpressions factored once (bit-identical to inlining).
        # fmt: off
        I3_13 = I3**(1/3)
        I3_23 = I3**(2/3)
        I3_43 = I3**(4/3)
        I3_53 = I3**(5/3)
        eC1 = np.exp(C1*(I1/I3_13 - 3))
        e4 = np.exp(C3*(I4 - 1)**2)
        em4 = np.exp(-ks*(I4 - 1))
        s4 = 1 + em4
        e6 = np.exp(C5*(I6 - 1)**2)
        em6 = np.exp(-ks*(I6 - 1))
        s6 = 1 + em6

        dWdI1 = C0*C1*eC1/I3_13 + Mu1/I3_13
        dWdI2 = Mu2/I3_23
        dWdI3 = -C0*C1*I1*eC1/(3*I3_43) - I1*Mu1/(3*I3_43) - 2*I2*Mu2/(3*I3_53) + K*(1 - 1/I3)/4
        dWdI4 = C2*C3*(2*I4 - 2)*e4/s4 + C2*ks*(e4 - 1)*em4/s4**2
        dWdI6 = C4*C5*(2*I6 - 2)*e6/s6 + C4*ks*(e6 - 1)*em6/s6**2
        dWdI8 = 2*C6*C7*I8*np.exp(C7*I8**2)
        # fmt: on

        dW = 2 * (
            dWdI1 * dI1dC
            + dWdI2 * dI2dC
            + dWdI3 * dI3dC
            + dWdI4 * dI4dC
            + dWdI6 * dI6dC
            + dWdI8 * dI8dC
        )

        return dW

    def Compute_d2Wde(self, hyperElasticState: HyperElasticState) -> FeArray:
        C0 = self.C0
        C1 = self.C1
        C2 = self.C2
        C3 = self.C3
        C4 = self.C4
        C5 = self.C5
        C6 = self.C6
        C7 = self.C7
        K = self.K
        Mu1 = self.Mu1
        Mu2 = self.Mu2
        T1 = self.T1
        T2 = self.T2
        ks = self.__ks

        I1 = hyperElasticState.Compute_I1()
        I2 = hyperElasticState.Compute_I2()
        I3 = hyperElasticState.Compute_I3()
        I4 = hyperElasticState.Compute_I4(T1)
        I6 = hyperElasticState.Compute_I6(T2)
        I8 = hyperElasticState.Compute_I8(T1, T2)

        dI1dC = hyperElasticState.Compute_dI1dC()
        dI2dC = hyperElasticState.Compute_dI2dC()
        dI3dC = hyperElasticState.Compute_dI3dC()
        dI4dC = hyperElasticState.Compute_dI4dC(T1)
        dI6dC = hyperElasticState.Compute_dI6dC(T2)
        dI8dC = hyperElasticState.Compute_dI8dC(T1, T2)

        d2I1dC = hyperElasticState.Compute_d2I1dC()
        d2I2dC = hyperElasticState.Compute_d2I2dC()
        d2I3dC = hyperElasticState.Compute_d2I3dC()
        # d2I4dC = d2I6dC = d2I8dC = 0 (I4/I6/I8 are linear in C), so their
        # `dWdI* * d2I*dC` contributions below are identically zero and are
        # dropped (each would otherwise broadcast/allocate a full (Ne,nPg,6,6)
        # zero array). Result is bit-identical.

        # Common subexpressions, factored once (bit-identical to inlining): the
        # isotropic exp/powers of I3 and the per-fiber sigmoid blocks. Only the
        # second-derivative coefficients are needed here (the dWdI4/dWdI6/dWdI8
        # first-derivative terms drop out with the zero d2I4/d2I6/d2I8dC above).
        # fmt: off
        I3_23 = I3**(2/3)
        I3_43 = I3**(4/3)
        I3_53 = I3**(5/3)
        I3_73 = I3**(7/3)
        I3_83 = I3**(8/3)
        eC1 = np.exp(C1*(I1/I3**(1/3) - 3))

        I4m1 = I4 - 1
        two_I4m2 = 2*I4 - 2
        e4 = np.exp(C3*(I4 - 1)**2)
        em4 = np.exp(-ks*(I4 - 1))
        s4 = 1 + em4
        I6m1 = I6 - 1
        two_I6m2 = 2*I6 - 2
        e6 = np.exp(C5*(I6 - 1)**2)
        em6 = np.exp(-ks*(I6 - 1))
        s6 = 1 + em6
        eC7 = np.exp(C7*I8**2)

        dWdI1 = C0*C1*eC1/I3**(1/3) + Mu1/I3**(1/3)
        dWdI2 = Mu2/I3_23
        dWdI3 = -C0*C1*I1*eC1/(3*I3_43) - I1*Mu1/(3*I3_43) - 2*I2*Mu2/(3*I3_53) + K*(1 - 1/I3)/4
        d2WdI1dI1 = C0*C1**2*eC1/I3_23
        d2WdI1dI3 = -C0*C1**2*I1*eC1/(3*I3_53) - C0*C1*eC1/(3*I3_43) - Mu1/(3*I3_43)
        d2WdI2dI3 = -2*Mu2/(3*I3_53)
        d2WdI3dI1 = -C0*C1**2*I1*eC1/(3*I3_53) - C0*C1*eC1/(3*I3_43) - Mu1/(3*I3_43)
        d2WdI3dI2 = -2*Mu2/(3*I3_53)
        d2WdI3dI3 = C0*C1**2*I1**2*eC1/(9*I3_83) + 4*C0*C1*I1*eC1/(9*I3_73) + 4*I1*Mu1/(9*I3_73) + 10*I2*Mu2/(9*I3_83) + K/(4*I3**2)
        d2WdI4dI4 = C2*C3**2*two_I4m2**2*e4/s4 + 2*C2*C3*ks*two_I4m2*e4*em4/s4**2 + 2*C2*C3*e4/s4 - C2*ks**2*(e4 - 1)*em4/s4**2 + 2*C2*ks**2*(e4 - 1)*np.exp(-2*ks*I4m1)/s4**3
        d2WdI6dI6 = C4*C5**2*two_I6m2**2*e6/s6 + 2*C4*C5*ks*two_I6m2*e6*em6/s6**2 + 2*C4*C5*e6/s6 - C4*ks**2*(e6 - 1)*em6/s6**2 + 2*C4*ks**2*(e6 - 1)*np.exp(-2*ks*I6m1)/s6**3
        d2WdI8dI8 = 4*C6*C7**2*I8**2*eC7 + 2*C6*C7*eC7
        # fmt: on

        d2W = 4 * (dWdI1 * d2I1dC + dWdI2 * d2I2dC + dWdI3 * d2I3dC) + 4 * (
            d2WdI1dI1 * TensorProd(dI1dC, dI1dC)
            + d2WdI1dI3 * TensorProd(dI1dC, dI3dC)
            + d2WdI2dI3 * TensorProd(dI2dC, dI3dC)
            + d2WdI3dI1 * TensorProd(dI3dC, dI1dC)
            + d2WdI3dI2 * TensorProd(dI3dC, dI2dC)
            + d2WdI3dI3 * TensorProd(dI3dC, dI3dC)
            + d2WdI4dI4 * TensorProd(dI4dC, dI4dC)
            + d2WdI6dI6 * TensorProd(dI6dC, dI6dC)
            + d2WdI8dI8 * TensorProd(dI8dC, dI8dC)
        )

        return d2W


# ----------------------------------------------
# AutoDiff
# ----------------------------------------------


def HyperElasticPotential(
    W: Callable, in_axes: Union[int, tuple] = 0
) -> tuple[Callable, Callable, Callable]:
    """Builds the ``(W, dWde, d2Wde)`` trio from the potential alone. Needs jax.

    Parameters
    ----------
    W : Callable
        ``W(C, *aux)`` for one point, with ``C`` the ``(3, 3)`` right Cauchy-Green tensor
    in_axes : int | tuple, optional
        as in :func:`~EasyFEA.Models._autodiff.Vmap_e_pg`, over ``(C, *aux)``
    """
    # imported here so jax stays optional
    from .._autodiff import Vmap_e_pg, Kelvin_to_tensor

    # differentiated w.r.t. the 6 independent components of C, not its 9 entries
    def W_kelvin(vec, *aux):
        return W(Kelvin_to_tensor(vec), *aux)

    # called first: it is guarded, so a missing jax is reported before jax is used
    W_field = Vmap_e_pg(W_kelvin, in_axes)

    import jax

    dW_field = Vmap_e_pg(jax.grad(W_kelvin), in_axes)
    d2W_field = Vmap_e_pg(jax.hessian(W_kelvin), in_axes)

    def Kelvin_C(state: HyperElasticState) -> FeArray.FeArrayALike:
        return Project_matrix_to_vector(state.Compute_C())

    def Compute_W(state: HyperElasticState, *aux) -> FeArray:
        return W_field(Kelvin_C(state), *aux)

    def Compute_dWde(state: HyperElasticState, *aux) -> FeArray:
        return state._Slice_Vector(2 * dW_field(Kelvin_C(state), *aux))

    def Compute_d2Wde(state: HyperElasticState, *aux) -> FeArray:
        return state._Slice_Matrix(4 * d2W_field(Kelvin_C(state), *aux))

    return (Compute_W, Compute_dWde, Compute_d2Wde)


class AutoDiff(_HyperElastic):
    """A law declared as a potential ``W(C)``, differentiated by jax.

    Needs ``pip install easyfea[jax]``. Runs a potential EasyFEA does not ship, with no derivative written by hand.
    """

    def __init__(
        self,
        dim: int,
        W: Callable,
        aux: tuple = (),
        in_axes: Union[int, tuple] = 0,
        thickness: float = 1.0,
    ):
        """Creates a hyperelastic law from its potential alone.

        Parameters
        ----------
        dim : int
            dimension (e.g 2 or 3)
        W : Callable
            ``W(C, *aux)`` for one material point, with ``C`` the ``(3, 3)`` right Cauchy-Green tensor
        aux : tuple, optional
            fields ``W`` takes after ``C``, fibre directions for instance. Held here because the operators call ``Compute_dWde(state)`` with nothing else.
        in_axes : int | tuple, optional
            which arguments vary per point, as :func:`jax.vmap` reads it
        thickness : float, optional
            thickness, by default 1.0
        """
        _HyperElastic.__init__(self, dim, thickness)

        self.__potential = W
        self.__in_axes = in_axes
        self.__aux = tuple(aux)
        self.__Build()

    def __Build(self) -> None:
        self.__W, self.__dWde, self.__d2Wde = HyperElasticPotential(
            self.__potential, self.__in_axes
        )

    __DERIVED = ("_AutoDiff__W", "_AutoDiff__dWde", "_AutoDiff__d2Wde")

    def __getstate__(self) -> dict:
        """Drops the jax closures, which pickle cannot take. ``W`` must be picklable: define it at module level and bind its parameters with :func:`functools.partial`."""
        return {k: v for k, v in self.__dict__.items() if k not in self.__DERIVED}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.__Build()

    def Compute_W(self, hyperElasticState: HyperElasticState) -> FeArray:
        return self.__W(hyperElasticState, *self.__aux)

    def Compute_dWde(self, hyperElasticState: HyperElasticState) -> FeArray:
        return self.__dWde(hyperElasticState, *self.__aux)

    def Compute_d2Wde(self, hyperElasticState: HyperElasticState) -> FeArray:
        return self.__d2Wde(hyperElasticState, *self.__aux)
