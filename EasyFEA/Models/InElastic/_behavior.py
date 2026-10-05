# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""The behavior contract: a developer writes ``Update`` at one 3D point, EasyFEA does the rest; jax is imported only once a behavior runs."""

import copy
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, ClassVar, NamedTuple, TypeVar

import numpy as np

from .._utils import _IModel
from ..Elastic._laws import _Elastic
from ...FEM._linalg import FeArray
from ...Utilities import _params, Tic
from ...Utilities._observers import _IObserver, Observable

if TYPE_CHECKING:
    from jax import Array
    from ...FEM import Kinematics
    from jax.typing import ArrayLike

X = TypeVar("X")

# ----------------------------------------------
# Helpers, at one point, on (6,) Kelvin vectors
# ----------------------------------------------

ONE = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
"""The identity, as a (6,) Kelvin vector."""

# numpy rather than jax: jax arrays built at import would be float32, before x64 is enabled
ZERO_TENSOR: "Array" = np.zeros(6)  # type: ignore[assignment]
"""The default of a tensor internal variable: a zero (6,) Kelvin vector."""

ZERO_SCALAR: "Array" = np.zeros(())  # type: ignore[assignment]
"""The default of a scalar internal variable."""


def Trace(sig: "Array") -> "Array":
    """Trace of a (6,) Kelvin vector."""
    return sig[:3].sum()


def Deviator(sig: "Array") -> "Array":
    """Deviatoric part of a (6,) Kelvin vector."""
    return sig - Trace(sig) / 3 * ONE


def Von_Mises_stress(sig: "Array") -> "Array":
    """``sqrt(3/2 s:s)``, finite at ``sig = 0`` so that it stays differentiable."""
    import jax.numpy as jnp

    s = Deviator(sig)
    return jnp.sqrt(1.5 * s @ s + 1e-300)


def Newton(
    residual: Callable[[X], X],
    x0: X,
    active: "ArrayLike" = True,
    tol: float = 1e-10,
    maxIter: int = 30,
) -> X:
    """Solves ``residual(x) = 0`` for an array or a NamedTuple of arrays, differentiably through the root; x stays x0 where not ``active``, and is NaN if it does not converge."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.flatten_util import ravel_pytree

    flat0, unravel = ravel_pytree(x0)

    def Flat_residual(flat):
        r, _ = ravel_pytree(residual(unravel(flat)))
        return jnp.where(active, r, flat - flat0)

    def Solve(F, u0):
        # F evaluated once per iteration: nested in another Newton, every extra call multiplies
        def Jacobian_and_residual(u):
            def F_twice(u):
                r = F(u)
                return r, r

            return jax.jacfwd(F_twice, has_aux=True)(u)

        # carry: (flat iterate, iterates checked, converged)
        def Continue(carry: tuple["Array", "Array", "Array"]) -> "Array":
            _, i, done = carry
            return (i <= maxIter) & ~done

        # checks the iterate, then steps from it: F is traced once, in the loop body only
        def Step(
            carry: tuple["Array", "Array", "Array"],
        ) -> tuple["Array", "Array", "Array"]:
            u, i, _ = carry
            J, r = Jacobian_and_residual(u)
            done = jnp.max(jnp.abs(r)) < tol
            u = jnp.where(done, u, u - jnp.linalg.solve(J, r))
            return u, i + 1, done

        u, _, done = lax.while_loop(Continue, Step, (u0, 0, False))
        return jnp.where(done, u, jnp.nan)

    def Tangent_solve(g, y):
        # g is linear, so where its jacobian is evaluated does not matter
        return jnp.linalg.solve(jax.jacfwd(g)(y), y)

    return unravel(lax.custom_root(Flat_residual, flat0, Solve, Tangent_solve))


# ----------------------------------------------
# The base class
# ----------------------------------------------

IDX_2D = np.array([0, 1, 5])
"""In-plane components [xx, yy, xy] of the (6,) Kelvin vector."""
ZZ = 2


class _NoInternals(NamedTuple):
    pass


class _NoExternals(NamedTuple):
    pass


class _Behavior(_IModel, _IObserver):
    """A material whose stress depends on its history: subclass it, declare ``Internals``, write :meth:`Update` at one 3D point."""

    Internals: ClassVar[type] = _NoInternals
    """The internal variables, a NamedTuple whose defaults are the virgin material."""

    Externals: ClassVar[type] = _NoExternals
    """The external variables, a NamedTuple with no defaults whose fields ``Update`` and ``Stress`` read as keywords at the end of the step."""

    _tol: float = _params.PositiveScalarParameter()
    """plane-stress eps_zz solve, on sig_zz scaled to a strain"""
    _maxIter: int = _params.PositiveScalarParameter()
    """plane-stress eps_zz solve"""

    __KERNEL_UPDATE = "_compiled"
    """Where the kernel of :meth:`Integrate` is cached."""
    __KERNEL_STRESS = "_compiledStress"
    """Where the kernel of :meth:`Compute_Sigma` is cached."""
    __KERNELS = (__KERNEL_UPDATE, __KERNEL_STRESS)

    def __init__(self, elastic: _Elastic):
        """``dim``, ``planeStress`` and ``thickness`` are the elastic model's."""
        assert isinstance(elastic, _Elastic), "elastic must be an elastic model"
        assert (
            not self.Externals._field_defaults
        ), "an external variable has no default: a reference value is a parameter"
        elastic._Add_observer(self)
        self.__elastic = elastic
        self._tol = 1e-10
        self._maxIter = 20

    @property
    def elastic(self) -> _Elastic:
        """The elastic model; modifying it rebuilds the kernel."""
        return self.__elastic

    @property
    def dim(self) -> int:
        return self.__elastic.dim

    @property
    def thickness(self) -> float:
        return self.__elastic.thickness

    @property
    def planeStress(self) -> bool:
        """the 2D model uses the plane-stress assumption (otherwise plane strain)"""
        return self.dim == 2 and self.__elastic.planeStress

    @property
    def C(self) -> "Array":
        """The 3D stiffness, in Kelvin-Mandel notation, whatever the model dimension; the point's inside ``Update`` and ``Stress``."""
        if "_C" not in self.__dict__:
            self._C = self.__elastic._Get_C_S(3)[0]
        return self._C  # type: ignore[return-value]

    def __With_stiffness(self, C: "Array") -> "_Behavior":
        """A copy whose ``C`` is the point's, for the kernel to trace, so that the elastic model may be heterogeneous."""
        point = copy.copy(self)
        point._C = C
        return point

    @property
    def externalNames(self) -> tuple[str, ...]:
        """The declared external variables, in order."""
        return self.Externals._fields

    def __C_e_pg(self, Ne: int, nPg: int) -> FeArray.FeArrayALike:
        return FeArray.broadcast(self.C, Ne, nPg, tensor_ndim=2)

    def _Update(
        self,
        observable: Observable,
        event: str,
    ) -> None:
        self.Need_Update()

    @abstractmethod
    def Update(
        self,
        eps: "Array",
        z: Any,
        dt: float,
        **external,
    ) -> tuple["Array", Any]:
        """(6,) strain, internal state ``z`` at the last converged step -> (6,) stress, new internal state."""

    @abstractmethod
    def Stress(
        self,
        eps: "Array",
        z: Any,
        **external,
    ) -> "Array":
        """(6,) strain, internal state ``z`` -> (6,) stress, with no step: ``Update`` returns ``Stress(eps, new)``."""

    def Virgin_internals(self) -> Any:
        """The internal variables of the virgin material; override it when their size depends on the instance."""
        return self.Internals()

    @property
    def coef(self) -> float:
        """Kelvin-Mandel coefficient, used when projecting result fields."""
        return np.sqrt(2)

    def __str__(self) -> str:
        text = f"{type(self).__name__}:"
        if self.dim == 2:
            simplification = "Plane Stress" if self.planeStress else "Plane Strain"
            text += f"\nthickness = {self.thickness:.2e} ({simplification})"
        return text

    def Need_Update(self, value=True) -> None:
        super().Need_Update(value)
        # jit captured the parameters at its first trace
        for cached in ("_C", *_Behavior.__KERNELS):
            self.__dict__.pop(cached, None)

    def __getstate__(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in _Behavior.__KERNELS}

    def __Kernel(
        self,
        name: str,
        point: Callable,
        in_axes: int | tuple = 0,
    ) -> Callable:
        """``point`` lifted to every Gauss point, built once until :meth:`Need_Update`."""
        if name not in self.__dict__:
            from .._autodiff import Enable_x64, Vmap_e_pg

            Enable_x64()
            self.__dict__[name] = Vmap_e_pg(point, in_axes)
        return self.__dict__[name]

    # --------------------------------------------------------------------------
    # The internal state at every point: one (Ne, nPg, ...) array per internal variable
    # --------------------------------------------------------------------------

    def Virgin_internals_e_pg(self, Ne: int, nPg: int) -> dict[str, FeArray]:
        """The virgin internal state at every point, by name; plane stress adds ``eps_zz``."""
        virgin = self.Virgin_internals()._asdict()
        if self.planeStress:
            virgin["eps_zz"] = ZERO_SCALAR
        # tensor_ndim, else a (6,) default reads as (Ne,) when Ne == 6; copied, since broadcast is a read-only view
        return {
            name: FeArray.broadcast(
                np.asarray(v, dtype=float), Ne, nPg, tensor_ndim=np.ndim(v)
            ).copy()
            for name, v in virgin.items()
        }

    # --------------------------------------------------------------------------
    # One point, then every point
    # --------------------------------------------------------------------------

    def __Point(
        self,
        eps: "Array",
        z: dict[str, "Array"],
        dt: float,
        external: dict,
        C: "Array",
    ) -> tuple["Array", "Array", dict[str, "Array"]]:
        """Stress, tangent and new internal state at one point, in the model dimension; the tangent is jacfwd through Update and the eps_zz solve."""
        import jax
        import jax.numpy as jnp

        point = self.__With_stiffness(C)
        Internals = type(self.Virgin_internals())
        zOld = Internals(**{name: z[name] for name in Internals._fields})

        def Update_named(eps6):
            sig6, new = point.Update(eps6, zOld, dt, **external)
            return sig6, new._asdict()

        def Stress(e):
            if self.dim == 3:
                sig, new = Update_named(e)
                return sig, (sig, new)
            eps6 = jnp.zeros(6).at[IDX_2D].set(e)
            if self.planeStress:
                eps6 = eps6.at[ZZ].set(self.__Eps_zz(Update_named, eps6, z["eps_zz"]))
            sig6, new = Update_named(eps6)
            if self.planeStress:
                new["eps_zz"] = eps6[ZZ]
            return sig6[IDX_2D], (sig6[IDX_2D], new)

        C_alg, (sig, z_new) = jax.jacfwd(Stress, has_aux=True)(eps)
        return sig, C_alg, z_new

    def __Point_stress(
        self,
        eps: "Array",
        z: dict[str, "Array"],
        external: dict,
        C: "Array",
    ) -> "Array":
        """Stress at one point, in the model dimension, under the internal state ``z``."""
        import jax.numpy as jnp

        point = self.__With_stiffness(C)
        Internals = type(self.Virgin_internals())
        state = Internals(**{name: z[name] for name in Internals._fields})
        if self.dim == 3:
            return point.Stress(eps, state, **external)
        eps6 = jnp.zeros(6).at[IDX_2D].set(eps)
        if self.planeStress:
            eps6 = eps6.at[ZZ].set(z["eps_zz"])
        return point.Stress(eps6, state, **external)[IDX_2D]

    def __Eps_zz(
        self,
        Update_named: Callable,
        eps6: "Array",
        eps_zz0: "Array",
    ) -> "Array":
        """``eps_zz`` such that ``sig_zz = 0``, from the last converged one."""
        import jax
        from jax import lax

        def Sig_zz(eps_zz):
            return Update_named(eps6.at[ZZ].set(eps_zz))[0][ZZ]

        # sig_zz / C_zz,zz: a strain, so that _tol does not depend on the stress unit
        _, C_zz = jax.jvp(Sig_zz, (eps_zz0,), (np.ones(()),))
        C_zz = lax.stop_gradient(C_zz)
        return Newton(
            lambda eps_zz: Sig_zz(eps_zz) / C_zz,
            eps_zz0,
            tol=self._tol,
            maxIter=self._maxIter,
        )

    def __External_e_pg(
        self,
        external: dict[str, FeArray.FeArrayALike],
        Ne: int,
        nPg: int,
    ) -> dict[str, FeArray.FeArrayALike]:
        """The declared external variables, each broadcast to ``(Ne, nPg)``; asserted here, since inside the trace a missing one fails obscurely."""
        declared = set(self.externalNames)
        missing = declared - set(external)
        unknown = set(external) - declared
        assert not missing, f"external variables missing: {sorted(missing)}"
        assert not unknown, f"{type(self).__name__} reads no {sorted(unknown)}"
        # vmap maps every one over (Ne, nPg), a scalar included
        return {
            name: np.broadcast_to(FeArray.broadcast(v, Ne, nPg), (Ne, nPg))
            for name, v in external.items()
        }

    def Integrate(
        self,
        kinematics: "Kinematics",
        z_e_pg: dict[str, FeArray] | None = None,
        dt: float = 0.0,
        **external,
    ) -> tuple[FeArray, FeArray, dict[str, FeArray]]:
        """Stress, consistent tangent and trial internal state at every Gauss point of `kinematics`, in the model dimension, from the internal state at the last converged step (virgin by default)."""
        return self._Integrate(kinematics.Compute_Epsilon(), z_e_pg, dt, **external)

    def _Integrate(
        self,
        eps_e_pg: FeArray.FeArrayALike,
        z_e_pg: dict[str, FeArray] | None = None,
        dt: float = 0.0,
        **external,
    ) -> tuple[FeArray, FeArray, dict[str, FeArray]]:
        """:meth:`Integrate` at a prescribed strain field, with no mesh — what :class:`MaterialPoint` drives."""
        tic = Tic()
        eps_e_pg = FeArray.asfearray(eps_e_pg)
        Ne, nPg = eps_e_pg.shape[:2]
        if z_e_pg is None:
            z_e_pg = self.Virgin_internals_e_pg(Ne, nPg)
        external = self.__External_e_pg(external, Ne, nPg)

        # dt is shared
        kernel = self.__Kernel(
            _Behavior.__KERNEL_UPDATE, self.__Point, (0, 0, None, 0, 0)
        )
        out = kernel(eps_e_pg, z_e_pg, dt, external, self.__C_e_pg(Ne, nPg))
        # copied, since jax hands out read-only buffers
        sig, C_alg = (FeArray.asfearray(np.array(a)) for a in out[:2])
        # in the declaration order, which jax sorts away
        z = {name: FeArray.asfearray(np.array(out[2][name])) for name in z_e_pg}

        failed = ~np.isfinite(sig).all(-1)
        assert not failed.any(), (
            f"constitutive integration did not converge at {int(failed.sum())} of "
            f"{failed.size} Gauss points - reduce the load step"
        )
        tic.Tac("Matrix", "Behavior integrate", False)
        return sig, C_alg, z

    def Compute_Sigma(
        self,
        kinematics: "Kinematics",
        z_e_pg: dict[str, FeArray],
        **external,
    ) -> FeArray:
        """Stress at every Gauss point of `kinematics`, in the model dimension, under the internal state ``z_e_pg``, with no step."""
        eps_e_pg = kinematics.Compute_Epsilon()
        Ne, nPg = eps_e_pg.shape[:2]
        external = self.__External_e_pg(external, Ne, nPg)
        kernel = self.__Kernel(_Behavior.__KERNEL_STRESS, self.__Point_stress)
        # copied, since jax hands out read-only buffers
        return FeArray.asfearray(
            np.array(kernel(eps_e_pg, z_e_pg, external, self.__C_e_pg(Ne, nPg)))
        )


# ----------------------------------------------
# One material point, with no mesh
# ----------------------------------------------

COMPONENTS = {"xx": 0, "yy": 1, "zz": 2, "yz": 3, "xz": 4, "xy": 5}
"""Kelvin-Mandel component names; the shear entries carry a sqrt(2)."""


class MaterialPoint:
    """Runs a 3D :class:`_Behavior` at one point: each component is strain-driven by its history, or stress-driven to a target (zero by default) by a Newton on the free strains."""

    _tol: float = 1e-9
    _maxIter: int = 50

    def __init__(self, behavior: _Behavior):
        assert isinstance(behavior, _Behavior), "behavior must be a _Behavior"
        assert behavior.dim == 3, "a material point runs the 3D behavior"
        self.behavior = behavior

    def Run(
        self,
        strain: dict[str, np.ndarray],
        stress: dict[str, np.ndarray] | None = None,
        dt: float = 0.0,
        external: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """``strain`` and ``stress`` as ``(nstep, 6)``, plus one ``(nstep, ...)`` entry per internal variable; each ``external`` is ``(nstep,)``."""
        assert strain, "at least one component must be strain-controlled"
        driven = {COMPONENTS[k]: np.asarray(v, dtype=float) for k, v in strain.items()}
        targets = {
            COMPONENTS[k]: np.asarray(v, dtype=float) for k, v in (stress or {}).items()
        }
        assert not (
            set(driven) & set(targets)
        ), "a component is either strain- or stress-driven"
        free = [i for i in range(6) if i not in driven]
        external = {k: np.asarray(v, dtype=float) for k, v in (external or {}).items()}

        eps = np.zeros(6)
        z = self.behavior.Virgin_internals_e_pg(1, 1)
        strains, stresses, states = [], [], []
        for k in range(len(next(iter(driven.values())))):
            for i, path in driven.items():
                eps[i] = path[k]
            target = np.array([targets[i][k] if i in targets else 0.0 for i in free])
            external_k = {name: v[k] for name, v in external.items()}

            for _ in range(self._maxIter):
                sig_e_pg, C_e_pg, zNew = self.behavior._Integrate(
                    eps[None, None], z, dt, **external_k
                )
                # one point, read back as plain vectors
                sig, C_alg = np.asarray(sig_e_pg)[0, 0], np.asarray(C_e_pg)[0, 0]
                r = sig[free] - target
                if not free or np.max(np.abs(r)) < self._tol:
                    break
                eps[free] -= np.linalg.solve(C_alg[np.ix_(free, free)], r)
            else:
                raise AssertionError(f"the stress targets were not reached at step {k}")

            z = zNew
            strains.append(eps.copy())
            stresses.append(sig)
            states.append({name: np.asarray(v)[0, 0] for name, v in z.items()})

        out = {"strain": np.array(strains), "stress": np.array(stresses)}
        for name in z:
            out[name] = np.array([state[name] for state in states])
        return out
