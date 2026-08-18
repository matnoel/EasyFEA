# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Driving a behaviour at a single material point, with no mesh and no solver."""

from typing import Optional

import numpy as np

from ._behaviour import Behaviour
from ..FEM._linalg import FeArray
from ..Utilities import _types

COMPONENTS = {"xx": 0, "yy": 1, "zz": 2, "yz": 3, "xz": 4, "xy": 5}
"""Kelvin-Mandel component names. The shear entries carry a sqrt(2)."""


class MaterialPoint:
    r"""Runs a :class:`~EasyFEA.Models.Behaviour` on one Gauss point.

    Each component is either **strain-controlled** — you give its history — or
    **stress-controlled**, where the strain is solved for until the stress reaches its target
    (zero unless you say otherwise). That is what lets the textbook cases be written down:
    uniaxial *stress* is ``eps_xx`` prescribed with everything else free, so the lateral
    contraction comes out of the solve rather than being assumed.

    It runs the law on a ``(1, 1)`` :class:`~EasyFEA.FEM.FeArray`, which is the same code path
    assembly uses — so this is the real behaviour, not a second implementation of it.

    Examples
    --------
    Uniaxial tension, load then unload::

        law = Models.Behaviour(3, Isotropic(3, E=210e3, v=0.3),
                               hardening=Models.IsotropicHardening.Linear(2000.0),
                               yieldSurface=Models.Yield.VonMises(250.0))
        path = np.concatenate([np.linspace(0, 5e-3, 40), np.linspace(5e-3, 0, 40)[1:]])
        res = MaterialPoint(law).Run(strain={"xx": path})
        sigma_xx, eps_xx = res["stress"][:, 0], res["strain"][:, 0]
    """

    _tol: float = 1e-9
    _maxIter: int = 50

    def __init__(self, behaviour: Behaviour):
        assert isinstance(behaviour, Behaviour), "behaviour must be a Behaviour"
        assert behaviour.dim == 3, "a material point runs the 3D behaviour"
        self.__behaviour = behaviour

    @property
    def behaviour(self) -> Behaviour:
        """The material being driven."""
        return self.__behaviour

    @staticmethod
    def __fe(vec: _types.FloatArray) -> FeArray.FeArrayALike:
        return FeArray.asfearray(vec[np.newaxis, np.newaxis])

    def Run(
        self,
        strain: dict[str, _types.FloatArray],
        stress: Optional[dict[str, _types.FloatArray]] = None,
        dt: float = 0.0,
    ) -> dict[str, _types.FloatArray]:
        """Walks the prescribed history, one step at a time.

        Parameters
        ----------
        strain : dict[str, FloatArray]
            strain-controlled components, e.g. ``{"xx": np.linspace(0, 5e-3, 50)}``
        stress : dict[str, FloatArray], optional
            stress targets for the remaining components; those left out are held at zero
        dt : float, optional
            time increment per step, needed by rate-dependent behaviours

        Returns
        -------
        dict
            ``strain`` and ``stress`` as ``(nstep, 6)``, ``state`` as ``(nstep, n_z)``, plus one
            entry per internal variable named after its slot (``eps_p``, ``alpha``, ``d``, ...).
        """
        assert strain, "at least one component must be strain-controlled"
        driven = {COMPONENTS[k]: np.asarray(v, dtype=float) for k, v in strain.items()}
        targets = {
            COMPONENTS[k]: np.asarray(v, dtype=float) for k, v in (stress or {}).items()
        }
        assert not (
            set(driven) & set(targets)
        ), "a component is either strain- or stress-driven"

        nstep = len(next(iter(driven.values())))
        free = [i for i in range(6) if i not in driven]

        eps = np.zeros(6)
        z = None
        history: list[tuple] = []

        for k in range(nstep):
            for i, path in driven.items():
                eps[i] = path[k]

            target = np.zeros(len(free))
            for j, i in enumerate(free):
                if i in targets:
                    target[j] = targets[i][k]

            # one point, so the fields are read back as plain 6-vectors
            sig = zNew = ok = None
            for _ in range(self._maxIter):
                sig_e_pg, C_e_pg, zNew, ok = self.__behaviour.Integrate(
                    self.__fe(eps), z, dt
                )
                sig = np.asarray(sig_e_pg)[0, 0]
                if not free:
                    break
                r = sig[free] - target
                if np.max(np.abs(r)) < self._tol:
                    break
                sub = np.asarray(C_e_pg)[0, 0][np.ix_(free, free)]
                eps[free] -= np.linalg.solve(sub, r)

            assert (
                ok is not None and ok.all()
            ), f"the behaviour did not converge at step {k} - reduce the step size"
            z = zNew
            history.append((eps.copy(), sig.copy(), np.asarray(z)[0, 0].copy()))

        out = {
            "strain": np.array([h[0] for h in history]),
            "stress": np.array([h[1] for h in history]),
            "state": np.array([h[2] for h in history]),
        }
        for name, slot in self.__behaviour.layout.slots.items():
            values = out["state"][:, slot]
            # str(), so callers get a plain-string dict whatever the layout keys are
            out[str(name)] = values[:, 0] if values.shape[1] == 1 else values
        return out
