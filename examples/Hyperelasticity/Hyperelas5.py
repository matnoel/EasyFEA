# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Hyperelas5
==========

A hyperelastic cantilever is deflected, released and left to vibrate freely — no external load
and no damping — so its total energy ``KE + W`` should stay constant. Whether it does depends on
**how the internal-force stress is sampled over each step**, and this example compares the choices
offered by ``Solver_Set_Stress``:

- ``gonzalez`` — the energy-momentum discrete-gradient stress: conserves ``KE + W`` to round-off,
  for any law, from a single stress evaluation.
- ``quadrature`` — the stress averaged along the step's strain path, at increasing resolution:
  a fixed Clenshaw-Curtis rule (``nPoints`` = 1, 2, 3, 5, 9 — midpoint, trapezoid, Simpson, …) or
  the adaptive per-element rule (``energyTol``). It conserves energy only up to its quadrature error,
  so a coarse rule (1–2 points) drifts while a fine one (9 points, or a tight ``energyTol``)
  matches ``gonzalez``.

The figures show each scheme's energy drift over time, the kinetic / strain-energy exchange of the
reference run, and how the adaptive rule spread its quadrature points across the mesh over time.
"""

import matplotlib.pyplot as plt
import numpy as np

from EasyFEA import (
    Terminal,
    Folder,
    ElemType,
    Models,
    Simulations,
    AlgoType,
    Tic,
    Matplotlib,
    PyVista,
)
from EasyFEA.Geoms import Domain

if __name__ == "__main__":
    Terminal.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    folder = Folder.Results_Dir()
    makeMovie = True
    result = "uy"

    ST = Simulations.HyperElastic.StressType
    list_config = [
        ("gonzalez", dict(stressType=ST.gonzalez)),
        # ("gonzalez_approx", dict(stressType=ST.gonzalez, useConsistentTangent=False)),
        # ("pointwise", dict(stressType=ST.pointwise)),
        ("1 pt (midpoint)", dict(stressType=ST.quadrature, nPoints=1)),
        # ("2 pts (trapezoid)", dict(stressType=ST.quadrature, nPoints=2)),
        ("3 pts (simpson)", dict(stressType=ST.quadrature, nPoints=3)),
        # ("5 pts", dict(stressType=ST.quadrature, nPoints=5)),
        ("9 pts", dict(stressType=ST.quadrature, nPoints=9)),
        ("energyTol=1e-6", dict(stressType=ST.quadrature, energyTol=1e-6)),
        # ("energyTol=1e-9", dict(stressType=ST.quadrature, energyTol=1e-9)),
        ("energyTol=1e-12", dict(stressType=ST.quadrature, energyTol=1e-12)),
    ]

    # geom
    L = 50
    h = 5

    # model
    K = 4.0e5

    # load
    deflection = L * 0.4
    nPreload = 1
    T = 3.0
    Nt = 50

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    domain = Domain((0, 0), (L, h), h / 3)
    mesh = domain.Mesh_2D([], ElemType.QUAD4, isOrganised=True)

    nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L)

    # ----------------------------------------------
    # Deflect (static), release, integrate freely
    # ----------------------------------------------

    def run(label, **stress):
        """`stress` is forwarded verbatim to Solver_Set_Stress."""

        Terminal.Section(label)

        mat = Models.HyperElastic.NeoHookean(2, K, thickness=h)
        simu = Simulations.HyperElastic(mesh, mat, maxIter=50)

        tic = Tic()

        # displacement-controlled preload, ramped
        for s in range(1, nPreload + 1):
            simu.Bc_Init()
            simu.add_dirichlet(nodesX0, [0, 0], simu.Get_unknowns())
            simu.add_dirichlet(nodesXL, [-deflection * s / nPreload], ["y"])
            simu.Solve()
        simu.Save_Iter()

        # release, then free vibration
        dt = T / Nt
        simu.Bc_Init()
        simu.Solver_Set_Hyperbolic_Algorithm(dt, algo=AlgoType.midpoint)
        simu.Solver_Set_Stress(**stress)
        simu.add_dirichlet(nodesX0, [0, 0], simu.Get_unknowns())

        problemType = simu.problemType

        list_t = [0.0]
        list_KE = [0.0]
        list_W = [0.0]
        list_W[0] = simu._Calc_W(returnScalar=True)
        list_newton = []

        M = None
        for i in range(1, Nt):
            simu.Solve()
            simu.Save_Iter()
            if M is None:
                M = simu.Get_K_C_M_F(problemType)[2]
            v = simu._Get_v_n(problemType)
            list_t.append(i * dt)
            list_KE.append(0.5 * float(v @ (M @ v)))
            list_W.append(simu._Calc_W(returnScalar=True))
            list_newton.append(simu.Get_results(-1)["newtonIter"])

        return (
            simu,
            np.array(list_t),
            np.array(list_KE),
            np.array(list_W),
            tic.Tac(),
            np.array(list_newton),
        )

    # parallel to list_config; each entry: (simu, times, KE, W, t, newtonIter)
    list_results: list[
        tuple[
            Simulations.HyperElastic,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            float,
            np.ndarray,
        ]
    ] = [run(label, **kw) for label, kw in list_config]

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    Terminal.Section("Results")

    simu, times, KE_ref, W_ref, _, _ = list_results[0]
    E0 = KE_ref[0] + W_ref[0]

    # one pass: print each scheme's drift/cost and plot its drift over time (log axis)
    ax1 = Matplotlib.Init_Axes()
    for (label, _), (_, _, KE, W, t, newton) in zip(list_config, list_results):
        drift = np.abs(KE + W - E0) / E0
        print(
            f"{label:19s}: max |KE+W-E0|/E0 = {drift.max():.2e} | "
            f"newton {newton.sum():4d} ({newton.mean():.2f}/step) | {t:.3f} s"
        )
        drift[drift <= 0] = np.nan  # t = 0 is exactly 0 by construction
        ax1.semilogy(times, drift, label=f"{label} ({t:.2f} s, {newton.sum()} it)")
    ax1.set_xlabel("time")
    ax1.set_ylabel("relative drift  |KE + W - E0| / E0")
    ax1.set_title("Energy drift vs time")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=7, ncol=2, loc="center left", framealpha=0.9)
    Matplotlib.Save_fig(folder, "energy drift")

    # reference scheme: kinetic / strain / total exchange
    ax2 = Matplotlib.Init_Axes()
    ax2.plot(times, KE_ref, label="KE")
    ax2.plot(times, W_ref, label="W")
    ax2.plot(times, KE_ref + W_ref, label="KE + W", color="k")
    ax2.set_xlabel("time")
    ax2.set_ylabel("energy")
    ax2.set_title(f"{list_config[0][0]}: kinetic / strain / total")
    ax2.legend()
    Matplotlib.Save_fig(folder, "energy")

    # adaptive refinement of the last (energyTol) run: share of the mesh at each Clenshaw–Curtis
    # level over time, stacked to 1 so band thickness reads directly as the fraction of elements —
    # most stay at the cheapest rule, more refine only during the hard-bending phases of the swing
    ax3 = Matplotlib.Init_Axes()
    simu1 = list_results[-1][0]
    nPts_e = [simu1.Get_results(i)["nPts_e"] for i in range(1, simu1.Niter)]
    levels = np.unique(np.concatenate(nPts_e))  # CC point counts actually used
    frac = np.array([[np.mean(step == lvl) for step in nPts_e] for lvl in levels])
    ax3.stackplot(times[1:], frac, labels=[f"{lvl} pts" for lvl in levels])
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("time")
    ax3.set_ylabel("fraction of elements")
    ax3.set_title(f"{list_config[-1][0]}: adaptive quadrature refinement")
    ax3.legend(loc="center left", fontsize=8)
    Matplotlib.Save_fig(folder, "adaptive points")

    if makeMovie:

        block = Domain((0, -h), (-h, 2 * h)).Mesh_2D()

        def DoAnim(plotter, i):
            simu.Set_Iter(i)
            PyVista.Plot(block, color="k", plotter=plotter)
            PyVista.Plot(simu, plotMesh=True, deformFactor=1, plotter=plotter)

            PyVista._setCameraPosition(
                plotter,
                2,
                bounds=(-2 * h, L * 1.2, -deflection, deflection, 0, 0),
            )
            plotter.hide_axes()

        PyVista.Movie_func(
            DoAnim,
            simu.Niter,
            folder,
            f"{result}.gif",
        )

    simu.Set_Iter(0)
    PyVista.Plot(simu, result, deformFactor=1.0, plotMesh=True).show()

    plt.show()
