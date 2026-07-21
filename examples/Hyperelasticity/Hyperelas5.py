# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Hyperelas5
==========

A hyperelastic cantilever is deflected, released and vibrates freely. The energy-momentum
``gonzalez`` scheme keeps the total energy ``KE + W`` constant; the ``midpoint`` rule drifts.
"""

# sphinx_gallery_thumbnail_number = -1

import matplotlib.pyplot as plt
import numpy as np

from EasyFEA import (
    Terminal,
    Folder,
    Matplotlib,
    ElemType,
    Models,
    Simulations,
    AlgoType,
    Tic,
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

    # schemes to compare, as (label, useGonzalez, useConsistentTangent).
    # The first one is the reference: shown on the right and animated.
    list_config = [
        ("gonzalez", True, True),
        ("gonzalez_noTangent", True, False),
        ("midpoint", False, True),
    ]

    # geom
    L = 50
    h = 5

    # model
    K = 4.0e5

    # load
    deflection = L * 0.4
    nPreload = 2
    T = 3.0
    Nt = 100

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

    def run(label, useGonzalez, useConsistentTangent) -> None:

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
        if useGonzalez:
            simu.Solver_Set_Gonzalez(useConsistentTangent)
        simu.add_dirichlet(nodesX0, [0, 0], simu.Get_unknowns())

        problemType = simu.problemType

        list_t = [0.0]
        list_KE = [0.0]
        list_W = [0.0]
        list_W[0] = simu._Calc_W(returnScalar=True)

        M = None
        for i in range(1, Nt):
            # for i in range(1, 10):
            simu.Solve()
            simu.Save_Iter()
            if M is None:
                M = simu.Get_K_C_M_F(problemType)[2]
            v = simu._Get_v_n(problemType)
            list_t.append(i * dt)
            list_KE.append(0.5 * float(v @ (M @ v)))
            list_W.append(simu._Calc_W(returnScalar=True))

        return (
            simu,
            np.array(list_t),
            np.array(list_KE),
            np.array(list_W),
            tic.Tac(),
        )

    # {label: (simu, times, KE, W, t)}
    runs: dict[str, tuple] = {cfg[0]: run(*cfg) for cfg in list_config}

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    Terminal.Section("Results")

    simu_ref, times, KE_ref, W_ref, _ = runs[list_config[0][0]]
    E0 = KE_ref[0] + W_ref[0]

    for label in runs:
        _, _, KE, W, _ = runs[label]
        print(f"{label:19s}: max |KE+W-E0|/E0 = {np.abs(KE + W - E0).max() / E0:.2e}")

    axs: list[plt.Axes]
    _, axs = plt.subplots(1, 2, figsize=(12, 4.5))

    # total energy of every scheme
    ax1 = axs[0]
    for label in runs:
        _, _, KE, W, t = runs[label]
        ls = "--" if label.endswith("noTangent") else None
        ax1.plot(times, KE + W, label=f"{label} ({t:.3f} s)", ls=ls)
    # ax1.axhline(E0, color="k", lw=0.6, ls="--")
    # ax1.set_ylim(E0 - 10, E0 + 10)
    ax1.set_xlabel("time")
    ax1.set_ylabel("total energy  KE + W")
    ax1.set_title("Total energy vs time")
    ax1.legend()

    # reference scheme: kinetic / strain / total exchange
    ax2 = axs[1]
    ax2.plot(times, KE_ref, label="KE")
    ax2.plot(times, W_ref, label="W")
    ax2.plot(times, KE_ref + W_ref, label="KE + W", color="k")
    ax2.set_xlabel("time")
    ax2.set_ylabel("energy")
    ax2.set_title(f"{list_config[0][0]}: kinetic / strain / total")
    ax2.legend()

    Matplotlib.Save_fig(folder, "energy")

    if makeMovie:
        Matplotlib.Movie_Simu(
            simu_ref,
            result,
            folder,
            f"{result}.gif",
            N=30,
            deformFactor=1,
            plotMesh=True,
        )

    simu_ref.Set_Iter(0)
    Matplotlib.Plot(simu_ref, result, deformFactor=1.0, plotMesh=True)
    plt.show()
