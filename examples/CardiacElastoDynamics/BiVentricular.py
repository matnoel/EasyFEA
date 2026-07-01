# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
BiVentricular
=============

Passive + active hyperelastic simulation of an bi-ventricular model.

Combines a ``Holzapfel-Ogden`` orthotropic law (fiber + sheet directions), an ``active stress`` along the fiber direction, and a ``following pressure`` on the endocardial surface. Time integration uses the midpoint hyperbolic scheme.

Reproduces *Benchmark 2: biventricular mechanics* (§4) of the cardiac elastodynamics benchmark published in Comput. Methods Appl. Mech. Engrg.: https://www.sciencedirect.com/science/article/pii/S0045782524007394

The ``mesh.xdmf`` / ``mesh.h5`` / ``data.vtk`` are available in the `cardiac_benchmark_toolkit <https://github.com/Reidmen/cardiac_benchmark_toolkit>`_.

This script is not run in the Sphinx gallery because it takes too long to complete and would blow up the documentation build — this large 3D, non-linear, transient problem is solver-bound and slow for *both* configurations, the bottleneck being the default direct linear solver. To markedly cut the solve time it is recommended to run either in parallel with MPI and PETSc (e.g. ``mpiexec -n <N> python BiVentricular.py`` with a PETSc-backed solver), or, on a single process, with the ``pypardiso`` solver.
"""

from enum import Enum

import numpy as np


from EasyFEA import (
    Terminal,
    Matplotlib,
    Folder,
    PyVista,
    MatrixType,
    Models,
    Simulations,
    AlgoType,
)

from MonoVentricular import CardiacElastoDynamics

from utils import (
    RESULTS_DIR,
    DATA_DIR,
    Get_biventricular,
    Get_stresses,
    Get_pressures,
)


class Config(str, Enum):
    A = "A"
    B = "B"
    C = "C"


if __name__ == "__main__":

    Terminal.Clear()

    # ----------------------------------------------
    # Config
    # ----------------------------------------------

    useCoarseConfig = True

    meshName = "biventricularCoarse" if useCoarseConfig else "biventricularFine"

    config = Config.A

    matrixType = MatrixType.rigi
    # matrixType = MatrixType.mass

    results_dir = Folder.Join(RESULTS_DIR, config.name, meshName)

    doSimu = True
    plotGraph = False
    plotParticles = False
    saveParticles = True
    makeMovie = True

    # ----------------------------------------------
    # time-history needed for plotting in both doSimu / Load_Simu flows

    Nt = 80 if useCoarseConfig else 1000

    times = np.linspace(0, 1, Nt + 1)
    dt = times[1] - times[0]

    sig_0 = {
        Config.A: 2 * 1e5,
        Config.B: 1 * 1e5,
        Config.C: 2 * 1e5,
    }[config]

    stresses = Get_stresses(times, sig_0=sig_0, t_sys=0.163, t_dias=0.5)
    pressures_lv = Get_pressures(times, alpha_mid=15, sig_pre=12000)
    pressures_rv = Get_pressures(
        times, alpha_pre=1, alpha_mid=10, sig_pre=3000, sig_mid=4000
    )

    results_dir += f"_dt{dt}_{matrixType}"

    if plotGraph:
        ax = Matplotlib.Init_Axes()
        ax.grid()
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$\tau(t)$ [Pa]")
        ax.plot(times, stresses)
        name = "active_pressure"
        Matplotlib.Save_fig(results_dir, name)

        ax = Matplotlib.Init_Axes()
        ax.set_xlabel(r"$t$ [s]")
        ax.grid()
        ax2 = ax.twinx()
        ax2.grid()

        ax.plot(times, pressures_lv, color="blue")
        ax.tick_params(axis="y", labelcolor="blue")
        ax.set_ylabel(r"Left ventricle $p(t)$ [Pa]", color="blue")
        ax2.plot(times, pressures_rv, color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.set_ylabel(r"Right ventricle $p(t)$ [Pa]", color="red")
        name = "pressure"
        Matplotlib.Save_fig(results_dir, name)

    if doSimu:

        # ----------------------------------------------
        # Mesh, fibers and sheets
        # ----------------------------------------------

        mesh, fibers_e_pg, sheets_e_pg = Get_biventricular(
            Folder.Join(DATA_DIR, meshName),
            matrixType=matrixType,
            plotMesh=False,
            plotTags=False,
            plotFibers=False,
        )

        # ----------------------------------------------
        # Material
        # ----------------------------------------------

        # solid
        a, a_f, a_fs, a_s = dict_a = {
            Config.A: (177, 55416, 648, 7443),
            Config.B: (295, 92360, 1080, 12405),
            Config.C: (19, 6157, 72, 827),
        }[config]
        b = 8.023
        b_f = 16.026
        b_fs = 11.436
        b_s = 11.12

        material = Models.HyperElastic.HolzapfelOgden(
            dim=3,
            C0=a / 2 / b,
            C1=b,
            C2=a_f / 2 / b_f,
            C3=b_f,
            C4=a_s / 2 / b_s,
            C5=b_s,
            C6=a_fs / 2 / b_fs,
            C7=b_fs,
            K=1e6,
            Mu1=0.0,
            Mu2=0.0,
            T1=fibers_e_pg,
            T2=sheets_e_pg,
            ks=100,
        )
        material.eta = 100.0
        material.Set_active_stress_vec(material.T1)

        # ----------------------------------------------
        # Simulation
        # ----------------------------------------------

        simu = CardiacElastoDynamics(
            mesh,
            material,
            folder=results_dir,
            alpha_top=1e6,
            alpha_epi=1e8,
        )

        simu.Solver_Set_Hyperbolic_Algorithm(dt, algo=AlgoType.midpoint)
        simu.rho = 1000

        for t in times:

            simu.Bc_Init()
            simu.Set_pressure(
                {
                    "endo_lv": np.interp(t + dt / 2, times, pressures_lv),
                    "endo_rv": np.interp(t + dt / 2, times, pressures_rv),
                }
            )
            material.active_stress = np.interp(t + dt / 2, times, stresses)
            simu.Solve()
            simu.Save_Iter()

        simu.Save(results_dir)

    else:
        simu = Simulations.Load_Simu(results_dir)

    simu._Gather()

    Niter = simu.Niter
    if plotParticles or saveParticles and simu.isGathered:

        coords = [
            (0.025, 0.03, 0),
            (0, 0.03, 0),
            (0.025, 0, 0.072),
        ]
        evalCoords = np.array(coords)

        evalElements = simu.mesh.groupElem._Get_nearby_elements(evalCoords)
        # print(evalElements)

        values = np.empty((Niter, len(coords), 3))
        for i in range(Niter):
            simu.Set_Iter(i)
            values[i] = simu.mesh.Evaluate_dofsValues_at_coordinates(
                evalCoords, simu.displacement, evalElements
            )

        times = times[:Niter]
        axs = Matplotlib.plt.subplots(3, 3, sharex=True)[1]

        for p, (particle, coord) in enumerate(zip(["p0", "p1", "p2"], coords)):

            for c, component in enumerate(["x", "y", "z"]):

                ax: Matplotlib.plt.Axes = axs[c, p]

                ax.grid()

                if c == 2:
                    ax.set_xlabel("Time [s]")
                if p == 0:
                    ax.set_ylabel(f"Displacement {component}-component [m]")
                if c == 0:
                    ax.set_title(f"Particle {particle}")

                ax.plot(times, values[:, p, c])

        width, height = ax.figure.get_size_inches()
        ax.figure.set_size_inches(width * 1.5, height * 2.5)
        Matplotlib.Save_fig(results_dir, "particles")

    if saveParticles and simu.isGathered:

        # per-iteration deformed volume
        volumes = np.empty(Niter)
        for i in range(Niter):
            simu.Set_Iter(i)
            deformed = simu.mesh.copy()
            deformed.coord += simu.displacement.reshape(-1, 3)
            volumes[i] = deformed.volume

        dict_particles = {
            "time": times,
            "displacement": {
                f"p{p}": {
                    "ux": values[:, p, 0],
                    "uy": values[:, p, 1],
                    "uz": values[:, p, 2],
                    "magnitude": np.linalg.norm(values[:, p, :], axis=1),
                }
                for p in range(3)
            },
            "stress": {
                "time": None,
                "p0": {"magnitude": None},
                "p1": {"magnitude": None},
                "p2": {"magnitude": None},
            },
            "volume": volumes,
        }
        Simulations.Save_pickle(dict_particles, results_dir, "particles")

    if makeMovie:
        values = [simu.Result("ux", iter=i) for i in range(simu.Niter)]
        clim = (np.min(values), np.max(values))
        PyVista.Movie_simu(
            simu,
            "ux",
            results_dir,
            "ux.gif",
            N=20,
            deformFactor=1.0,
            clim=clim,
            plotMesh=True,
        )

    Matplotlib.plt.show()
