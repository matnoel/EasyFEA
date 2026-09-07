# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
MonoVentricular
===============

Passive + active hyperelastic simulation of an ellipsoidal left-ventricle model.

Combines a ``Holzapfel-Ogden`` orthotropic law (fiber + sheet directions), an ``active stress`` along the fiber direction, and a ``following pressure`` on the endocardial surface. Time integration uses the midpoint hyperbolic scheme.

Reproduces *Benchmark 1: monoventricular mechanics* (§3) of the cardiac elastodynamics benchmark published in Comput. Methods Appl. Mech. Engrg.: https://www.sciencedirect.com/science/article/pii/S0045782524007394

The ``mesh.msh`` / ``fiber.vtu`` / ``sheet.vtu`` files read for ``fiberSource="vtu"`` are generated beforehand with the `cardiac_benchmark_toolkit <https://github.com/Reidmen/cardiac_benchmark_toolkit>`_ — see the module docstring of ``utils.py`` for the exact procedure. (``fiberSource="analytic"`` builds the fibers/sheets directly in EasyFEA and needs no external data.)

With ``useCoarseConfig=False`` this becomes a large 3D, non-linear, transient problem for which the default direct solver is slow. In that case it is recommended to run it either in parallel with MPI and PETSc (e.g. ``mpiexec -n <N> python MonoVentricular.py`` with a PETSc-backed solver), or, on a single process, with the ``pypardiso`` solver — both markedly cut the solve time. The default ``useCoarseConfig=True`` is light enough to run as-is.
"""

from enum import Enum

import numpy as np

from EasyFEA import Terminal, Matplotlib, Folder, PyVista, MatrixType, Simulations

from utils import (
    RESULTS_DIR,
    DATA_DIR,
    Get_config_ellipsoid,
    Get_material,
    Get_simu,
    Get_stresses,
    Get_pressures,
)


class Config(str, Enum):
    step0A = "step0A"  # active_stress
    step0B = "step0B"  # pressure
    step1 = "step1"  # active_stress + pressure


if __name__ == "__main__":

    Terminal.Clear()

    # ----------------------------------------------
    # Config
    # ----------------------------------------------

    useCoarseConfig = True

    meshName = "ellipsoid0.03" if useCoarseConfig else "ellipsoid0.005"

    config = Config.step1

    fiberSource = "analytic"
    # fiberSource = "vtu"

    matrixType = MatrixType.rigi
    # matrixType = MatrixType.mass
    # matrixType = 15

    results_dir = Folder.Join(RESULTS_DIR, config.name, meshName)

    doSimu = True
    plotGraph = False
    plotParticles = True
    saveParticles = True
    makeMovie = True

    # ----------------------------------------------
    # time-history needed for plotting in both doSimu / Load_Simu flows

    Nt = 80 if useCoarseConfig else 1000

    times = np.linspace(0, 1, Nt + 1)
    dt = times[1] - times[0]

    stresses = Get_stresses(times)
    pressures = Get_pressures(times)

    results_dir += f"_dt{dt}_{fiberSource}_{matrixType}"

    if plotGraph:
        ax = Matplotlib.Init_Axes()
        ax.grid()
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$\tau(t)$ [Pa]")
        ax.plot(times, stresses)
        name = "active_pressure"
        Matplotlib.Save_fig(results_dir, name)

        ax = Matplotlib.Init_Axes()
        ax.grid()
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$p(t)$ [Pa]")
        ax.plot(times, pressures)
        name = "pressure"
        Matplotlib.Save_fig(results_dir, name)

    if config is Config.step0B:
        stresses *= 0
    if config is Config.step0A:
        pressures *= 0

    if doSimu:

        # ----------------------------------------------
        # Mesh, fibers and sheets
        # ----------------------------------------------

        mesh, fibers_e_pg, sheets_e_pg = Get_config_ellipsoid(
            Folder.Join(DATA_DIR, meshName),
            matrixType=matrixType,
            fiberSource=fiberSource,
            plotMesh=False,
            plotTags=False,
            plotFibers=False,
        )

        # ----------------------------------------------
        # Material
        # ----------------------------------------------

        # solid
        a, a_f, a_fs, a_s = 59.0, 18472.0, 216.0, 2481.0

        material = Get_material(
            fibers_e_pg,
            sheets_e_pg,
            a,
            a_f,
            a_fs,
            a_s,
            useJax=False,
        )

        # ----------------------------------------------
        # Simulation
        # ----------------------------------------------

        simu, endoTerms = Get_simu(
            mesh, material, dt, ["endo"], folder=results_dir, matrixType=matrixType
        )

        for t in times:
            simu.Bc_Init()
            endoTerms["endo"].Set(pressure=np.interp(t + dt / 2, times, pressures))
            material.active_stress = np.interp(t + dt / 2, times, stresses)
            simu.Solve()
            simu.Save_Iter()

        simu.Save(results_dir)

    else:
        simu = Simulations.Load_Simu(results_dir)

    simu._Gather()

    if plotParticles and simu.isGathered:

        coords = [(0.025, 0.03, 0), (0, 0.03, 0)]
        evalCoords = np.array(coords)
        evalElements = simu.mesh.groupElem._Get_nearby_elements(evalCoords)

        Niter = simu.Niter
        values = np.empty((Niter, len(coords), 3))
        for i in range(Niter):
            simu.Set_Iter(i)
            values[i] = simu.mesh.Evaluate_dofsValues_at_coordinates(
                evalCoords, simu.displacement, evalElements
            )

        times = times[:Niter]
        axs = Matplotlib.plt.subplots(3, 2, sharex=True)[1]

        for p, (particle, coord) in enumerate(zip(["p0", "p1"], coords)):

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
                for p in range(2)
            },
            "stress": {
                "time": None,
                "p0": {"magnitude": None},
                "p1": {"magnitude": None},
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
