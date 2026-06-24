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
from EasyFEA.FEM import Operators
from EasyFEA.Utilities import _params

from utils import RESULTS_DIR, DATA_DIR, Get_config, Get_values


class CardiacElastoDynamics(Simulations.HyperElastic):

    pressure = _params.ScalarParameter()

    def __init__(
        self, mesh, model, folder="", tolConv=0.00001, maxIter=20, verbosity=False
    ):
        super().__init__(mesh, model, folder, tolConv, maxIter, verbosity)
        self.pressure = 0.0

    def Construct_local_matrix_system(self, problemType):

        assert isinstance(self.material, Models.HyperElastic.HolzapfelOgden)
        nPg = self.material.T1.shape[1]

        results = super().Construct_local_matrix_system(problemType, nPg)

        # current Newton-Raphson iterate (updated via u += delta_u)
        displacement = self._Solver_Get_Newton_Raphson_current_solution()
        if self.algo in AlgoType.Get_Hyperbolic_Types():
            displacement, _, _ = self._Solver_Evaluate_u_v_a_for_time_scheme(
                problemType, displacement
            )

        for groupElem in self.mesh.Get_list_groupElem(self.dim - 1):

            # Following pressure — tracks the deformed normal, so it depends on
            # the current iterate / pressure and is rebuilt every Newton step.
            tangent_e, residual_e = Operators.NonLinear.FollowingPressure(
                groupElem,
                displacement,
                self.pressure,
                groupElem.Get_Elements_Tag("endo"),
                MatrixType.mass,
            )

            # top — isotropic surface mass penalty (Robin α·u + β·u̇ = 0)
            M_e = Operators.Bilinear.UV(groupElem, dof_n=3)

            Ktop_e = np.zeros_like(M_e)
            Ctop_e = np.zeros_like(M_e)
            if "top" in groupElem.elementTags:
                top_e = groupElem.Get_Elements_Tag("top")
                alpha_top = 1e5
                Ktop_e[top_e] = alpha_top * M_e[top_e]

                beta_top = 5e3
                Ctop_e[top_e] = beta_top * M_e[top_e]

            # epi — normal-direction mass penalty (Robin α·(u·n̂) + β·(u̇·n̂) = 0)
            Ms_e = Operators.Bilinear.MassAlongNormal(groupElem)

            Kepi_e = np.zeros_like(Ms_e)
            Cepi_e = np.zeros_like(Ms_e)
            if "epi" in groupElem.elementTags:
                epi_e = groupElem.Get_Elements_Tag("epi")
                alpha_epi = 1e8
                Kepi_e[epi_e] = alpha_epi * Ms_e[epi_e]

                beta_epi = 5e3
                Cepi_e[epi_e] = beta_epi * Ms_e[epi_e]

            # Penalty residual contribution: −K_penalty · u_t at current iterate
            K_penalty_e = Ktop_e + Kepi_e
            assembly_e = groupElem.Get_assembly_e(self.dim)
            u_e = displacement[assembly_e]  # (Ne_surf, nPe·3)
            f_penalty_e = np.einsum("eij,ej->ei", K_penalty_e, u_e)

            results[groupElem] = (
                tangent_e + K_penalty_e,
                Ctop_e + Cepi_e,
                None,
                residual_e - f_penalty_e,
            )

        return results


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

    ellipsoid = "ellipsoid0.03" if useCoarseConfig else "ellipsoid0.005"

    config = Config.step1

    fiberSource = "analytic"
    # fiberSource = "vtu"

    matrixType = MatrixType.rigi
    # matrixType = MatrixType.mass
    # matrixType = 15

    results_dir = Folder.Join(RESULTS_DIR, config.name, ellipsoid)

    doSimu = True
    plotGraph = False
    plotParticles = True
    saveParticles = True
    makeMovie = True

    # ----------------------------------------------
    # time-history needed for plotting in both doSimu / Load_Simu flows

    Nt = 80 if useCoarseConfig else 1000

    t_values, activeStress_values, pressure_values = Get_values(Tmax=1.0, Nt=Nt)
    dt = t_values[1] - t_values[0]
    results_dir += f"_dt{dt}_{fiberSource}_{matrixType}"

    if plotGraph:
        ax = Matplotlib.Init_Axes()
        ax.grid()
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$\tau(t)$ [Pa]")
        ax.plot(t_values, activeStress_values)
        name = "active_pressure"
        Matplotlib.Save_fig(results_dir, name)

        ax = Matplotlib.Init_Axes()
        ax.grid()
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$p(t)$ [Pa]")
        ax.plot(t_values, pressure_values)
        name = "pressure"
        Matplotlib.Save_fig(results_dir, name)

    if config is Config.step0B:
        activeStress_values *= 0
    if config is Config.step0A:
        pressure_values *= 0

    if doSimu:

        # ----------------------------------------------
        # Mesh, fibers and sheets
        # ----------------------------------------------

        mesh, fibers_e_pg, sheets_e_pg = Get_config(
            Folder.Join(DATA_DIR, ellipsoid),
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
        a = 59.0
        a_f = 18472.0
        a_fs = 216.0
        a_s = 2481.0
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

        simu = CardiacElastoDynamics(mesh, material, folder=results_dir)

        simu.Solver_Set_Hyperbolic_Algorithm(dt, algo=AlgoType.midpoint)
        simu.rho = 1000

        for t in t_values:
            simu.Bc_Init()
            simu.pressure = np.interp(t + dt / 2, t_values, pressure_values)
            material.active_stress = np.interp(
                t + dt / 2, t_values, activeStress_values
            )
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

        times = t_values[:Niter]
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
